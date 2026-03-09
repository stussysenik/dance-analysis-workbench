from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
import wave
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeAlias

import cv2
import numpy as np

from dance_analysis.adapters import CRadioV4Adapter, SapiensPoseAdapter
from dance_analysis.config import RUNS_DIR, RuntimeProfile, ensure_directories, load_runtime_profiles
from dance_analysis.contracts import (
    AnalysisResult,
    AnalyzeVideoRequest,
    BeatEvent,
    BiomechFrame,
    BoundingBox,
    MusicAnalysis,
    PoseFrame,
    Point,
    RenderBundle,
    TrackFrame,
    TrackRecord,
)

JOINT_WEIGHTS = {
    "head": 0.08,
    "shoulder_l": 0.17,
    "shoulder_r": 0.17,
    "hip_l": 0.24,
    "hip_r": 0.24,
    "foot_l": 0.05,
    "foot_r": 0.05,
}
FeatureVector: TypeAlias = np.ndarray


class DanceAnalysisPipeline:
    def __init__(self) -> None:
        self.profiles = load_runtime_profiles()
        self.backbone = CRadioV4Adapter()
        self.pose_adapter = SapiensPoseAdapter()

    def analyze(self, request: AnalyzeVideoRequest) -> AnalysisResult:
        ensure_directories()
        profile = self.profiles[request.runtime_profile]
        capture = cv2.VideoCapture(str(request.video_path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {request.video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        start_frame = max(0, int(request.segment.start_sec * fps))
        end_frame = int(min(total_frames - 1 if total_frames else start_frame, request.segment.end_sec * fps))

        frames, frame_indices, timestamps = _read_segment_frames(capture, start_frame, end_frame, profile)
        capture.release()
        if not frames:
            raise ValueError("Selected segment did not contain readable frames")

        tracks = _track_dancers(
            frames=frames,
            frame_indices=frame_indices,
            timestamps=timestamps,
            target_dancers=request.target_dancers,
            threshold=profile.detection_threshold,
            stability_threshold=profile.track_stability_threshold,
            correction_hints=request.correction_hints,
            backbone=self.backbone if profile.prefer_model_adapters else None,
        )
        poses = _estimate_poses(tracks, self.pose_adapter)
        biomechanics = _score_biomechanics(poses)
        music = _analyze_music(
            request=request,
            timestamps=timestamps,
            biomechanics=biomechanics,
            confidence_threshold=profile.beat_confidence_threshold,
        )
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        render_bundle = _render_outputs(
            run_id=run_id,
            request=request,
            frames=frames,
            timestamps=timestamps,
            tracks=tracks,
            poses=poses,
            biomechanics=biomechanics,
            music=music,
        )

        summary = {
            "runtime_profile": request.runtime_profile,
            "target_dancers": request.target_dancers,
            "frame_count": len(frames),
            "backbone_status": self.backbone.status().reason,
            "pose_status": self.pose_adapter.status().reason,
            "track_warnings": sum(len(track.warnings) for track in tracks),
            "music_confidence": music.confidence,
        }
        result = AnalysisResult(
            request=request,
            tracks=tracks,
            poses=poses,
            biomechanics=biomechanics,
            music=music,
            render_bundle=render_bundle,
            summary=summary,
        )
        render_bundle.json_path.write_text(result.model_dump_json(indent=2, exclude_none=True))
        return result


def compute_center_of_mass(joints: dict[str, Point]) -> Point:
    total_weight = sum(JOINT_WEIGHTS[name] for name in joints if name in JOINT_WEIGHTS)
    if total_weight == 0:
        return Point(x=0.0, y=0.0)
    x = sum(joints[name].x * JOINT_WEIGHTS[name] for name in joints if name in JOINT_WEIGHTS) / total_weight
    y = sum(joints[name].y * JOINT_WEIGHTS[name] for name in joints if name in JOINT_WEIGHTS) / total_weight
    return Point(x=float(x), y=float(y))


def score_on_beat(beat_times: list[float], motion_event_times: list[float], beat_period: float) -> float:
    if not beat_times or not motion_event_times or beat_period <= 0:
        return 0.0
    distances = [min(abs(event - beat) for beat in beat_times) for event in motion_event_times]
    return float(np.mean([math.exp(-(distance / beat_period) ** 2) for distance in distances]))


def _read_segment_frames(
    capture: cv2.VideoCapture,
    start_frame: int,
    end_frame: int,
    profile: RuntimeProfile,
) -> tuple[list[np.ndarray], list[int], list[float]]:
    frames: list[np.ndarray] = []
    frame_indices: list[int] = []
    timestamps: list[float] = []
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current = start_frame
    collected = 0
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    while current <= end_frame and collected < profile.max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        if (current - start_frame) % profile.sample_stride == 0:
            frames.append(frame)
            frame_indices.append(current)
            timestamps.append(current / fps)
            collected += 1
        current += 1
    return frames, frame_indices, timestamps


def _track_dancers(
    frames: list[np.ndarray],
    frame_indices: list[int],
    timestamps: list[float],
    target_dancers: int,
    threshold: float,
    stability_threshold: float,
    correction_hints: str,
    backbone: CRadioV4Adapter | None = None,
) -> list[TrackRecord]:
    seeded_centers = _parse_seed_centers(correction_hints)
    history: dict[int, list[TrackFrame]] = {i: [] for i in range(target_dancers)}
    warnings: dict[int, list[str]] = defaultdict(list)

    previous_centers = seeded_centers
    previous_features: list[FeatureVector | None] = [None] * target_dancers
    use_backbone = bool(backbone and backbone.status().available)
    for frame, frame_index, timestamp in zip(frames, frame_indices, timestamps, strict=False):
        candidates = _detect_people_boxes(frame, target_dancers, threshold)
        if not candidates:
            height, width = frame.shape[:2]
            candidates = [BoundingBox(x=width // 4, y=height // 8, width=width // 2, height=int(height * 0.8))]
        candidate_features = [backbone.extract_features(frame, box) for box in candidates] if use_backbone else None
        assignments, assignment_features = _assign_boxes(
            candidates,
            previous_centers,
            target_dancers,
            previous_features=previous_features,
            candidate_features=candidate_features,
        )
        current_centers: list[Point] = []
        current_features: list[FeatureVector | None] = []
        for slot, box in enumerate(assignments):
            confidence = _box_confidence(box, frame.shape[:2])
            unstable = confidence < stability_threshold
            history[slot].append(
                TrackFrame(
                    frame_index=frame_index,
                    timestamp_sec=timestamp,
                    box=box,
                    confidence=confidence,
                    unstable=unstable,
                )
            )
            current_centers.append(box.centroid)
            current_features.append(assignment_features[slot] if slot < len(assignment_features) else None)
            if unstable:
                warnings[slot].append(f"Low-confidence track near {timestamp:.2f}s")
        previous_centers = current_centers
        previous_features = current_features

    return [
        TrackRecord(dancer_id=f"dancer_{slot + 1}", frames=frames_for_track, warnings=warnings[slot])
        for slot, frames_for_track in history.items()
    ]


def _detect_people_boxes(frame: np.ndarray, target_dancers: int, threshold: float) -> list[BoundingBox]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = gray.shape
    min_area = max(500, int(height * width * 0.02 * threshold))
    boxes: list[BoundingBox] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h >= min_area and h > height * 0.18:
            boxes.append(BoundingBox(x=x, y=y, width=w, height=h))
    boxes.sort(key=lambda item: item.width * item.height, reverse=True)
    return boxes[:target_dancers]


def _parse_seed_centers(correction_hints: str) -> list[Point]:
    points: list[Point] = []
    for chunk in correction_hints.split(";"):
        parts = [part.strip() for part in chunk.split(",")]
        if len(parts) != 2:
            continue
        try:
            points.append(Point(x=float(parts[0]), y=float(parts[1])))
        except ValueError:
            continue
    return points


def _assign_boxes(
    candidates: list[BoundingBox],
    previous_centers: list[Point],
    target_dancers: int,
    previous_features: list[FeatureVector | None] | None = None,
    candidate_features: list[FeatureVector] | None = None,
) -> tuple[list[BoundingBox], list[FeatureVector | None]]:
    assigned: list[BoundingBox] = []
    assigned_features: list[FeatureVector | None] = []
    remaining = list(enumerate(candidates))
    for slot in range(target_dancers):
        if previous_centers and slot < len(previous_centers) and remaining:
            remaining.sort(
                key=lambda item: _assignment_cost(
                    candidate=item[1],
                    previous_center=previous_centers[slot],
                    previous_feature=previous_features[slot] if previous_features and slot < len(previous_features) else None,
                    candidate_feature=candidate_features[item[0]] if candidate_features else None,
                )
            )
            index, candidate = remaining.pop(0)
            assigned.append(candidate)
            assigned_features.append(candidate_features[index] if candidate_features else None)
        elif remaining:
            index, candidate = remaining.pop(0)
            assigned.append(candidate)
            assigned_features.append(candidate_features[index] if candidate_features else None)
    while len(assigned) < target_dancers:
        assigned.append(assigned[-1] if assigned else BoundingBox(x=0, y=0, width=32, height=32))
        assigned_features.append(assigned_features[-1] if assigned_features else None)
    return assigned, assigned_features


def _assignment_cost(
    candidate: BoundingBox,
    previous_center: Point,
    previous_feature: FeatureVector | None,
    candidate_feature: FeatureVector | None,
) -> tuple[float, float]:
    appearance_cost = _appearance_distance(previous_feature, candidate_feature)
    spatial_cost = (candidate.centroid.x - previous_center.x) ** 2 + (candidate.centroid.y - previous_center.y) ** 2
    return appearance_cost, spatial_cost


def _appearance_distance(previous_feature: FeatureVector | None, candidate_feature: FeatureVector | None) -> float:
    if previous_feature is None or candidate_feature is None:
        return 1.0
    denom = float(np.linalg.norm(previous_feature) * np.linalg.norm(candidate_feature))
    if denom <= 0:
        return 1.0
    similarity = float(np.dot(previous_feature, candidate_feature) / denom)
    return 1.0 - similarity


def _box_confidence(box: BoundingBox, frame_shape: tuple[int, int]) -> float:
    height, width = frame_shape
    area_ratio = (box.width * box.height) / max(1, width * height)
    aspect_penalty = min(1.0, box.height / max(box.width, 1))
    return float(max(0.2, min(0.98, area_ratio * 6.0 + aspect_penalty * 0.3)))


def _estimate_poses(tracks: list[TrackRecord], pose_adapter: SapiensPoseAdapter) -> list[PoseFrame]:
    poses: list[PoseFrame] = []
    for track in tracks:
        for frame in track.frames:
            poses.append(
                pose_adapter.estimate_pose(
                    dancer_id=track.dancer_id,
                    frame_index=frame.frame_index,
                    timestamp_sec=frame.timestamp_sec,
                    box=frame.box,
                )
            )
    return poses


def _score_biomechanics(poses: list[PoseFrame]) -> list[BiomechFrame]:
    previous_com: dict[str, Point] = {}
    previous_velocity: dict[str, float] = defaultdict(float)
    travel_distance: dict[str, float] = defaultdict(float)
    output: list[BiomechFrame] = []

    for pose in poses:
        joints = {joint.name: joint.point for joint in pose.joints}
        com = compute_center_of_mass(joints)
        prev = previous_com.get(pose.dancer_id, com)
        dt = 1.0 / 30.0
        displacement = math.dist((com.x, com.y), (prev.x, prev.y))
        velocity = displacement / dt
        acceleration = (velocity - previous_velocity[pose.dancer_id]) / dt
        previous_velocity[pose.dancer_id] = velocity
        previous_com[pose.dancer_id] = com
        travel_distance[pose.dancer_id] += displacement
        expansion = _joint_span(pose.joints)
        warnings = []
        if pose.confidence < 0.6:
            warnings.append("Approximate pose; COM is a proxy")
        output.append(
            BiomechFrame(
                dancer_id=pose.dancer_id,
                frame_index=pose.frame_index,
                timestamp_sec=pose.timestamp_sec,
                center_of_mass=com,
                velocity=float(velocity),
                acceleration=float(acceleration),
                travel_distance=float(travel_distance[pose.dancer_id]),
                expansion=float(expansion),
                confidence=pose.confidence,
                warnings=warnings,
            )
        )
    return output


def _joint_span(joints: list) -> float:
    xs = [joint.point.x for joint in joints]
    ys = [joint.point.y for joint in joints]
    return float((max(xs) - min(xs)) * (max(ys) - min(ys)))


def _analyze_music(
    request: AnalyzeVideoRequest,
    timestamps: list[float],
    biomechanics: list[BiomechFrame],
    confidence_threshold: float,
) -> MusicAnalysis:
    audio_path = request.audio_path or _extract_audio(request.video_path)
    motion_event_times = _motion_events_from_biomech(biomechanics)
    if audio_path is None or not audio_path.exists():
        return MusicAnalysis(
            bpm=0.0,
            beat_times=[],
            motion_event_times=motion_event_times,
            on_beat_score=0.0,
            confidence=0.0,
            warnings=["Audio unavailable; beat score skipped"],
        )

    samples, sr = _read_wav(audio_path)
    envelope, frame_hop = _energy_envelope(samples, sr)
    bpm, beat_times, confidence = _estimate_beats(envelope, frame_hop, sr)
    score = score_on_beat(beat_times, motion_event_times, 60.0 / bpm if bpm else 0.0)
    warnings = []
    if confidence < confidence_threshold:
        warnings.append("Beat estimate is low confidence")
    if not beat_times:
        warnings.append("No stable beat grid detected")
    return MusicAnalysis(
        bpm=float(bpm),
        beat_times=beat_times,
        motion_event_times=motion_event_times,
        on_beat_score=float(score),
        confidence=float(confidence),
        warnings=warnings,
    )


def _extract_audio(video_path: Path) -> Path | None:
    ffmpeg = _which("ffmpeg")
    if ffmpeg is None:
        return None
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        output = Path(handle.name)
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-ac",
        "1",
        "-ar",
        "22050",
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, check=False)
    if result.returncode != 0:
        output.unlink(missing_ok=True)
        return None
    return output


def _which(binary: str) -> str | None:
    return shutil.which(binary)


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        frames = handle.readframes(handle.getnframes())
        sr = handle.getframerate()
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        channels = handle.getnchannels()
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, sr


def _energy_envelope(samples: np.ndarray, sr: int, window: int = 1024, hop: int = 512) -> tuple[np.ndarray, int]:
    if samples.size < window:
        return np.zeros(1, dtype=np.float32), hop
    energies = []
    for start in range(0, samples.size - window, hop):
        chunk = samples[start : start + window]
        energies.append(float(np.sqrt(np.mean(chunk**2))))
    envelope = np.asarray(energies, dtype=np.float32)
    envelope = envelope - envelope.mean()
    envelope[envelope < 0] = 0
    return envelope, hop


def _estimate_beats(envelope: np.ndarray, hop: int, sr: int) -> tuple[float, list[float], float]:
    if envelope.size < 4 or float(envelope.max(initial=0.0)) <= 0:
        return 0.0, [], 0.0
    min_bpm = 70
    max_bpm = 180
    min_lag = max(1, int((60 / max_bpm) * sr / hop))
    max_lag = max(min_lag + 1, int((60 / min_bpm) * sr / hop))
    autocorr = np.correlate(envelope, envelope, mode="full")[envelope.size - 1 :]
    band = autocorr[min_lag:max_lag]
    if band.size == 0:
        return 0.0, [], 0.0
    best_lag = int(np.argmax(band)) + min_lag
    bpm = 60.0 * sr / (best_lag * hop)
    beat_period = best_lag * hop / sr
    peaks = np.where(envelope > envelope.mean() + envelope.std() * 0.75)[0]
    if peaks.size == 0:
        return bpm, [], 0.2
    first_time = peaks[0] * hop / sr
    duration = envelope.size * hop / sr
    beat_times = list(np.arange(first_time, duration, beat_period))
    confidence = float(min(0.95, max(0.2, band.max() / max(autocorr[0], 1e-6))))
    return bpm, beat_times, confidence


def _motion_events_from_biomech(biomechanics: list[BiomechFrame]) -> list[float]:
    if not biomechanics:
        return []
    per_dancer: dict[str, list[BiomechFrame]] = defaultdict(list)
    for frame in biomechanics:
        per_dancer[frame.dancer_id].append(frame)
    events: list[float] = []
    for frames in per_dancer.values():
        velocities = np.asarray([frame.velocity for frame in frames], dtype=np.float32)
        if velocities.size == 0:
            continue
        threshold = float(velocities.mean() + velocities.std())
        for frame in frames:
            if frame.velocity >= threshold:
                events.append(frame.timestamp_sec)
    events.sort()
    return events


def _render_outputs(
    run_id: str,
    request: AnalyzeVideoRequest,
    frames: list[np.ndarray],
    timestamps: list[float],
    tracks: list[TrackRecord],
    poses: list[PoseFrame],
    biomechanics: list[BiomechFrame],
    music: MusicAnalysis,
) -> RenderBundle:
    output_dir = RUNS_DIR / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_path = output_dir / "preview.jpg"
    video_path = output_dir / "annotated.mp4"
    json_path = output_dir / "analysis.json"

    frame_map = {pose.frame_index: [] for pose in poses}
    for pose in poses:
        frame_map[pose.frame_index].append(pose)
    biomech_map = {frame.frame_index: [] for frame in biomechanics}
    for frame in biomechanics:
        biomech_map[frame.frame_index].append(frame)

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height))
    com_trails: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for frame, timestamp, index in zip(frames, timestamps, [t.frame_index for t in tracks[0].frames], strict=False):
        annotated = frame.copy()
        if "boxes" in request.overlay_layers or not request.overlay_layers:
            for track in tracks:
                track_frame = next(item for item in track.frames if item.frame_index == index)
                _draw_box(annotated, track.dancer_id, track_frame.box, track_frame.unstable)
        if "skeleton" in request.overlay_layers or not request.overlay_layers:
            for pose in frame_map.get(index, []):
                _draw_pose(annotated, pose)
        if "com" in request.overlay_layers or not request.overlay_layers:
            for biomech in biomech_map.get(index, []):
                point = (int(biomech.center_of_mass.x), int(biomech.center_of_mass.y))
                com_trails[biomech.dancer_id].append(point)
                _draw_trail(annotated, com_trails[biomech.dancer_id])
        if "beats" in request.overlay_layers or not request.overlay_layers:
            nearest = min((abs(timestamp - beat) for beat in music.beat_times), default=999.0)
            if nearest < 0.08:
                cv2.putText(annotated, "BEAT", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 215, 255), 2)
        writer.write(annotated)
        if index == tracks[0].frames[0].frame_index:
            cv2.imwrite(str(preview_path), annotated)
    writer.release()
    if not preview_path.exists():
        cv2.imwrite(str(preview_path), frames[0])
    return RenderBundle(run_id=run_id, output_dir=output_dir, annotated_video_path=video_path, preview_frame_path=preview_path, json_path=json_path)


def _draw_box(frame: np.ndarray, dancer_id: str, box: BoundingBox, unstable: bool) -> None:
    color = (0, 140, 255) if dancer_id.endswith("1") else (102, 255, 102)
    if unstable:
        color = (0, 0, 255)
    cv2.rectangle(frame, (box.x, box.y), (box.x + box.width, box.y + box.height), color, 2)
    cv2.putText(frame, dancer_id, (box.x, max(20, box.y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def _draw_pose(frame: np.ndarray, pose: PoseFrame) -> None:
    joint_map = {joint.name: (int(joint.point.x), int(joint.point.y)) for joint in pose.joints}
    for point in joint_map.values():
        cv2.circle(frame, point, 3, (255, 255, 255), -1)
    for start, end in [("head", "shoulder_l"), ("head", "shoulder_r"), ("shoulder_l", "hip_l"), ("shoulder_r", "hip_r"), ("hip_l", "foot_l"), ("hip_r", "foot_r")]:
        cv2.line(frame, joint_map[start], joint_map[end], (255, 220, 0), 1)


def _draw_trail(frame: np.ndarray, trail: list[tuple[int, int]]) -> None:
    for index in range(1, len(trail)):
        cv2.line(frame, trail[index - 1], trail[index], (255, 0, 120), 2)
