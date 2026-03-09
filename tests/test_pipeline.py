from __future__ import annotations

import math
import wave
from pathlib import Path

import cv2
import numpy as np

from dance_analysis.contracts import AnalyzeVideoRequest, SegmentSelection
from dance_analysis.contracts import BoundingBox, Point
from dance_analysis.pipeline import DanceAnalysisPipeline, _assign_boxes


def _write_test_video(path: Path, frame_count: int = 36) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240))
    for index in range(frame_count):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        left_x = 20 + index * 2
        right_x = 210 - index
        cv2.rectangle(frame, (left_x, 40), (left_x + 42, 190), (255, 255, 255), -1)
        cv2.rectangle(frame, (right_x, 55), (right_x + 45, 205), (220, 220, 220), -1)
        writer.write(frame)
    writer.release()


def _write_test_wav(path: Path, duration_sec: float = 1.2, sr: int = 22050) -> None:
    t = np.arange(int(duration_sec * sr)) / sr
    signal = np.zeros_like(t)
    for beat in [0.0, 0.4, 0.8]:
        pulse = np.sin(2 * math.pi * 440 * t) * np.exp(-((t - beat) ** 2) / 0.0008)
        signal += pulse
    signal = np.clip(signal / np.max(np.abs(signal)), -1.0, 1.0)
    pcm = (signal * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(pcm.tobytes())


def test_pipeline_analyzes_synthetic_video(tmp_path: Path) -> None:
    video_path = tmp_path / "battle.mp4"
    audio_path = tmp_path / "battle.wav"
    _write_test_video(video_path)
    _write_test_wav(audio_path)

    request = AnalyzeVideoRequest(
        video_path=video_path,
        segment=SegmentSelection(start_sec=0.0, end_sec=1.0),
        target_dancers=2,
        runtime_profile="balanced",
        overlay_layers=["boxes", "skeleton", "com", "beats"],
        audio_path=audio_path,
    )
    result = DanceAnalysisPipeline().analyze(request)

    assert len(result.tracks) == 2
    assert len(result.biomechanics) > 0
    assert result.render_bundle.annotated_video_path.exists()
    assert result.render_bundle.json_path.exists()
    assert result.music.bpm > 0


def test_assign_boxes_prefers_appearance_when_features_exist() -> None:
    left = BoundingBox(x=10, y=10, width=40, height=120)
    right = BoundingBox(x=200, y=10, width=40, height=120)
    previous_centers = [Point(x=20, y=40), Point(x=220, y=40)]
    previous_features = [
        np.asarray([1.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 1.0], dtype=np.float32),
    ]
    candidate_features = [
        np.asarray([0.0, 1.0], dtype=np.float32),
        np.asarray([1.0, 0.0], dtype=np.float32),
    ]

    assigned, _ = _assign_boxes(
        candidates=[left, right],
        previous_centers=previous_centers,
        target_dancers=2,
        previous_features=previous_features,
        candidate_features=candidate_features,
    )

    assert assigned == [right, left]
