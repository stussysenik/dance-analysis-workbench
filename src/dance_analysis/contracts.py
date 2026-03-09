from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class Point(BaseModel):
    x: float
    y: float


class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int

    @property
    def centroid(self) -> Point:
        return Point(x=self.x + self.width / 2.0, y=self.y + self.height / 2.0)


class TrackFrame(BaseModel):
    frame_index: int
    timestamp_sec: float
    box: BoundingBox
    confidence: float = Field(ge=0.0, le=1.0)
    unstable: bool = False


class TrackRecord(BaseModel):
    dancer_id: str
    frames: list[TrackFrame]
    warnings: list[str] = Field(default_factory=list)


class PoseJoint(BaseModel):
    name: str
    point: Point
    confidence: float = Field(ge=0.0, le=1.0)


class PoseFrame(BaseModel):
    dancer_id: str
    frame_index: int
    timestamp_sec: float
    joints: list[PoseJoint]
    confidence: float = Field(ge=0.0, le=1.0)


class BiomechFrame(BaseModel):
    dancer_id: str
    frame_index: int
    timestamp_sec: float
    center_of_mass: Point
    velocity: float
    acceleration: float
    travel_distance: float
    expansion: float
    confidence: float = Field(ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)


class BeatEvent(BaseModel):
    timestamp_sec: float
    strength: float


class MusicAnalysis(BaseModel):
    bpm: float
    beat_times: list[float]
    motion_event_times: list[float]
    on_beat_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)


class SegmentSelection(BaseModel):
    start_sec: float = Field(ge=0.0)
    end_sec: float = Field(gt=0.0)


class AnalyzeVideoRequest(BaseModel):
    video_path: Path
    segment: SegmentSelection
    target_dancers: int = Field(ge=1, le=2)
    runtime_profile: str
    overlay_layers: list[str] = Field(default_factory=list)
    correction_hints: str = ""
    audio_path: Path | None = None


class RenderBundle(BaseModel):
    run_id: str
    output_dir: Path
    annotated_video_path: Path
    preview_frame_path: Path
    json_path: Path


class AnalysisResult(BaseModel):
    request: AnalyzeVideoRequest
    tracks: list[TrackRecord]
    poses: list[PoseFrame]
    biomechanics: list[BiomechFrame]
    music: MusicAnalysis
    render_bundle: RenderBundle
    summary: dict[str, object]
