from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "runtime_profiles.json"
ARTIFACTS_DIR = ROOT / "artifacts"
RUNS_DIR = ARTIFACTS_DIR / "runs"


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    sample_stride: int
    max_frames: int
    prefer_model_adapters: bool
    enable_3d_refine: bool
    detection_threshold: float
    track_stability_threshold: float
    beat_confidence_threshold: float


def ensure_directories() -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    RUNS_DIR.mkdir(exist_ok=True)


def load_runtime_profiles() -> dict[str, RuntimeProfile]:
    payload = json.loads(CONFIG_PATH.read_text())
    profiles: dict[str, RuntimeProfile] = {}
    for name, data in payload["profiles"].items():
        profiles[name] = RuntimeProfile(name=name, **data)
    return profiles
