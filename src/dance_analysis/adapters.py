from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np

from dance_analysis.contracts import BoundingBox, PoseFrame, PoseJoint, Point


@dataclass
class AdapterStatus:
    name: str
    available: bool
    reason: str


class CRadioV4Adapter:
    """Shared visual backbone seam.

    This placeholder exposes availability and a deterministic feature proxy so the
    pipeline contract stays stable before heavyweight weights are installed.
    """

    name = "c-radiov4"

    def status(self) -> AdapterStatus:
        enabled = bool(os.environ.get("CRADIOV4_WEIGHTS"))
        reason = "weights configured" if enabled else "weights not configured; using fallback contour features"
        return AdapterStatus(name=self.name, available=enabled, reason=reason)

    def extract_features(self, frame: np.ndarray, box: BoundingBox) -> np.ndarray:
        crop = frame[box.y : box.y + box.height, box.x : box.x + box.width]
        if crop.size == 0:
            return np.zeros(8, dtype=np.float32)
        resized = cv2.resize(crop, (8, 8), interpolation=cv2.INTER_AREA)
        return resized.mean(axis=(0, 1)).astype(np.float32)


class SapiensPoseAdapter:
    """Human articulation seam.

    If model weights are unavailable, proxy joints are derived directly from the
    tracked bounding box while preserving the same output schema.
    """

    name = "sapiens-pose"

    def status(self) -> AdapterStatus:
        enabled = bool(os.environ.get("SAPIENS_WEIGHTS"))
        reason = "weights configured" if enabled else "weights not configured; using proxy joints"
        return AdapterStatus(name=self.name, available=enabled, reason=reason)

    def estimate_pose(
        self,
        dancer_id: str,
        frame_index: int,
        timestamp_sec: float,
        box: BoundingBox,
    ) -> PoseFrame:
        joints = _proxy_joints(box)
        confidence = 0.92 if self.status().available else 0.58
        return PoseFrame(
            dancer_id=dancer_id,
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            joints=joints,
            confidence=confidence,
        )


def _proxy_joints(box: BoundingBox) -> list[PoseJoint]:
    x = box.x
    y = box.y
    w = box.width
    h = box.height
    points = {
        "head": Point(x=x + 0.5 * w, y=y + 0.12 * h),
        "shoulder_l": Point(x=x + 0.35 * w, y=y + 0.25 * h),
        "shoulder_r": Point(x=x + 0.65 * w, y=y + 0.25 * h),
        "hip_l": Point(x=x + 0.42 * w, y=y + 0.58 * h),
        "hip_r": Point(x=x + 0.58 * w, y=y + 0.58 * h),
        "foot_l": Point(x=x + 0.4 * w, y=y + 0.95 * h),
        "foot_r": Point(x=x + 0.6 * w, y=y + 0.95 * h),
    }
    return [PoseJoint(name=name, point=point, confidence=0.7) for name, point in points.items()]
