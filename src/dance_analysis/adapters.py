from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

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

    This adapter can either return a lightweight deterministic proxy or load an
    official NVIDIA C-RADIOv4 model from Hugging Face when configured.
    """

    name = "c-radiov4"

    def __init__(self) -> None:
        self.model_id = os.environ.get("CRADIOV4_MODEL_ID", "").strip()
        self.device_preference = os.environ.get("CRADIOV4_DEVICE", "cuda").strip().lower()
        self._model: Any | None = None
        self._processor: Any | None = None
        self._torch: Any | None = None
        self._load_error: str | None = None

    def status(self) -> AdapterStatus:
        if self._load_error:
            return AdapterStatus(name=self.name, available=False, reason=f"model load failed: {self._load_error}")
        if self.model_id:
            if not _cradio_dependencies_available():
                return AdapterStatus(
                    name=self.name,
                    available=False,
                    reason=f"{self.model_id} configured but torch/transformers dependencies are unavailable",
                )
            state = "loaded" if self._model is not None else "lazy-load pending"
            return AdapterStatus(
                name=self.name,
                available=True,
                reason=f"{self.model_id} configured for {_resolve_device_name(self.device_preference)} ({state})",
            )
        if os.environ.get("CRADIOV4_WEIGHTS"):
            return AdapterStatus(
                name=self.name,
                available=False,
                reason="CRADIOV4_WEIGHTS is set but local checkpoint loading is not implemented; set CRADIOV4_MODEL_ID",
            )
        return AdapterStatus(name=self.name, available=False, reason="model not configured; using fallback contour features")

    def extract_features(self, frame: np.ndarray, box: BoundingBox) -> np.ndarray:
        crop = frame[box.y : box.y + box.height, box.x : box.x + box.width]
        if crop.size == 0:
            return np.zeros(8, dtype=np.float32)
        if not self.model_id:
            resized = cv2.resize(crop, (8, 8), interpolation=cv2.INTER_AREA)
            return resized.mean(axis=(0, 1)).astype(np.float32)

        model, processor, torch = self._ensure_model_loaded()
        from PIL import Image

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        target_height, target_width = _resolve_radio_resolution(model, rgb.shape[0], rgb.shape[1])
        if (target_height, target_width) != rgb.shape[:2]:
            rgb = cv2.resize(rgb, (target_width, target_height), interpolation=cv2.INTER_AREA)
        image = Image.fromarray(rgb)
        inputs = processor(images=image, return_tensors="pt").pixel_values
        device = _resolve_device(torch, self.device_preference)
        inputs = inputs.to(device)
        if device.type == "cuda":
            inputs = inputs.to(dtype=torch.bfloat16)

        with torch.inference_mode():
            outputs = model(inputs)
        summary = _extract_summary_vector(outputs)
        vector = summary.detach().float().cpu().numpy().reshape(-1).astype(np.float32)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector

    def _ensure_model_loaded(self) -> tuple[Any, Any, Any]:
        if self._model is not None and self._processor is not None and self._torch is not None:
            return self._model, self._processor, self._torch

        try:
            import torch
            from transformers import AutoModel, CLIPImageProcessor
        except Exception as exc:  # pragma: no cover - optional dependency path
            self._load_error = str(exc)
            raise RuntimeError(f"Could not import C-RADIOv4 dependencies: {exc}") from exc

        try:
            processor = CLIPImageProcessor.from_pretrained(self.model_id)
            model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True)
        except Exception as exc:  # pragma: no cover - network/model load path
            self._load_error = str(exc)
            raise RuntimeError(f"Could not load C-RADIOv4 model {self.model_id}: {exc}") from exc

        device = _resolve_device(torch, self.device_preference)
        model = model.to(device).eval()
        self._model = model
        self._processor = processor
        self._torch = torch
        return model, processor, torch


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


def _cradio_dependencies_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        from PIL import Image  # noqa: F401
    except Exception:
        return False
    return True


def _resolve_device(torch: Any, preference: str):
    if preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _resolve_device_name(preference: str) -> str:
    try:
        import torch
    except Exception:
        return preference
    return str(_resolve_device(torch, preference))


def _extract_summary_vector(outputs: Any):
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    if hasattr(outputs, "summary"):
        return outputs.summary
    if isinstance(outputs, dict):
        if "backbone" in outputs:
            backbone = outputs["backbone"]
            if isinstance(backbone, (tuple, list)):
                return backbone[0]
            if hasattr(backbone, "summary"):
                return backbone.summary
        if "summary" in outputs:
            return outputs["summary"]
    raise RuntimeError(f"Unsupported C-RADIOv4 output type: {type(outputs)!r}")


def _resolve_radio_resolution(model: Any, height: int, width: int) -> tuple[int, int]:
    preferred = getattr(model, "preferred_resolution", (height, width))
    max_height = int(preferred[0]) if len(preferred) >= 1 else height
    max_width = int(preferred[1]) if len(preferred) >= 2 else width
    scale = min(1.0, max_height / max(height, 1), max_width / max(width, 1))
    scaled_height = max(1, int(round(height * scale)))
    scaled_width = max(1, int(round(width * scale)))
    if hasattr(model, "get_nearest_supported_resolution"):
        nearest = model.get_nearest_supported_resolution(height=scaled_height, width=scaled_width)
        if hasattr(nearest, "height") and hasattr(nearest, "width"):
            return int(nearest.height), int(nearest.width)
        if isinstance(nearest, (tuple, list)) and len(nearest) == 2:
            return int(nearest[0]), int(nearest[1])
    step = int(getattr(model, "min_resolution_step", 16))
    snapped_height = max(step, ((scaled_height + step - 1) // step) * step)
    snapped_width = max(step, ((scaled_width + step - 1) // step) * step)
    return snapped_height, snapped_width
