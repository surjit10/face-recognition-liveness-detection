"""ONNX presentation-attack inference with the frozen preprocessing contract."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .config import sha256_file


class PadInferenceError(RuntimeError):
    """PAD preprocessing or inference could not produce a trustworthy score."""


@dataclass(frozen=True)
class PadPreprocessConfig:
    crop_scale: float = 1.3
    max_clipped_fraction: float = 0.2
    input_size: int = 224
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    def validate(self) -> None:
        if not np.isfinite(self.crop_scale) or self.crop_scale < 1.0:
            raise PadInferenceError("PAD crop scale must be finite and at least 1.0")
        if not 0.0 <= self.max_clipped_fraction < 1.0:
            raise PadInferenceError("Maximum clipped fraction must be in [0, 1)")
        if len(self.mean) != 3 or len(self.std) != 3:
            raise PadInferenceError("PAD mean and standard deviation must have three channels")
        if not all(np.isfinite(value) for value in (*self.mean, *self.std)):
            raise PadInferenceError("PAD normalization values must be finite")
        if self.input_size <= 0 or any(value <= 0 for value in self.std):
            raise PadInferenceError("PAD input size and standard deviations must be positive")


def prepare_pad_tensor(
    frame_bgr: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float],
    config: PadPreprocessConfig,
) -> np.ndarray:
    """Create one RGB/ImageNet-normalized PAD tensor from an OpenCV BGR frame."""
    config.validate()
    if (
        not isinstance(frame_bgr, np.ndarray)
        or frame_bgr.dtype != np.uint8
        or frame_bgr.ndim != 3
        or frame_bgr.shape[2] != 3
        or frame_bgr.shape[0] == 0
        or frame_bgr.shape[1] == 0
    ):
        raise PadInferenceError("Frame must be a non-empty uint8 BGR image")
    box = np.asarray(bbox_xyxy, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise PadInferenceError("Face box must contain four finite values")
    x1, y1, x2, y2 = box.tolist()
    if x2 <= x1 or y2 <= y1:
        raise PadInferenceError("Face box has non-positive area")
    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = config.crop_scale * max(x2 - x1, y2 - y1)
    requested = (
        math.floor(center_x - side / 2.0),
        math.floor(center_y - side / 2.0),
        math.ceil(center_x + side / 2.0),
        math.ceil(center_y + side / 2.0),
    )
    height, width = frame_bgr.shape[:2]
    clipped = (
        max(0, requested[0]),
        max(0, requested[1]),
        min(width, requested[2]),
        min(height, requested[3]),
    )
    requested_area = max(0, requested[2] - requested[0]) * max(
        0, requested[3] - requested[1]
    )
    clipped_area = max(0, clipped[2] - clipped[0]) * max(0, clipped[3] - clipped[1])
    if requested_area <= 0 or clipped_area <= 0:
        raise PadInferenceError("Expanded PAD crop is empty")
    clipped_fraction = 1.0 - clipped_area / requested_area
    if clipped_fraction > config.max_clipped_fraction:
        raise PadInferenceError(
            f"Expanded PAD crop clipped by {clipped_fraction:.4f}, exceeding policy"
        )
    crop = frame_bgr[clipped[1] : clipped[3], clipped[0] : clipped[2]]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(
        rgb, (config.input_size, config.input_size), interpolation=cv2.INTER_LINEAR
    )
    values = resized.astype(np.float32) / 255.0
    mean = np.asarray(config.mean, dtype=np.float32)
    std = np.asarray(config.std, dtype=np.float32)
    values = (values - mean) / std
    tensor = np.transpose(values, (2, 0, 1))[None, ...]
    if not np.isfinite(tensor).all():
        raise PadInferenceError("PAD preprocessing produced non-finite values")
    return np.ascontiguousarray(tensor, dtype=np.float32)


class OnnxPadModel:
    """Hash-verified CPU/GPU ONNX model; providers are never silently replaced."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        expected_sha256: str,
        providers: Sequence[str],
        preprocess: PadPreprocessConfig,
    ) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise PadInferenceError(f"PAD model is absent: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_sha256:
            raise PadInferenceError(
                f"PAD model hash mismatch: expected {expected_sha256}, got {actual_hash}"
            )
        preprocess.validate()
        if not providers:
            raise PadInferenceError("At least one explicit ONNX provider is required")
        import onnxruntime as ort

        unavailable = set(providers).difference(ort.get_available_providers())
        if unavailable:
            raise PadInferenceError(
                f"Requested ONNX providers are unavailable: {sorted(unavailable)}"
            )
        try:
            session = ort.InferenceSession(str(path), providers=list(providers))
        except Exception as exc:
            raise PadInferenceError(f"Unable to load PAD ONNX model: {exc}") from exc
        active = tuple(session.get_providers())
        if not active or active[0] != providers[0]:
            raise PadInferenceError(
                f"PAD model did not activate provider {providers[0]}: {active}"
            )
        inputs, outputs = session.get_inputs(), session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise PadInferenceError("PAD model must expose exactly one input and one output")
        shape = inputs[0].shape
        if len(shape) != 4 or shape[1:] != [3, preprocess.input_size, preprocess.input_size]:
            raise PadInferenceError(f"Unexpected PAD input shape: {shape}")
        self._session = session
        self._input_name = inputs[0].name
        self._output_name = outputs[0].name
        self.preprocess = preprocess
        self.model_sha256 = actual_hash

    def score_live(
        self, frame_bgr: np.ndarray, bbox_xyxy: tuple[float, float, float, float]
    ) -> float:
        tensor = prepare_pad_tensor(frame_bgr, bbox_xyxy, self.preprocess)
        try:
            output = self._session.run([self._output_name], {self._input_name: tensor})[0]
        except Exception as exc:
            raise PadInferenceError(f"PAD inference failed: {type(exc).__name__}") from exc
        logits = np.asarray(output, dtype=np.float32)
        if logits.size != 1 or not np.isfinite(logits).all():
            raise PadInferenceError(f"Unexpected PAD output shape or value: {logits.shape}")
        logit = float(logits.reshape(-1)[0])
        if logit >= 0:
            score = 1.0 / (1.0 + math.exp(-logit))
        else:
            exp_value = math.exp(logit)
            score = exp_value / (1.0 + exp_value)
        return float(score)
