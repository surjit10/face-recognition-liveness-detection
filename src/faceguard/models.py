"""Explicit, fail-closed InsightFace detection and embedding adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import sha256_file


class ModelPackError(RuntimeError):
    """The configured face model package is missing, ambiguous, or incompatible."""


@dataclass(frozen=True)
class EmbeddingObservation:
    face_count: int
    embedding: np.ndarray | None
    detection_score: float | None
    bbox_xyxy: tuple[float, float, float, float] | None
    error: str | None
    face_boxes_xyxy: tuple[tuple[float, float, float, float], ...] = ()


class InsightFaceEmbedder:
    """Load one local InsightFace pack without permitting automatic downloads."""

    def __init__(
        self,
        *,
        model_root: str | Path,
        pack: str = "buffalo_m",
        providers: Sequence[str] = ("CPUExecutionProvider",),
        det_size: tuple[int, int] = (640, 640),
        expected_embedding_dim: int = 512,
        expected_hashes: dict[str, str] | None = None,
    ) -> None:
        self.model_root = Path(model_root).resolve()
        self.pack = pack
        self.pack_dir = self.model_root / "models" / pack
        self.providers = tuple(providers)
        self.det_size = det_size
        self.expected_embedding_dim = expected_embedding_dim
        if not self.pack_dir.is_dir():
            raise ModelPackError(
                f"Model pack is absent: {self.pack_dir}. Install it explicitly; "
                "automatic downloads are disabled by this wrapper."
            )
        self.model_files = sorted(self.pack_dir.glob("*.onnx"))
        if len(self.model_files) < 2:
            raise ModelPackError(
                f"Expected detector and recognizer ONNX files in {self.pack_dir}"
            )
        if not self.providers:
            raise ModelPackError("At least one explicit ONNX Runtime provider is required")
        if not expected_hashes:
            raise ModelPackError("Frozen detector/recognizer file hashes are required")
        for filename, expected_hash in expected_hashes.items():
            model_path = self.pack_dir / filename
            if not model_path.is_file():
                raise ModelPackError(f"Frozen model file is absent: {model_path}")
            actual_hash = sha256_file(model_path)
            if actual_hash != expected_hash:
                raise ModelPackError(
                    f"Model hash mismatch for {filename}: expected {expected_hash}, got {actual_hash}"
                )

        import onnxruntime as ort
        from insightface.app import FaceAnalysis

        available = set(ort.get_available_providers())
        unavailable = [provider for provider in self.providers if provider not in available]
        if unavailable:
            raise ModelPackError(
                f"Requested providers unavailable: {unavailable}; available={sorted(available)}"
            )
        try:
            self._app = FaceAnalysis(
                name=self.pack,
                root=str(self.model_root),
                allowed_modules=["detection", "recognition"],
                providers=list(self.providers),
            )
            ctx_id = 0 if self.providers[0] == "CUDAExecutionProvider" else -1
            self._app.prepare(ctx_id=ctx_id, det_size=self.det_size)
        except Exception as exc:
            raise ModelPackError(
                f"Unable to initialize frozen model pack {self.pack}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        runtime_providers: dict[str, tuple[str, ...]] = {}
        for task_name, model in self._app.models.items():
            session = getattr(model, "session", None)
            get_providers = getattr(session, "get_providers", None)
            if get_providers is None:
                raise ModelPackError(f"Model task has no inspectable ONNX session: {task_name}")
            task_providers = tuple(get_providers())
            if not task_providers or task_providers[0] != self.providers[0]:
                raise ModelPackError(
                    f"Model task {task_name} did not activate requested primary provider "
                    f"{self.providers[0]}: {task_providers}"
                )
            runtime_providers[task_name] = task_providers
        self.runtime_providers = runtime_providers

    @property
    def model_hashes(self) -> dict[str, str]:
        return {path.name: sha256_file(path) for path in self.model_files}

    def embed_rgb(self, image_rgb: np.ndarray) -> EmbeddingObservation:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            return EmbeddingObservation(0, None, None, None, "unsupported_image_shape")
        if image_rgb.dtype != np.uint8:
            return EmbeddingObservation(0, None, None, None, "unsupported_image_dtype")
        image_bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])
        faces = self._app.get(image_bgr)
        face_count = len(faces)
        face_boxes: list[tuple[float, float, float, float]] = []
        for detected_face in faces:
            bbox_array = np.asarray(
                getattr(detected_face, "bbox", None), dtype=np.float32
            ).reshape(-1)
            if bbox_array.shape != (4,) or not np.isfinite(bbox_array).all():
                continue
            bbox = tuple(float(value) for value in bbox_array)
            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                face_boxes.append(bbox)
        observed_boxes = tuple(face_boxes)
        if face_count != 1:
            error = "no_face" if face_count == 0 else "multiple_faces"
            return EmbeddingObservation(
                face_count,
                None,
                None,
                None,
                error,
                observed_boxes,
            )

        face = faces[0]
        raw = getattr(face, "embedding", None)
        if raw is None:
            return EmbeddingObservation(1, None, None, None, "embedding_missing")
        embedding = np.asarray(raw, dtype=np.float32).reshape(-1)
        if embedding.shape != (self.expected_embedding_dim,):
            return EmbeddingObservation(1, None, None, None, "embedding_dimension")
        if not np.isfinite(embedding).all():
            return EmbeddingObservation(1, None, None, None, "embedding_non_finite")
        norm = float(np.linalg.norm(embedding))
        if not np.isfinite(norm) or norm <= 1e-12:
            return EmbeddingObservation(1, None, None, None, "embedding_zero_norm")
        embedding = np.ascontiguousarray(embedding / norm, dtype=np.float32)

        if len(observed_boxes) != 1:
            return EmbeddingObservation(1, None, None, None, "bbox_invalid")
        bbox = observed_boxes[0]
        detection_score = float(face.det_score)
        if not np.isfinite(detection_score):
            return EmbeddingObservation(1, None, None, None, "detect_score_invalid")
        return EmbeddingObservation(
            1,
            embedding,
            detection_score,
            bbox,
            None,
            observed_boxes,
        )
