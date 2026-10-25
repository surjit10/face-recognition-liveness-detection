"""Strict runtime configuration loader; incomplete candidates cannot start."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ConfigurationError, load_yaml_mapping
from .pad import PadPreprocessConfig
from .pipeline import PipelinePolicy


def _exact(mapping: Any, expected: set[str], location: str) -> dict[str, Any]:
    if not isinstance(mapping, dict):
        raise ConfigurationError(f"{location} must be a mapping")
    keys = set(mapping)
    if keys != expected:
        raise ConfigurationError(
            f"{location} keys mismatch; missing={sorted(expected - keys)}, "
            f"unknown={sorted(keys - expected)}"
        )
    return mapping


def _text(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{location} must be a non-empty string")
    return value


def _number(value: Any, location: str) -> float:
    if not isinstance(value, (int, float)):
        raise ConfigurationError(f"{location} must be a concrete number")
    return float(value)


def _hash(value: Any, location: str) -> str:
    result = _text(value, location).lower()
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ConfigurationError(f"{location} must be a SHA-256 hex digest")
    return result


def _path(root: Path, value: Any, location: str) -> Path:
    relative = Path(_text(value, location))
    if relative.is_absolute() or ".." in relative.parts:
        raise ConfigurationError(f"{location} must be a project-relative safe path")
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ConfigurationError(f"{location} escapes the project root")
    return result


@dataclass(frozen=True)
class RuntimeConfig:
    release_id: str
    providers: tuple[str, ...]
    insightface_model_root: Path
    insightface_pack: str
    detector_file: str
    detector_sha256: str
    recognizer_file: str
    recognizer_sha256: str
    gallery_path: Path
    gallery_sha256: str
    pad_path: Path
    pad_sha256: str
    pad_preprocess: PadPreprocessConfig
    pipeline_policy: PipelinePolicy
    frame_queue_size: int


def load_runtime_config(path: str | Path, *, project_root: str | Path) -> RuntimeConfig:
    document = _exact(
        load_yaml_mapping(path),
        {"schema_version", "release_id", "providers", "face", "pad", "runtime"},
        "runtime config",
    )
    if document["schema_version"] != 1:
        raise ConfigurationError("runtime config schema_version must be 1")
    providers = document["providers"]
    if not isinstance(providers, list) or not providers or not all(
        isinstance(value, str) and value for value in providers
    ):
        raise ConfigurationError("providers must be a non-empty string list")
    face = _exact(
        document["face"],
        {
            "model_root",
            "pack",
            "detector_file",
            "detector_sha256",
            "recognizer_file",
            "recognizer_sha256",
            "gallery_file",
            "gallery_sha256",
            "threshold",
            "min_top1_top2_margin",
        },
        "face",
    )
    pad = _exact(
        document["pad"],
        {
            "file",
            "sha256",
            "threshold",
            "crop_scale",
            "max_clipped_fraction",
            "input_size",
            "mean",
            "std",
        },
        "pad",
    )
    runtime = _exact(
        document["runtime"],
        {
            "pad_min_observations",
            "max_track_gap_ms",
            "min_iou",
            "min_detection_score",
            "min_face_size_px",
            "min_blur_variance",
            "min_mean_brightness",
            "max_mean_brightness",
            "frame_queue_size",
        },
        "runtime",
    )
    root = Path(project_root).resolve()
    mean, std = pad["mean"], pad["std"]
    if not isinstance(mean, list) or len(mean) != 3 or not isinstance(std, list) or len(std) != 3:
        raise ConfigurationError("pad.mean and pad.std must contain exactly three numbers")
    preprocess = PadPreprocessConfig(
        crop_scale=_number(pad["crop_scale"], "pad.crop_scale"),
        max_clipped_fraction=_number(
            pad["max_clipped_fraction"], "pad.max_clipped_fraction"
        ),
        input_size=int(_number(pad["input_size"], "pad.input_size")),
        mean=tuple(_number(value, "pad.mean[]") for value in mean),
        std=tuple(_number(value, "pad.std[]") for value in std),
    )
    policy = PipelinePolicy(
        pad_threshold=_number(pad["threshold"], "pad.threshold"),
        recognition_threshold=_number(face["threshold"], "face.threshold"),
        recognition_min_margin=_number(
            face["min_top1_top2_margin"], "face.min_top1_top2_margin"
        ),
        pad_min_observations=int(
            _number(runtime["pad_min_observations"], "runtime.pad_min_observations")
        ),
        max_track_gap_ms=_number(runtime["max_track_gap_ms"], "runtime.max_track_gap_ms"),
        min_iou=_number(runtime["min_iou"], "runtime.min_iou"),
        min_detection_score=_number(
            runtime["min_detection_score"], "runtime.min_detection_score"
        ),
        min_face_size_px=int(
            _number(runtime["min_face_size_px"], "runtime.min_face_size_px")
        ),
        min_blur_variance=_number(
            runtime["min_blur_variance"], "runtime.min_blur_variance"
        ),
        min_mean_brightness=_number(
            runtime["min_mean_brightness"], "runtime.min_mean_brightness"
        ),
        max_mean_brightness=_number(
            runtime["max_mean_brightness"], "runtime.max_mean_brightness"
        ),
    )
    preprocess.validate()
    policy.validate()
    frame_queue_size = int(
        _number(runtime["frame_queue_size"], "runtime.frame_queue_size")
    )
    if frame_queue_size <= 0:
        raise ConfigurationError("runtime.frame_queue_size must be positive")
    return RuntimeConfig(
        release_id=_text(document["release_id"], "release_id"),
        providers=tuple(providers),
        insightface_model_root=_path(root, face["model_root"], "face.model_root"),
        insightface_pack=_text(face["pack"], "face.pack"),
        detector_file=_text(face["detector_file"], "face.detector_file"),
        detector_sha256=_hash(face["detector_sha256"], "face.detector_sha256"),
        recognizer_file=_text(face["recognizer_file"], "face.recognizer_file"),
        recognizer_sha256=_hash(face["recognizer_sha256"], "face.recognizer_sha256"),
        gallery_path=_path(root, face["gallery_file"], "face.gallery_file"),
        gallery_sha256=_hash(face["gallery_sha256"], "face.gallery_sha256"),
        pad_path=_path(root, pad["file"], "pad.file"),
        pad_sha256=_hash(pad["sha256"], "pad.sha256"),
        pad_preprocess=preprocess,
        pipeline_policy=policy,
        frame_queue_size=frame_queue_size,
    )
