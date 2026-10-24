"""Non-destructive preparation of context-preserving recognition images."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageOps

try:
    import cv2
except ModuleNotFoundError:  # permits manifest/crop helpers in audit-only environments
    cv2 = None  # type: ignore[assignment]

from .audit import SUPPORTED_SUFFIXES
from .config import sha256_file


class PreprocessingError(RuntimeError):
    """Recognition preprocessing cannot continue safely."""


MANIFEST_COLUMNS = (
    "schema_version",
    "candidate_person_id",
    "source_relative_path",
    "source_sha256",
    "source_width",
    "source_height",
    "exif_orientation",
    "face_count",
    "detection_score",
    "bbox_xywh",
    "landmarks_xy",
    "selected_face_index",
    "selection_policy",
    "selected_crop_factor",
    "output_face_count_yunet",
    "secondary_faces_masked",
    "secondary_mask_boxes_xyxy",
    "preprocess_status",
    "crop_box_xyxy",
    "padding_ltrb",
    "output_relative_path",
    "output_sha256",
    "output_width",
    "output_height",
    "error",
)

_DETECTOR: Any | None = None
_SETTINGS: dict[str, Any] = {}


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _candidate_files(root: Path) -> list[Path]:
    files = [
        path
        for person_dir in sorted(root.iterdir(), key=lambda item: item.name)
        if person_dir.is_dir() and not person_dir.name.startswith(".")
        for path in sorted(person_dir.iterdir(), key=lambda item: item.name)
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    if not files:
        raise PreprocessingError(f"No supported images found under {root}")
    return files


def _output_relative_path(source_relative_path: Path) -> Path:
    return source_relative_path.with_suffix(".jpg")


def _validate_unique_outputs(paths: Iterable[Path], root: Path) -> None:
    seen: dict[str, Path] = {}
    for path in paths:
        relative = path.relative_to(root)
        output = _output_relative_path(relative)
        key = output.as_posix().casefold()
        previous = seen.get(key)
        if previous is not None:
            raise PreprocessingError(
                f"Sources would collide at {output}: {previous} and {relative}"
            )
        seen[key] = relative


def _init_worker(model_path: str, settings: dict[str, Any]) -> None:
    global _DETECTOR, _SETTINGS
    if cv2 is None:
        raise PreprocessingError(
            "OpenCV is required; run this command with the project virtual environment"
        )
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    _SETTINGS = settings
    _DETECTOR = cv2.FaceDetectorYN.create(
        model_path,
        "",
        (320, 320),
        settings["score_threshold"],
        settings["nms_threshold"],
        settings["top_k"],
    )


def _detect_faces(image_rgb: np.ndarray) -> np.ndarray:
    if _DETECTOR is None:
        raise PreprocessingError("Worker detector is not initialized")
    height, width = image_rgb.shape[:2]
    scale = min(1.0, _SETTINGS["detector_max_side"] / max(height, width))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    image_bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])
    if scale < 1.0:
        image_bgr = cv2.resize(
            image_bgr,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )
    _DETECTOR.setInputSize((resized_width, resized_height))
    _, faces = _DETECTOR.detect(image_bgr)
    if faces is None:
        return np.empty((0, 15), dtype=np.float32)
    result = np.asarray(faces, dtype=np.float32).copy()
    result[:, :14] /= scale
    return result


def _channel_median(image_rgb: np.ndarray) -> np.ndarray:
    step_y = max(1, image_rgb.shape[0] // 128)
    step_x = max(1, image_rgb.shape[1] // 128)
    sample = image_rgb[::step_y, ::step_x]
    return np.median(sample.reshape(-1, 3), axis=0).astype(np.uint8)


def _largest_face_index(faces: np.ndarray) -> int:
    if len(faces) == 0:
        raise PreprocessingError("Cannot select a face from an empty detection result")
    areas = np.asarray(faces[:, 2] * faces[:, 3], dtype=np.float64)
    if not np.isfinite(areas).all() or np.any(areas <= 0.0):
        raise PreprocessingError("Detected face boxes contain invalid areas")
    return int(np.argmax(areas))


def _expanded_square_crop(
    image_rgb: np.ndarray,
    bbox_xywh: tuple[float, float, float, float],
    crop_factor: float,
) -> tuple[np.ndarray, tuple[int, int, int, int], tuple[int, int, int, int]]:
    height, width = image_rgb.shape[:2]
    x, y, box_width, box_height = bbox_xywh
    side = max(2, int(math.ceil(max(box_width, box_height) * crop_factor)))
    center_x = x + box_width / 2.0
    center_y = y + box_height / 2.0
    left = int(math.floor(center_x - side / 2.0))
    top = int(math.floor(center_y - side / 2.0))
    right = left + side
    bottom = top + side

    source_left = max(0, left)
    source_top = max(0, top)
    source_right = min(width, right)
    source_bottom = min(height, bottom)
    if source_right <= source_left or source_bottom <= source_top:
        raise PreprocessingError("Expanded face crop does not intersect the image")

    pad_left = source_left - left
    pad_top = source_top - top
    pad_right = right - source_right
    pad_bottom = bottom - source_bottom
    canvas = np.empty((side, side, 3), dtype=np.uint8)
    canvas[...] = _channel_median(image_rgb)
    canvas[
        pad_top : side - pad_bottom,
        pad_left : side - pad_right,
    ] = image_rgb[source_top:source_bottom, source_left:source_right]
    return (
        np.ascontiguousarray(canvas),
        (left, top, right, bottom),
        (pad_left, pad_top, pad_right, pad_bottom),
    )


def _mask_secondary_faces(
    context_rgb: np.ndarray,
    crop_box_xyxy: tuple[int, int, int, int],
    faces: np.ndarray,
    selected_index: int,
    margin_ratio: float = 0.15,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Replace non-selected detected face regions with a neutral solid color."""
    output = context_rgb.copy()
    crop_left, crop_top, _, _ = crop_box_xyxy
    height, width = output.shape[:2]
    fill = _channel_median(output)
    masked_source_boxes: list[tuple[int, int, int, int]] = []
    for index, face in enumerate(faces):
        if index == selected_index:
            continue
        x, y, box_width, box_height = (float(value) for value in face[:4])
        margin_x = box_width * margin_ratio
        margin_y = box_height * margin_ratio
        source_left = int(math.floor(x - margin_x))
        source_top = int(math.floor(y - margin_y))
        source_right = int(math.ceil(x + box_width + margin_x))
        source_bottom = int(math.ceil(y + box_height + margin_y))
        left = max(0, source_left - crop_left)
        top = max(0, source_top - crop_top)
        right = min(width, source_right - crop_left)
        bottom = min(height, source_bottom - crop_top)
        if right <= left or bottom <= top:
            continue
        output[top:bottom, left:right] = fill
        masked_source_boxes.append(
            (source_left, source_top, source_right, source_bottom)
        )
    return np.ascontiguousarray(output), masked_source_boxes


def _resize_square(image_rgb: np.ndarray, size: int) -> np.ndarray:
    interpolation = cv2.INTER_AREA if max(image_rgb.shape[:2]) > size else cv2.INTER_CUBIC
    return cv2.resize(image_rgb, (size, size), interpolation=interpolation)


def _resize_max_side(image_rgb: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    if max(height, width) <= max_side:
        return np.ascontiguousarray(image_rgb)
    scale = max_side / max(height, width)
    output_width = max(1, int(round(width * scale)))
    output_height = max(1, int(round(height * scale)))
    return cv2.resize(
        image_rgb,
        (output_width, output_height),
        interpolation=cv2.INTER_AREA,
    )


def _save_jpeg_atomic(image_rgb: np.ndarray, path: Path, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w+b", suffix=".jpg", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        Image.fromarray(image_rgb, mode="RGB").save(
            temporary,
            format="JPEG",
            quality=quality,
            subsampling=0,
            optimize=False,
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _process_one(arguments: tuple[str, str, str]) -> dict[str, Any]:
    source_string, input_root_string, staging_root_string = arguments
    source = Path(source_string)
    input_root = Path(input_root_string)
    staging_root = Path(staging_root_string)
    source_relative = source.relative_to(input_root)
    output_relative = _output_relative_path(source_relative)
    output_path = staging_root / output_relative
    row: dict[str, Any] = {
        "schema_version": 1,
        "candidate_person_id": source_relative.parts[0],
        "source_relative_path": source_relative.as_posix(),
        "source_sha256": "",
        "source_width": "",
        "source_height": "",
        "exif_orientation": "",
        "face_count": "",
        "detection_score": "",
        "bbox_xywh": "",
        "landmarks_xy": "",
        "selected_face_index": "",
        "selection_policy": "",
        "selected_crop_factor": "",
        "output_face_count_yunet": "",
        "secondary_faces_masked": 0,
        "secondary_mask_boxes_xyxy": "",
        "preprocess_status": "error",
        "crop_box_xyxy": "",
        "padding_ltrb": "",
        "output_relative_path": "",
        "output_sha256": "",
        "output_width": "",
        "output_height": "",
        "error": "",
    }
    try:
        row["source_sha256"] = sha256_file(source)
        with Image.open(source) as image:
            orientation = int(image.getexif().get(274, 1) or 1)
            image = ImageOps.exif_transpose(image).convert("RGB")
            image_rgb = np.asarray(image, dtype=np.uint8)
        height, width = image_rgb.shape[:2]
        row["source_width"] = width
        row["source_height"] = height
        row["exif_orientation"] = orientation
        faces = _detect_faces(image_rgb)
        face_count = len(faces)
        row["face_count"] = face_count

        if face_count == 1 or (
            face_count > 1 and _SETTINGS["multiple_face_policy"] == "largest"
        ):
            selected_index = 0 if face_count == 1 else _largest_face_index(faces)
            face = faces[selected_index]
            bbox = tuple(float(value) for value in face[:4])
            landmarks = np.asarray(face[4:14], dtype=np.float32).reshape(5, 2)
            row["detection_score"] = float(face[14])
            row["bbox_xywh"] = json.dumps(bbox)
            row["landmarks_xy"] = json.dumps(landmarks.tolist(), separators=(",", ":"))
            row["selected_face_index"] = selected_index
            if face_count == 1:
                factors = (_SETTINGS["crop_factor"],)
                row["selection_policy"] = "only_detected_face"
            else:
                factors = tuple(_SETTINGS["multiple_crop_factors"])
                row["selection_policy"] = "largest_bbox_area"

            output_rgb = np.empty((0, 0, 3), dtype=np.uint8)
            crop_box = (0, 0, 0, 0)
            padding = (0, 0, 0, 0)
            verified_face_count: int | str = ""
            selected_factor = factors[-1]
            for factor in factors:
                context, candidate_box, candidate_padding = _expanded_square_crop(
                    image_rgb, bbox, factor
                )
                masked_boxes: list[tuple[int, int, int, int]] = []
                if face_count > 1:
                    context, masked_boxes = _mask_secondary_faces(
                        context, candidate_box, faces, selected_index
                    )
                candidate = _resize_square(context, _SETTINGS["crop_size"])
                output_rgb = candidate
                crop_box = candidate_box
                padding = candidate_padding
                selected_factor = factor
                if face_count == 1:
                    break
                verified_face_count = len(_detect_faces(candidate))
                if verified_face_count == 1:
                    break

            row["selected_crop_factor"] = selected_factor
            row["output_face_count_yunet"] = verified_face_count
            row["secondary_faces_masked"] = len(masked_boxes)
            row["secondary_mask_boxes_xyxy"] = json.dumps(masked_boxes)
            if face_count == 1:
                row["preprocess_status"] = "context_crop_single_face"
            elif verified_face_count == 1:
                row["preprocess_status"] = "context_crop_largest_of_multiple_masked"
            else:
                row["preprocess_status"] = (
                    "context_crop_largest_of_multiple_masked_review"
                )
            row["crop_box_xyxy"] = json.dumps(crop_box)
            row["padding_ltrb"] = json.dumps(padding)
        else:
            output_rgb = _resize_max_side(image_rgb, _SETTINGS["fallback_max_side"])
            row["preprocess_status"] = (
                "full_frame_no_face" if face_count == 0 else "full_frame_multiple_faces"
            )

        _save_jpeg_atomic(output_rgb, output_path, _SETTINGS["jpeg_quality"])
        row["output_relative_path"] = output_relative.as_posix()
        row["output_sha256"] = sha256_file(output_path)
        row["output_width"] = output_rgb.shape[1]
        row["output_height"] = output_rgb.shape[0]
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def prepare_recognition_context_dataset(
    *,
    input_root: str | Path,
    output_root: str | Path,
    manifest_path: str | Path,
    report_path: str | Path,
    detector_model: str | Path,
    workers: int = 1,
    detector_max_side: int = 960,
    crop_factor: float = 2.0,
    crop_size: int = 768,
    fallback_max_side: int = 1600,
    score_threshold: float = 0.6,
    jpeg_quality: int = 95,
    multiple_face_policy: str = "full_frame",
) -> dict[str, Any]:
    source_root = Path(input_root).resolve(strict=True)
    destination = Path(output_root).resolve()
    manifest = Path(manifest_path).resolve()
    report = Path(report_path).resolve()
    model = Path(detector_model).resolve(strict=True)
    if not source_root.is_dir():
        raise PreprocessingError(f"Input root is not a directory: {source_root}")
    if workers not in {1, 2}:
        raise PreprocessingError("workers must be 1 or 2 to keep resource use bounded")
    if crop_factor < 1.5 or crop_factor > 3.0:
        raise PreprocessingError("crop_factor must be in [1.5, 3.0]")
    if crop_size < 256 or crop_size > 2048:
        raise PreprocessingError("crop_size must be in [256, 2048]")
    if fallback_max_side < 640 or fallback_max_side > 4096:
        raise PreprocessingError("fallback_max_side must be in [640, 4096]")
    if multiple_face_policy not in {"full_frame", "largest"}:
        raise PreprocessingError("multiple_face_policy must be 'full_frame' or 'largest'")
    for path in (destination, manifest, report):
        if path.exists():
            raise PreprocessingError(f"Refusing to overwrite existing output: {path}")

    sources = _candidate_files(source_root)
    _validate_unique_outputs(sources, source_root)
    staging = destination.with_name(f"{destination.name}.partial")
    if staging.exists():
        raise PreprocessingError(f"Partial output already exists: {staging}")
    staging.mkdir(parents=True)

    settings = {
        "detector_max_side": detector_max_side,
        "crop_factor": crop_factor,
        "crop_size": crop_size,
        "fallback_max_side": fallback_max_side,
        "score_threshold": score_threshold,
        "nms_threshold": 0.3,
        "top_k": 5000,
        "jpeg_quality": jpeg_quality,
        "multiple_face_policy": multiple_face_policy,
        "multiple_crop_factors": [2.0, 1.8, 1.6, 1.5, 1.4, 1.3],
    }
    arguments = [
        (path.as_posix(), source_root.as_posix(), staging.as_posix()) for path in sources
    ]
    if workers == 1:
        _init_worker(model.as_posix(), settings)
        rows = [_process_one(argument) for argument in arguments]
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(model.as_posix(), settings),
        ) as executor:
            rows = list(executor.map(_process_one, arguments, chunksize=1))

    failures = [row for row in rows if row["error"]]
    if failures:
        _atomic_csv(manifest, rows)
        raise PreprocessingError(
            f"{len(failures)} samples failed; partial outputs and manifest were retained"
        )

    os.replace(staging, destination)
    _atomic_csv(manifest, rows)
    status_counts = Counter(str(row["preprocess_status"]) for row in rows)
    report_value = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_root_name": source_root.name,
        "output_root_name": destination.name,
        "source_samples": len(rows),
        "output_samples": len(rows),
        "source_subject_folders": len({row["candidate_person_id"] for row in rows}),
        "status_counts": dict(sorted(status_counts.items())),
        "workers": workers,
        "opencv_threads_per_worker": 1,
        "detector": "OpenCV YuNet context-crop proposal only",
        "detector_model_sha256": sha256_file(model),
        "settings": settings,
        "manifest_sha256": sha256_file(manifest),
        "content_multiset_sha256": hashlib.sha256(
            "".join(sorted(str(row["output_sha256"]) for row in rows)).encode("ascii")
        ).hexdigest(),
        "limitations": [
            "The YuNet result proposes a context crop; it is not final SCRFD validation.",
            (
                "Multiple-face inputs select the largest detected box and require manual "
                "identity review."
                if multiple_face_policy == "largest"
                else "Multiple-face inputs are retained as full-frame fallbacks."
            ),
            "No-face inputs are retained as full-frame fallbacks.",
            "ArcFace extraction must still reject images that fail its single-face gate.",
            "Original source files were not modified or deleted.",
        ],
        "privacy": "restricted biometric; do not publish",
    }
    _atomic_json(report, report_value)
    return report_value
