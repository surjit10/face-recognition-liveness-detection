"""Pure mappings from verification results to operator-safe presentation values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


REASON_MESSAGES = {
    "no_face": "LOOKING FOR FACE",
    "multiple_faces": "MULTIPLE FACES",
    "collecting_pad_evidence": "CHECKING LIVENESS",
    "pad_below_threshold": "POSSIBLE SPOOF / REJECTED",
    "face_too_small": "MOVE CLOSER",
    "face_too_blurry": "HOLD STILL",
    "exposure_out_of_range": "IMPROVE LIGHTING",
    "low_detection_score": "CENTER YOUR FACE",
}

REASON_GUIDANCE = {
    "no_face": "Position one face inside the camera view.",
    "multiple_faces": "Only one person may be visible.",
    "collecting_pad_evidence": "Hold still while liveness evidence is collected.",
    "pad_below_threshold": (
        "Liveness could not be confirmed. Face the camera, improve lighting and retry."
    ),
    "face_too_small": "Move closer to the camera.",
    "face_too_blurry": "Hold still and keep the face in focus.",
    "exposure_out_of_range": "Use even lighting and avoid strong backlight.",
    "low_detection_score": "Center one clearly visible face in the camera view.",
    "missing_face_box": "Center one clearly visible face in the camera view.",
    "empty_face_crop": "Move fully into the camera view.",
    "timestamp_regression": "The stream was reset for timing safety.",
    "face_model_error": "Face analysis failed safely. Try restarting the session.",
    "pad_model_error": "Liveness analysis failed safely. Try restarting the session.",
    "gallery_error": "Gallery matching failed safely. Check system readiness.",
}

STATUS_COLORS = {
    "verified": "success",
    "spoof": "warning",
    "unknown": "warning",
    "retry": "neutral",
    "ended": "neutral",
}


@dataclass(frozen=True)
class FaceAnnotation:
    bbox_xyxy: tuple[float, float, float, float]
    label: str
    tone: str


def operator_message(
    result: Any,
    *,
    identity_labels: Mapping[str, str] | None = None,
) -> str:
    """Return a concise message without exposing identity before full acceptance."""
    person_id = getattr(result, "person_id", None)
    status = getattr(result, "status", "retry")
    if status == "verified" and person_id:
        display_label = (
            identity_labels.get(str(person_id), str(person_id))
            if identity_labels is not None
            else str(person_id)
        )
        return f"VERIFIED: {display_label}"
    if status == "spoof":
        return "POSSIBLE SPOOF / REJECTED"
    if status == "unknown":
        return "UNKNOWN"
    return REASON_MESSAGES.get(getattr(result, "reason", ""), "HOLD STILL")


def operator_guidance(result: Any) -> str:
    """Return a human-readable next action for the current result."""
    status = getattr(result, "status", "retry")
    if status == "verified" and getattr(result, "person_id", None):
        return "Liveness passed and the gallery accepted this identity."
    if status == "verified":
        return "Identity output was withheld because the accepted label was missing."
    if status == "unknown":
        return "Liveness passed, but no enrolled identity met the threshold."
    return REASON_GUIDANCE.get(
        getattr(result, "reason", ""),
        "Keep one face centered, well lit and steady.",
    )


def status_color_key(result: Any) -> str:
    return STATUS_COLORS.get(getattr(result, "status", "retry"), "neutral")


def face_annotations(
    result: Any,
    *,
    identity_labels: Mapping[str, str] | None = None,
) -> tuple[FaceAnnotation, ...]:
    """Build memory-only overlay labels without weakening single-face verification."""
    boxes = tuple(getattr(result, "face_boxes_xyxy", ()) or ())
    if not boxes:
        return ()
    reason = getattr(result, "reason", "")
    status = getattr(result, "status", "retry")
    person_id = getattr(result, "person_id", None)

    if reason == "multiple_faces" or len(boxes) > 1:
        return tuple(
            FaceAnnotation(box, f"FACE {index}: IDENTITY NOT CHECKED", "danger")
            for index, box in enumerate(boxes, start=1)
        )

    box = boxes[0]
    if status == "verified" and person_id:
        label = (
            identity_labels.get(str(person_id), str(person_id))
            if identity_labels is not None
            else str(person_id)
        )
        return (FaceAnnotation(box, f"VERIFIED: {label}", "success"),)
    if status == "unknown":
        return (FaceAnnotation(box, "UNKNOWN", "warning"),)
    if status == "spoof":
        return (FaceAnnotation(box, "LIVENESS NOT CONFIRMED", "warning"),)
    if reason == "collecting_pad_evidence":
        return (FaceAnnotation(box, "CHECKING LIVENESS", "neutral"),)
    return (FaceAnnotation(box, "FACE DETECTED: HOLD STILL", "neutral"),)


def safe_event(sequence: int, result: Any) -> dict[str, object]:
    """Add sequence metadata to the pipeline's log-safe event."""
    return {"sequence": int(sequence), **result.event()}
