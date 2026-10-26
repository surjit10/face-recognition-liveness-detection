from types import SimpleNamespace

import pytest

from faceguard.presentation import (
    face_annotations,
    operator_guidance,
    operator_message,
    safe_event,
    status_color_key,
)


def _result(status="retry", reason="no_face", person_id=None):
    return SimpleNamespace(
        status=status,
        reason=reason,
        person_id=person_id,
        event=lambda: {
            "status": status,
            "person_id": person_id if status == "verified" else None,
        },
    )


@pytest.mark.parametrize(
    ("result", "message", "tone"),
    [
        (_result(reason="no_face"), "LOOKING FOR FACE", "neutral"),
        (_result(reason="multiple_faces"), "MULTIPLE FACES", "neutral"),
        (_result(reason="collecting_pad_evidence"), "CHECKING LIVENESS", "neutral"),
        (
            _result(status="spoof", reason="pad_below_threshold"),
            "POSSIBLE SPOOF / REJECTED",
            "warning",
        ),
        (_result(status="unknown", reason="below_threshold"), "UNKNOWN", "warning"),
        (_result(status="verified", reason="accepted", person_id="000"), "VERIFIED: 000", "success"),
    ],
)
def test_operator_mapping_is_fail_closed(result, message, tone) -> None:
    assert operator_message(result) == message
    assert status_color_key(result) == tone
    assert operator_guidance(result)


def test_verified_without_identity_does_not_display_a_placeholder_identity() -> None:
    result = _result(status="verified", reason="accepted", person_id=None)
    assert operator_message(result) == "HOLD STILL"


def test_private_display_label_does_not_change_canonical_event_identity() -> None:
    result = _result(status="verified", reason="accepted", person_id="000")
    assert operator_message(
        result, identity_labels={"000": "Display Name"}
    ) == "VERIFIED: Display Name"
    assert safe_event(1, result)["person_id"] == "000"


def test_safe_event_uses_pipeline_event_without_pixels() -> None:
    event = safe_event(9, _result(status="unknown", reason="below_threshold"))
    assert event["sequence"] == 9
    assert "frame" not in event
    assert "embedding" not in event
    assert "bbox" not in event


def test_verified_face_annotation_uses_private_display_label() -> None:
    result = _result(status="verified", reason="accepted", person_id="000")
    result.face_boxes_xyxy = ((10.0, 20.0, 80.0, 90.0),)
    annotations = face_annotations(
        result,
        identity_labels={"000": "Display Name"},
    )
    assert len(annotations) == 1
    assert annotations[0].label == "VERIFIED: Display Name"
    assert annotations[0].tone == "success"


def test_multiple_faces_are_pointed_out_without_identity_matching() -> None:
    result = _result(status="retry", reason="multiple_faces")
    result.face_boxes_xyxy = (
        (10.0, 20.0, 80.0, 90.0),
        (100.0, 30.0, 160.0, 100.0),
    )
    annotations = face_annotations(result, identity_labels={"000": "Display Name"})
    assert [item.label for item in annotations] == [
        "FACE 1: IDENTITY NOT CHECKED",
        "FACE 2: IDENTITY NOT CHECKED",
    ]
    assert all(item.tone == "danger" for item in annotations)
