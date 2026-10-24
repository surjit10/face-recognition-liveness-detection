from pathlib import Path

import numpy as np
import pytest

from faceguard.preprocessing import (
    PreprocessingError,
    _expanded_square_crop,
    _largest_face_index,
    _mask_secondary_faces,
    _validate_unique_outputs,
)


def test_expanded_crop_keeps_context_and_records_padding() -> None:
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    image[2:12, 0:8] = (10, 20, 30)
    crop, crop_box, padding = _expanded_square_crop(
        image, (0.0, 2.0, 8.0, 10.0), crop_factor=2.0
    )

    assert crop.shape == (20, 20, 3)
    assert crop_box == (-6, -3, 14, 17)
    assert padding == (6, 3, 0, 0)


def test_output_name_collisions_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "source"
    first = root / "0001" / "face.jpg"
    second = root / "0001" / "face.png"
    first.parent.mkdir(parents=True)
    first.touch()
    second.touch()

    with pytest.raises(PreprocessingError, match="collide"):
        _validate_unique_outputs([first, second], root)


def test_largest_face_selection_uses_bbox_area() -> None:
    faces = np.zeros((3, 15), dtype=np.float32)
    faces[:, 2:4] = ((10, 20), (30, 10), (12, 40))

    assert _largest_face_index(faces) == 2


def test_secondary_faces_are_masked_without_touching_selected_face() -> None:
    context = np.full((100, 100, 3), 50, dtype=np.uint8)
    context[20:50, 20:50] = 200
    context[60:80, 70:90] = 150
    faces = np.zeros((2, 15), dtype=np.float32)
    faces[0, :4] = (20, 20, 30, 30)
    faces[1, :4] = (70, 60, 20, 20)

    masked, boxes = _mask_secondary_faces(
        context, (0, 0, 100, 100), faces, selected_index=0, margin_ratio=0.0
    )

    assert boxes == [(70, 60, 90, 80)]
    assert np.array_equal(masked[20:50, 20:50], context[20:50, 20:50])
    assert not np.array_equal(masked[60:80, 70:90], context[60:80, 70:90])
