import numpy as np
import pytest

from faceguard.pad import PadInferenceError, PadPreprocessConfig, prepare_pad_tensor


def test_pad_preprocessing_has_frozen_shape_and_rgb_order() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:, :, 2] = 255  # red in BGR storage
    tensor = prepare_pad_tensor(
        frame,
        (25.0, 25.0, 75.0, 75.0),
        PadPreprocessConfig(max_clipped_fraction=0.2),
    )
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == np.float32
    assert tensor[0, 0].mean() > tensor[0, 2].mean()


def test_pad_preprocessing_rejects_excessive_edge_clipping() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(PadInferenceError, match="clipped"):
        prepare_pad_tensor(
            frame,
            (-30.0, -30.0, 20.0, 20.0),
            PadPreprocessConfig(max_clipped_fraction=0.1),
        )
