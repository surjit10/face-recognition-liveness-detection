from pathlib import Path

import pytest

from faceguard.models import InsightFaceEmbedder, ModelPackError


def test_missing_model_pack_fails_before_download(tmp_path: Path) -> None:
    with pytest.raises(ModelPackError, match="automatic downloads are disabled"):
        InsightFaceEmbedder(
            model_root=tmp_path,
            pack="buffalo_m",
            expected_hashes={"detector.onnx": "a" * 64},
        )
