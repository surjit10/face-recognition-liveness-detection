import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "faceguard_verify_private_models", PROJECT_ROOT / "scripts" / "verify_private_models.py"
)
assert SPEC is not None and SPEC.loader is not None
VPM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VPM)


def test_verify_fails_on_missing_config(tmp_path: Path):
    non_existent = tmp_path / "missing.yaml"
    with pytest.raises(VPM.PrivateVerificationError, match="Config file not found"):
        VPM.verify_private_artifacts(non_existent, project_root=tmp_path)


def test_verify_real_private_artifacts_if_present():
    real_config = PROJECT_ROOT / "configs/runtime.yaml"
    if not (PROJECT_ROOT / "models/private/insightface/models/buffalo_m").is_dir():
        pytest.skip("Private models not mounted in this environment")

    report = VPM.verify_private_artifacts(real_config, project_root=PROJECT_ROOT)
    assert report["event"] == "runtime_ready"
    assert report["camera_opened"] is False
    assert report["gallery_identities"] >= 2
    assert "embedding" not in report
