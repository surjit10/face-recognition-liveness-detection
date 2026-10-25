import importlib.util
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "faceguard_run_app", PROJECT_ROOT / "scripts" / "run_app.py"
)
assert SPEC is not None and SPEC.loader is not None
RUN_APP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN_APP)


def test_app_defaults_to_authorized_camera_zero() -> None:
    args = RUN_APP.parse_args([])
    assert args.camera == 0
    assert args.config == PROJECT_ROOT / "configs/runtime.yaml"
    assert args.identity_labels == PROJECT_ROOT / "configs/private/identity_labels.yaml"


def test_app_rejects_unapproved_camera_index() -> None:
    with pytest.raises(SystemExit) as exc:
        RUN_APP.parse_args(["--camera", "1"])
    assert exc.value.code == 2
