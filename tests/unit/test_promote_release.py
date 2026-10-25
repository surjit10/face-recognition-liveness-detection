import importlib.util
import json
from pathlib import Path
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "faceguard_promote_release", PROJECT_ROOT / "scripts" / "promote_release.py"
)
assert SPEC is not None and SPEC.loader is not None
PR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PR)


@pytest.fixture
def promotion_env(tmp_path: Path):
    model_dir = tmp_path / "models/private/insightface/models/buffalo_m"
    model_dir.mkdir(parents=True)
    det_file = model_dir / "det.onnx"
    det_file.write_bytes(b"detector_weights")
    rec_file = model_dir / "rec.onnx"
    rec_file.write_bytes(b"recognizer_weights")

    pad_dir = tmp_path / "models/private/pad"
    pad_dir.mkdir(parents=True)
    pad_file = pad_dir / "pad.onnx"
    pad_file.write_bytes(b"pad_weights")

    gallery_dir = tmp_path / "models/private/gallery"
    gallery_dir.mkdir(parents=True)
    gallery_file = gallery_dir / "gallery.npz"
    gallery_file.write_bytes(b"gallery_weights")

    lock_file = tmp_path / "requirements-runtime.lock"
    lock_file.write_text("package==1.0.0\n", encoding="utf-8")

    from faceguard.config import sha256_file

    config = {
        "schema_version": 1,
        "release_id": "proto-v1",
        "providers": ["CPUExecutionProvider"],
        "face": {
            "model_root": "models/private/insightface",
            "pack": "buffalo_m",
            "detector_file": "det.onnx",
            "detector_sha256": sha256_file(det_file),
            "recognizer_file": "rec.onnx",
            "recognizer_sha256": sha256_file(rec_file),
            "gallery_file": "models/private/gallery/gallery.npz",
            "gallery_sha256": sha256_file(gallery_file),
            "threshold": 0.65,
            "min_top1_top2_margin": 0.0,
        },
        "pad": {
            "file": "models/private/pad/pad.onnx",
            "sha256": sha256_file(pad_file),
            "threshold": 0.75,
            "crop_scale": 1.3,
            "max_clipped_fraction": 0.2,
            "input_size": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "runtime": {
            "pad_min_observations": 5,
            "max_track_gap_ms": 500.0,
            "min_iou": 0.3,
            "min_detection_score": 0.6,
            "min_face_size_px": 80,
            "min_blur_variance": 40.0,
            "min_mean_brightness": 40.0,
            "max_mean_brightness": 215.0,
            "frame_queue_size": 2,
        },
    }
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text(yaml.dump(config), encoding="utf-8")
    return tmp_path, config_path


def test_promotion_checks_succeed_for_academic_prototype(promotion_env):
    root, config_path = promotion_env
    checks = PR.run_checks(
        config_path, scope="academic-prototype", project_root=root, run_pytest=False
    )
    assert checks["manifest"]["status"] == "passed"
    assert checks["runtime_config"]["status"] == "passed"
    assert checks["scope"]["status"] == "passed"


def test_promotion_checks_block_production_security(promotion_env):
    root, config_path = promotion_env
    with pytest.raises(PR.PromotionError, match="Promotion to 'production-security' is blocked"):
        PR.run_checks(
            config_path, scope="production-security", project_root=root, run_pytest=False
        )


def test_record_promotion_receipt(promotion_env, tmp_path):
    receipt_dir = tmp_path / "receipts"
    checks = {"test_check": {"status": "passed"}}
    receipt_file = PR.record_promotion(
        "proto-v1", "academic-prototype", "reviewer_surjit", checks, output_dir=receipt_dir
    )

    assert receipt_file.is_file()
    data = json.loads(receipt_file.read_text(encoding="utf-8"))
    assert data["status"] == "promoted"
    assert data["promoted_by"] == "reviewer_surjit"
    assert data["release_id"] == "proto-v1"
    assert len(data["limitations"]) > 0
