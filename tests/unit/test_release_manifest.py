import importlib.util
import json
from pathlib import Path
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "faceguard_generate_release_manifest",
    PROJECT_ROOT / "scripts" / "generate_release_manifest.py",
)
assert SPEC is not None and SPEC.loader is not None
GRM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GRM)


@pytest.fixture
def mock_release_env(tmp_path: Path):
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
        "release_id": "test-v1",
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
            "min_top1_top2_margin": 0.05,
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
        "runtime": {"pad_min_observations": 5},
    }
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text(yaml.dump(config), encoding="utf-8")
    return tmp_path, config_path


def test_build_manifest_succeeds_with_valid_files(mock_release_env):
    root, config_path = mock_release_env
    manifest = GRM.build_manifest(config_path, project_root=root)

    assert manifest["schema_version"] == 1
    assert manifest["release_id"] == "test-v1"
    assert "detector" in manifest["artifacts"]
    assert "recognizer" in manifest["artifacts"]
    assert "pad_onnx" in manifest["artifacts"]
    assert "gallery" in manifest["artifacts"]


def test_missing_artifact_raises_error(mock_release_env):
    root, config_path = mock_release_env
    (root / "models/private/pad/pad.onnx").unlink()

    with pytest.raises(GRM.ReleaseManifestError, match="Required artifact pad_onnx missing"):
        GRM.build_manifest(config_path, project_root=root)


def test_hash_mismatch_raises_error(mock_release_env):
    root, config_path = mock_release_env
    (root / "models/private/pad/pad.onnx").write_bytes(b"tampered_weights")

    with pytest.raises(GRM.ReleaseManifestError, match="hash mismatch"):
        GRM.build_manifest(config_path, project_root=root)


def test_write_manifest_creates_valid_json(mock_release_env, tmp_path):
    root, config_path = mock_release_env
    manifest = GRM.build_manifest(config_path, project_root=root)
    out_file = tmp_path / "manifest.json"

    GRM.write_manifest(manifest, out_file)
    assert out_file.is_file()
    loaded = json.loads(out_file.read_text(encoding="utf-8"))
    assert loaded["release_id"] == "test-v1"
