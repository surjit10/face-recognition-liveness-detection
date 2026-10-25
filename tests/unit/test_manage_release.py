import importlib.util
import json
from pathlib import Path
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "faceguard_manage_release", PROJECT_ROOT / "scripts" / "manage_release.py"
)
assert SPEC is not None and SPEC.loader is not None
MR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MR)


@pytest.fixture
def releases_setup(tmp_path: Path):
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

    def make_config(release_id: str, threshold: float):
        cfg = {
            "schema_version": 1,
            "release_id": release_id,
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
                "threshold": threshold,
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
                "pad_min_observations": 8,
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
        cfg_p = tmp_path / f"{release_id}.yaml"
        cfg_p.write_text(yaml.dump(cfg), encoding="utf-8")
        return cfg_p

    cfg_v1 = make_config("release-v1", 0.65)
    cfg_v2 = make_config("release-v2", 0.70)
    releases_dir = tmp_path / "releases"

    return tmp_path, releases_dir, cfg_v1, cfg_v2


def test_register_and_switch_release(releases_setup):
    root, rel_dir, cfg_v1, cfg_v2 = releases_setup

    res1 = MR.register_release(cfg_v1, releases_dir=rel_dir, project_root=root)
    assert res1["status"] == "registered"
    assert res1["release_id"] == "release-v1"

    res2 = MR.register_release(cfg_v2, releases_dir=rel_dir, project_root=root)
    assert res2["status"] == "registered"
    assert res2["release_id"] == "release-v2"

    # Switch to v1
    sw1 = MR.switch_release("release-v1", releases_dir=rel_dir, project_root=root)
    assert sw1["status"] == "activated"
    assert sw1["active_release"] == "release-v1"

    status = MR.release_status(rel_dir)
    assert status["active_release"] == "release-v1"
    assert status["previous_release"] is None

    # Switch to v2
    sw2 = MR.switch_release("release-v2", releases_dir=rel_dir, project_root=root)
    assert sw2["status"] == "activated"
    assert sw2["active_release"] == "release-v2"
    assert sw2["previous_release"] == "release-v1"

    status = MR.release_status(rel_dir)
    assert status["active_release"] == "release-v2"
    assert status["previous_release"] == "release-v1"


def test_rollback_restores_previous_release(releases_setup):
    root, rel_dir, cfg_v1, cfg_v2 = releases_setup

    MR.register_release(cfg_v1, releases_dir=rel_dir, project_root=root)
    MR.register_release(cfg_v2, releases_dir=rel_dir, project_root=root)

    MR.switch_release("release-v1", releases_dir=rel_dir, project_root=root)
    MR.switch_release("release-v2", releases_dir=rel_dir, project_root=root)

    rb = MR.rollback_release(releases_dir=rel_dir, project_root=root)
    assert rb["status"] == "rolled_back"
    assert rb["active_release"] == "release-v1"
    assert rb["rolled_back_from"] == "release-v2"

    status = MR.release_status(rel_dir)
    assert status["active_release"] == "release-v1"


def test_corrupted_candidate_fails_and_preserves_active(releases_setup):
    root, rel_dir, cfg_v1, cfg_v2 = releases_setup

    MR.register_release(cfg_v1, releases_dir=rel_dir, project_root=root)
    MR.register_release(cfg_v2, releases_dir=rel_dir, project_root=root)
    MR.switch_release("release-v1", releases_dir=rel_dir, project_root=root)

    # Corrupt release_manifest.json of v2
    v2_manifest = rel_dir / "release-v2" / "release_manifest.json"
    data = json.loads(v2_manifest.read_text(encoding="utf-8"))
    data["artifacts"]["pad_onnx"]["sha256"] = "0" * 64
    v2_manifest.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(MR.ReleaseError, match="hash verification failed"):
        MR.switch_release("release-v2", releases_dir=rel_dir, project_root=root)

    status = MR.release_status(rel_dir)
    assert status["active_release"] == "release-v1"
