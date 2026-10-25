import importlib.util
import json
import shutil
from pathlib import Path
import numpy as np
import pytest

from faceguard.gallery import Gallery, GalleryError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "faceguard_manage_gallery", PROJECT_ROOT / "scripts" / "manage_gallery.py"
)
assert SPEC is not None and SPEC.loader is not None
MG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MG)


@pytest.fixture
def sample_gallery_bundle(tmp_path: Path):
    emb = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    pids = np.array(["user_a", "user_a", "user_b", "user_c"])
    model_hash = "a" * 64
    manifest_hash = "b" * 64
    hashes = np.array(["1" * 64, "2" * 64, "3" * 64, "4" * 64])

    gallery = Gallery(
        emb,
        pids,
        embedding_model_sha256=model_hash,
        source_sha256=hashes,
    )

    gallery_path = tmp_path / "gallery.npz"
    gallery_hash = gallery.save(gallery_path, manifest_sha256=manifest_hash)

    json_path = tmp_path / "gallery.json"
    json_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "gallery_file": str(gallery_path),
                "gallery_sha256": gallery_hash,
                "gallery_manifest_sha256": manifest_hash,
                "embedding_model_sha256": model_hash,
                "identities": 3,
                "vectors": 4,
            }
        ),
        encoding="utf-8",
    )
    return gallery_path, json_path


def test_list_and_inspect_identities(sample_gallery_bundle):
    gallery_path, _ = sample_gallery_bundle
    listed = MG.list_identities(gallery_path)
    assert listed["identities_count"] == 3
    assert listed["vectors_count"] == 4
    assert len(listed["identities"]) == 3

    inspected = MG.inspect_identity(gallery_path, "user_a")
    assert inspected["person_id"] == "user_a"
    assert inspected["vectors_count"] == 2


def test_delete_dry_run_preserves_files(sample_gallery_bundle):
    gallery_path, json_path = sample_gallery_bundle
    original_npz_bytes = gallery_path.read_bytes()
    original_json_bytes = json_path.read_bytes()

    res = MG.delete_identity(gallery_path, "user_a", dry_run=True, confirmed=False)
    assert res["status"] == "dry_run"
    assert res["vectors_to_remove"] == 2
    assert res["projected_identities"] == 2

    assert gallery_path.read_bytes() == original_npz_bytes
    assert json_path.read_bytes() == original_json_bytes


def test_delete_without_confirmation_raises(sample_gallery_bundle):
    gallery_path, _ = sample_gallery_bundle
    with pytest.raises(GalleryError, match="requires explicit confirmation"):
        MG.delete_identity(gallery_path, "user_a", dry_run=False, confirmed=False)


def test_delete_confirmed_removes_subject_and_updates_sidecar(sample_gallery_bundle):
    gallery_path, json_path = sample_gallery_bundle
    res = MG.delete_identity(gallery_path, "user_a", dry_run=False, confirmed=True)

    assert res["status"] == "deleted"
    assert res["remaining_identities"] == 2
    assert res["remaining_vectors"] == 2

    # Check updated files
    verified = MG.verify_gallery(gallery_path)
    assert verified["valid"] is True
    assert verified["identities"] == 2
    assert verified["vectors"] == 2
    assert verified["sidecar_consistent"] is True

    # User A is gone, B and C remain
    listed = MG.list_identities(gallery_path)
    pids = [entry["person_id"] for entry in listed["identities"]]
    assert "user_a" not in pids
    assert "user_b" in pids
    assert "user_c" in pids


def test_delete_unknown_identity_raises(sample_gallery_bundle):
    gallery_path, _ = sample_gallery_bundle
    with pytest.raises(GalleryError, match="Person ID not found"):
        MG.delete_identity(gallery_path, "non_existent_user", dry_run=False, confirmed=True)


def test_delete_leaving_less_than_two_identities_is_blocked(sample_gallery_bundle):
    gallery_path, _ = sample_gallery_bundle
    # Delete user_a -> 2 remain (user_b, user_c)
    MG.delete_identity(gallery_path, "user_a", dry_run=False, confirmed=True)
    # Attempting to delete user_b should leave only user_c, which is <2
    with pytest.raises(GalleryError, match="must retain at least two identities"):
        MG.delete_identity(gallery_path, "user_b", dry_run=False, confirmed=True)
