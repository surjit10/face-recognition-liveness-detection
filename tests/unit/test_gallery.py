from pathlib import Path

import numpy as np
import pytest

from faceguard.config import sha256_file
from faceguard.gallery import Gallery, GalleryError, l2_normalize


MODEL_HASH = "a" * 64


def _gallery() -> Gallery:
    return Gallery(
        np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32),
        np.asarray(["person-a", "person-a", "person-b"]),
        embedding_model_sha256=MODEL_HASH,
        source_sha256=np.asarray(["1" * 64, "2" * 64, "3" * 64]),
    )


def test_l2_normalize_rejects_invalid_vectors() -> None:
    assert np.allclose(l2_normalize(np.asarray([3.0, 4.0])), [0.6, 0.8])
    with pytest.raises(GalleryError):
        l2_normalize(np.zeros(2))
    with pytest.raises(GalleryError):
        l2_normalize(np.asarray([np.nan, 1.0]))


def test_gallery_accepts_only_threshold_and_margin_pass() -> None:
    gallery = _gallery()
    accepted = gallery.search(np.asarray([1.0, 0.0]), threshold=1.0, min_margin=0.5)
    assert accepted.status == "verified"
    assert accepted.person_id == "person-a"

    below = gallery.search(np.asarray([0.8, 0.6]), threshold=0.99, min_margin=0.0)
    assert below.status == "unknown"
    assert below.person_id is None
    assert below.reason == "below_threshold"

    ambiguous = gallery.search(np.asarray([0.71, 0.70]), threshold=0.5, min_margin=0.1)
    assert ambiguous.status == "unknown"
    assert ambiguous.reason == "ambiguous_margin"


def test_gallery_tie_is_unknown() -> None:
    gallery = Gallery(
        np.eye(2, dtype=np.float32),
        np.asarray(["person-a", "person-b"]),
        embedding_model_sha256=MODEL_HASH,
    )
    result = gallery.search(
        np.asarray([1.0, 1.0]), threshold=0.1, min_margin=0.0
    )
    assert result.status == "unknown"
    assert result.reason == "exact_or_numerical_tie"
    ranking = gallery.rank(np.asarray([1.0, 1.0]))
    assert ranking.tied is True
    assert ranking.margin == pytest.approx(0.0)


def test_gallery_save_load_enroll_and_remove(tmp_path: Path) -> None:
    gallery = _gallery()
    assert gallery.identity_ids == ("person-a", "person-b")
    assert gallery.vector_count("person-a") == 2
    assert gallery.source_hash_count("person-a") == 2
    assert gallery.vector_count("missing") == 0
    gallery.enroll(
        "person-c",
        np.asarray([[0.7, -0.7]]),
        source_sha256=np.asarray(["4" * 64]),
    )
    assert gallery.identity_count == 3
    assert gallery.remove("person-c") == 1
    target = tmp_path / "gallery.npz"
    gallery.save(target, manifest_sha256="f" * 64)
    loaded = Gallery.load(
        target,
        expected_sha256=sha256_file(target),
        expected_embedding_model_sha256=MODEL_HASH,
    )
    assert loaded.size == 3
    assert loaded.identity_count == 2
    with pytest.raises(GalleryError, match="model"):
        Gallery.load(
            target,
            expected_sha256=sha256_file(target),
            expected_embedding_model_sha256="b" * 64,
        )


def test_gallery_rejects_duplicate_sources_and_unsafe_removal() -> None:
    with pytest.raises(GalleryError, match="Duplicate"):
        Gallery(
            np.eye(2, dtype=np.float32),
            np.asarray(["a", "b"]),
            embedding_model_sha256=MODEL_HASH,
            source_sha256=np.asarray(["1" * 64, "1" * 64]),
        )
    with pytest.raises(GalleryError, match="fewer than two"):
        _gallery().remove("person-b")
