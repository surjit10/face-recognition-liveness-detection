"""Protected in-memory ArcFace gallery with fail-closed open-set matching."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import sha256_file


class GalleryError(RuntimeError):
    """A gallery artifact or operation violates the recognition contract."""


def l2_normalize(values: np.ndarray) -> np.ndarray:
    """Return finite float32 unit vectors without silently repairing bad rows."""
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim not in (1, 2) or matrix.size == 0:
        raise GalleryError("Embeddings must be a non-empty 1D or 2D array")
    if not np.isfinite(matrix).all():
        raise GalleryError("Embeddings contain non-finite values")
    axis = 0 if matrix.ndim == 1 else 1
    norms = np.linalg.norm(matrix, axis=axis, keepdims=True)
    if np.any(norms <= 1e-12):
        raise GalleryError("Embeddings contain a zero-norm vector")
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


@dataclass(frozen=True)
class GalleryMatch:
    status: str
    person_id: str | None
    top1_score: float
    top2_score: float
    margin: float
    reason: str


@dataclass(frozen=True)
class GalleryRanking:
    top1_person_id: str
    top1_score: float
    top2_score: float
    margin: float
    tied: bool


class Gallery:
    """Vectorized per-person max scoring over individual enrollment vectors."""

    def __init__(
        self,
        embeddings: np.ndarray,
        person_ids: np.ndarray,
        *,
        embedding_model_sha256: str,
        source_sha256: np.ndarray | None = None,
    ) -> None:
        matrix = l2_normalize(embeddings)
        if matrix.ndim != 2:
            raise GalleryError("Gallery embeddings must be a two-dimensional matrix")
        labels = np.asarray(person_ids, dtype=np.str_).reshape(-1)
        if len(labels) != len(matrix) or any(not value.strip() for value in labels):
            raise GalleryError("Gallery person IDs must be non-empty and align with vectors")
        if len(set(labels.tolist())) < 2:
            raise GalleryError("At least two enrolled identities are required")
        if len(embedding_model_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in embedding_model_sha256.lower()
        ):
            raise GalleryError("Embedding model SHA-256 must be 64 hexadecimal characters")
        hashes = (
            np.asarray(source_sha256, dtype=np.str_).reshape(-1)
            if source_sha256 is not None
            else np.full(len(matrix), "", dtype=np.str_)
        )
        if len(hashes) != len(matrix):
            raise GalleryError("Source hashes must align with gallery vectors")
        nonempty_hashes = [value for value in hashes.tolist() if value]
        if len(nonempty_hashes) != len(set(nonempty_hashes)):
            raise GalleryError("Duplicate source hashes are not allowed in a gallery")
        self._embeddings = matrix
        self._person_ids = labels
        self._source_sha256 = hashes
        self.embedding_model_sha256 = embedding_model_sha256.lower()
        self._refresh_index()

    def _refresh_index(self) -> None:
        self._identities = np.asarray(sorted(set(self._person_ids.tolist())), dtype=np.str_)
        self._row_indices = {
            person_id: np.flatnonzero(self._person_ids == person_id)
            for person_id in self._identities.tolist()
        }

    @property
    def size(self) -> int:
        return len(self._embeddings)

    @property
    def identity_count(self) -> int:
        return len(self._identities)

    @property
    def identity_ids(self) -> tuple[str, ...]:
        """Return enrolled identity references without exposing gallery vectors."""
        return tuple(str(value) for value in self._identities.tolist())

    def vector_count(self, person_id: str) -> int:
        """Return the number of enrollment vectors for one identity."""
        rows = self._row_indices.get(person_id)
        return 0 if rows is None else int(len(rows))

    def source_hash_count(self, person_id: str) -> int:
        """Return non-empty source references without exposing their values."""
        rows = self._row_indices.get(person_id)
        if rows is None:
            return 0
        return sum(bool(self._source_sha256[index]) for index in rows)

    def search(
        self,
        embedding: np.ndarray,
        *,
        threshold: float,
        min_margin: float,
    ) -> GalleryMatch:
        if not np.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
            raise GalleryError("Recognition threshold must be finite and within [-1, 1]")
        if not np.isfinite(min_margin) or min_margin < 0.0 or min_margin > 2.0:
            raise GalleryError("Recognition margin must be finite and within [0, 2]")
        ranking = self.rank(embedding)
        top1 = ranking.top1_score
        top2 = ranking.top2_score
        margin = ranking.margin
        tied = ranking.tied
        accepted = top1 >= threshold and margin >= min_margin and not tied
        if tied:
            reason = "exact_or_numerical_tie"
        elif top1 < threshold:
            reason = "below_threshold"
        elif margin < min_margin:
            reason = "ambiguous_margin"
        else:
            reason = "accepted"
        return GalleryMatch(
            status="verified" if accepted else "unknown",
            person_id=ranking.top1_person_id if accepted else None,
            top1_score=top1,
            top2_score=top2,
            margin=margin,
            reason=reason,
        )

    def rank(self, embedding: np.ndarray) -> GalleryRanking:
        """Return top-two per-person scores without making an acceptance decision."""
        query = l2_normalize(embedding)
        if query.ndim != 1 or query.shape[0] != self._embeddings.shape[1]:
            raise GalleryError("Query embedding dimension does not match the gallery")
        vector_scores = self._embeddings @ query
        person_scores = np.asarray(
            [float(np.max(vector_scores[rows])) for rows in self._row_indices.values()],
            dtype=np.float64,
        )
        order = np.argsort(-person_scores, kind="stable")
        top1_index, top2_index = int(order[0]), int(order[1])
        top1 = float(person_scores[top1_index])
        top2 = float(person_scores[top2_index])
        margin = top1 - top2
        tied = bool(np.isclose(top1, top2, rtol=0.0, atol=1e-7))
        return GalleryRanking(
            top1_person_id=str(self._identities[top1_index]),
            top1_score=top1,
            top2_score=top2,
            margin=margin,
            tied=tied,
        )

    def enroll(
        self,
        person_id: str,
        embeddings: np.ndarray,
        *,
        source_sha256: np.ndarray | None = None,
    ) -> None:
        person_id = person_id.strip()
        if not person_id:
            raise GalleryError("Person ID must not be empty")
        vectors = l2_normalize(embeddings)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[1] != self._embeddings.shape[1]:
            raise GalleryError("Enrollment embedding dimension does not match the gallery")
        hashes = (
            np.asarray(source_sha256, dtype=np.str_).reshape(-1)
            if source_sha256 is not None
            else np.full(len(vectors), "", dtype=np.str_)
        )
        if len(hashes) != len(vectors):
            raise GalleryError("Enrollment source hashes must align with vectors")
        existing = {value for value in self._source_sha256.tolist() if value}
        incoming = [value for value in hashes.tolist() if value]
        if len(incoming) != len(set(incoming)) or existing.intersection(incoming):
            raise GalleryError("Duplicate source hash in enrollment")
        self._embeddings = np.concatenate((self._embeddings, vectors), axis=0)
        self._person_ids = np.concatenate(
            (self._person_ids, np.asarray([person_id] * len(vectors), dtype=np.str_))
        )
        self._source_sha256 = np.concatenate((self._source_sha256, hashes))
        self._refresh_index()

    def remove(self, person_id: str) -> int:
        keep = self._person_ids != person_id
        removed = int(np.sum(~keep))
        if removed == 0:
            return 0
        remaining_ids = set(self._person_ids[keep].tolist())
        if len(remaining_ids) < 2:
            raise GalleryError("Removal would leave fewer than two gallery identities")
        self._embeddings = self._embeddings[keep]
        self._person_ids = self._person_ids[keep]
        self._source_sha256 = self._source_sha256[keep]
        self._refresh_index()
        return removed

    def save(self, path: str | Path, *, manifest_sha256: str) -> str:
        if len(manifest_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in manifest_sha256.lower()
        ):
            raise GalleryError("Manifest SHA-256 must be 64 hexadecimal characters")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w+b", dir=target.parent, prefix=f".{target.name}.", delete=False
        ) as handle:
            np.savez_compressed(
                handle,
                embeddings=self._embeddings,
                person_ids=self._person_ids,
                source_sha256=self._source_sha256,
                embedding_model_sha256=np.asarray(
                    [self.embedding_model_sha256], dtype=np.str_
                ),
                manifest_sha256=np.asarray([manifest_sha256], dtype=np.str_),
                schema_version=np.asarray([1], dtype=np.int64),
            )
            temporary = Path(handle.name)
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
        return sha256_file(target)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_sha256: str,
        expected_embedding_model_sha256: str,
    ) -> "Gallery":
        gallery_path = Path(path)
        if not gallery_path.is_file():
            raise GalleryError(f"Gallery file is absent: {gallery_path}")
        actual_hash = sha256_file(gallery_path)
        if actual_hash != expected_sha256:
            raise GalleryError(
                f"Gallery hash mismatch: expected {expected_sha256}, got {actual_hash}"
            )
        try:
            with np.load(gallery_path, allow_pickle=False) as archive:
                required = {
                    "embeddings",
                    "person_ids",
                    "source_sha256",
                    "embedding_model_sha256",
                }
                missing = required.difference(archive.files)
                if missing:
                    raise GalleryError(f"Gallery fields are missing: {sorted(missing)}")
                stored_model_hash = str(archive["embedding_model_sha256"].reshape(-1)[0])
                gallery = cls(
                    archive["embeddings"],
                    archive["person_ids"],
                    embedding_model_sha256=stored_model_hash,
                    source_sha256=archive["source_sha256"],
                )
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise GalleryError(f"Unable to load gallery: {exc}") from exc
        if gallery.embedding_model_sha256 != expected_embedding_model_sha256.lower():
            raise GalleryError("Gallery embedding model does not match the active recognizer")
        return gallery
