import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from faceguard.config import sha256_file
from faceguard.extraction import _embed_with_padding_retries, extract_embeddings
from faceguard.models import EmbeddingObservation


class _FakeEmbedder:
    def __init__(self, **_: object) -> None:
        self.model_hashes = {"detector.onnx": "a" * 64, "recognizer.onnx": "b" * 64}
        self.runtime_providers = {"detection": ("CPUExecutionProvider",)}

    def embed_rgb(self, image_rgb: np.ndarray) -> EmbeddingObservation:
        assert image_rgb.shape == (12, 16, 3)
        vector = np.zeros(512, dtype=np.float32)
        vector[0] = 1.0
        return EmbeddingObservation(1, vector, 0.99, (1.0, 1.0, 10.0, 10.0), None)


def test_private_extraction_outputs_normalized_matrix(
    tmp_path: Path, monkeypatch
) -> None:
    dataset = tmp_path / "dataset"
    image_path = dataset / "0001" / "a.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (16, 12), color=(10, 20, 30)).save(image_path, format="JPEG")

    manifest = tmp_path / "inventory.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "candidate_person_id",
                "relative_path",
                "sha256",
                "decode_ok",
                "include",
                "split",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "candidate_person_id": "0001",
                "relative_path": "0001/a.jpg",
                "sha256": sha256_file(image_path),
                "decode_ok": "true",
                "include": "UNRESOLVED",
                "split": "",
            }
        )

    decisions = tmp_path / "decisions.yaml"
    decisions.write_text(
        """decisions:
  - id: FRAD_PERMISSION
    state: VERIFIED
    value:
      local_processing: true
      kaggle_cloud_transfer: false
  - id: INSIGHTFACE_PACK
    state: VERIFIED
    value:
      selected_pack: buffalo_m
      detector_file: detector.onnx
      detector_sha256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
      recognizer_file: recognizer.onnx
      recognizer_sha256: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
""",
        encoding="utf-8",
    )
    monkeypatch.setattr("faceguard.extraction.InsightFaceEmbedder", _FakeEmbedder)
    output = tmp_path / "private"
    report = extract_embeddings(
        dataset_root=dataset,
        manifest_path=manifest,
        decisions_path=decisions,
        model_root=tmp_path / "models",
        output_dir=output,
        processing_context="local",
        mode="audit",
        providers=("CPUExecutionProvider",),
    )
    assert report["embedded_samples"] == 1
    with np.load(output / "frad_embeddings.npz", allow_pickle=False) as archive:
        assert archive["embeddings"].shape == (1, 512)
        assert np.linalg.norm(archive["embeddings"][0]) == 1.0
        assert archive["person_ids"].tolist() == ["0001"]
        assert archive["selected_input_variants"].tolist() == ["original"]


class _PaddingRetryEmbedder(_FakeEmbedder):
    def embed_rgb(self, image_rgb: np.ndarray) -> EmbeddingObservation:
        if image_rgb.shape == (12, 16, 3):
            return EmbeddingObservation(0, None, None, None, "no_face")
        assert image_rgb.shape == (18, 24, 3)
        vector = np.zeros(512, dtype=np.float32)
        vector[1] = 1.0
        return EmbeddingObservation(1, vector, 0.75, (5.0, 4.0, 19.0, 15.0), None)


class _MultiFaceRetryEmbedder(_FakeEmbedder):
    def __init__(self) -> None:
        self.calls = 0

    def embed_rgb(self, image_rgb: np.ndarray) -> EmbeddingObservation:
        self.calls += 1
        if self.calls == 1:
            return EmbeddingObservation(0, None, None, None, "no_face")
        return EmbeddingObservation(2, None, None, None, "multiple_faces")


def test_no_face_can_be_retried_on_a_recorded_neutral_canvas(
    tmp_path: Path, monkeypatch
) -> None:
    dataset = tmp_path / "dataset"
    image_path = dataset / "0001" / "a.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (16, 12), color=(10, 20, 30)).save(image_path, format="JPEG")

    manifest = tmp_path / "inventory.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "candidate_person_id",
                "relative_path",
                "sha256",
                "decode_ok",
                "include",
                "split",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "candidate_person_id": "0001",
                "relative_path": "0001/a.jpg",
                "sha256": sha256_file(image_path),
                "decode_ok": "true",
                "include": "UNRESOLVED",
                "split": "",
            }
        )

    decisions = tmp_path / "decisions.yaml"
    decisions.write_text(
        """decisions:
  - id: FRAD_PERMISSION
    state: VERIFIED
    value:
      local_processing: true
      kaggle_cloud_transfer: false
  - id: INSIGHTFACE_PACK
    state: VERIFIED
    value:
      selected_pack: buffalo_m
      detector_file: detector.onnx
      detector_sha256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
      recognizer_file: recognizer.onnx
      recognizer_sha256: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
""",
        encoding="utf-8",
    )
    monkeypatch.setattr("faceguard.extraction.InsightFaceEmbedder", _PaddingRetryEmbedder)
    output = tmp_path / "private"
    report = extract_embeddings(
        dataset_root=dataset,
        manifest_path=manifest,
        decisions_path=decisions,
        model_root=tmp_path / "models",
        output_dir=output,
        processing_context="local",
        mode="audit",
        providers=("CPUExecutionProvider",),
        retry_padding_ratios=(0.25,),
    )

    assert report["embedded_samples"] == 1
    assert report["rescued_samples"] == 1
    assert report["embedded_by_input_variant"] == {"neutral_pad_0.25_per_side": 1}
    with np.load(output / "frad_embeddings.npz", allow_pickle=False) as archive:
        assert archive["selected_input_variants"].tolist() == [
            "neutral_pad_0.25_per_side"
        ]
    with (output / "frad_embedding_metadata.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        row = next(csv.DictReader(handle))
    assert json.loads(row["padding_ltrb"]) == [4, 3, 4, 3]
    assert json.loads(row["bbox_xyxy"]) == [1.0, 1.0, 15.0, 12.0]
    attempts = json.loads(row["detection_attempts_json"])
    assert [attempt["error"] for attempt in attempts] == ["no_face", None]


def test_padding_retries_stop_at_multiple_faces() -> None:
    embedder = _MultiFaceRetryEmbedder()
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    observation, variant, _, attempts = _embed_with_padding_retries(
        embedder, image, (0.25, 0.5)
    )

    assert observation.error == "multiple_faces"
    assert variant == "neutral_pad_0.25_per_side"
    assert embedder.calls == 2
    assert len(attempts) == 2
