import csv
from pathlib import Path

import numpy as np
from PIL import Image

from faceguard.config import sha256_file
from faceguard.extraction import extract_embeddings
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
