from types import SimpleNamespace

import numpy as np

from faceguard.models import InsightFaceEmbedder


def test_embedder_preserves_multiple_face_boxes_for_memory_only_display() -> None:
    embedder = InsightFaceEmbedder.__new__(InsightFaceEmbedder)
    embedder.expected_embedding_dim = 512
    embedder._app = SimpleNamespace(
        get=lambda _: [
            SimpleNamespace(bbox=np.array([1, 2, 20, 30]), det_score=0.9),
            SimpleNamespace(bbox=np.array([40, 5, 70, 38]), det_score=0.8),
        ]
    )

    observation = embedder.embed_rgb(np.zeros((64, 80, 3), dtype=np.uint8))

    assert observation.error == "multiple_faces"
    assert observation.face_count == 2
    assert observation.embedding is None
    assert observation.face_boxes_xyxy == (
        (1.0, 2.0, 20.0, 30.0),
        (40.0, 5.0, 70.0, 38.0),
    )
