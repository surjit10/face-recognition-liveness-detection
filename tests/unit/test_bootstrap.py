from types import SimpleNamespace

from faceguard.bootstrap import build_readiness_report


def test_readiness_report_contains_no_biometric_payload() -> None:
    config = SimpleNamespace(
        release_id="test-release",
        providers=("CPUExecutionProvider",),
        insightface_pack="buffalo_m",
    )
    verifier = SimpleNamespace(
        analyzer=SimpleNamespace(
            runtime_providers={
                "detection": ("CPUExecutionProvider",),
                "recognition": ("CPUExecutionProvider",),
            },
            model_hashes={"det.onnx": "a" * 64, "rec.onnx": "b" * 64},
        ),
        pad_model=SimpleNamespace(model_sha256="c" * 64),
        gallery=SimpleNamespace(size=12, identity_count=3),
    )

    report = build_readiness_report(verifier, config)

    assert report["event"] == "runtime_ready"
    assert report["camera_opened"] is False
    assert report["gallery_identities"] == 3
    rendered = str(report).lower()
    assert "embedding" not in rendered
    assert "frame" not in rendered
    assert "person_id" not in rendered
