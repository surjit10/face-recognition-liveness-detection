import pytest

from faceguard.telemetry import RollingMetrics


def test_rolling_metrics_report_age_latency_and_counts() -> None:
    metrics = RollingMetrics(window=10)
    for latency, age in zip((10, 20, 30, 40, 50), (100, 200, 300, 400, 500)):
        metrics.add(latency, age)

    assert metrics.processed == 5
    assert metrics.median_age() == 300
    assert metrics.p95_age() == 500
    assert metrics.median_latency() == 30


def test_rolling_metrics_discard_old_samples() -> None:
    metrics = RollingMetrics(window=2)
    for age in (100, 200, 300):
        metrics.add(age, age)

    assert metrics.median_age() == 250
    assert metrics.p95_age() == 300


def test_metrics_window_must_hold_two_samples() -> None:
    with pytest.raises(ValueError):
        RollingMetrics(window=1)
