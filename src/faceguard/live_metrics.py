"""Opt-in, loopback-only live runtime metrics for the MLOps dashboard.

``LiveMetricsHub`` aggregates per-frame timing and pipeline-state counts from the
inference loop. ``LiveMetricsServer`` exposes them on 127.0.0.1 as JSON (for the
dashboard), Prometheus text (for a scraper) and serves the dashboard page itself, so
the browser polls a same-origin endpoint and no CORS header is ever sent.

Privacy: only timings, counters, pipeline state names and fixed reason codes leave the
process. Person IDs, similarity scores, PAD scores, track IDs, frames, embeddings and
operator messages (which can contain a display name) are never recorded.
"""

from __future__ import annotations

from collections import Counter, deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import resource
import sys
import threading
import time
from typing import Any

SCHEMA_VERSION = 1
BIND_HOSTS = {"127.0.0.1", "localhost"}
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}  # accepted in the Host header
_REASON_CODE = re.compile(r"^[a-z][a-z0-9_]{0,47}$")
_HISTORY_SECONDS = 120.0


def _rss_mb() -> float | None:
    """Current resident memory from /proc (Linux); None where unavailable."""
    try:
        with open("/proc/self/status", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024.0, 1)
    except (OSError, ValueError, IndexError):
        return None
    return None


def _peak_rss_mb() -> float:
    # ru_maxrss is KiB on Linux and bytes on macOS (same rule as benchmark_runtime.py).
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor, 1)


def _percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percent / 100.0 * len(ordered)) - 1)
    return float(ordered[index])


def load_performance_targets(path: Path) -> dict[str, float]:
    """Read the frozen benchmark objectives so live values are judged by the same gate."""
    import yaml

    objectives = (yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}).get(
        "objectives", {}
    )
    mapping = {
        "min_fps": "minimum_processed_fps",
        "max_p95_frame_age_ms": "maximum_p95_capture_to_decision_ms",
        "max_peak_memory_mb": "maximum_peak_memory_mb",
    }
    return {
        key: float(objectives[source])
        for key, source in mapping.items()
        if isinstance(objectives.get(source), (int, float))
    }


class LiveMetricsHub:
    """Thread-safe aggregate of the newest processed frames; holds no biometric values."""

    def __init__(
        self,
        *,
        source: str,
        window: int = 60,
        targets: dict[str, float] | None = None,
        clock=time.monotonic,
    ) -> None:
        if source not in {"camera", "video", "desktop"}:
            raise ValueError("source must be camera, video or desktop")
        if window < 2:
            raise ValueError("window must be at least two frames")
        self.source = source
        self.targets = dict(targets or {})
        self._clock = clock
        self._lock = threading.Lock()
        self._window = window
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._started = self._clock()
            self._latencies: deque[float] = deque(maxlen=self._window)
            self._ages: deque[float] = deque(maxlen=self._window)
            self._times: deque[float] = deque(maxlen=self._window)
            self._history: deque[tuple[float, float, float]] = deque()
            self._states: Counter[str] = Counter()
            self._reasons: Counter[str] = Counter()
            self._events: Counter[str] = Counter()
            self._processed = 0
            self._captured = 0
            self._dropped = 0
            self._queue_depth = 0
            self._state: str | None = None
            self._status: str | None = None
            self._pad_observations = 0
            self._last_frame_at: float | None = None

    def record(
        self,
        result: Any,
        *,
        age_ms: float,
        captured: int,
        dropped: int,
        queue_depth: int = 0,
    ) -> None:
        """Add one processed frame. Only ``state``, ``status``, ``reason``,
        ``latency_ms`` and ``pad_observations`` are read from ``result``."""
        state = getattr(getattr(result, "state", None), "value", None) or "UNKNOWN_STATE"
        reason = str(getattr(result, "reason", "") or "")
        if not _REASON_CODE.match(reason):
            reason = "other"
        latency = float(getattr(result, "latency_ms", 0.0) or 0.0)
        age = max(0.0, float(age_ms))
        now = self._clock()
        with self._lock:
            self._processed += 1
            self._captured = max(self._captured, int(captured))
            self._dropped = max(self._dropped, int(dropped))
            self._queue_depth = max(0, int(queue_depth))
            self._latencies.append(latency)
            self._ages.append(age)
            self._times.append(now)
            elapsed = now - self._started
            self._history.append((round(elapsed, 3), round(latency, 2), round(age, 2)))
            while self._history and elapsed - self._history[0][0] > _HISTORY_SECONDS:
                self._history.popleft()
            self._states[state] += 1
            self._reasons[reason] += 1
            self._state = state
            self._status = str(getattr(result, "status", "") or "") or None
            self._pad_observations = int(getattr(result, "pad_observations", 0) or 0)
            self._last_frame_at = now

    def note_event(self, name: str) -> None:
        """Count an operational event such as ``camera_reconnected``."""
        if not _REASON_CODE.match(name):
            raise ValueError("event names must be lower_snake_case codes")
        with self._lock:
            self._events[name] += 1

    def snapshot(self) -> dict[str, Any]:
        now = self._clock()
        with self._lock:
            times = list(self._times)
            span = times[-1] - times[0] if len(times) >= 2 else 0.0
            fps = (len(times) - 1) / span if span > 0 else 0.0
            latencies = list(self._latencies)
            ages = list(self._ages)
            captured = max(self._captured, self._processed)
            snapshot = {
                "schema_version": SCHEMA_VERSION,
                "source": self.source,
                "uptime_s": round(now - self._started, 2),
                "seconds_since_last_frame": (
                    None if self._last_frame_at is None else round(now - self._last_frame_at, 2)
                ),
                "frames": {
                    "processed": self._processed,
                    "captured": captured,
                    "dropped": self._dropped,
                    "drop_ratio": round(self._dropped / captured, 4) if captured else 0.0,
                    "queue_depth": self._queue_depth,
                },
                "window": {
                    "frames": len(latencies),
                    "fps": round(fps, 2),
                    "latency_ms_p50": round(_percentile(latencies, 50.0), 2),
                    "latency_ms_p95": round(_percentile(latencies, 95.0), 2),
                    "frame_age_ms_p50": round(_percentile(ages, 50.0), 2),
                    "frame_age_ms_p95": round(_percentile(ages, 95.0), 2),
                },
                "memory_mb": {"rss": _rss_mb(), "peak_rss": _peak_rss_mb()},
                "pipeline": {
                    "state": self._state,
                    "status": self._status,
                    "pad_observations": self._pad_observations,
                    "state_counts": dict(sorted(self._states.items())),
                    "reason_counts": dict(sorted(self._reasons.items())),
                },
                "events": dict(sorted(self._events.items())),
                "history": [list(point) for point in self._history],
                "targets": self.targets,
                "privacy": "aggregate timings and counters only; no identities, scores, "
                "frames or embeddings",
            }
        return snapshot


def prometheus_text(snapshot: dict[str, Any]) -> str:
    """Render a snapshot in the Prometheus text exposition format (0.0.4)."""
    lines: list[str] = []

    def metric(name: str, kind: str, help_text: str, samples) -> None:
        lines.append(f"# HELP faceguard_{name} {help_text}")
        lines.append(f"# TYPE faceguard_{name} {kind}")
        for labels, value in samples:
            if value is None:
                continue
            label_text = (
                "{" + ",".join(f'{k}="{v}"' for k, v in sorted(labels.items())) + "}"
                if labels
                else ""
            )
            lines.append(f"faceguard_{name}{label_text} {float(value):g}")

    frames, window, memory = snapshot["frames"], snapshot["window"], snapshot["memory_mb"]
    source = {"source": snapshot["source"]}
    metric("frames_processed_total", "counter", "Frames run through the pipeline.",
           [(source, frames["processed"])])
    metric("frames_captured_total", "counter", "Frames read from the source.",
           [(source, frames["captured"])])
    metric("frames_dropped_total", "counter", "Stale frames dropped by the latest-frame queue.",
           [(source, frames["dropped"])])
    metric("frame_queue_depth", "gauge", "Frames waiting in the bounded queue.",
           [(source, frames["queue_depth"])])
    metric("processed_fps", "gauge", "Processed frames per second over the rolling window.",
           [(source, window["fps"])])
    metric("latency_ms", "gauge", "Pipeline latency over the rolling window.",
           [({**source, "quantile": "0.5"}, window["latency_ms_p50"]),
            ({**source, "quantile": "0.95"}, window["latency_ms_p95"])])
    metric("frame_age_ms", "gauge", "Capture-to-decision frame age over the rolling window.",
           [({**source, "quantile": "0.5"}, window["frame_age_ms_p50"]),
            ({**source, "quantile": "0.95"}, window["frame_age_ms_p95"])])
    metric("memory_rss_mb", "gauge", "Resident memory of the runtime process.",
           [(source, memory["rss"])])
    metric("memory_peak_rss_mb", "gauge", "Peak resident memory of the runtime process.",
           [(source, memory["peak_rss"])])
    metric("pipeline_state_total", "counter", "Processed frames by pipeline state.",
           [({**source, "state": k}, v) for k, v in snapshot["pipeline"]["state_counts"].items()])
    metric("pipeline_reason_total", "counter", "Processed frames by decision reason code.",
           [({**source, "reason": k}, v)
            for k, v in snapshot["pipeline"]["reason_counts"].items()])
    metric("runtime_events_total", "counter", "Operational events such as camera reconnects.",
           [({**source, "event": k}, v) for k, v in snapshot["events"].items()])
    return "\n".join(lines) + "\n"


class _Handler(BaseHTTPRequestHandler):
    server: "_Server"
    server_version = "FaceGuardMetrics/1"
    sys_version = ""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib name
        return  # keep the operator console clean; requests carry no useful audit data

    def _host_allowed(self) -> bool:
        # DNS-rebinding guard: a hostile page can resolve its own name to 127.0.0.1,
        # but its requests still carry that foreign name in the Host header.
        host = (self.headers.get("Host") or "").strip().lower()
        if host.startswith("["):
            name = host[1:].split("]", 1)[0]
        else:
            name = host.rsplit(":", 1)[0] if host.count(":") == 1 else host
        return name in LOOPBACK_HOSTS

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_HEAD(self) -> None:  # noqa: N802 - stdlib naming
        self.do_GET()

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        if not self._host_allowed():
            self._send(HTTPStatus.FORBIDDEN, b"forbidden host\n", "text/plain; charset=utf-8")
            return
        path = self.path.split("?", 1)[0]
        if path == "/metrics.json":
            body = json.dumps(self.server.hub.snapshot(), sort_keys=True).encode("utf-8")
            self._send(HTTPStatus.OK, body, "application/json")
        elif path == "/metrics":
            body = prometheus_text(self.server.hub.snapshot()).encode("utf-8")
            self._send(HTTPStatus.OK, body, "text/plain; version=0.0.4; charset=utf-8")
        elif path == "/healthz":
            self._send(HTTPStatus.OK, b"ok\n", "text/plain; charset=utf-8")
        elif path in {"/", "/index.html"} and self.server.dashboard is not None:
            self._send(HTTPStatus.OK, self.server.dashboard.read_bytes(),
                       "text/html; charset=utf-8")
        else:
            self._send(HTTPStatus.NOT_FOUND, b"not found\n", "text/plain; charset=utf-8")


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, hub: LiveMetricsHub, dashboard: Path | None) -> None:
        self.hub = hub
        self.dashboard = dashboard
        super().__init__(address, _Handler)


class LiveMetricsServer:
    """Background HTTP server bound to a loopback address only."""

    def __init__(
        self,
        hub: LiveMetricsHub,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        dashboard: Path | None = None,
    ) -> None:
        if host not in BIND_HOSTS:
            raise ValueError("live metrics may only bind to 127.0.0.1 (loopback)")
        if not 0 <= int(port) <= 65535:
            raise ValueError("port must be within 0-65535")
        if dashboard is not None and not Path(dashboard).is_file():
            raise ValueError(f"dashboard file not found: {dashboard}")
        self._server = _Server((host, int(port)), hub, Path(dashboard) if dashboard else None)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="faceguard-live-metrics",
            daemon=True,
        )

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/"

    def start(self) -> "LiveMetricsServer":
        self._thread.start()
        return self

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2.0)
