"""Aggregate runtime metrics and an optional isolated Tkinter dashboard."""

from __future__ import annotations

import math
import multiprocessing
import queue
import statistics
import sys
import time
from collections import deque
from dataclasses import dataclass

BACKGROUND = "#0f1115"
TILE = "#171a21"
MUTED = "#9aa4b2"
TEXT = "#f5f7fa"
STATUS_COLORS = {
    "verified": "#1f6f43",
    "spoof": "#8a1c1c",
    "unknown": "#7a4b00",
    "retry": "#263142",
}


class RollingMetrics:
    """Fixed-window processing and frame-age statistics; no biometric values."""

    def __init__(self, window: int = 60) -> None:
        if window < 2:
            raise ValueError("Metrics window must be at least two samples")
        self._ages: deque[float] = deque(maxlen=window)
        self._latencies: deque[float] = deque(maxlen=window)
        self._times: deque[float] = deque(maxlen=window)
        self.processed = 0
        self.started = time.monotonic()

    def add(self, latency_ms: float, age_ms: float) -> None:
        self.processed += 1
        self._latencies.append(float(latency_ms))
        self._ages.append(float(age_ms))
        self._times.append(time.monotonic())

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.started

    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        return 0.0 if span <= 0 else (len(self._times) - 1) / span

    def median_age(self) -> float:
        return float(statistics.median(self._ages)) if self._ages else 0.0

    def p95_age(self) -> float:
        return self._percentile(self._ages, 95.0)

    def median_latency(self) -> float:
        return float(statistics.median(self._latencies)) if self._latencies else 0.0

    @staticmethod
    def _percentile(values: deque[float], percent: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = max(0, math.ceil(percent / 100.0 * len(ordered)) - 1)
        return float(ordered[index])


@dataclass(frozen=True)
class MetricsSnapshot:
    message: str
    state: str
    status: str
    fps: float
    median_age_ms: float
    p95_age_ms: float
    median_latency_ms: float
    processed: int
    captured: int
    dropped: int
    pad_observations: int
    pad_required: int
    elapsed_s: float


class MetricsDashboard:
    """Controller for a separate dashboard process; window close requests shutdown."""

    def __init__(self, *, pad_required: int) -> None:
        method = "spawn" if sys.platform in ("darwin", "win32") else "fork"
        context = multiprocessing.get_context(method)
        self.pad_required = max(1, int(pad_required))
        self._updates = context.Queue(maxsize=2)
        self._errors = context.Queue(maxsize=1)
        self._ready = context.Event()
        self._available = context.Value("b", 0)
        self._closed = context.Event()
        self._process = context.Process(
            target=_dashboard_main,
            args=(
                self._updates,
                self.pad_required,
                self._ready,
                self._available,
                self._errors,
                self._closed,
            ),
            daemon=True,
        )
        self._process.start()
        self._ready.wait(timeout=5.0)
        self.error: str | None = None
        if not self._available.value:
            try:
                self.error = self._errors.get(timeout=0.2)
            except queue.Empty:
                self.error = "dashboard process did not become ready"

    @property
    def available(self) -> bool:
        return self.error is None and self._process.is_alive()

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    def update(self, snapshot: MetricsSnapshot) -> None:
        try:
            self._updates.put_nowait(snapshot)
        except queue.Full:
            try:
                self._updates.get_nowait()
            except queue.Empty:
                pass
            try:
                self._updates.put_nowait(snapshot)
            except queue.Full:
                pass

    def close(self) -> None:
        if self._process.is_alive():
            try:
                self._updates.put_nowait(None)
            except queue.Full:
                pass
            self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)
        self._closed.set()


class _DashboardWindow:
    def __init__(self, root, tk, pad_required: int) -> None:
        self.root = root
        self.tk = tk
        self.pad_required = pad_required
        self._build()

    def _build(self) -> None:
        tk = self.tk
        self.root.title("FaceGuard AI · runtime metrics")
        self.root.configure(bg=BACKGROUND)
        self.root.geometry("480x600")
        self.root.minsize(420, 560)

        header = tk.Frame(self.root, bg=BACKGROUND)
        header.pack(fill="x", padx=22, pady=(20, 8))
        tk.Label(
            header,
            text="FACEGUARD AI",
            font=("DejaVu Sans", 18, "bold"),
            fg=TEXT,
            bg=BACKGROUND,
        ).pack(anchor="w")
        tk.Label(
            header,
            text="Local prototype · camera 0 · CPUExecutionProvider",
            font=("DejaVu Sans", 9),
            fg=MUTED,
            bg=BACKGROUND,
        ).pack(anchor="w", pady=(2, 0))

        self._status = tk.Label(
            self.root,
            text="Starting…",
            font=("DejaVu Sans", 20, "bold"),
            fg=TEXT,
            bg=STATUS_COLORS["retry"],
            anchor="w",
            padx=16,
            pady=12,
        )
        self._status.pack(fill="x", padx=22, pady=(6, 4))
        self._detail = tk.Label(
            self.root,
            text="waiting for frames",
            font=("DejaVu Sans", 9),
            fg=MUTED,
            bg=BACKGROUND,
            anchor="w",
        )
        self._detail.pack(fill="x", padx=22)

        grid = tk.Frame(self.root, bg=BACKGROUND)
        grid.pack(fill="both", expand=True, padx=17, pady=8)
        labels = (
            ("fps", "PROCESSED FPS"),
            ("age_median", "MEDIAN FRAME AGE"),
            ("age_p95", "P95 FRAME AGE"),
            ("latency", "PIPELINE MEDIAN"),
            ("processed", "PROCESSED / CAPTURED"),
            ("dropped", "DROPPED FRAMES"),
            ("pad", "PAD OBSERVATIONS"),
            ("elapsed", "ELAPSED"),
        )
        self._values: dict[str, object] = {}
        for index, (key, label) in enumerate(labels):
            tile = tk.Frame(grid, bg=TILE, padx=14, pady=10)
            tile.grid(row=index // 2, column=index % 2, sticky="nsew", padx=5, pady=5)
            tk.Label(
                tile,
                text=label,
                font=("DejaVu Sans", 8, "bold"),
                fg=MUTED,
                bg=TILE,
                anchor="w",
            ).pack(fill="x")
            value = tk.Label(
                tile,
                text="—",
                font=("DejaVu Sans", 18, "bold"),
                fg=TEXT,
                bg=TILE,
                anchor="w",
            )
            value.pack(fill="x", pady=(2, 0))
            self._values[key] = value
        for column in (0, 1):
            grid.columnconfigure(column, weight=1)

        self._pad_canvas = tk.Canvas(
            self.root, height=8, bg="#20242c", highlightthickness=0
        )
        self._pad_canvas.pack(fill="x", padx=22, pady=(4, 6))
        self._pad_fill = self._pad_canvas.create_rectangle(
            0, 0, 1, 8, fill="#4fd1c5", outline=""
        )
        tk.Label(
            self.root,
            text="No frames, crops, embeddings or identity histories are retained.",
            font=("DejaVu Sans", 8),
            fg=MUTED,
            bg=BACKGROUND,
            anchor="w",
        ).pack(fill="x", padx=22, pady=(0, 18))

    def render(self, snapshot: MetricsSnapshot) -> None:
        color = STATUS_COLORS.get(snapshot.status, STATUS_COLORS["retry"])
        self._status.configure(text=snapshot.message, bg=color)
        self._detail.configure(
            text=f"{snapshot.state.replace('_', ' ')} · {snapshot.status.upper()}"
        )
        values = {
            "fps": f"{snapshot.fps:.1f} fps",
            "age_median": f"{snapshot.median_age_ms:.0f} ms",
            "age_p95": f"{snapshot.p95_age_ms:.0f} ms",
            "latency": f"{snapshot.median_latency_ms:.0f} ms",
            "processed": f"{snapshot.processed} / {snapshot.captured}",
            "dropped": str(snapshot.dropped),
            "pad": f"{snapshot.pad_observations} / {snapshot.pad_required}",
            "elapsed": f"{snapshot.elapsed_s:.0f} s",
        }
        for key, text in values.items():
            self._values[key].configure(text=text)
        width = max(1, self._pad_canvas.winfo_width())
        ratio = min(1.0, snapshot.pad_observations / max(1, snapshot.pad_required))
        self._pad_canvas.coords(self._pad_fill, 0, 0, max(1, int(width * ratio)), 8)


def _dashboard_main(
    updates,
    pad_required: int,
    ready,
    available,
    errors,
    closed,
) -> None:
    try:
        import signal

        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (ImportError, OSError, ValueError):
        pass
    try:
        import tkinter as tk

        root = tk.Tk()
        window = _DashboardWindow(root, tk, pad_required)
        available.value = 1
        ready.set()

        def poll() -> None:
            if closed.is_set():
                root.destroy()
                return
            while True:
                try:
                    snapshot = updates.get_nowait()
                except queue.Empty:
                    break
                if snapshot is None:
                    root.destroy()
                    return
                window.render(snapshot)
            root.after(50, poll)

        root.protocol("WM_DELETE_WINDOW", closed.set)
        root.after(0, poll)
        root.mainloop()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        try:
            errors.put_nowait(f"{type(exc).__name__}: {exc}")
        except Exception:
            pass
        ready.set()
    finally:
        closed.set()
