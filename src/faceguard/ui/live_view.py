"""Single-window live preview and operator controls."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Mapping

import cv2
from PIL import Image, ImageTk

from ..presentation import (
    face_annotations,
    operator_guidance,
    operator_message,
    status_color_key,
)
from ..session import UiSnapshot
from .theme import COLORS, FONT_FAMILY, button


class LiveView(tk.Frame):
    """Responsive presentation-only view; it owns no camera or model state."""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        on_check: Callable[[], None],
        on_start: Callable[[], None],
        on_stop: Callable[[], None],
        identity_labels: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(parent, bg=COLORS["background"])
        self._photo: ImageTk.PhotoImage | None = None
        self._on_check = on_check
        self._on_start = on_start
        self._on_stop = on_stop
        self._identity_labels = dict(identity_labels or {})
        self._build()

    def _build(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = tk.Frame(self, bg=COLORS["background"])
        header.grid(row=0, column=0, sticky="ew", padx=30, pady=(24, 18))
        header.grid_columnconfigure(1, weight=1)

        mark = tk.Canvas(
            header,
            width=44,
            height=44,
            bg=COLORS["background"],
            highlightthickness=0,
        )
        mark.grid(row=0, column=0, rowspan=2, padx=(0, 13))
        mark.create_oval(3, 3, 41, 41, fill=COLORS["accent"], outline="")
        mark.create_oval(13, 11, 31, 29, outline=COLORS["text"], width=2)
        mark.create_arc(10, 20, 34, 39, start=12, extent=156, style="arc", outline=COLORS["text"], width=2)

        tk.Label(
            header,
            text="FACEGUARD AI",
            font=(FONT_FAMILY, 18, "bold"),
            fg=COLORS["text"],
            bg=COLORS["background"],
            anchor="w",
        ).grid(row=0, column=1, sticky="w")
        tk.Label(
            header,
            text="Local vision verification workspace",
            font=(FONT_FAMILY, 9),
            fg=COLORS["muted"],
            bg=COLORS["background"],
            anchor="w",
        ).grid(row=1, column=1, sticky="w", pady=(1, 0))

        badge = tk.Label(
            header,
            text="ACADEMIC PROTOTYPE",
            font=(FONT_FAMILY, 8, "bold"),
            fg="#BFD2FF",
            bg="#172846",
            padx=12,
            pady=7,
        )
        badge.grid(row=0, column=2, rowspan=2, sticky="e")

        body = tk.Frame(self, bg=COLORS["background"])
        body.grid(row=1, column=0, sticky="nsew", padx=30)
        body.grid_columnconfigure(0, weight=3, uniform="body")
        body.grid_columnconfigure(1, weight=2, uniform="body")
        body.grid_rowconfigure(0, weight=1)

        preview_card = tk.Frame(
            body,
            bg=COLORS["surface"],
            highlightbackground=COLORS["border"],
            highlightthickness=1,
        )
        preview_card.grid(row=0, column=0, sticky="nsew", padx=(0, 9))
        preview_card.grid_columnconfigure(0, weight=1)
        preview_card.grid_rowconfigure(1, weight=1)

        preview_header = tk.Frame(preview_card, bg=COLORS["surface"])
        preview_header.grid(row=0, column=0, sticky="ew", padx=18, pady=(15, 12))
        preview_header.grid_columnconfigure(1, weight=1)
        tk.Label(
            preview_header,
            text="LIVE CAMERA",
            font=(FONT_FAMILY, 9, "bold"),
            fg=COLORS["muted"],
            bg=COLORS["surface"],
        ).grid(row=0, column=0, sticky="w")
        self.source_label = tk.Label(
            preview_header,
            text="CAMERA 0 · OFF",
            font=(FONT_FAMILY, 8, "bold"),
            fg=COLORS["subtle"],
            bg=COLORS["surface"],
        )
        self.source_label.grid(row=0, column=1, sticky="e")

        self.video = tk.Label(
            preview_card,
            text="Camera preview is off\n\nRun the system check, then start camera 0.",
            font=(FONT_FAMILY, 12),
            fg=COLORS["muted"],
            bg=COLORS["video"],
            justify="center",
        )
        self.video.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))

        side = tk.Frame(body, bg=COLORS["background"])
        side.grid(row=0, column=1, sticky="nsew", padx=(9, 0))
        side.grid_columnconfigure(0, weight=1)
        side.grid_rowconfigure(2, weight=1)

        status_card = tk.Frame(
            side,
            bg=COLORS["surface"],
            highlightbackground=COLORS["border"],
            highlightthickness=1,
        )
        status_card.grid(row=0, column=0, sticky="ew")
        status_card.grid_columnconfigure(0, weight=1)

        tk.Label(
            status_card,
            text="VERIFICATION",
            font=(FONT_FAMILY, 9, "bold"),
            fg=COLORS["muted"],
            bg=COLORS["surface"],
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=18, pady=(16, 10))
        self.status_banner = tk.Label(
            status_card,
            text="CHECKING SYSTEM",
            font=(FONT_FAMILY, 18, "bold"),
            fg=COLORS["text"],
            bg=COLORS["neutral_soft"],
            anchor="w",
            padx=15,
            pady=13,
        )
        self.status_banner.grid(row=1, column=0, sticky="ew", padx=18)
        self.guidance = tk.Label(
            status_card,
            text="Validating local models, gallery and runtime configuration.",
            font=(FONT_FAMILY, 9),
            fg=COLORS["muted"],
            bg=COLORS["surface"],
            justify="left",
            anchor="nw",
            wraplength=340,
        )
        self.guidance.grid(row=2, column=0, sticky="ew", padx=18, pady=(10, 14))
        self.pad_progress = ttk.Progressbar(
            status_card,
            orient="horizontal",
            mode="determinate",
            maximum=8,
            style="FaceGuard.Horizontal.TProgressbar",
        )
        self.pad_progress.grid(row=3, column=0, sticky="ew", padx=18)
        self.pad_label = tk.Label(
            status_card,
            text="PAD evidence 0 / 8",
            font=(FONT_FAMILY, 8),
            fg=COLORS["subtle"],
            bg=COLORS["surface"],
            anchor="w",
        )
        self.pad_label.grid(row=4, column=0, sticky="ew", padx=18, pady=(6, 16))

        metrics_card = tk.Frame(
            side,
            bg=COLORS["surface"],
            highlightbackground=COLORS["border"],
            highlightthickness=1,
        )
        metrics_card.grid(row=1, column=0, sticky="ew", pady=(16, 0))
        tk.Label(
            metrics_card,
            text="SESSION DIAGNOSTICS",
            font=(FONT_FAMILY, 9, "bold"),
            fg=COLORS["muted"],
            bg=COLORS["surface"],
            anchor="w",
        ).grid(row=0, column=0, columnspan=2, sticky="ew", padx=18, pady=(15, 10))
        self._metrics: dict[str, tk.Label] = {}
        metric_rows = (
            ("fps", "PROCESSED FPS"),
            ("latency", "MEDIAN LATENCY"),
            ("age", "MEDIAN FRAME AGE"),
            ("frames", "PROCESSED / CAPTURED"),
            ("dropped", "DROPPED FRAMES"),
            ("elapsed", "SESSION TIME"),
        )
        for row, (key, title) in enumerate(metric_rows, start=1):
            tk.Label(
                metrics_card,
                text=title,
                font=(FONT_FAMILY, 8, "bold"),
                fg=COLORS["subtle"],
                bg=COLORS["surface"],
                anchor="w",
            ).grid(row=row, column=0, sticky="w", padx=(18, 8), pady=5)
            value = tk.Label(
                metrics_card,
                text="--",
                font=(FONT_FAMILY, 9, "bold"),
                fg=COLORS["text"],
                bg=COLORS["surface"],
                anchor="e",
            )
            value.grid(row=row, column=1, sticky="e", padx=(8, 18), pady=5)
            self._metrics[key] = value
        metrics_card.grid_columnconfigure(0, weight=1)
        tk.Frame(metrics_card, bg=COLORS["surface"], height=10).grid(
            row=len(metric_rows) + 1, column=0, columnspan=2
        )

        self.readiness = tk.Label(
            side,
            text="Runtime validation in progress",
            font=(FONT_FAMILY, 8),
            fg=COLORS["muted"],
            bg=COLORS["background"],
            justify="left",
            anchor="sw",
            wraplength=340,
        )
        self.readiness.grid(row=2, column=0, sticky="sew", pady=(14, 0))

        controls = tk.Frame(self, bg=COLORS["background"])
        controls.grid(row=2, column=0, sticky="ew", padx=30, pady=(18, 13))
        controls.grid_columnconfigure(3, weight=1)
        self.check_button = button(
            controls, text="Check system", command=self._on_check
        )
        self.check_button.grid(row=0, column=0, padx=(0, 9))
        self.start_button = button(
            controls, text="Start camera", command=self._on_start, primary=True
        )
        self.start_button.grid(row=0, column=1, padx=9)
        self.stop_button = button(
            controls, text="Stop", command=self._on_stop, destructive=True
        )
        self.stop_button.grid(row=0, column=2, padx=9)
        self.session_label = tk.Label(
            controls,
            text="LOCAL ONLY  ·  NO RECORDING",
            font=(FONT_FAMILY, 8, "bold"),
            fg=COLORS["subtle"],
            bg=COLORS["background"],
            anchor="e",
        )
        self.session_label.grid(row=0, column=3, sticky="e")

        footer = tk.Label(
            self,
            text="Frames, face crops and embeddings are processed in memory and are not saved.",
            font=(FONT_FAMILY, 8),
            fg=COLORS["muted"],
            bg=COLORS["surface"],
            padx=30,
            pady=10,
            anchor="w",
        )
        footer.grid(row=3, column=0, sticky="ew")
        self.set_controls(check=False, start=False, stop=False)

    def set_controls(self, *, check: bool, start: bool, stop: bool) -> None:
        self.check_button.configure(state="normal" if check else "disabled")
        self.start_button.configure(state="normal" if start else "disabled")
        self.stop_button.configure(state="normal" if stop else "disabled")

    def set_status(self, message: str, detail: str, tone: str = "neutral") -> None:
        soft = COLORS.get(f"{tone}_soft", COLORS["neutral_soft"])
        self.status_banner.configure(text=message, bg=soft)
        self.guidance.configure(text=detail)

    def set_readiness(self, text: str) -> None:
        self.readiness.configure(text=text)

    def set_source_active(self, active: bool) -> None:
        self.source_label.configure(
            text="CAMERA 0 · LIVE" if active else "CAMERA 0 · OFF",
            fg=COLORS["success"] if active else COLORS["subtle"],
        )

    def clear_preview(self, message: str = "Camera preview is off") -> None:
        self._photo = None
        self.video.configure(image="", text=message)
        self.set_source_active(False)

    def render_snapshot(self, snapshot: UiSnapshot, *, pad_required: int) -> None:
        display_frame = snapshot.frame_bgr.copy()
        height_px, width_px = display_frame.shape[:2]
        bgr_colors = {
            "success": (106, 163, 22),
            "warning": (60, 162, 230),
            "danger": (111, 90, 239),
            "neutral": (186, 144, 113),
        }
        for annotation in face_annotations(
            snapshot.result,
            identity_labels=self._identity_labels,
        ):
            x1, y1, x2, y2 = annotation.bbox_xyxy
            left = max(0, min(width_px - 1, int(round(x1))))
            top = max(0, min(height_px - 1, int(round(y1))))
            right = max(0, min(width_px - 1, int(round(x2))))
            bottom = max(0, min(height_px - 1, int(round(y2))))
            if right <= left or bottom <= top:
                continue
            color = bgr_colors.get(annotation.tone, bgr_colors["neutral"])
            cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
            (text_width, text_height), baseline = cv2.getTextSize(
                annotation.label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                1,
            )
            label_top = max(0, top - text_height - baseline - 9)
            label_right = min(width_px - 1, left + text_width + 12)
            cv2.rectangle(
                display_frame,
                (left, label_top),
                (label_right, top),
                color,
                -1,
            )
            cv2.putText(
                display_frame,
                annotation.label,
                (left + 6, max(text_height + 2, top - baseline - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        width = max(480, self.video.winfo_width() - 8)
        height = max(360, self.video.winfo_height() - 8)
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(image=image)
        self.video.configure(image=self._photo, text="")
        self.set_source_active(True)

        result = snapshot.result
        self.set_status(
            operator_message(result, identity_labels=self._identity_labels),
            operator_guidance(result),
            status_color_key(result),
        )
        observations = min(result.pad_observations, pad_required)
        self.pad_progress.configure(maximum=max(1, pad_required), value=observations)
        self.pad_label.configure(text=f"PAD evidence {observations} / {pad_required}")

        metrics = snapshot.metrics
        values = {
            "fps": f"{metrics.fps:.1f} fps",
            "latency": f"{metrics.median_latency_ms:.0f} ms",
            "age": f"{metrics.median_age_ms:.0f} ms",
            "frames": f"{metrics.processed} / {metrics.captured}",
            "dropped": str(metrics.dropped),
            "elapsed": f"{metrics.elapsed_s:.0f} s",
        }
        for key, value in values.items():
            self._metrics[key].configure(text=value)

    def reset_metrics(self, *, pad_required: int) -> None:
        for label in self._metrics.values():
            label.configure(text="--")
        self.pad_progress.configure(maximum=max(1, pad_required), value=0)
        self.pad_label.configure(text=f"PAD evidence 0 / {pad_required}")
