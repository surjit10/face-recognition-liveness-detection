"""Unified navigation and non-verification screens for the desktop prototype."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from datetime import date, datetime
from typing import Callable, Mapping, Sequence

import cv2
from PIL import Image, ImageTk

from ..checkins import CheckInEvent
from .theme import COLORS, FONT_FAMILY, button


class NavigationShell(tk.Frame):
    """Persistent local-app navigation around the existing verification view."""

    LABELS = (
        ("verify", "Verify and attendance"),
        ("enroll", "Enroll person"),
        ("users", "Enrolled users"),
        ("history", "Attendance register"),
        ("settings", "Settings"),
    )

    def __init__(
        self,
        parent: tk.Misc,
        *,
        on_navigate: Callable[[str], None],
    ) -> None:
        super().__init__(parent, bg=COLORS["background"])
        self._on_navigate = on_navigate
        self._buttons: dict[str, tk.Button] = {}
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        sidebar = tk.Frame(self, bg=COLORS["surface"], width=206)
        sidebar.grid(row=0, column=0, sticky="ns")
        sidebar.grid_propagate(False)
        brand = tk.Frame(sidebar, bg=COLORS["surface"])
        brand.pack(fill="x", padx=20, pady=(26, 26))
        mark = tk.Canvas(
            brand, width=38, height=38, bg=COLORS["surface"], highlightthickness=0
        )
        mark.pack(side="left", padx=(0, 11))
        mark.create_oval(2, 2, 36, 36, fill=COLORS["accent"], outline="")
        mark.create_oval(11, 9, 27, 25, outline="#FFFFFF", width=2)
        mark.create_arc(8, 18, 30, 34, start=15, extent=150, style="arc", outline="#FFFFFF", width=2)
        brand_text = tk.Frame(brand, bg=COLORS["surface"])
        brand_text.pack(side="left", fill="x", expand=True)
        tk.Label(
            brand_text,
            text="FACEGUARD",
            font=(FONT_FAMILY, 14, "bold"),
            fg=COLORS["text"],
            bg=COLORS["surface"],
            anchor="w",
        ).pack(fill="x")
        tk.Label(
            brand_text,
            text="ATTENDANCE SYSTEM",
            font=(FONT_FAMILY, 7, "bold"),
            fg=COLORS["subtle"],
            bg=COLORS["surface"],
            anchor="w",
        ).pack(fill="x", pady=(2, 0))
        for key, label in self.LABELS:
            item = tk.Button(
                sidebar,
                text=label,
                command=lambda target=key: self._on_navigate(target),
                font=(FONT_FAMILY, 10, "bold"),
                fg=COLORS["muted"],
                bg=COLORS["surface"],
                activeforeground=COLORS["text"],
                activebackground=COLORS["surface_hover"],
                relief="flat",
                bd=0,
                padx=22,
                pady=13,
                anchor="w",
                cursor="hand2",
            )
            item.pack(fill="x", padx=8, pady=2)
            self._buttons[key] = item
        self.content = tk.Frame(self, bg=COLORS["background"])
        self.content.grid(row=0, column=1, sticky="nsew")
        self.content.grid_rowconfigure(0, weight=1)
        self.content.grid_columnconfigure(0, weight=1)
        self._toast_after: str | None = None
        self._toast = tk.Label(
            self,
            text="",
            font=(FONT_FAMILY, 9, "bold"),
            fg=COLORS["text"],
            bg=COLORS["neutral_soft"],
            padx=18,
            pady=12,
            highlightbackground=COLORS["border"],
            highlightthickness=1,
        )

    def select(self, name: str) -> None:
        for key, item in self._buttons.items():
            selected = key == name
            item.configure(
                bg=COLORS["surface_hover"] if selected else COLORS["surface"],
                fg=COLORS["text"] if selected else COLORS["muted"],
            )

    def set_visible_pages(self, allowed: set[str]) -> None:
        """Hide navigation that the configured local role cannot access."""
        for item in self._buttons.values():
            item.pack_forget()
        for key, _label in self.LABELS:
            if key in allowed:
                self._buttons[key].pack(fill="x", padx=8, pady=2)

    def show_toast(self, message: str, *, tone: str = "neutral", duration_ms: int = 3500) -> None:
        if self._toast_after is not None:
            self.after_cancel(self._toast_after)
        self._toast.configure(
            text=message,
            bg=COLORS.get(f"{tone}_soft", COLORS["neutral_soft"]),
        )
        self._toast.place(relx=0.985, rely=0.975, anchor="se")
        self._toast.lift()
        self._toast_after = self.after(duration_ms, self._hide_toast)

    def _hide_toast(self) -> None:
        self._toast.place_forget()
        self._toast_after = None


def _page_header(parent: tk.Misc, title: str, subtitle: str) -> tk.Frame:
    header = tk.Frame(parent, bg=COLORS["background"])
    tk.Label(
        header,
        text=title,
        font=(FONT_FAMILY, 20, "bold"),
        fg=COLORS["text"],
        bg=COLORS["background"],
        anchor="w",
    ).pack(fill="x")
    tk.Label(
        header,
        text=subtitle,
        font=(FONT_FAMILY, 9),
        fg=COLORS["muted"],
        bg=COLORS["background"],
        anchor="w",
    ).pack(fill="x", pady=(4, 0))
    return header


class EnrollmentView(tk.Frame):
    SAMPLE_TARGET = 5

    def __init__(
        self,
        parent: tk.Misc,
        *,
        on_start: Callable[[], None],
        on_stop: Callable[[], None],
        on_capture: Callable[[], None],
        on_confirm: Callable[[], None],
        on_clear: Callable[[], None],
    ) -> None:
        super().__init__(parent, bg=COLORS["background"])
        self._photo: ImageTk.PhotoImage | None = None
        self.identity_var = tk.StringVar()
        self.consent_var = tk.BooleanVar(value=False)
        self.review_var = tk.BooleanVar(value=False)
        self._step_labels: list[tuple[tk.Label, tk.Label]] = []
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)
        _page_header(
            self,
            "Guided enrollment",
            "Capture quality-checked ArcFace vectors from one consenting person. Raw images are not saved.",
        ).grid(row=0, column=0, sticky="ew", padx=30, pady=(26, 18))
        stepper = tk.Frame(self, bg=COLORS["surface"], highlightthickness=1, highlightbackground=COLORS["border"])
        stepper.grid(row=1, column=0, sticky="ew", padx=30, pady=(0, 16))
        for column, title in enumerate(("Consent", "Capture", "Review", "Complete"), start=1):
            stepper.grid_columnconfigure(column - 1, weight=1)
            cell = tk.Frame(stepper, bg=COLORS["surface"])
            cell.grid(row=0, column=column - 1, sticky="ew", padx=10, pady=12)
            number = tk.Label(
                cell,
                text=str(column),
                width=2,
                font=(FONT_FAMILY, 9, "bold"),
                fg=COLORS["muted"],
                bg=COLORS["surface_alt"],
                padx=5,
                pady=4,
            )
            number.pack(side="left")
            label = tk.Label(
                cell,
                text=title.upper(),
                font=(FONT_FAMILY, 8, "bold"),
                fg=COLORS["muted"],
                bg=COLORS["surface"],
                padx=8,
            )
            label.pack(side="left")
            self._step_labels.append((number, label))
        body = tk.Frame(self, bg=COLORS["background"])
        body.grid(row=2, column=0, sticky="nsew", padx=30, pady=(0, 26))
        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=2)
        body.grid_rowconfigure(0, weight=1)

        preview_card = tk.Frame(body, bg=COLORS["surface"], highlightthickness=1, highlightbackground=COLORS["border"])
        preview_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        preview_card.grid_columnconfigure(0, weight=1)
        preview_card.grid_rowconfigure(1, weight=1)
        tk.Label(
            preview_card,
            text="ENROLLMENT CAMERA",
            font=(FONT_FAMILY, 9, "bold"),
            fg=COLORS["muted"],
            bg=COLORS["surface"],
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=18, pady=15)
        self.video = tk.Label(
            preview_card,
            text="Camera preview is off",
            font=(FONT_FAMILY, 11),
            fg=COLORS["muted"],
            bg=COLORS["video"],
        )
        self.video.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))

        form = tk.Frame(body, bg=COLORS["surface"], highlightthickness=1, highlightbackground=COLORS["border"])
        form.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        form.grid_columnconfigure(0, weight=1)
        tk.Label(
            form,
            text="ENROLLMENT DETAILS",
            font=(FONT_FAMILY, 9, "bold"),
            fg=COLORS["muted"],
            bg=COLORS["surface"],
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 8))
        tk.Label(
            form,
            text="Non-sensitive identity label",
            font=(FONT_FAMILY, 9),
            fg=COLORS["text"],
            bg=COLORS["surface"],
            anchor="w",
        ).grid(row=1, column=0, sticky="ew", padx=20, pady=(8, 5))
        self.identity_entry = tk.Entry(
            form,
            textvariable=self.identity_var,
            font=(FONT_FAMILY, 11),
            fg=COLORS["text"],
            bg=COLORS["surface_alt"],
            insertbackground=COLORS["text"],
            relief="flat",
        )
        self.identity_entry.grid(row=2, column=0, sticky="ew", padx=20, ipady=10)
        self.consent_check = tk.Checkbutton(
            form,
            text="Consent for local biometric enrollment is recorded",
            variable=self.consent_var,
            font=(FONT_FAMILY, 9),
            fg=COLORS["text"],
            bg=COLORS["surface"],
            activebackground=COLORS["surface"],
            activeforeground=COLORS["text"],
            selectcolor=COLORS["surface_alt"],
            anchor="w",
            wraplength=330,
        )
        self.consent_check.grid(row=3, column=0, sticky="ew", padx=16, pady=(14, 4))
        self.progress = ttk.Progressbar(
            form,
            maximum=self.SAMPLE_TARGET,
            style="FaceGuard.Horizontal.TProgressbar",
        )
        self.progress.grid(row=4, column=0, sticky="ew", padx=20, pady=(18, 5))
        self.sample_label = tk.Label(
            form,
            text=f"0 / {self.SAMPLE_TARGET} quality samples",
            font=(FONT_FAMILY, 9, "bold"),
            fg=COLORS["text"],
            bg=COLORS["surface"],
            anchor="w",
        )
        self.sample_label.grid(row=5, column=0, sticky="ew", padx=20)
        self.guidance = tk.Label(
            form,
            text="Record consent, start the camera, then capture several clear samples.",
            font=(FONT_FAMILY, 9),
            fg=COLORS["muted"],
            bg=COLORS["surface"],
            justify="left",
            anchor="nw",
            wraplength=330,
        )
        self.guidance.grid(row=6, column=0, sticky="ew", padx=20, pady=(8, 10))
        self.review_check = tk.Checkbutton(
            form,
            text="I reviewed the possible duplicate identity warning",
            variable=self.review_var,
            font=(FONT_FAMILY, 9),
            fg=COLORS["warning"],
            bg=COLORS["surface"],
            activebackground=COLORS["surface"],
            activeforeground=COLORS["warning"],
            selectcolor=COLORS["surface_alt"],
            anchor="w",
            wraplength=330,
        )
        self.review_check.grid(row=7, column=0, sticky="ew", padx=16, pady=4)
        self.review_check.grid_remove()
        self.summary = tk.Label(
            form,
            text="Consent: pending    Samples: 0/5    Raw images: not retained",
            font=(FONT_FAMILY, 8),
            fg=COLORS["muted"],
            bg=COLORS["surface_alt"],
            justify="left",
            anchor="w",
            padx=12,
            pady=10,
            wraplength=330,
        )
        self.summary.grid(row=8, column=0, sticky="ew", padx=20, pady=(10, 0))
        controls = tk.Frame(form, bg=COLORS["surface"])
        controls.grid(row=9, column=0, sticky="ew", padx=20, pady=(12, 20))
        self.start_button = button(controls, text="Start camera", command=on_start, primary=True)
        self.start_button.grid(row=0, column=0, padx=(0, 6), pady=5)
        self.stop_button = button(controls, text="Stop", command=on_stop)
        self.stop_button.grid(row=0, column=1, padx=6, pady=5)
        self.capture_button = button(controls, text="Capture sample", command=on_capture, primary=True)
        self.capture_button.grid(row=1, column=0, padx=(0, 6), pady=5)
        self.confirm_button = button(controls, text="Confirm enrollment", command=on_confirm, primary=True)
        self.confirm_button.grid(row=1, column=1, padx=6, pady=5)
        self.clear_button = button(controls, text="Clear", command=on_clear, destructive=True)
        self.clear_button.grid(row=2, column=0, padx=(0, 6), pady=5)
        self.set_camera_controls(start=False, stop=False, capture=False)
        self.confirm_button.configure(state="disabled")
        self.set_step(1)

    @property
    def identity_label(self) -> str:
        return self.identity_var.get().strip()

    def set_camera_controls(self, *, start: bool, stop: bool, capture: bool) -> None:
        self.start_button.configure(state="normal" if start else "disabled")
        self.stop_button.configure(state="normal" if stop else "disabled")
        self.capture_button.configure(state="normal" if capture else "disabled")

    def set_step(self, active: int) -> None:
        for index, (number, label) in enumerate(self._step_labels, start=1):
            complete = index < active
            selected = index == active
            number.configure(
                text=str(index),
                bg=(
                    COLORS["success"]
                    if complete
                    else COLORS["accent"]
                    if selected
                    else COLORS["surface_alt"]
                ),
                fg=COLORS["text"] if complete or selected else COLORS["muted"],
            )
            label.configure(fg=COLORS["text"] if selected else COLORS["muted"])

    def render_frame(self, frame_bgr) -> None:
        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        image.thumbnail(
            (max(480, self.video.winfo_width() - 8), max(360, self.video.winfo_height() - 8)),
            Image.Resampling.LANCZOS,
        )
        self._photo = ImageTk.PhotoImage(image=image)
        self.video.configure(image=self._photo, text="")

    def clear_preview(self, text: str = "Camera preview is off") -> None:
        self._photo = None
        self.video.configure(image="", text=text)

    def set_samples(
        self,
        count: int,
        *,
        guidance: str,
        duplicate_ids: Sequence[str] = (),
    ) -> None:
        self.progress.configure(value=min(count, self.SAMPLE_TARGET))
        self.sample_label.configure(text=f"{count} / {self.SAMPLE_TARGET} quality samples")
        consent = "recorded" if self.consent_var.get() else "pending"
        duplicate = f"    Review: {', '.join(duplicate_ids)}" if duplicate_ids else ""
        self.summary.configure(
            text=(
                f"Consent: {consent}    Samples: {count}/{self.SAMPLE_TARGET}"
                f"{duplicate}\nRaw images: not retained"
            )
        )
        if duplicate_ids:
            rendered = ", ".join(duplicate_ids)
            self.guidance.configure(
                text=f"Possible duplicate enrollment: {rendered}. Review before confirming.",
                fg=COLORS["warning"],
            )
            self.review_check.grid()
        else:
            self.guidance.configure(text=guidance, fg=COLORS["muted"])
            self.review_check.grid_remove()
            self.review_var.set(False)
        ready = count >= self.SAMPLE_TARGET
        self.confirm_button.configure(state="normal" if ready else "disabled")
        self.set_step(3 if ready else 2 if self.consent_var.get() else 1)

    def set_busy(self, busy: bool, detail: str) -> None:
        self.guidance.configure(text=detail, fg=COLORS["muted"])
        state = "disabled" if busy else "normal"
        self.identity_entry.configure(state=state)
        self.consent_check.configure(state=state)
        self.clear_button.configure(state=state)

    def reset(self, *, preserve_identity: bool = False) -> None:
        if not preserve_identity:
            self.identity_var.set("")
            self.consent_var.set(False)
        self.review_var.set(False)
        self.review_check.grid_remove()
        self.progress.configure(value=0)
        self.sample_label.configure(text=f"0 / {self.SAMPLE_TARGET} quality samples")
        self.guidance.configure(
            text="Record consent, start the camera, then capture several clear samples.",
            fg=COLORS["muted"],
        )
        self.confirm_button.configure(state="disabled")
        self.summary.configure(
            text="Consent: pending    Samples: 0/5\nRaw images: not retained"
        )
        self.set_step(1)
        self.clear_preview()

    def mark_complete(self) -> None:
        self.set_step(4)
        self.summary.configure(text="Enrollment saved. Raw images were not retained")


class UsersView(tk.Frame):
    def __init__(
        self,
        parent: tk.Misc,
        *,
        on_refresh: Callable[[], None],
        on_inspect: Callable[[], None],
        on_delete: Callable[[], None],
    ) -> None:
        super().__init__(parent, bg=COLORS["background"])
        self.search_var = tk.StringVar()
        self._all_rows: list[dict] = []
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)
        _page_header(
            self,
            "Enrolled users",
            "Authorized operator view. Deletion requires explicit confirmation and reloads the active gallery.",
        ).grid(row=0, column=0, sticky="ew", padx=30, pady=(26, 16))
        controls = tk.Frame(self, bg=COLORS["background"])
        controls.grid(row=1, column=0, sticky="ew", padx=30, pady=(0, 12))
        controls.grid_columnconfigure(0, weight=1)
        search = tk.Entry(
            controls,
            textvariable=self.search_var,
            font=(FONT_FAMILY, 10),
            fg=COLORS["text"],
            bg=COLORS["surface_alt"],
            insertbackground=COLORS["text"],
            relief="flat",
        )
        search.grid(row=0, column=0, sticky="ew", padx=(0, 10), ipady=9)
        self.search_var.trace_add("write", lambda *_: self._filter())
        button(controls, text="Refresh", command=on_refresh).grid(row=0, column=1, padx=5)
        button(controls, text="Inspect", command=on_inspect).grid(row=0, column=2, padx=5)
        button(controls, text="Delete", command=on_delete, destructive=True).grid(row=0, column=3, padx=(5, 0))
        body = tk.Frame(self, bg=COLORS["background"])
        body.grid(row=2, column=0, sticky="nsew", padx=30, pady=(0, 26))
        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=2)
        body.grid_rowconfigure(0, weight=1)
        list_frame = tk.Frame(body, bg=COLORS["surface"])
        list_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)
        self.tree = ttk.Treeview(list_frame, columns=("id", "label", "vectors", "date"), show="headings")
        for key, title, width in (
            ("id", "Identity reference", 180),
            ("label", "Display label", 180),
            ("vectors", "Samples", 80),
            ("date", "Enrolled", 190),
        ):
            self.tree.heading(key, text=title)
            self.tree.column(key, width=width, anchor="w")
        self.tree.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.empty_label = tk.Label(
            list_frame,
            text="No enrolled identities match this search.",
            font=(FONT_FAMILY, 10),
            fg=COLORS["muted"],
            bg=COLORS["surface"],
        )
        self.details = tk.Label(
            body,
            text="Select an identity and choose Inspect.",
            font=(FONT_FAMILY, 9),
            fg=COLORS["muted"],
            bg=COLORS["surface"],
            justify="left",
            anchor="nw",
            padx=20,
            pady=20,
            wraplength=330,
        )
        self.details.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

    @property
    def selected_person_id(self) -> str | None:
        selection = self.tree.selection()
        return None if not selection else str(self.tree.item(selection[0], "values")[0])

    def set_rows(self, rows: Sequence[dict], labels: Mapping[str, str]) -> None:
        self._all_rows = [dict(row, display_label=labels.get(row["person_id"], row["person_id"])) for row in rows]
        self._filter()

    def _filter(self) -> None:
        needle = self.search_var.get().strip().casefold()
        self.tree.delete(*self.tree.get_children())
        for row in self._all_rows:
            haystack = f"{row['person_id']} {row['display_label']}".casefold()
            if needle and needle not in haystack:
                continue
            self.tree.insert(
                "",
                "end",
                values=(
                    row["person_id"],
                    row["display_label"],
                    row["vectors"],
                    row.get("enrolled_at") or "Legacy / unknown",
                ),
            )
        if self.tree.get_children():
            self.empty_label.place_forget()
        else:
            self.empty_label.place(relx=0.5, rely=0.5, anchor="center")

    def show_details(self, data: Mapping[str, object]) -> None:
        consent = data.get("consent_recorded")
        consent_text = "Recorded" if consent is True else "Legacy / unavailable"
        self.details.configure(
            text=(
                f"Identity: {data.get('person_id')}\n\n"
                f"Enrollment vectors: {data.get('vectors_count')}\n"
                f"Source references: {data.get('source_hashes_count')}\n"
                f"Enrolled: {data.get('enrolled_at') or 'Legacy / unknown'}\n"
                f"Consent metadata: {consent_text}\n"
                f"Capture method: {data.get('capture_method')}\n"
                f"Raw images retained: No"
            )
        )

    def set_status(self, text: str) -> None:
        self.details.configure(text=text)


class HistoryView(tk.Frame):
    def __init__(
        self,
        parent: tk.Misc,
        *,
        on_refresh: Callable[[], None],
        on_export: Callable[[], None],
        on_delete_selected: Callable[[], None],
        on_clear: Callable[[], None],
        on_lock: Callable[[], None],
    ) -> None:
        super().__init__(parent, bg=COLORS["background"])
        self._events: tuple[CheckInEvent, ...] = ()
        self._labels: Mapping[str, str] = {}
        self.month_var = tk.StringVar(value="All months")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)
        _page_header(
            self,
            "Attendance register",
            "Verified attendance organized by month and day. No face images or embeddings are stored.",
        ).grid(row=0, column=0, sticky="ew", padx=30, pady=(26, 16))
        controls = tk.Frame(self, bg=COLORS["background"])
        controls.grid(row=1, column=0, sticky="ew", padx=30, pady=(0, 12))
        button(controls, text="Refresh", command=on_refresh).pack(side="left", padx=(0, 6))
        button(controls, text="Export CSV", command=on_export, primary=True).pack(side="left", padx=6)
        button(
            controls,
            text="Delete selected",
            command=on_delete_selected,
            destructive=True,
        ).pack(side="left", padx=6)
        button(controls, text="Delete all history", command=on_clear, destructive=True).pack(side="left", padx=6)
        self.lock_admin_button = button(controls, text="Lock admin", command=on_lock)
        self.month_label = tk.Label(
            controls,
            text="MONTH",
            font=(FONT_FAMILY, 8, "bold"),
            fg=COLORS["muted"],
            bg=COLORS["background"],
        )
        self.month_label.pack(side="left", padx=(22, 6))
        self.month_filter = ttk.Combobox(
            controls,
            textvariable=self.month_var,
            state="readonly",
            width=18,
            values=("All months",),
        )
        self.month_filter.pack(side="left")
        self.month_filter.bind("<<ComboboxSelected>>", lambda _event: self._render_events())
        self.status = tk.Label(
            controls,
            text="",
            font=(FONT_FAMILY, 9),
            fg=COLORS["muted"],
            bg=COLORS["background"],
            anchor="e",
        )
        self.status.pack(side="right", fill="x", expand=True)
        table = tk.Frame(self, bg=COLORS["surface"])
        table.grid(row=2, column=0, sticky="nsew", padx=30, pady=(0, 26))
        table.grid_rowconfigure(0, weight=1)
        table.grid_columnconfigure(0, weight=1)
        self.tree = ttk.Treeview(
            table,
            columns=("time", "identity", "status"),
            show="tree headings",
        )
        self.tree.heading("#0", text="Month / day")
        self.tree.column("#0", width=260, anchor="w")
        for key, title, width in (
            ("time", "Time", 150),
            ("identity", "Identity", 240),
            ("status", "Attendance", 140),
        ):
            self.tree.heading(key, text=title)
            self.tree.column(key, width=width, anchor="w")
        self.tree.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.empty_label = tk.Label(
            table,
            text="No attendance yet. A successful verification will appear here automatically.",
            font=(FONT_FAMILY, 10),
            fg=COLORS["muted"],
            bg=COLORS["surface"],
        )

    def set_events(self, events: Sequence[CheckInEvent], labels: Mapping[str, str]) -> None:
        self._events = tuple(events)
        self._labels = labels
        months = sorted(
            {event.local_date[:7] for event in events if event.local_date}, reverse=True
        )
        month_labels = [
            datetime.strptime(month, "%Y-%m").strftime("%B %Y") for month in months
        ]
        current = self.month_var.get()
        values = ("All months", *month_labels)
        self.month_filter.configure(values=values)
        if current not in values:
            self.month_var.set("All months")
        self._render_events()

    @property
    def selected_event_id(self) -> int | None:
        selected = self.tree.selection()
        if not selected or not selected[0].startswith("event:"):
            return None
        try:
            return int(selected[0].split(":", 1)[1])
        except ValueError:
            return None

    def show_admin_lock(self) -> None:
        self.lock_admin_button.pack(
            side="left", padx=(14, 6), before=self.month_label
        )

    def _render_events(self) -> None:
        self.tree.delete(*self.tree.get_children())
        selected_month = self.month_var.get()
        month_nodes: dict[str, str] = {}
        day_nodes: dict[tuple[str, str], str] = {}
        shown = 0
        for event in self._events:
            try:
                attendance_day = date.fromisoformat(event.local_date)
                occurred = datetime.fromisoformat(event.occurred_at).astimezone()
            except ValueError:
                continue
            month_label = attendance_day.strftime("%B %Y")
            if selected_month != "All months" and month_label != selected_month:
                continue
            month_key = attendance_day.strftime("%Y-%m")
            month_node = month_nodes.get(month_key)
            if month_node is None:
                month_node = self.tree.insert(
                    "", "end", text=f"  {month_label}", open=True, values=("", "", "")
                )
                month_nodes[month_key] = month_node
            day_key = (month_key, event.local_date)
            day_node = day_nodes.get(day_key)
            if day_node is None:
                day_node = self.tree.insert(
                    month_node,
                    "end",
                    text=f"    {attendance_day.strftime('%A, %d %B')}",
                    open=True,
                    values=("", "", ""),
                )
                day_nodes[day_key] = day_node
            self.tree.insert(
                day_node,
                "end",
                iid=f"event:{event.event_id}",
                text="      Attendance",
                values=(
                    occurred.strftime("%I:%M %p"),
                    self._labels.get(event.identity_reference, event.identity_reference),
                    "Present" if event.outcome == "verified" else event.outcome.title(),
                ),
            )
            shown += 1
        if shown:
            self.empty_label.place_forget()
        else:
            self.empty_label.place(relx=0.5, rely=0.5, anchor="center")
        self.set_status(f"{shown} attendance record(s)")

    def set_status(self, text: str, *, error: bool = False) -> None:
        self.status.configure(text=text, fg=COLORS["danger"] if error else COLORS["muted"])


class SettingsView(tk.Frame):
    def __init__(
        self,
        parent: tk.Misc,
        *,
        on_reset_demo: Callable[[], None],
        on_lock: Callable[[], None],
    ) -> None:
        super().__init__(parent, bg=COLORS["background"])
        self.grid_columnconfigure(0, weight=1)
        _page_header(
            self,
            "Workspace settings",
            "Local privacy, operator access, and presentation controls.",
        ).grid(row=0, column=0, sticky="ew", padx=30, pady=(26, 18))
        cards = tk.Frame(self, bg=COLORS["background"])
        cards.grid(row=1, column=0, sticky="nsew", padx=30)
        cards.grid_columnconfigure(0, weight=1, uniform="settings")
        cards.grid_columnconfigure(1, weight=1, uniform="settings")
        system = tk.Frame(cards, bg=COLORS["surface"], highlightthickness=1, highlightbackground=COLORS["border"])
        system.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        tk.Label(
            system,
            text="SYSTEM STATUS",
            font=(FONT_FAMILY, 9, "bold"),
            fg=COLORS["muted"],
            bg=COLORS["surface"],
            anchor="w",
        ).pack(fill="x", padx=22, pady=(20, 10))
        self.details = tk.Label(
            system,
            text="System check in progress…",
            font=(FONT_FAMILY, 10),
            fg=COLORS["text"],
            bg=COLORS["surface"],
            justify="left",
            anchor="nw",
            padx=22,
            pady=8,
            wraplength=420,
        )
        self.details.pack(fill="both", expand=True, pady=(0, 18))
        privacy = tk.Frame(cards, bg=COLORS["surface"], highlightthickness=1, highlightbackground=COLORS["border"])
        privacy.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        tk.Label(
            privacy,
            text="PRIVACY & ACCESS",
            font=(FONT_FAMILY, 9, "bold"),
            fg=COLORS["muted"],
            bg=COLORS["surface"],
            anchor="w",
        ).pack(fill="x", padx=22, pady=(20, 10))
        self.operator_status = tk.Label(
            privacy,
            text="Supervised operator mode",
            font=(FONT_FAMILY, 10, "bold"),
            fg=COLORS["success"],
            bg=COLORS["surface"],
            anchor="w",
        )
        self.operator_status.pack(fill="x", padx=22, pady=(4, 8))
        tk.Label(
            privacy,
            text=(
                "Local processing only\n"
                "Camera images are not retained\n"
                "Attendance data stays in private storage\n"
                "Decision thresholds are fixed"
            ),
            font=(FONT_FAMILY, 9),
            fg=COLORS["text"],
            bg=COLORS["surface"],
            justify="left",
            anchor="nw",
        ).pack(fill="x", padx=22, pady=8)
        actions = tk.Frame(privacy, bg=COLORS["surface"])
        actions.pack(fill="x", padx=22, pady=(12, 22))
        self.lock_button = button(actions, text="Lock operator", command=on_lock)
        self.lock_button.pack(side="left", padx=(0, 6))
        button(actions, text="Reset demo session", command=on_reset_demo, destructive=True).pack(
            side="left", padx=6
        )
        tk.Label(
            self,
            text=(
                "Demonstration scope: Not certified for access-control or consequential decisions."
            ),
            font=(FONT_FAMILY, 9),
            fg=COLORS["warning"],
            bg=COLORS["warning_soft"],
            justify="left",
            anchor="w",
            padx=18,
            pady=14,
            wraplength=900,
        ).grid(row=2, column=0, sticky="ew", padx=30, pady=18)

    def update_info(
        self,
        *,
        provider: str,
        gallery_identities: int,
        gallery_vectors: int,
    ) -> None:
        self.details.configure(
            text=(
                "Models and gallery verified\n\n"
                f"Inference provider      {provider}\n"
                "Authorized camera       Camera 0\n"
                f"Enrolled identities     {gallery_identities}\n"
                f"Gallery vectors         {gallery_vectors}\n\n"
                "Camera access starts only when requested."
            )
        )

    def set_operator_state(self, *, configured: bool, unlocked: bool) -> None:
        if not configured:
            self.operator_status.configure(
                text="Supervised operator mode: PIN not configured", fg=COLORS["warning"]
            )
            self.lock_button.configure(state="disabled")
        elif unlocked:
            self.operator_status.configure(text="Operator session unlocked", fg=COLORS["success"])
            self.lock_button.configure(state="normal")
        else:
            self.operator_status.configure(text="Operator session locked", fg=COLORS["warning"])
            self.lock_button.configure(state="disabled")
