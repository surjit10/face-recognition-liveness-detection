"""Bright professional visual system for the local desktop application."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


COLORS = {
    "background": "#F3F7FC",
    "surface": "#FFFFFF",
    "surface_alt": "#EDF3FA",
    "surface_hover": "#E1EBF8",
    "border": "#D4E0ED",
    "text": "#12233F",
    "muted": "#5F7188",
    "subtle": "#8392A6",
    "accent": "#2563EB",
    "accent_hover": "#1D4ED8",
    "success": "#0F9F6E",
    "success_soft": "#DCFCEB",
    "warning": "#C97808",
    "warning_soft": "#FFF2D8",
    "danger": "#DC3655",
    "danger_soft": "#FFE5EA",
    "neutral": "#4F6F98",
    "neutral_soft": "#E4EDF7",
    "video": "#101827",
}

FONT_FAMILY = "DejaVu Sans"


def configure_root(root: tk.Misc) -> None:
    root.configure(bg=COLORS["background"])
    try:
        detected = float(root.winfo_fpixels("1i")) / 72.0
        root.tk.call("tk", "scaling", max(1.0, min(1.35, detected)))
    except (tk.TclError, ValueError):
        pass
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure(
        "FaceGuard.Horizontal.TProgressbar",
        troughcolor=COLORS["surface_alt"],
        background=COLORS["accent"],
        bordercolor=COLORS["surface_alt"],
        lightcolor=COLORS["accent"],
        darkcolor=COLORS["accent"],
        thickness=8,
    )
    style.configure(
        "Treeview",
        background=COLORS["surface"],
        fieldbackground=COLORS["surface"],
        foreground=COLORS["text"],
        bordercolor=COLORS["border"],
        rowheight=30,
        font=(FONT_FAMILY, 9),
    )
    style.configure(
        "Treeview.Heading",
        background=COLORS["surface_alt"],
        foreground=COLORS["text"],
        relief="flat",
        font=(FONT_FAMILY, 9, "bold"),
    )
    style.map(
        "Treeview",
        background=[("selected", COLORS["accent"])],
        foreground=[("selected", "#FFFFFF")],
    )
    style.configure(
        "Vertical.TScrollbar",
        background=COLORS["surface_alt"],
        troughcolor=COLORS["surface"],
        bordercolor=COLORS["surface"],
        arrowcolor=COLORS["muted"],
    )


def button(
    parent: tk.Misc,
    *,
    text: str,
    command,
    primary: bool = False,
    destructive: bool = False,
) -> tk.Button:
    background = COLORS["accent"] if primary else COLORS["surface_alt"]
    active = COLORS["accent_hover"] if primary else COLORS["surface_hover"]
    if destructive:
        background = COLORS["danger_soft"]
        active = COLORS["danger"]
    foreground = "#FFFFFF" if primary else COLORS["text"]
    result = tk.Button(
        parent,
        text=text,
        command=command,
        font=(FONT_FAMILY, 10, "bold"),
        fg=foreground,
        bg=background,
        activeforeground="#FFFFFF" if primary or destructive else COLORS["text"],
        activebackground=active,
        disabledforeground=COLORS["subtle"],
        relief="flat",
        bd=0,
        padx=18,
        pady=11,
        cursor="hand2",
        highlightthickness=1,
        highlightbackground=background,
        highlightcolor=COLORS["accent"],
        takefocus=True,
    )
    result.bind("<Enter>", lambda _event: result.configure(bg=active))
    result.bind("<Leave>", lambda _event: result.configure(bg=background))
    return result
