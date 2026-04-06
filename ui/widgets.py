"""
ui/widgets.py — Reusable tkinter widget classes for NeuroStream.

Keeps app_view.py lean by extracting self-contained UI components.
"""

import tkinter as tk


# ══════════════════════════════════════════════════════════════════════════════
# Tooltip
# ══════════════════════════════════════════════════════════════════════════════

class Tooltip:
    """
    Hover tooltip for any tkinter widget.

    Usage:
        tip = Tooltip(some_widget, "Helpful description here.")
        tip.update_text("New text")   # change dynamically
    """

    _DELAY_MS = 350

    def __init__(self, widget: tk.Widget, text: str, *, bg: str = "#24292f",
                 fg: str = "#ffffff"):
        self._widget = widget
        self._text = text
        self._bg = bg
        self._fg = fg
        self._tip: tk.Toplevel | None = None
        self._after_id: str | None = None

        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._cancel, add="+")
        widget.bind("<ButtonPress>", self._cancel, add="+")

    def update_text(self, text: str) -> None:
        self._text = text
        if self._tip is not None:
            for child in self._tip.winfo_children():
                if isinstance(child, tk.Label):
                    child.config(text=text)

    def _schedule(self, _event: tk.Event) -> None:
        self._cancel()
        self._after_id = self._widget.after(self._DELAY_MS, self._show)

    def _cancel(self, _event: tk.Event | None = None) -> None:
        if self._after_id is not None:
            self._widget.after_cancel(self._after_id)
            self._after_id = None
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None

    def _show(self) -> None:
        if not self._text:
            return
        x = self._widget.winfo_rootx() + 12
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 4
        self._tip = tw = tk.Toplevel(self._widget)
        tw.wm_overrideredirect(True)
        tw.attributes("-topmost", True)
        tw.wm_geometry(f"+{x}+{y}")
        lbl = tk.Label(
            tw, text=self._text, justify="left", wraplength=280,
            bg=self._bg, fg=self._fg,
            font=("Helvetica Neue", 9), padx=8, pady=5,
            relief=tk.SOLID, bd=1,
        )
        lbl.pack()


# ══════════════════════════════════════════════════════════════════════════════
# Collapsible Section
# ══════════════════════════════════════════════════════════════════════════════

class CollapsibleSection(tk.Frame):
    """
    A section with a clickable header that toggles child visibility.

    Usage:
        cs = CollapsibleSection(parent, title="Advanced", collapsed=True)
        cs.pack(fill=tk.X)
        tk.Label(cs.content, text="...").pack()
    """

    _ARROW_OPEN = "\u25bc"    # ▼
    _ARROW_CLOSED = "\u25b6"  # ▶

    def __init__(self, parent: tk.Widget, *, title: str = "Advanced",
                 collapsed: bool = True, bg: str = "#f6f8fa",
                 fg: str = "#0969da", accent: str = "#0969da"):
        super().__init__(parent, bg=bg)
        self._collapsed = collapsed
        self._bg = bg

        # Header row
        header = tk.Frame(self, bg=bg, cursor="hand2")
        header.pack(fill=tk.X, padx=8, pady=(6, 0))

        self._arrow = tk.Label(
            header, text=self._ARROW_CLOSED if collapsed else self._ARROW_OPEN,
            font=("Helvetica Neue", 11, "bold"), bg=bg, fg=fg,
        )
        self._arrow.pack(side=tk.LEFT, padx=(0, 4))

        self._title_lbl = tk.Label(
            header, text=title,
            font=("Helvetica Neue", 11, "bold"),
            bg=bg, fg=fg,
        )
        self._title_lbl.pack(side=tk.LEFT)

        for w in (header, self._arrow, self._title_lbl):
            w.bind("<Button-1>", lambda e: self.toggle())
            w.bind("<Enter>", lambda e: self._title_lbl.config(
                font=("Helvetica Neue", 11, "bold underline")))
            w.bind("<Leave>", lambda e: self._title_lbl.config(
                font=("Helvetica Neue", 11, "bold")))

        # Content frame
        self.content = tk.Frame(self, bg=bg)
        if not collapsed:
            self.content.pack(fill=tk.X)

    def toggle(self) -> None:
        self._collapsed = not self._collapsed
        if self._collapsed:
            self.content.pack_forget()
            self._arrow.config(text=self._ARROW_CLOSED)
        else:
            self.content.pack(fill=tk.X)
            self._arrow.config(text=self._ARROW_OPEN)
        self.event_generate("<Configure>")

    @property
    def is_collapsed(self) -> bool:
        return self._collapsed


# ══════════════════════════════════════════════════════════════════════════════
# Phase Indicator
# ══════════════════════════════════════════════════════════════════════════════

class PhaseIndicator(tk.Frame):
    """
    Horizontal pipeline: Configure → Training → Streaming.

    Call .set_phase(state) with one of: "idle", "training", "ready",
    "streaming", "paused", "complete".
    """

    _PHASES = ["Configure", "Training", "Streaming"]

    # Mapping from AppUI state strings to (active_index, completed_indices)
    _STATE_MAP = {
        "idle":      (0, set()),
        "training":  (1, {0}),
        "ready":     (1, {0, 1}),
        "streaming": (2, {0, 1}),
        "paused":    (2, {0, 1}),
        "complete":  (2, {0, 1, 2}),
    }

    def __init__(self, parent: tk.Widget, *, bg: str = "#f6f8fa",
                 accent: str = "#0969da", green: str = "#1a7f37",
                 grey: str = "#8c959f"):
        super().__init__(parent, bg=bg)
        self._accent = accent
        self._green = green
        self._grey = grey
        self._bg = bg

        self._labels: list[tk.Label] = []
        self._arrows: list[tk.Label] = []

        for i, name in enumerate(self._PHASES):
            if i > 0:
                arrow = tk.Label(
                    self, text=" \u203a ", font=("Helvetica Neue", 12),
                    bg=bg, fg=grey,
                )
                arrow.pack(side=tk.LEFT)
                self._arrows.append(arrow)
            lbl = tk.Label(
                self, text=f"  {name}  ",
                font=("Helvetica Neue", 10, "bold"),
                bg=bg, fg=grey, padx=6, pady=3,
            )
            lbl.pack(side=tk.LEFT)
            self._labels.append(lbl)

        self.set_phase("idle")

    def set_phase(self, state: str) -> None:
        active_idx, completed = self._STATE_MAP.get(state, (0, set()))
        for i, lbl in enumerate(self._labels):
            if i in completed and i != active_idx:
                lbl.config(fg=self._green, bg=self._bg,
                           font=("Helvetica Neue", 10))
            elif i == active_idx:
                lbl.config(fg="white", bg=self._accent,
                           font=("Helvetica Neue", 10, "bold"))
            else:
                lbl.config(fg=self._grey, bg=self._bg,
                           font=("Helvetica Neue", 10))


# ══════════════════════════════════════════════════════════════════════════════
# Welcome Overlay
# ══════════════════════════════════════════════════════════════════════════════

class WelcomeOverlay(tk.Frame):
    """
    Semi-opaque overlay with onboarding instructions.
    Uses place() so it layers on top of pack()-managed siblings.

    Call .dismiss() to hide, .show() to re-show.
    """

    def __init__(self, parent: tk.Widget, *, bg: str = "#f6f8fa",
                 fg: str = "#24292f", accent: str = "#0969da",
                 logo_image: tk.PhotoImage | None = None):
        super().__init__(parent, bg=bg)

        spacer = tk.Frame(self, bg=bg, height=40)
        spacer.pack()

        if logo_image is not None:
            tk.Label(self, image=logo_image, bg=bg).pack(pady=(0, 10))

        tk.Label(
            self, text="Welcome to NeuroStream",
            font=("Helvetica Neue", 22, "bold"), bg=bg, fg=fg,
        ).pack(pady=(10, 6))

        tk.Label(
            self, text="Real-time EEG Motor Imagery Decoding",
            font=("Helvetica Neue", 12), bg=bg, fg="#57606a",
        ).pack(pady=(0, 20))

        steps = [
            ("1.", "Set your data folder and press", "Train & Load"),
            ("2.", "Watch the model predict", "LEFT / RIGHT"),
            ("3.", "Observe predictions refine as", "more EEG arrives"),
        ]
        for num, desc, bold in steps:
            row = tk.Frame(self, bg=bg)
            row.pack(anchor="w", padx=60, pady=3)
            tk.Label(
                row, text=num, font=("Helvetica Neue", 12, "bold"),
                bg=bg, fg=accent, width=3, anchor="e",
            ).pack(side=tk.LEFT)
            tk.Label(
                row, text=f"  {desc} ",
                font=("Helvetica Neue", 12), bg=bg, fg=fg,
            ).pack(side=tk.LEFT)
            tk.Label(
                row, text=bold,
                font=("Helvetica Neue", 12, "bold"), bg=bg, fg=accent,
            ).pack(side=tk.LEFT)

        tk.Label(
            self, text="",
            font=("Helvetica Neue", 10), bg=bg, fg="#57606a",
        ).pack(pady=(24, 0))

    def show(self) -> None:
        self.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.lift()

    def dismiss(self) -> None:
        self.place_forget()


# ══════════════════════════════════════════════════════════════════════════════
# Tween Engine
# ══════════════════════════════════════════════════════════════════════════════

class TweenEngine:
    """
    Lightweight linear-interpolation engine for canvas redraws.

    Stores the "current displayed" value and smoothly transitions to a
    new target over a configurable duration using root.after().

    Usage:
        tween = TweenEngine(root)
        tween.animate(
            key="confidence",
            start=np.array([0.5, 0.5]),
            end=np.array([0.3, 0.7]),
            duration_ms=120,
            on_frame=lambda v: draw_confidence(canvas, v, ...),
        )
    """

    _FRAME_MS = 16   # ~60 fps

    def __init__(self, root: tk.Tk):
        self._root = root
        self._active: dict[str, str | None] = {}   # key -> after_id

    def animate(
        self,
        key: str,
        start,
        end,
        duration_ms: int,
        on_frame,
    ) -> None:
        """
        Tween from start to end, calling on_frame(interpolated_value) each
        frame.  If a tween with the same key is already running, it is
        cancelled and replaced.
        """
        # Cancel any existing tween for this key
        if key in self._active and self._active[key] is not None:
            self._root.after_cancel(self._active[key])
            self._active[key] = None

        import numpy as np
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        total_frames = max(1, duration_ms // self._FRAME_MS)

        def _step(frame: int) -> None:
            t = min(1.0, frame / total_frames)
            # Ease-out quadratic for smoother deceleration
            t = 1.0 - (1.0 - t) ** 2
            current = start + (end - start) * t
            on_frame(current)
            if frame < total_frames:
                aid = self._root.after(self._FRAME_MS,
                                       lambda: _step(frame + 1))
                self._active[key] = aid
            else:
                self._active[key] = None

        _step(0)

    def cancel(self, key: str) -> None:
        if key in self._active and self._active[key] is not None:
            self._root.after_cancel(self._active[key])
            self._active[key] = None


# ══════════════════════════════════════════════════════════════════════════════
# Card Frame
# ══════════════════════════════════════════════════════════════════════════════

def make_card(parent: tk.Widget, title: str, *,
              bg: str = "#f6f8fa", card_bg: str = "#ffffff",
              fg: str = "#57606a") -> tk.Frame:
    """
    Creates a card-style container with a subtle title and white background.
    Returns the inner content frame to pack children into.
    """
    outer = tk.Frame(parent, bg=bg)
    tk.Label(
        outer, text=title,
        font=("Helvetica Neue", 9), bg=bg, fg=fg,
    ).pack(anchor="w", padx=2)
    card = tk.Frame(outer, bg=card_bg, bd=1, relief=tk.SOLID,
                    highlightbackground="#d0d7de", highlightthickness=1)
    card.pack(fill=tk.BOTH, expand=True, pady=(2, 0))
    return outer, card
