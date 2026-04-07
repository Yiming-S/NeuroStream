"""
ui/app_view.py — tkinter application window.

Threading model
───────────────
Heavy work (data loading + training) runs in a daemon thread.
Online streaming uses a background acquisition/inference thread that
produces ProgressEvent and TrialResult into queue.Queue objects.
The Tk main thread polls these queues via root.after() and updates the UI.
FBCSP pipelines fall back to the legacy root.after()-driven replay.
"""

import queue
import threading
import time
import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional

import numpy as np

from config import BCIConfig
from data_engine import DataEngine
from model import BCIModel
from sources import ProgressEvent, ReplaySource, RingBuffer, TrialResult
from streaming import StreamingSimulator
from .plots import (
    draw_accuracy_curve,
    draw_band_power,
    draw_confidence,
    draw_confusion_matrix,
    draw_progressive_accuracy,
    draw_trial_chart,
)
from .widgets import (
    CollapsibleSection,
    PhaseIndicator,
    Tooltip,
    TweenEngine,
    WelcomeOverlay,
    make_card,
)

_LOGO_PATH = Path(__file__).resolve().parent / "img" / "neurostream.png"


class AppUI:
    """
    Main application window.

    Layout
    ──────
    ┌──────────────────────────────────────────────┐
    │          BCI Real-Time Motor Imagery Demo     │
    ├──────────────────────┬───────────────────────┤
    │   Settings Panel     │   Display Panel        │
    │   • Data Folder      │   • Countdown / bar    │
    │   • Action buttons   │   • Prediction/Actual  │
    │   • Parameters (↕)   │   • Confidence bar     │
    │                      │   • Band Power         │
    │                      │   • Trial History      │
    │                      │   • Confusion Matrix   │
    └──────────────────────┴───────────────────────┘
    """

    DISPLAY_INTERVAL_MS = 2500
    ACTUAL_DELAY_MS     = 1500

    # Colour palette
    BG_COLOR = "#f6f8fa"
    FG_COLOR = "#24292f"
    ACCENT   = "#0969da"
    GREEN    = "#1a7f37"
    RED      = "#cf222e"
    YELLOW   = "#9a6700"

    # Button state colours
    _BTN_TRAIN_ON   = "#0969da"
    _BTN_START_ON   = "#1a7f37"
    _BTN_STOP_ON    = "#cf222e"
    _BTN_STOP_PULSE = "#fa4549"
    _BTN_OFF        = "#eaeef2"
    _BTN_FG_OFF     = "#8c959f"

    # ------------------------------------------------------------------
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("NeuroStream — Motor Imagery Demo")
        self.root.configure(bg=self.BG_COLOR)
        self.root.resizable(True, True)

        self.config    = BCIConfig()
        self.data_engine: Optional[DataEngine]         = None
        self.model:       Optional[BCIModel]           = None
        self.simulator:   Optional[StreamingSimulator] = None

        # Online architecture state
        self._source: Optional[ReplaySource] = None
        self._buffer: Optional[RingBuffer] = None
        self._trial_onsets: List[int] = []
        self._trial_labels: List[Optional[int]] = []
        self._n_total_trials: int = 0
        self._progress_queue: queue.Queue = queue.Queue()
        self._result_queue: queue.Queue = queue.Queue()
        self._stream_done = threading.Event()
        self._acq_thread: Optional[threading.Thread] = None
        self._displaying: bool = False
        self._demo_result_buffer: deque = deque()

        # Runtime state
        self._running   = False
        self._paused    = False
        self._trial_idx = 0
        self._correct   = 0
        self._total     = 0
        self._history:    list = []
        self._btn_enabled: dict = {}
        self._btns:        dict = {}

        # Visualisation state
        self._sfreq:      float = 250.0
        self._conf_matrix: list = [[0, 0], [0, 0]]
        self._prog_sample_points: List[int] = []
        self._prog_time_labels: List[float] = []
        self._prog_accuracy: dict = {}  # ns -> [correct, total]
        self._last_proba: np.ndarray = np.array([0.5, 0.5])
        self._last_bp: np.ndarray = np.array([0.0, 0.0])
        self._showing_summary: bool = False

        self._build_ui()
        self._tween = TweenEngine(self.root)

    # ══════════════════════════════════════════════════════════════════
    # UI Construction
    # ══════════════════════════════════════════════════════════════════

    def _build_ui(self) -> None:
        header_frame = tk.Frame(self.root, bg=self.BG_COLOR)
        header_frame.pack(fill=tk.X, padx=20, pady=(14, 10))
        header_frame.grid_columnconfigure(0, weight=1)
        header_frame.grid_columnconfigure(1, weight=0)
        header_frame.grid_columnconfigure(2, weight=1)

        title_block = tk.Frame(header_frame, bg=self.BG_COLOR)
        title_block.grid(row=0, column=1, sticky="n")

        # App icon / logo
        try:
            from PIL import Image, ImageTk
            _img = Image.open(_LOGO_PATH).resize((84, 84), Image.LANCZOS)
            self._logo_img = ImageTk.PhotoImage(_img)
            tk.Label(
                header_frame, image=self._logo_img, bg=self.BG_COLOR,
            ).grid(row=0, column=0, rowspan=2, sticky="w", padx=(44, 18))
        except Exception:
            pass  # skip logo if Pillow is unavailable

        tk.Label(
            title_block,
            text="BCI  Real-Time Motor Imagery Demo",
            font=("Helvetica Neue", 20, "bold"),
            bg=self.BG_COLOR, fg=self.FG_COLOR,
        ).pack()

        tk.Label(
            title_block,
            text="Zhou2016  ·  Cross-Subject Classification  ·  CSP + LDA / SVM",
            font=("Helvetica Neue", 10),
            bg=self.BG_COLOR, fg="#57606a",
        ).pack(pady=(6, 0))

        tk.Label(
            title_block,
            text="Yiming Shen  ·  Department of Mathematics  ·  "
                 "University of Massachusetts Boston",
            font=("Helvetica Neue", 9),
            bg=self.BG_COLOR, fg="#8c959f",
        ).pack(pady=(3, 0))

        tk.Frame(self.root, height=1, bg="#d0d7de").pack(fill=tk.X, padx=20)

        main_frame = tk.Frame(self.root, bg=self.BG_COLOR)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=8)

        self._build_settings_panel(main_frame)
        self._build_display_panel(main_frame)

        self.status_bar = tk.Label(
            self.root,
            text="Ready. Configure parameters and press [Train & Load].",
            font=("Helvetica Neue", 10),
            bg="#f0f3f6", fg="#57606a",
            anchor="w", padx=8,
        )
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=(8, 0))

    # ------------------------------------------------------------------
    def _build_settings_panel(self, parent: tk.Frame) -> None:
        panel = tk.LabelFrame(
            parent,
            text=" Parameters ",
            font=("Helvetica Neue", 11, "bold"),
            bg=self.BG_COLOR, fg=self.ACCENT,
            relief=tk.RIDGE, bd=2,
        )
        panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 12), pady=4)

        self._entries: dict = {}
        self._sliders: dict = {}

        # ── Section 1: Data Folder (always visible) ───────────────────
        tk.Label(
            panel, text="Data Folder:",
            font=("Helvetica Neue", 10, "bold"),
            bg=self.BG_COLOR, fg=self.YELLOW, anchor="w",
        ).pack(fill=tk.X, padx=10, pady=(8, 0))

        path_row = tk.Frame(panel, bg=self.BG_COLOR)
        path_row.pack(fill=tk.X, padx=10, pady=(2, 4))

        self._data_path_var = tk.StringVar(value="")
        path_entry = tk.Entry(
            path_row, textvariable=self._data_path_var,
            font=("Helvetica Neue", 9),
            bg="#ffffff", fg=self.FG_COLOR,
            insertbackground=self.FG_COLOR, relief=tk.FLAT, bd=4,
        )
        path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        PLACEHOLDER = "Paste folder path here  (e.g. …/EEG_MotorImagery)"
        path_entry.insert(0, PLACEHOLDER)
        path_entry.config(fg="#57606a")

        def _on_focus_in(e):
            if self._data_path_var.get() == PLACEHOLDER:
                path_entry.delete(0, tk.END)
                path_entry.config(fg=self.FG_COLOR)

        def _on_focus_out(e):
            if not self._data_path_var.get().strip():
                path_entry.insert(0, PLACEHOLDER)
                path_entry.config(fg="#57606a")

        path_entry.bind("<FocusIn>",  _on_focus_in)
        path_entry.bind("<FocusOut>", _on_focus_out)

        def browse_folder():
            chosen = filedialog.askdirectory(
                title="Select MNE data folder (contains MNE-zhou-2016/)",
                initialdir=self._data_path_var.get(),
            )
            if chosen:
                self._data_path_var.set(chosen)

        browse_frame = tk.Frame(path_row, bg="#24292f", cursor="hand2")
        browse_frame.pack(side=tk.LEFT, padx=(4, 0))
        browse_lbl = tk.Label(
            browse_frame, text="Browse",
            font=("Helvetica Neue", 9, "bold"),
            bg="#24292f", fg="white",
            padx=8, pady=3, cursor="hand2",
        )
        browse_lbl.pack()
        for w in (browse_frame, browse_lbl):
            w.bind("<Button-1>", lambda e: browse_folder())
            w.bind("<Enter>",    lambda e: browse_frame.config(bg="#444d56") or browse_lbl.config(bg="#444d56"))
            w.bind("<Leave>",    lambda e: browse_frame.config(bg="#24292f") or browse_lbl.config(bg="#24292f"))

        self.lbl_data_status = tk.Label(
            panel, text="",
            font=("Helvetica Neue", 9),
            bg=self.BG_COLOR, fg="#57606a", anchor="w",
        )
        self.lbl_data_status.pack(fill=tk.X, padx=10, pady=(0, 2))

        # ── Section 2: Action Buttons (always visible) ────────────────
        tk.Frame(panel, height=1, bg="#333").pack(fill=tk.X, padx=10, pady=(4, 2))

        def _make_btn(name: str, text: str, handler):
            bg = self._BTN_TRAIN_ON if name == "train" else self._BTN_OFF
            fg = "white" if name == "train" else self._BTN_FG_OFF
            f  = tk.Frame(panel, bg=bg)
            f.pack(fill=tk.X, padx=10, pady=2)
            dot = tk.Label(f, text="●", font=("Helvetica Neue", 10),
                           bg=bg, fg="#8c959f")
            dot.pack(side=tk.LEFT, padx=(10, 4))
            lbl = tk.Label(f, text=text,
                           font=("Helvetica Neue", 13, "bold"),
                           bg=bg, fg=fg, pady=8, padx=4, anchor="w")
            lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)

            def on_click(e, h=handler, n=name):
                if self._btn_enabled.get(n):
                    h()
            for w in (f, dot, lbl):
                w.bind("<Button-1>", on_click)
            self._btns[name] = (f, dot, lbl)

        _make_btn("train",   "⚙   Train & Load", self._on_train)
        _make_btn("start",   "▶   Start Stream",  self._on_start)
        _make_btn("pause",   "⏸   Pause",          self._on_pause)
        _make_btn("stop",    "⏹   Stop",           self._on_stop)
        _make_btn("summary", "📊   Summary",       self._on_summary)
        self._update_button_states("idle")

        # ── Section 3: Parameters (scrollable) ────────────────────────
        tk.Frame(panel, height=1, bg="#333").pack(fill=tk.X, padx=10, pady=(4, 0))

        scroll_outer = tk.Frame(panel, bg=self.BG_COLOR)
        scroll_outer.pack(fill=tk.BOTH, expand=True)

        self._param_sb = ttk.Scrollbar(scroll_outer, orient="vertical")
        sc = tk.Canvas(scroll_outer, bg=self.BG_COLOR, highlightthickness=0,
                       yscrollcommand=self._param_sb.set)
        self._param_sb.config(command=sc.yview)
        sc.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner = tk.Frame(sc, bg=self.BG_COLOR)
        win_id = sc.create_window((0, 0), window=inner, anchor="nw")

        def _sync_scroll(e=None):
            sc.configure(scrollregion=sc.bbox("all"))
            if inner.winfo_reqheight() > sc.winfo_height():
                self._param_sb.pack(side=tk.RIGHT, fill=tk.Y)
            else:
                self._param_sb.pack_forget()

        def _fit_width(e):
            sc.itemconfig(win_id, width=e.width)
            _sync_scroll()

        inner.bind("<Configure>", _sync_scroll)
        sc.bind("<Configure>", _fit_width)
        sc.bind("<MouseWheel>",
                lambda e: sc.yview_scroll(int(-1 * e.delta / 120), "units"))
        inner.bind("<MouseWheel>",
                   lambda e: sc.yview_scroll(int(-1 * e.delta / 120), "units"))

        # ── Parameter section helpers ──────────────────────────────────
        def add_slider(parent_frame, label, key, from_, to, default, is_int=False):
            row = tk.Frame(parent_frame, bg=self.BG_COLOR)
            row.pack(fill=tk.X, padx=6, pady=2)
            tk.Label(row, text=label, width=9, anchor="w",
                     font=("Helvetica Neue", 9),
                     bg=self.BG_COLOR, fg=self.FG_COLOR).pack(side=tk.LEFT)
            var = tk.DoubleVar(value=default)
            fmt = "d" if is_int else ".1f"
            val_lbl = tk.Label(
                row, text=f"{int(default) if is_int else default:{fmt}}",
                width=5, anchor="e",
                font=("Helvetica Neue", 9, "bold"),
                bg=self.BG_COLOR, fg=self.ACCENT,
            )

            def _on_slide(v, lbl=val_lbl, ii=is_int):
                lbl.config(text=str(round(float(v))) if ii else f"{float(v):.1f}")

            ttk.Scale(row, from_=from_, to=to, variable=var,
                      orient=tk.HORIZONTAL, command=_on_slide
                      ).pack(side=tk.LEFT, fill=tk.X, expand=True)
            val_lbl.pack(side=tk.RIGHT, padx=(4, 0))
            self._sliders[key] = var

        def section(title, color="#0969da", parent=None):
            frm = tk.LabelFrame(
                parent or inner, text=f" {title} ",
                font=("Helvetica Neue", 9, "bold"),
                bg=self.BG_COLOR, fg=color,
                relief=tk.GROOVE, bd=1, padx=4, pady=3,
            )
            frm.pack(fill=tk.X, padx=8, pady=(6, 0))
            return frm

        # Training Setup
        tr = section("Training Setup", color="#1a7f37")

        def add_entry_row(parent, label, key, default, width=8):
            row = tk.Frame(parent, bg=self.BG_COLOR)
            row.pack(fill=tk.X, padx=6, pady=2)
            tk.Label(row, text=label, width=18, anchor="w",
                     font=("Helvetica Neue", 9),
                     bg=self.BG_COLOR, fg=self.FG_COLOR).pack(side=tk.LEFT)
            var = tk.StringVar(value=str(default))
            tk.Entry(row, textvariable=var, width=width,
                     font=("Helvetica Neue", 9),
                     bg="#ffffff", fg=self.FG_COLOR,
                     insertbackground=self.FG_COLOR,
                     relief=tk.FLAT, bd=3).pack(side=tk.LEFT)
            self._entries[key] = var
            return row

        # ── Evaluation Protocol selector ─────────────────────────────────
        protocol_row = tk.Frame(tr, bg=self.BG_COLOR)
        protocol_row.pack(fill=tk.X, padx=6, pady=(4, 2))
        tk.Label(protocol_row, text="Evaluation Protocol:", width=18, anchor="w",
                 font=("Helvetica Neue", 9, "bold"),
                 bg=self.BG_COLOR, fg=self.FG_COLOR).pack(side=tk.LEFT)
        self._protocol_var = tk.StringVar(value=self.config.evaluation_protocol)
        protocol_cb = ttk.Combobox(
            protocol_row, textvariable=self._protocol_var,
            values=["Cross-Subject", "Cross-Session"],
            width=14, state="readonly",
        )
        protocol_cb.pack(side=tk.LEFT)

        self._protocol_tooltip = Tooltip(protocol_cb, "")

        # ── Cross-Subject parameter rows ─────────────────────────────────
        self._cross_subject_rows = tk.Frame(tr, bg=self.BG_COLOR)
        self._cross_subject_rows.pack(fill=tk.X)
        add_entry_row(self._cross_subject_rows,
                      "Train Subjects:",  "train_subjects",
                      "1,2", width=8)
        add_entry_row(self._cross_subject_rows,
                      "Test Subject:",    "test_subject",
                      self.config.test_subject, width=5)

        # ── Cross-Session parameter rows ─────────────────────────────────
        self._cross_session_rows = tk.Frame(tr, bg=self.BG_COLOR)
        # (not packed initially — shown only when Cross-Session is selected)
        add_entry_row(self._cross_session_rows,
                      "Subject:",         "cross_session_subject",
                      self.config.cross_session_subject, width=5)
        add_entry_row(self._cross_session_rows,
                      "Train Sessions:",  "train_sessions",
                      "0,1", width=8)
        add_entry_row(self._cross_session_rows,
                      "Test Session:",    "test_session",
                      self.config.test_session, width=5)

        def _on_protocol_change(*_):
            protocol = self._protocol_var.get()
            if protocol == "Cross-Subject":
                self._cross_session_rows.pack_forget()
                self._cross_subject_rows.pack(fill=tk.X)
                self._protocol_tooltip.update_text(
                    "Train on data from multiple subjects. "
                    "Test on a different held-out subject.")
            else:  # Cross-Session
                self._cross_subject_rows.pack_forget()
                self._cross_session_rows.pack(fill=tk.X)
                self._protocol_tooltip.update_text(
                    "Train on earlier sessions of one subject. "
                    "Test on a later session of the same subject. "
                    "Zhou2016 has sessions 0, 1, 2.")
            self.root.after(50, lambda: inner.event_generate("<Configure>"))

        protocol_cb.bind("<<ComboboxSelected>>", _on_protocol_change)
        _on_protocol_change()   # set initial state

        # ── Feature Extraction ──────────────────────────────────────────
        feat_row = tk.Frame(tr, bg=self.BG_COLOR)
        feat_row.pack(fill=tk.X, padx=6, pady=2)
        tk.Label(feat_row, text="Feature Extraction:", width=16, anchor="w",
                 font=("Helvetica Neue", 9),
                 bg=self.BG_COLOR, fg=self.FG_COLOR).pack(side=tk.LEFT)
        self._feat_var = tk.StringVar(value="CSP")
        feat_cb = ttk.Combobox(feat_row, textvariable=self._feat_var,
                               values=["CSP", "FBCSP", "TS (Riemannian)"],
                               width=14, state="readonly")
        feat_cb.pack(side=tk.LEFT)

        self._feat_tooltip = Tooltip(feat_cb, "")

        # ── Classifier ──────────────────────────────────────────────────
        clf_row = tk.Frame(tr, bg=self.BG_COLOR)
        clf_row.pack(fill=tk.X, padx=6, pady=2)
        tk.Label(clf_row, text="Classifier:", width=16, anchor="w",
                 font=("Helvetica Neue", 9),
                 bg=self.BG_COLOR, fg=self.FG_COLOR).pack(side=tk.LEFT)
        self._clf_var = tk.StringVar(value=self.config.clf_type)
        self._clf_cb  = ttk.Combobox(clf_row, textvariable=self._clf_var,
                                     values=["LDA", "SVM"],
                                     width=8, state="readonly")
        self._clf_cb.pack(side=tk.LEFT)

        # Wire up the dynamic update
        def _on_feat_change(*_):
            feat = self._feat_var.get()
            if feat == "CSP":
                self._sp_frame.pack(fill=tk.X, padx=8, pady=(6, 0))
                self._bands_row.pack_forget()
                self._clf_cb.config(values=["LDA", "SVM"])
                if self._clf_var.get() == "MDM":
                    self._clf_var.set("LDA")
                self._feat_tooltip.update_text(
                    "Single bandpass → CSP spatial filters → LDA/SVM. "
                    "Classic MOABB baseline (Jayaram & Barachant 2018).")
            elif feat == "FBCSP":
                self._sp_frame.pack(fill=tk.X, padx=8, pady=(6, 0))
                self._bands_row.pack(fill=tk.X, padx=6, pady=2)
                self._clf_cb.config(values=["LDA", "SVM"])
                if self._clf_var.get() == "MDM":
                    self._clf_var.set("LDA")
                self._feat_tooltip.update_text(
                    "FilterBankLeftRightImagery (moabb): CSP per band, "
                    "features concatenated → LDA/SVM. Ang et al. 2012.")
            else:  # TS (Riemannian)
                self._sp_frame.pack_forget()
                self._clf_cb.config(values=["LDA", "SVM", "MDM"])
                self._feat_tooltip.update_text(
                    "pyriemann: covariances → Tangent Space (LDA/SVM) "
                    "or Minimum Distance to Mean (MDM). "
                    "Barachant et al. 2012/2013.")
            self.root.after(50, lambda: inner.event_generate("<Configure>"))

        feat_cb.bind("<<ComboboxSelected>>", _on_feat_change)

        # ── Advanced (collapsible) ────────────────────────────────────
        self._advanced = CollapsibleSection(
            inner, title="\u2699  Advanced Parameters  \u25b8", collapsed=True,
            bg=self.BG_COLOR, fg=self.ACCENT, accent=self.ACCENT,
        )
        self._advanced.pack(fill=tk.X)

        # Bandpass Filter
        bp = section("Bandpass Filter", parent=self._advanced.content)
        Tooltip(bp, "Retain mu (8-12 Hz) and beta (13-30 Hz) motor rhythms, reject noise.")
        add_slider(bp, "Low:",  "f_low",  1.0, 30.0, self.config.f_low)
        add_slider(bp, "High:", "f_high", 10.0, 60.0, self.config.f_high)

        # Epoch Window
        ep = section("Epoch Window", parent=self._advanced.content)
        Tooltip(ep, "EEG segment per trial, relative to stimulus onset (seconds).")
        add_slider(ep, "Start:", "t_min", -1.0, 2.0, self.config.t_min)
        add_slider(ep, "End:",   "t_max",  0.5, 8.0, self.config.t_max)

        # Progressive Prediction
        pg = section("Progressive Prediction", color="#7d4e00",
                     parent=self._advanced.content)
        Tooltip(pg, "Refresh tentative predictions during EEG collection at sub-windows.")

        prog_toggle_row = tk.Frame(pg, bg=self.BG_COLOR)
        prog_toggle_row.pack(fill=tk.X, padx=6, pady=(2, 0))
        self._prog_var = tk.BooleanVar(value=self.config.progressive)
        tk.Checkbutton(
            prog_toggle_row,
            text="Enable",
            variable=self._prog_var,
            font=("Helvetica Neue", 9),
            bg=self.BG_COLOR,
            fg=self.FG_COLOR,
            activebackground=self.BG_COLOR,
            activeforeground=self.FG_COLOR,
            selectcolor="#ffffff",
            highlightthickness=0,
            anchor="w",
        ).pack(side=tk.LEFT)

        prog_step_row = tk.Frame(pg, bg=self.BG_COLOR)
        prog_step_row.pack(fill=tk.X, padx=6, pady=2)
        tk.Label(
            prog_step_row, text="Step:", width=9, anchor="w",
            font=("Helvetica Neue", 9),
            bg=self.BG_COLOR, fg=self.FG_COLOR,
        ).pack(side=tk.LEFT)
        prog_step_var = tk.DoubleVar(value=self.config.prog_step)
        self._sliders["prog_step"] = prog_step_var
        self._prog_step_value_lbl = tk.Label(
            prog_step_row, text=f"{self.config.prog_step:.2f} s",
            width=7, anchor="e",
            font=("Helvetica Neue", 9, "bold"),
            bg=self.BG_COLOR, fg=self.ACCENT,
        )

        def _on_prog_step_slide(v):
            self._prog_step_value_lbl.config(text=f"{float(v):.2f} s")

        self._prog_step_scale = ttk.Scale(
            prog_step_row,
            from_=0.25,
            to=1.0,
            variable=prog_step_var,
            orient=tk.HORIZONTAL,
            command=_on_prog_step_slide,
        )
        self._prog_step_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._prog_step_value_lbl.pack(side=tk.RIGHT, padx=(4, 0))

        def _sync_progressive_controls(*_):
            enabled = self._prog_var.get()
            if enabled:
                self._prog_step_scale.state(["!disabled"])
                self._prog_step_value_lbl.config(fg=self.ACCENT)
            else:
                self._prog_step_scale.state(["disabled"])
                self._prog_step_value_lbl.config(fg=self._BTN_FG_OFF)

        self._prog_var.trace_add("write", _sync_progressive_controls)
        _sync_progressive_controls()

        # Spatial Filters — shown for CSP and FBCSP, hidden for TS
        self._sp_frame = section("Spatial Filters",
                                 parent=self._advanced.content)
        add_slider(self._sp_frame, "Filters (n):", "csp_components", 2, 14,
                   self.config.csp_components, is_int=True)

        # Bands row — only shown for FBCSP
        self._bands_row = tk.Frame(self._sp_frame, bg=self.BG_COLOR)
        tk.Label(self._bands_row, text="Freq. Bands:", width=12, anchor="w",
                 font=("Helvetica Neue", 9),
                 bg=self.BG_COLOR, fg=self.FG_COLOR).pack(side=tk.LEFT)
        self._fb_bands_var = tk.StringVar(value=self.config.fb_bands)
        tk.Entry(self._bands_row, textvariable=self._fb_bands_var, width=22,
                 font=("Helvetica Neue", 9),
                 bg="#ffffff", fg=self.FG_COLOR,
                 insertbackground=self.FG_COLOR,
                 relief=tk.FLAT, bd=3).pack(side=tk.LEFT)
        tk.Label(self._bands_row, text=" Hz  (fmin-fmax, …)",
                 font=("Helvetica Neue", 8),
                 bg=self.BG_COLOR, fg="#57606a").pack(side=tk.LEFT)

        _on_feat_change()   # set initial state (after _sp_frame exists)

        tk.Frame(inner, height=8, bg=self.BG_COLOR).pack()   # bottom padding

    # ------------------------------------------------------------------
    def _build_display_panel(self, parent: tk.Frame) -> None:
        outer = tk.LabelFrame(
            parent,
            text=" Live Feed ",
            font=("Helvetica Neue", 11, "bold"),
            bg=self.BG_COLOR, fg=self.ACCENT,
            relief=tk.RIDGE, bd=2,
        )
        outer.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=4)
        self._display_panel = outer   # keep reference for overlay

        # Scrollable interior
        self._display_sb = ttk.Scrollbar(outer, orient="vertical")
        disp_canvas = tk.Canvas(outer, bg=self.BG_COLOR, highlightthickness=0,
                                yscrollcommand=self._display_sb.set)
        self._display_sb.config(command=disp_canvas.yview)
        disp_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        panel = tk.Frame(disp_canvas, bg=self.BG_COLOR)
        win_id = disp_canvas.create_window((0, 0), window=panel, anchor="nw")

        def _sync_display_scroll(e=None):
            disp_canvas.configure(scrollregion=disp_canvas.bbox("all"))
            if panel.winfo_reqheight() > disp_canvas.winfo_height():
                self._display_sb.pack(side=tk.RIGHT, fill=tk.Y)
            else:
                self._display_sb.pack_forget()

        def _fit_display_width(e):
            disp_canvas.itemconfig(win_id, width=e.width)
            _sync_display_scroll()

        panel.bind("<Configure>", _sync_display_scroll)
        disp_canvas.bind("<Configure>", _fit_display_width)
        disp_canvas.bind("<MouseWheel>",
                         lambda e: disp_canvas.yview_scroll(
                             int(-1 * e.delta / 120), "units"))
        panel.bind("<MouseWheel>",
                   lambda e: disp_canvas.yview_scroll(
                       int(-1 * e.delta / 120), "units"))

        # Phase indicator
        self._phase_indicator = PhaseIndicator(
            panel, bg=self.BG_COLOR,
            accent=self.ACCENT, green=self.GREEN, grey="#8c959f",
        )
        self._phase_indicator.pack(fill=tk.X, padx=10, pady=(6, 2))

        def make_label(text, font_size=14, color=None):
            lbl = tk.Label(panel, text=text,
                           font=("Helvetica Neue", font_size),
                           bg=self.BG_COLOR, fg=color or self.FG_COLOR)
            lbl.pack(pady=6)
            return lbl

        self.lbl_trial     = make_label("Trial:  —", 12, "#888")
        self.lbl_countdown = make_label("EEG Window:  — s", 20, self.YELLOW)

        self.progress_bar = ttk.Progressbar(
            panel, orient="horizontal", mode="determinate", maximum=100,
        )
        self.progress_bar.pack(fill=tk.X, padx=30, pady=(0, 8))

        self.lbl_status    = make_label("Status:  Idle", 13, "#57606a")

        tk.Frame(panel, height=1, bg="#d0d7de").pack(fill=tk.X, padx=20, pady=6)

        self.lbl_prediction = make_label("Prediction:  —", 26, self.ACCENT)
        self.lbl_actual     = make_label("Actual:  —",     20, self.GREEN)

        # Confidence bar
        tk.Label(panel, text="Classifier Confidence",
                 font=("Helvetica Neue", 9), bg=self.BG_COLOR, fg="#57606a").pack()
        self._conf_canvas = tk.Canvas(panel, height=36, bg="#eaeef2",
                                      highlightthickness=0)
        self._conf_canvas.pack(fill=tk.X, padx=20, pady=(2, 8))

        tk.Frame(panel, height=1, bg="#d0d7de").pack(fill=tk.X, padx=20, pady=4)

        self.lbl_accuracy  = make_label("Online Accuracy:  —", 14, self.FG_COLOR)
        self.lbl_train_acc = make_label("—", 11, "#888")

        # Trial history chart (card)
        chart_outer, chart_card = make_card(
            panel, "Trial-by-Trial History",
            bg=self.BG_COLOR, card_bg="#ffffff", fg="#57606a",
        )
        chart_outer.pack(fill=tk.X, padx=16, pady=(8, 4))
        self._chart_canvas = tk.Canvas(chart_card, height=100, bg="#ffffff",
                                       highlightthickness=0)
        self._chart_canvas.pack(fill=tk.X, padx=6, pady=6)

        # Band Power + Confusion Matrix (side by side, cards)
        vis_row = tk.Frame(panel, bg=self.BG_COLOR)
        vis_row.pack(fill=tk.X, padx=16, pady=(0, 8))

        bp_outer, bp_card = make_card(
            vis_row, "Band Power  (\u03bc / \u03b2)",
            bg=self.BG_COLOR, card_bg="#ffffff", fg="#57606a",
        )
        bp_outer.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._bp_canvas = tk.Canvas(bp_card, height=80, bg="#ffffff",
                                    highlightthickness=0)
        self._bp_canvas.pack(fill=tk.X, padx=6, pady=6)

        tk.Frame(vis_row, width=8, bg=self.BG_COLOR).pack(side=tk.LEFT)

        cm_outer, cm_card = make_card(
            vis_row, "Confusion Matrix",
            bg=self.BG_COLOR, card_bg="#ffffff", fg="#57606a",
        )
        cm_outer.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._cm_canvas = tk.Canvas(cm_card, height=80, bg="#ffffff",
                                    highlightthickness=0)
        self._cm_canvas.pack(fill=tk.X, padx=6, pady=6)

        # Welcome overlay — shown until first training completes
        # Placed on outer LabelFrame so it covers the scroll area
        logo_img = getattr(self, "_logo_img", None)
        self._welcome = WelcomeOverlay(
            outer, bg=self.BG_COLOR, fg=self.FG_COLOR,
            accent=self.ACCENT, logo_image=logo_img,
        )
        self._welcome.show()

    # ══════════════════════════════════════════════════════════════════
    # Parameter Reading
    # ══════════════════════════════════════════════════════════════════

    def _read_config(self) -> bool:
        """Parse UI entries into self.config. Returns False on validation error."""
        try:
            raw_path = self._data_path_var.get().strip()
            self.config.data_path      = "" if raw_path.startswith("Paste folder") else raw_path
            self.config.f_low          = round(self._sliders["f_low"].get(), 1)
            self.config.f_high         = round(self._sliders["f_high"].get(), 1)
            self.config.t_min          = round(self._sliders["t_min"].get(), 1)
            self.config.t_max          = round(self._sliders["t_max"].get(), 1)
            self.config.progressive    = self._prog_var.get()
            self.config.prog_step      = round(self._sliders["prog_step"].get(), 2)
            self.config.csp_components = round(self._sliders["csp_components"].get())
            self.config.clf_type       = self._clf_var.get()
            feat = self._feat_var.get()
            clf  = self._clf_var.get()
            if feat == "CSP":
                self.config.pipeline_type = "CSP"
            elif feat == "FBCSP":
                self.config.pipeline_type = "FBCSP"
                # Validate and store frequency bands string
                raw_bands = self._fb_bands_var.get().strip()
                for b in raw_bands.split(","):
                    lo, hi = (float(x) for x in b.strip().split("-"))
                    if lo >= hi:
                        raise ValueError(f"Band {b.strip()}: fmin must be < fmax.")
                self.config.fb_bands = raw_bands
            else:  # TS (Riemannian)
                if clf == "MDM":
                    self.config.pipeline_type = "MDM"
                elif clf == "SVM":
                    self.config.pipeline_type = "TS+SVM"
                else:
                    self.config.pipeline_type = "TS+LDA"
            # Evaluation protocol and its specific parameters
            self.config.evaluation_protocol = self._protocol_var.get()

            if self.config.evaluation_protocol == "Cross-Subject":
                raw_subjects = self._entries["train_subjects"].get()
                self.config.train_subjects = [
                    int(s.strip()) for s in raw_subjects.split(",")
                ]
                self.config.test_subject = int(self._entries["test_subject"].get())
            else:  # Cross-Session
                self.config.cross_session_subject = int(
                    self._entries["cross_session_subject"].get()
                )
                raw_sessions = self._entries["train_sessions"].get()
                self.config.train_sessions = [
                    int(s.strip()) for s in raw_sessions.split(",")
                ]
                self.config.test_session = int(self._entries["test_session"].get())

        except ValueError as exc:
            messagebox.showerror("Parameter Error", f"Invalid parameter value:\n{exc}")
            return False

        if self.config.f_low >= self.config.f_high:
            messagebox.showerror("Parameter Error", "f_low must be < f_high.")
            return False
        if self.config.t_min >= self.config.t_max:
            messagebox.showerror("Parameter Error", "t_min must be < t_max.")
            return False
        if self.config.csp_components < 2:
            messagebox.showerror("Parameter Error", "CSP components must be ≥ 2.")
            return False
        if self.config.progressive and self.config.prog_step < 0.25:
            messagebox.showerror(
                "Parameter Error",
                "Progressive step must be at least 0.25 seconds.",
            )
            return False

        duration = self.config.t_max - self.config.t_min
        if self.config.progressive and self.config.prog_step > duration:
            messagebox.showerror(
                "Parameter Error",
                "Progressive step must be ≤ epoch duration.",
            )
            return False

        if self.config.evaluation_protocol == "Cross-Session":
            if self.config.test_session in self.config.train_sessions:
                messagebox.showerror(
                    "Parameter Error",
                    f"Test Session ({self.config.test_session}) must not overlap "
                    "with Train Sessions.",
                )
                return False

        return True

    # ══════════════════════════════════════════════════════════════════
    # Button Callbacks
    # ══════════════════════════════════════════════════════════════════

    def _on_train(self) -> None:
        if not self._read_config():
            return

        data_path = self.config.data_path
        if not data_path:
            messagebox.showerror(
                "No Data Path",
                "Please enter the folder path that contains MNE-zhou-2016/",
            )
            return

        if DataEngine.data_exists(data_path):
            self.lbl_data_status.config(text="Local data found.", fg=self.GREEN)
        else:
            self.lbl_data_status.config(text="Data NOT found at this path.", fg=self.RED)
            answer = messagebox.askyesno(
                title="Data Not Found",
                message=(
                    f"Zhou2016 data was not found at:\n{data_path}\n\n"
                    "Download now? (~200 MB)\n\n"
                    "Select [No] to pick a different folder first."
                ),
            )
            if not answer:
                self._set_status("Aborted. Please set the correct Data Folder path.")
                return
            self.lbl_data_status.config(
                text="Downloading…  (existing files are never deleted)", fg=self.YELLOW
            )

        self._update_button_states("training")
        self._set_status("Loading data & training model …  (this may take ~1 min)")
        self.lbl_train_acc.config(text="Pre-training in progress…")
        threading.Thread(target=self._train_worker, daemon=True).start()

    def _on_start(self) -> None:
        if self.simulator is None and self._source is None:
            return
        self._running = True
        self._paused = False
        self._trial_idx = 0
        self._correct   = 0
        self._total     = 0
        self._history   = []
        self._conf_matrix = [[0, 0], [0, 0]]
        self._prog_accuracy = {ns: [0, 0] for ns in self._prog_sample_points}
        self._last_proba = np.array([0.5, 0.5])
        self._last_bp = np.array([0.0, 0.0])
        self._showing_summary = False
        self._displaying = False
        self._demo_result_buffer.clear()
        self._tween.cancel("confidence")
        self._tween.cancel("band_power")
        for c in (self._chart_canvas, self._conf_canvas,
                  self._bp_canvas, self._cm_canvas):
            c.delete("all")
        self._update_button_states("streaming")
        self._pulse_stop()

        if self._source is not None:
            # ── Online architecture path ──
            self._source.start()
            self._buffer.reset()
            self._stream_done.clear()
            while not self._progress_queue.empty():
                self._progress_queue.get_nowait()
            while not self._result_queue.empty():
                self._result_queue.get_nowait()
            self._acq_thread = threading.Thread(
                target=self._acquisition_worker, daemon=True)
            self._acq_thread.start()
            self._set_status("Streaming (online) …")
            self.root.after(20, self._ui_poll)
        else:
            # ── FBCSP legacy path ──
            self.simulator.reset()
            self._set_status("Streaming …")
            self.root.after(200, self._stream_loop)

    def _on_pause(self) -> None:
        if not self._running:
            return
        self._paused = not self._paused
        _, _, lbl = self._btns["pause"]
        if self._paused:
            lbl.config(text="▶   Resume")
            if isinstance(self._source, ReplaySource):
                self._source.pause()
            self._update_button_states("paused")
            self._set_status("Paused.")
        else:
            # Restore live UI if we were showing summary
            if self._showing_summary:
                self._showing_summary = False
                self.lbl_trial.config(fg="#888", font=("Helvetica Neue", 12))
                self.lbl_countdown.config(
                    fg=self.YELLOW, font=("Helvetica Neue", 20))
                self.lbl_prediction.config(
                    text="Prediction:  —", fg=self.ACCENT,
                    font=("Helvetica Neue", 26))
                self.lbl_actual.config(
                    text="Actual:  —", fg="#8c959f",
                    font=("Helvetica Neue", 20))
                self._conf_canvas.config(height=36)
                self._conf_canvas.delete("all")
                self._bp_canvas.delete("all")
            if isinstance(self._source, ReplaySource):
                self._source.resume()
            lbl.config(text="⏸   Pause")
            self._update_button_states("streaming")
            self._pulse_stop()
            self._set_status("Streaming …")
            if self._source is not None:
                # Worker thread is still alive (it continues when _paused=False)
                self.root.after(20, self._ui_poll)
            else:
                self.root.after(200, self._stream_loop)

    def _on_summary(self) -> None:
        """Show in-progress summary while paused."""
        if not self._paused or not self._running:
            return
        self._show_summary(final=False)

    def _on_stop(self) -> None:
        self._running = False
        self._paused  = False
        self._showing_summary = False
        if self._source is not None:
            self._source.stop()
        _, _, lbl = self._btns["pause"]
        lbl.config(text="⏸   Pause")
        self._update_button_states("ready")
        # Show summary if at least one trial was completed
        if self._total > 0:
            self._show_summary(final=True)
        else:
            self._set_status("Stopped.")

    # ══════════════════════════════════════════════════════════════════
    # Training Thread
    # ══════════════════════════════════════════════════════════════════

    def _compute_time_points(self) -> list[float]:
        """Return sub-window checkpoints in seconds, always including full duration."""
        duration = self.config.t_max - self.config.t_min
        step = self.config.prog_step
        points: list[float] = []
        t = step
        while t < duration - 1e-9:
            points.append(round(t, 3))
            t += step
        points.append(round(duration, 3))
        return points

    def _train_worker(self) -> None:
        try:
            self.data_engine = DataEngine(self.config)
            X_train, y_train, ea_matrix = self.data_engine.get_train_data()

            duration = self.config.t_max - self.config.t_min
            sfreq_train = X_train.shape[2] / duration
            self.model = BCIModel(self.config)
            self.model.set_ea_matrix(ea_matrix)

            if (
                self.config.progressive
                and self.config.pipeline_type in {"CSP", "FBCSP"}
            ):
                min_samples = int(round(self.config.prog_step * sfreq_train))
                if min_samples < self.config.csp_components:
                    raise ValueError(
                        f"Progressive step {self.config.prog_step}s gives "
                        f"{min_samples} samples at {sfreq_train:.0f} Hz, "
                        f"but CSP needs at least {self.config.csp_components} "
                        f"samples. Increase step or reduce CSP components."
                    )

            if self.config.progressive:
                time_points = self._compute_time_points()
                full_ns = X_train.shape[2]
                sample_points = sorted({
                    max(1, min(full_ns, int(round(t * sfreq_train))))
                    for t in time_points
                })
                if sample_points[-1] != full_ns:
                    sample_points.append(full_ns)
                self.model.build(sample_points=sample_points)
                self._prog_sample_points = sample_points
                self._prog_time_labels = [
                    round(ns / sfreq_train, 3) for ns in sample_points
                ]
            else:
                self.model.build()
                self._prog_sample_points = []
                self._prog_time_labels = []

            train_acc = self.model.train(X_train, y_train)

            # ── Set up data source ──
            use_legacy = (self.config.pipeline_type == "FBCSP")
            if use_legacy:
                X_test, y_test, sfreq = self.data_engine.get_test_data(
                    apply_ea=True)
                self._sfreq = sfreq
                self.simulator = StreamingSimulator(X_test, y_test)
                self._source = None
            else:
                X_test, y_test, sfreq = self.data_engine.get_test_data(
                    apply_ea=False)
                self._sfreq = sfreq
                self.simulator = None
                source = ReplaySource(
                    X_test, y_test, sfreq,
                    gap_s=self.config.replay_gap_s)
                self._source = source
                self._trial_onsets = source.get_trial_onsets()
                self._trial_labels = source.get_trial_labels()
                self._n_total_trials = source.get_n_trials()
                buf_cap = int(self.config.buffer_capacity_s * sfreq)
                self._buffer = RingBuffer(source.get_n_channels(), buf_cap)

            n_trials = len(X_test)
            self.root.after(0, lambda: self._on_train_done(train_acc, n_trials))
        except Exception as exc:
            err_msg = str(exc)
            self.root.after(0, lambda: self._on_train_error(err_msg))

    def _on_train_done(self, train_acc: float, n_test_trials: int) -> None:
        if self.config.evaluation_protocol == "Cross-Session":
            subj = self.config.cross_session_subject
            sess_str = ", ".join(str(s) for s in self.config.train_sessions)
            self.lbl_train_acc.config(
                text=f"Pre-train Accuracy:  {train_acc:.1%}   "
                     f"[S{subj}, sessions {sess_str}]"
            )
            self._set_status(
                f"Model ready.  S{subj} session {self.config.test_session}: "
                f"{n_test_trials} trials queued."
            )
        else:
            subj_str = ", ".join(f"S{s}" for s in self.config.train_subjects)
            self.lbl_train_acc.config(
                text=f"Pre-train Accuracy:  {train_acc:.1%}   "
                     f"[trained on: {subj_str}]"
            )
            self._set_status(
                f"Model ready.  Test subject {self.config.test_subject}: "
                f"{n_test_trials} trials queued."
            )
        self._welcome.dismiss()
        self._update_button_states("ready")

    def _on_train_error(self, msg: str) -> None:
        messagebox.showerror("Training Error", msg)
        self._update_button_states("idle")
        self._set_status("Training failed. See error dialog.")

    # ══════════════════════════════════════════════════════════════════
    # Online Architecture — background worker + UI poll
    # ══════════════════════════════════════════════════════════════════

    _ACQ_SLEEP_S = 0.005   # ~200 Hz polling in worker thread
    _UI_POLL_MS  = 20      # Tk main thread poll interval

    def _acquisition_worker(self) -> None:
        """
        Background thread: read source → write buffer → inference.
        Produces ProgressEvent and TrialResult into queues.
        Sets _stream_done when all trials are finished.
        Must NOT touch any Tk widget.
        """
        sfreq = self._sfreq
        epoch_samples = int(round(
            (self.config.t_max - self.config.t_min) * sfreq))
        n_trials = self._n_total_trials

        current_trial = 0
        last_prog_idx = 0
        correct = 0
        total = 0
        conf_matrix = [[0, 0], [0, 0]]

        while self._running:
            if self._paused:
                time.sleep(0.02)
                continue

            # 1. Read from source → write to buffer
            chunk = self._source.read_chunk()
            if chunk is not None:
                self._buffer.write(chunk)

            # 2. All trials done?
            if current_trial >= n_trials:
                if self._source.is_exhausted():
                    self._stream_done.set()
                    return
                time.sleep(self._ACQ_SLEEP_S)
                continue

            # 3. Check current trial
            onset = self._trial_onsets[current_trial]
            samples_available = self._buffer.write_pos - onset

            if samples_available <= 0:
                time.sleep(self._ACQ_SLEEP_S)
                continue

            # 4. Progressive predictions
            if self.config.progressive and self._prog_sample_points:
                n_prog = len(self._prog_sample_points) - 1
                while (last_prog_idx < n_prog
                       and samples_available
                       >= self._prog_sample_points[last_prog_idx]):
                    ns = self._prog_sample_points[last_prog_idx]
                    epoch_slice = self._buffer.read(onset, ns)
                    if epoch_slice is not None:
                        epoch_slice = self.model.apply_ea(epoch_slice)
                        pred, proba = self.model.predict_at(
                            epoch_slice, ns)
                        self._progress_queue.put(ProgressEvent(
                            trial_idx=current_trial,
                            n_samples=ns,
                            pred_label=pred,
                            proba=proba.copy(),
                            epoch_slice=epoch_slice.copy(),
                        ))
                        # Track progressive accuracy
                        true_lbl = self._trial_labels[current_trial]
                        if ns in self._prog_accuracy:
                            self._prog_accuracy[ns][1] += 1
                            if (true_lbl is not None
                                    and pred == true_lbl):
                                self._prog_accuracy[ns][0] += 1
                    last_prog_idx += 1

            # 5. Full epoch arrived → final prediction
            if samples_available >= epoch_samples:
                full_epoch = self._buffer.read(onset, epoch_samples)
                if full_epoch is not None:
                    full_epoch = self.model.apply_ea(full_epoch)
                    try:
                        pred = self.model.predict(full_epoch)
                        proba = self.model.predict_proba_single(
                            full_epoch)
                    except Exception:
                        pred = -1
                        proba = np.array([0.5, 0.5])

                    true_label = self._trial_labels[current_trial]
                    total += 1
                    if (true_label is not None
                            and pred == true_label):
                        correct += 1
                    if (true_label is not None
                            and 0 <= true_label <= 1
                            and 0 <= pred <= 1):
                        conf_matrix[true_label][pred] += 1

                    self._result_queue.put(TrialResult(
                        trial_idx=current_trial,
                        true_label=true_label,
                        pred_label=pred,
                        proba=proba.copy(),
                        epoch=full_epoch.copy(),
                        correct_so_far=correct,
                        total_so_far=total,
                        conf_matrix_snapshot=[
                            row[:] for row in conf_matrix],
                    ))
                    current_trial += 1
                    last_prog_idx = 0

            time.sleep(self._ACQ_SLEEP_S)

    # ------------------------------------------------------------------

    def _ui_poll(self) -> None:
        """
        Tk main thread: drain both queues, update UI.
        Called every _UI_POLL_MS via root.after.
        """
        if not self._running:
            return

        # ── Check completion ──
        if self._stream_done.is_set():
            # Drain remaining results into metrics
            while not self._result_queue.empty():
                try:
                    r = self._result_queue.get_nowait()
                    self._apply_result_metrics(r)
                    if self.config.presentation_mode == "demo":
                        self._demo_result_buffer.append(r)
                except queue.Empty:
                    break
            # Demo: let backlog finish playing before summary
            if (self.config.presentation_mode == "demo"
                    and (self._displaying or self._demo_result_buffer)):
                # Fall through to demo display logic below
                pass
            else:
                self._on_stop()
                self._show_summary()
                return

        # ── Drain progress queue → tentative UI ──
        latest_progress = None  # type: Optional[ProgressEvent]
        while not self._progress_queue.empty():
            try:
                latest_progress = self._progress_queue.get_nowait()
            except queue.Empty:
                break
        if (latest_progress is not None
                and not self._displaying
                and not self._showing_summary):
            self._display_progress(latest_progress)

        # ── Drain result queue → metrics + display ──
        if self.config.presentation_mode == "live":
            results = []  # type: List[TrialResult]
            while not self._result_queue.empty():
                try:
                    results.append(self._result_queue.get_nowait())
                except queue.Empty:
                    break
            if results:
                for r in results:
                    self._apply_result_metrics(r)
                self._display_final_result(results[-1])
        else:
            # Demo: drain into persistent backlog
            while not self._result_queue.empty():
                try:
                    r = self._result_queue.get_nowait()
                    self._apply_result_metrics(r)
                    self._demo_result_buffer.append(r)
                except queue.Empty:
                    break
            # Display next from backlog if UI is idle
            if not self._displaying and self._demo_result_buffer:
                r = self._demo_result_buffer.popleft()
                self._display_final_result(r)
            # Check again: if stream done and backlog empty and not displaying
            if (self._stream_done.is_set()
                    and not self._displaying
                    and not self._demo_result_buffer):
                self._on_stop()
                self._show_summary()
                return

        self.root.after(self._UI_POLL_MS, self._ui_poll)

    # ------------------------------------------------------------------

    def _apply_result_metrics(self, r: TrialResult) -> None:
        """Update cumulative counters from a TrialResult. No UI animation."""
        self._trial_idx = r.trial_idx + 1
        self._correct = r.correct_so_far
        self._total = r.total_so_far
        self._conf_matrix = r.conf_matrix_snapshot
        if r.true_label is not None:
            self._history.append(r.pred_label == r.true_label)

    def _display_progress(self, evt: ProgressEvent) -> None:
        """Update tentative prediction UI from a ProgressEvent."""
        pred_name = StreamingSimulator.label_name(evt.pred_label)
        t_label = evt.n_samples / self._sfreq
        total_s = self.config.t_max - self.config.t_min

        self.lbl_trial.config(
            text=f"Trial:  {evt.trial_idx + 1}  /  {self._n_total_trials}")
        self.lbl_prediction.config(
            text=f"Prediction:  {pred_name}  (tentative, {t_label:.1f}s)",
            fg="#6e7781")
        self.lbl_status.config(
            text=f"Status:  Predicting  ({t_label:.1f}s / {total_s:.1f}s)",
            fg=self.YELLOW)
        pct = min(100, int(t_label / total_s * 100))
        self.progress_bar["value"] = pct
        self.lbl_countdown.config(
            text=f"EEG Window:  {t_label:.1f} s  /  {total_s:.1f} s")
        self._animate_confidence(evt.proba)
        self._animate_band_power(evt.epoch_slice)

    def _display_final_result(self, r: TrialResult) -> None:
        """Animate the final prediction + reveal actual for one trial."""
        self._displaying = True
        pred_name = StreamingSimulator.label_name(r.pred_label)
        true_name = (StreamingSimulator.label_name(r.true_label)
                     if r.true_label is not None else "\u2014")
        is_correct = ((r.pred_label == r.true_label)
                      if r.true_label is not None else None)

        self.lbl_trial.config(
            text=f"Trial:  {r.trial_idx + 1}  /  {self._n_total_trials}",
            fg=self.ACCENT, font=("Helvetica Neue", 14, "bold"))
        self.root.after(250, lambda: self.lbl_trial.config(
            fg="#888", font=("Helvetica Neue", 12)))
        self.lbl_status.config(
            text="Status:  Prediction Ready", fg=self.GREEN)
        self.lbl_prediction.config(
            text=f"Prediction:  {pred_name}", fg=self.ACCENT,
            font=("Helvetica Neue", 26))
        self.progress_bar["value"] = 100
        self.lbl_countdown.config(
            text=f"EEG Window:  {self.config.t_max - self.config.t_min:.1f} s"
                 f"  /  {self.config.t_max - self.config.t_min:.1f} s")
        self._animate_confidence(r.proba)
        self._animate_band_power(r.epoch)

        acc = self._correct / self._total if self._total else 0.0
        self.lbl_accuracy.config(
            text=f"Online Accuracy:  {acc:.1%}"
                 f"   ({self._correct} correct  /  {self._total} trials)")

        def reveal_actual():
            if is_correct is not None:
                flash_color = self.GREEN if is_correct else self.RED
                self.lbl_prediction.config(fg="white", bg=flash_color)
                self.lbl_actual.config(
                    text=f"Actual:  {true_name}", fg=flash_color)
                self.root.after(350, lambda: self.lbl_prediction.config(
                    fg=flash_color, bg=self.BG_COLOR))
            else:
                self.lbl_actual.config(
                    text="Actual:  \u2014 (no ground truth)")
            draw_trial_chart(self._chart_canvas, self._history,
                             self._CHART_MAX_BARS,
                             self.GREEN, self.RED, self.ACCENT, self.FG_COLOR)
            draw_confusion_matrix(self._cm_canvas, self._conf_matrix,
                                  self.FG_COLOR, self.GREEN, self.RED)

        def mark_done():
            self._displaying = False

        if self.config.presentation_mode == "demo":
            self.root.after(self.ACTUAL_DELAY_MS, reveal_actual)
            self.root.after(self.DISPLAY_INTERVAL_MS, mark_done)
        else:
            reveal_actual()
            mark_done()

    # ══════════════════════════════════════════════════════════════════
    # Legacy Streaming Loop (FBCSP fallback, root.after driven)
    # ══════════════════════════════════════════════════════════════════

    _COUNTDOWN_TICK_MS = 50

    def _stream_loop(self) -> None:
        if not self._running or self._paused:
            return

        trial = self.simulator.next_trial()
        if trial is None:
            self._on_stop()
            self._show_summary()
            return

        epoch, true_label = trial
        self._trial_idx += 1
        self._total     += 1

        self.lbl_trial.config(
            text=f"Trial:  {self._trial_idx}  /  {len(self.simulator.X)}",
            fg=self.ACCENT, font=("Helvetica Neue", 14, "bold"),
        )
        self.root.after(250, lambda: self.lbl_trial.config(
            fg="#888", font=("Helvetica Neue", 12),
        ))
        self.lbl_status.config(text="Status:  Collecting EEG …", fg=self.YELLOW)
        self.lbl_prediction.config(text="Prediction:  —", fg=self.ACCENT)
        self.lbl_actual.config(text="Actual:  —", fg="#8c959f")
        self.progress_bar["value"] = 0

        # Clear all visualisation canvases so previous-trial results are not
        # visible during the current trial's EEG collection window.
        self._conf_canvas.delete("all")
        self._bp_canvas.delete("all")

        total_ms = int((self.config.t_max - self.config.t_min) * 1000)
        self._run_countdown(
            epoch, true_label, elapsed_ms=0, total_ms=total_ms, next_prog_idx=0
        )

    def _run_countdown(
        self,
        epoch: np.ndarray,
        true_label: int,
        elapsed_ms: int,
        total_ms: int,
        next_prog_idx: int = 0,
    ) -> None:
        if not self._running or self._paused:
            return

        remaining_s = max(0.0, (total_ms - elapsed_ms) / 1000.0)
        pct         = min(100, int(elapsed_ms / total_ms * 100))

        self.lbl_countdown.config(
            text=f"EEG Window:  {elapsed_ms / 1000:.1f} s  /  {total_ms / 1000:.1f} s"
                 f"   ({remaining_s:.1f} s left)"
        )
        self.progress_bar["value"] = pct

        if self.config.progressive and self._prog_sample_points:
            n_prog = len(self._prog_sample_points) - 1
            while (
                next_prog_idx < n_prog
                and elapsed_ms >= self._prog_time_labels[next_prog_idx] * 1000 - 1e-3
            ):
                ns = self._prog_sample_points[next_prog_idx]
                pred_label, proba = self.model.predict_at(epoch, ns)
                # Track progressive accuracy
                if ns in self._prog_accuracy:
                    self._prog_accuracy[ns][1] += 1
                    if pred_label == true_label:
                        self._prog_accuracy[ns][0] += 1
                self._update_progressive_ui(
                    epoch, pred_label, proba, ns, self._prog_time_labels[next_prog_idx]
                )
                next_prog_idx += 1

        if elapsed_ms >= total_ms:
            self.lbl_status.config(text="Status:  Processing …", fg=self.YELLOW)
            self.root.after(80, lambda: self._do_predict(epoch, true_label))
        else:
            self.root.after(
                self._COUNTDOWN_TICK_MS,
                lambda idx=next_prog_idx: self._run_countdown(
                    epoch, true_label,
                    elapsed_ms + self._COUNTDOWN_TICK_MS,
                    total_ms,
                    idx,
                ),
            )

    def _update_progressive_ui(
        self,
        epoch: np.ndarray,
        pred_label: int,
        proba: np.ndarray,
        n_samples: int,
        t_label: float,
    ) -> None:
        """Refresh tentative prediction widgets without updating final metrics."""
        pred_name = StreamingSimulator.label_name(pred_label)
        total_s = self.config.t_max - self.config.t_min

        self.lbl_prediction.config(
            text=f"Prediction:  {pred_name}  (tentative, {t_label:.1f}s)",
            fg="#6e7781",
        )
        self.lbl_status.config(
            text=f"Status:  Predicting  ({t_label:.1f}s / {total_s:.1f}s)",
            fg=self.YELLOW,
        )

        self._animate_confidence(proba)
        if epoch.ndim == 2:
            sub_epoch = epoch[:, :n_samples]
        else:
            sub_epoch = epoch[:, :n_samples, :]
        self._animate_band_power(sub_epoch)

    def _do_predict(self, epoch: np.ndarray, true_label: int) -> None:
        try:
            pred_label = self.model.predict(epoch)
            proba      = self.model.predict_proba_single(epoch)
        except Exception as exc:
            pred_label = -1
            proba      = np.array([0.5, 0.5])
            print(f"  [Predict ERROR] {exc}")

        pred_name  = StreamingSimulator.label_name(pred_label)
        true_name  = StreamingSimulator.label_name(true_label)
        is_correct = (pred_label == true_label)
        if is_correct:
            self._correct += 1

        self.lbl_status.config(text="Status:  Prediction Ready", fg=self.GREEN)
        self.lbl_prediction.config(text=f"Prediction:  {pred_name}", fg=self.ACCENT)
        self.progress_bar["value"] = 100

        self._animate_confidence(proba)
        self._animate_band_power(epoch)

        def reveal_actual():
            flash_color = self.GREEN if is_correct else self.RED
            self.lbl_prediction.config(
                fg="white", bg=flash_color,
            )
            self.lbl_actual.config(
                text=f"Actual:  {true_name}",
                fg=flash_color,
            )
            self.root.after(350, lambda: self.lbl_prediction.config(
                fg=flash_color, bg=self.BG_COLOR,
            ))
            acc = self._correct / self._total if self._total else 0.0
            self.lbl_accuracy.config(
                text=f"Online Accuracy:  {acc:.1%}"
                     f"   ({self._correct} correct  /  {self._total} trials)"
            )
            self._history.append(is_correct)
            if 0 <= true_label <= 1 and 0 <= pred_label <= 1:
                self._conf_matrix[true_label][pred_label] += 1
            draw_trial_chart(self._chart_canvas, self._history,
                             self._CHART_MAX_BARS,
                             self.GREEN, self.RED, self.ACCENT, self.FG_COLOR)
            draw_confusion_matrix(self._cm_canvas, self._conf_matrix,
                                  self.FG_COLOR, self.GREEN, self.RED)

        self.root.after(self.ACTUAL_DELAY_MS, reveal_actual)
        self.root.after(self.DISPLAY_INTERVAL_MS, self._stream_loop)

    # ══════════════════════════════════════════════════════════════════
    # Summary View
    # ══════════════════════════════════════════════════════════════════

    def _generate_conclusion(self, acc: float) -> tuple[str, str]:
        """
        Return (performance_line, progressive_line) for the summary view.
        """
        # ── Performance rating ──
        if acc > 0.80:
            perf = ("Excellent decoding performance — "
                    "strong separability between classes.")
        elif acc > 0.70:
            perf = ("Good performance — "
                    "reliable above chance with room for improvement.")
        elif acc > 0.60:
            perf = ("Moderate performance — "
                    "above chance but noisy; consider tuning parameters.")
        elif acc > 0.50:
            perf = ("Weak performance — "
                    "the model struggles to distinguish LEFT from RIGHT.")
        else:
            perf = ("At or below chance level — "
                    "no meaningful discrimination detected.")

        # ── Progressive trend ──
        prog = ""
        if self._prog_accuracy and self._prog_time_labels:
            sorted_ns = sorted(self._prog_accuracy.keys())
            accs = []
            for ns in sorted_ns:
                c, t = self._prog_accuracy[ns]
                accs.append(c / t if t > 0 else 0.0)

            if len(accs) >= 2 and len(self._prog_time_labels) >= 2:
                t_first = self._prog_time_labels[0]
                t_last = self._prog_time_labels[-1]
                a_first = accs[0]
                a_last = accs[-1]

                # Find the steepest gain segment
                max_gain = 0.0
                gain_start_idx = 0
                for i in range(len(accs) - 1):
                    gain = accs[i + 1] - accs[i]
                    if gain > max_gain:
                        max_gain = gain
                        gain_start_idx = i

                prog = (
                    f"Accuracy improves from {a_first:.0%} at "
                    f"{t_first:.1f}s to {a_last:.0%} at {t_last:.1f}s."
                )
                if max_gain > 0.02 and len(self._prog_time_labels) > gain_start_idx + 1:
                    t_gs = self._prog_time_labels[gain_start_idx]
                    t_ge = self._prog_time_labels[gain_start_idx + 1]
                    prog += (
                        f"  Steepest gain between {t_gs:.1f}–{t_ge:.1f}s "
                        f"(+{max_gain:.0%})."
                    )

        return perf, prog

    def _show_summary(self, final: bool = True) -> None:
        """
        Replace live-feed widgets with a summary view.

        final=True  — demo is over (all trials done or stopped)
        final=False — in-progress preview while paused
        """
        self._showing_summary = True
        acc = self._correct / self._total if self._total else 0.0

        if final:
            self._phase_indicator.set_phase("complete")
            title = "Demo Complete"
            title_fg = self.GREEN
            status_text = (f"Final Accuracy   "
                           f"({self._correct} / {self._total} correct)")
        else:
            title = f"Summary  ({self._trial_idx} / {len(self.simulator.X)} trials)"
            title_fg = self.ACCENT
            status_text = (f"In-progress Accuracy   "
                           f"({self._correct} / {self._total} correct)")

        self.lbl_trial.config(text=title, fg=title_fg,
                              font=("Helvetica Neue", 12, "bold"))
        self.lbl_countdown.config(
            text=f"{acc:.1%}", fg=title_fg,
            font=("Helvetica Neue", 36, "bold"),
        )
        self.progress_bar["value"] = (
            100 if final
            else int(self._trial_idx / max(1, len(self.simulator.X)) * 100)
        )
        self.lbl_status.config(text=status_text, fg=self.FG_COLOR)

        # Dynamic conclusion
        perf_line, prog_line = self._generate_conclusion(acc)
        self.lbl_prediction.config(
            text=perf_line, bg=self.BG_COLOR, fg=self.FG_COLOR,
            font=("Helvetica Neue", 11),
        )
        self.lbl_actual.config(
            text=prog_line, fg="#57606a",
            font=("Helvetica Neue", 10),
        )

        # Confidence canvas → Progressive Accuracy Curve (needs more height)
        self._conf_canvas.config(height=100)
        if self._prog_accuracy and self._prog_time_labels:
            draw_progressive_accuracy(
                self._conf_canvas, self._prog_accuracy,
                self._prog_time_labels,
                self.ACCENT, self.GREEN, self.RED, self.FG_COLOR,
            )
        else:
            self._conf_canvas.delete("all")

        # Band power → cleared
        self._bp_canvas.delete("all")

        # Trial chart → cumulative accuracy curve
        draw_accuracy_curve(
            self._chart_canvas, self._history,
            self.ACCENT, self.GREEN, self.FG_COLOR,
        )

        # Confusion matrix
        draw_confusion_matrix(
            self._cm_canvas, self._conf_matrix,
            self.FG_COLOR, self.GREEN, self.RED,
        )

        self._set_status(
            f"{'Demo complete' if final else 'Paused — summary preview'}!  "
            f"Accuracy: {acc:.1%}  ({self._correct}/{self._total})"
        )

    # ══════════════════════════════════════════════════════════════════
    # Button State Management
    # ══════════════════════════════════════════════════════════════════

    def _update_button_states(self, state: str) -> None:
        ON_TR = (True,  self._BTN_TRAIN_ON, "white",         "#8c959f")
        ON_ST = (True,  self._BTN_START_ON, "white",         "#8c959f")
        ON_PA = (True,  "#7d4e00",          "white",         self.YELLOW)
        ON_SP = (True,  self._BTN_STOP_ON,  "white",         self.RED)
        ON_SU = (True,  self.ACCENT,        "white",         "#8c959f")
        OFF   = (False, self._BTN_OFF,      self._BTN_FG_OFF, "#8c959f")

        table = {
            #          train  start  pause  stop   summary
            "idle"     : [ON_TR, OFF,   OFF,   OFF,   OFF  ],
            "training" : [OFF,   OFF,   OFF,   OFF,   OFF  ],
            "ready"    : [ON_TR, ON_ST, OFF,   OFF,   OFF  ],
            "streaming": [OFF,   OFF,   ON_PA, ON_SP, OFF  ],
            "paused"   : [OFF,   ON_ST, ON_PA, ON_SP, ON_SU],
        }

        for (name, btn_widgets), cfg in zip(self._btns.items(), table[state]):
            enabled, bg, fg, dot_color = cfg
            self._btn_enabled[name] = enabled
            f, dot, lbl = btn_widgets
            f.config(bg=bg, cursor="hand2" if enabled else "")
            dot.config(bg=bg, fg=dot_color)
            lbl.config(bg=bg, fg=fg)

        if hasattr(self, "_phase_indicator"):
            self._phase_indicator.set_phase(state)

    def _pulse_stop(self, bright: bool = True) -> None:
        if not self._running or self._paused:
            return
        f, dot, lbl = self._btns["stop"]
        color = self._BTN_STOP_PULSE if bright else self._BTN_STOP_ON
        f.config(bg=color)
        dot.config(bg=color)
        lbl.config(bg=color)
        self.root.after(550, lambda: self._pulse_stop(not bright))

    # ══════════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════════

    _CHART_MAX_BARS = 60
    _TWEEN_MS = 120

    def _animate_confidence(self, proba: np.ndarray) -> None:
        """Smoothly tween the confidence bar from its last value to proba."""
        old = self._last_proba.copy()
        self._last_proba = proba.copy()
        self._tween.animate(
            key="confidence",
            start=old, end=proba,
            duration_ms=self._TWEEN_MS,
            on_frame=lambda v: draw_confidence(
                self._conf_canvas, v, self.ACCENT, self.FG_COLOR),
        )

    def _animate_band_power(self, epoch: np.ndarray) -> None:
        """Smoothly tween band power bars from last value to current."""
        from .plots import band_power
        if epoch.ndim == 3:
            draw_band_power(self._bp_canvas, epoch, self._sfreq,
                            self.ACCENT, self.GREEN, self.FG_COLOR)
            return
        new_bp = np.array([
            band_power(epoch, self._sfreq, 8, 12),
            band_power(epoch, self._sfreq, 13, 30),
        ])
        old = self._last_bp.copy()
        self._last_bp = new_bp.copy()
        self._tween.animate(
            key="band_power",
            start=old, end=new_bp,
            duration_ms=self._TWEEN_MS,
            on_frame=lambda v: self._draw_bp_from_values(v),
        )

    def _draw_bp_from_values(self, values: np.ndarray) -> None:
        """Redraw band power bars from pre-computed [mu_frac, beta_frac]."""
        import tkinter as tk
        c = self._bp_canvas
        c.delete("all")
        c.update_idletasks()
        W, H = c.winfo_width(), c.winfo_height()
        if W < 10 or H < 10:
            return
        mu_frac, beta_frac = float(values[0]), float(values[1])
        bands = [
            ("\u03bc  8\u201312 Hz",  mu_frac,   self.ACCENT),
            ("\u03b2  13\u201330 Hz", beta_frac, self.GREEN),
        ]
        label_h = 16
        plot_h = H - label_h - 8
        n = len(bands)
        gap = 10
        bar_w = (W - (n + 1) * gap) // n
        for i, (label, frac, color) in enumerate(bands):
            x0 = gap + i * (bar_w + gap)
            x1 = x0 + bar_w
            c.create_rectangle(x0, 4, x1, H - label_h - 4,
                               fill="#d0d7de", outline="")
            fill_h = max(2, int(frac * plot_h))
            c.create_rectangle(x0, H - label_h - 4 - fill_h, x1,
                               H - label_h - 4, fill=color, outline="")
            c.create_text((x0 + x1) // 2, H - label_h // 2 - 2,
                          text=f"{label}  {frac:.0%}",
                          fill=self.FG_COLOR, font=("Helvetica Neue", 7),
                          anchor="center")

    def _set_status(self, msg: str) -> None:
        self.status_bar.config(text=f"  {msg}")
        print(f"[STATUS] {msg}")
