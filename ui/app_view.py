"""
ui/app_view.py — tkinter application window.

Threading model
───────────────
Heavy work (data loading + training) runs in a daemon thread and posts
results back via root.after(0, callback).
Streaming is driven entirely by root.after() — no time.sleep(), no threads.
"""

import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import numpy as np

from config import BCIConfig
from data_engine import DataEngine
from model import BCIModel
from streaming import StreamingSimulator
from .plots import (
    draw_band_power,
    draw_confidence,
    draw_confusion_matrix,
    draw_trial_chart,
)


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
        self.root.title("BCI Streaming Demo — Motor Imagery")
        self.root.configure(bg=self.BG_COLOR)
        self.root.resizable(True, True)

        self.config    = BCIConfig()
        self.data_engine: Optional[DataEngine]         = None
        self.model:       Optional[BCIModel]           = None
        self.simulator:   Optional[StreamingSimulator] = None

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

        self._build_ui()

    # ══════════════════════════════════════════════════════════════════
    # UI Construction
    # ══════════════════════════════════════════════════════════════════

    def _build_ui(self) -> None:
        tk.Label(
            self.root,
            text="BCI  Real-Time Motor Imagery Demo",
            font=("Helvetica Neue", 20, "bold"),
            bg=self.BG_COLOR, fg=self.FG_COLOR,
        ).pack(pady=(18, 2))

        tk.Label(
            self.root,
            text="Zhou2016  ·  Cross-Subject Classification  ·  CSP + LDA / SVM",
            font=("Helvetica Neue", 10),
            bg=self.BG_COLOR, fg="#57606a",
        ).pack(pady=(0, 10))

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

        _make_btn("train", "⚙   Train & Load", self._on_train)
        _make_btn("start", "▶   Start Stream",  self._on_start)
        _make_btn("pause", "⏸   Pause",          self._on_pause)
        _make_btn("stop",  "⏹   Stop",           self._on_stop)
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

        def section(title, color="#0969da"):
            frm = tk.LabelFrame(
                inner, text=f" {title} ",
                font=("Helvetica Neue", 9, "bold"),
                bg=self.BG_COLOR, fg=color,
                relief=tk.GROOVE, bd=1, padx=4, pady=3,
            )
            frm.pack(fill=tk.X, padx=8, pady=(6, 0))
            return frm

        def hint(parent_frame, text):
            tk.Label(parent_frame, text=text,
                     font=("Helvetica Neue", 8), bg=self.BG_COLOR, fg="#57606a",
                     wraplength=230, justify="left",
                     ).pack(anchor="w", padx=2, pady=(0, 2))

        # Bandpass Filter
        bp = section("Bandpass Filter")
        hint(bp, "Retain mu (8-12 Hz) and beta (13-30 Hz) motor rhythms, reject noise")
        add_slider(bp, "Low:",  "f_low",  1.0, 30.0, self.config.f_low)
        add_slider(bp, "High:", "f_high", 10.0, 60.0, self.config.f_high)

        # Epoch Window
        ep = section("Epoch Window")
        hint(ep, "EEG segment per trial, relative to stimulus onset (seconds)")
        add_slider(ep, "Start:", "t_min", -1.0, 2.0, self.config.t_min)
        add_slider(ep, "End:",   "t_max",  0.5, 8.0, self.config.t_max)

        # Spatial Filters — shown for CSP and FBCSP, hidden for TS
        self._sp_frame = section("Spatial Filters")
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

        # Protocol description label
        self._protocol_hint_lbl = tk.Label(
            tr, text="", font=("Helvetica Neue", 8),
            bg=self.BG_COLOR, fg="#57606a", wraplength=230, justify="left",
        )
        self._protocol_hint_lbl.pack(anchor="w", padx=8, pady=(0, 4))

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
                self._protocol_hint_lbl.config(
                    text="Train on data from multiple subjects. "
                         "Test on a different held-out subject.")
            else:  # Cross-Session
                self._cross_subject_rows.pack_forget()
                self._cross_session_rows.pack(fill=tk.X)
                self._protocol_hint_lbl.config(
                    text="Train on earlier sessions of one subject. "
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

        # Hint label that updates when feature extraction changes
        self._feat_hint_lbl = tk.Label(tr, text="",
                                       font=("Helvetica Neue", 8),
                                       bg=self.BG_COLOR, fg="#57606a",
                                       wraplength=230, justify="left")
        self._feat_hint_lbl.pack(anchor="w", padx=8, pady=(0, 2))

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
                self._sp_frame.pack(fill=tk.X, padx=8, pady=(6, 0), before=tr)
                self._bands_row.pack_forget()
                self._clf_cb.config(values=["LDA", "SVM"])
                if self._clf_var.get() == "MDM":
                    self._clf_var.set("LDA")
                self._feat_hint_lbl.config(
                    text="Single bandpass → CSP spatial filters → LDA/SVM. "
                         "Classic MOABB baseline (Jayaram & Barachant 2018).")
            elif feat == "FBCSP":
                self._sp_frame.pack(fill=tk.X, padx=8, pady=(6, 0), before=tr)
                self._bands_row.pack(fill=tk.X, padx=6, pady=2)
                self._clf_cb.config(values=["LDA", "SVM"])
                if self._clf_var.get() == "MDM":
                    self._clf_var.set("LDA")
                self._feat_hint_lbl.config(
                    text="FilterBankLeftRightImagery (moabb): CSP per band, "
                         "features concatenated → LDA/SVM. Ang et al. 2012.")
            else:  # TS (Riemannian)
                self._sp_frame.pack_forget()
                self._clf_cb.config(values=["LDA", "SVM", "MDM"])
                self._feat_hint_lbl.config(
                    text="pyriemann: covariances → Tangent Space (LDA/SVM) "
                         "or Minimum Distance to Mean (MDM). "
                         "Barachant et al. 2012/2013.")
            self.root.after(50, lambda: inner.event_generate("<Configure>"))

        feat_cb.bind("<<ComboboxSelected>>", _on_feat_change)
        _on_feat_change()   # set initial state

        tk.Frame(inner, height=8, bg=self.BG_COLOR).pack()   # bottom padding

    # ------------------------------------------------------------------
    def _build_display_panel(self, parent: tk.Frame) -> None:
        panel = tk.LabelFrame(
            parent,
            text=" Live Feed ",
            font=("Helvetica Neue", 11, "bold"),
            bg=self.BG_COLOR, fg=self.ACCENT,
            relief=tk.RIDGE, bd=2,
        )
        panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=4)

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

        # Trial history chart
        tk.Frame(panel, height=2, bg="#333").pack(fill=tk.X, padx=20, pady=(8, 4))
        tk.Label(panel, text="Trial-by-Trial History",
                 font=("Helvetica Neue", 9), bg=self.BG_COLOR, fg="#57606a").pack()
        self._chart_canvas = tk.Canvas(panel, height=100, bg="#eaeef2",
                                       highlightthickness=0)
        self._chart_canvas.pack(fill=tk.X, padx=20, pady=(2, 4))

        # Band Power + Confusion Matrix (side by side)
        vis_row = tk.Frame(panel, bg=self.BG_COLOR)
        vis_row.pack(fill=tk.X, padx=20, pady=(0, 8))

        bp_col = tk.Frame(vis_row, bg=self.BG_COLOR)
        bp_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(bp_col, text="Band Power  (μ / β)",
                 font=("Helvetica Neue", 9), bg=self.BG_COLOR, fg="#57606a").pack()
        self._bp_canvas = tk.Canvas(bp_col, height=80, bg="#eaeef2",
                                    highlightthickness=0)
        self._bp_canvas.pack(fill=tk.X, pady=(2, 0))

        tk.Frame(vis_row, width=8, bg=self.BG_COLOR).pack(side=tk.LEFT)

        cm_col = tk.Frame(vis_row, bg=self.BG_COLOR)
        cm_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(cm_col, text="Confusion Matrix",
                 font=("Helvetica Neue", 9), bg=self.BG_COLOR, fg="#57606a").pack()
        self._cm_canvas = tk.Canvas(cm_col, height=80, bg="#eaeef2",
                                    highlightthickness=0)
        self._cm_canvas.pack(fill=tk.X, pady=(2, 0))

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
        if self.simulator is None:
            return
        self._running = True
        self._trial_idx = 0
        self._correct   = 0
        self._total     = 0
        self._history   = []
        self._conf_matrix = [[0, 0], [0, 0]]
        self.simulator.reset()
        for c in (self._chart_canvas, self._conf_canvas,
                  self._bp_canvas, self._cm_canvas):
            c.delete("all")
        self._update_button_states("streaming")
        self._pulse_stop()
        self._set_status("Streaming …")
        self.root.after(200, self._stream_loop)

    def _on_pause(self) -> None:
        if not self._running:
            return
        self._paused = not self._paused
        _, _, lbl = self._btns["pause"]
        if self._paused:
            lbl.config(text="▶   Resume")
            self._update_button_states("paused")
            self._set_status("Paused.")
        else:
            lbl.config(text="⏸   Pause")
            self._update_button_states("streaming")
            self._pulse_stop()
            self._set_status("Streaming …")
            self.root.after(200, self._stream_loop)

    def _on_stop(self) -> None:
        self._running = False
        self._paused  = False
        _, _, lbl = self._btns["pause"]
        lbl.config(text="⏸   Pause")
        self._update_button_states("ready")
        self._set_status("Stopped.")

    # ══════════════════════════════════════════════════════════════════
    # Training Thread
    # ══════════════════════════════════════════════════════════════════

    def _train_worker(self) -> None:
        try:
            self.data_engine = DataEngine(self.config)
            X_train, y_train = self.data_engine.get_train_data()

            self.model = BCIModel(self.config)
            self.model.build()
            train_acc = self.model.train(X_train, y_train)

            X_test, y_test, sfreq = self.data_engine.get_test_data()
            self._sfreq    = sfreq
            self.simulator = StreamingSimulator(X_test, y_test)

            self.root.after(0, lambda: self._on_train_done(train_acc, len(X_test)))
        except Exception as exc:
            err_msg = str(exc)
            self.root.after(0, lambda: self._on_train_error(err_msg))

    def _on_train_done(self, train_acc: float, n_test_trials: int) -> None:
        subj_str = ", ".join(f"S{s}" for s in self.config.train_subjects)
        self.lbl_train_acc.config(
            text=f"Pre-train Accuracy:  {train_acc:.1%}   [trained on: {subj_str}]"
        )
        self._set_status(
            f"Model ready.  Test subject {self.config.test_subject}: "
            f"{n_test_trials} trials queued."
        )
        self._update_button_states("ready")

    def _on_train_error(self, msg: str) -> None:
        messagebox.showerror("Training Error", msg)
        self._update_button_states("idle")
        self._set_status("Training failed. See error dialog.")

    # ══════════════════════════════════════════════════════════════════
    # Streaming Loop (root.after driven)
    # ══════════════════════════════════════════════════════════════════

    _COUNTDOWN_TICK_MS = 50

    def _stream_loop(self) -> None:
        if not self._running or self._paused:
            return

        trial = self.simulator.next_trial()
        if trial is None:
            self._on_stop()
            acc = self._correct / self._total if self._total else 0.0
            self._set_status(
                f"Demo complete!  Final accuracy: {acc:.1%}  "
                f"({self._correct}/{self._total})"
            )
            return

        epoch, true_label = trial
        self._trial_idx += 1
        self._total     += 1

        self.lbl_trial.config(
            text=f"Trial:  {self._trial_idx}  /  {len(self.simulator.X)}"
        )
        self.lbl_status.config(text="Status:  Collecting EEG …", fg=self.YELLOW)
        self.lbl_prediction.config(text="Prediction:  —", fg=self.ACCENT)
        self.lbl_actual.config(text="Actual:  —", fg="#8c959f")
        self.progress_bar["value"] = 0

        # Clear all visualisation canvases so previous-trial results are not
        # visible during the current trial's EEG collection window.
        self._conf_canvas.delete("all")
        self._bp_canvas.delete("all")

        total_ms = int((self.config.t_max - self.config.t_min) * 1000)
        self._run_countdown(epoch, true_label, elapsed_ms=0, total_ms=total_ms)

    def _run_countdown(
        self, epoch: np.ndarray, true_label: int, elapsed_ms: int, total_ms: int
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

        if elapsed_ms >= total_ms:
            self.lbl_status.config(text="Status:  Processing …", fg=self.YELLOW)
            self.root.after(80, lambda: self._do_predict(epoch, true_label))
        else:
            self.root.after(
                self._COUNTDOWN_TICK_MS,
                lambda: self._run_countdown(
                    epoch, true_label,
                    elapsed_ms + self._COUNTDOWN_TICK_MS,
                    total_ms,
                ),
            )

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

        draw_confidence(self._conf_canvas, proba, self.ACCENT, self.FG_COLOR)
        draw_band_power(self._bp_canvas, epoch, self._sfreq,
                        self.ACCENT, self.GREEN, self.FG_COLOR)

        def reveal_actual():
            self.lbl_prediction.config(fg=self.GREEN if is_correct else self.RED)
            self.lbl_actual.config(
                text=f"Actual:  {true_name}",
                fg=self.GREEN if is_correct else self.RED,
            )
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
    # Button State Management
    # ══════════════════════════════════════════════════════════════════

    def _update_button_states(self, state: str) -> None:
        ON_TR = (True,  self._BTN_TRAIN_ON, "white",         "#8c959f")
        ON_ST = (True,  self._BTN_START_ON, "white",         "#8c959f")
        ON_PA = (True,  "#7d4e00",          "white",         self.YELLOW)
        ON_SP = (True,  self._BTN_STOP_ON,  "white",         self.RED)
        OFF   = (False, self._BTN_OFF,      self._BTN_FG_OFF, "#8c959f")

        table = {
            "idle"     : [ON_TR, OFF,   OFF,   OFF  ],
            "training" : [OFF,   OFF,   OFF,   OFF  ],
            "ready"    : [ON_TR, ON_ST, OFF,   OFF  ],
            "streaming": [OFF,   OFF,   ON_PA, ON_SP],
            "paused"   : [OFF,   ON_ST, ON_PA, ON_SP],
        }

        for (name, btn_widgets), cfg in zip(self._btns.items(), table[state]):
            enabled, bg, fg, dot_color = cfg
            self._btn_enabled[name] = enabled
            f, dot, lbl = btn_widgets
            f.config(bg=bg, cursor="hand2" if enabled else "")
            dot.config(bg=bg, fg=dot_color)
            lbl.config(bg=bg, fg=fg)

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

    def _set_status(self, msg: str) -> None:
        self.status_bar.config(text=f"  {msg}")
        print(f"[STATUS] {msg}")
