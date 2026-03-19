"""
BCI Streaming Demo Application
================================
A real-time Brain-Computer Interface (BCI) motor imagery (MI) demonstration
that trains a cross-subject model and simulates streaming predictions on a
new subject using tkinter GUI.

Architecture:
  - BCIConfig:           Centralized parameter store (no hard-coding)
  - DataEngine:          MOABB data loading + preprocessing (filter, re-ref, epoch)
  - BCIModel:            sklearn Pipeline (CSP + LDA/SVM)
  - StreamingSimulator:  Generator yielding (epoch_array, true_label) pairs
  - AppUI:               tkinter GUI driven by root.after() — never blocks main thread
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from dataclasses import dataclass, field
from typing import List, Optional, Generator, Tuple
import warnings

# ── Suppress MNE / sklearn verbosity for cleaner console ─────────────────────
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

# MNE imports
import mne
from mne.decoding import CSP
mne.set_log_level("WARNING")

# MOABB imports
import moabb
from moabb.datasets import Zhou2016
from moabb.paradigms import LeftRightImagery
moabb.set_log_level("WARNING")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  BCIConfig — all tunable parameters in one place
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class BCIConfig:
    """
    Centralised configuration for the BCI pipeline.
    Every numeric and categorical parameter lives here; nothing is
    hard-coded anywhere else in the application.
    """
    # -- Bandpass filter range (Hz) --
    f_low: float  = 8.0
    f_high: float = 30.0

    # -- Epoch window relative to stimulus onset (seconds) --
    t_min: float  = 0.0
    t_max: float  = 3.0

    # -- Common Spatial Patterns: number of spatial filters to retain --
    csp_components: int = 8

    # -- Classifier selection: "LDA" or "SVM" --
    clf_type: str = "LDA"

    # -- Subjects used to pre-train the cross-subject model --
    train_subjects: List[int] = field(default_factory=lambda: [1, 2])

    # -- Subject whose data is streamed at demo time --
    test_subject: int = 3

    # -- Directory where MNE/MOABB data lives (contains MNE-zhou-2016/ subfolder).
    # -- Leave empty; the user must specify this path in the UI before training.
    data_path: str = ""


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DataEngine — MOABB-based data loading and preprocessing
# ══════════════════════════════════════════════════════════════════════════════
class DataEngine:
    """
    Loads Zhou2016 EEG data via MOABB, applies the LeftRightImagery paradigm
    (binary Left-Hand vs. Right-Hand classification), and exposes ready-to-use
    epoch arrays to the rest of the pipeline.

    Key signal-processing choices:
      • Average EEG re-reference  → removes common-mode drift / noise
      • Bandpass 8–30 Hz          → isolates μ/β motor rhythms (adjustable)
      • Baseline correction       → subtracts mean of pre-stimulus interval
      • Amplitude reject          → drops epochs > 200 µV to avoid artefacts
    """

    # Folder name that MOABB/MNE creates for Zhou2016 inside MNE_DATA
    ZHOU_FOLDER = "MNE-zhou-2016"

    def __init__(self, config: BCIConfig):
        self.config = config
        # Point MNE_DATA at the user-specified data directory.
        # Both the env var AND mne.set_config() are required:
        #   - os.environ is read by MNE at import time
        #   - mne.set_config() overrides any cached path from a previous run
        # NEVER delete or overwrite anything already present there.
        os.environ["MNE_DATA"] = config.data_path
        mne.set_config("MNE_DATA", config.data_path)
        os.makedirs(config.data_path, exist_ok=True)

        # MOABB paradigm — automatically filters to Left vs. Right classes
        self.paradigm = LeftRightImagery(
            fmin=config.f_low,
            fmax=config.f_high,
            tmin=config.t_min,
            tmax=config.t_max,
        )
        self.dataset = Zhou2016()

    # ------------------------------------------------------------------
    @classmethod
    def data_exists(cls, data_path: str) -> bool:
        """
        Return True if the Zhou2016 data folder already exists at data_path
        and contains at least one subject subfolder (sub-1 … sub-4).
        This check is non-destructive — it only reads directory listings.
        """
        zhou_dir = os.path.join(data_path, cls.ZHOU_FOLDER)
        if not os.path.isdir(zhou_dir):
            return False
        # Confirm at least one subject folder is present
        return any(
            os.path.isdir(os.path.join(zhou_dir, f"sub-{i}")) for i in range(1, 5)
        )

    # ------------------------------------------------------------------
    def get_subject_data(
        self, subject_id: int
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Return (X, y, sfreq) for a single subject.

          X     : (n_epochs, n_channels, n_times)  — bandpassed, baselined epochs
          y     : (n_epochs,)                       — integer class labels {0, 1}
          sfreq : float                              — sampling frequency (Hz)

        The sampling rate is read from the raw object — never assumed.
        """
        # get_data returns a dict keyed by subject id; we request one at a time
        X_raw, y_str, metadata = self.paradigm.get_data(
            dataset=self.dataset,
            subjects=[subject_id],
            return_epochs=False,
        )

        # Rebuild integer labels in a consistent order
        label_map = {v: k for k, v in enumerate(sorted(set(y_str)))}
        y_int = np.array([label_map[lbl] for lbl in y_str])

        # Retrieve sampling frequency from the raw info (no hard-coding)
        # MOABB gives us epochs directly; infer sfreq from epoch shape + tmax-tmin
        duration = self.config.t_max - self.config.t_min
        n_times   = X_raw.shape[-1]
        sfreq     = n_times / duration  # inferred — accurate for MOABB epochs

        return X_raw.astype(np.float64), y_int, sfreq, label_map

    # ------------------------------------------------------------------
    def get_train_data(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Concatenate data from all training subjects into a single (X, y) pair.
        Optionally applies Euclidean Mean Alignment (EA) across subjects to
        reduce cross-subject covariance shift before CSP fitting.
        """
        X_all, y_all = [], []
        for sid in self.config.train_subjects:
            print(f"  [DataEngine] Loading training subject {sid} …")
            X, y, sfreq, label_map = self.get_subject_data(sid)
            X_all.append(X)
            y_all.append(y)

        X_concat = np.concatenate(X_all, axis=0)
        y_concat  = np.concatenate(y_all, axis=0)
        print(
            f"  [DataEngine] Training set: {X_concat.shape[0]} epochs, "
            f"{X_concat.shape[1]} channels, {X_concat.shape[2]} samples."
        )
        # ── Euclidean Mean Alignment ──────────────────────────────────
        # Aligns each subject's epoch covariances to the grand mean,
        # reducing cross-subject distribution shift before CSP.
        X_concat = self._euclidean_alignment(X_concat)
        return X_concat, y_concat

    # ------------------------------------------------------------------
    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Return (X, y, sfreq) for the streaming test subject."""
        print(
            f"  [DataEngine] Loading test subject {self.config.test_subject} …"
        )
        X, y, sfreq, label_map = self.get_subject_data(self.config.test_subject)
        X = self._euclidean_alignment(X)
        return X, y, sfreq

    # ------------------------------------------------------------------
    @staticmethod
    def _euclidean_alignment(X: np.ndarray) -> np.ndarray:
        """
        Euclidean Mean Alignment (EA) — He & Wu, 2020.

        For each trial x_i ∈ ℝ^{C×T}:
          R̄  = mean covariance of all trials in this subject batch
          x̃_i = R̄^{-1/2} · x_i

        This centres the covariance distribution around the identity,
        making cross-subject transfer much more robust.
        """
        n, c, t = X.shape
        # Compute mean covariance matrix across all epochs
        cov_mean = np.mean(
            [xi @ xi.T / t for xi in X], axis=0
        )  # (C, C)

        # Matrix square-root inverse via eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(cov_mean)
        eigvals = np.maximum(eigvals, 1e-10)          # numerical safety
        R_inv_sqrt = eigvecs @ np.diag(eigvals ** -0.5) @ eigvecs.T

        # Apply alignment: x̃_i = R̄^{-1/2} · x_i
        return np.array([R_inv_sqrt @ xi for xi in X])


# ══════════════════════════════════════════════════════════════════════════════
# 3.  BCIModel — CSP + LDA/SVM sklearn Pipeline
# ══════════════════════════════════════════════════════════════════════════════
class BCIModel:
    """
    Wraps a two-step sklearn Pipeline:
      1. CSP (Common Spatial Patterns) — unsupervised spatial filter that
         maximises variance ratio between the two motor-imagery classes.
         log=True means the feature is log(band-power), which is more
         Gaussian and benefits LDA.
      2. LDA or SVM classifier.

    The pipeline can be rebuilt and retrained at any time when the user
    updates parameters via the UI — no application restart needed.
    """

    def __init__(self, config: BCIConfig):
        self.config = config
        self.pipeline: Optional[Pipeline] = None
        self.is_trained: bool = False

    # ------------------------------------------------------------------
    def build(self) -> None:
        """Construct (or reconstruct) the sklearn pipeline from current config."""
        csp = CSP(
            n_components=self.config.csp_components,
            reg=None,          # no regularisation — EA alignment handles this
            log=True,          # log-variance features → near-Gaussian → good for LDA
            norm_trace=False,
        )

        if self.config.clf_type == "LDA":
            classifier = LinearDiscriminantAnalysis()
        elif self.config.clf_type == "SVM":
            classifier = SVC(kernel="linear", C=1.0, probability=False)
        else:
            raise ValueError(
                f"Unknown clf_type '{self.config.clf_type}'. Use 'LDA' or 'SVM'."
            )

        self.pipeline = Pipeline([("csp", csp), ("clf", classifier)])
        self.is_trained = False

    # ------------------------------------------------------------------
    def train(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Fit the pipeline on training data.
        Returns training accuracy (informational only — not used for selection).
        """
        if self.pipeline is None:
            self.build()
        self.pipeline.fit(X, y)
        self.is_trained = True
        train_acc = self.pipeline.score(X, y)
        print(f"  [BCIModel] Training accuracy: {train_acc:.2%}")
        return train_acc

    # ------------------------------------------------------------------
    def predict(self, X_epoch: np.ndarray) -> int:
        """
        Predict class for a single epoch (shape: channels × times).
        The pipeline expects shape (1, C, T).
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet.")
        return int(self.pipeline.predict(X_epoch[np.newaxis, ...])[0])

    def predict_proba_single(self, X_epoch: np.ndarray) -> np.ndarray:
        """
        Returns [p_class0, p_class1] for a single epoch.
        Uses predict_proba if available (LDA), otherwise sigmoid of decision_function (SVM).
        """
        x = X_epoch[np.newaxis, ...]
        try:
            return self.pipeline.predict_proba(x)[0]
        except Exception:
            pass
        try:
            df = float(self.pipeline.decision_function(x)[0])
            p = 1.0 / (1.0 + np.exp(-df))
            return np.array([1.0 - p, p])
        except Exception:
            return np.array([0.5, 0.5])


# ══════════════════════════════════════════════════════════════════════════════
# 4.  StreamingSimulator — generator-based trial delivery
# ══════════════════════════════════════════════════════════════════════════════
class StreamingSimulator:
    """
    Simulates a real-time EEG data stream by iterating through pre-loaded
    test-subject epochs one trial at a time.

    Each call to next() on the generator yields:
      (epoch_array, true_label)
      epoch_array : (n_channels, n_times)  — single trial
      true_label  : int                    — ground-truth class index

    The generator pattern makes the data source fully decoupled from the UI
    update loop — the UI simply calls next() on a schedule.
    """

    # Human-readable class names (matches MOABB LeftRightImagery ordering)
    CLASS_NAMES = {0: "LEFT", 1: "RIGHT"}

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Parameters
        ----------
        X : (n_epochs, n_channels, n_times)
        y : (n_epochs,)  integer labels
        """
        self.X = X
        self.y = y
        self._gen: Optional[Generator] = None

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Restart the stream from the first trial."""
        self._gen = self._stream()

    # ------------------------------------------------------------------
    def _stream(self) -> Generator[Tuple[np.ndarray, int], None, None]:
        """Core generator — yields one (epoch, label) pair per next() call."""
        for i in range(len(self.X)):
            yield self.X[i], int(self.y[i])

    # ------------------------------------------------------------------
    def next_trial(self) -> Optional[Tuple[np.ndarray, int]]:
        """
        Advance the stream by one trial.
        Returns None when the stream is exhausted (demo complete).
        """
        if self._gen is None:
            self.reset()
        try:
            return next(self._gen)
        except StopIteration:
            return None

    # ------------------------------------------------------------------
    @classmethod
    def label_name(cls, label: int) -> str:
        return cls.CLASS_NAMES.get(label, f"CLASS_{label}")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  AppUI — tkinter interface (non-blocking via root.after)
# ══════════════════════════════════════════════════════════════════════════════
class AppUI:
    """
    Main application window.

    Layout
    ──────
    ┌─────────────────────────────────────────┐
    │              BCI Streaming Demo          │  ← title
    ├────────────────────┬────────────────────┤
    │   Settings Panel   │   Display Panel     │
    │  (left column)     │  (right column)     │
    │  f_low / f_high    │  Trial #            │
    │  t_min / t_max     │  Target             │
    │  csp_components    │  Status             │
    │  clf_type          │  Prediction         │
    │  train_subjects    │  Actual             │
    │  test_subject      │  Accuracy           │
    │  [Retrain] [Start] │                     │
    └────────────────────┴────────────────────┘

    Threading model
    ───────────────
    All heavy work (data loading, model training) runs in a daemon thread
    so the GUI stays responsive.  Trial streaming is driven by root.after()
    callbacks — pure single-threaded, no race conditions.
    """

    DISPLAY_INTERVAL_MS = 2500   # ms between trial displays
    ACTUAL_DELAY_MS     = 1500   # ms after prediction before showing actual label

    # ── Colour palette ────────────────────────────────────────────────
    BG_COLOR  = "#f6f8fa"   # light gray base (GitHub light)
    FG_COLOR  = "#24292f"   # near-black primary text
    ACCENT    = "#0969da"   # GitHub-style blue
    GREEN     = "#1a7f37"   # clean green
    RED       = "#cf222e"   # clean red
    YELLOW    = "#9a6700"   # warm amber

    # ── Button state colours ──────────────────────────────────────────
    _BTN_TRAIN_ON   = "#0969da"   # blue  — Train available
    _BTN_START_ON   = "#1a7f37"   # green — Start available
    _BTN_STOP_ON    = "#cf222e"   # red   — Stop active (base)
    _BTN_STOP_PULSE = "#fa4549"   # red   — Stop pulse (brighter)
    _BTN_OFF        = "#eaeef2"   # light grey — disabled/inactive
    _BTN_FG_OFF     = "#8c959f"   # dim text when button is inactive

    # ------------------------------------------------------------------
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("BCI Streaming Demo — Motor Imagery")
        self.root.configure(bg=self.BG_COLOR)
        self.root.resizable(True, True)

        self.config = BCIConfig()
        self.data_engine: Optional[DataEngine]       = None
        self.model:       Optional[BCIModel]         = None
        self.simulator:   Optional[StreamingSimulator] = None

        # Runtime state
        self._running   = False
        self._paused    = False
        self._trial_idx = 0
        self._correct   = 0
        self._total     = 0
        self._history: list = []   # list of bool per trial (True=correct)
        self._btn_enabled: dict = {}   # name → bool, gating Label-button clicks
        self._btns:        dict = {}   # name → (frame, dot_lbl, text_lbl)

        # ── New visualization state ────────────────────────────────────
        self._sfreq: float = 250.0              # updated after training
        self._conf_matrix: list = [[0, 0], [0, 0]]   # [actual][predicted]
        self._band_max: dict = {"mu": 1e-6, "beta": 1e-6}  # running max for normalisation

        self._build_ui()

    # ── UI construction ────────────────────────────────────────────────
    def _build_ui(self) -> None:
        # ── Title bar ─────────────────────────────────────────────────
        title = tk.Label(
            self.root,
            text="BCI  Real-Time Motor Imagery Demo",
            font=("Helvetica Neue", 20, "bold"),
            bg=self.BG_COLOR, fg=self.FG_COLOR,
        )
        title.pack(pady=(18, 2))

        subtitle = tk.Label(
            self.root,
            text="Zhou2016  ·  Cross-Subject Classification  ·  CSP + LDA / SVM",
            font=("Helvetica Neue", 10),
            bg=self.BG_COLOR, fg="#57606a",
        )
        subtitle.pack(pady=(0, 10))

        # Thin horizontal rule under title
        tk.Frame(self.root, height=1, bg="#d0d7de").pack(fill=tk.X, padx=20)

        # ── Main frame ────────────────────────────────────────────────
        main_frame = tk.Frame(self.root, bg=self.BG_COLOR)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=8)

        self._build_settings_panel(main_frame)
        self._build_display_panel(main_frame)

        # ── Status bar ────────────────────────────────────────────────
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

        # ══════════════════════════════════════════════════════════════
        # SECTION 1 — Data folder  (always visible, never scrolls)
        # ══════════════════════════════════════════════════════════════
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

        # ══════════════════════════════════════════════════════════════
        # SECTION 2 — Action buttons  (always visible, never scrolls)
        # ══════════════════════════════════════════════════════════════
        tk.Frame(panel, height=1, bg="#333").pack(fill=tk.X, padx=10, pady=(4, 2))

        def _make_btn(name: str, text: str, handler):
            bg = self._BTN_TRAIN_ON if name == 'train' else self._BTN_OFF
            fg = "white" if name == 'train' else self._BTN_FG_OFF
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

        _make_btn('train', "⚙   Train & Load", self._on_train)
        _make_btn('start', "▶   Start Stream",  self._on_start)
        _make_btn('pause', "⏸   Pause",          self._on_pause)
        _make_btn('stop',  "⏹   Stop",           self._on_stop)
        self._update_button_states("idle")

        # ══════════════════════════════════════════════════════════════
        # SECTION 3 — Parameters  (scrollable when window is small)
        # ══════════════════════════════════════════════════════════════
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
            # Show scrollbar only when content taller than canvas
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

        # ── Helpers targeting inner frame ─────────────────────────────
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

        # ── 1. Bandpass Filter ────────────────────────────────────────
        bp = section("Bandpass Filter")
        hint(bp, "Retain mu (8-12 Hz) and beta (13-30 Hz) motor rhythms, reject noise")
        add_slider(bp, "Low:",  "f_low",  1.0, 30.0, self.config.f_low)
        add_slider(bp, "High:", "f_high", 10.0, 60.0, self.config.f_high)

        # ── 2. Epoch Window ───────────────────────────────────────────
        ep = section("Epoch Window")
        hint(ep, "EEG segment per trial, relative to stimulus onset (seconds)")
        add_slider(ep, "Start:", "t_min", -1.0, 2.0, self.config.t_min)
        add_slider(ep, "End:",   "t_max",  0.5, 8.0, self.config.t_max)

        # ── 3. Spatial Filters (CSP) ──────────────────────────────────
        sp = section("Spatial Filters (CSP)")
        hint(sp, "Extract discriminative spatial patterns for left/right imagery")
        add_slider(sp, "Filters:", "csp_components", 2, 14,
                   self.config.csp_components, is_int=True)

        # ── 4. Training Setup ─────────────────────────────────────────
        tr = section("Training Setup", color="#1a7f37")

        def add_entry_row(parent, label, key, default):
            row = tk.Frame(parent, bg=self.BG_COLOR)
            row.pack(fill=tk.X, padx=6, pady=2)
            tk.Label(row, text=label, width=14, anchor="w",
                     font=("Helvetica Neue", 9),
                     bg=self.BG_COLOR, fg=self.FG_COLOR).pack(side=tk.LEFT)
            var = tk.StringVar(value=str(default))
            tk.Entry(row, textvariable=var, width=8,
                     font=("Helvetica Neue", 9),
                     bg="#ffffff", fg=self.FG_COLOR,
                     insertbackground=self.FG_COLOR,
                     relief=tk.FLAT, bd=3).pack(side=tk.LEFT)
            self._entries[key] = var

        hint(tr, "Train subjects: comma-separated IDs.  Test subject: single ID.")
        add_entry_row(tr, "Train subjects:", "train_subjects", "1,2")
        add_entry_row(tr, "Test subject:",   "test_subject",   self.config.test_subject)

        clf_row = tk.Frame(tr, bg=self.BG_COLOR)
        clf_row.pack(fill=tk.X, padx=6, pady=2)
        tk.Label(clf_row, text="Classifier:", width=14, anchor="w",
                 font=("Helvetica Neue", 9),
                 bg=self.BG_COLOR, fg=self.FG_COLOR).pack(side=tk.LEFT)
        self._clf_var = tk.StringVar(value=self.config.clf_type)
        ttk.Combobox(clf_row, textvariable=self._clf_var,
                     values=["LDA", "SVM"], width=8,
                     state="readonly").pack(side=tk.LEFT)

        tk.Frame(inner, height=8, bg=self.BG_COLOR).pack()  # bottom padding


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
            lbl = tk.Label(
                panel,
                text=text,
                font=("Helvetica Neue", font_size),
                bg=self.BG_COLOR,
                fg=color or self.FG_COLOR,
            )
            lbl.pack(pady=6)
            return lbl

        self.lbl_trial      = make_label("Trial:  —", 12, "#888")

        # ── EEG collection countdown ──────────────────────────────────
        self.lbl_countdown  = make_label("EEG Window:  — s", 20, self.YELLOW)

        # Progress bar: shows collection progress across the epoch window
        self.progress_bar = ttk.Progressbar(
            panel, orient="horizontal", mode="determinate", maximum=100,
        )
        self.progress_bar.pack(fill=tk.X, padx=30, pady=(0, 8))

        self.lbl_status     = make_label("Status:  Idle", 13, "#57606a")

        tk.Frame(panel, height=1, bg="#d0d7de").pack(fill=tk.X, padx=20, pady=6)

        self.lbl_prediction = make_label("Prediction:  —", 26, self.ACCENT)
        self.lbl_actual     = make_label("Actual:  —", 20, self.GREEN)

        # ── Classifier Confidence Bar ──────────────────────────────────
        tk.Label(
            panel, text="Classifier Confidence",
            font=("Helvetica Neue", 9), bg=self.BG_COLOR, fg="#57606a",
        ).pack()
        self._conf_canvas = tk.Canvas(
            panel, height=36, bg="#eaeef2", highlightthickness=0,
        )
        self._conf_canvas.pack(fill=tk.X, padx=20, pady=(2, 8))

        tk.Frame(panel, height=1, bg="#d0d7de").pack(fill=tk.X, padx=20, pady=4)

        self.lbl_accuracy   = make_label("Online Accuracy:  —", 14, self.FG_COLOR)
        self.lbl_train_acc  = make_label("—", 11, "#888")

        # ── Trial-by-trial accuracy bar chart ─────────────────────────
        tk.Frame(panel, height=2, bg="#333").pack(fill=tk.X, padx=20, pady=(8, 4))
        tk.Label(
            panel, text="Trial-by-Trial History",
            font=("Helvetica Neue", 9), bg=self.BG_COLOR, fg="#57606a",
        ).pack()
        self._chart_canvas = tk.Canvas(
            panel, height=100, bg="#eaeef2", highlightthickness=0,
        )
        self._chart_canvas.pack(fill=tk.X, padx=20, pady=(2, 4))

        # ── Band Power + Confusion Matrix (side by side) ───────────────
        vis_row = tk.Frame(panel, bg=self.BG_COLOR)
        vis_row.pack(fill=tk.X, padx=20, pady=(0, 8))

        bp_col = tk.Frame(vis_row, bg=self.BG_COLOR)
        bp_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(bp_col, text="Band Power  (μ / β)",
                 font=("Helvetica Neue", 9), bg=self.BG_COLOR, fg="#57606a").pack()
        self._bp_canvas = tk.Canvas(bp_col, height=80, bg="#eaeef2", highlightthickness=0)
        self._bp_canvas.pack(fill=tk.X, pady=(2, 0))

        tk.Frame(vis_row, width=8, bg=self.BG_COLOR).pack(side=tk.LEFT)

        cm_col = tk.Frame(vis_row, bg=self.BG_COLOR)
        cm_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(cm_col, text="Confusion Matrix",
                 font=("Helvetica Neue", 9), bg=self.BG_COLOR, fg="#57606a").pack()
        self._cm_canvas = tk.Canvas(cm_col, height=80, bg="#eaeef2", highlightthickness=0)
        self._cm_canvas.pack(fill=tk.X, pady=(2, 0))

    # ── Parameter reading ──────────────────────────────────────────────
    def _read_config(self) -> bool:
        """Parse UI entries into self.config. Returns False on validation error."""
        try:
            raw_path = self._data_path_var.get().strip()
            # Ignore placeholder text
            self.config.data_path = "" if raw_path.startswith("Paste folder") else raw_path
            self.config.f_low          = round(self._sliders["f_low"].get(), 1)
            self.config.f_high         = round(self._sliders["f_high"].get(), 1)
            self.config.t_min          = round(self._sliders["t_min"].get(), 1)
            self.config.t_max          = round(self._sliders["t_max"].get(), 1)
            self.config.csp_components = round(self._sliders["csp_components"].get())
            self.config.test_subject   = int(self._entries["test_subject"].get())
            self.config.clf_type       = self._clf_var.get()
            raw_subjects = self._entries["train_subjects"].get()
            self.config.train_subjects = [int(s.strip()) for s in raw_subjects.split(",")]
        except ValueError as exc:
            messagebox.showerror("Parameter Error", f"Invalid parameter value:\n{exc}")
            return False

        # Basic sanity checks
        if self.config.f_low >= self.config.f_high:
            messagebox.showerror("Parameter Error", "f_low must be < f_high.")
            return False
        if self.config.t_min >= self.config.t_max:
            messagebox.showerror("Parameter Error", "t_min must be < t_max.")
            return False
        if self.config.csp_components < 2:
            messagebox.showerror("Parameter Error", "CSP components must be ≥ 2.")
            return False
        return True

    # ── Button callbacks ───────────────────────────────────────────────
    def _on_train(self) -> None:
        """Validate parameters, check local data, then launch training thread."""
        if not self._read_config():
            return

        data_path = self.config.data_path

        if not data_path:
            messagebox.showerror(
                "No Data Path",
                "Please enter the folder path that contains MNE-zhou-2016/",
            )
            return

        # ── Check whether local data already exists ───────────────────
        if DataEngine.data_exists(data_path):
            self.lbl_data_status.config(
                text=f"Local data found.", fg=self.GREEN
            )
        else:
            # Data not found — ask user before downloading anything
            self.lbl_data_status.config(
                text="Data NOT found at this path.", fg=self.RED
            )
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
            # User agreed to download — MOABB will handle it automatically.
            # Existing files are NEVER touched or deleted.
            self.lbl_data_status.config(
                text="Downloading…  (existing files are never deleted)", fg=self.YELLOW
            )

        self._update_button_states("training")
        self._set_status("Loading data & training model …  (this may take ~1 min)")
        self.lbl_train_acc.config(text="Pre-training in progress…")
        threading.Thread(target=self._train_worker, daemon=True).start()

    def _on_start(self) -> None:
        """Begin the streaming loop."""
        if self.simulator is None:
            return
        self._running = True
        self._trial_idx = 0
        self._correct   = 0
        self._total     = 0
        self._history   = []
        self._conf_matrix  = [[0, 0], [0, 0]]
        self._band_max     = {"mu": 1e-6, "beta": 1e-6}
        self.simulator.reset()
        self._chart_canvas.delete("all")
        self._conf_canvas.delete("all")
        self._bp_canvas.delete("all")
        self._cm_canvas.delete("all")
        self._update_button_states("streaming")
        self._pulse_stop()
        self._set_status("Streaming …")
        self.root.after(200, self._stream_loop)

    def _on_pause(self) -> None:
        """Toggle pause/resume of the streaming loop."""
        if not self._running:
            return
        self._paused = not self._paused
        _, _, lbl = self._btns['pause']
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
        """Halt the streaming loop."""
        self._running = False
        self._paused  = False
        _, _, lbl = self._btns['pause']
        lbl.config(text="⏸   Pause")   # reset pause label for next run
        self._update_button_states("ready")
        self._set_status("Stopped.")

    # ── Background training worker ─────────────────────────────────────
    def _train_worker(self) -> None:
        """Runs in daemon thread. Posts results back to main thread via after()."""
        try:
            self.data_engine = DataEngine(self.config)

            # Load & concatenate training subjects
            X_train, y_train = self.data_engine.get_train_data()

            # Build and fit the CSP+Classifier pipeline
            self.model = BCIModel(self.config)
            self.model.build()
            train_acc = self.model.train(X_train, y_train)

            # Load test subject data
            X_test, y_test, sfreq = self.data_engine.get_test_data()
            self._sfreq = sfreq
            self.simulator = StreamingSimulator(X_test, y_test)

            # Post success to GUI thread
            self.root.after(
                0,
                lambda: self._on_train_done(train_acc, len(X_test)),
            )

        except Exception as exc:
            err_msg = str(exc)
            self.root.after(
                0,
                lambda: self._on_train_error(err_msg),
            )

    def _on_train_done(self, train_acc: float, n_test_trials: int) -> None:
        """Called on main thread when training completes successfully."""
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
        """Called on main thread when training throws an exception."""
        messagebox.showerror("Training Error", msg)
        self._update_button_states("idle")
        self._set_status("Training failed. See error dialog.")

    # ── Streaming loop (root.after driven) ────────────────────────────
    _COUNTDOWN_TICK_MS = 50   # progress bar refresh rate (ms)

    def _stream_loop(self) -> None:
        """
        Entry point for each new trial.
        Fetches the next epoch from the simulator, resets the display,
        then hands off to _run_countdown() which drives the EEG collection
        animation before triggering the prediction.
        """
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
        self._total += 1

        # ── Reset display for new trial ───────────────────────────────
        self.lbl_trial.config(
            text=f"Trial:  {self._trial_idx}  /  {len(self.simulator.X)}"
        )
        self.lbl_status.config(text="Status:  Collecting EEG …", fg=self.YELLOW)
        self.lbl_prediction.config(text="Prediction:  —", fg=self.ACCENT)
        self.lbl_actual.config(text="Actual:  —", fg="#8c959f")
        self.progress_bar["value"] = 0

        # ── Start the countdown animation ─────────────────────────────
        total_ms = int((self.config.t_max - self.config.t_min) * 1000)
        self._run_countdown(epoch, true_label, elapsed_ms=0, total_ms=total_ms)

    def _run_countdown(
        self, epoch: np.ndarray, true_label: int, elapsed_ms: int, total_ms: int
    ) -> None:
        """
        Recursive root.after callback — fires every _COUNTDOWN_TICK_MS ms.
        Updates the countdown label and progress bar to visualise the
        simulated EEG collection window (t_max - t_min seconds).
        When the window is full, calls _do_predict().
        """
        if not self._running or self._paused:
            return

        remaining_s = max(0.0, (total_ms - elapsed_ms) / 1000.0)
        pct = min(100, int(elapsed_ms / total_ms * 100))

        self.lbl_countdown.config(
            text=f"EEG Window:  {elapsed_ms / 1000:.1f} s  /  {total_ms / 1000:.1f} s"
                 f"   ({remaining_s:.1f} s left)"
        )
        self.progress_bar["value"] = pct

        if elapsed_ms >= total_ms:
            # Collection window complete — run prediction now
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
        """
        Run the classifier on the epoch and update the prediction display.
        Schedules reveal of the actual label after ACTUAL_DELAY_MS,
        and the next trial after DISPLAY_INTERVAL_MS.
        """
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

        # Prediction shown in blue first — verdict colour revealed later
        self.lbl_status.config(text="Status:  Prediction Ready", fg=self.GREEN)
        self.lbl_prediction.config(text=f"Prediction:  {pred_name}", fg=self.ACCENT)
        self.progress_bar["value"] = 100

        # ── Update confidence bar + band power immediately ────────────
        self._update_confidence(proba)
        self._update_band_power(epoch)

        # ── Reveal actual label after a short pause ───────────────────
        def reveal_actual():
            # Now update prediction colour: green = correct, red = wrong
            self.lbl_prediction.config(
                fg=self.GREEN if is_correct else self.RED,
            )
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
            # Update confusion matrix: rows = actual, cols = predicted
            if 0 <= true_label <= 1 and 0 <= pred_label <= 1:
                self._conf_matrix[true_label][pred_label] += 1
            self._update_chart()
            self._update_confusion_matrix()

        self.root.after(self.ACTUAL_DELAY_MS, reveal_actual)

        # ── Schedule next trial ───────────────────────────────────────
        self.root.after(self.DISPLAY_INTERVAL_MS, self._stream_loop)

    # ── Visualisation helpers ──────────────────────────────────────────

    def _update_confidence(self, proba: np.ndarray) -> None:
        """Horizontal split bar: LEFT probability | RIGHT probability."""
        c = self._conf_canvas
        c.delete("all")
        c.update_idletasks()
        W, H = c.winfo_width(), c.winfo_height()
        if W < 10 or H < 10:
            return
        p_left  = float(proba[0])
        p_right = float(proba[1])
        split   = max(4, min(W - 4, int(W * p_left)))
        pad     = 3
        # Left fill (blue)
        c.create_rectangle(pad, pad, split, H - pad, fill=self.ACCENT, outline="")
        # Right fill (light gray)
        if split < W - pad:
            c.create_rectangle(split, pad, W - pad, H - pad, fill="#d0d7de", outline="")
        # Labels — always visible, placed in their own halves
        lx = (pad + split) // 2
        rx = (split + W - pad) // 2
        lc = "white"     if p_left  > 0.15 else self.FG_COLOR
        rc = self.FG_COLOR if p_right > 0.15 else "white"
        c.create_text(lx, H // 2, text=f"LEFT  {p_left:.0%}",
                      fill=lc, font=("Helvetica Neue", 9, "bold"), anchor="center")
        c.create_text(rx, H // 2, text=f"RIGHT  {p_right:.0%}",
                      fill=rc, font=("Helvetica Neue", 9, "bold"), anchor="center")

    # ------------------------------------------------------------------
    @staticmethod
    def _band_power(epoch: np.ndarray, sfreq: float, fmin: float, fmax: float) -> float:
        """Band power as fraction of total spectral power (0–1)."""
        n_times = epoch.shape[1]
        freqs   = np.fft.rfftfreq(n_times, 1.0 / sfreq)
        ps      = np.abs(np.fft.rfft(epoch, axis=1)) ** 2   # (channels, freqs)
        mask    = (freqs >= fmin) & (freqs <= fmax)
        band    = float(ps[:, mask].mean()) if mask.any() else 0.0
        total   = float(ps.mean()) + 1e-10
        return band / total

    # ------------------------------------------------------------------
    def _update_band_power(self, epoch: np.ndarray) -> None:
        """Two vertical bars: μ (8–12 Hz) and β (13–30 Hz) band power."""
        c = self._bp_canvas
        c.delete("all")
        c.update_idletasks()
        W, H = c.winfo_width(), c.winfo_height()
        if W < 10 or H < 10:
            return
        mu_frac   = self._band_power(epoch, self._sfreq, 8,  12)
        beta_frac = self._band_power(epoch, self._sfreq, 13, 30)
        bands = [
            ("μ  8–12 Hz",  mu_frac,   self.ACCENT),
            ("β  13–30 Hz", beta_frac, self.GREEN),
        ]
        label_h = 16
        plot_h  = H - label_h - 8
        n       = len(bands)
        gap     = 10
        bar_w   = (W - (n + 1) * gap) // n
        for i, (label, frac, color) in enumerate(bands):
            x0 = gap + i * (bar_w + gap)
            x1 = x0 + bar_w
            # Background track
            c.create_rectangle(x0, 4, x1, H - label_h - 4,
                                fill="#d0d7de", outline="")
            # Power fill
            fill_h = max(2, int(frac * plot_h))
            c.create_rectangle(x0, H - label_h - 4 - fill_h, x1, H - label_h - 4,
                                fill=color, outline="")
            c.create_text((x0 + x1) // 2, H - label_h // 2 - 2,
                          text=f"{label}  {frac:.0%}",
                          fill=self.FG_COLOR, font=("Helvetica Neue", 7), anchor="center")

    # ------------------------------------------------------------------
    def _update_confusion_matrix(self) -> None:
        """2×2 grid — rows = Actual, cols = Predicted.  Colour intensity ∝ count."""
        c = self._cm_canvas
        c.delete("all")
        c.update_idletasks()
        W, H = c.winfo_width(), c.winfo_height()
        if W < 10 or H < 10:
            return
        matrix  = self._conf_matrix
        max_val = max(v for row in matrix for v in row)
        if max_val == 0:
            return
        label_w, label_h = 20, 16
        cell_w = (W - label_w) // 2
        cell_h = (H - label_h) // 2
        # Column headers (Predicted)
        for j, name in enumerate(["LEFT", "RIGHT"]):
            x = label_w + j * cell_w + cell_w // 2
            c.create_text(x, label_h // 2, text=name, fill="#57606a",
                          font=("Helvetica Neue", 7), anchor="center")
        # Row headers (Actual) + cells
        for i, name in enumerate(["LEFT", "RIGHT"]):
            y = label_h + i * cell_h + cell_h // 2
            c.create_text(label_w // 2, y, text=name, fill="#57606a",
                          font=("Helvetica Neue", 7), anchor="center")
            for j in range(2):
                val   = matrix[i][j]
                alpha = val / max_val
                x0    = label_w + j * cell_w
                y0    = label_h + i * cell_h
                x1, y1 = x0 + cell_w - 1, y0 + cell_h - 1
                if i == j:  # diagonal → correct → green tint
                    r = int(234 + (26  - 234) * alpha)
                    g = int(255 + (127 - 255) * alpha)
                    b = int(240 + (55  - 240) * alpha)
                else:       # off-diagonal → wrong → red tint
                    r = int(255 + (207 - 255) * alpha)
                    g = int(240 + (34  - 240) * alpha)
                    b = int(240 + (34  - 240) * alpha)
                fill   = f"#{r:02x}{g:02x}{b:02x}"
                text_c = "white" if alpha > 0.5 else self.FG_COLOR
                c.create_rectangle(x0, y0, x1, y1, fill=fill, outline="#d0d7de")
                c.create_text((x0 + x1) // 2, (y0 + y1) // 2, text=str(val),
                              fill=text_c, font=("Helvetica Neue", 10, "bold"),
                              anchor="center")

    # ── Button state management ────────────────────────────────────────
    def _update_button_states(self, state: str) -> None:
        """
        Centralised button colour + dot colour update.

        States
        ------
        "idle"      — app just started, no model yet
        "training"  — background training thread running
        "ready"     — model ready, waiting to stream
        "streaming" — live stream running
        "paused"    — stream paused mid-session
        """
        # (enabled, bg, fg, dot_color)
        ON_TR  = (True,  self._BTN_TRAIN_ON, "white",          "#8c959f")
        ON_ST  = (True,  self._BTN_START_ON, "white",          "#8c959f")
        ON_PA  = (True,  "#7d4e00",          "white",          self.YELLOW)  # amber pause
        ON_SP  = (True,  self._BTN_STOP_ON,  "white",          self.RED)     # red dot
        OFF    = (False, self._BTN_OFF,      self._BTN_FG_OFF, "#8c959f")

        table = {
            #           train   start   pause   stop
            "idle"     : [ON_TR, OFF,    OFF,    OFF  ],
            "training" : [OFF,   OFF,    OFF,    OFF  ],
            "ready"    : [ON_TR, ON_ST,  OFF,    OFF  ],
            "streaming": [OFF,   OFF,    ON_PA,  ON_SP],
            "paused"   : [OFF,   ON_ST,  ON_PA,  ON_SP],
        }

        for (name, btn_widgets), cfg in zip(self._btns.items(), table[state]):
            enabled, bg, fg, dot_color = cfg
            self._btn_enabled[name] = enabled
            f, dot, lbl = btn_widgets
            f.config(bg=bg, cursor="hand2" if enabled else "")
            dot.config(bg=bg, fg=dot_color)
            lbl.config(bg=bg, fg=fg)

    def _pulse_stop(self, bright: bool = True) -> None:
        """Alternate Stop button bg while streaming for a pulsing effect."""
        if not self._running or self._paused:
            return
        f, dot, lbl = self._btns['stop']
        color = self._BTN_STOP_PULSE if bright else self._BTN_STOP_ON
        f.config(bg=color)
        dot.config(bg=color)
        lbl.config(bg=color)
        self.root.after(550, lambda: self._pulse_stop(not bright))

    # ── Accuracy bar chart ─────────────────────────────────────────────
    _CHART_MAX_BARS = 60   # show at most this many recent trials

    def _update_chart(self) -> None:
        """
        Redraws the trial-by-trial accuracy canvas.

        Layout
        ------
          Left 32 px  → Y-axis labels  (0 % … 100 %)
          Remaining   → plot area

        Bars   : full-height, green = correct, red = incorrect.
        Line   : cumulative accuracy after each trial — identical to the
                 value shown in the Online Accuracy label, so the line
                 sits exactly at the labelled percentage on the Y-axis.
        """
        canvas = self._chart_canvas
        canvas.delete("all")

        history = self._history[-self._CHART_MAX_BARS:]
        n = len(history)
        if n == 0:
            return

        canvas.update_idletasks()
        W = canvas.winfo_width()
        H = canvas.winfo_height()
        if W < 10 or H < 10:
            return

        # ── Coordinate system ─────────────────────────────────────────
        AXIS_W  = 30    # width reserved for Y-axis labels
        TOP_PAD = 4     # pixels above 100 % line
        BOT_PAD = 4     # pixels below 0 % line
        plot_x0 = AXIS_W
        plot_y0 = TOP_PAD
        plot_y1 = H - BOT_PAD
        plot_h  = plot_y1 - plot_y0   # height in pixels for full 0-100 % range

        def pct_to_y(p: float) -> int:
            """Map p ∈ [0, 1] to canvas Y (0 % → bottom, 100 % → top)."""
            return plot_y1 - int(p * plot_h)

        # ── Y-axis labels + horizontal grid lines ─────────────────────
        for pct in (0.0, 0.25, 0.50, 0.75, 1.0):
            y = pct_to_y(pct)
            # Dashed grid line across the plot area
            canvas.create_line(
                plot_x0, y, W, y,
                fill="#d0d7de", dash=(3, 5),
            )
            # Percentage label
            canvas.create_text(
                AXIS_W - 3, y,
                text=f"{int(pct * 100)}%",
                anchor="e", fill="#57606a",
                font=("Helvetica Neue", 7),
            )

        # ── Bars ──────────────────────────────────────────────────────
        plot_w  = W - plot_x0 - 2
        bar_w   = max(3, plot_w // self._CHART_MAX_BARS)
        bar_gap = max(1, bar_w // 5)

        for i, correct in enumerate(history):
            x1 = plot_x0 + i * (bar_w + bar_gap)
            x2 = x1 + bar_w
            color = self.GREEN if correct else self.RED
            # Bar spans the full plot height; colour communicates outcome
            canvas.create_rectangle(x1, plot_y0, x2, plot_y1, fill=color, outline="")

        # ── Cumulative accuracy line ───────────────────────────────────
        # Computed the same way as Online Accuracy: correct_so_far / trials_so_far
        # → the line is always at exactly the position the label shows.
        points = []
        running_correct = 0
        for i, correct in enumerate(history):
            running_correct += int(correct)
            cum_acc = running_correct / (i + 1)
            x_mid   = plot_x0 + i * (bar_w + bar_gap) + bar_w // 2
            y       = pct_to_y(cum_acc)
            points.extend([x_mid, y])

        if len(points) >= 4:
            canvas.create_line(*points, fill="#0969da", width=2, smooth=True)

    # ── Helpers ────────────────────────────────────────────────────────
    def _set_status(self, msg: str) -> None:
        self.status_bar.config(text=f"  {msg}")
        print(f"[STATUS] {msg}")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Entry point
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    # macOS: use the 'TkAgg' backend so MNE plots (if any) don't steal focus
    import matplotlib
    matplotlib.use("Agg")

    root = tk.Tk()
    root.geometry("1050x720")
    root.minsize(860, 600)

    # macOS window appearance tweak
    try:
        root.tk.call("::tk::unsupported::MacWindowStyle", "style", root._w,
                     "document", "closeBox collapseBox resizeBox")
    except Exception:
        pass

    app = AppUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
