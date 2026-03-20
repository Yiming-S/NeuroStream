"""
data_engine.py — MOABB-based EEG data loading and preprocessing.

Supports two evaluation protocols (driven by BCIConfig.evaluation_protocol):

  "Cross-Subject"
      Training data : all sessions of subjects listed in config.train_subjects
      Test data     : all sessions of config.test_subject
      Alignment     : Euclidean Alignment (EA) applied per-subject batch

  "Cross-Session"
      Training data : sessions config.train_sessions of config.cross_session_subject
      Test data     : session  config.test_session   of config.cross_session_subject
      Alignment     : Euclidean Alignment applied per-session batch

Paradigm selection (driven by BCIConfig.pipeline_type):
  CSP / TS family  → LeftRightImagery          → 3-D array (n_trials, channels, times)
  FBCSP            → FilterBankLeftRightImagery → 4-D array (n_trials, channels, times, bands)
"""

import os

import mne
import moabb
import numpy as np
from moabb.datasets import Zhou2016
from moabb.paradigms import FilterBankLeftRightImagery, LeftRightImagery

from config import BCIConfig

mne.set_log_level("WARNING")
moabb.set_log_level("WARNING")

_FILTERBANK_PIPELINES = {"FBCSP"}
_RIEMANNIAN_PIPELINES = {"TS+LDA", "TS+SVM"}


class DataEngine:
    """
    Loads Zhou2016 EEG data via MOABB.

    Automatically selects the correct MOABB paradigm and evaluation protocol
    based on BCIConfig, then exposes get_train_data() and get_test_data()
    with shapes expected by BCIModel.
    """

    ZHOU_FOLDER = "MNE-zhou-2016"

    def __init__(self, config: BCIConfig):
        self.config = config

        # Both env var AND mne.set_config() are required — env var is read at
        # import time, set_config() overrides any path cached from a prior run.
        # Neither call deletes or overwrites existing data files.
        os.environ["MNE_DATA"] = config.data_path
        mne.set_config("MNE_DATA", config.data_path)
        os.makedirs(config.data_path, exist_ok=True)

        self.dataset = Zhou2016()

        self._use_filterbank = config.pipeline_type in _FILTERBANK_PIPELINES
        self._use_riemannian = config.pipeline_type in _RIEMANNIAN_PIPELINES

        if self._use_filterbank:
            bands = [
                [float(x) for x in band.strip().split("-")]
                for band in config.fb_bands.split(",")
                if band.strip()
            ]
            self.paradigm = FilterBankLeftRightImagery(
                filters=bands,
                tmin=config.t_min,
                tmax=config.t_max,
            )
        else:
            self.paradigm = LeftRightImagery(
                fmin=config.f_low,
                fmax=config.f_high,
                tmin=config.t_min,
                tmax=config.t_max,
            )

    # ──────────────────────────────────────────────────────────────────
    # Public helpers
    # ──────────────────────────────────────────────────────────────────

    @classmethod
    def data_exists(cls, data_path: str) -> bool:
        """
        Return True if the Zhou2016 data folder already exists at data_path
        and contains at least one subject subfolder (sub-1 … sub-4).
        Non-destructive — only reads directory listings.
        """
        zhou_dir = os.path.join(data_path, cls.ZHOU_FOLDER)
        if not os.path.isdir(zhou_dir):
            return False
        return any(
            os.path.isdir(os.path.join(zhou_dir, f"sub-{i}")) for i in range(1, 5)
        )

    # ──────────────────────────────────────────────────────────────────
    # Data loading — public API consumed by BCIModel and StreamingSimulator
    # ──────────────────────────────────────────────────────────────────

    def get_train_data(self):
        """
        Load and return training epochs according to the evaluation protocol.

        Returns
        -------
        X : np.ndarray
            Shape (n_trials, n_channels, n_times)      for CSP / Riemannian
            Shape (n_trials, n_channels, n_times, n_bands) for FBCSP
        y : np.ndarray  shape (n_trials,)
        """
        if self.config.evaluation_protocol == "Cross-Subject":
            return self._load_cross_subject_train()
        else:
            return self._load_cross_session_train()

    def get_test_data(self):
        """
        Load and return test epochs according to the evaluation protocol.

        Returns
        -------
        X     : np.ndarray  (same shape convention as get_train_data)
        y     : np.ndarray  shape (n_trials,)
        sfreq : float  sampling frequency in Hz
        """
        if self.config.evaluation_protocol == "Cross-Subject":
            return self._load_cross_subject_test()
        else:
            return self._load_cross_session_test()

    # ──────────────────────────────────────────────────────────────────
    # Cross-Subject loading
    # ──────────────────────────────────────────────────────────────────

    def _load_cross_subject_train(self):
        X_all, y_all = [], []
        for subject_id in self.config.train_subjects:
            print(f"  [DataEngine] Cross-Subject  — loading subject {subject_id} …")
            X, y, _, _ = self._load_subject(subject_id)
            X_all.append(X)
            y_all.append(y)

        X_concat = np.concatenate(X_all, axis=0)
        y_concat  = np.concatenate(y_all, axis=0)
        self._log_shape("Training", X_concat)

        if not self._use_riemannian:
            X_concat = self._apply_euclidean_alignment(X_concat)
        return X_concat, y_concat

    def _load_cross_subject_test(self):
        subject_id = self.config.test_subject
        print(f"  [DataEngine] Cross-Subject  — loading test subject {subject_id} …")
        X, y, sfreq, _ = self._load_subject(subject_id)
        if not self._use_riemannian:
            X = self._apply_euclidean_alignment(X)
        return X, y, sfreq

    # ──────────────────────────────────────────────────────────────────
    # Cross-Session loading
    # ──────────────────────────────────────────────────────────────────

    def _load_cross_session_train(self):
        subject_id = self.config.cross_session_subject
        X_all, y_all = [], []
        for session_index in self.config.train_sessions:
            print(
                f"  [DataEngine] Cross-Session  — subject {subject_id}, "
                f"session {session_index} (train) …"
            )
            X, y, _, _ = self._load_session(subject_id, session_index)
            X_all.append(X)
            y_all.append(y)

        X_concat = np.concatenate(X_all, axis=0)
        y_concat  = np.concatenate(y_all, axis=0)
        self._log_shape("Training", X_concat)

        if not self._use_riemannian:
            X_concat = self._apply_euclidean_alignment(X_concat)
        return X_concat, y_concat

    def _load_cross_session_test(self):
        subject_id   = self.config.cross_session_subject
        session_index = self.config.test_session
        print(
            f"  [DataEngine] Cross-Session  — subject {subject_id}, "
            f"session {session_index} (test) …"
        )
        X, y, sfreq, _ = self._load_session(subject_id, session_index)
        if not self._use_riemannian:
            X = self._apply_euclidean_alignment(X)
        return X, y, sfreq

    # ──────────────────────────────────────────────────────────────────
    # MOABB data fetchers
    # ──────────────────────────────────────────────────────────────────

    def _load_subject(self, subject_id: int):
        """Load all sessions for one subject. Returns (X, y, sfreq, label_map)."""
        X_raw, y_str, _ = self.paradigm.get_data(
            dataset=self.dataset,
            subjects=[subject_id],
            return_epochs=False,
        )
        return self._finalise(X_raw, y_str)

    def _load_session(self, subject_id: int, session_index: int):
        """
        Load a single session for one subject.

        Zhou2016 session IDs are string integers: '0', '1', '2'.
        """
        session_id = str(session_index)
        X_raw, y_str, metadata = self.paradigm.get_data(
            dataset=self.dataset,
            subjects=[subject_id],
            return_epochs=False,
        )
        mask  = metadata["session"].values == session_id
        X_raw = X_raw[mask]
        y_str = y_str[mask]
        return self._finalise(X_raw, y_str)

    def _finalise(self, X_raw: np.ndarray, y_str: np.ndarray):
        """Convert string labels to integers and infer sfreq."""
        label_map = {v: k for k, v in enumerate(sorted(set(y_str)))}
        y_int     = np.array([label_map[lbl] for lbl in y_str])
        duration  = self.config.t_max - self.config.t_min
        n_times   = X_raw.shape[2]
        sfreq     = n_times / duration
        return X_raw.astype(np.float64), y_int, sfreq, label_map

    # ──────────────────────────────────────────────────────────────────
    # Euclidean Alignment
    # ──────────────────────────────────────────────────────────────────

    def _apply_euclidean_alignment(self, X: np.ndarray) -> np.ndarray:
        """
        Dispatch to the appropriate alignment function based on data dimensions.

        3-D input (n, C, T)      → align the whole batch as one set.
        4-D input (n, C, T, B)   → align each frequency band independently.
        """
        if X.ndim == 3:
            return self._euclidean_alignment(X)
        elif X.ndim == 4:
            return np.stack(
                [self._euclidean_alignment(X[..., band]) for band in range(X.shape[-1])],
                axis=-1,
            )
        else:
            raise ValueError(f"Expected 3-D or 4-D input, got shape {X.shape}")

    @staticmethod
    def _euclidean_alignment(X: np.ndarray) -> np.ndarray:
        """
        Euclidean Mean Alignment — He & Wu (2020).

        For each trial x_i in R^{C x T}:
            mean_covariance  = mean of (x_i @ x_i.T / T) over all trials
            aligned_x_i      = mean_covariance^{-1/2} @ x_i

        Centres the covariance distribution around the identity matrix,
        reducing cross-subject and cross-session covariance shift.
        """
        n, channels, time_points = X.shape
        mean_covariance = np.mean([x @ x.T / time_points for x in X], axis=0)

        eigenvalues, eigenvectors = np.linalg.eigh(mean_covariance)
        eigenvalues           = np.maximum(eigenvalues, 1e-10)   # numerical safety
        inverse_sqrt_matrix   = (
            eigenvectors @ np.diag(eigenvalues ** -0.5) @ eigenvectors.T
        )
        return np.array([inverse_sqrt_matrix @ x for x in X])

    # ──────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _log_shape(label: str, X: np.ndarray) -> None:
        shape_str = (
            f"{X.shape[0]} trials, {X.shape[1]} channels, {X.shape[2]} samples"
            + (f", {X.shape[3]} bands" if X.ndim == 4 else "")
        )
        print(f"  [DataEngine] {label} set: {shape_str}.")
