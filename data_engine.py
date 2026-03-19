"""
data_engine.py — MOABB-based EEG data loading and preprocessing.

Key signal-processing choices:
  • Average EEG re-reference  → removes common-mode drift / noise
  • Bandpass 8–30 Hz          → isolates μ/β motor rhythms (adjustable)
  • Euclidean Mean Alignment  → reduces cross-subject covariance shift
"""

import os

import mne
import moabb
import numpy as np
from moabb.datasets import Zhou2016
from moabb.paradigms import LeftRightImagery

from config import BCIConfig

mne.set_log_level("WARNING")
moabb.set_log_level("WARNING")


class DataEngine:
    """
    Loads Zhou2016 EEG data via MOABB, applies the LeftRightImagery paradigm
    (binary Left-Hand vs. Right-Hand classification), and exposes ready-to-use
    epoch arrays to the rest of the pipeline.
    """

    # Folder name that MOABB/MNE creates for Zhou2016 inside MNE_DATA
    ZHOU_FOLDER = "MNE-zhou-2016"

    def __init__(self, config: BCIConfig):
        self.config = config
        # Both the env var AND mne.set_config() are required:
        #   - os.environ is read by MNE at import time
        #   - mne.set_config() overrides any cached path from a previous run
        # NEVER delete or overwrite anything already present there.
        os.environ["MNE_DATA"] = config.data_path
        mne.set_config("MNE_DATA", config.data_path)
        os.makedirs(config.data_path, exist_ok=True)

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
        Non-destructive — only reads directory listings.
        """
        zhou_dir = os.path.join(data_path, cls.ZHOU_FOLDER)
        if not os.path.isdir(zhou_dir):
            return False
        return any(
            os.path.isdir(os.path.join(zhou_dir, f"sub-{i}")) for i in range(1, 5)
        )

    # ------------------------------------------------------------------
    def get_subject_data(self, subject_id: int):
        """
        Return (X, y, sfreq, label_map) for a single subject.

          X         : (n_epochs, n_channels, n_times)
          y         : (n_epochs,)  integer class labels {0, 1}
          sfreq     : float  — sampling frequency (Hz)
          label_map : dict   — {string_label: int_label}
        """
        X_raw, y_str, metadata = self.paradigm.get_data(
            dataset=self.dataset,
            subjects=[subject_id],
            return_epochs=False,
        )
        label_map = {v: k for k, v in enumerate(sorted(set(y_str)))}
        y_int = np.array([label_map[lbl] for lbl in y_str])

        duration = self.config.t_max - self.config.t_min
        n_times  = X_raw.shape[-1]
        sfreq    = n_times / duration   # inferred — accurate for MOABB epochs

        return X_raw.astype(np.float64), y_int, sfreq, label_map

    # ------------------------------------------------------------------
    def get_train_data(self):
        """
        Concatenate data from all training subjects and apply EA alignment.
        Returns (X, y).
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
        X_concat = self._euclidean_alignment(X_concat)
        return X_concat, y_concat

    # ------------------------------------------------------------------
    def get_test_data(self):
        """Return (X, y, sfreq) for the streaming test subject."""
        print(f"  [DataEngine] Loading test subject {self.config.test_subject} …")
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

        Centres the covariance distribution around the identity,
        making cross-subject transfer more robust.
        """
        n, c, t  = X.shape
        cov_mean = np.mean([xi @ xi.T / t for xi in X], axis=0)

        eigvals, eigvecs = np.linalg.eigh(cov_mean)
        eigvals    = np.maximum(eigvals, 1e-10)   # numerical safety
        R_inv_sqrt = eigvecs @ np.diag(eigvals ** -0.5) @ eigvecs.T

        return np.array([R_inv_sqrt @ xi for xi in X])
