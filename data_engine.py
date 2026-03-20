"""
data_engine.py — MOABB-based EEG data loading and preprocessing.

Paradigm selection (driven by BCIConfig.pipeline_type):
  • CSP / TS / MDM  → LeftRightImagery      → 3-D array (n, C, T)
  • FBCSP           → FilterBankLeftRightImagery (moabb) → 4-D array (n, C, T, B)

Alignment:
  • CSP, FBCSP  → Euclidean Mean Alignment (He & Wu 2020)
  • TS+*, MDM   → no EA; pyriemann operates on covariance matrices natively
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

# Pipeline types that need the FilterBank paradigm (4-D output)
_FB_PIPELINES = {"FBCSP"}

# Pipeline types that operate on raw covariance matrices (no EA)
_RIEMANNIAN_PIPELINES = {"TS+LDA", "TS+SVM", "MDM"}


class DataEngine:
    """
    Loads Zhou2016 EEG data via MOABB.

    Automatically selects the correct MOABB paradigm based on pipeline_type:
      - LeftRightImagery             for CSP / Riemannian families
      - FilterBankLeftRightImagery   for FBCSP

    Exposes get_train_data() and get_test_data() with shapes expected by
    BCIModel for each pipeline family.
    """

    ZHOU_FOLDER = "MNE-zhou-2016"

    def __init__(self, config: BCIConfig):
        self.config = config
        # Both env var AND mne.set_config() required — see comments in original.
        os.environ["MNE_DATA"] = config.data_path
        mne.set_config("MNE_DATA", config.data_path)
        os.makedirs(config.data_path, exist_ok=True)

        self.dataset = Zhou2016()
        self._is_fb         = config.pipeline_type in _FB_PIPELINES
        self._is_riemannian = config.pipeline_type in _RIEMANNIAN_PIPELINES

        if self._is_fb:
            # moabb.paradigms.FilterBankLeftRightImagery returns
            # X of shape (n_trials, n_channels, n_times, n_bands)
            # Parse "8-12,12-16,..." → [[8,12],[12,16],...]
            bands = [
                [float(x) for x in b.strip().split("-")]
                for b in config.fb_bands.split(",")
                if b.strip()
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

          X         : (n_epochs, n_channels, n_times)       — CSP / Riemannian
                      (n_epochs, n_channels, n_times, n_bands) — FBCSP
          y         : (n_epochs,)  integer class labels {0, 1}
          sfreq     : float  — sampling frequency (Hz)
          label_map : dict   — {string_label: int_label}
        """
        X_raw, y_str, _ = self.paradigm.get_data(
            dataset=self.dataset,
            subjects=[subject_id],
            return_epochs=False,
        )
        label_map = {v: k for k, v in enumerate(sorted(set(y_str)))}
        y_int = np.array([label_map[lbl] for lbl in y_str])

        duration = self.config.t_max - self.config.t_min
        # For 4-D (FBCSP): n_times is axis=2; for 3-D: axis=2 as well
        n_times = X_raw.shape[2]
        sfreq   = n_times / duration

        return X_raw.astype(np.float64), y_int, sfreq, label_map

    # ------------------------------------------------------------------
    def get_train_data(self):
        """
        Concatenate data from all training subjects.
        Applies EA for CSP/FBCSP; skips it for Riemannian pipelines.
        Returns (X, y).
        """
        X_all, y_all = [], []
        for sid in self.config.train_subjects:
            print(f"  [DataEngine] Loading training subject {sid} …")
            X, y, _, _ = self.get_subject_data(sid)
            X_all.append(X)
            y_all.append(y)

        X_concat = np.concatenate(X_all, axis=0)
        y_concat  = np.concatenate(y_all, axis=0)

        ndim = X_concat.ndim
        print(
            f"  [DataEngine] Training set: {X_concat.shape[0]} epochs, "
            f"{X_concat.shape[1]} channels, {X_concat.shape[2]} samples"
            + (f", {X_concat.shape[3]} bands." if ndim == 4 else ".")
        )

        if not self._is_riemannian:
            X_concat = self._apply_ea(X_concat)

        return X_concat, y_concat

    # ------------------------------------------------------------------
    def get_test_data(self):
        """Return (X, y, sfreq) for the streaming test subject."""
        print(f"  [DataEngine] Loading test subject {self.config.test_subject} …")
        X, y, sfreq, _ = self.get_subject_data(self.config.test_subject)
        if not self._is_riemannian:
            X = self._apply_ea(X)
        return X, y, sfreq

    # ------------------------------------------------------------------
    def _apply_ea(self, X: np.ndarray) -> np.ndarray:
        """
        Euclidean Mean Alignment — He & Wu 2020.

        Supports both 3-D (n, C, T) and 4-D (n, C, T, B) inputs.
        For 4-D, EA is applied independently per frequency band.
        """
        if X.ndim == 3:
            return self._euclidean_alignment(X)
        elif X.ndim == 4:
            # Shape: (n_trials, n_channels, n_times, n_bands)
            aligned = np.stack(
                [self._euclidean_alignment(X[..., b]) for b in range(X.shape[-1])],
                axis=-1,
            )
            return aligned
        else:
            raise ValueError(f"Expected 3-D or 4-D input, got shape {X.shape}")

    @staticmethod
    def _euclidean_alignment(X: np.ndarray) -> np.ndarray:
        """
        Euclidean Mean Alignment (EA) — He & Wu, 2020.

        For each trial x_i ∈ ℝ^{C×T}:
          R̄  = mean covariance across all trials
          x̃_i = R̄^{-1/2} · x_i

        Centres the covariance distribution around the identity,
        reducing cross-subject covariance shift.
        """
        n, c, t  = X.shape
        cov_mean = np.mean([xi @ xi.T / t for xi in X], axis=0)

        eigvals, eigvecs = np.linalg.eigh(cov_mean)
        eigvals    = np.maximum(eigvals, 1e-10)
        R_inv_sqrt = eigvecs @ np.diag(eigvals ** -0.5) @ eigvecs.T

        return np.array([R_inv_sqrt @ xi for xi in X])
