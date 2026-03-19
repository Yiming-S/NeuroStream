"""
model.py — CSP + LDA/SVM sklearn Pipeline wrapper.
"""

from typing import Optional

import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from config import BCIConfig


class BCIModel:
    """
    Wraps a two-step sklearn Pipeline:
      1. CSP — maximises variance ratio between the two motor-imagery classes.
         log=True → log-band-power features, more Gaussian, better for LDA.
      2. LDA or SVM classifier.

    Can be rebuilt and retrained at any time when the user updates parameters.
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
            reg=None,
            log=True,
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
        """Fit the pipeline. Returns training accuracy."""
        if self.pipeline is None:
            self.build()
        self.pipeline.fit(X, y)
        self.is_trained = True
        train_acc = self.pipeline.score(X, y)
        print(f"  [BCIModel] Training accuracy: {train_acc:.2%}")
        return train_acc

    # ------------------------------------------------------------------
    def predict(self, X_epoch: np.ndarray) -> int:
        """Predict class for a single epoch (shape: channels × times)."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet.")
        return int(self.pipeline.predict(X_epoch[np.newaxis, ...])[0])

    # ------------------------------------------------------------------
    def predict_proba_single(self, X_epoch: np.ndarray) -> np.ndarray:
        """
        Returns [p_class0, p_class1] for a single epoch.
        Uses predict_proba if available (LDA), otherwise sigmoid of
        decision_function (SVM).
        """
        x = X_epoch[np.newaxis, ...]
        try:
            return self.pipeline.predict_proba(x)[0]
        except Exception:
            pass
        try:
            df = float(self.pipeline.decision_function(x)[0])
            p  = 1.0 / (1.0 + np.exp(-df))
            return np.array([1.0 - p, p])
        except Exception:
            return np.array([0.5, 0.5])
