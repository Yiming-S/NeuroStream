"""
model.py — sklearn Pipeline factory supporting multiple BCI feature-extraction
           and classification strategies.

Supported pipeline_type values (BCIConfig.pipeline_type):

  "CSP"    mne.decoding.CSP + LDA/SVM
           Standard single-band approach used as the baseline in most MOABB
           benchmarks (Jayaram & Barachant 2018).

  "FBCSP"  moabb.pipelines.utils.FilterBank(CSP) + LDA/SVM
           Applies CSP independently on each MOABB filter-bank frequency band
           and concatenates the resulting log-power features.
           Reference: Ang et al. (2012) FBCSP.

  "TS+LDA" pyriemann Covariances → TangentSpace → LDA
  "TS+SVM" pyriemann Covariances → TangentSpace → SVM
           Project the Riemannian manifold of SPD matrices to the tangent
           space at the Fréchet mean, then classify in Euclidean space.
           Reference: Barachant et al. (2013), MOABB benchmark top-performer.

  "MDM"    pyriemann Covariances → MDM (Minimum Distance to Mean)
           Pure Riemannian classifier; no hyperparameters.
           Reference: Barachant et al. (2012).
"""

from typing import Optional

import numpy as np
from mne.decoding import CSP
from moabb.pipelines.utils import FilterBank
from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from config import BCIConfig


class BCIModel:
    """
    Constructs and wraps the full sklearn-compatible pipeline for the
    chosen pipeline_type.  All variants expose the same .train() /
    .predict() / .predict_proba_single() interface so the rest of the
    application is pipeline-agnostic.
    """

    def __init__(self, config: BCIConfig):
        self.config = config
        self.pipeline: Optional[Pipeline] = None
        self.is_trained: bool = False

    # ------------------------------------------------------------------
    def build(self) -> None:
        """Construct (or reconstruct) the sklearn pipeline from current config."""
        pt = self.config.pipeline_type

        if pt == "CSP":
            self.pipeline = self._build_csp()
        elif pt == "FBCSP":
            self.pipeline = self._build_fbcsp()
        elif pt == "TS+LDA":
            self.pipeline = self._build_tangentspace(clf="LDA")
        elif pt == "TS+SVM":
            self.pipeline = self._build_tangentspace(clf="SVM")
        elif pt == "MDM":
            self.pipeline = self._build_mdm()
        else:
            raise ValueError(
                f"Unknown pipeline_type '{pt}'. "
                "Choose: CSP | FBCSP | TS+LDA | TS+SVM | MDM"
            )

        self.is_trained = False
        print(f"  [BCIModel] Pipeline built: {pt}")

    # ------------------------------------------------------------------
    # Private pipeline builders
    # ------------------------------------------------------------------

    def _build_csp(self) -> Pipeline:
        """
        mne.decoding.CSP + LDA or SVM.
        Input: (n_trials, n_channels, n_times)
        """
        csp = CSP(
            n_components=self.config.csp_components,
            reg=None,
            log=True,          # log-band-power → more Gaussian, better for LDA
            norm_trace=False,
        )
        clf = self._make_clf(self.config.clf_type)
        return Pipeline([("csp", csp), ("clf", clf)])

    def _build_fbcsp(self) -> Pipeline:
        """
        moabb.pipelines.utils.FilterBank(CSP) + LDA or SVM.
        Input: (n_trials, n_channels, n_times, n_bands)  — from FilterBankLeftRightImagery.

        FilterBank applies the inner estimator independently to each band
        (axis=-1) and concatenates the resulting feature vectors.
        """
        csp = CSP(
            n_components=self.config.csp_components,
            reg=None,
            log=True,
            norm_trace=False,
        )
        fb  = FilterBank(estimator=csp, flatten=True)
        clf = self._make_clf(self.config.clf_type)
        return Pipeline([("fb_csp", fb), ("clf", clf)])

    def _build_tangentspace(self, clf: str) -> Pipeline:
        """
        pyriemann Covariances → TangentSpace → LDA or SVM.
        Input: (n_trials, n_channels, n_times)

        Covariances("oas") uses Oracle Approximating Shrinkage — robust
        estimation even with limited samples (recommended in MOABB benchmarks).
        """
        cov = Covariances(estimator="oas")
        ts  = TangentSpace(metric="riemann")
        return Pipeline([("cov", cov), ("ts", ts), ("clf", self._make_clf(clf))])

    def _build_mdm(self) -> Pipeline:
        """
        pyriemann Covariances → MDM.
        Input: (n_trials, n_channels, n_times)

        MDM classifies by Riemannian distance to class means — zero
        hyperparameters and excellent cross-subject generalisation.
        """
        cov = Covariances(estimator="oas")
        mdm = MDM(metric={"mean": "riemann", "distance": "riemann"})
        return Pipeline([("cov", cov), ("clf", mdm)])

    # ------------------------------------------------------------------
    @staticmethod
    def _make_clf(clf_type: str):
        if clf_type == "LDA":
            return LinearDiscriminantAnalysis()
        elif clf_type == "SVM":
            return SVC(kernel="linear", C=1.0, probability=False)
        else:
            raise ValueError(f"Unknown clf_type '{clf_type}'. Use 'LDA' or 'SVM'.")

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
        """Predict class for a single epoch."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet.")
        return int(self.pipeline.predict(X_epoch[np.newaxis, ...])[0])

    # ------------------------------------------------------------------
    def predict_proba_single(self, X_epoch: np.ndarray) -> np.ndarray:
        """
        Returns [p_class0, p_class1] for a single epoch.
        Tries predict_proba first (LDA, MDM), then sigmoid of
        decision_function (linear SVM).
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
