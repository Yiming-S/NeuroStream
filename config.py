"""
config.py — Centralised parameter store for the BCI pipeline.
All tunable values live here; nothing is hard-coded elsewhere.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class BCIConfig:
    """Every numeric and categorical parameter in one place."""

    # ── Signal processing ──────────────────────────────────────────────

    # Bandpass filter range (Hz)
    f_low: float  = 8.0
    f_high: float = 30.0

    # Epoch window relative to stimulus onset (seconds)
    t_min: float = 0.0
    t_max: float = 3.0

    # ── Feature extraction ─────────────────────────────────────────────

    # Number of spatial filters retained per class (CSP / FBCSP only)
    csp_components: int = 8

    # Pipeline / feature extraction method:
    #   "CSP"    — bandpass → mne.decoding.CSP → LDA/SVM
    #   "FBCSP"  — FilterBankLeftRightImagery (moabb) → FilterBank(CSP) → LDA/SVM
    #   "TS+LDA" — raw epochs → pyriemann Covariances → TangentSpace → LDA
    #   "TS+SVM" — raw epochs → pyriemann Covariances → TangentSpace → SVM
    pipeline_type: str = "CSP"

    # Classifier for CSP / FBCSP pipelines
    clf_type: str = "LDA"

    # FBCSP frequency bands (comma-separated "fmin-fmax" pairs, Hz)
    # Defaults mirror moabb.FilterBankLeftRightImagery
    fb_bands: str = "8-12,12-16,16-20,20-24,24-28,28-32"

    # ── Evaluation protocol ────────────────────────────────────────────

    # "Cross-Subject"  — train on data from multiple subjects, test on a
    #                    held-out subject (different person entirely).
    # "Cross-Session"  — train on earlier sessions of one subject, test on
    #                    a later session of the same subject.
    evaluation_protocol: str = "Cross-Subject"

    # ── Cross-Subject parameters (used when evaluation_protocol == "Cross-Subject") ──

    # Subject IDs used for training
    train_subjects: List[int] = field(default_factory=lambda: [1, 2])

    # Subject ID used for streaming / testing
    test_subject: int = 3

    # ── Cross-Session parameters (used when evaluation_protocol == "Cross-Session") ──

    # The single subject whose sessions are split into train / test
    cross_session_subject: int = 1

    # Session indices used for training (Zhou2016 has sessions 0, 1, 2)
    train_sessions: List[int] = field(default_factory=lambda: [0, 1])

    # Session index used for streaming / testing
    test_session: int = 2

    # ── Data location ──────────────────────────────────────────────────

    # Directory where MNE/MOABB data lives (contains MNE-zhou-2016/ subfolder)
    data_path: str = ""
