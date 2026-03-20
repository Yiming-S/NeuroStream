"""
config.py — Centralised parameter store for the BCI pipeline.
All tunable values live here; nothing is hard-coded elsewhere.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class BCIConfig:
    """Every numeric and categorical parameter in one place."""

    # -- Bandpass filter range (Hz) --
    f_low: float  = 8.0
    f_high: float = 30.0

    # -- Epoch window relative to stimulus onset (seconds) --
    t_min: float  = 0.0
    t_max: float  = 3.0

    # -- Common Spatial Patterns: number of spatial filters to retain --
    csp_components: int = 8

    # -- Pipeline / feature extraction method --
    # "CSP"    : bandpass → CSP (mne.decoding.CSP) → LDA/SVM
    # "FBCSP"  : FilterBankLeftRightImagery (moabb) → FilterBank(CSP) → LDA/SVM
    # "TS+LDA" : raw epochs → Covariances (pyriemann) → TangentSpace → LDA
    # "TS+SVM" : raw epochs → Covariances (pyriemann) → TangentSpace → SVM
    # "MDM"    : raw epochs → Covariances (pyriemann) → MDM classifier
    pipeline_type: str = "CSP"

    # -- Classifier for CSP / FBCSP pipelines: "LDA" or "SVM" --
    clf_type: str = "LDA"

    # -- FBCSP frequency bands, each entry is [fmin, fmax] in Hz --
    # Default mirrors moabb.FilterBankLeftRightImagery defaults.
    fb_bands: str = "8-12,12-16,16-20,20-24,24-28,28-32"

    # -- Subjects used to pre-train the cross-subject model --
    train_subjects: List[int] = field(default_factory=lambda: [1, 2])

    # -- Subject whose data is streamed at demo time --
    test_subject: int = 3

    # -- Directory where MNE/MOABB data lives (contains MNE-zhou-2016/ subfolder) --
    data_path: str = ""
