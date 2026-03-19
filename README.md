# NeuroStream

**Real-Time BCI Motor Imagery Streaming Demo**

A macOS desktop application that trains a cross-subject Brain-Computer Interface (BCI) model on EEG motor imagery data, then simulates live streaming predictions — all within a single Python file and a clean light-theme GUI.

---

## What It Does

1. **Train** — loads EEG data for subjects 1 and 2 from the [Zhou2016](https://doi.org/10.1371/journal.pone.0162657) dataset via [MOABB](https://moabb.neurotechx.com), applies Euclidean Mean Alignment (EA) for cross-subject domain adaptation, and fits a CSP + LDA/SVM pipeline.
2. **Stream** — replays subject 3's trials one at a time, showing a simulated EEG collection window with a live countdown, then displays the classifier's prediction vs. the ground truth.
3. **Visualise** — four real-time panels update after every trial:

| Panel | What it shows |
|---|---|
| Classifier Confidence | LEFT / RIGHT split bar with probabilities |
| Band Power | μ (8–12 Hz) and β (13–30 Hz) relative power bars |
| Trial History | Per-trial correct/incorrect bars + cumulative accuracy line |
| Confusion Matrix | 2×2 colour-coded LEFT vs. RIGHT classification counts |

---

## Screenshots

> Train subjects, select your data folder, and click **Train & Load** — the model is ready in under a minute on a standard laptop.

---

## Signal Processing Pipeline

```
Raw EEG  →  Bandpass Filter (8–30 Hz)  →  Epoch (0–3 s post-cue)
         →  Euclidean Mean Alignment (EA)
         →  CSP (8 spatial filters, log-variance features)
         →  LDA  (or SVM, selectable in UI)
         →  LEFT / RIGHT prediction
```

**Key choices:**
- **EA** (He & Wu, 2020) — aligns each subject's covariance distribution to the identity, substantially reducing cross-subject shift without requiring a calibration session.
- **CSP log-variance** — produces near-Gaussian features that benefit LDA's Gaussian class-conditional assumption.
- **No hard-coded values** — every parameter (filter range, epoch window, CSP components, classifier type, train/test subjects) is exposed in the UI and stored in `BCIConfig`.

---

## Requirements

- Python 3.9+
- macOS (tested on macOS 14 Sonoma; tkinter is bundled with the official Python installer)

Install dependencies:

```bash
pip install -r requirements.txt
```

| Package | Purpose |
|---|---|
| `numpy` | Numerical arrays |
| `scipy` | Signal processing utilities |
| `scikit-learn` | CSP, LDA, SVM, Pipeline |
| `mne` | EEG data structures and decoding |
| `moabb` | Zhou2016 dataset loading and LeftRightImagery paradigm |
| `matplotlib` | Required by MNE internals (Agg backend only — no window opened) |

---

## Usage

### 1. Get the data

The app supports **local data** — if you already have the Zhou2016 dataset downloaded, point the app to that folder and it will skip the download entirely. Your existing files are never deleted or overwritten.

Expected folder layout:
```
/your/data/path/
└── MNE-zhou-2016/
    ├── sub-1/
    ├── sub-2/
    ├── sub-3/
    └── sub-4/
```

If you do not have the data, the app will ask whether to download it (~200 MB via MOABB).

### 2. Run

```bash
python main.py
```

### 3. Workflow in the GUI

1. Set the **Data Folder** path (paste or Browse).
2. Adjust parameters in the scrollable **Parameters** panel (optional).
3. Click **Train & Load** — training completes in the background; the GUI stays responsive.
4. Click **Start Stream** to begin the trial-by-trial simulation.
5. Use **Pause / Resume** and **Stop** as needed.

---

## Architecture

```
BCIConfig           # Dataclass — all parameters in one place
DataEngine          # MOABB loading, EA alignment, epoch arrays
BCIModel            # sklearn Pipeline: CSP → LDA/SVM
StreamingSimulator  # Generator yielding (epoch, label) per trial
AppUI               # tkinter GUI, root.after() driven — never blocks
```

The GUI **never blocks the main thread**:
- Training runs in a `daemon` thread and posts results back via `root.after(0, callback)`.
- Streaming is driven entirely by `root.after()` callbacks — no `time.sleep()`, no threads.

---

## Parameters

| Parameter | Default | Description |
|---|---|---|
| Bandpass Low | 8.0 Hz | Lower edge of the bandpass filter |
| Bandpass High | 30.0 Hz | Upper edge (covers μ + β motor rhythms) |
| Epoch Start | 0.0 s | Trial window start relative to cue onset |
| Epoch End | 3.0 s | Trial window end |
| CSP Filters | 8 | Number of spatial filters to retain |
| Classifier | LDA | `LDA` or `SVM` (linear kernel) |
| Train Subjects | 1, 2 | Comma-separated subject IDs for training |
| Test Subject | 3 | Subject ID for streaming simulation |

---

## Dataset

**Zhou2016** — publicly available EEG motor imagery dataset.

> Zhou, B., Wu, X., Lv, Z., Zhang, L., & Guo, X. (2016).
> *A Fully Automated Trial Selection Method for Optimization of Motor Imagery Based Brain-Computer Interface.*
> PLOS ONE. https://doi.org/10.1371/journal.pone.0162657

Accessed via [MOABB](https://moabb.neurotechx.com) (`moabb.datasets.Zhou2016`).

---

## Key References

- He, H., & Wu, D. (2020). Transfer learning for brain-computer interfaces: A Euclidean space data alignment approach. *IEEE Transactions on Biomedical Engineering*, 67(2), 399-410.
- Blankertz, B., Tomioka, R., Lemm, S., Kawanabe, M., & Muller, K. R. (2008). Optimizing spatial filters for robust EEG single-trial analysis. *IEEE Signal Processing Magazine*, 25(1), 41-56.

---

## License

MIT — see [LICENSE](LICENSE).
