# NeuroStream

<p align="center">
  <img src="assets/neurostream-overview.svg" alt="NeuroStream desktop app overview" width="980">
</p>

**Real-time BCI motor imagery streaming demo for macOS.**

NeuroStream is a desktop app that trains a cross-subject EEG motor imagery classifier on the [Zhou2016](https://doi.org/10.1371/journal.pone.0162657) dataset through [MOABB](https://moabb.neurotechx.com), then replays a held-out subject as a pseudo-online stream. The UI shows prediction confidence, band-power summaries, trial history, and a live confusion matrix in one window.

## Highlights

- Cross-subject pipeline: Euclidean Alignment (EA) plus CSP plus `LDA` or `SVM`
- Desktop UI built with `tkinter`, background training thread, and `root.after()` driven streaming
- Uses an existing local `MNE-zhou-2016/` folder when available; no forced redownload
- All core parameters are exposed in the UI and stored in `BCIConfig`

## What The App Shows

| Area | Purpose |
|---|---|
| Data Folder | Point the app at a local Zhou2016 download |
| Parameters | Tune filter range, epoch window, CSP filters, classifier, and subject split |
| Train and Load | Build the model without freezing the GUI |
| Live Feed | Watch the pseudo-online trial countdown and prediction state |
| Confidence | Compare LEFT vs RIGHT probabilities |
| Band Power | Inspect relative mu and beta power for the current trial |
| Trial History | Track recent hits and misses plus cumulative accuracy |
| Confusion Matrix | See running class-level performance |

## Signal Processing Pipeline

```text
Raw EEG
  -> Bandpass Filter (8-30 Hz)
  -> Epoch (0-3 s post-cue)
  -> Euclidean Alignment
  -> CSP (log-variance features)
  -> LDA or linear SVM
  -> LEFT / RIGHT prediction
```

## Quick Start

### Requirements

- Python 3.9+
- macOS with `tkinter` available in the Python installation

Install dependencies:

```bash
pip install -r requirements.txt
```

### Expected Dataset Layout

```text
/your/data/path/
└── MNE-zhou-2016/
    ├── sub-1/
    ├── sub-2/
    ├── sub-3/
    └── sub-4/
```

If the dataset is not already present, the app can offer a MOABB download for Zhou2016.

### Run

```bash
python main.py
```

### Typical Workflow

1. Paste or browse to the folder that contains `MNE-zhou-2016/`.
2. Adjust the parameters in the left panel if needed.
3. Click `Train & Load`.
4. Click `Start Stream` to replay the held-out subject trial by trial.
5. Use `Pause` or `Stop` while inspecting the live metrics.

## Project Structure

```text
main.py            # App entry point
config.py          # Central parameter dataclass
data_engine.py     # MOABB loading and Euclidean Alignment
model.py           # CSP + LDA/SVM pipeline
streaming.py       # Trial streaming simulator
ui/app_view.py     # Tkinter layout and app state
ui/plots.py        # Custom canvas charts
```

## Default Parameters

| Parameter | Default |
|---|---|
| Bandpass Low | `8.0 Hz` |
| Bandpass High | `30.0 Hz` |
| Epoch Start | `0.0 s` |
| Epoch End | `3.0 s` |
| CSP Filters | `8` |
| Classifier | `LDA` |
| Train Subjects | `1, 2` |
| Test Subject | `3` |

## Dataset

**Zhou2016** is a public EEG motor imagery dataset exposed in MOABB as `moabb.datasets.Zhou2016`.

Reference:

- Zhou, B., Wu, X., Lv, Z., Zhang, L., and Guo, X. (2016). *A Fully Automated Trial Selection Method for Optimization of Motor Imagery Based Brain-Computer Interface.* PLOS ONE.

## Key References

- He, H., and Wu, D. (2020). Transfer learning for brain-computer interfaces: A Euclidean space data alignment approach.
- Blankertz, B., Tomioka, R., Lemm, S., Kawanabe, M., and Muller, K. R. (2008). Optimizing spatial filters for robust EEG single-trial analysis.

## License

MIT. See [LICENSE](LICENSE).
