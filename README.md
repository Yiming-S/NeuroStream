<p align="center">
  <img src="ui/img/neurostream.svg" alt="NeuroStream Logo" width="120">
</p>

<h1 align="center">NeuroStream</h1>

<p align="center"><strong>Real-time BCI motor imagery decoding with progressive prediction</strong></p>

![NeuroStream overview](assets/neurostream-overview.svg)

NeuroStream is a desktop application that trains cross-subject (or cross-session) EEG motor imagery classifiers on the [Zhou2016](https://doi.org/10.1371/journal.pone.0162657) dataset via [MOABB](https://moabb.neurotechx.com), then replays a held-out subject as a pseudo-online stream with **progressive prediction** — the model begins predicting as soon as 0.5 s of EEG is available and continuously refines its output as more data arrives within each trial.

## How It Works

### Traditional approach

In a conventional offline evaluation, the entire EEG epoch (e.g. 3 s) is collected first, then classified once. The user sees nothing until the trial is over.

### Progressive prediction in NeuroStream

NeuroStream simulates what a real-time BCI would do: it starts predicting **early** and **updates continuously**.

During training, the system builds **one independent classifier per time checkpoint** (e.g. at 0.5 s, 1.0 s, 1.5 s, ... up to the full 3.0 s window). Each sub-model is trained on epochs truncated to its corresponding length. During streaming, as the simulated EEG collection progresses, NeuroStream feeds the available data to the matching sub-model and refreshes the prediction, confidence bar, and spectral features in real time:

```text
Trial starts
  |
  |-- 0.5 s  ->  0.5 s model predicts  ->  UI updates (tentative)
  |-- 1.0 s  ->  1.0 s model predicts  ->  UI updates (tentative)
  |-- 1.5 s  ->  1.5 s model predicts  ->  UI updates (tentative)
  |-- 2.0 s  ->  2.0 s model predicts  ->  UI updates (tentative)
  |-- 2.5 s  ->  2.5 s model predicts  ->  UI updates (tentative)
  |-- 3.0 s  ->  full model predicts   ->  FINAL result, accuracy updated
```

The confidence bar and band power visualisations transition smoothly between updates (120 ms ease-out tween) rather than jumping, giving the feel of a continuously flowing data stream.

### Why 0.5 s?

The default update step of 0.5 s is grounded in established online BCI practice:

- The Graz BCI group uses a **1 s sliding window advanced in 0.5 s steps** as the standard for online motor imagery feedback (Muller-Putz et al., 2010; Pfurtscheller & Neuper, 2001).
- Townsend et al. (2004) demonstrated that continuous sliding-window classification at sub-second intervals is feasible for asynchronous BCI control.
- Wolpaw et al. (2002) established that feedback latency within roughly 0.5–1.0 s is the practical upper bound for responsive BCI interaction.

Shorter steps (< 0.25 s) yield too few samples for stable covariance estimation in CSP; longer steps (> 1.0 s) reduce the number of visible updates per trial, weakening the progressive effect. The step is user-adjustable from 0.25 s to 1.0 s in the Advanced Parameters panel.

## Evaluation Protocols

NeuroStream supports two evaluation modes:

| Protocol | Training data | Test data | Use case |
| --- | --- | --- | --- |
| **Cross-Subject** | All sessions of subjects 1, 2 | All sessions of subject 3 | Generalisation across individuals |
| **Cross-Session** | Sessions 0, 1 of a single subject | Session 2 of the same subject | Within-subject session transfer |

Both protocols apply **Euclidean Alignment** (He & Wu, 2020) to reduce covariance shift between data batches (for CSP and FBCSP pipelines).

## Pipelines

All pipelines are implemented using MOABB and pyriemann building blocks.

### CSP

```text
Raw EEG
  -> Bandpass filter (configurable, default 8-30 Hz)
  -> Euclidean Alignment (He & Wu 2020)
  -> mne.decoding.CSP  (log-variance features)
  -> LDA or linear SVM
```

Classic MOABB baseline — Jayaram & Barachant (2018).

### FBCSP

```text
Raw EEG
  -> moabb.paradigms.FilterBankLeftRightImagery
     (configurable bands, default: 8-12, 12-16, 16-20, 20-24, 24-28, 28-32 Hz)
  -> Euclidean Alignment per band
  -> moabb.pipelines.utils.FilterBank(CSP)  — CSP applied per band, features concatenated
  -> LDA or linear SVM
```

Ang et al. (2012).

### TS+LDA / TS+SVM

```text
Raw EEG
  -> Bandpass filter
  -> pyriemann.estimation.Covariances (OAS estimator)
  -> pyriemann.tangentspace.TangentSpace (Riemannian metric)
  -> LDA or linear SVM
```

Barachant et al. (2013). Top-performing family in MOABB motor imagery benchmarks. An MDM (Minimum Distance to Mean) classifier is also available as a pure Riemannian alternative.

## Quick Start

### Requirements

- Python 3.9+
- macOS with `tkinter` available

### Run (recommended)

```bash
bash run.sh
```

`run.sh` automatically creates a virtual environment, installs all dependencies, and launches the app. Safe to run multiple times — it skips setup if the environment already exists.

### Manual setup (alternative)

```bash
pip install -r requirements.txt
python main.py
```

### Dataset Layout

```text
/your/data/path/
└── MNE-zhou-2016/
    ├── sub-1/
    ├── sub-2/
    ├── sub-3/
    └── sub-4/
```

If not present, MOABB can download it automatically on first run.

### Typical Workflow

1. Paste or browse to the folder containing `MNE-zhou-2016/`.
2. Select **Feature Extraction** method and **Classifier**.
3. Click **Train & Load** (training takes ~1 min).
4. Click **Start Stream** to replay the held-out subject trial by trial.
5. Watch predictions refine in real time as each trial's EEG window fills.
6. After all trials, a **Summary View** shows final accuracy and the cumulative accuracy curve.

## Project Structure

```text
NeuroStream/
├── main.py            Entry point (window init + mainloop)
├── config.py          BCIConfig dataclass — all tunable parameters
├── data_engine.py     MOABB data loading, paradigm selection, EA alignment
├── model.py           Pipeline factory with multi-window progressive support
├── streaming.py       Trial-by-trial streaming simulator
├── ui/
│   ├── app_view.py    Tkinter layout, state machine, training/streaming control
│   ├── plots.py       Canvas drawing functions (confidence, band power, charts)
│   └── widgets.py     Reusable UI components (tooltip, collapsible section, phase indicator)
├── requirements.txt
└── run.sh             One-command launcher (venv + deps + run)
```

## Default Parameters

| Parameter | Default |
| --- | --- |
| Feature Extraction | `CSP` |
| Classifier | `LDA` |
| Bandpass | `8.0 - 30.0 Hz` |
| Epoch Window | `0.0 - 3.0 s` |
| Progressive Step | `0.5 s` |
| CSP Filters | `8` |
| FBCSP Bands | `8-12, 12-16, 16-20, 20-24, 24-28, 28-32 Hz` |
| Evaluation Protocol | `Cross-Subject` |
| Train Subjects | `1, 2` |
| Test Subject | `3` |

## References

- **Zhou2016 dataset**: Zhou et al. (2016). *A Fully Automated Trial Selection Method for Optimization of Motor Imagery Based BCI.* PLOS ONE.
- **MOABB**: Jayaram & Barachant (2018). *MOABB: trustworthy algorithm benchmarking for BCIs.* J. Neural Eng.
- **Euclidean Alignment**: He & Wu (2020). *Transfer learning for BCIs: A Euclidean space data alignment approach.* IEEE Trans. Biomed. Eng.
- **FBCSP**: Ang et al. (2012). *Filter bank common spatial pattern algorithm on BCI competition IV.* Front. Hum. Neurosci.
- **Tangent Space / Riemannian geometry**: Barachant et al. (2013). *Classification of covariance matrices using a Riemannian-based kernel.* Signal Processing.
- **CSP**: Blankertz et al. (2008). *Optimizing spatial filters for robust EEG single-trial analysis.* IEEE Signal Process. Mag.
- **BCI feedback principles**: Wolpaw et al. (2002). *Brain-computer interfaces for communication and control.* Clin. Neurophysiol.
- **Online MI feedback timing**: Pfurtscheller & Neuper (2001). *Motor imagery and direct brain-computer communication.* Proc. IEEE.
- **Sliding-window MI feedback**: Muller-Putz et al. (2010). *Temporal coding of brain patterns for direct limb control in humans.* Front. Neurosci.
- **Continuous asynchronous BCI**: Townsend, Graimann & Pfurtscheller (2004). *Continuous EEG classification during motor imagery — simulation of an asynchronous BCI.* IEEE Trans. Neural Syst. Rehabil. Eng.

## License

MIT. See [LICENSE](LICENSE).
