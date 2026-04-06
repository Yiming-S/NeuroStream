"""
ui/plots.py — Standalone canvas drawing functions.

Each function takes explicit canvas and data arguments so it can be
called from anywhere without coupling to AppUI internals.
"""

import tkinter as tk

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# Classifier Confidence
# ══════════════════════════════════════════════════════════════════════════════

def draw_confidence(
    canvas: tk.Canvas,
    proba: np.ndarray,
    accent: str,
    fg: str,
) -> None:
    """Horizontal split bar: LEFT probability | RIGHT probability."""
    c = canvas
    c.delete("all")
    c.update_idletasks()
    W, H = c.winfo_width(), c.winfo_height()
    if W < 10 or H < 10:
        return

    p_left  = float(proba[0])
    p_right = float(proba[1])
    split   = max(4, min(W - 4, int(W * p_left)))
    pad     = 3

    c.create_rectangle(pad, pad, split, H - pad, fill=accent, outline="")
    if split < W - pad:
        c.create_rectangle(split, pad, W - pad, H - pad, fill="#d0d7de", outline="")

    lx = (pad + split) // 2
    rx = (split + W - pad) // 2
    lc = "white" if p_left  > 0.15 else fg
    rc = fg      if p_right > 0.15 else "white"
    c.create_text(lx, H // 2, text=f"LEFT  {p_left:.0%}",
                  fill=lc, font=("Helvetica Neue", 9, "bold"), anchor="center")
    c.create_text(rx, H // 2, text=f"RIGHT  {p_right:.0%}",
                  fill=rc, font=("Helvetica Neue", 9, "bold"), anchor="center")


# ══════════════════════════════════════════════════════════════════════════════
# Band Power
# ══════════════════════════════════════════════════════════════════════════════

def band_power(epoch: np.ndarray, sfreq: float, fmin: float, fmax: float) -> float:
    """Band power as fraction of total spectral power (0–1)."""
    if epoch.ndim == 3:
        return 0.0
    n_times = epoch.shape[1]
    freqs   = np.fft.rfftfreq(n_times, 1.0 / sfreq)
    ps      = np.abs(np.fft.rfft(epoch, axis=1)) ** 2   # (channels, freqs)
    mask    = (freqs >= fmin) & (freqs <= fmax)
    b       = float(ps[:, mask].mean()) if mask.any() else 0.0
    total   = float(ps.mean()) + 1e-10
    return b / total


def draw_band_power(
    canvas: tk.Canvas,
    epoch: np.ndarray,
    sfreq: float,
    accent: str,
    green: str,
    fg: str,
) -> None:
    """Two vertical bars: μ (8–12 Hz) and β (13–30 Hz) band power."""
    c = canvas
    c.delete("all")
    c.update_idletasks()
    W, H = c.winfo_width(), c.winfo_height()
    if W < 10 or H < 10:
        return

    if epoch.ndim == 3:
        c.create_text(
            W // 2, H // 2,
            text="N/A (Filter Bank)",
            fill="#57606a",
            font=("Helvetica Neue", 9, "bold"),
            anchor="center",
        )
        return
    mu_frac   = band_power(epoch, sfreq, 8,  12)
    beta_frac = band_power(epoch, sfreq, 13, 30)
    bands = [
        ("μ  8–12 Hz",  mu_frac,   accent),
        ("β  13–30 Hz", beta_frac, green),
    ]

    label_h = 16
    plot_h  = H - label_h - 8
    n       = len(bands)
    gap     = 10
    bar_w   = (W - (n + 1) * gap) // n

    for i, (label, frac, color) in enumerate(bands):
        x0 = gap + i * (bar_w + gap)
        x1 = x0 + bar_w
        c.create_rectangle(x0, 4, x1, H - label_h - 4, fill="#d0d7de", outline="")
        fill_h = max(2, int(frac * plot_h))
        c.create_rectangle(x0, H - label_h - 4 - fill_h, x1, H - label_h - 4,
                            fill=color, outline="")
        c.create_text((x0 + x1) // 2, H - label_h // 2 - 2,
                      text=f"{label}  {frac:.0%}",
                      fill=fg, font=("Helvetica Neue", 7), anchor="center")


# ══════════════════════════════════════════════════════════════════════════════
# Confusion Matrix
# ══════════════════════════════════════════════════════════════════════════════

def draw_confusion_matrix(
    canvas: tk.Canvas,
    conf_matrix: list,
    fg: str,
    green: str,
    red: str,
) -> None:
    """2×2 grid — rows = Actual, cols = Predicted. Colour intensity ∝ count."""
    c = canvas
    c.delete("all")
    c.update_idletasks()
    W, H = c.winfo_width(), c.winfo_height()
    if W < 10 or H < 10:
        return

    max_val = max(v for row in conf_matrix for v in row)
    if max_val == 0:
        return

    label_w, label_h = 20, 16
    cell_w = (W - label_w) // 2
    cell_h = (H - label_h) // 2

    # Column headers (Predicted)
    for j, name in enumerate(["LEFT", "RIGHT"]):
        x = label_w + j * cell_w + cell_w // 2
        c.create_text(x, label_h // 2, text=name, fill="#57606a",
                      font=("Helvetica Neue", 7), anchor="center")

    # Row headers (Actual) + cells
    for i, name in enumerate(["LEFT", "RIGHT"]):
        y = label_h + i * cell_h + cell_h // 2
        c.create_text(label_w // 2, y, text=name, fill="#57606a",
                      font=("Helvetica Neue", 7), anchor="center")
        for j in range(2):
            val   = conf_matrix[i][j]
            alpha = val / max_val
            x0    = label_w + j * cell_w
            y0    = label_h + i * cell_h
            x1, y1 = x0 + cell_w - 1, y0 + cell_h - 1

            if i == j:   # diagonal → correct → green tint
                r = int(234 + (26  - 234) * alpha)
                g = int(255 + (127 - 255) * alpha)
                b = int(240 + (55  - 240) * alpha)
            else:        # off-diagonal → wrong → red tint
                r = int(255 + (207 - 255) * alpha)
                g = int(240 + (34  - 240) * alpha)
                b = int(240 + (34  - 240) * alpha)

            fill   = f"#{r:02x}{g:02x}{b:02x}"
            text_c = "white" if alpha > 0.5 else fg
            c.create_rectangle(x0, y0, x1, y1, fill=fill, outline="#d0d7de")
            c.create_text((x0 + x1) // 2, (y0 + y1) // 2, text=str(val),
                          fill=text_c, font=("Helvetica Neue", 10, "bold"),
                          anchor="center")


# ══════════════════════════════════════════════════════════════════════════════
# Trial History Chart
# ══════════════════════════════════════════════════════════════════════════════

def draw_trial_chart(
    canvas: tk.Canvas,
    history: list,
    max_bars: int,
    green: str,
    red: str,
    accent: str,
    fg: str,
) -> None:
    """
    Bar chart of recent trial outcomes + cumulative accuracy line.

    Layout
    ------
      Left 30 px → Y-axis labels (0% … 100%)
      Remaining  → plot area
    Bars   : green = correct, red = incorrect.
    Line   : cumulative accuracy — matches the Online Accuracy label exactly.
    """
    c = canvas
    c.delete("all")

    h = history[-max_bars:]
    n = len(h)
    if n == 0:
        return

    c.update_idletasks()
    W = c.winfo_width()
    H = c.winfo_height()
    if W < 10 or H < 10:
        return

    AXIS_W  = 30
    TOP_PAD = 4
    BOT_PAD = 4
    plot_x0 = AXIS_W
    plot_y0 = TOP_PAD
    plot_y1 = H - BOT_PAD
    plot_h  = plot_y1 - plot_y0

    def pct_to_y(p: float) -> int:
        return plot_y1 - int(p * plot_h)

    # Y-axis + grid lines
    for pct in (0.0, 0.25, 0.50, 0.75, 1.0):
        y = pct_to_y(pct)
        c.create_line(plot_x0, y, W, y, fill="#d0d7de", dash=(3, 5))
        c.create_text(AXIS_W - 3, y, text=f"{int(pct * 100)}%",
                      anchor="e", fill="#57606a", font=("Helvetica Neue", 7))

    # Bars
    plot_w  = W - plot_x0 - 2
    bar_w   = max(3, plot_w // max_bars)
    bar_gap = max(1, bar_w // 5)

    for i, correct in enumerate(h):
        x1 = plot_x0 + i * (bar_w + bar_gap)
        x2 = x1 + bar_w
        c.create_rectangle(x1, plot_y0, x2, plot_y1,
                            fill=green if correct else red, outline="")

    # Cumulative accuracy line
    points = []
    running_correct = 0
    for i, correct in enumerate(h):
        running_correct += int(correct)
        cum_acc = running_correct / (i + 1)
        x_mid   = plot_x0 + i * (bar_w + bar_gap) + bar_w // 2
        y       = pct_to_y(cum_acc)
        points.extend([x_mid, y])

    if len(points) >= 4:
        c.create_line(*points, fill=accent, width=2, smooth=True)


# ══════════════════════════════════════════════════════════════════════════════
# Accuracy Curve (Summary View)
# ══════════════════════════════════════════════════════════════════════════════

def draw_accuracy_curve(
    canvas: tk.Canvas,
    history: list,
    accent: str,
    green: str,
    fg: str,
) -> None:
    """
    Full-canvas cumulative accuracy curve with axis labels.
    Used in the post-demo summary view.
    """
    c = canvas
    c.delete("all")
    c.update_idletasks()
    W = c.winfo_width()
    H = c.winfo_height()
    if W < 10 or H < 10 or not history:
        return

    AXIS_W  = 38
    AXIS_H  = 18
    TOP_PAD = 6
    plot_x0 = AXIS_W
    plot_x1 = W - 6
    plot_y0 = TOP_PAD
    plot_y1 = H - AXIS_H
    plot_w  = plot_x1 - plot_x0
    plot_h  = plot_y1 - plot_y0

    def pct_to_y(p: float) -> int:
        return plot_y1 - int(p * plot_h)

    def trial_to_x(i: int) -> int:
        return plot_x0 + int(i / max(len(history) - 1, 1) * plot_w)

    # Grid lines and Y-axis labels
    for pct in (0.0, 0.25, 0.50, 0.75, 1.0):
        y = pct_to_y(pct)
        c.create_line(plot_x0, y, plot_x1, y, fill="#d0d7de", dash=(3, 5))
        c.create_text(AXIS_W - 3, y, text=f"{int(pct * 100)}%",
                      anchor="e", fill="#57606a", font=("Helvetica Neue", 7))

    # X-axis labels
    n = len(history)
    for frac in (0.0, 0.5, 1.0):
        idx = int(frac * (n - 1)) if n > 1 else 0
        x = trial_to_x(idx)
        c.create_text(x, plot_y1 + 10, text=str(idx + 1),
                      anchor="center", fill="#57606a",
                      font=("Helvetica Neue", 7))

    # Axis title
    c.create_text((plot_x0 + plot_x1) // 2, H - 2, text="Trial",
                  anchor="s", fill="#57606a", font=("Helvetica Neue", 7))

    # 50% chance line
    y50 = pct_to_y(0.5)
    c.create_line(plot_x0, y50, plot_x1, y50, fill="#cf222e",
                  dash=(6, 4), width=1)
    c.create_text(plot_x1 + 2, y50, text="chance", anchor="w",
                  fill="#cf222e", font=("Helvetica Neue", 6))

    # Cumulative accuracy curve
    points = []
    running_correct = 0
    for i, correct in enumerate(history):
        running_correct += int(correct)
        cum_acc = running_correct / (i + 1)
        points.extend([trial_to_x(i), pct_to_y(cum_acc)])

    if len(points) >= 4:
        c.create_line(*points, fill=accent, width=2, smooth=True)

    # Final accuracy dot
    if points:
        final_x, final_y = points[-2], points[-1]
        final_acc = running_correct / n
        c.create_oval(final_x - 4, final_y - 4, final_x + 4, final_y + 4,
                      fill=green, outline="")
        c.create_text(final_x, final_y - 10, text=f"{final_acc:.1%}",
                      fill=green, font=("Helvetica Neue", 9, "bold"),
                      anchor="s")


# ══════════════════════════════════════════════════════════════════════════════
# Progressive Accuracy Curve (Summary View)
# ══════════════════════════════════════════════════════════════════════════════

def draw_progressive_accuracy(
    canvas: tk.Canvas,
    prog_accuracy: dict,
    time_labels: list,
    accent: str,
    green: str,
    red: str,
    fg: str,
) -> None:
    """
    Accuracy at each progressive time window across all trials.

    prog_accuracy : {n_samples: [correct, total], ...}
    time_labels   : corresponding seconds for each n_samples key
    """
    c = canvas
    c.delete("all")
    c.update_idletasks()
    W = c.winfo_width()
    H = c.winfo_height()
    if W < 10 or H < 10:
        return

    # Build sorted (time_s, accuracy) pairs
    sorted_ns = sorted(prog_accuracy.keys())
    if not sorted_ns or not time_labels:
        return
    ns_to_time = {}
    for ns, t in zip(sorted(prog_accuracy.keys()), sorted(time_labels)):
        ns_to_time[ns] = t

    points_data = []
    for ns in sorted_ns:
        correct, total = prog_accuracy[ns]
        if total == 0:
            continue
        points_data.append((ns_to_time.get(ns, 0), correct / total))

    if not points_data:
        c.create_text(W // 2, H // 2, text="No progressive data",
                      fill="#57606a", font=("Helvetica Neue", 9),
                      anchor="center")
        return

    # Layout
    AXIS_W = 38
    AXIS_H = 20
    TOP_PAD = 10
    plot_x0 = AXIS_W
    plot_x1 = W - 10
    plot_y0 = TOP_PAD
    plot_y1 = H - AXIS_H
    plot_w = plot_x1 - plot_x0
    plot_h = plot_y1 - plot_y0

    t_min_val = points_data[0][0]
    t_max_val = points_data[-1][0]
    t_range = t_max_val - t_min_val if t_max_val > t_min_val else 1.0

    def t_to_x(t: float) -> int:
        return plot_x0 + int((t - t_min_val) / t_range * plot_w)

    def acc_to_y(a: float) -> int:
        return plot_y1 - int(a * plot_h)

    # Grid + Y axis
    for pct in (0.0, 0.25, 0.50, 0.75, 1.0):
        y = acc_to_y(pct)
        c.create_line(plot_x0, y, plot_x1, y, fill="#d0d7de", dash=(3, 5))
        c.create_text(AXIS_W - 3, y, text=f"{int(pct * 100)}%",
                      anchor="e", fill="#57606a", font=("Helvetica Neue", 7))

    # X axis labels
    for t, _ in points_data:
        x = t_to_x(t)
        c.create_text(x, plot_y1 + 10, text=f"{t:.1f}s",
                      anchor="center", fill="#57606a",
                      font=("Helvetica Neue", 7))

    # Axis title
    c.create_text((plot_x0 + plot_x1) // 2, H - 2,
                  text="EEG window length",
                  anchor="s", fill="#57606a", font=("Helvetica Neue", 7))

    # 50% chance line
    y50 = acc_to_y(0.5)
    c.create_line(plot_x0, y50, plot_x1, y50, fill=red,
                  dash=(6, 4), width=1)

    # Accuracy curve
    line_points = []
    for t, acc in points_data:
        line_points.extend([t_to_x(t), acc_to_y(acc)])

    if len(line_points) >= 4:
        c.create_line(*line_points, fill=accent, width=2, smooth=True)

    # Dots + labels
    for t, acc in points_data:
        x, y = t_to_x(t), acc_to_y(acc)
        c.create_oval(x - 3, y - 3, x + 3, y + 3, fill=accent, outline="")

    # Label first and last
    if points_data:
        t0, a0 = points_data[0]
        c.create_text(t_to_x(t0), acc_to_y(a0) - 8, text=f"{a0:.0%}",
                      fill="#57606a", font=("Helvetica Neue", 7, "bold"),
                      anchor="s")
        tN, aN = points_data[-1]
        c.create_text(t_to_x(tN), acc_to_y(aN) - 8, text=f"{aN:.0%}",
                      fill=green, font=("Helvetica Neue", 8, "bold"),
                      anchor="s")
