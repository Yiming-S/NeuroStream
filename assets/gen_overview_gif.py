"""
Generate an animated GIF illustrating NeuroStream's progressive prediction.
Run: python assets/gen_overview_gif.py
Output: assets/neurostream-overview.gif
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math

OUT = Path(__file__).parent / "neurostream-overview.gif"
W, H = 720, 380
BG = "#f6f8fa"
FG = "#24292f"
ACCENT = "#0969da"
GREEN = "#1a7f37"
RED = "#cf222e"
GREY = "#8c959f"
LIGHT = "#d0d7de"
WHITE = "#ffffff"
YELLOW = "#9a6700"

# Progressive prediction data: (time_s, p_left, pred_label)
STEPS = [
    (0.0,  0.50, None),
    (0.5,  0.55, "LEFT"),
    (1.0,  0.48, "RIGHT"),
    (1.5,  0.40, "RIGHT"),
    (2.0,  0.35, "RIGHT"),
    (2.5,  0.30, "RIGHT"),
    (3.0,  0.24, "RIGHT"),
]
TRUE_LABEL = "RIGHT"
TOTAL_S = 3.0


def get_font(size, bold=False):
    names = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSText.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


font_title = get_font(22)
font_big = get_font(28)
font_med = get_font(16)
font_sm = get_font(13)
font_xs = get_font(11)


def ease_out(t):
    return 1.0 - (1.0 - t) ** 2


def lerp(a, b, t):
    return a + (b - a) * t


def draw_frame(step_idx, sub_t):
    """Draw one frame. sub_t in [0, 1] for tween between steps."""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # Current and previous step
    cur = STEPS[step_idx]
    prev = STEPS[max(0, step_idx - 1)] if sub_t < 1.0 and step_idx > 0 else cur
    t = ease_out(sub_t)

    time_s = lerp(prev[0], cur[0], t)
    p_left = lerp(prev[1], cur[1], t)
    p_right = 1.0 - p_left
    pred = cur[2] if sub_t > 0.3 else (prev[2] if step_idx > 0 else None)

    # ── Title bar ──
    d.text((W // 2, 18), "NeuroStream", fill=FG, font=font_title, anchor="mt")
    d.text((W // 2, 44), "Progressive Prediction Demo", fill=GREY, font=font_xs,
           anchor="mt")

    # ── Phase indicator ──
    phases = ["Configure", "Training", "Streaming"]
    phase_y = 65
    total_w = 0
    for p in phases:
        total_w += d.textlength(p, font=font_xs) + 30
    total_w += 40  # arrows
    x_start = (W - total_w) // 2
    cx = x_start
    for i, p in enumerate(phases):
        tw = d.textlength(p, font=font_xs) + 20
        if i == 2:  # Streaming active
            d.rounded_rectangle([cx, phase_y - 10, cx + tw, phase_y + 12],
                                radius=4, fill=ACCENT)
            d.text((cx + tw // 2, phase_y + 1), p, fill=WHITE, font=font_xs,
                   anchor="mm")
        elif i < 2:  # completed
            d.text((cx + tw // 2, phase_y + 1), p, fill=GREEN, font=font_xs,
                   anchor="mm")
        cx += tw
        if i < 2:
            d.text((cx + 5, phase_y + 1), "›", fill=GREY, font=font_med, anchor="mm")
            cx += 20

    y = 90

    # ── EEG Window + Progress bar ──
    d.text((W // 2, y), f"EEG Window:  {time_s:.1f} s  /  {TOTAL_S:.0f}.0 s",
           fill=YELLOW, font=font_med, anchor="mt")
    y += 28

    bar_x0, bar_x1 = 60, W - 60
    bar_h = 10
    d.rounded_rectangle([bar_x0, y, bar_x1, y + bar_h], radius=4, fill=LIGHT)
    pct = min(1.0, time_s / TOTAL_S)
    fill_x = bar_x0 + int((bar_x1 - bar_x0) * pct)
    if fill_x > bar_x0 + 4:
        d.rounded_rectangle([bar_x0, y, fill_x, y + bar_h], radius=4, fill=ACCENT)
    y += 22

    # ── Prediction ──
    is_final = (step_idx == len(STEPS) - 1 and sub_t >= 1.0)
    if pred is None:
        pred_text = "Prediction:  —"
        pred_color = GREY
    elif is_final:
        pred_text = f"Prediction:  {pred}"
        pred_color = GREEN if pred == TRUE_LABEL else RED
    else:
        pred_text = f"Prediction:  {pred}  (tentative, {cur[0]:.1f}s)"
        pred_color = GREY

    d.text((W // 2, y + 4), pred_text, fill=pred_color, font=font_big, anchor="mt")
    y += 42

    # ── Actual (only at final) ──
    if is_final:
        d.text((W // 2, y), f"Actual:  {TRUE_LABEL}  ✓", fill=GREEN, font=font_med,
               anchor="mt")
    y += 28

    # ── Confidence bar ──
    d.text((W // 2, y), "Classifier Confidence", fill=GREY, font=font_xs, anchor="mt")
    y += 16
    cb_x0, cb_x1 = 50, W - 50
    cb_h = 28
    split = cb_x0 + int((cb_x1 - cb_x0) * p_left)
    d.rounded_rectangle([cb_x0, y, cb_x1, y + cb_h], radius=6, fill=LIGHT)
    if split > cb_x0 + 4:
        d.rounded_rectangle([cb_x0, y, split, y + cb_h], radius=6, fill=ACCENT)
    if split < cb_x1 - 4:
        d.rounded_rectangle([split, y, cb_x1, y + cb_h], radius=6, fill=LIGHT)

    lx = (cb_x0 + split) // 2
    rx = (split + cb_x1) // 2
    d.text((lx, y + cb_h // 2), f"LEFT  {p_left:.0%}",
           fill=WHITE if p_left > 0.15 else FG, font=font_xs, anchor="mm")
    d.text((rx, y + cb_h // 2), f"RIGHT  {p_right:.0%}",
           fill=FG if p_right > 0.15 else WHITE, font=font_xs, anchor="mm")
    y += cb_h + 16

    # ── Band power bars (simplified) ──
    bp_x0 = 50
    bp_w = 130
    bp_h_max = 60
    mu_frac = lerp(0.1, 0.22, pct)
    beta_frac = lerp(0.08, 0.18, pct)

    d.text((bp_x0 + bp_w // 2, y), "Band Power", fill=GREY, font=font_xs, anchor="mt")
    y_bp = y + 16
    for i, (label, frac, color) in enumerate([
        ("μ 8-12", mu_frac, ACCENT), ("β 13-30", beta_frac, GREEN)
    ]):
        bx = bp_x0 + i * 70
        d.rectangle([bx, y_bp, bx + 50, y_bp + bp_h_max], fill=LIGHT)
        fill_h = max(2, int(frac * bp_h_max))
        d.rectangle([bx, y_bp + bp_h_max - fill_h, bx + 50, y_bp + bp_h_max],
                     fill=color)
        d.text((bx + 25, y_bp + bp_h_max + 10), f"{label} {frac:.0%}",
               fill=FG, font=font_xs, anchor="mt")

    # ── Mini confusion matrix ──
    cm_x0 = W - 200
    d.text((cm_x0 + 65, y), "Confusion Matrix", fill=GREY, font=font_xs, anchor="mt")
    cm_y = y + 16
    cell = 30
    # Simulated counts that build up
    n_done = max(0, step_idx - 1) * 3
    cm = [[int(n_done * 0.4), int(n_done * 0.1)],
          [int(n_done * 0.08), int(n_done * 0.42)]]
    labels_cm = ["L", "R"]
    for i in range(2):
        for j in range(2):
            x0 = cm_x0 + 20 + j * cell
            y0 = cm_y + i * cell
            c = GREEN if i == j else RED
            alpha = min(1.0, cm[i][j] / max(1, max(v for row in cm for v in row)))
            # Blend color
            r = int(lerp(240, int(c[1:3], 16), alpha))
            g = int(lerp(248, int(c[3:5], 16), alpha))
            b = int(lerp(248, int(c[5:7], 16), alpha))
            fill_c = f"#{r:02x}{g:02x}{b:02x}"
            d.rectangle([x0, y0, x0 + cell, y0 + cell], fill=fill_c, outline=LIGHT)
            d.text((x0 + cell // 2, y0 + cell // 2), str(cm[i][j]),
                   fill=FG, font=font_xs, anchor="mm")
        d.text((cm_x0 + 10, cm_y + i * cell + cell // 2), labels_cm[i],
               fill=GREY, font=font_xs, anchor="mm")
    for j in range(2):
        d.text((cm_x0 + 20 + j * cell + cell // 2, cm_y - 8), labels_cm[j],
               fill=GREY, font=font_xs, anchor="mm")

    return img


# Generate frames
frames = []
TWEEN_FRAMES = 6  # frames per transition

for step_idx in range(len(STEPS)):
    if step_idx == 0:
        # Initial state — hold briefly
        frames.extend([draw_frame(0, 1.0)] * 3)
    else:
        # Tween from previous to current
        for f in range(TWEEN_FRAMES):
            t = f / TWEEN_FRAMES
            frames.append(draw_frame(step_idx, t))
        # Hold at current step
        hold = 4 if step_idx < len(STEPS) - 1 else 10
        frames.append(draw_frame(step_idx, 1.0))
        frames.extend([draw_frame(step_idx, 1.0)] * hold)

# Save GIF
frames[0].save(
    OUT,
    save_all=True,
    append_images=frames[1:],
    duration=80,  # ms per frame
    loop=0,
)
print(f"Saved {OUT}  ({len(frames)} frames)")
