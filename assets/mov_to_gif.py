"""
Convert screen recording to an optimized GIF for README.
Extracts a portion, downscales, reduces frame rate.

Usage: python assets/mov_to_gif.py
"""

import imageio
from PIL import Image
from pathlib import Path

INPUT = Path("/Users/yiming/Desktop/Screen Recording 2026-04-06 at 20.37.51.mov")
OUTPUT = Path(__file__).parent / "neurostream-overview.gif"

# Parameters
TARGET_W = 540          # output width (height scales proportionally)
SKIP_START_S = 0.0      # skip first N seconds
SKIP_END_S = 2.0        # skip last N seconds
SAMPLE_EVERY = 7        # take every Nth frame (reduces size)
GIF_FRAME_MS = 120      # ms per frame in output GIF
MAX_FRAMES = 120        # cap total frames

reader = imageio.get_reader(str(INPUT))
meta = reader.get_meta_data()
fps = meta.get("fps", 30)
n_frames = reader.count_frames()
duration_s = n_frames / fps

start_frame = int(SKIP_START_S * fps)
end_frame = int((duration_s - SKIP_END_S) * fps)

print(f"Input: {n_frames} frames, {fps:.1f} fps, {duration_s:.1f}s")
print(f"Extracting frames {start_frame}-{end_frame}, every {SAMPLE_EVERY}th")

frames = []
for i, frame in enumerate(reader):
    if i < start_frame:
        continue
    if i >= end_frame:
        break
    if (i - start_frame) % SAMPLE_EVERY != 0:
        continue

    img = Image.fromarray(frame)
    # Scale down
    ratio = TARGET_W / img.width
    new_h = int(img.height * ratio)
    img = img.resize((TARGET_W, new_h), Image.LANCZOS)
    frames.append(img)
    if len(frames) >= MAX_FRAMES:
        break

reader.close()

print(f"Collected {len(frames)} frames, quantizing and saving GIF...")

# Quantize to 128 colors for smaller file
quantized = [f.quantize(colors=128, method=Image.MEDIANCUT, dither=0) for f in frames]

quantized[0].save(
    str(OUTPUT),
    save_all=True,
    append_images=quantized[1:],
    duration=GIF_FRAME_MS,
    loop=0,
    optimize=True,
)

size_mb = OUTPUT.stat().st_size / 1024 / 1024
print(f"Saved {OUTPUT} ({len(frames)} frames, {size_mb:.1f} MB)")

if size_mb > 10:
    print("WARNING: GIF is large. Consider increasing SAMPLE_EVERY or reducing TARGET_W.")
