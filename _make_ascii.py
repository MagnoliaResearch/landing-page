#!/usr/bin/env python3
"""Extract a swaying magnolia silhouette from the source video and emit ASCII frames."""

import json
from pathlib import Path

import imageio
import numpy as np
from PIL import Image, ImageFilter

SRC = "YTDown.com_YouTube_Magnolia-flowers-in-the-wind-in-slow-mo_Media_vdna_eJ-7SM_001_1080p.mp4"
OUT = Path("magnolia-ascii.js")
CHARS = " .:-=+*#%@"
COLS, ROWS = 224, 80
# Denser temporal sample so adjacent frames can crossfade smoothly
START, COUNT, STEP = 12, 60, 2


def silhouette(rgb):
    a = rgb.astype(np.float32)
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    sky = ((b > r + 6) & (b > g + 4) & (luma > 148)) | ((luma > 188) & (b > r))
    strength = np.clip((190 - luma) / 190.0, 0, 1)
    strength = np.where(~sky, np.clip(strength * 1.35 + 0.22, 0, 1), 0)
    gray = Image.fromarray((strength * 255).astype(np.uint8))
    return gray.filter(ImageFilter.GaussianBlur(0.55))


def to_ascii(gray):
    g = np.asarray(gray.resize((COLS, ROWS), Image.Resampling.LANCZOS), dtype=np.float32)
    g = np.clip((g - 10) * 1.18, 0, 255)
    # gamma so sky stays empty and branches read as a silhouette
    t = (g / 255.0) ** 1.35
    n = len(CHARS) - 1
    lines = []
    for y in range(ROWS):
        row = []
        for x in range(COLS):
            v = t[y, x]
            if v < 0.07:
                row.append(" ")
            else:
                row.append(CHARS[int(v * n + 1e-6)])
        lines.append("".join(row))
    return "\n".join(lines)


def main():
    wanted = [START + i * STEP for i in range(COUNT)]
    wanted_set = set(wanted)
    collected = {}
    reader = imageio.get_reader(SRC)
    for i, frame in enumerate(reader):
        if i in wanted_set:
            collected[i] = to_ascii(silhouette(frame))
            print(f"frame {i}")
        if i >= wanted[-1]:
            break
    reader.close()
    frames = [collected[i] for i in wanted]
    OUT.write_text(
        "window.MAGNOLIA_ASCII = "
        + json.dumps(frames)
        + ";\n"
    )
    Path("_frames/preview.txt").write_text(frames[0])
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes, {len(frames)} frames)")


if __name__ == "__main__":
    main()
