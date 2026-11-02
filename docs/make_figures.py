"""Build the README cover from the four stage frames in docs/pipeline.jpg.

    python docs/make_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

DOCS = Path(__file__).resolve().parent

BG, CARD, MUTED, ACCENT = "#121419", "#1c2029", "#9aa0ab", "#4f8ff7"
plt.rcParams.update({"font.family": "DejaVu Sans"})


def stage_frames() -> list[Image.Image]:
    """The four stage frames of pipeline.jpg, without their title bands."""
    im = Image.open(DOCS / "pipeline.jpg").convert("RGB")
    w, h = im.size
    gray = im.convert("L").load()

    def runs(n: int, white) -> list[tuple[int, int]]:
        out, start = [], None
        for i in range(n):
            if not white(i) and start is None:
                start = i
            elif white(i) and start is not None:
                out.append((start, i))
                start = None
        return out

    cols = runs(w, lambda x: sum(gray[x, y] > 240 for y in range(0, h, 3)) > 0.9 * (h // 3))
    rows = runs(h, lambda y: sum(gray[x, y] > 240 for x in range(0, w, 3)) > 0.9 * (w // 3))
    frames_rows = sorted(rows, key=lambda r: r[1] - r[0])[-2:]
    frames_rows.sort()
    return [im.crop((x0, y0, x1, y1)) for y0, y1 in frames_rows for x0, x1 in cols[:2]]


frames = stage_frames()
aspect = frames[0].width / frames[0].height

fig_w, fig_h = 16, 6.2
fig = plt.figure(figsize=(fig_w, fig_h), dpi=100, facecolor=BG)
fig.text(0.03, 0.91, "Tracking Prayer Areas for Per-Zone Crowd Counting", color="white",
         fontsize=27, fontweight="bold", va="center")
fig.text(0.03, 0.845, "The prayer area moves, so the zone has to move with it  ·  "
         "SAM 2.1 or homography tracks the area, CLIP-EBC counts only inside it",
         color=MUTED, fontsize=13.5, va="center")

left, right, gap = 0.03, 0.97, 0.0125
w = (right - left - 3 * gap) / 4
img_h = w * fig_w / aspect / fig_h
img_y = 0.775 - img_h
labels = [
    ("1  SAM 2.1", "TRACKS THE PRAYER AREA"),
    ("2  CLIP-EBC", "COUNTS INSIDE THE MASK"),
    ("3  HOMOGRAPHY", "ANCHORS A GROUND REGION"),
    ("4  OPTICAL FLOW", "COUNTS ONLY THE WALKERS"),
]
for i, (frame, (big, small)) in enumerate(zip(frames, labels)):
    x = left + i * (w + gap)
    ax = fig.add_axes([x, img_y, w, img_h])
    ax.imshow(frame, aspect="auto")
    ax.set_axis_off()
    fig.add_artist(Rectangle((x, img_y - 0.1), w, 0.095, transform=fig.transFigure, color=CARD))
    fig.text(x + 0.009, img_y - 0.033, big, color=ACCENT, fontsize=14, fontweight="bold", va="center")
    fig.text(x + 0.009, img_y - 0.075, small, color="#c9ced6", fontsize=10.5, fontweight="bold", va="center")

cards = [
    ("2 trackers", "SAM 2.1 when the area changes shape,\nhomography when the camera moves"),
    ("1 mask format", "either tracker feeds either counter:\nobj_<id> → (T, H, W) NPZ"),
    ("4 GB GPU", "runs on an MX150 with\nsam2.1_hiera_tiny at 512 px"),
    ("0 identities", "density maps count people\nwithout recognising anyone"),
]
cw = (right - left - 3 * gap) / 4
card_y, card_h = 0.1, 0.175
for i, (big, small) in enumerate(cards):
    x = left + i * (cw + gap)
    fig.add_artist(Rectangle((x, card_y), cw, card_h, transform=fig.transFigure, color=CARD))
    fig.add_artist(Rectangle((x, card_y), 0.0022, card_h, transform=fig.transFigure, color=ACCENT))
    fig.text(x + 0.012, card_y + 0.122, big, color="white", fontsize=21, fontweight="bold", va="center")
    fig.text(x + 0.012, card_y + 0.048, small, color=MUTED, fontsize=11.5, va="center", linespacing=1.3)

fig.text(0.03, 0.045, "Frames: a public TV broadcast of Masjid al-Haram, published on YouTube. "
         "Counts are model estimates; this clip has no ground truth.", color="#6b7280", fontsize=10.5,
         va="center")
fig.savefig(DOCS / "header.png", facecolor=BG)
plt.close(fig)
print("wrote docs/header.png")
