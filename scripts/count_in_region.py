"""Count people inside a SAM2-tracked region per frame with CLIP-EBC.

Consumes the per-frame mask NPZ produced by `run_sam2.py --export-masks` and
runs the fine-tuned CLIP-EBC ViT-B/16 (LoRA) model on each frame, restricting
the density-map sum to the mask for the chosen object id.

Outputs:
  - CSV with per-frame (frame_idx, count_region, count_total)
  - annotated MP4 with the mask tint + count text
"""

import argparse
import csv
import shutil
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
from _ffmpeg import find_ffmpeg  # noqa: E402
FFMPEG = find_ffmpeg()

PALETTE = [
    (255,  64,  64),
    ( 64, 200,  80),
    ( 80, 140, 255),
    (255, 200,  60),
    (220,  90, 220),
]

NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
WINDOW_SIZE = 224
STRIDE = 224

DEFAULT_BINS = [(0, 0), (1, 1), (2, 2), (3, 3), (4, float('inf'))]
DEFAULT_ANCHOR_POINTS = [0, 1, 2, 3, 5.71828]
CLIP_ARGS = {"prompt_type": "word", "num_vpt": 32, "vpt_drop": 0.0, "deep_vpt": True}


def color_for(obj_id: int) -> tuple[int, int, int]:
    return PALETTE[(obj_id - 1) % len(PALETTE)]


def overlay(frame: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    layer = np.zeros_like(frame)
    layer[mask] = color
    return cv2.addWeighted(frame, 1.0, layer, alpha, 0.0)


def find_density_peaks(density: np.ndarray, min_distance: int = 4, threshold_rel: float = 0.1) -> np.ndarray:
    """Return (N, 2) array of (x, y) pixel locations of local maxima in density.

    A pixel is a peak if it equals the max in its (2*min_distance+1) neighborhood
    and its value is above threshold_rel * density.max().
    """
    if density.size == 0 or float(density.max()) <= 0.0:
        return np.empty((0, 2), dtype=np.int32)
    k = 2 * min_distance + 1
    kernel = np.ones((k, k), dtype=np.uint8)
    dilated = cv2.dilate(density, kernel)
    is_peak = (density == dilated) & (density > threshold_rel * float(density.max()))
    ys, xs = np.nonzero(is_peak)
    return np.stack([xs, ys], axis=1).astype(np.int32)


def load_model(weights_path: Path, lora_rank: int, device: torch.device):
    from models import get_model

    checkpoint = torch.load(str(weights_path), map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        bins = checkpoint.get("bins", DEFAULT_BINS)
        anchor_points = checkpoint.get("anchor_points", DEFAULT_ANCHOR_POINTS)
    else:
        bins, anchor_points = DEFAULT_BINS, DEFAULT_ANCHOR_POINTS

    print(f"  bins: {bins}")
    print(f"  anchor_points: {anchor_points}")

    model = get_model(
        backbone="clip_vit_b_16",
        input_size=224,
        reduction=8,
        bins=bins,
        anchor_points=anchor_points,
        **CLIP_ARGS,
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    if any("lora_" in k for k in state_dict.keys()):
        print(f"  injecting LoRA (rank={lora_rank})")
        from lora_utils import inject_lora_into_clip_vit
        model = inject_lora_into_clip_vit(model, rank=lora_rank)

    new_state = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print(f"  loaded (missing={len(missing)}, unexpected={len(unexpected)})")

    model = model.to(device)
    model.eval()
    return model


def preprocess_frame(pil_image: Image.Image, device: torch.device) -> torch.Tensor:
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    t = T.ToTensor()(pil_image)
    t = NORMALIZE(t)
    return t.unsqueeze(0).to(device)


@torch.no_grad()
def sliding_window_predict(model, image_tensor: torch.Tensor) -> torch.Tensor:
    _, _, height, width = image_tensor.shape
    positions: list[tuple[int, int]] = []

    num_h = max((height - WINDOW_SIZE) // STRIDE + 1, 1)
    num_w = max((width - WINDOW_SIZE) // STRIDE + 1, 1)
    for i in range(num_h):
        for j in range(num_w):
            positions.append((i * STRIDE, j * STRIDE))

    last_w = max(width - WINDOW_SIZE, 0)
    if last_w > (num_w - 1) * STRIDE:
        for i in range(num_h):
            positions.append((i * STRIDE, last_w))

    last_h = max(height - WINDOW_SIZE, 0)
    if last_h > (num_h - 1) * STRIDE:
        for j in range(num_w):
            positions.append((last_h, j * STRIDE))

    if last_h > (num_h - 1) * STRIDE and last_w > (num_w - 1) * STRIDE:
        positions.append((last_h, last_w))

    pred_density = torch.zeros((1, 1, height, width), device=image_tensor.device)
    count_map = torch.zeros((1, 1, height, width), device=image_tensor.device)

    for h_start, w_start in positions:
        h_end = h_start + WINDOW_SIZE
        w_end = w_start + WINDOW_SIZE
        window = image_tensor[:, :, h_start:h_end, w_start:w_end]
        window_pred = model(window)
        if isinstance(window_pred, tuple):
            _, window_pred = window_pred

        if window_pred.shape[-2:] != (WINDOW_SIZE, WINDOW_SIZE):
            orig = window_pred.sum()
            window_pred = F.interpolate(window_pred, size=(WINDOW_SIZE, WINDOW_SIZE),
                                        mode="bilinear", align_corners=False)
            new = window_pred.sum()
            if new > 0:
                window_pred = window_pred * (orig / new)

        pred_density[:, :, h_start:h_end, w_start:w_end] += window_pred
        count_map[:, :, h_start:h_end, w_start:w_end] += 1

    return pred_density / (count_map + 1e-8)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--frames",   required=True, type=Path)
    p.add_argument("--masks",    required=True, type=Path)
    p.add_argument("--out",      required=True, type=Path)
    p.add_argument("--csv",      required=True, type=Path)
    p.add_argument("--weights",  type=Path, required=True,
                   help="CLIP-EBC checkpoint: original pretrained or LoRA fine-tuned")
    p.add_argument("--clip-ebc", type=Path, default=ROOT / "external" / "CLIP-EBC")
    p.add_argument("--obj-id",   type=int, default=1)
    p.add_argument("--fps",      type=int, default=4)
    p.add_argument("--lora-rank", type=int, default=16)
    args = p.parse_args()

    if not FFMPEG.exists():
        raise SystemExit(f"ffmpeg not found at {FFMPEG}")
    if not args.weights.exists():
        raise SystemExit(f"weights not found: {args.weights}")
    if not args.clip_ebc.exists():
        raise SystemExit(f"CLIP-EBC repo not found: {args.clip_ebc}")

    sys.path.insert(1, str(args.clip_ebc))  # after scripts/, so the bundled lora_utils wins

    frame_files = sorted(args.frames.glob("*.jpg"))
    if not frame_files:
        raise SystemExit(f"no frames in {args.frames}")
    total = len(frame_files)

    masks_npz = np.load(args.masks)
    key = f"obj_{args.obj_id}"
    if key not in masks_npz.files:
        raise SystemExit(f"{key} not in {args.masks}; available: {masks_npz.files}")
    masks = masks_npz[key]
    if masks.shape[0] != total:
        raise SystemExit(f"mask frames ({masks.shape[0]}) != frame files ({total})")
    _, mask_h, mask_w = masks.shape
    print(f"loaded {total} frames and masks ({mask_h}x{mask_w}) for {key}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading CLIP-EBC from {args.weights} on {device}")
    model = load_model(args.weights, args.lora_rank, device)

    render_dir = args.out.parent / f"_render_{args.out.stem}"
    if render_dir.exists():
        shutil.rmtree(render_dir)
    render_dir.mkdir(parents=True)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    color = color_for(args.obj_id)

    with args.csv.open("w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["frame_idx", "count_region", "count_total"])

        for i, fp in enumerate(tqdm(frame_files, desc="count")):
            img_bgr = cv2.imread(str(fp))
            if img_bgr is None:
                raise SystemExit(f"failed to read {fp}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]

            mask = masks[i]
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.uint8), (w, h),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)

            pil = Image.fromarray(img_rgb)
            tensor = preprocess_frame(pil, device)
            density = sliding_window_predict(model, tensor).squeeze().cpu().numpy()

            density_pos = np.maximum(density, 0)
            count_total = float(density_pos.sum())
            count_region = float((density_pos * mask.astype(np.float32)).sum())

            writer.writerow([i, f"{count_region:.3f}", f"{count_total:.3f}"])

            density_in_region = density_pos * mask.astype(np.float32)

            mask_3c = np.repeat(mask[:, :, None], 3, axis=2)
            dimmed = (img_bgr.astype(np.float32) * 0.55).astype(np.uint8)
            rendered = np.where(mask_3c, dimmed, img_bgr)

            peaks = find_density_peaks(density_in_region, min_distance=4, threshold_rel=0.08)
            for (px, py) in peaks:
                cv2.circle(rendered, (int(px), int(py)), 5, (255, 120, 40), -1, cv2.LINE_AA)
                cv2.circle(rendered, (int(px), int(py)), 2, (255, 230, 200), -1, cv2.LINE_AA)

            mask_u8 = mask.astype(np.uint8)
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rendered, contours, -1, color, 2)

            label = f"region: {count_region:.0f}  total: {count_total:.0f}"
            cv2.putText(rendered, label, (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(rendered, label, (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(rendered, f"frame {i+1}/{total}", (10, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(rendered, f"frame {i+1}/{total}", (10, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            cv2.imwrite(str(render_dir / f"{i:05d}.jpg"), rendered, [cv2.IMWRITE_JPEG_QUALITY, 92])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", str(args.fps),
        "-i", str(render_dir / "%05d.jpg"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        str(args.out),
    ]
    print("muxing:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    shutil.rmtree(render_dir)
    print(f"done: {args.out}  csv: {args.csv}")


if __name__ == "__main__":
    main()
