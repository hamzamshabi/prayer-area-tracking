"""Count only people who are WALKING (Tawaf) inside a tracked region.

Pipeline (per frame):
  density   = CLIP-EBC sliding window (per-pixel)
  region    = boolean mask from SAM2 / homography NPZ
  motion    = dense optical flow gated by tangential direction around the Kaaba center
  walking   = sum(density * region * motion)

Output: annotated MP4 (region outline, green tint on walking pixels, live counts).
"""

import argparse
import json
import shutil
import subprocess
import sys
from collections import OrderedDict, deque
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

NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
WINDOW_SIZE = 224
STRIDE = 224

DEFAULT_BINS = [(0, 0), (1, 1), (2, 2), (3, 3), (4, float('inf'))]
DEFAULT_ANCHOR_POINTS = [0, 1, 2, 3, 5.71828]
CLIP_ARGS = {"prompt_type": "word", "num_vpt": 32, "vpt_drop": 0.0, "deep_vpt": True}


# ---------------------------------------------------------------------------
# CLIP-EBC loading & inference (mirrors count_in_region.py)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Motion analysis
# ---------------------------------------------------------------------------


def build_radial_tangential(h: int, w: int, center: tuple[int, int]):
    """Per-pixel unit vectors: r_hat (radial out from center), t_hat (90° CCW)."""
    cx, cy = center
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = xs - cx
    dy = ys - cy
    r = np.sqrt(dx * dx + dy * dy)
    r_safe = np.maximum(r, 1.0)
    rx = dx / r_safe
    ry = dy / r_safe
    tx = -ry
    ty = rx
    return rx, ry, tx, ty


def farneback_flow(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
    return cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--frames",    required=True, type=Path)
    p.add_argument("--masks",     type=Path, default=None,
                   help="NPZ from run_sam2.py / track_region_homography.py (per-frame masks)")
    p.add_argument("--polygon",   type=Path, default=None,
                   help="static polygon JSON from draw_polygon.py (rasterized once, reused for every frame)")
    p.add_argument("--center",    required=True, type=Path)
    p.add_argument("--out",       required=True, type=Path)
    p.add_argument("--weights",   type=Path, required=True,
                   help="CLIP-EBC checkpoint: original pretrained or LoRA fine-tuned")
    p.add_argument("--clip-ebc",  type=Path, default=ROOT / "external" / "CLIP-EBC")
    p.add_argument("--obj-id",    type=int, default=1)
    p.add_argument("--fps",       type=int, default=4)
    p.add_argument("--lora-rank", type=int, default=16)

    p.add_argument("--flow-threshold", type=float, default=0.6,
                   help="min |tangential flow| in scaled-px/frame to count as walking")
    p.add_argument("--tangent-ratio",  type=float, default=1.5,
                   help="|tang| must exceed this multiple of |rad| to count as walking")
    p.add_argument("--smooth-frames",  type=int,   default=3,
                   help="rolling mean window over flow vectors")
    p.add_argument("--flow-downscale", type=float, default=0.5,
                   help="compute flow on downscaled frame for speed")

    args = p.parse_args()

    if not FFMPEG.exists():
        raise SystemExit(f"ffmpeg not found at {FFMPEG}")
    if not args.weights.exists():
        raise SystemExit(f"weights not found: {args.weights}")
    if not args.clip_ebc.exists():
        raise SystemExit(f"CLIP-EBC repo not found: {args.clip_ebc}")
    if not args.center.exists():
        raise SystemExit(f"center json not found: {args.center}")
    if (args.masks is None) == (args.polygon is None):
        raise SystemExit("provide exactly one of --masks or --polygon")
    if args.masks is not None and not args.masks.exists():
        raise SystemExit(f"masks not found: {args.masks}")
    if args.polygon is not None and not args.polygon.exists():
        raise SystemExit(f"polygon not found: {args.polygon}")

    sys.path.insert(1, str(args.clip_ebc))  # after scripts/, so the bundled lora_utils wins

    center_payload = json.loads(args.center.read_text())
    cx, cy = center_payload["center"]
    print(f"Kaaba center (full-res): ({cx}, {cy})")

    frame_files = sorted(args.frames.glob("*.jpg"))
    if not frame_files:
        raise SystemExit(f"no frames in {args.frames}")
    total = len(frame_files)

    polygon_pts: np.ndarray | None = None
    masks: np.ndarray | None = None
    if args.masks is not None:
        masks_npz = np.load(args.masks)
        key = f"obj_{args.obj_id}"
        if key not in masks_npz.files:
            raise SystemExit(f"{key} not in {args.masks}; available: {masks_npz.files}")
        masks = masks_npz[key]
        if masks.shape[0] != total:
            raise SystemExit(f"mask frames ({masks.shape[0]}) != frame files ({total})")
        print(f"loaded {total} frames + masks for {key}")
    else:
        poly_payload = json.loads(args.polygon.read_text())
        prompts = poly_payload.get("prompts", [])
        match = next((pr for pr in prompts if pr.get("obj_id") == args.obj_id), None)
        if match is None or "polygon" not in match:
            raise SystemExit(f"no polygon for obj_id={args.obj_id} in {args.polygon}")
        polygon_pts = np.array(match["polygon"], dtype=np.int32)
        print(f"loaded {total} frames + static polygon ({len(polygon_pts)} vertices)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading CLIP-EBC from {args.weights} on {device}")
    model = load_model(args.weights, args.lora_rank, device)

    render_dir = args.out.parent / f"_render_{args.out.stem}"
    if render_dir.exists():
        shutil.rmtree(render_dir)
    render_dir.mkdir(parents=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    flow_buf: deque = deque(maxlen=max(args.smooth_frames, 1))
    prev_gray_small = None
    rad_basis: tuple[np.ndarray, ...] | None = None  # (rx, ry, tx, ty) at flow scale
    static_mask: np.ndarray | None = None  # cached when --polygon mode
    scale = float(args.flow_downscale)
    if not (0.0 < scale <= 1.0):
        raise SystemExit("--flow-downscale must be in (0, 1]")

    for i, fp in enumerate(tqdm(frame_files, desc="walk")):
        img_bgr = cv2.imread(str(fp))
        if img_bgr is None:
            raise SystemExit(f"failed to read {fp}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        if masks is not None:
            mask = masks[i]
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.uint8), (w, h),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            if static_mask is None or static_mask.shape != (h, w):
                m = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(m, [polygon_pts], 1)
                static_mask = m.astype(bool)
            mask = static_mask

        # --- optical flow at downscaled resolution ---
        if scale < 1.0:
            sw, sh = max(int(round(w * scale)), 1), max(int(round(h * scale)), 1)
            small = cv2.resize(img_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
        else:
            sw, sh = w, h
            small = img_bgr
        gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if rad_basis is None:
            cxs = cx * (sw / w)
            cys = cy * (sh / h)
            rad_basis = build_radial_tangential(sh, sw, (int(round(cxs)), int(round(cys))))

        if prev_gray_small is None:
            motion_small = np.zeros((sh, sw), dtype=bool)
        else:
            flow = farneback_flow(prev_gray_small, gray_small)  # (sh, sw, 2): u=dx, v=dy
            flow_buf.append(flow)
            flow_mean = np.mean(np.stack(flow_buf, axis=0), axis=0)
            u = flow_mean[..., 0]
            v = flow_mean[..., 1]
            rx, ry, tx, ty = rad_basis
            tang = np.abs(u * tx + v * ty)
            rad  = np.abs(u * rx + v * ry)
            motion_small = (tang > args.flow_threshold) & (tang > args.tangent_ratio * rad)
        prev_gray_small = gray_small

        if scale < 1.0:
            motion = cv2.resize(motion_small.astype(np.uint8), (w, h),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            motion = motion_small

        # --- CLIP-EBC density ---
        pil = Image.fromarray(img_rgb)
        tensor = preprocess_frame(pil, device)
        density = sliding_window_predict(model, tensor).squeeze().cpu().numpy()
        density_pos = np.maximum(density, 0)

        m_region = mask.astype(np.float32)
        m_walk   = (mask & motion).astype(np.float32)
        count_total   = float(density_pos.sum())
        count_region  = float((density_pos * m_region).sum())
        count_walking = float((density_pos * m_walk).sum())

        # --- annotate ---
        rendered = img_bgr.copy()
        # dim everything outside the region
        mask_3c = np.repeat(mask[:, :, None], 3, axis=2)
        dimmed = (rendered.astype(np.float32) * 0.55).astype(np.uint8)
        rendered = np.where(mask_3c, rendered, dimmed)
        # green tint on walking pixels
        walk_layer = np.zeros_like(rendered)
        walk_layer[mask & motion] = (60, 220, 80)
        rendered = cv2.addWeighted(rendered, 1.0, walk_layer, 0.45, 0.0)
        # region contour
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rendered, contours, -1, (255, 200, 60), 2)
        # Kaaba center marker
        cv2.drawMarker(rendered, (cx, cy), (0, 255, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)

        label = f"walking: {count_walking:.0f}  in_region: {count_region:.0f}  total: {count_total:.0f}"
        cv2.putText(rendered, label, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(rendered, label, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(rendered, f"frame {i+1}/{total}", (10, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(rendered, f"frame {i+1}/{total}", (10, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.imwrite(str(render_dir / f"{i:05d}.jpg"), rendered, [cv2.IMWRITE_JPEG_QUALITY, 92])

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
    print(f"done: {args.out}")


if __name__ == "__main__":
    main()
