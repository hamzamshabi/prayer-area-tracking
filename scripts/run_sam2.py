"""Run SAM 2.1 video predictor on extracted frames; render annotated MP4.

Each prompt entry in the JSON may carry either:
  "box":     [x1, y1, x2, y2]                — seeds via add_new_points_or_box
  "polygon": [[x1, y1], [x2, y2], ...]       — rasterized and seeded via add_new_mask
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from sam2.build_sam import build_sam2_video_predictor

ROOT = Path(__file__).resolve().parent.parent
from _ffmpeg import find_ffmpeg  # noqa: E402
FFMPEG = find_ffmpeg()

CFG = {
    "tiny":      ("configs/sam2.1/sam2.1_hiera_t.yaml",  "checkpoints/sam2.1_hiera_tiny.pt"),
    "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "checkpoints/sam2.1_hiera_base_plus.pt"),
}

PALETTE = [
    (255,  64,  64),
    ( 64, 200,  80),
    ( 80, 140, 255),
    (255, 200,  60),
    (220,  90, 220),
]


def color_for(obj_id: int) -> tuple[int, int, int]:
    return PALETTE[(obj_id - 1) % len(PALETTE)]


def overlay(frame: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    layer = np.zeros_like(frame)
    layer[mask] = color
    return cv2.addWeighted(frame, 1.0, layer, alpha, 0.0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--frames", required=True, type=Path)
    p.add_argument("--prompt", required=True, type=Path)
    p.add_argument("--out",    required=True, type=Path)
    p.add_argument("--model",  choices=list(CFG), default="tiny")
    p.add_argument("--fps",    type=int, default=4)
    p.add_argument("--export-masks", type=Path, default=None,
                   help="optional .npz path; saves per-frame bool masks keyed by obj_<id>")
    args = p.parse_args()

    if not FFMPEG.exists():
        raise SystemExit(f"ffmpeg not found at {FFMPEG}")
    cfg_rel, ckpt_rel = CFG[args.model]
    ckpt = ROOT / ckpt_rel
    if not ckpt.exists():
        raise SystemExit(f"checkpoint missing: {ckpt}")

    prompt = json.loads(args.prompt.read_text())
    prompt_frame_idx = int(prompt["frame_idx"])
    prompts = prompt["prompts"]

    frame_files = sorted(args.frames.glob("*.jpg"))
    if not frame_files:
        raise SystemExit(f"no frames in {args.frames}")
    total = len(frame_files)
    probe = cv2.imread(str(frame_files[0]))
    if probe is None:
        raise SystemExit(f"failed to read {frame_files[0]}")
    frame_h, frame_w = probe.shape[:2]
    print(f"loaded {total} frames ({frame_w}x{frame_h}) from {args.frames}; model={args.model}")

    device = torch.device("cuda")
    predictor = build_sam2_video_predictor(config_file=cfg_rel, ckpt_path=str(ckpt), device=device)

    state = predictor.init_state(
        video_path=str(args.frames),
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
        async_loading_frames=True,
    )

    masks_per_frame: dict[int, dict[int, np.ndarray]] = {}

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for entry in prompts:
            obj_id = int(entry["obj_id"])
            if "polygon" in entry:
                pts = np.array(entry["polygon"], dtype=np.int32).reshape(-1, 1, 2)
                mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
                cv2.fillPoly(mask, [pts], 1)
                predictor.add_new_mask(
                    inference_state=state,
                    frame_idx=prompt_frame_idx,
                    obj_id=obj_id,
                    mask=mask.astype(bool),
                )
            elif "box" in entry:
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=prompt_frame_idx,
                    obj_id=obj_id,
                    box=np.array(entry["box"], dtype=np.float32),
                )
            else:
                raise SystemExit(f"prompt entry missing 'box' or 'polygon': {entry}")
        for f_idx, obj_ids, mask_logits in tqdm(
            predictor.propagate_in_video(state), total=total, desc="propagate"
        ):
            masks_per_frame[f_idx] = {
                int(oid): (mask_logits[i, 0] > 0.0).cpu().numpy()
                for i, oid in enumerate(obj_ids)
            }

    if args.export_masks is not None:
        all_obj_ids = sorted({oid for frame_masks in masks_per_frame.values() for oid in frame_masks})
        stacks = {}
        for oid in all_obj_ids:
            frames_stack = np.zeros((total, frame_h, frame_w), dtype=bool)
            for i in range(total):
                m = masks_per_frame.get(i, {}).get(oid)
                if m is None:
                    continue
                if m.shape != (frame_h, frame_w):
                    m = cv2.resize(m.astype(np.uint8), (frame_w, frame_h),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
                frames_stack[i] = m
            stacks[f"obj_{oid}"] = frames_stack
        args.export_masks.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.export_masks, **stacks)
        print(f"exported masks: {args.export_masks} ({list(stacks)}, shape={frames_stack.shape})")

    render_dir = args.out.parent / f"_render_{args.out.stem}"
    if render_dir.exists():
        shutil.rmtree(render_dir)
    render_dir.mkdir(parents=True)

    for i, fp in enumerate(tqdm(frame_files, desc="render")):
        img = cv2.imread(str(fp))
        masks = masks_per_frame.get(i, {})
        for obj_id, m in masks.items():
            c = color_for(obj_id)
            if m.shape != img.shape[:2]:
                m = cv2.resize(m.astype(np.uint8), (img.shape[1], img.shape[0]),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
            img = overlay(img, m, c)

        for entry in prompts:
            c = color_for(int(entry["obj_id"]))
            if "polygon" in entry:
                pts = np.array(entry["polygon"], dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(img, [pts], isClosed=True, color=c, thickness=1)
                label_pt = tuple(int(v) for v in entry["polygon"][0])
            else:
                x1, y1, x2, y2 = [int(v) for v in entry["box"]]
                cv2.rectangle(img, (x1, y1), (x2, y2), c, 1)
                label_pt = (x1, y1)
            cv2.putText(img, f"id={entry['obj_id']}", (label_pt[0], max(0, label_pt[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)

        cv2.putText(img, f"frame {i+1}/{total}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"frame {i+1}/{total}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.imwrite(str(render_dir / f"{i:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])

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
    print(f"done: {args.out}")


if __name__ == "__main__":
    main()
