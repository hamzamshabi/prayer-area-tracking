"""Track a fixed physical ground region across frames via per-frame homography.

Replaces SAM 2.1 for the "hold this patch of ground anchored" use case. Flat ground
seen from above is roughly planar, so a 3x3 homography from the reference frame to frame i
maps a user-drawn polygon pixel-exact. No appearance drift.

Consumes the same polygon JSON produced by `draw_polygon.py` and writes an NPZ
with the same `obj_<id>` -> (T, H, W) bool layout as `run_sam2.py --export-masks`,
so `count_in_region.py` can consume the output unchanged.
"""

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
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


def color_for(obj_id: int) -> tuple[int, int, int]:
    return PALETTE[(obj_id - 1) % len(PALETTE)]


def overlay(frame: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    layer = np.zeros_like(frame)
    layer[mask] = color
    return cv2.addWeighted(frame, 1.0, layer, alpha, 0.0)


def build_detector(name: str, max_features: int):
    if name == "orb":
        return cv2.ORB_create(nfeatures=max_features, fastThreshold=10), cv2.NORM_HAMMING
    if name == "sift":
        sift = cv2.SIFT_create(nfeatures=max_features)
        return sift, cv2.NORM_L2
    raise SystemExit(f"unknown detector: {name}")


def homography_valid(H: np.ndarray, max_det_ratio: float = 1e-3) -> bool:
    if H is None or not np.isfinite(H).all():
        return False
    det = abs(float(np.linalg.det(H[:2, :2])))
    return det > max_det_ratio


def estimate_homography(
    detector,
    matcher,
    kp_ref,
    des_ref,
    frame_gray: np.ndarray,
    ransac_thresh: float,
    min_inliers: int,
    max_reproj_err: float,
) -> tuple[np.ndarray | None, int, int, float]:
    kp_i, des_i = detector.detectAndCompute(frame_gray, None)
    if des_i is None or des_ref is None or len(kp_i) < 4:
        return None, 0, 0, float("inf")

    raw = matcher.knnMatch(des_i, des_ref, k=2)
    good = []
    for pair in raw:
        if len(pair) < 2:
            continue
        a, b = pair
        if a.distance < 0.75 * b.distance:
            good.append(a)
    n_matches = len(good)
    if n_matches < min_inliers:
        return None, n_matches, 0, float("inf")

    dst_pts = np.float32([kp_i[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    src_pts = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, inlier_mask = cv2.findHomography(
        src_pts, dst_pts, cv2.RANSAC, ransac_thresh, maxIters=2000, confidence=0.999
    )
    if not homography_valid(H):
        return None, n_matches, 0, float("inf")

    inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    if inliers < min_inliers:
        return None, n_matches, inliers, float("inf")

    mask_bool = inlier_mask.ravel().astype(bool)
    src_in = src_pts[mask_bool]
    dst_in = dst_pts[mask_bool]
    proj = cv2.perspectiveTransform(src_in, H)
    err = float(np.median(np.linalg.norm(proj - dst_in, axis=2)))
    if err > max_reproj_err:
        return None, n_matches, inliers, err

    return H, n_matches, inliers, err


def rasterize_polygon(poly_xy: np.ndarray, h: int, w: int) -> np.ndarray:
    buf = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(buf, [poly_xy.astype(np.int32)], 1)
    return buf.astype(bool)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--frames",        required=True, type=Path)
    p.add_argument("--prompt",        required=True, type=Path)
    p.add_argument("--out",           required=True, type=Path)
    p.add_argument("--export-masks",  type=Path, default=None,
                   help="optional .npz path; saves per-frame bool masks keyed by obj_<id>")
    p.add_argument("--fps",           type=int, default=4)
    p.add_argument("--detector",      choices=["orb", "sift"], default="orb")
    p.add_argument("--max-features",  type=int, default=4000)
    p.add_argument("--ransac-thresh", type=float, default=3.0)
    p.add_argument("--min-inliers",   type=int, default=25)
    p.add_argument("--max-reproj",    type=float, default=5.0)
    p.add_argument("--diagnostics",   type=Path, default=None,
                   help="optional CSV: frame_idx, n_matches, n_inliers, reproj_err, fallback")
    args = p.parse_args()

    if not FFMPEG.exists():
        raise SystemExit(f"ffmpeg not found at {FFMPEG}")

    prompt = json.loads(args.prompt.read_text())
    prompt_frame_idx = int(prompt["frame_idx"])
    entries = prompt["prompts"]
    polygons: dict[int, np.ndarray] = {}
    for entry in entries:
        if "polygon" not in entry:
            raise SystemExit(
                f"prompt entry for obj_id={entry.get('obj_id')} has no polygon. "
                f"Homography tracker needs polygons — author one with "
                f"scripts/draw_polygon.py."
            )
        obj_id = int(entry["obj_id"])
        polygons[obj_id] = np.array(entry["polygon"], dtype=np.float32).reshape(-1, 1, 2)

    frame_files = sorted(args.frames.glob("*.jpg"))
    if not frame_files:
        raise SystemExit(f"no frames in {args.frames}")
    total = len(frame_files)
    probe = cv2.imread(str(frame_files[0]))
    if probe is None:
        raise SystemExit(f"failed to read {frame_files[0]}")
    frame_h, frame_w = probe.shape[:2]
    print(f"loaded {total} frames ({frame_w}x{frame_h}) from {args.frames}; "
          f"detector={args.detector}, ref_frame={prompt_frame_idx}")

    if prompt_frame_idx < 0 or prompt_frame_idx >= total:
        raise SystemExit(
            f"prompt frame_idx={prompt_frame_idx} out of range for {total} frames"
        )
    ref_img = cv2.imread(str(frame_files[prompt_frame_idx]))
    if ref_img is None:
        raise SystemExit(f"failed to read reference frame {frame_files[prompt_frame_idx]}")
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

    detector, norm = build_detector(args.detector, args.max_features)
    matcher = cv2.BFMatcher(norm)
    kp_ref, des_ref = detector.detectAndCompute(ref_gray, None)
    if des_ref is None or len(kp_ref) < args.min_inliers:
        raise SystemExit(
            f"reference frame yielded only {0 if des_ref is None else len(kp_ref)} "
            f"features; need >= {args.min_inliers}. Try --detector sift or a different "
            f"--prompt-frame."
        )
    print(f"reference features: {len(kp_ref)}")

    masks_per_frame: dict[int, dict[int, np.ndarray]] = {}
    poly_per_frame: dict[int, dict[int, np.ndarray]] = {}
    fallback_flags: list[bool] = [False] * total
    diag_rows: list[tuple[int, int, int, float, bool]] = []

    last_good_H: np.ndarray | None = np.eye(3, dtype=np.float64)

    for i, fp in enumerate(tqdm(frame_files, desc="homography")):
        frame = cv2.imread(str(fp))
        if frame is None:
            raise SystemExit(f"failed to read {fp}")

        if i == prompt_frame_idx:
            H = np.eye(3, dtype=np.float64)
            n_matches = len(kp_ref)
            n_inliers = len(kp_ref)
            err = 0.0
            fallback = False
            last_good_H = H
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            H, n_matches, n_inliers, err = estimate_homography(
                detector, matcher, kp_ref, des_ref, gray,
                args.ransac_thresh, args.min_inliers, args.max_reproj,
            )
            if H is None:
                H = last_good_H
                fallback = True
            else:
                fallback = False
                last_good_H = H

        fallback_flags[i] = fallback
        diag_rows.append((i, n_matches, n_inliers, err, fallback))

        masks: dict[int, np.ndarray] = {}
        polys_i: dict[int, np.ndarray] = {}
        for obj_id, poly_ref in polygons.items():
            poly_i = cv2.perspectiveTransform(poly_ref, H)
            masks[obj_id] = rasterize_polygon(poly_i, frame_h, frame_w)
            polys_i[obj_id] = poly_i
        masks_per_frame[i] = masks
        poly_per_frame[i] = polys_i

    if args.export_masks is not None:
        all_obj_ids = sorted(polygons.keys())
        stacks = {}
        for oid in all_obj_ids:
            frames_stack = np.zeros((total, frame_h, frame_w), dtype=bool)
            for i in range(total):
                m = masks_per_frame.get(i, {}).get(oid)
                if m is None:
                    continue
                frames_stack[i] = m
            stacks[f"obj_{oid}"] = frames_stack
        args.export_masks.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.export_masks, **stacks)
        print(f"exported masks: {args.export_masks} ({list(stacks)}, shape={frames_stack.shape})")

    if args.diagnostics is not None:
        args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
        with args.diagnostics.open("w", newline="") as fcsv:
            w = csv.writer(fcsv)
            w.writerow(["frame_idx", "n_matches", "n_inliers", "reproj_err", "fallback"])
            for row in diag_rows:
                w.writerow([row[0], row[1], row[2], f"{row[3]:.3f}", int(row[4])])
        n_fb = sum(fallback_flags)
        print(f"diagnostics: {args.diagnostics} (fallback on {n_fb}/{total} frames)")

    render_dir = args.out.parent / f"_render_{args.out.stem}"
    if render_dir.exists():
        shutil.rmtree(render_dir)
    render_dir.mkdir(parents=True)

    for i, fp in enumerate(tqdm(frame_files, desc="render")):
        img = cv2.imread(str(fp))
        masks = masks_per_frame.get(i, {})
        polys_i = poly_per_frame.get(i, {})
        for obj_id, m in masks.items():
            c = color_for(obj_id)
            img = overlay(img, m, c)
            poly_i = polys_i.get(obj_id)
            if poly_i is not None:
                cv2.polylines(img, [poly_i.astype(np.int32)], isClosed=True, color=c, thickness=2)
                first_pt = poly_i.reshape(-1, 2)[0]
                label_pt = (int(first_pt[0]), int(first_pt[1]))
                cv2.putText(img, f"id={obj_id}", (label_pt[0], max(0, label_pt[1] - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)

        if fallback_flags[i]:
            red_layer = np.zeros_like(img)
            red_layer[:, :] = (0, 0, 180)
            img = cv2.addWeighted(img, 1.0, red_layer, 0.15, 0.0)
            cv2.putText(img, "FALLBACK", (10, frame_h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        _, n_matches, n_inliers, err, _ = diag_rows[i]
        hud = f"m={n_matches} in={n_inliers} err={err:.2f}px"
        cv2.putText(img, hud, (10, frame_h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, hud, (10, frame_h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

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
