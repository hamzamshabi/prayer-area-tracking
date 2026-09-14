"""Draw one bounding box on a chosen prompt frame and save it as JSON."""

import argparse
import json
from pathlib import Path

import cv2


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--frames", required=True, type=Path, help="directory of extracted frames")
    p.add_argument("--prompt-frame", type=int, default=30)
    p.add_argument("--obj-id", type=int, default=1)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    frame_path = args.frames / f"{args.prompt_frame:05d}.jpg"
    if not frame_path.exists():
        raise SystemExit(f"prompt frame not found: {frame_path}")

    img = cv2.imread(str(frame_path))
    if img is None:
        raise SystemExit(f"failed to read {frame_path}")

    print("Drag a rectangle around the prayer area, then press ENTER (or SPACE).")
    print("Press 'c' inside the selector to cancel.")
    win = "draw_box: drag rectangle, ENTER to accept"
    x, y, w, h = cv2.selectROI(win, img, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    if w == 0 or h == 0:
        raise SystemExit("no box drawn")

    box = [int(x), int(y), int(x + w), int(y + h)]
    payload = {
        "frame_idx": args.prompt_frame,
        "prompts": [{"obj_id": args.obj_id, "box": box}],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}: {payload}")


if __name__ == "__main__":
    main()
