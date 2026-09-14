"""Click the Kaaba center on a chosen frame and save it as JSON.

Output JSON:
  {"frame_idx": N, "center": [x, y]}

Controls:
  left-click   set / move the center
  ENTER        accept and save (need a point)
  ESC          cancel
"""

import argparse
import json
from pathlib import Path

import cv2


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--frames", required=True, type=Path)
    p.add_argument("--prompt-frame", type=int, default=30)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    frame_path = args.frames / f"{args.prompt_frame:05d}.jpg"
    if not frame_path.exists():
        raise SystemExit(f"prompt frame not found: {frame_path}")

    img = cv2.imread(str(frame_path))
    if img is None:
        raise SystemExit(f"failed to read {frame_path}")

    state: dict[str, tuple[int, int] | None] = {"point": None}
    win = "draw_point: L-click set, ENTER accept, ESC cancel"

    def redraw() -> None:
        canvas = img.copy()
        pt = state["point"]
        if pt is not None:
            cv2.drawMarker(canvas, pt, (0, 255, 255),
                           markerType=cv2.MARKER_CROSS, markerSize=24, thickness=2)
            cv2.circle(canvas, pt, 12, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"({pt[0]}, {pt[1]})", (pt[0] + 14, pt[1] - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        status = "click to set Kaaba center" if pt is None else "ENTER to save, click to move"
        cv2.putText(canvas, status, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(win, canvas)

    def on_mouse(event: int, x: int, y: int, flags: int, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            state["point"] = (x, y)
            redraw()

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 13:      # ENTER
            if state["point"] is None:
                print("click to set the center first.")
                continue
            break
        if key == 27:      # ESC
            cv2.destroyAllWindows()
            raise SystemExit("cancelled")
    cv2.destroyAllWindows()

    x, y = state["point"]
    payload = {"frame_idx": args.prompt_frame, "center": [int(x), int(y)]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}: center=({x}, {y})")


if __name__ == "__main__":
    main()
