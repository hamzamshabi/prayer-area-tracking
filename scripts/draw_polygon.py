"""Draw a polygon on a chosen prompt frame and save it as JSON.

Controls:
  left-click   add vertex
  right-click  remove last vertex
  ENTER        accept and save (need >= 3 vertices)
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
    p.add_argument("--obj-id", type=int, default=1)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    frame_path = args.frames / f"{args.prompt_frame:05d}.jpg"
    if not frame_path.exists():
        raise SystemExit(f"prompt frame not found: {frame_path}")

    img = cv2.imread(str(frame_path))
    if img is None:
        raise SystemExit(f"failed to read {frame_path}")

    verts: list[tuple[int, int]] = []
    win = "draw_polygon: L-click add, R-click undo, ENTER accept, ESC cancel"

    def redraw() -> None:
        canvas = img.copy()
        if len(verts) >= 2:
            for i in range(len(verts) - 1):
                cv2.line(canvas, verts[i], verts[i + 1], (0, 255, 255), 2)
        if len(verts) >= 3:
            cv2.line(canvas, verts[-1], verts[0], (0, 200, 200), 1, cv2.LINE_AA)
        for i, v in enumerate(verts):
            cv2.circle(canvas, v, 4, (0, 0, 255), -1)
            cv2.putText(canvas, str(i + 1), (v[0] + 6, v[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"vertices: {len(verts)}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(win, canvas)

    def on_mouse(event: int, x: int, y: int, flags: int, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            verts.append((x, y))
            redraw()
        elif event == cv2.EVENT_RBUTTONDOWN and verts:
            verts.pop()
            redraw()

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 13:      # ENTER
            if len(verts) < 3:
                print("need at least 3 vertices — keep clicking.")
                continue
            break
        if key == 27:      # ESC
            cv2.destroyAllWindows()
            raise SystemExit("cancelled")
    cv2.destroyAllWindows()

    payload = {
        "frame_idx": args.prompt_frame,
        "prompts": [{"obj_id": args.obj_id, "polygon": [[int(x), int(y)] for x, y in verts]}],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}: {len(verts)} vertices")


if __name__ == "__main__":
    main()
