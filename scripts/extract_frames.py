"""Extract a clip from a video as zero-padded JPEGs for SAM 2 video predictor."""

import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
from _ffmpeg import find_ffmpeg  # noqa: E402
FFMPEG = find_ffmpeg()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    p.add_argument("--duration", type=float, default=60.0, help="clip duration in seconds")
    p.add_argument("--fps", type=float, default=4.0)
    p.add_argument("--max-side", type=int, default=512, help="resize so longest side <= this")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    if not args.video.exists():
        raise SystemExit(f"video not found: {args.video}")
    if not FFMPEG.exists():
        raise SystemExit(f"ffmpeg not found at {FFMPEG}")

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    vf = (
        f"fps={args.fps},"
        f"scale='if(gt(iw,ih),min({args.max_side},iw),-2)':'if(gt(iw,ih),-2,min({args.max_side},ih))'"
    )
    cmd = [
        str(FFMPEG), "-hide_banner", "-loglevel", "error",
        "-ss", str(args.start), "-t", str(args.duration),
        "-i", str(args.video),
        "-vf", vf,
        "-q:v", "2",
        "-start_number", "0",
        str(args.out / "%05d.jpg"),
    ]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    n = len(list(args.out.glob("*.jpg")))
    print(f"wrote {n} frames to {args.out}")


if __name__ == "__main__":
    main()
