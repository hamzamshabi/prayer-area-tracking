"""Pre-run environment check. Exits non-zero on the first failure."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
from _ffmpeg import find_ffmpeg  # noqa: E402
FFMPEG = find_ffmpeg()
SAM2_DIR = ROOT / "external" / "sam2"
CKPT_TINY = ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt"
CKPT_BPLUS = ROOT / "checkpoints" / "sam2.1_hiera_base_plus.pt"
VIDEOS = ROOT / "videos"


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"[ OK ] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def main() -> None:
    try:
        import torch
    except ImportError:
        fail("torch not installed. Run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")

    if not torch.cuda.is_available():
        fail("CUDA not available. Install the CUDA build of torch and ensure NVIDIA driver is loaded.")
    name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    ok(f"CUDA OK: {name} ({total_gb:.1f} GB)")
    if total_gb < 3.5:
        warn(f"Only {total_gb:.1f} GB VRAM detected. Even sam2.1_hiera_tiny may OOM.")

    if not FFMPEG.exists():
        fail("ffmpeg not found. Put it on PATH, set $FFMPEG, or place it in tools/ffmpeg/.")
    try:
        out = subprocess.run([str(FFMPEG), "-version"], capture_output=True, text=True, check=True)
        first = out.stdout.splitlines()[0] if out.stdout else ""
        ok(f"ffmpeg OK: {first}")
    except subprocess.CalledProcessError as e:
        fail(f"ffmpeg failed to run: {e}")

    if not CKPT_TINY.exists() or CKPT_TINY.stat().st_size == 0:
        fail(f"Missing checkpoint {CKPT_TINY}. Download from https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt")
    ok(f"checkpoint OK: sam2.1_hiera_tiny.pt ({CKPT_TINY.stat().st_size / 1024**2:.1f} MB)")
    if not CKPT_BPLUS.exists():
        warn("sam2.1_hiera_base_plus.pt not present (stretch goal — fine for now)")
    else:
        ok(f"checkpoint OK: sam2.1_hiera_base_plus.pt ({CKPT_BPLUS.stat().st_size / 1024**2:.1f} MB)")

    if not SAM2_DIR.exists():
        fail(f"SAM 2 repo not cloned at {SAM2_DIR}. See README install steps.")
    try:
        import sam2  # noqa: F401
        ok("sam2 import OK")
    except ImportError as e:
        fail(f"sam2 not installed (pip install -e external/sam2). Error: {e}")

    videos = sorted(VIDEOS.glob("*.mp4")) if VIDEOS.exists() else []
    if not videos:
        warn(f"no .mp4 in {VIDEOS} yet (see README: Get a video)")
    else:
        ok(f"videos: {', '.join(v.name for v in videos)}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
