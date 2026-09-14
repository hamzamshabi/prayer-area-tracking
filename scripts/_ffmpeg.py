"""Locate ffmpeg: $FFMPEG, then tools/ffmpeg/, then PATH."""

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXE = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"


def find_ffmpeg() -> Path:
    """Return the first ffmpeg found. If none is found, return the expected
    tools/ffmpeg/ path so callers' existence checks print a useful message."""
    env = os.environ.get("FFMPEG")
    if env:
        return Path(env)
    local = ROOT / "tools" / "ffmpeg" / EXE
    if local.exists():
        return local
    found = shutil.which("ffmpeg")
    return Path(found) if found else local
