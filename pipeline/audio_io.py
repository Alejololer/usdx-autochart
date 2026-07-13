"""Audio loading via ffmpeg (decodes mp3/ogg/avi without extra Python codecs)."""
from __future__ import annotations

import subprocess
from typing import Tuple

import numpy as np


def load_mono(path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Decode any ffmpeg-readable file to a mono float32 array at `sr`."""
    cmd = [
        "ffmpeg", "-v", "error", "-i", path,
        "-ac", "1", "-ar", str(sr), "-f", "f32le", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    audio = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    return audio, sr


def duration_seconds(path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", path,
    ]
    out = subprocess.run(cmd, capture_output=True, encoding="utf-8",
                         errors="replace", check=True).stdout
    try:
        return float(out.strip())
    except ValueError:
        return 0.0
