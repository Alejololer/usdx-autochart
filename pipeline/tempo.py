"""Tempo / BPM estimation (informational; the chart grid uses a fixed musical
resolution, but a detected tempo is handy metadata and for future beat-aligned
quantization)."""
from __future__ import annotations

import numpy as np


def estimate_bpm(audio: np.ndarray, sr: int) -> float:
    import librosa

    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    try:
        return float(np.atleast_1d(tempo)[0])
    except (TypeError, IndexError):
        return float(tempo)
