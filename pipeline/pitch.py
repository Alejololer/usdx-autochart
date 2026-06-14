"""Vocal pitch (f0) extraction.

GPU path: torchcrepe (CREPE) on CUDA - fast and accurate.
CPU fallback: librosa.pyin.
Both return (times, f0_hz, confidence) numpy arrays; unvoiced frames get
f0=0, confidence=0.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def _device(prefer: str = "auto") -> str:
    if prefer != "auto":
        return prefer
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def extract_pitch_crepe(
    audio: np.ndarray, sr: int,
    fmin: float = 65.0, fmax: float = 1000.0, hop: int = 256,
    device: str = "cuda", model: str = "full",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch
    import torchcrepe

    # CREPE expects 16 kHz mono float32 in shape (1, samples)
    tensor = torch.tensor(audio, dtype=torch.float32, device=device)[None]
    f0, periodicity = torchcrepe.predict(
        tensor, sr, hop_length=hop, fmin=fmin, fmax=fmax,
        model=model, batch_size=2048, device=device, return_periodicity=True,
    )
    # Light cleanup: zero out low-confidence frames
    periodicity = torchcrepe.filter.median(periodicity, 3)
    f0 = torchcrepe.filter.mean(f0, 3)

    f0 = f0.squeeze(0).cpu().numpy().astype(np.float64)
    conf = periodicity.squeeze(0).cpu().numpy().astype(np.float64)
    f0 = np.where(conf > 0, f0, 0.0)
    times = np.arange(len(f0)) * hop / sr
    return times, f0, conf


def extract_pitch_pyin(
    audio: np.ndarray, sr: int,
    fmin: float = 65.0, fmax: float = 1000.0, hop: int = 256,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import librosa

    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio, sr=sr, fmin=fmin, fmax=fmax, hop_length=hop,
    )
    times = librosa.times_like(f0, sr=sr, hop_length=hop)
    f0 = np.nan_to_num(f0, nan=0.0)
    conf = np.where(voiced_flag, voiced_prob, 0.0)
    return times, f0.astype(np.float64), conf.astype(np.float64)


def extract_pitch(
    audio: np.ndarray, sr: int,
    fmin: float = 65.0, fmax: float = 1000.0, hop: int = 256,
    device: str = "auto",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use CREPE on GPU when available, else pYIN on CPU."""
    dev = _device(device)
    if dev == "cuda":
        try:
            return extract_pitch_crepe(audio, sr, fmin, fmax, hop, device="cuda")
        except Exception as e:  # noqa: BLE001
            print(f"[pitch] CREPE/GPU failed ({e!r}); falling back to pYIN/CPU")
    return extract_pitch_pyin(audio, sr, fmin, fmax, hop)
