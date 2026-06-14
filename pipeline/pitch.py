"""Vocal pitch (f0) extraction.

GPU path: torchcrepe (CREPE) on CUDA - fast and accurate.
CPU fallback: librosa.pyin.
Both return (times, f0_hz, confidence) numpy arrays; unvoiced frames get
f0=0, confidence=0.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

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


def _dominant_speaker(diarization, t0: float, t1: float) -> Optional[str]:
    """Speaker with the most overlap of [t0, t1) (None if no overlap)."""
    best_spk, best_ov = None, 0.0
    for start, end, spk in diarization:
        ov = min(end, t1) - max(start, t0)
        if ov > best_ov:
            best_ov, best_spk = ov, spk
    return best_spk


def _nearest_by_register(per_spk_pitches: dict, midi: int) -> Optional[str]:
    """Speaker whose running-median pitch is closest to `midi` (register
    continuity tie-break when diarization doesn't disambiguate)."""
    best_spk, best_d = None, float("inf")
    for spk, pitches in per_spk_pitches.items():
        if not pitches:
            continue
        med = float(np.median(pitches))
        d = abs(med - midi)
        if d < best_d:
            best_d, best_spk = d, spk
    return best_spk


def extract_pitch_per_speaker(
    vocals_wav: str,
    diarization: Sequence[Tuple[float, float, str]],
    *, hop: int = 256, sr: int = 16000, device: str = "auto",
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per-singer pitch tracks via Spotify basic-pitch (multi-f0).

    Returns ``{speaker_label: (times, f0_hz, conf)}`` on the same hop grid as
    ``extract_pitch`` so ``assemble._median_pitch`` works unchanged. Each
    transcribed note is assigned to a singer by diarization-dominant speaker,
    with register-continuity as the tie-break when the window has no clear owner.

    Optional and fails soft: returns ``{}`` if basic-pitch isn't installed or the
    inference fails (caller then uses the shared mono pitch track). Deterministic:
    basic-pitch is a fixed CNN with no sampling; events are processed in time order.
    """
    if not diarization:
        return {}
    try:
        from basic_pitch.inference import predict
    except Exception as e:  # noqa: BLE001
        print(f"[pitch] basic-pitch unavailable ({e!r}); per-speaker pitch off")
        return {}
    try:
        _model_out, _midi, note_events = predict(vocals_wav)
    except Exception as e:  # noqa: BLE001
        print(f"[pitch] basic-pitch inference failed ({e!r}); per-speaker pitch off")
        return {}

    # note_events: (start_s, end_s, pitch_midi, amplitude, pitch_bends)
    events = sorted(note_events, key=lambda e: (float(e[0]), int(e[2])))
    running: Dict[str, List[int]] = {}
    assigned: List[Tuple[str, float, float, int, float]] = []
    for ev in events:
        start, end, midi = float(ev[0]), float(ev[1]), int(ev[2])
        amp = float(ev[3]) if len(ev) > 3 else 1.0
        spk = _dominant_speaker(diarization, start, end)
        if spk is None:
            spk = _nearest_by_register(running, midi)
        if spk is None:
            continue
        assigned.append((spk, start, end, midi, amp))
        running.setdefault(spk, []).append(midi)

    if not assigned:
        return {}
    max_end = max(a[2] for a in assigned)
    n = int(math.ceil(max_end * sr / hop)) + 1
    times = np.arange(n) * hop / sr
    out: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for spk in sorted({a[0] for a in assigned}):
        f0 = np.zeros(n, dtype=np.float64)
        conf = np.zeros(n, dtype=np.float64)
        for s, st, en, midi, amp in assigned:
            if s != spk:
                continue
            i0 = max(0, int(round(st * sr / hop)))
            i1 = min(n, int(round(en * sr / hop)))
            if i1 <= i0:
                i1 = min(n, i0 + 1)
            f0[i0:i1] = 440.0 * 2.0 ** ((midi - 69) / 12.0)
            conf[i0:i1] = min(1.0, max(0.6, amp))  # ensure > _median_pitch's 0.5 gate
        out[spk] = (times, f0, conf)
    return out
