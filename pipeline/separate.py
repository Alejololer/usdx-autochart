"""Vocal separation via Demucs, used as a library so we avoid torchaudio's
file I/O (which on torch 2.9+ routes through torchcodec and breaks on Windows).

Audio is decoded with the ffmpeg CLI and the vocal stem is written with
soundfile. Returns a path to a mono vocals WAV, or None on failure.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Optional

import numpy as np


def _pick_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def _decode(path: str, sr: int, channels: int) -> np.ndarray:
    """Decode any ffmpeg-readable file to float32 (channels, samples)."""
    cmd = ["ffmpeg", "-v", "error", "-i", path,
           "-ac", str(channels), "-ar", str(sr), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    audio = np.frombuffer(raw, dtype=np.float32)
    return audio.reshape(-1, channels).T


def separate_vocals(audio_path: str, out_dir: Optional[str] = None,
                    model_name: str = "htdemucs") -> Optional[str]:
    out_dir = out_dir or tempfile.mkdtemp(prefix="demucs_")
    os.makedirs(out_dir, exist_ok=True)
    try:
        import torch
        import soundfile as sf
        from demucs.apply import apply_model
        from demucs.pretrained import get_model

        device = _pick_device()
        model = get_model(model_name)
        model.to(device).eval()

        wav_np = _decode(audio_path, model.samplerate, model.audio_channels)
        wav = torch.tensor(wav_np, device=device)
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / (ref.std() + 1e-8)

        with torch.no_grad():
            sources = apply_model(model, wav[None], device=device, progress=False)[0]
        sources = sources * ref.std() + ref.mean()

        vocals = sources[model.sources.index("vocals")]   # (channels, samples)
        mono = vocals.mean(0).cpu().numpy().astype(np.float32)

        out_path = os.path.join(out_dir, "vocals.wav")
        sf.write(out_path, mono, model.samplerate)
        return out_path
    except Exception as e:  # noqa: BLE001
        print(f"[separate] demucs failed: {e!r}")
        return None
