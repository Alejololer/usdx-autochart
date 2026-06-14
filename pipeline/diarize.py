"""Speaker diarization of a vocal stem via pyannote.audio.

Returns a list of ``(start_s, end_s, speaker_label)`` segments, used by
assemble.py to attribute each word/line to a singer for the P1/P2 duet split
(replacing the pitch-register approximation, which collapses when a singer
leaves their usual register or both sing in unison).

Like separate.py this is **optional and fails soft**: ML imports are lazy, any
failure (no HF token, model load error, CUDA OOM) logs and returns ``[]`` so the
caller falls back to register clustering. No behaviour change when unavailable.

Requirements (install separately, mirrors how demucs is handled):
  pip install pyannote.audio
  export HF_TOKEN=...        # accept terms at hf.co/pyannote/speaker-diarization-3.1

Determinism: we pin the model id, force ``num_speakers=2`` (no speaker-count
estimation), and seed torch. The arbitrary "SPEAKER_00/01" labels don't matter -
assemble maps them to P1/P2 by earliest onset.

Windows/torchcodec note: pyannote 4.x decodes audio through torchcodec, whose
shared libs often fail to load on Windows (same gotcha as torchaudio.save - see
separate.py). We sidestep it by decoding the WAV ourselves (soundfile) and passing
an in-memory ``{"waveform", "sample_rate"}`` dict, so pyannote never touches the file.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

_MODEL = "pyannote/speaker-diarization-3.1"


def _pick_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def _patch_torchaudio() -> None:
    """torch>=2.9 (the cu128 build here) removed torchaudio's legacy file-I/O API
    that pyannote 3.x / speechbrain probe at import (``AudioMetaData``,
    ``list_audio_backends``, ...). We never hit torchaudio file I/O - we pass an
    in-memory waveform - so stubbing the missing names is enough to let it import."""
    try:
        import torchaudio
    except Exception:  # noqa: BLE001
        return
    if not hasattr(torchaudio, "AudioMetaData"):
        try:
            from torchaudio.backend.common import AudioMetaData
        except Exception:  # noqa: BLE001
            class AudioMetaData:  # minimal stand-in; only used for typing
                pass
        torchaudio.AudioMetaData = AudioMetaData
    # legacy backend probes removed in torch 2.9 - safe no-op stubs
    if not hasattr(torchaudio, "list_audio_backends"):
        # pyannote io.py picks "soundfile" when present; advertise it (it is installed)
        torchaudio.list_audio_backends = lambda: ["soundfile"]
    if not hasattr(torchaudio, "get_audio_backend"):
        torchaudio.get_audio_backend = lambda: None
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda *a, **k: None


def _patch_speechbrain() -> None:
    """speechbrain 1.x registers *lazy* proxies for optional integrations (k2,
    huggingface wordemb, ...). pytorch-lightning's from_pretrained calls
    ``inspect.stack()`` → ``inspect.getmodule`` → ``getattr(mod, "__file__", None)``,
    which on a lazy proxy triggers a real import of an absent optional dep and
    raises ``ImportError``. Because that's not ``AttributeError``, getattr's default
    can't absorb it and loading aborts. We make the proxy raise ``AttributeError``
    for dunder probes so inspection skips it; genuine attribute access still imports.
    Call *after* speechbrain is imported."""
    try:
        from speechbrain.utils import importutils as iu
    except Exception:  # noqa: BLE001
        return
    cls = getattr(iu, "LazyModule", None)
    if cls is None or getattr(cls, "_usdx_patched", False):
        return
    orig = cls.__getattr__

    def _safe_getattr(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)  # let inspect's getattr(...,None) default win
        return orig(self, name)

    cls.__getattr__ = _safe_getattr
    cls._usdx_patched = True


def _patch_hf_hub() -> None:
    """pyannote 3.x passes the removed ``use_auth_token`` kwarg to
    huggingface_hub download fns (renamed ``token`` in hub >= 1.0). Translate it,
    scoped here so the working whisper/demucs paths keep hub untouched. Must run
    *before* ``import pyannote`` so pyannote picks up the wrapped functions."""
    try:
        import huggingface_hub as hub
    except Exception:  # noqa: BLE001
        return

    def _wrap(fn):
        if getattr(fn, "_usdx_wrapped", False):
            return fn

        def inner(*args, **kwargs):
            if "use_auth_token" in kwargs:
                kwargs["token"] = kwargs.pop("use_auth_token")
            return fn(*args, **kwargs)

        inner._usdx_wrapped = True
        return inner

    for name in ("hf_hub_download", "snapshot_download"):
        if hasattr(hub, name):
            setattr(hub, name, _wrap(getattr(hub, name)))


def _load_waveform(wav_path: str):
    """Decode to a (1, samples) float32 torch tensor + sample rate, without
    going through torchcodec."""
    import numpy as np
    import soundfile as sf
    import torch

    data, sr = sf.read(wav_path, dtype="float32", always_2d=True)  # (samples, ch)
    mono = data.mean(axis=1).astype(np.float32)                    # (samples,)
    return torch.from_numpy(mono)[None, :], int(sr)                # (1, samples)


def diarize(wav_path: str, num_speakers: int = 2,
            device: str = "auto") -> List[Tuple[float, float, str]]:
    """Diarize a vocal WAV into ``[(start, end, speaker)]`` segments (seconds).
    Returns ``[]`` on any failure (graceful fallback to register clustering)."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[diarize] HF_TOKEN not set; falling back to register clustering")
        return []
    try:
        import torch
        _patch_torchaudio()
        _patch_hf_hub()
        from pyannote.audio import Pipeline
        _patch_speechbrain()  # after speechbrain is imported, to override its lazy proxy

        torch.manual_seed(0)
        # torch >= 2.6 defaults torch.load(weights_only=True), which rejects the
        # official pyannote checkpoint. We trust that source, so force the legacy
        # behaviour for the duration of the load+inference, then restore it.
        # add_safe_globals covers callers that bind torch.load directly.
        try:
            torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
        except Exception:  # noqa: BLE001
            pass
        _orig_load = torch.load

        def _load_compat(*a, **k):
            k["weights_only"] = False
            return _orig_load(*a, **k)

        torch.load = _load_compat
        try:
            # pyannote 4.x uses `token=`; older builds used `use_auth_token=`.
            try:
                pipeline = Pipeline.from_pretrained(_MODEL, token=token)
            except TypeError:
                pipeline = Pipeline.from_pretrained(_MODEL, use_auth_token=token)
            if pipeline is None:
                print("[diarize] pipeline load returned None (model terms accepted?)")
                return []
            dev = device if device != "auto" else _pick_device()
            try:
                pipeline.to(torch.device(dev))
            except Exception:  # noqa: BLE001 - CPU fallback if .to() unsupported
                pass

            waveform, sr = _load_waveform(wav_path)
            annotation = pipeline({"waveform": waveform, "sample_rate": sr},
                                  num_speakers=num_speakers)
        finally:
            torch.load = _orig_load
        segments: List[Tuple[float, float, str]] = [
            (float(turn.start), float(turn.end), str(speaker))
            for turn, _track, speaker in annotation.itertracks(yield_label=True)
        ]
        segments.sort(key=lambda s: s[0])
        return segments
    except Exception as e:  # noqa: BLE001
        print(f"[diarize] pyannote failed: {e!r}")
        return []
