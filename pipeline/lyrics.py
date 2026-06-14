"""Lyrics transcription with word-level timestamps via faster-whisper (CPU,
ctranslate2 - no torch required). Returns a list of {"text","start","end"}.
"""
from __future__ import annotations

from typing import List, Optional


def _pick_device(prefer: str = "auto") -> str:
    if prefer != "auto":
        return prefer
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def transcribe_words(
    audio_path: str,
    model_size: str = "small",
    language: Optional[str] = "es",
    compute_type: Optional[str] = None,
    vad: bool = True,
    beam_size: int = 5,
    device: str = "auto",
) -> List[dict]:
    from faster_whisper import WhisperModel

    dev = _pick_device(device)
    # float16 on GPU, int8 on CPU
    if compute_type is None:
        compute_type = "float16" if dev == "cuda" else "int8"
    try:
        model = WhisperModel(model_size, device=dev, compute_type=compute_type)
    except Exception as e:  # noqa: BLE001 - GPU libs (cuDNN) missing, etc.
        print(f"[lyrics] {dev}/{compute_type} unavailable ({e!r}); using cpu/int8")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _info = model.transcribe(
        audio_path,
        language=language,
        word_timestamps=True,
        beam_size=beam_size,
        # temperature=0 (no sampling fallback) -> deterministic, reproducible
        temperature=0,
        # On clean (separated) vocals, VAD can clip sung vowels; allow disabling.
        vad_filter=vad,
        vad_parameters=dict(min_silence_duration_ms=300) if vad else None,
        condition_on_previous_text=False,
    )

    words: List[dict] = []
    for seg in segments:
        for w in (seg.words or []):
            text = w.word.strip()
            if not text:
                continue
            words.append({"text": text, "start": float(w.start), "end": float(w.end)})
    return words
