"""CTC forced alignment of canonical lyrics via torchaudio's bundled MMS_FA.

Aligns the *full canonical text* against (ideally) a separated vocal stem, so
every canonical word gets a measured start/end: nothing is skipped when
Whisper's recall is low, elongated vowels keep their true extent, and phrase
starts are acoustic rather than attention artifacts. Each word also carries
``char_spans`` (per normalized character) so assemble can cut syllables at
acoustic boundaries instead of spreading proportionally.

Fail-soft per repo convention: lazy imports, broad except, returns None so the
caller keeps the Whisper-driven path. Deterministic (fixed CNN, no sampling).
Audio is decoded via audio_io.load_mono (ffmpeg) - never torchaudio I/O.
"""
from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Tuple

import numpy as np

from . import audio_io
from .align import tokenize_lines

# below this fraction of a-z-romanizable words, bail (ja/kana etc. keep Whisper)
MIN_ROMANIZABLE = 0.8
# wav2vec2 attention is O(n^2): run emissions in chunks, then ONE Viterbi pass
CHUNK_S = 30.0
# sanity-gate thresholds, calibrated on the Phase-0 library run
# (eval_runs/replay-n100-seed0): catastrophic misalignments show a low mean
# word posterior or words placed where nobody sings; good songs sit at
# coverage >~0.9 and score >~0.2. CTC scores on singing are systematically
# low even when timing is right, so the score floor alone must stay permissive.
MIN_SCORE = 0.15
MIN_VOICED_COVERAGE = 0.85

_MODEL = {}  # device -> loaded MMS_FA model (~1.2 GB download; load once per process)


def norm_word(word: str) -> str:
    """uroman-lite for the MMS_FA token set: NFD accent-strip, lowercase,
    a-z only (n~ -> n, a' -> a; digits/kana/punctuation drop to '')."""
    w = unicodedata.normalize("NFD", word.lower())
    w = "".join(c for c in w if not unicodedata.combining(c))
    return re.sub(r"[^a-z]", "", w)


def _pick_device(prefer: str = "auto") -> str:
    if prefer != "auto":
        return prefer
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def passes_gate(fa: dict, pitch_track: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> bool:
    """Song-level sanity check on an ``align_words`` result: words of a drifted
    alignment land in instrumental (unvoiced) regions, so require most aligned
    words to overlap voiced f0 frames, plus a permissive mean-score floor.
    On the Phase-0 sample this catches 7/12 catastrophes at 4/88 false rejects
    (a false reject only means keeping today's Whisper-merge timing)."""
    words = [w for w in fa["words"] if w["score"] > 0]
    if not words or fa["score"] < MIN_SCORE:
        return False
    times, f0, conf = pitch_track
    voiced = (conf > 0.5) & (f0 > 0)
    hit = 0
    for w in words:
        m = (times >= w["start"] - 0.05) & (times < w["end"] + 0.05)
        if bool(np.any(voiced & m)):
            hit += 1
    return hit >= MIN_VOICED_COVERAGE * len(words)


def align_words(lyric_lines: List[str], audio_path: str,
                device: str = "auto") -> Optional[dict]:
    """Force-align canonical lyric lines against the audio.

    Returns ``{"words": [...], "score": float}`` where each word dict has the
    pipeline shape (text/start/end/line_index/adlib) plus ``score`` (mean token
    posterior, 0 for unalignable words), ``norm`` (the aligned a-z string) and
    ``char_spans`` ([(start_s, end_s)] per char of ``norm``). Returns None when
    alignment is unavailable or the text doesn't romanize - caller falls back.
    """
    try:
        return _align(lyric_lines, audio_path, _pick_device(device))
    except Exception as e:  # noqa: BLE001 - optional stage, fail soft
        print(f"[forced_align] failed: {e!r}")
        return None


def _align(lyric_lines: List[str], audio_path: str, dev: str) -> Optional[dict]:
    toks = tokenize_lines(lyric_lines)
    if not toks:
        return None
    norms = [norm_word(t["text"]) for t in toks]
    if sum(1 for n in norms if n) < MIN_ROMANIZABLE * len(norms):
        print("[forced_align] lyrics don't romanize to a-z; skipping")
        return None

    import torch
    import torchaudio

    bundle = torchaudio.pipelines.MMS_FA
    model = _MODEL.get(dev)
    if model is None:
        model = bundle.get_model(with_star=False).to(dev).eval()
        _MODEL[dev] = model
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()

    audio, sr = audio_io.load_mono(audio_path, sr=bundle.sample_rate)
    if not len(audio):
        return None
    wav = torch.from_numpy(audio)

    # chunked emissions, concatenated for a single Viterbi over the whole song
    chunk = int(CHUNK_S * sr)
    bounds = list(range(0, len(audio), chunk))
    if len(bounds) > 1 and len(audio) - bounds[-1] < sr // 2:
        bounds.pop()  # merge a <0.5 s tail into the previous chunk
    parts = []
    with torch.inference_mode():
        for i, b in enumerate(bounds):
            e = bounds[i + 1] if i + 1 < len(bounds) else len(audio)
            em, _ = model(wav[None, b:e].to(dev))
            parts.append(em[0].cpu())
    emission = torch.cat(parts)
    spf = (len(audio) / sr) / emission.size(0)  # seconds per emission frame

    alignable = [i for i, n in enumerate(norms) if n]
    spans = aligner(emission, tokenizer([norms[i] for i in alignable]))

    words: List[Optional[dict]] = [None] * len(toks)
    for i, sp in zip(alignable, spans):
        frames = sum(s.end - s.start for s in sp) or 1
        words[i] = {
            "text": toks[i]["text"],
            "start": sp[0].start * spf,
            "end": sp[-1].end * spf,
            "line_index": toks[i]["line_index"],
            "adlib": toks[i]["adlib"],
            "score": float(sum(s.score * (s.end - s.start) for s in sp) / frames),
            "norm": norms[i],
            "char_spans": [(s.start * spf, s.end * spf) for s in sp],
        }
    # unalignable words (digits, kana...): wedge between aligned neighbours
    for i, w in enumerate(words):
        if w is not None:
            continue
        prev_end = next((words[j]["end"] for j in range(i - 1, -1, -1)
                         if words[j]), 0.0)
        nxt = next((words[j]["start"] for j in range(i + 1, len(words))
                    if words[j]), prev_end)
        words[i] = {
            "text": toks[i]["text"], "start": prev_end,
            "end": max(nxt, prev_end),
            "line_index": toks[i]["line_index"], "adlib": toks[i]["adlib"],
            "score": 0.0, "norm": norms[i], "char_spans": [],
        }
    scores = [w["score"] for w in words if w["score"] > 0]
    if not scores:
        return None
    return {"words": words, "score": float(sum(scores) / len(scores))}
