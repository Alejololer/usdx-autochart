"""Turn MIR results (timed words + a pitch track) into a USDX Chart.

Inputs:
  words: list of {"text", "start", "end"[, "line_index", "new_sentence"]}
         If "line_index" is present (from align.py against canonical lyrics),
         sentence breaks follow the canonical lyric lines; otherwise they fall
         on silence gaps.
  pitch: (times[], f0_hz[], confidence[]) numpy arrays, frame-wise
  duration: audio length in seconds

Picks a beat grid, syllabifies each word, assigns a pitch per syllable, and
groups words into sentences. With duet detection on, sentences are split between
two singers by pitch register (deterministic 2-means) and written as P1/P2.
"""
from __future__ import annotations

import math
import re
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .usdx_writer import Chart, Line, Note

# performer separators that signal a duet/collab in an artist string
_MULTI_ARTIST = re.compile(
    r"(\s+y\s+|\s+e\s+|\s*&\s*|\s*,\s*|\s+feat\.?\s+|\s+ft\.?\s+|\s+con\s+"
    r"|\s+vs\.?\s+|\s+with\s+|\s+\+\s+|\s+x\s+)", re.IGNORECASE)


def split_artists(artist: str) -> List[str]:
    parts = _MULTI_ARTIST.split(artist)
    return [p.strip() for p in parts if p and not _MULTI_ARTIST.match(p) and len(p.strip()) > 1]


def is_multi_artist(artist: str) -> bool:
    return len(split_artists(artist)) >= 2

VOWELS = set("aeiouáéíóúüAEIOUÁÉÍÓÚÜ")
INSEPARABLE = {"pr", "br", "tr", "dr", "cr", "gr", "fr",
               "pl", "bl", "cl", "gl", "fl", "ll", "rr", "ch"}


def syllabify_es(word: str) -> List[str]:
    """Heuristic Spanish syllabification. Good enough for an auto-draft."""
    core = word.strip()
    if not core:
        return [word]
    letters = list(core)
    nuclei: List[int] = []
    i = 0
    while i < len(letters):
        if letters[i] in VOWELS:
            nuclei.append(i)
            while i + 1 < len(letters) and letters[i + 1] in VOWELS:
                i += 1
        i += 1
    if len(nuclei) <= 1:
        return [word]

    cuts: List[int] = []
    for a, b in zip(nuclei, nuclei[1:]):
        consonants = letters[a + 1:b]
        n = len(consonants)
        if n == 0:
            cut = b
        elif n == 1:
            cut = b - 1
        else:
            pair = (consonants[-2] + consonants[-1]).lower()
            cut = b - 2 if pair in INSEPARABLE else b - 1
        cuts.append(cut)

    parts: List[str] = []
    prev = 0
    for c in cuts:
        parts.append("".join(letters[prev:c]))
        prev = c
    parts.append("".join(letters[prev:]))
    return [p for p in parts if p]


def _median_pitch(times, f0, conf, t0: float, t1: float) -> Optional[float]:
    mask = (times >= t0) & (times < t1) & (conf > 0.5) & (f0 > 0)
    vals = f0[mask]
    if vals.size == 0:
        return None
    return 69.0 + 12.0 * math.log2(float(np.median(vals)) / 440.0)


def _kmeans2(values: Sequence[float], iters: int = 25):
    """Deterministic 1-D 2-means. Returns (labels, center_lo, center_hi)."""
    v = np.asarray(values, dtype=float)
    c0, c1 = float(v.min()), float(v.max())
    labels = np.zeros(len(v), dtype=int)
    for _ in range(iters):
        labels = (np.abs(v - c1) < np.abs(v - c0)).astype(int)
        if labels.any():
            c1 = float(v[labels == 1].mean())
        if (labels == 0).any():
            c0 = float(v[labels == 0].mean())
    lo, hi = sorted((c0, c1))
    # relabel so 0 = low register, 1 = high register
    labels = (np.abs(v - hi) < np.abs(v - lo)).astype(int)
    return labels, lo, hi


def _build_track(sentences: List[List[Note]]) -> List[Line]:
    """Lines for one track, with '-' break beats (first line has none)."""
    lines: List[Line] = []
    prev_end: Optional[int] = None
    for sent in sentences:
        if not sent:
            continue
        if prev_end is None:
            lines.append(Line(notes=sent))
        else:
            brk = min(prev_end, sent[0].start_beat - 1)
            lines.append(Line(break_beat=brk, notes=sent))
        prev_end = sent[-1].start_beat + sent[-1].duration
    return lines


def assemble(
    words: Sequence[dict],
    pitch: Tuple[np.ndarray, np.ndarray, np.ndarray],
    duration: float,
    title: str,
    artist: str,
    audio: str,
    *,
    target_grid_s: float = 0.0625,
    sentence_gap_s: float = 0.6,
    language: Optional[str] = None,
    cover: Optional[str] = None,
    video: Optional[str] = None,
    duet: str = "auto",            # "auto" | "yes" | "no"
    p1: Optional[str] = None,
    p2: Optional[str] = None,
    duet_min_sep: float = 6.0,     # semitones between registers to call a duet
) -> Chart:
    times, f0, conf = pitch
    words = [w for w in words if w.get("text", "").strip()]
    if not words:
        raise ValueError("no words to assemble")

    bpm = round(15.0 / target_grid_s, 2)
    spb = 15.0 / bpm
    first = words[0]["start"]
    gap_ms = max(0.0, first * 1000.0 - 200.0)
    t0 = gap_ms / 1000.0

    def to_beat(t: float) -> int:
        return max(0, int(round((t - t0) / spb)))

    has_lineinfo = all("line_index" in w for w in words)

    raw: List[dict] = []
    prev_end = first
    prev_line = words[0].get("line_index")
    for wi, w in enumerate(words):
        if has_lineinfo:
            new_sentence = wi > 0 and w["line_index"] != prev_line
            prev_line = w["line_index"]
        else:
            new_sentence = (w["start"] - prev_end) > sentence_gap_s
        prev_end = w["end"]

        sylls = syllabify_es(w["text"])
        wstart, wend = w["start"], max(w["end"], w["start"] + 0.08)
        span = wend - wstart
        weights = [max(1, len(s)) for s in sylls]
        total = sum(weights)
        acc = 0.0
        for si, (s, wt) in enumerate(zip(sylls, weights)):
            sb = wstart + span * (acc / total)
            acc += wt
            se = wstart + span * (acc / total)
            midi = _median_pitch(times, f0, conf, sb, se)
            text = (" " + s) if (si == 0 and wi > 0 and not new_sentence) else s
            raw.append({
                "beat": to_beat(sb), "endbeat": to_beat(se), "midi": midi,
                "text": text, "new_sentence": new_sentence and si == 0,
            })

    known = [r["midi"] for r in raw if r["midi"] is not None]
    fallback = float(np.median(known)) if known else 60.0
    for r in raw:
        if r["midi"] is None:
            r["midi"] = fallback
    offset = round(float(np.median([r["midi"] for r in raw]))) - 12

    notes: List[Tuple[bool, Note]] = []
    cursor = -1
    for r in raw:
        start = max(r["beat"], cursor + 1)
        dur = max(1, r["endbeat"] - r["beat"])
        cursor = start + dur
        notes.append((r["new_sentence"],
                      Note(start, dur, int(round(r["midi"])) - offset, r["text"])))

    # group into sentences
    sentences: List[List[Note]] = []
    cur: List[Note] = []
    for is_new, note in notes:
        if is_new and cur:
            sentences.append(cur)
            cur = []
        cur.append(note)
    if cur:
        sentences.append(cur)

    preview = round(min(duration * 0.25, max(0.0, t0 + 30 * spb)), 2) if duration > 0 else None
    base = dict(title=title, artist=artist, bpm=bpm, gap_ms=round(gap_ms),
                audio=audio, language=language, creator="usdx-autochart",
                cover=cover, video=video, preview_start=preview)

    # --- duet decision ---
    # In "auto" mode only split when the artist string names multiple performers
    # (e.g. "Carlos Baute y Marta Sánchez"); a single wide-range singer must not
    # be split. The pitch clustering then decides which singer sings each line.
    do_duet = False
    labels = None
    allow = duet == "yes" or (duet == "auto" and is_multi_artist(artist))
    if allow and len(sentences) >= 6:
        med = [float(np.median([n.pitch for n in s])) for s in sentences]
        labels, lo, hi = _kmeans2(med)
        share_hi = float(labels.mean())
        if duet == "yes" or (hi - lo >= duet_min_sep and 0.15 <= share_hi <= 0.85):
            do_duet = True

    if do_duet:
        t_lo = [s for s, lab in zip(sentences, labels) if lab == 0]
        t_hi = [s for s, lab in zip(sentences, labels) if lab == 1]
        names = split_artists(artist)
        # track 0 = low register, track 1 = high register; performers are usually
        # listed lead-first, so name them in order when we have two.
        n1 = p1 or (names[0] if len(names) >= 2 else "Singer 1")
        n2 = p2 or (names[1] if len(names) >= 2 else "Singer 2")
        return Chart(lines=[], tracks=[_build_track(t_lo), _build_track(t_hi)],
                     p1=n1, p2=n2, **base)

    return Chart(lines=_build_track(sentences), **base)
