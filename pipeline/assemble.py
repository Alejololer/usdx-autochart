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
two singers and written as P1/P2 - by speaker diarization when a `diarization`
segment list is supplied (lead = earliest onset = P1), otherwise by pitch register
(deterministic 2-means). `pitch_per_speaker` (Phase 2 multi-f0) gives each note its
own singer's f0 where available; `unison` duplicates shared lines into both tracks;
`drop_adlibs` removes ()-bracketed canonical ad-libs (flagged by align.py).
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


# --- speaker diarization helpers (Phase 1/3) ---------------------------------
# Diarization is a list of (start_s, end_s, speaker_label) segments produced by
# pipeline/diarize.py. These helpers map words/sentences to a speaker and decide
# the P1/P2 track split when diarization is available; _kmeans2 stays the fallback.

def _order_speakers(diarization) -> List[str]:
    """Distinct speaker labels ordered by first onset (lead singer first)."""
    first: dict = {}
    for start, _end, spk in diarization:
        if spk not in first or start < first[spk]:
            first[spk] = start
    return sorted(first, key=lambda s: (first[s], s))


def _word_speaker(diarization, t0: float, t1: float) -> Optional[str]:
    """Speaker of the segment with the most temporal overlap with [t0, t1);
    if nothing overlaps, the nearest segment by gap. None if no diarization."""
    if not diarization:
        return None
    best_spk, best_ov = None, 0.0
    for start, end, spk in diarization:
        ov = min(end, t1) - max(start, t0)
        if ov > best_ov:
            best_ov, best_spk = ov, spk
    if best_spk is not None:
        return best_spk
    best_spk, best_gap = None, float("inf")
    for start, end, spk in diarization:
        gap = t0 - end if end < t0 else (start - t1 if start > t1 else 0.0)
        if gap < best_gap:
            best_gap, best_spk = gap, spk
    return best_spk


def _majority(speakers: Sequence[Optional[str]]) -> Optional[str]:
    """Most common non-None speaker; deterministic tie-break by label."""
    vals = [s for s in speakers if s is not None]
    if not vals:
        return None
    return min(set(vals), key=lambda s: (-vals.count(s), s))


def _assign_tracks_from_diarization(sentence_speakers, ordered) -> Optional[np.ndarray]:
    """Map each sentence's majority speaker to track 0 (lead) / 1 (second).
    Returns labels, or None if the split is too lopsided to trust (caller then
    falls back to register clustering)."""
    if len(ordered) < 2:
        return None
    lead, second = ordered[0], ordered[1]
    labels = []
    for spk in sentence_speakers:
        labels.append(1 if spk == second else 0)  # unknown/third -> lead track
    labels = np.asarray(labels, dtype=int)
    share = float((labels == 1).mean())
    if (labels == 1).sum() < 2 or (labels == 0).sum() < 2 or not (0.10 <= share <= 0.90):
        return None
    return labels


def _speaker_active_fraction(diarization, spk, t0: float, t1: float) -> float:
    if t1 <= t0:
        return 0.0
    covered = 0.0
    for start, end, s in diarization:
        if s != spk:
            continue
        covered += max(0.0, min(end, t1) - max(start, t0))
    return covered / (t1 - t0)


def _is_unison(sent: List[Note], diarization, lead, second,
               t0: float, spb: float, thresh: float = 0.4) -> bool:
    """A sentence is unison when both singers are active over >= thresh of its
    span (used by the --unison policy)."""
    if not sent or lead is None or second is None:
        return False
    s_start = t0 + sent[0].start_beat * spb
    s_end = t0 + (sent[-1].start_beat + sent[-1].duration) * spb
    return (_speaker_active_fraction(diarization, lead, s_start, s_end) >= thresh
            and _speaker_active_fraction(diarization, second, s_start, s_end) >= thresh)


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
    diarization: Optional[Sequence[Tuple[float, float, str]]] = None,
    pitch_per_speaker: Optional[dict] = None,
    unison: str = "both",          # "both" | "lead" (shared chorus lines)
    drop_adlibs: bool = False,     # drop ()-bracketed canonical ad-libs
) -> Chart:
    times, f0, conf = pitch
    words = [w for w in words if w.get("text", "").strip()]
    if drop_adlibs:
        words = [w for w in words if not w.get("adlib")]
    if not words:
        raise ValueError("no words to assemble")
    pps = pitch_per_speaker or {}

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
        # speaker for this word (Phase 1 attribution + Phase 2 per-voice pitch)
        wspk = _word_speaker(diarization, wstart, wend) if diarization else None
        spk_pitch = pps.get(wspk) if wspk is not None else None
        weights = [max(1, len(s)) for s in sylls]
        total = sum(weights)
        acc = 0.0
        for si, (s, wt) in enumerate(zip(sylls, weights)):
            sb = wstart + span * (acc / total)
            acc += wt
            se = wstart + span * (acc / total)
            midi = None
            if spk_pitch is not None:
                midi = _median_pitch(spk_pitch[0], spk_pitch[1], spk_pitch[2], sb, se)
            if midi is None:  # fall back to the shared mono pitch track
                midi = _median_pitch(times, f0, conf, sb, se)
            text = (" " + s) if (si == 0 and wi > 0 and not new_sentence) else s
            raw.append({
                "beat": to_beat(sb), "endbeat": to_beat(se), "midi": midi,
                "text": text, "new_sentence": new_sentence and si == 0,
                "speaker": wspk,
            })

    known = [r["midi"] for r in raw if r["midi"] is not None]
    fallback = float(np.median(known)) if known else 60.0
    for r in raw:
        if r["midi"] is None:
            r["midi"] = fallback
    offset = round(float(np.median([r["midi"] for r in raw]))) - 12

    notes: List[Tuple[bool, Note, Optional[str]]] = []
    cursor = -1
    for r in raw:
        start = max(r["beat"], cursor + 1)
        dur = max(1, r["endbeat"] - r["beat"])
        cursor = start + dur
        notes.append((r["new_sentence"],
                      Note(start, dur, int(round(r["midi"])) - offset, r["text"]),
                      r.get("speaker")))

    # group into sentences, tracking each sentence's majority speaker
    sentences: List[List[Note]] = []
    sentence_speakers: List[Optional[str]] = []
    cur: List[Note] = []
    cur_spk: List[Optional[str]] = []
    for is_new, note, spk in notes:
        if is_new and cur:
            sentences.append(cur)
            sentence_speakers.append(_majority(cur_spk))
            cur, cur_spk = [], []
        cur.append(note)
        cur_spk.append(spk)
    if cur:
        sentences.append(cur)
        sentence_speakers.append(_majority(cur_spk))

    preview = round(min(duration * 0.25, max(0.0, t0 + 30 * spb)), 2) if duration > 0 else None
    base = dict(title=title, artist=artist, bpm=bpm, gap_ms=round(gap_ms),
                audio=audio, language=language, creator="usdx-autochart",
                cover=cover, video=video, preview_start=preview)

    # --- duet decision ---
    # In "auto" mode only split when the artist string names multiple performers
    # (e.g. "Carlos Baute y Marta Sánchez"); a single wide-range singer must not
    # be split. Speaker diarization decides the split when available; pitch-register
    # clustering (_kmeans2) is the deterministic fallback when it isn't.
    do_duet = False
    labels = None
    used_diarization = False
    ordered = _order_speakers(diarization) if diarization else []
    allow = duet == "yes" or (duet == "auto" and is_multi_artist(artist))

    if allow and len(sentences) >= 6 and len(ordered) >= 2:
        dl = _assign_tracks_from_diarization(sentence_speakers, ordered)
        if dl is not None:
            labels, do_duet, used_diarization = dl, True, True

    if allow and not do_duet and len(sentences) >= 6:
        med = [float(np.median([n.pitch for n in s])) for s in sentences]
        labels, lo, hi = _kmeans2(med)
        share_hi = float(labels.mean())
        if duet == "yes" or (hi - lo >= duet_min_sep and 0.15 <= share_hi <= 0.85):
            do_duet = True

    if do_duet:
        names = split_artists(artist)
        # track 0 = lead (earliest onset when diarized, else low register);
        # performers are usually listed lead-first, so name them in order.
        n1 = p1 or (names[0] if len(names) >= 2 else "Singer 1")
        n2 = p2 or (names[1] if len(names) >= 2 else "Singer 2")
        lead = ordered[0] if used_diarization else None
        second = ordered[1] if used_diarization else None
        t1: List[List[Note]] = []
        t2: List[List[Note]] = []
        for sent, lab in zip(sentences, labels):
            # unison policy only applies when we have diarization to detect it
            if (used_diarization and unison == "both"
                    and _is_unison(sent, diarization, lead, second, t0, spb)):
                t1.append(sent)
                t2.append(sent)
            elif lab == 0:
                t1.append(sent)
            else:
                t2.append(sent)
        return Chart(lines=[], tracks=[_build_track(t1), _build_track(t2)],
                     p1=n1, p2=n2, **base)

    return Chart(lines=_build_track(sentences), **base)
