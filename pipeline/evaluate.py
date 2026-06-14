"""Score a generated Chart against a reference (gold) Chart.

Everything is compared in the time domain (seconds) so that differing BPM/GAP
choices between the two charts don't bias the result. Reports note-count ratio,
onset-timing error on matched notes, relative-pitch contour correlation, and a
lyric-similarity ratio.
"""
from __future__ import annotations

import difflib
import re
import statistics
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .usdx_writer import Chart


@dataclass
class TimedNote:
    start: float
    end: float
    pitch: int
    text: str


def _flatten(chart: Chart) -> List[TimedNote]:
    out: List[TimedNote] = []
    tracks = chart.tracks if chart.tracks else [chart.lines]
    for track in tracks:
        for line in track:
            for n in line.notes:
                out.append(TimedNote(
                    chart.beat_to_time(n.start_beat),
                    chart.beat_to_time(n.start_beat + n.duration),
                    n.pitch, n.text,
                ))
    out.sort(key=lambda t: t.start)
    return out


def _match(gen: List[TimedNote], ref: List[TimedNote], tol: float = 0.3):
    """Greedy nearest-onset matching within `tol` seconds."""
    pairs: List[Tuple[TimedNote, TimedNote]] = []
    used = [False] * len(gen)
    for r in ref:
        best, bestd = -1, tol
        for i, g in enumerate(gen):
            if used[i]:
                continue
            d = abs(g.start - r.start)
            if d < bestd:
                best, bestd = i, d
        if best >= 0:
            used[best] = True
            pairs.append((gen[best], r))
    return pairs


def _norm_text(notes: List[TimedNote]) -> str:
    s = "".join(n.text for n in notes).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\sáéíóúñü]", "", s)).strip()


def evaluate(generated: Chart, reference: Chart, onset_tol: float = 0.3) -> dict:
    gen = _flatten(generated)
    ref = _flatten(reference)
    pairs = _match(gen, ref, onset_tol)

    onset_err = [abs(g.start - r.start) for g, r in pairs]
    # relative pitch: subtract each side's median so absolute offset is ignored
    if pairs:
        gmed = statistics.median(g.pitch for g, _ in pairs)
        rmed = statistics.median(r.pitch for _, r in pairs)
        gp = np.array([g.pitch - gmed for g, _ in pairs], dtype=float)
        rp = np.array([r.pitch - rmed for _, r in pairs], dtype=float)
        pitch_corr = float(np.corrcoef(gp, rp)[0, 1]) if len(gp) > 1 and gp.std() and rp.std() else 0.0
        pitch_within2 = float(np.mean(np.abs(gp - rp) <= 2))
    else:
        pitch_corr, pitch_within2 = 0.0, 0.0

    lyric_ratio = difflib.SequenceMatcher(
        None, _norm_text(gen), _norm_text(ref)
    ).ratio()

    return {
        "gen_notes": len(gen),
        "ref_notes": len(ref),
        "note_count_ratio": round(len(gen) / len(ref), 3) if ref else 0.0,
        "matched": len(pairs),
        "match_rate_vs_ref": round(len(pairs) / len(ref), 3) if ref else 0.0,
        "onset_err_ms_median": round(statistics.median(onset_err) * 1000, 1) if onset_err else None,
        "onset_err_ms_mean": round(statistics.mean(onset_err) * 1000, 1) if onset_err else None,
        "pitch_contour_corr": round(pitch_corr, 3),
        "pitch_within_2st_rate": round(pitch_within2, 3),
        "lyric_similarity": round(lyric_ratio, 3),
        "gen_span_s": round(gen[-1].end - gen[0].start, 1) if gen else 0.0,
        "ref_span_s": round(ref[-1].end - ref[0].start, 1) if ref else 0.0,
    }


def format_report(metrics: dict) -> str:
    lines = ["=== generated vs reference ==="]
    for k, v in metrics.items():
        lines.append(f"  {k:24s}: {v}")
    return "\n".join(lines)
