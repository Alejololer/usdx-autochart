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


def _flatten_track(track_lines, chart: Chart) -> List[TimedNote]:
    out: List[TimedNote] = []
    for line in track_lines:
        for n in line.notes:
            out.append(TimedNote(
                chart.beat_to_time(n.start_beat),
                chart.beat_to_time(n.start_beat + n.duration),
                n.pitch, n.text,
            ))
    out.sort(key=lambda t: t.start)
    return out


def _flatten(chart: Chart) -> List[TimedNote]:
    tracks = chart.tracks if chart.tracks else [chart.lines]
    out: List[TimedNote] = []
    for track in tracks:
        out.extend(_flatten_track(track, chart))
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


def _pair_metrics(gen: List[TimedNote], ref: List[TimedNote],
                  onset_tol: float) -> dict:
    """Per-track metrics for one gen<->ref track pairing (same math as
    ``evaluate`` but scoped to a single track)."""
    pairs = _match(gen, ref, onset_tol)
    onset_err = [abs(g.start - r.start) for g, r in pairs]
    if pairs:
        gmed = statistics.median(g.pitch for g, _ in pairs)
        rmed = statistics.median(r.pitch for _, r in pairs)
        gp = np.array([g.pitch - gmed for g, _ in pairs], dtype=float)
        rp = np.array([r.pitch - rmed for _, r in pairs], dtype=float)
        pitch_corr = float(np.corrcoef(gp, rp)[0, 1]) if len(gp) > 1 and gp.std() and rp.std() else 0.0
        pitch_within2 = float(np.mean(np.abs(gp - rp) <= 2))
    else:
        pitch_corr, pitch_within2 = 0.0, 0.0
    lyric = difflib.SequenceMatcher(None, _norm_text(gen), _norm_text(ref)).ratio()
    return {
        "gen_notes": len(gen),
        "ref_notes": len(ref),
        "note_count_ratio": round(len(gen) / len(ref), 3) if ref else 0.0,
        "matched": len(pairs),
        "match_rate_vs_ref": round(len(pairs) / len(ref), 3) if ref else 0.0,
        "onset_err_ms_median": round(statistics.median(onset_err) * 1000, 1) if onset_err else None,
        "pitch_contour_corr": round(pitch_corr, 3),
        "pitch_within_2st_rate": round(pitch_within2, 3),
        "lyric_similarity": round(lyric, 3),
    }


def evaluate_duet(generated: Chart, reference: Chart, onset_tol: float = 0.3) -> dict:
    """Score a 2-track duet per singer. Tries both gen<->ref pairings, keeps the
    one with more matched notes, and reports per-track metrics plus
    ``singer_assignment_accuracy`` (matched ref notes whose nearest gen note lands
    on the agreeing track). Returns ``{"duet": False}`` if either side isn't a
    2-track duet - the caller should fall back to the flattened ``evaluate``."""
    if not (generated.tracks and len(generated.tracks) >= 2
            and reference.tracks and len(reference.tracks) >= 2):
        return {"duet": False}

    g = [_flatten_track(t, generated) for t in generated.tracks[:2]]
    r = [_flatten_track(t, reference) for t in reference.tracks[:2]]

    # ref-track-index -> gen-track-index for each candidate pairing
    pairings = {"direct": {0: 0, 1: 1}, "swap": {0: 1, 1: 0}}

    def total_matched(mapping) -> int:
        return sum(len(_match(g[mapping[rt]], r[rt], onset_tol)) for rt in (0, 1))

    best = max(pairings, key=lambda name: total_matched(pairings[name]))
    mapping = pairings[best]

    p1m = _pair_metrics(g[mapping[0]], r[0], onset_tol)
    p2m = _pair_metrics(g[mapping[1]], r[1], onset_tol)

    # singer-assignment accuracy: for each ref note, compare the nearest gen note
    # on its paired track against the nearest on the other track. Correct iff the
    # paired track is at least as close (ties -> correct, so unison/identical
    # charts score 1.0). Notes with no gen note within tol on either track are
    # unattributed and excluded.
    def _nearest(track: List[TimedNote], t: float) -> Optional[float]:
        best = None
        for n in track:
            d = abs(n.start - t)
            if d <= onset_tol and (best is None or d < best):
                best = d
        return best

    correct = total = 0
    for rt in (0, 1):
        want, other = mapping[rt], mapping[1 - rt]
        for rn in r[rt]:
            d_same = _nearest(g[want], rn.start)
            d_other = _nearest(g[other], rn.start)
            if d_same is None and d_other is None:
                continue
            total += 1
            if d_same is not None and (d_other is None or d_same <= d_other):
                correct += 1

    return {
        "duet": True,
        "pairing": best,
        "singer_assignment_accuracy": round(correct / total, 3) if total else 0.0,
        "attributed_notes": total,
        "p1": p1m,
        "p2": p2m,
    }


def format_report_duet(metrics: dict) -> str:
    if not metrics.get("duet"):
        return "(not a 2-track duet on both sides; use --eval for flattened score)"
    lines = ["=== generated vs reference (per-singer) ==="]
    for k in ("pairing", "singer_assignment_accuracy", "attributed_notes"):
        lines.append(f"  {k:28s}: {metrics[k]}")
    for trk in ("p1", "p2"):
        lines.append(f"  [{trk.upper()}]")
        for k, v in metrics[trk].items():
            lines.append(f"    {k:26s}: {v}")
    return "\n".join(lines)
