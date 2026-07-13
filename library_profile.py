"""Style profile of UltraStar charts: what hand-made .txt files look like.

Scans a directory tree of charts (e.g. the hand-made community library) and
dumps per-metric distributions as JSON. The percentiles derived from the
library are the provenance for the style constants in pipeline/assemble.py;
rerun this after changing the library to re-check them. Also point it at a
generated song folder to compare its distributions against the library's.

Usage:
    python library_profile.py --lib "D:/Canciones Karaoke"
    python library_profile.py --lib "C:/.../USDX/game/songs" --out gen_profile.json

All time metrics are in the time domain (seconds/ms) so differing BPM/GAP
choices don't bias the stats (same rationale as evaluate.py).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from pipeline.usdx_parse import read_file

PCTS = (10, 25, 50, 75, 90, 95, 99)


def _dist(values: List[float]) -> dict:
    if not values:
        return {"n": 0}
    a = np.asarray(values, dtype=float)
    d = {"n": int(a.size), "mean": round(float(a.mean()), 3)}
    for p in PCTS:
        d[f"p{p}"] = round(float(np.percentile(a, p)), 3)
    return d


def profile_chart(path: Path) -> Dict[str, list]:
    """Per-chart metric samples; raises on parse failure."""
    text = path.read_text(encoding="utf-8", errors="replace")
    if "#RELATIVE" in text.upper().split("E\n")[0]:
        # relative-mode beats restart per line; our parser reads them as absolute
        raise ValueError("relative-mode chart")
    chart = read_file(str(path))
    if not chart.lines or chart.bpm <= 0:
        raise ValueError("empty or headerless chart")
    spb = chart.seconds_per_beat()

    m: Dict[str, list] = {
        "note_duration_ms": [], "pitch_step": [], "notes_per_line": [],
        "line_span_s": [], "intra_line_gap_ms": [], "inter_line_gap_s": [],
    }
    n_notes = 0
    n_hold = 0
    n_golden = 0
    pitches: List[int] = []
    prev_line_end = None
    for line in chart.lines:
        sung = [n for n in line.notes if n.kind != "freestyle"]
        if not sung:
            continue
        m["notes_per_line"].append(len(sung))
        span = (sung[-1].start_beat + sung[-1].duration - sung[0].start_beat) * spb
        m["line_span_s"].append(span)
        if prev_line_end is not None:
            m["inter_line_gap_s"].append(max(0.0, (sung[0].start_beat - prev_line_end) * spb))
        prev_line_end = sung[-1].start_beat + sung[-1].duration
        for a, b in zip(sung, sung[1:]):
            m["pitch_step"].append(abs(b.pitch - a.pitch))
            m["intra_line_gap_ms"].append(
                max(0.0, (b.start_beat - (a.start_beat + a.duration)) * spb * 1000.0))
        for n in sung:
            n_notes += 1
            m["note_duration_ms"].append(n.duration * spb * 1000.0)
            pitches.append(n.pitch)
            if n.text.strip().startswith("~"):
                n_hold += 1
            if n.kind in ("golden", "rapgolden"):
                n_golden += 1

    m["hold_note_fraction"] = [n_hold / n_notes] if n_notes else []
    m["golden_fraction"] = [n_golden / n_notes] if n_notes else []
    m["pitch_range"] = [max(pitches) - min(pitches)] if pitches else []
    return m


def profile_library(lib: Path) -> dict:
    pooled: Dict[str, list] = {}
    charts = 0
    skipped: List[str] = []
    for path in sorted(lib.rglob("*.txt")):
        if path.name.lower() in ("license.txt", "readme.txt"):
            continue
        try:
            m = profile_chart(path)
        except Exception as e:  # noqa: BLE001 - skip unparseable, keep scanning
            skipped.append(f"{path.name}: {e}")
            continue
        charts += 1
        for k, v in m.items():
            pooled.setdefault(k, []).extend(v)
    return {
        "library": str(lib),
        "charts": charts,
        "skipped": len(skipped),
        "skipped_files": skipped[:20],
        "metrics": {k: _dist(v) for k, v in sorted(pooled.items())},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lib", required=True, help="directory tree of song folders")
    ap.add_argument("--out", default=None, help="write JSON here (default: stdout)")
    args = ap.parse_args()

    result = profile_library(Path(args.lib))
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out} ({result['charts']} charts, {result['skipped']} skipped)")
    else:
        print(text)


if __name__ == "__main__":
    main()
