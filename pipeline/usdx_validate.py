"""Load-guard: re-check a generated chart against the rules USDX enforces in
src/base/USong.pas before it will show a song. Returns a list of problems;
empty list == USDX should load it.
"""
from __future__ import annotations

from typing import List

from .usdx_writer import Chart

MIN_BPM = 1.0  # UNote.pas MIN_BPM guard


def validate(chart: Chart) -> List[str]:
    problems: List[str] = []

    # Required headers (USong.pas done-flags: TITLE|ARTIST|AUDIO|BPM)
    if not chart.title.strip():
        problems.append("missing #TITLE")
    if not chart.artist.strip():
        problems.append("missing #ARTIST")
    if not chart.audio.strip():
        problems.append("missing #AUDIO/#MP3")
    if chart.bpm < MIN_BPM:
        problems.append(f"#BPM below MIN_BPM ({chart.bpm})")

    tracks = chart.tracks if chart.tracks else [chart.lines]
    all_notes = [n for track in tracks for line in track for n in line.notes]
    if not all_notes:
        problems.append("no notes")
        return problems

    # USDX needs at least one scored (non-freestyle) note for a singable line.
    if all(n.kind == "freestyle" for n in all_notes):
        problems.append("all notes are freestyle (nothing to score)")

    # Beats must be monotonically non-decreasing within each track.
    for ti, track in enumerate(tracks):
        last = -1
        for line in track:
            for n in line.notes:
                if n.start_beat < last:
                    problems.append(
                        f"track {ti}: non-monotonic beat {n.start_beat} after {last}")
                    break
                last = n.start_beat
                if n.duration < 1:
                    problems.append(
                        f"track {ti}: zero/negative duration at beat {n.start_beat}")
                    break

    return problems
