"""Serialize an in-memory chart to a UltraStar Deluxe song.txt.

The format and timing model are taken straight from the USDX source:
  - src/base/UNote.pas  GetTimeFromBeat:  time = GAP/1000 + Beat*60/Song.BPM
  - src/base/USong.pas  line 1209:        Song.BPM = fileBPM * 4
=> one USDX beat lasts  15 / fileBPM  seconds, and
   beat = round( (time - GAP/1000) / (15 / fileBPM) )

Required headers (USDX refuses to load otherwise): TITLE, ARTIST, BPM, and
AUDIO (or legacy MP3). We emit a modern 1.0.0 header set but keep MP3 as an
alias so the file also loads on older builds, matching the bundled references.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# USDX note type -> leading character (see USong.pas note grammar / UMusic.pas TNoteType)
NOTE_CHARS = {
    "normal": ":",
    "golden": "*",
    "freestyle": "F",
    "rap": "R",
    "rapgolden": "G",
}


@dataclass
class Note:
    start_beat: int
    duration: int          # in beats; >=1 (zero-duration becomes freestyle in USDX)
    pitch: int             # integer semitone, 0 = C (USDX renders pitch mod 12)
    text: str
    kind: str = "normal"   # key of NOTE_CHARS


@dataclass
class Line:
    """A sentence. `start_beat` is where the '-' break is written before it."""
    notes: List[Note] = field(default_factory=list)
    break_beat: Optional[int] = None  # beat written on the '- <beat>' break line


@dataclass
class Chart:
    title: str
    artist: str
    bpm: float                  # the value written to #BPM (the "file BPM")
    gap_ms: float               # #GAP in milliseconds
    audio: str                  # filename referenced by #AUDIO / #MP3
    lines: List[Line] = field(default_factory=list)
    language: Optional[str] = None
    genre: Optional[str] = None
    year: Optional[int] = None
    creator: str = "usdx-autochart"
    cover: Optional[str] = None
    background: Optional[str] = None
    video: Optional[str] = None
    preview_start: Optional[float] = None  # seconds
    # Duet: when set, `tracks` holds one list of Lines per singer and is written
    # as P1/P2 blocks. `lines` is then ignored. p1/p2 are the singer names.
    tracks: Optional[List[List[Line]]] = None
    p1: Optional[str] = None
    p2: Optional[str] = None

    def seconds_per_beat(self) -> float:
        return 15.0 / self.bpm

    def beat_to_time(self, beat: int) -> float:
        return self.gap_ms / 1000.0 + beat * self.seconds_per_beat()


def _fmt_float(value: float) -> str:
    """USDX accepts both '.' and ',' decimals (StrToFloatI18n). We write '.'.
    Trim trailing zeros so 240.0 -> '240', 240.04 -> '240.04'."""
    s = f"{value:.4f}".rstrip("0").rstrip(".")
    return s if s else "0"


def render(chart: Chart) -> str:
    out: List[str] = []
    add = out.append

    add(f"#TITLE:{chart.title}")
    add(f"#ARTIST:{chart.artist}")
    if chart.language:
        add(f"#LANGUAGE:{chart.language}")
    if chart.genre:
        add(f"#GENRE:{chart.genre}")
    if chart.year:
        add(f"#YEAR:{chart.year}")
    add(f"#CREATOR:{chart.creator}")
    # AUDIO is the modern required tag; MP3 kept for backward compatibility.
    add(f"#MP3:{chart.audio}")
    add(f"#AUDIO:{chart.audio}")
    if chart.cover:
        add(f"#COVER:{chart.cover}")
    if chart.background:
        add(f"#BACKGROUND:{chart.background}")
    if chart.video:
        add(f"#VIDEO:{chart.video}")
    is_duet = bool(chart.tracks)
    if is_duet:
        add(f"#P1:{chart.p1 or 'P1'}")
        add(f"#P2:{chart.p2 or 'P2'}")
    add(f"#BPM:{_fmt_float(chart.bpm)}")
    add(f"#GAP:{_fmt_float(chart.gap_ms)}")
    if chart.preview_start is not None:
        add(f"#PREVIEWSTART:{_fmt_float(chart.preview_start)}")

    def emit_lines(lines: List[Line]) -> None:
        for line in lines:
            if line.break_beat is not None:
                add(f"- {line.break_beat}")
            for n in line.notes:
                ch = NOTE_CHARS.get(n.kind, ":")
                add(f"{ch} {n.start_beat} {n.duration} {n.pitch} {n.text}")

    if is_duet:
        for i, track in enumerate(chart.tracks):
            add(f"P{i + 1}")
            emit_lines(track)
    else:
        emit_lines(chart.lines)

    add("E")
    return "\n".join(out) + "\n"


def write(chart: Chart, path: str, encoding: str = "utf-8") -> None:
    with open(path, "w", encoding=encoding, newline="\r\n") as f:
        f.write(render(chart))
