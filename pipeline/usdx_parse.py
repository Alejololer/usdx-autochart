"""Parse a UltraStar .txt chart into a Chart (used for the load-guard and for
scoring generated output against the bundled reference charts).

Mirrors the grammar in src/base/USong.pas (ReadTXTHeader + LoadOpenedSong):
  header lines start with '#TAG:value'
  note lines:   <type> <startBeat> <durationBeats> <pitch> <lyric>
  break lines:  - <startBeat> [relativeOffset]
  duet tracks:  P1 / P2     end marker: E
"""
from __future__ import annotations

import re
from typing import List, Optional

from .usdx_writer import Chart, Line, Note, NOTE_CHARS

CHAR_TO_KIND = {v: k for k, v in NOTE_CHARS.items()}


def _to_float(value: str) -> float:
    # USDX StrToFloatI18n accepts both ',' and '.' as the decimal separator.
    return float(value.strip().replace(",", "."))


def parse(text: str, *, keep_tracks: bool = False) -> Chart:
    """Parse a chart. By default duet ``P1``/``P2`` markers are flattened into a
    single ``Chart.lines`` (back-compat for tests and the time-domain ``--eval``).
    With ``keep_tracks=True`` the P-blocks are kept as separate ``Chart.tracks``
    (used by ``evaluate_duet`` to score per-singer attribution)."""
    header = {}
    lines: List[Line] = []
    tracks: List[List[Line]] = []
    cur = Line()  # first sentence has no preceding break
    started = False

    def target() -> List[Line]:
        # where flush() appends: the current track in keep_tracks mode, else lines
        return tracks[-1] if (keep_tracks and tracks) else lines

    def flush():
        nonlocal cur
        if cur.notes or cur.break_beat is not None:
            target().append(cur)
        cur = Line()

    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        if line.startswith("#") and not started:
            if ":" in line:
                tag, _, val = line[1:].partition(":")
                header[tag.strip().upper()] = val.strip()
            continue

        token = line[0]
        if token in CHAR_TO_KIND:
            started = True
            # split into 4 fields + remainder lyric (lyric may contain spaces)
            m = re.match(r"^(.)\s+(-?\d+)\s+(\d+)\s+(-?\d+)\s?(.*)$", line)
            if not m:
                continue
            _, sb, dur, pitch, lyric = m.groups()
            cur.notes.append(
                Note(int(sb), int(dur), int(pitch), lyric, CHAR_TO_KIND[token])
            )
        elif token == "-":
            started = True
            parts = line.split()
            flush()
            cur.break_beat = int(parts[1]) if len(parts) > 1 else None
        elif token in ("E",):
            break
        elif token in ("P", "p"):
            # duet track marker. The gold charts use "P 1"/"P 2" (with a space);
            # our writer emits "P1"/"P2" - both have first char 'P', which is all
            # we key on. Flatten by default; start a new track when keep_tracks.
            started = True
            if keep_tracks:
                flush()
                tracks.append([])
            continue

    flush()

    audio = header.get("AUDIO") or header.get("MP3") or ""
    # singer names: gold uses #DUETSINGERP1/P2, our writer uses #P1/#P2
    p1 = header.get("DUETSINGERP1") or header.get("P1")
    p2 = header.get("DUETSINGERP2") or header.get("P2")
    keep = keep_tracks and bool(tracks)
    return Chart(
        title=header.get("TITLE", ""),
        artist=header.get("ARTIST", ""),
        bpm=_to_float(header["BPM"]) if "BPM" in header else 0.0,
        gap_ms=_to_float(header.get("GAP", "0")),
        audio=audio,
        lines=[] if keep else lines,
        tracks=tracks if keep else None,
        p1=p1,
        p2=p2,
        language=header.get("LANGUAGE"),
        genre=header.get("GENRE"),
        year=int(header["YEAR"]) if header.get("YEAR", "").isdigit() else None,
        cover=header.get("COVER"),
        video=header.get("VIDEO"),
    )


def read_file(path: str, *, keep_tracks: bool = False) -> Chart:
    # Reference charts are CP1252; fall back through utf-8.
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                return parse(f.read(), keep_tracks=keep_tracks)
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return parse(f.read(), keep_tracks=keep_tracks)
