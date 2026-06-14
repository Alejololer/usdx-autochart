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


def parse(text: str) -> Chart:
    header = {}
    lines: List[Line] = []
    cur = Line()  # first sentence has no preceding break
    started = False

    def flush():
        nonlocal cur
        if cur.notes or cur.break_beat is not None:
            lines.append(cur)
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
            # duet track marker (P1/P2) - flatten for scoring purposes
            continue

    flush()

    audio = header.get("AUDIO") or header.get("MP3") or ""
    return Chart(
        title=header.get("TITLE", ""),
        artist=header.get("ARTIST", ""),
        bpm=_to_float(header["BPM"]) if "BPM" in header else 0.0,
        gap_ms=_to_float(header.get("GAP", "0")),
        audio=audio,
        lines=lines,
        language=header.get("LANGUAGE"),
        genre=header.get("GENRE"),
        year=int(header["YEAR"]) if header.get("YEAR", "").isdigit() else None,
        cover=header.get("COVER"),
        video=header.get("VIDEO"),
    )


def read_file(path: str) -> Chart:
    # Reference charts are CP1252; fall back through utf-8.
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                return parse(f.read())
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return parse(f.read())
