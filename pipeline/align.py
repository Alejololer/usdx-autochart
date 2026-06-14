"""Align canonical lyrics (from LRCLIB) to Whisper's audio-timed words.

Result: the canonical words, in order, each with a start/end time taken from
the matching Whisper word (audio-accurate) or interpolated when Whisper missed
it. Whisper words with no canonical match (spoken intros, hallucinations on the
separated stem) are dropped. Canonical line breaks become USDX sentence breaks.

This makes the *text* deterministic (it's the looked-up lyric, not a guess)
while keeping *timing* from the audio.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from typing import List, Optional


def _key(word: str) -> str:
    w = unicodedata.normalize("NFKD", word.lower())
    w = "".join(c for c in w if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", w)


def _clean_display(word: str) -> str:
    # strip surrounding punctuation but keep internal apostrophes (e.g. pa')
    return word.strip().strip(".,;:!?\"“”()¿¡").strip()


def tokenize_lines(lines: List[str]) -> List[dict]:
    """[{text, key, line_index, new_line}] for every canonical word."""
    toks: List[dict] = []
    for li, line in enumerate(lines):
        words = [w for w in re.split(r"\s+", line) if w.strip()]
        for wi, w in enumerate(words):
            disp = _clean_display(w)
            if not disp:
                continue
            toks.append({
                "text": disp,
                "key": _key(disp),
                "line_index": li,
                "new_line": (wi == 0 and li > 0),
            })
    return [t for t in toks if t["key"]]


def build_words(lyric_lines: List[str], whisper_words: List[dict]) -> Optional[List[dict]]:
    """Whisper-driven alignment: keep every Whisper word (its audio timing is
    what we trust), but where it lines up with the canonical lyric, replace the
    text with the canonical word and adopt the canonical line break. Drop only
    the leading words before the first canonical match (spoken/hallucinated
    intro). Returns word dicts with text/start/end/line_index, or None if the
    overlap is too poor to trust.
    """
    canonical = tokenize_lines(lyric_lines)
    if not canonical or not whisper_words:
        return None

    a = [t["key"] for t in canonical]
    b = [_key(w["text"]) for w in whisper_words]
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)

    match: dict = {}  # whisper index -> canonical token
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("equal", "replace"):
            n = min(i2 - i1, j2 - j1)
            for k in range(n):
                match[j1 + k] = canonical[i1 + k]

    if len(match) < 0.3 * len(canonical):
        return None  # too little overlap to trust the lookup

    j_first = min(match)
    out: List[dict] = []
    prev_line: Optional[int] = None
    for j in range(j_first, len(whisper_words)):
        w = whisper_words[j]
        tok = match.get(j)
        if tok is not None:
            text, line = tok["text"], tok["line_index"]
        else:
            text, line = w["text"].strip(), prev_line   # extra sung word (repeat/adlib)
        out.append({
            "text": text,
            "start": float(w["start"]),
            "end": float(w["end"]),
            "line_index": line if line is not None else (prev_line or 0),
        })
        if line is not None:
            prev_line = line
    return out
