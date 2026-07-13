"""Align canonical lyrics (from LRCLIB) to Whisper's audio-timed words.

Result: the canonical words, in order, each with a start/end time taken from
the matching Whisper word (audio-accurate) or interpolated when Whisper missed
it. Whisper words with no canonical match are dropped when they form the
leading run (spoken intro) or a long run outside the song's vocabulary
(subtitle-credit hallucinations, invented outros); unmatched words that look
like lyrics — short runs, repeated choruses — are kept. Canonical line breaks
become USDX sentence breaks.

This makes the *text* deterministic (it's the looked-up lyric, not a guess)
while keeping *timing* from the audio.

Words bracketed as ``(...)`` in the canonical lyric (backing-vocal ad-libs) are
flagged ``adlib=True`` so assemble can optionally drop them. This only works when
canonical lyrics are found and aligned; with raw Whisper there is no paren signal.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from typing import List, Optional


# longest run of consecutive unmatched Whisper words kept as sung extras;
# longer runs are dropped as hallucinations (credits, invented outros)
_MAX_UNMATCHED_RUN = 4


def _key(word: str) -> str:
    w = unicodedata.normalize("NFKD", word.lower())
    w = "".join(c for c in w if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", w)


def _clean_display(word: str) -> str:
    # strip surrounding punctuation but keep internal apostrophes (e.g. pa')
    return word.strip().strip(".,;:!?\"“”()¿¡").strip()


def _adlib_flags(raw_words: List[str]) -> List[bool]:
    """Per-word flag: True if the word lies inside ``(...)`` parentheses.

    Lyric databases bracket backing-vocal ad-libs, e.g. ``Te amo (te amo) ya``.
    Depth is tracked across words so multi-word spans are covered. A word that
    only opens or only closes a paren is itself part of the span. Unbalanced /
    nested parens are handled gracefully (depth never goes below 0); an unclosed
    ``(`` at line end keeps flagging the rest of that line (best-effort)."""
    flags: List[bool] = []
    depth = 0
    for w in raw_words:
        opens = w.count("(")
        closes = w.count(")")
        # inside if we entered with depth>0, or this word carries any paren
        inside = depth > 0 or opens > 0 or closes > 0
        depth = max(0, depth + opens - closes)
        flags.append(inside)
    return flags


def tokenize_lines(lines: List[str]) -> List[dict]:
    """[{text, key, line_index, new_line, adlib}] for every canonical word.

    ``adlib`` is detected from the raw ``(...)`` parentheses *before*
    ``_clean_display`` strips them, so the display text/key stay clean."""
    toks: List[dict] = []
    for li, line in enumerate(lines):
        words = [w for w in re.split(r"\s+", line) if w.strip()]
        flags = _adlib_flags(words)
        for wi, w in enumerate(words):
            disp = _clean_display(w)
            if not disp:
                continue
            toks.append({
                "text": disp,
                "key": _key(disp),
                "line_index": li,
                "new_line": (wi == 0 and li > 0),
                "adlib": flags[wi],
            })
    return [t for t in toks if t["key"]]


def build_words(lyric_lines: List[str], whisper_words: List[dict]) -> Optional[List[dict]]:
    """Whisper-driven alignment: keep the Whisper words (their audio timing is
    what we trust), but where they line up with the canonical lyric, replace the
    text with the canonical word and adopt the canonical line break. Unmatched
    Whisper words are dropped when they lie before the first match, or form a
    run of > _MAX_UNMATCHED_RUN consecutive words mostly *outside* the song's
    vocabulary — hallucinations ("Subtítulos realizados por...", invented
    outros). Long in-vocabulary runs are repeated choruses (canonical text
    lists them once; difflib matches monotonically) and are kept, as are short
    unmatched runs (real sung words the lyric DB missed). Returns word dicts
    with text/start/end/line_index, or None if the overlap is too poor to trust.
    """
    canonical = tokenize_lines(lyric_lines)
    if not canonical or not whisper_words:
        return None

    a = [t["key"] for t in canonical]
    b = [_key(w["text"]) for w in whisper_words]
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)

    match: dict = {}  # whisper index -> canonical token
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag not in ("equal", "replace"):
            continue
        n = min(i2 - i1, j2 - j1)
        for k in range(n):
            # a "replace" pairs whatever falls between anchors; only trust it
            # when the words actually resemble each other, else a spoken
            # count-in gets labeled as the first lyric line at the wrong time
            if tag == "replace" and difflib.SequenceMatcher(
                    a=canonical[i1 + k]["key"], b=b[j1 + k]).ratio() < 0.5:
                continue
            match[j1 + k] = canonical[i1 + k]

    if len(match) < 0.3 * len(canonical):
        return None  # too little overlap to trust the lookup

    # emit matched words plus unmatched runs that look like real lyrics. Long
    # runs of words outside the song's vocabulary are hallucinations (subtitle
    # credits, invented outros); long runs *inside* it are repeated choruses the
    # canonical text lists once (difflib matches monotonically) - keep those.
    vocab = {t["key"] for t in canonical}

    def _run_is_lyrics(run: List[int]) -> bool:
        if len(run) <= _MAX_UNMATCHED_RUN:
            return True
        inv = sum(1 for j in run if _key(whisper_words[j]["text"]) in vocab)
        return inv >= 0.5 * len(run)

    keep: List[int] = []
    run: List[int] = []
    for j in range(min(match), len(whisper_words)):
        if j in match:
            if _run_is_lyrics(run):
                keep.extend(run)
            run = []
            keep.append(j)
        else:
            run.append(j)
    if run and _run_is_lyrics(run):  # trailing run: same vocabulary test
        keep.extend(run)

    out: List[dict] = []
    prev_line: Optional[int] = None
    for j in keep:
        w = whisper_words[j]
        tok = match.get(j)
        if tok is not None:
            text, line, adlib = tok["text"], tok["line_index"], tok["adlib"]
        else:
            # extra sung word with no canonical match; not a bracketed ad-lib
            text, line, adlib = w["text"].strip(), prev_line, False
        out.append({
            "text": text,
            "start": float(w["start"]),
            "end": float(w["end"]),
            "line_index": line if line is not None else (prev_line or 0),
            "adlib": adlib,
        })
        if line is not None:
            prev_line = line
    return out
