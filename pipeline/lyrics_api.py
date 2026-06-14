"""Look up canonical lyrics by artist/title from LRCLIB (free, no API key).

LRCLIB returns plain lyrics and, often, synced lyrics (LRC, per-line
timestamps). We use the canonical *text* and *line structure* for deterministic
recognition; absolute LRC times are crowd-sourced and can drift, so note timing
still comes from the audio (Whisper word times). See align.py.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from typing import List, Optional, Tuple

API = "https://lrclib.net/api/get"
UA = "usdx-autochart/0.1 (https://github.com/UltraStar-Deluxe)"

_LRC_TS = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\]")


def _get(url: str, timeout: float = 15.0) -> Optional[dict]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except Exception:  # noqa: BLE001 - network/404 -> caller falls back
        return None


def parse_synced(lrc: str) -> List[Tuple[float, str]]:
    """Parse LRC into [(seconds, line_text)] sorted by time."""
    out: List[Tuple[float, str]] = []
    for raw in lrc.splitlines():
        stamps = _LRC_TS.findall(raw)
        text = _LRC_TS.sub("", raw).strip()
        if not text:
            continue
        for mm, ss in stamps:
            out.append((int(mm) * 60 + float(ss), text))
    out.sort(key=lambda x: x[0])
    return out


def fetch_lyrics(artist: str, title: str,
                 duration: Optional[float] = None) -> Optional[dict]:
    """Return {"plain": str, "synced": [(t, line)], "lines": [str]} or None."""
    params = {"artist_name": artist, "track_name": title}
    if duration:
        params["duration"] = str(int(round(duration)))
    data = _get(API + "?" + urllib.parse.urlencode(params))
    # retry without duration (LRCLIB is strict about exact duration match)
    if not data and duration:
        params.pop("duration")
        data = _get(API + "?" + urllib.parse.urlencode(params))
    if not data or data.get("instrumental"):
        return None

    plain = (data.get("plainLyrics") or "").strip()
    synced = parse_synced(data.get("syncedLyrics") or "")
    lines = [ln.strip() for ln in plain.splitlines() if ln.strip()]
    if not lines and synced:
        lines = [t for _, t in synced]
    if not lines:
        return None
    return {"plain": plain, "synced": synced, "lines": lines}
