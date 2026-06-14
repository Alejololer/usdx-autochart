"""Metadata + cover handling so a single bare .mp3 yields a complete USDX song
folder. Title/artist resolve from (CLI override) > "Artist - Title" filename >
ID3 tags > Unknown. Cover art is extracted from the file if embedded, otherwise
a simple placeholder is generated.
"""
from __future__ import annotations

import os
import subprocess
from typing import Optional, Tuple


def read_tags(path: str) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format_tags=title,artist,album", "-of", "default=noprint_wrappers=1", path],
        capture_output=True, text=True,
    ).stdout
    tags: dict = {}
    for line in out.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            tags[k.replace("TAG:", "").strip().lower()] = v.strip()
    return tags


def resolve_title_artist(path: str, title: Optional[str],
                         artist: Optional[str]) -> Tuple[str, str]:
    base = os.path.splitext(os.path.basename(path))[0]
    tags = read_tags(path)
    if not title:
        if " - " in base:
            title = base.split(" - ", 1)[1]
        else:
            title = tags.get("title") or base or "Untitled"
    if not artist:
        if " - " in base:
            artist = base.split(" - ", 1)[0]
        else:
            artist = tags.get("artist") or "Unknown"
    return title.strip(), artist.strip()


def _has_embedded_art(path: str) -> bool:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v",
         "-show_entries", "stream_disposition=attached_pic",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True,
    ).stdout
    return "1" in out.split()


def _extract_art(path: str, out_path: str) -> bool:
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path,
                        "-an", "-frames:v", "1", out_path],
                       check=True, capture_output=True)
        return os.path.isfile(out_path) and os.path.getsize(out_path) > 0
    except subprocess.CalledProcessError:
        return False


def _make_placeholder(out_path: str, title: str, artist: str) -> bool:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:  # noqa: BLE001
        return False

    size = 512
    img = Image.new("RGB", (size, size))
    px = img.load()
    for y in range(size):  # vertical gradient
        t = y / size
        px_row = (int(30 + 40 * t), int(20 + 30 * t), int(60 + 80 * t))
        for x in range(size):
            px[x, y] = px_row
    draw = ImageDraw.Draw(img)

    def font(sz: int):
        for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
            try:
                return ImageFont.truetype(name, sz)
            except Exception:  # noqa: BLE001
                continue
        return ImageFont.load_default()

    def centered(text: str, y: int, f, fill):
        w = draw.textlength(text, font=f)
        draw.text(((size - w) / 2, y), text, font=f, fill=fill)

    centered(artist[:28], 200, font(34), (235, 235, 245))
    centered(title[:30], 250, font(28), (180, 200, 255))
    centered("usdx-autochart", size - 40, font(16), (140, 140, 160))
    img.save(out_path, "JPEG", quality=88)
    return True


def ensure_cover(audio_path: str, song_dir: str, title: str,
                 artist: str) -> Optional[str]:
    """Put a cover.jpg in song_dir; return its basename or None."""
    out_path = os.path.join(song_dir, "cover.jpg")
    if _has_embedded_art(audio_path) and _extract_art(audio_path, out_path):
        return "cover.jpg"
    if _make_placeholder(out_path, title, artist):
        return "cover.jpg"
    return None
