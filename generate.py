"""CLI: audio file -> UltraStar Deluxe song folder (auto-draft).

  python generate.py "song.mp3" --title "..." --artist "..." [--separate]
       [--outdir songs] [--lang es] [--whisper small] [--eval reference.txt]

Stages: (optional) Demucs vocal separation -> librosa pYIN pitch ->
faster-whisper word timestamps -> assemble USDX chart -> validate -> write
folder. With --eval, scores the result against a reference chart.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

from pipeline import (audio_io, align, assemble, evaluate, lyrics, lyrics_api,
                      metadata, pitch, separate, tempo)
from pipeline import usdx_parse, usdx_validate, usdx_writer


def log(msg: str) -> None:
    print(f"[autochart] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--title", default=None)
    ap.add_argument("--artist", default=None)
    ap.add_argument("--lang", default="es")
    ap.add_argument("--whisper", default="small")
    ap.add_argument("--no-vad", action="store_true", help="disable Whisper VAD filter")
    ap.add_argument("--separate", action="store_true", help="run Demucs first")
    ap.add_argument("--device", default="auto", help="auto|cuda|cpu")
    ap.add_argument("--duet", default="auto", help="auto|yes|no (P1/P2 split)")
    ap.add_argument("--no-lyrics-api", action="store_true",
                    help="skip LRCLIB canonical lyrics lookup")
    ap.add_argument("--outdir", default="songs")
    ap.add_argument("--grid", type=float, default=0.0625, help="seconds per beat")
    ap.add_argument("--eval", default=None, help="reference .txt to score against")
    ap.add_argument("--dump-txt", default=None, help="also write the .txt here")
    args = ap.parse_args()

    audio_path = args.audio
    if not os.path.isfile(audio_path):
        log(f"no such file: {audio_path}")
        return 2

    title, artist = metadata.resolve_title_artist(audio_path, args.title, args.artist)

    try:
        import torch
        dev = "cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"
        if dev == "cuda":
            log(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            log("running on CPU (no CUDA)")
    except Exception:  # noqa: BLE001
        dev = "cpu"

    duration = audio_io.duration_seconds(audio_path)
    log(f"duration {duration:.1f}s  title={title!r} artist={artist!r}")

    analysis_path = audio_path
    if args.separate:
        log("separating vocals (Demucs)...")
        voc = separate.separate_vocals(audio_path)
        if voc:
            analysis_path = voc
            log(f"vocals: {voc}")
        else:
            log("Demucs unavailable; analysing the full mix")

    log("loading audio for pitch...")
    audio, sr = audio_io.load_mono(analysis_path, sr=16000)

    try:
        bpm_est = tempo.estimate_bpm(audio, sr)
        log(f"estimated musical tempo ~{bpm_est:.1f} BPM")
    except Exception as e:  # noqa: BLE001 - informational only
        log(f"tempo estimate skipped: {e}")

    log(f"extracting pitch ({'CREPE/GPU' if dev == 'cuda' else 'pYIN/CPU'})...")
    pitch_track = pitch.extract_pitch(audio, sr, device=dev)
    voiced = int((pitch_track[2] > 0.5).sum())
    log(f"pitch frames voiced: {voiced}/{len(pitch_track[0])}")

    log(f"transcribing lyrics (faster-whisper {args.whisper})...")
    words = lyrics.transcribe_words(analysis_path, model_size=args.whisper,
                                    language=args.lang, vad=not args.no_vad, device=dev)
    log(f"whisper words: {len(words)}")

    # Deterministic recognition: replace Whisper's guessed text with canonical
    # lyrics looked up by artist/title, keeping Whisper's audio-derived timing.
    if not args.no_lyrics_api:
        log("looking up canonical lyrics (LRCLIB)...")
        canon = lyrics_api.fetch_lyrics(artist, title, duration)
        if canon:
            aligned = align.build_words(canon["lines"], words)
            if aligned:
                log(f"aligned to canonical lyrics: {len(aligned)} words "
                    f"({len(canon['lines'])} lines)")
                words = aligned
            else:
                log("alignment too weak; keeping Whisper transcription")
        else:
            log("no canonical lyrics found; keeping Whisper transcription")

    log("assembling chart...")
    chart = assemble.assemble(
        words, pitch_track, duration,
        title=title, artist=artist, audio=os.path.basename(audio_path),
        target_grid_s=args.grid, language=args.lang.upper(), duet=args.duet,
    )
    if chart.tracks:
        log(f"DUET detected -> P1 {len(chart.tracks[0])} lines, "
            f"P2 {len(chart.tracks[1])} lines")

    problems = usdx_validate.validate(chart)
    if problems:
        log("VALIDATION PROBLEMS:")
        for p in problems:
            log(f"  - {p}")
    else:
        log("validation OK (USDX should load it)")

    song_dir = os.path.join(args.outdir, f"{artist} - {title}")
    os.makedirs(song_dir, exist_ok=True)
    audio_out = os.path.join(song_dir, os.path.basename(audio_path))
    if os.path.abspath(audio_out) != os.path.abspath(audio_path):
        shutil.copy2(audio_path, audio_out)

    # cover art (extracted from the file or generated) -> #COVER
    cover = metadata.ensure_cover(audio_path, song_dir, title, artist)
    if cover:
        chart.cover = cover
        log(f"cover: {cover}")

    txt_path = os.path.join(song_dir, f"{artist} - {title}.txt")
    usdx_writer.write(chart, txt_path)
    log(f"wrote {txt_path}")
    log(f"song folder ready: {song_dir}  "
        f"({', '.join(sorted(os.listdir(song_dir)))})")
    if args.dump_txt:
        usdx_writer.write(chart, args.dump_txt)

    if args.eval:
        ref = usdx_parse.read_file(args.eval)
        metrics = evaluate.evaluate(chart, ref)
        print(evaluate.format_report(metrics))

    return 0


if __name__ == "__main__":
    sys.exit(main())
