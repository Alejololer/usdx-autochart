# usdx-autochart

Turns an uploaded audio track into a **singable UltraStar Deluxe song folder**
(an auto-draft meant for manual polish in the in-game editor). Standalone Python;
does not modify the USDX codebase — it only targets its file format.

## Pipeline

```
audio ─▶ (optional) Demucs vocal separation
      ─▶ librosa pYIN        → per-frame pitch (f0)
      ─▶ faster-whisper      → word-level lyric timestamps
      ─▶ assemble.py         → syllabify, quantize to beats, group sentences
      ─▶ usdx_validate.py    → load-guard (mirrors USong.pas rules)
      ─▶ usdx_writer.py      → song.txt + copied audio (a USDX song folder)
```

### Timing model (verified against USDX source)
- `src/base/UNote.pas` `GetTimeFromBeat`: `time = GAP/1000 + Beat*60/Song.BPM`
- `src/base/USong.pas:1209`: `Song.BPM = fileBPM * 4`
- ⇒ one USDX beat = `15 / fileBPM` seconds; we write a high `#BPM` for a fine grid.

## Install

```bash
pip install -r requirements.txt        # core + pitch + lyrics (CPU, no torch)
pip install demucs                      # optional vocal separation (pulls torch)
```
Needs `ffmpeg`/`ffprobe` on PATH.

## Use

CLI:
```bash
python generate.py "benchmarks/Alejandro Sanz - Corazón partío/Alejandro Sanz - Corazón partío.mp3" \
    --lang es --whisper small --outdir songs \
    --eval "benchmarks/Alejandro Sanz - Corazón partío/Alejandro Sanz - Corazón partío.txt"
```

Web upload:
```bash
uvicorn app.main:app          # http://127.0.0.1:8000
```

## Tests / evaluation
```bash
python -m tests.test_core     # format math + writer/parser, no ML needed
```
`pipeline/evaluate.py` scores a generated chart against a gold reference in the
time domain (note-count ratio, onset error ms, relative-pitch contour
correlation, lyric similarity) so differing BPM/GAP choices don't bias results.

## Golden benchmarks

`benchmarks/` ships two hand-made reference charts (gold `.txt` + input `.mp3` +
`.avi` video + cover) used to track generation quality, committed via **Git LFS**
(run `git lfs install` once, then `git lfs pull` to fetch the media):
- **Solo** — `Alejandro Sanz - Corazón partío/` (should stay single-track).
- **Duet** — `Carlos Baute y Marta Sánchez - Colgando en tus manos/` (should split
  `P1`/`P2`; the `[MULTI].txt` is the per-singer gold — score it with `--eval-duet`).

See `CLAUDE.md` for the expected metric ranges and how to regenerate/score both.

## Status / limits
- Output is a **draft** — expect to fix timing/lyrics in the in-game editor.
- Duet `P1`/`P2` splitting is implemented (diarization + register clustering); see
  `CLAUDE.md`. Separating two singers from one mixed vocal stem is the hard limit.
- Accuracy depends on vocal clarity; `--separate` helps on dense mixes.
