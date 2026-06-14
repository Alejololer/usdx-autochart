# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Scope

This file covers **`usdx-autochart/`** only — a standalone Python tool that turns an
audio track into a singable UltraStar Deluxe song folder (an auto-draft for manual
polish in the in-game editor). It does **not** modify the surrounding USDX Pascal
codebase; it only targets USDX's `song.txt` file format. Treat the parent repo as
read-only reference (the timing model is derived from `../src/base/`).

## Commands

```bash
# Install (run from usdx-autochart/)
pip install -r requirements.txt        # core + pitch + lyrics (CPU, no torch)
pip install demucs                      # optional vocal separation (pulls torch)
# Needs ffmpeg/ffprobe on PATH.

# Generate a song folder
python generate.py "Artist - Title.mp3" --lang es --whisper small --outdir songs \
    --eval "reference.txt"             # --eval scores output vs a gold chart

# Web upload service (same pipeline, one job per upload, returns a zip)
uvicorn app.main:app                    # http://127.0.0.1:8000

# Tests — core format/timing math, no ML deps needed
python -m tests.test_core               # run from usdx-autochart/
```

There is no pytest harness; `tests/test_core.py` is a plain script with `assert`s
and `print`s, run as a module. Run individual checks by calling their `test_*`
functions in a quick `python -c`/REPL, or temporarily editing the `__main__` block.

## GPU setup (critical for usable output)

GPU is essentially required for good results: CPU pYIN on a full mix finds almost no
voiced frames; CREPE on GPU over a separated vocal stem finds thousands. On this
machine (RTX 5060 Ti, Blackwell sm_120) torch **must** come from the CUDA 12.8 wheels:

```bash
pip install --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Plain `pip install torch` silently installs the **CPU** build (`+cpu`,
`torch.cuda.is_available() == False`). The pipeline auto-detects CUDA and falls back
to CPU; `--device cpu` forces it.

## Architecture

The pipeline is a chain of independent `pipeline/` modules orchestrated twice — once
by `generate.py` (CLI) and once by `app/main.py` (web, in a background thread per
job). Both call the same functions in the same order; keep them in sync when changing
the pipeline.

```
audio ─▶ separate.py   (optional Demucs vocal stem)
      ─▶ pitch.py       CREPE/GPU or pYIN/CPU  → (times, f0_hz, confidence)
      ─▶ lyrics.py      faster-whisper          → word-level [{text,start,end}]
      ─▶ lyrics_api.py + align.py  (canonical lyrics overlay, see below)
      ─▶ assemble.py    syllabify + beat-quantize → Chart (in-memory)
      ─▶ usdx_validate.py  load-guard (mirrors USong.pas)
      ─▶ usdx_writer.py    Chart → song.txt + copied audio
```

**The `Chart`/`Line`/`Note` dataclasses in `usdx_writer.py` are the central data
model.** Everything downstream of `assemble` operates on a `Chart`; `usdx_parse.py`
reads a `song.txt` back into one (used by tests and `--eval`).

### Timing model — the one thing that must be exact
Verified against the USDX Pascal source and asserted in `tests/test_core.py`:
- `../src/base/USong.pas:1209`: `Song.BPM = fileBPM * 4`
- `../src/base/UNote.pas` `GetTimeFromBeat`: `time = GAP/1000 + Beat*60/Song.BPM`
- ⇒ one USDX beat = `15 / fileBPM` seconds; `beat = round((time - GAP/1000) / (15/fileBPM))`

We write a deliberately high `#BPM` (the "file BPM", default `15/0.0625 = 240`) to get
a fine quantization grid. `Chart.beat_to_time` / `seconds_per_beat` encode this; don't
hardcode the factor elsewhere.

### Deterministic lyric recognition (align.py)
Whisper's transcription is a guess. When `lyrics_api.py` finds canonical lyrics on
LRCLIB (free, no key), `align.py` does a **Whisper-driven** merge: it keeps every
sung word's audio timing but replaces the *text* with the matched canonical word and
adopts canonical **line breaks as USDX sentence breaks** (`line_index`). It drops the
pre-lyric intro (fixes GAP drift from intro hallucination) and bails (returns `None`,
keeping raw Whisper) if overlap is under 30%. Whisper runs `temperature=0` for
reproducibility. Skip with `--no-lyrics-api`.

**Ad-lib avoider:** lyric DBs bracket backing-vocal ad-libs as `(...)`. `_adlib_flags`
detects those spans during `tokenize_lines` (before `_clean_display` strips the parens),
sets `adlib=True` on each affected canonical word, and `build_words` propagates it.
`assemble(drop_adlibs=True)` (CLI default `--adlibs drop`) drops flagged words before
sentence grouping, so whole-ad-lib lines vanish cleanly. **Only active when canonical
lyrics are found** — raw Whisper words carry no paren signal.

### assemble.py — chart construction
Syllabifies each word (heuristic Spanish `syllabify_es`), spreads syllables across the
word's time span, assigns each the median MIDI pitch of voiced frames in its window,
quantizes onsets to beats (enforcing strictly increasing beats), and groups into
sentences (by `line_index` if aligned, else by silence gap). A global pitch `offset`
re-centers notes into a singable octave.

**Duet split:** only attempted when the artist string names multiple performers
(`is_multi_artist`: " y ", "&", "feat", ",", …) so a single wide-range singer is never
split. Written as `#P1`/`#P2` + `P1`/`P2` blocks; control with `--duet auto|yes|no`.
The **label source** is, in order of preference:
1. **Speaker diarization** (`pipeline/diarize.py`, pyannote 3.1) when a `diarization`
   segment list is passed — each word is attributed to the most-overlapping speaker,
   sentences take the majority, and tracks are ordered **lead-first by earliest onset**
   (P1 = first to sing). Requires `--separate` (it diarizes the vocal stem) + `HF_TOKEN`.
   Flag `--diarize auto|yes|no` (auto = on for multi-artist).
2. **Register clustering** (`_kmeans2` on per-sentence median pitch) — the deterministic
   fallback used when diarization is unavailable or its split is too lopsided.

**Per-voice pitch (`--multif0 auto|no`):** with diarization on, `extract_pitch_per_speaker`
(basic-pitch multi-f0) gives each note its own singer's f0, falling back to the shared
mono track per-note. **Unison (`--unison both|lead`, default both):** sentences where
both singers are active over ≥40% of the span are duplicated into both tracks (`both`)
or kept on the lead (`lead`); only applies when diarized. Legacy limitation (no
diarization): both singers share one separated stem so CREPE tracks a single pitch.

### Validation (usdx_validate.py)
Re-checks the generated `Chart` against the rules USDX enforces before it will load a
song (required headers, ≥1 non-freestyle note, monotonic beats, duration ≥ 1). Returns
a list of problem strings; empty == USDX should load it. **Mirror any new USDX load
rule here** when you find one in `../src/base/USong.pas`.

### Evaluation (evaluate.py)
`--eval` scores the generated chart against a gold reference **entirely in the time
domain** (seconds), so differing BPM/GAP choices don't bias results. Reports note-count
ratio, onset error (ms), relative-pitch contour correlation (medians subtracted), and
lyric similarity. It **flattens** both tracks (`_flatten`), so it can't measure singer
attribution.

`--eval-duet REF` scores per-singer: `usdx_parse.read_file(ref, keep_tracks=True)` keeps
the P1/P2 blocks (default parse still flattens, for back-compat), then `evaluate_duet`
tries both gen↔ref pairings, keeps the better, and reports per-track metrics plus
**`singer_assignment_accuracy`** (ref notes whose nearest gen note lands on the paired
track; ties count correct so identical/unison charts score 1.0). The bundled
`…Colgando…[MULTI].txt` (P1 Carlos / P2 Marta) is the duet ground truth.

### Benchmark charts
Two hand-made charts in the parent repo are the standing benchmarks for generation
quality — they exercise both code paths and `tests/test_core.py` parses them:
- `../Alejandro Sanz - Corazón partío/` — **solo** benchmark. Should stay solo (single
  performer). Reference replication: note-ratio ~1.10, match ~0.74, onset ~104 ms,
  pitch corr ~0.56, lyric sim ~0.46.
- `../Carlos Baute y Marta Sánchez - Colgando en tus manos/` — **duet** benchmark.
  Should split into P1/P2. Reference replication: note-ratio ~1.05, match ~0.77;
  flattened pitch corr only ~0.25 — inherent, since both singers share one separated
  stem. The folder also ships a `[MULTI].txt` (P1 Carlos / P2 Marta) — use it with
  `--eval-duet` to score per-singer attribution. Measured `singer_assignment_accuracy`:
  **diarization 0.586 vs register-clustering 0.512** (diarization wins, but neither
  hits the 0.80 stretch target — separating two singers from one mixed vocal stem is
  the fundamental limit). `--unison both` (default) duplicates shared lines into both
  tracks, which inflates the *flattened* note-count/lyric numbers vs the single-track
  plain `.txt`; score duets with `--eval-duet`, not `--eval`.

Regenerate and score against either with `--eval`, e.g.:
```bash
python generate.py "../Alejandro Sanz - Corazón partío/Alejandro Sanz - Corazón partío.mp3" \
    --separate --device cuda --lang es \
    --eval "../Alejandro Sanz - Corazón partío/Alejandro Sanz - Corazón partío.txt"
```
When changing the pipeline, run both and check the metrics don't regress.

## Conventions / gotchas

- **`song.txt` is written CRLF, UTF-8** (`usdx_writer.write`). The required headers are
  `#TITLE`, `#ARTIST`, `#BPM`, and `#AUDIO` (USDX refuses to load otherwise); `#MP3` is
  emitted as a backward-compat alias.
- **Never call `torchaudio.save`.** torch ≥2.9 routes it through `torchcodec`, which is
  painful on Windows. `separate.py` deliberately runs Demucs as a library
  (`apply_model`), decodes via the ffmpeg CLI, and writes stems with `soundfile`.
- **ML imports are lazy** — done inside functions, not at module top — so the core path
  and `tests/test_core.py` run without torch/librosa installed. Preserve this. The
  optional deps `pyannote.audio` (diarization) and `basic-pitch` (per-voice pitch)
  follow the same fail-soft pattern as demucs: lazy import, broad `except`, return
  `[]`/`{}` so the pipeline falls back. Never hard-fail on a missing optional dep.
- **Diarization needs `HF_TOKEN` from the env** (accept the model terms at
  `hf.co/pyannote/speaker-diarization-3.1`). `diarize.py` reads it from the environment
  only — never hardcode or commit a token. No token ⇒ logs and falls back to clustering.
- **Determinism** (the project promise): pyannote runs with `num_speakers=2` fixed +
  `torch.manual_seed(0)`; P1/P2 are assigned by earliest onset (not pyannote's arbitrary
  labels). basic-pitch is a fixed CNN with no sampling. When changing these, keep
  same-input→same-output.
- The web service (`app/main.py`) never trusts client filenames for paths
  (`safe_component`/`safe_audio_name`); keep that when touching upload handling.
- Output is explicitly a **draft**; v1 favors load-safety and reproducibility over
  polish. Expect users to fix timing/lyrics in the USDX editor.
