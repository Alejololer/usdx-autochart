# Handoff: Per-Voice (true per-singer) Duet Generation

Status: **planning**. Owner: _unassigned_. Branch base: `feature/usdx-autochart`.

This document hands off the next major piece of usdx-autochart: generating a
*real* melody+lyric track per singer in a duet, instead of the current
pitch-register approximation.

---

## 1. Why (the gap this closes)

Today the duet split is a **post-hoc approximation**, not real per-voice analysis:

- `pipeline/separate.py:separate_vocals()` runs Demucs `--two-stems=vocals`,
  which yields **one combined vocal stem** — both singers mixed together.
- `pipeline/pitch.py` (CREPE/torchcrepe) is **monophonic**: when two singers
  sing at once it locks onto a single f0 (usually the louder/lower voice).
- `pipeline/assemble.py` decides the duet split with
  `is_multi_artist(artist)` (gate) + `_kmeans2()` clustering of **per-line median
  pitch** into low/high registers → tracks `t_lo` / `t_hi`.

Consequences, measured against the bundled reference (`Carlos Baute y Marta
Sánchez - Colgando en tus manos`):

| metric (flattened eval) | current |
|---|---|
| note-count ratio | ~1.05 |
| onset match vs ref | ~0.77 |
| **pitch contour corr** | **~0.25** |

The low pitch correlation is the symptom. Register clustering misassigns lines
when a singer sings outside their usual register, and **unison/harmony sections
collapse to one voice**. We need a per-voice signal.

## 2. Goal & success criteria

Produce, for a 2-singer track, two USDX tracks (`P1`/`P2`) where each carries
that singer's own notes and lyrics, including correct attribution during
alternation and sensible handling of unison.

Success (vs the reference duet, using a **new per-track evaluator**, see §6):
- Singer-assignment accuracy ≥ 0.80 (fraction of matched notes whose P1/P2 label
  agrees with the reference, under best P1↔P2 pairing).
- Per-track pitch contour corr ≥ 0.5 (today's flattened number is ~0.25).
- No regression on the solo path (must still detect single-singer → one track).
- Deterministic: same input → same output (fixed models, no sampling).

## 3. Proposed architecture (phased)

Build incrementally; each phase is independently shippable and testable.

### Phase 1 — Diarization-driven assignment (biggest win, lowest risk)
Replace the register-clustering label source with **speaker diarization** of the
combined vocal stem.

- New `pipeline/diarize.py`: `diarize(vocals_wav) -> [(start, end, speaker)]`
  using `pyannote.audio` (`pyannote/speaker-diarization-3.1`), `num_speakers=2`
  when `is_multi_artist`. Runs on CUDA (RTX 5060 Ti present).
- In `assemble.assemble()`: assign each **word** (we already have word-level
  start/end from `align.build_words`) to the speaker whose segment overlaps it
  most. Aggregate to per-sentence majority for the line→track decision. Keep
  `_kmeans2` register clustering as the **fallback** when diarization is
  unavailable (no HF token / model load fails).
- Determinism: pin model revision; pyannote inference is deterministic given a
  fixed seed and no VAD randomness — verify and document.

This alone should fix most alternating-duet misassignments. Pitch still comes
from the mono CREPE stem (acceptable for alternation; wrong only during overlap).

### Phase 2 — Per-voice pitch during overlap (multi-f0)
Give each track its own pitch where singers overlap.

- Option A (recommended first try): **multi-f0 / polyphonic transcription** on
  the vocal stem via Spotify `basic-pitch` (lightweight, CPU/GPU, no HF gate).
  It returns simultaneous notes; assign each note to a singer by (a) the
  diarization-dominant speaker in that window and (b) register continuity within
  each singer's recent pitch track.
- Option B: **two-speaker source separation** (SpeechBrain `sepformer-libri2mix`
  / Conv-TasNet) applied to the vocal stem to get two singer waveforms, then run
  the existing CREPE pitch per waveform. Risk: these models are trained on
  *speech*, quality on sung/harmony vocals is unproven — spike before committing.

Wire whichever wins into `pipeline/pitch.py` as `extract_pitch_per_speaker(...)`
and have `assemble` use the speaker-specific f0 for each note when available.

### Phase 3 — Unison policy
When both sing the same words simultaneously (choruses), decide:
- duplicate the shared lines into **both** P1 and P2 tracks (common UltraStar
  practice; what the human chart often does), vs
- keep them on the lead track only.
Make it a flag (`--unison both|lead`, default `both`) and detect unison as
"diarization shows both speakers active" or "multi-f0 shows ≥2 stable notes".

## 4. Exact integration points (files/functions)

- `pipeline/separate.py` — keep `separate_vocals`; the vocals WAV it returns is
  the input to diarization/multi-f0.
- **new** `pipeline/diarize.py` — `diarize(wav, num_speakers=2)`.
- `pipeline/pitch.py` — add per-speaker / multi-f0 extraction alongside existing
  `extract_pitch`.
- `pipeline/assemble.py` — the duet block (search for `do_duet`, `_kmeans2`,
  `split_artists`). Change the **label source** from register clustering to
  diarization; keep clustering as fallback. `_build_track()` and the P1/P2 Chart
  construction stay as-is.
- `generate.py` — add `--diarize {auto,yes,no}` and `--unison {both,lead}`; thread
  the vocals WAV (already produced when `--separate`) into the new stages.
- `app/main.py` — mirror the new flags in `run_job`.
- `requirements.txt` — add `pyannote.audio` (note: needs `HF_TOKEN` + accepting
  model terms) and, for Phase 2, `basic-pitch` (or `speechbrain`). Keep them
  optional with graceful fallback, matching how `demucs` is handled.
- `pipeline/evaluate.py` — add per-track scoring (see §6).

## 5. Dependencies & environment

- `pyannote.audio` requires a Hugging Face token and one-time acceptance of the
  model license. Read `HF_TOKEN` from env; if absent, log and fall back to
  register clustering (do **not** hard-fail).
- GPU: all of pyannote / basic-pitch / sepformer run on CUDA; the cu128 torch
  build is already installed (see `memory/usdx-autochart-tool.md`).
- Determinism caveat: confirm pyannote pinned-revision runs are reproducible;
  if not, set seeds and document any residual nondeterminism.

## 6. Evaluation plan (must build first)

The current `evaluate.evaluate()` **flattens** both tracks, so it cannot measure
singer attribution. Add:

- `evaluate_duet(generated, reference)`:
  1. Parse reference P1/P2 as **separate** tracks (extend `usdx_parse` to retain
     the `P1`/`P2` markers instead of flattening — currently it drops them).
  2. Try both pairings (genP1↔refP1/genP2↔refP2 and the swap); keep the better.
  3. Report per-track note-match, per-track pitch corr, and **singer-assignment
     accuracy** = matched notes with agreeing track label / total matched.
- Ground truth is the bundled `...Colgando...[MULTI].txt` (P1=Carlos line 14,
  P2=Marta line 494) — verified to use sequential `P 1` / `P 2` blocks.
- Keep the solo regression check: `Alejandro Sanz - Corazón partío` must stay
  single-track.

## 7. Risks / open questions

- **Diarization on singing**: pyannote is trained on speech; harmony and fast
  alternation may degrade it. Spike on the reference duet before building out.
- **Unison attribution**: fundamentally ambiguous from one stem; Phase 2 multi-f0
  is the only real fix, and assigning simultaneous notes to singers is heuristic.
- **HF gating** breaks "works offline / no key" — that's why it must be optional.
- **Determinism** of pyannote/basic-pitch needs confirming for the project's
  "deterministic recognition" promise.
- Beat-grid alignment between two tracks: both tracks share one `#BPM`/`#GAP`
  (already true in our writer) — keep per-note beats absolute.

## 8. First concrete steps for the next session

1. Extend `usdx_parse` to retain P1/P2 tracks and add `evaluate_duet`
   (no new deps). Establish the baseline per-track numbers with today's output.
2. Spike `pyannote/speaker-diarization-3.1` on the Colgando vocal stem; eyeball
   segment quality and measure singer-assignment accuracy if we assign by
   diarization vs the current register clustering.
3. If the spike beats clustering, implement Phase 1 behind `--diarize`.
4. Then evaluate whether Phase 2 (multi-f0) is worth it for the pitch number.

---

_Context for whoever picks this up_: the pipeline, USDX timing model, GPU/cu128
setup, and the torchaudio/torchcodec Windows gotcha are recorded in
`C:\Users\acarl\.claude\projects\C--Users-acarl-Documents-USDX\memory\usdx-autochart-tool.md`.
The format spec lives in `README.md` and the USDX source (`src/base/USong.pas`,
`src/base/UNote.pas`).
