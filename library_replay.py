"""Stage-level eval harness (Phase 0 of STRATEGY-2026-07-13): score pipeline
stages *independently* against gold charts from the karaoke library, instead of
only comparing final output (batch_eval) or pooled style stats (library_profile).

Per song, against the gold chart's own words/times (human-verified for exactly
that audio file):

  whisper : word recall + onset/end error of text-matched words
  fa      : MMS_FA forced alignment of the gold lyric text (stand-in for
            canonical lyrics) -> same timing metrics; recall ~1 by construction
  syll    : interior-syllable onset error - proportional spread inside the
            *gold* word span (today's rule, with perfect word timing) vs FA
            char-span cuts (raw), on words where syllabify_es agrees with the
            gold syllable count
  pitch   : CREPE median MIDI inside *gold* note boundaries vs gold pitch,
            relative (medians subtracted): within-2-semitones rate + contour
            correlation - isolates pitch error from timing error
  bpm     : tempo.estimate_bpm (full mix) vs the gold #BPM header, mod
            power-of-two multiple - the Phase-2 musical-grid gate

Resumable like batch_eval: run dir keyed by n+seed, songs with an existing
result/error JSON are skipped; stems/whisper/pitch/FA are cached per song under
cache/<slug>/. Sample is stratified: language group (es/en/other) x gold
words-per-minute tercile. Gold onsets are themselves ~50 ms-grid quantized and
stylistic - differences below ~25-50 ms are noise, not signal.

  python library_replay.py --n 100 --seed 0
  python library_replay.py --n 5 --seed 0            # smoke run
  python library_replay.py --n 100 --seed 0 --aggregate-only
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import math
import os
import random
import re
import statistics
import sys
from collections import defaultdict

import numpy as np

from batch_eval import LANG_CODES, scan_library, slugify
from pipeline import audio_io, forced_align, separate, tempo, usdx_parse
from pipeline.align import _key
from pipeline.assemble import syllabify_es


def log(msg: str) -> None:
    print(f"[replay] {msg}", flush=True)


# ---------- gold chart -> time-domain events ----------

def is_relative(gold_path: str) -> bool:
    """#RELATIVE charts restart beats per line; usdx_parse doesn't model that."""
    try:
        with open(gold_path, "rb") as f:
            head = f.read(4096).decode("latin-1")
        return bool(re.search(r"(?im)^#RELATIVE\s*:\s*yes", head))
    except OSError:
        return False


def gold_words(chart) -> list[dict]:
    """Time-domain words from a parsed gold chart. A note is a syllable; a new
    word starts at a line break, after a note with trailing space, or on a note
    with leading space. '~'/empty notes are holds extending the previous
    syllable. Each word: {text, start, end, line_index, sylls:[{text,start,end}]}."""
    words: list[dict] = []
    for li, line in enumerate(chart.lines):
        word = None
        prev_trail = False
        for n in line.notes:
            raw, txt = n.text, n.text.strip()
            s = chart.beat_to_time(n.start_beat)
            e = chart.beat_to_time(n.start_beat + n.duration)
            if txt in ("~", ""):  # hold continuation
                if word:
                    word["end"] = max(word["end"], e)
                    word["sylls"][-1]["end"] = max(word["sylls"][-1]["end"], e)
                if raw:
                    prev_trail = raw[-1:].isspace()
                continue
            if word is None or prev_trail or raw[:1].isspace():
                if word:
                    words.append(word)
                word = {"text": txt, "start": s, "end": e, "line_index": li,
                        "sylls": [{"text": txt, "start": s, "end": e}]}
            else:
                word["text"] += txt
                word["end"] = max(word["end"], e)
                word["sylls"].append({"text": txt, "start": s, "end": e})
            prev_trail = raw[-1:].isspace() if raw else False
        if word:
            words.append(word)
    return words


def gold_notes(chart) -> list[tuple]:
    """Pitched notes as (start_s, end_s, pitch); freestyle/rap carry no pitch."""
    out = []
    for line in chart.lines:
        for n in line.notes:
            if n.kind in ("normal", "golden"):
                out.append((chart.beat_to_time(n.start_beat),
                            chart.beat_to_time(n.start_beat + n.duration),
                            n.pitch))
    return out


# ---------- matching + metrics ----------

def match_words(gold: list[dict], hyp: list[dict]) -> list[tuple[int, int]]:
    """Text-match gold<->hypothesis words (difflib on normalized keys, same
    rules as align.build_words). Returns (gold_idx, hyp_idx) pairs."""
    ga = [(i, _key(w["text"])) for i, w in enumerate(gold)]
    ha = [(j, _key(w["text"])) for j, w in enumerate(hyp)]
    ga = [(i, k) for i, k in ga if k]
    ha = [(j, k) for j, k in ha if k]
    sm = difflib.SequenceMatcher(a=[k for _, k in ga], b=[k for _, k in ha],
                                 autojunk=False)
    pairs = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag not in ("equal", "replace"):
            continue
        for k in range(min(i2 - i1, j2 - j1)):
            if tag == "replace" and difflib.SequenceMatcher(
                    a=ga[i1 + k][1], b=ha[j1 + k][1]).ratio() < 0.5:
                continue
            pairs.append((ga[i1 + k][0], ha[j1 + k][0]))
    return pairs


def _err_stats(errs: list[float]) -> dict | None:
    if not errs:
        return None
    a = sorted(abs(x) for x in errs)
    return {"n": len(a),
            "median_ms": round(a[len(a) // 2] * 1000, 1),
            "mae_ms": round(sum(a) / len(a) * 1000, 1),
            "p90_ms": round(a[int(0.9 * (len(a) - 1))] * 1000, 1),
            "bias_ms": round(statistics.median(errs) * 1000, 1)}


def timing_stats(gold: list[dict], hyp: list[dict],
                 pairs: list[tuple[int, int]]) -> dict:
    onset = [hyp[j]["start"] - gold[i]["start"] for i, j in pairs]
    end = [hyp[j]["end"] - gold[i]["end"] for i, j in pairs]
    return {"recall": round(len(pairs) / len(gold), 3) if gold else 0.0,
            "onset": _err_stats(onset), "end": _err_stats(end)}


def syllable_metrics(gwords: list[dict], fa_by_gi: dict[int, dict]) -> dict:
    """Interior-syllable onset error on multi-syllable words where syllabify_es
    agrees with the gold note count. 'prop' = proportional char-weight spread
    inside the *gold* word span (isolates the spreading rule from word-timing
    error); 'fa' = raw FA char-span cuts."""
    prop, facut = [], []
    n_words = 0
    for gi, gw in enumerate(gwords):
        sylls = gw["sylls"]
        if len(sylls) < 2:
            continue
        ours = syllabify_es(gw["text"])
        if len(ours) != len(sylls):
            continue
        n_words += 1
        span = gw["end"] - gw["start"]
        weights = [max(1, len(s)) for s in ours]
        total = sum(weights)
        acc = 0
        for i in range(1, len(ours)):
            acc += weights[i - 1]
            prop.append(gw["start"] + span * acc / total - sylls[i]["start"])
        fw = fa_by_gi.get(gi)
        if fw and fw.get("char_spans"):
            norm_s = [forced_align.norm_word(s) for s in ours]
            if all(norm_s) and "".join(norm_s) == fw.get("norm"):
                pos = 0
                for i in range(1, len(ours)):
                    pos += len(norm_s[i - 1])
                    facut.append(fw["char_spans"][pos][0] - sylls[i]["start"])

    def med(errs):
        return (round(statistics.median([abs(x) for x in errs]) * 1000, 1)
                if errs else None)

    return {"n_words": n_words, "n_prop": len(prop), "n_fa": len(facut),
            "prop_median_ms": med(prop), "fa_median_ms": med(facut)}


def pitch_metrics(notes: list[tuple], times, f0, conf) -> dict:
    """CREPE median MIDI inside *gold* note boundaries vs gold pitch (relative)."""
    gold_p, crepe_m = [], []
    for s, e, p in notes:
        mask = (times >= s) & (times < e) & (conf > 0.5) & (f0 > 0)
        if mask.sum() < 2:
            continue
        crepe_m.append(69.0 + 12.0 * math.log2(float(np.median(f0[mask])) / 440.0))
        gold_p.append(p)
    out = {"n_notes": len(notes), "n_scored": len(gold_p),
           "coverage": round(len(gold_p) / len(notes), 3) if notes else 0.0}
    if len(gold_p) >= 10:
        g = np.array(gold_p, float)
        c = np.array(crepe_m, float)
        g -= np.median(g)
        c -= np.median(c)
        out["within_2st"] = round(float(np.mean(np.abs(g - c) <= 2)), 3)
        out["contour_corr"] = (round(float(np.corrcoef(g, c)[0, 1]), 3)
                               if g.std() and c.std() else 0.0)
    return out


def bpm_metric(est: float | None, file_bpm: float) -> dict | None:
    """Deviation of the estimate from the gold #BPM, mod power-of-two multiple
    (gold file BPMs are the musical BPM x2/x4 for grid fineness)."""
    if not est or est <= 0 or not file_bpm:
        return None
    k = file_bpm / est
    p = 2.0 ** round(math.log2(k))
    return {"est": round(est, 1), "file_bpm": file_bpm, "mult": p,
            "dev": round(abs(k / p - 1), 4)}


# ---------- per-song run (cached, resumable) ----------

def _run_song(song: dict, run_dir: str, slug: str, device: str,
              whisper_size: str) -> dict:
    chart = usdx_parse.read_file(song["gold"])
    gwords = gold_words(chart)
    if len(gwords) < 30:
        raise ValueError(f"only {len(gwords)} gold words")
    notes = gold_notes(chart)
    cache = os.path.join(run_dir, "cache", slug)
    os.makedirs(cache, exist_ok=True)

    stem = os.path.join(cache, "vocals.wav")
    if not os.path.exists(stem):
        log("  demucs...")
        stem = separate.separate_vocals(song["mp3"], out_dir=cache) or song["mp3"]
    used_stem = os.path.basename(stem) == "vocals.wav"

    wj = os.path.join(cache, "whisper.json")
    if os.path.exists(wj):
        with open(wj, encoding="utf-8") as f:
            wwords = json.load(f)
    else:
        from pipeline import lyrics
        lang = LANG_CODES.get((chart.language or "").strip().lower())
        log(f"  whisper {whisper_size} (lang={lang or 'auto'})...")
        wwords = lyrics.transcribe_words(stem, model_size=whisper_size,
                                         language=lang, device=device)
        with open(wj, "w", encoding="utf-8") as f:
            json.dump(wwords, f, ensure_ascii=False)

    fj = os.path.join(cache, "fa.json")
    if os.path.exists(fj):
        with open(fj, encoding="utf-8") as f:
            fa = json.load(f)
        if fa.get("failed"):
            fa = None
    else:
        by_line: dict[int, list[str]] = defaultdict(list)
        for w in gwords:
            by_line[w["line_index"]].append(w["text"])
        lines = [" ".join(by_line[li]) for li in sorted(by_line)]
        log("  forced alignment (MMS_FA)...")
        fa = forced_align.align_words(lines, stem, device=device)
        with open(fj, "w", encoding="utf-8") as f:
            json.dump(fa if fa else {"failed": True}, f, ensure_ascii=False)

    pz = os.path.join(cache, "pitch.npz")
    if os.path.exists(pz):
        d = np.load(pz)
        times, f0, conf = d["t"], d["f0"], d["conf"]
    else:
        from pipeline import pitch as pitch_mod
        log("  pitch (CREPE)...")
        audio, sr = audio_io.load_mono(stem, sr=16000)
        times, f0, conf = pitch_mod.extract_pitch(audio, sr, device=device)
        np.savez_compressed(pz, t=times, f0=f0, conf=conf)

    try:
        mix, msr = audio_io.load_mono(song["mp3"], sr=16000)
        bpm_est = tempo.estimate_bpm(mix, msr)
    except Exception:  # noqa: BLE001 - informational metric
        bpm_est = None

    span = gwords[-1]["end"] - gwords[0]["start"]
    w_pairs = match_words(gwords, wwords)
    result = {
        "song": song["name"],
        "lang_group": song.get("lang_group"),
        "wpm": round(len(gwords) / (span / 60), 1) if span > 0 else None,
        "n_gold_words": len(gwords),
        "stem": used_stem,
        "whisper": {"n_words": len(wwords), **timing_stats(gwords, wwords, w_pairs)},
        "pitch": pitch_metrics(notes, times, f0, conf),
        "bpm": bpm_metric(bpm_est, chart.bpm),
    }
    if fa:
        fa_words = fa["words"]
        f_pairs = match_words(gwords, fa_words)
        aligned = sum(1 for w in fa_words if w["score"] > 0)
        result["fa"] = {"ok": True, "score": round(fa["score"], 3),
                        "aligned_frac": round(aligned / len(fa_words), 3),
                        **timing_stats(gwords, fa_words, f_pairs)}
        result["syll"] = syllable_metrics(gwords, {i: fa_words[j] for i, j in f_pairs})
    else:
        result["fa"] = {"ok": False}
        result["syll"] = syllable_metrics(gwords, {})
    return result


def run_song(song: dict, run_dir: str, device: str, whisper_size: str) -> None:
    slug = slugify(song["name"])
    rj = os.path.join(run_dir, "results", f"{slug}.json")
    ej = os.path.join(run_dir, "results", f"{slug}.error.json")
    if os.path.exists(rj) or os.path.exists(ej):
        log(f"[skip] {song['name']} (already done)")
        return
    log(f"[run ] {song['name']}")
    try:
        res = _run_song(song, run_dir, slug, device, whisper_size)
        with open(rj, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    except Exception as e:  # noqa: BLE001 - one bad song must not kill the run
        log(f"       FAILED: {e!r}")
        with open(ej, "w", encoding="utf-8") as f:
            json.dump({"error": repr(e)}, f)
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


# ---------- sampling ----------

def build_manifest(lib: str) -> list[dict]:
    """Usable songs with lang_group + gold wpm; cached (one full-library parse)."""
    os.makedirs("eval_runs", exist_ok=True)
    path = os.path.join("eval_runs", "library_manifest.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        if m.get("lib") == lib:
            return m["songs"]
    songs = scan_library(lib)
    out = []
    for i, s in enumerate(songs):
        if i and i % 1000 == 0:
            log(f"scanning gold charts {i}/{len(songs)}...")
        try:
            if is_relative(s["gold"]):
                continue
            chart = usdx_parse.read_file(s["gold"])
            gw = gold_words(chart)
            if len(gw) < 30:
                continue
            span = gw[-1]["end"] - gw[0]["start"]
            if span < 60:
                continue
            code = LANG_CODES.get((chart.language or "").strip().lower(), "other")
            out.append({"name": s["name"], "mp3": s["mp3"], "gold": s["gold"],
                        "lang_group": code if code in ("es", "en") else "other",
                        "wpm": round(len(gw) / (span / 60), 1)})
        except Exception:  # noqa: BLE001 - unparseable gold: not usable
            continue
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"lib": lib, "songs": out}, f, ensure_ascii=False)
    log(f"manifest: {len(out)} usable songs (cached at {path})")
    return out


def stratified_sample(manifest: list[dict], n: int, seed: int):
    """Proportional allocation over lang_group x wpm tercile, seeded."""
    wpms = sorted(s["wpm"] for s in manifest)
    t1, t2 = wpms[len(wpms) // 3], wpms[2 * len(wpms) // 3]

    def terc(w):
        return 0 if w < t1 else (1 if w < t2 else 2)

    strata: dict[tuple, list] = defaultdict(list)
    for s in manifest:
        strata[(s["lang_group"], terc(s["wpm"]))].append(s)
    keys = sorted(strata)
    total = len(manifest)
    alloc = {k: max(1, round(n * len(strata[k]) / total)) for k in keys}
    while sum(alloc.values()) > n:
        k = max(keys, key=lambda k: alloc[k])
        alloc[k] -= 1
    while sum(alloc.values()) < n:
        k = max(keys, key=lambda k: len(strata[k]) - alloc[k])
        alloc[k] += 1
    rng = random.Random(seed)
    sample = []
    for k in keys:
        sample.extend(rng.sample(strata[k], min(alloc[k], len(strata[k]))))
    return sample, (t1, t2)


# ---------- aggregation ----------

FLAT_FIELDS = ["song", "lang_group", "wpm", "stem", "n_gold_words",
               "w_recall", "w_onset_med_ms", "w_onset_mae_ms", "w_onset_p90_ms",
               "w_bias_ms",
               "fa_ok", "fa_score", "fa_aligned_frac", "fa_recall",
               "fa_onset_med_ms", "fa_onset_mae_ms", "fa_onset_p90_ms",
               "fa_bias_ms",
               "syll_n_words", "syll_prop_med_ms", "syll_fa_med_ms",
               "pitch_coverage", "pitch_within_2st", "pitch_corr",
               "bpm_est", "bpm_mult", "bpm_dev", "error"]

KEY_METRICS = ["w_recall", "w_onset_med_ms", "fa_recall", "fa_onset_med_ms",
               "syll_prop_med_ms", "syll_fa_med_ms",
               "pitch_within_2st", "pitch_corr", "bpm_dev"]


def _flat(r: dict) -> dict:
    w, f = r.get("whisper") or {}, r.get("fa") or {}
    wo, fo = w.get("onset") or {}, f.get("onset") or {}
    sy, pi, bp = r.get("syll") or {}, r.get("pitch") or {}, r.get("bpm") or {}
    return {
        "stem": r.get("stem"), "n_gold_words": r.get("n_gold_words"),
        "w_recall": w.get("recall"), "w_onset_med_ms": wo.get("median_ms"),
        "w_onset_mae_ms": wo.get("mae_ms"), "w_onset_p90_ms": wo.get("p90_ms"),
        "w_bias_ms": wo.get("bias_ms"),
        "fa_ok": f.get("ok"), "fa_score": f.get("score"),
        "fa_aligned_frac": f.get("aligned_frac"), "fa_recall": f.get("recall"),
        "fa_onset_med_ms": fo.get("median_ms"), "fa_onset_mae_ms": fo.get("mae_ms"),
        "fa_onset_p90_ms": fo.get("p90_ms"), "fa_bias_ms": fo.get("bias_ms"),
        "syll_n_words": sy.get("n_words"),
        "syll_prop_med_ms": sy.get("prop_median_ms"),
        "syll_fa_med_ms": sy.get("fa_median_ms"),
        "pitch_coverage": pi.get("coverage"),
        "pitch_within_2st": pi.get("within_2st"), "pitch_corr": pi.get("contour_corr"),
        "bpm_est": bp.get("est"), "bpm_mult": bp.get("mult"), "bpm_dev": bp.get("dev"),
    }


def aggregate(sample: list[dict], run_dir: str, tercs) -> None:
    rows = []
    for s in sample:
        slug = slugify(s["name"])
        row = {"song": s["name"], "lang_group": s["lang_group"], "wpm": s["wpm"]}
        rj = os.path.join(run_dir, "results", f"{slug}.json")
        ej = os.path.join(run_dir, "results", f"{slug}.error.json")
        if os.path.exists(rj):
            with open(rj, encoding="utf-8") as f:
                row.update(_flat(json.load(f)))
        elif os.path.exists(ej):
            with open(ej, encoding="utf-8") as f:
                row["error"] = json.load(f).get("error")
        else:
            row["error"] = "not run"
        rows.append(row)

    csv_path = os.path.join(run_dir, "results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FLAT_FIELDS)
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r.get("w_recall") is not None]
    failed = [r for r in rows if r.get("error")]
    t1, t2 = tercs

    def med_line(rs, label):
        parts = [f"  {label:24s}"]
        for k in KEY_METRICS:
            vals = [r[k] for r in rs if r.get(k) is not None]
            parts.append(f"{statistics.median(vals):9.3f}" if vals else "       --")
        print("".join(parts))

    print(f"\n=== library replay summary ({len(ok)} scored, {len(failed)} failed,"
          f" {len(rows)} total; wpm terciles at {t1:.0f}/{t2:.0f}) ===")
    hdr = ["w_rec", "w_onset", "fa_rec", "fa_onset", "sy_prop", "sy_fa",
           "p_2st", "p_corr", "bpm_dev"]
    print("  " + " " * 24 + "".join(f"{h:>9s}" for h in hdr))
    med_line(ok, "ALL (median)")
    for lg in ("es", "en", "other"):
        rs = [r for r in ok if r["lang_group"] == lg]
        if rs:
            med_line(rs, f"lang={lg} (n={len(rs)})")
    for ti, (lo, hi) in enumerate([(0, t1), (t1, t2), (t2, float("inf"))]):
        rs = [r for r in ok if lo <= r["wpm"] < hi]
        if rs:
            med_line(rs, f"wpm {lo:.0f}-{hi if hi != float('inf') else 999:.0f}"
                         f" (n={len(rs)})")
    fa_ok = [r for r in ok if r.get("fa_ok")]
    print(f"\n  FA succeeded on {len(fa_ok)}/{len(ok)} scored songs")
    bpm_good = [r for r in ok if r.get("bpm_dev") is not None and r["bpm_dev"] < 0.02]
    bpm_all = [r for r in ok if r.get("bpm_dev") is not None]
    if bpm_all:
        print(f"  BPM estimate within 2% of gold (mod octave): "
              f"{len(bpm_good)}/{len(bpm_all)}")
    for r in failed:
        print(f"  FAILED: {r['song']}: {r['error']}")
    print(f"\n  full table: {csv_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default=r"D:\Canciones Karaoke")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--whisper", default="small")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    manifest = build_manifest(args.lib)
    if not manifest:
        return 2
    sample, tercs = stratified_sample(manifest, args.n, args.seed)
    counts = ", ".join(
        f"{lg}={sum(1 for s in sample if s['lang_group'] == lg)}"
        for lg in ("es", "en", "other"))
    log(f"sample: {len(sample)} songs ({counts})")

    run_dir = os.path.join("eval_runs", f"replay-n{args.n}-seed{args.seed}")
    for sub in ("results", "cache"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)

    if not args.aggregate_only:
        for i, song in enumerate(sample, 1):
            log(f"[{i}/{len(sample)}]")
            run_song(song, run_dir, args.device, args.whisper)

    aggregate(sample, run_dir, tercs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
