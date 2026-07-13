"""Batch accuracy eval: run generate.py over a random sample of an existing
karaoke library (folders of "Artist - Title.mp3" + gold .txt) and aggregate
the --eval metrics.

  python batch_eval.py                       # 30 random songs from D:\\Canciones Karaoke
  python batch_eval.py --n 100 --seed 1 --lib "D:/Canciones Karaoke"

Resumable: the run directory is keyed by seed+n; songs with an existing
result/error JSON are skipped, so re-running continues where it left off.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import statistics
import subprocess
import sys

from pipeline import usdx_parse

# gold-chart #LANGUAGE header -> Whisper language code; unknown -> auto-detect
LANG_CODES = {
    "spanish": "es", "español": "es", "espanol": "es",
    "english": "en", "french": "fr", "français": "fr", "german": "de",
    "deutsch": "de", "italian": "it", "portuguese": "pt", "português": "pt",
    "japanese": "ja", "korean": "ko", "chinese": "zh",
    "catalan": "ca", "catalán": "ca", "gallego": "gl",
    "dutch": "nl", "swedish": "sv", "finnish": "fi", "russian": "ru",
    "polish": "pl", "turkish": "tr", "latin": "la",
}

METRICS = ["note_count_ratio", "match_rate_vs_ref", "onset_err_ms_median",
           "pitch_contour_corr", "pitch_within_2st_rate", "lyric_similarity"]


def scan_library(lib: str) -> list[dict]:
    """Folders with exactly one .mp3 and a gold (non-MULTI) .txt."""
    songs = []
    for name in sorted(os.listdir(lib)):
        d = os.path.join(lib, name)
        if not os.path.isdir(d):
            continue
        try:
            files = os.listdir(d)
        except OSError:
            continue
        mp3s = [f for f in files if f.lower().endswith(".mp3")]
        txts = [f for f in files if f.lower().endswith(".txt")]
        golds = [f for f in txts if "[multi]" not in f.lower()]
        multis = [f for f in txts if "[multi]" in f.lower()]
        if len(mp3s) != 1 or not golds:
            continue
        songs.append({
            "name": name,
            "mp3": os.path.join(d, mp3s[0]),
            "gold": os.path.join(d, sorted(golds, key=len)[0]),
            "multi": os.path.join(d, multis[0]) if multis else None,
        })
    return songs


def gold_lang(gold_path: str) -> str:
    try:
        ref = usdx_parse.read_file(gold_path)
        return LANG_CODES.get((ref.language or "").strip().lower(), "auto")
    except Exception:  # noqa: BLE001 - unparseable gold: let whisper detect
        return "auto"


def slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")[:80]


def run_song(song: dict, run_dir: str, device: str, timeout: int,
             diarize: str | None = None) -> None:
    slug = slugify(song["name"])
    result_json = os.path.join(run_dir, "results", f"{slug}.json")
    error_json = os.path.join(run_dir, "results", f"{slug}.error.json")
    if os.path.exists(result_json) or os.path.exists(error_json):
        print(f"  [skip] {song['name']} (already done)")
        return

    lang = gold_lang(song["gold"])
    cmd = [sys.executable, "generate.py", song["mp3"],
           "--separate", "--device", device, "--lang", lang,
           "--outdir", os.path.join(run_dir, "songs"),
           "--eval", song["gold"], "--eval-json", result_json]
    if song["multi"]:
        cmd += ["--eval-duet", song["multi"]]
    if diarize:
        cmd += ["--diarize", diarize]

    print(f"  [run ] {song['name']} (lang={lang}"
          f"{', duet-ref' if song['multi'] else ''})")
    log_path = os.path.join(run_dir, "logs", f"{slug}.log")
    err = None
    try:
        with open(log_path, "w", encoding="utf-8", errors="replace") as logf:
            # ponytail: sequential subprocess per song = crash/OOM isolation;
            # parallelize only if GPU idle time ever matters
            p = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT,
                               timeout=timeout)
        if p.returncode != 0:
            err = f"exit code {p.returncode}"
    except subprocess.TimeoutExpired:
        err = f"timeout after {timeout}s"
    if err is None and not os.path.exists(result_json):
        err = "no metrics produced"
    if err:
        print(f"         FAILED: {err}")
        with open(error_json, "w", encoding="utf-8") as f:
            json.dump({"error": err}, f)


def aggregate(sample: list[dict], run_dir: str) -> None:
    rows = []
    for song in sample:
        slug = slugify(song["name"])
        row = {"song": song["name"], "lang": gold_lang(song["gold"]),
               "has_multi_ref": bool(song["multi"])}
        rj = os.path.join(run_dir, "results", f"{slug}.json")
        ej = os.path.join(run_dir, "results", f"{slug}.error.json")
        if os.path.exists(rj):
            with open(rj, encoding="utf-8") as f:
                m = json.load(f)
            row.update({k: m.get(k) for k in METRICS})
            row["gen_is_duet"] = m.get("gen_is_duet")
            row["problems"] = "; ".join(m.get("problems") or [])
            duet = m.get("duet") or {}
            row["singer_assignment_accuracy"] = duet.get(
                "singer_assignment_accuracy") if duet.get("duet") else None
        elif os.path.exists(ej):
            with open(ej, encoding="utf-8") as f:
                row["error"] = json.load(f).get("error")
        else:
            row["error"] = "not run"
        rows.append(row)

    fields = ["song", "lang", "has_multi_ref", "gen_is_duet", *METRICS,
              "singer_assignment_accuracy", "problems", "error"]
    csv_path = os.path.join(run_dir, "results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r.get("note_count_ratio") is not None]
    failed = [r for r in rows if r.get("error")]
    print(f"\n=== batch eval summary ({len(ok)} scored, {len(failed)} failed, "
          f"{len(rows)} total) ===")
    for k in METRICS + ["singer_assignment_accuracy"]:
        vals = [r[k] for r in ok if r.get(k) is not None]
        if vals:
            print(f"  {k:28s} median {statistics.median(vals):7.3f}   "
                  f"mean {statistics.fmean(vals):7.3f}   n={len(vals)}")
    langs = sorted({r["lang"] for r in ok})
    if len(langs) > 1:
        print("  per-language median match_rate_vs_ref:")
        for lg in langs:
            vals = [r["match_rate_vs_ref"] for r in ok
                    if r["lang"] == lg and r.get("match_rate_vs_ref") is not None]
            if vals:
                print(f"    {lg:5s} {statistics.median(vals):.3f}  (n={len(vals)})")
    for r in failed:
        print(f"  FAILED: {r['song']}: {r['error']}")
    print(f"\n  full table: {csv_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default=r"D:\Canciones Karaoke")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--timeout", type=int, default=900, help="per-song seconds")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="just rebuild results.csv + summary from existing JSONs")
    ap.add_argument("--duets-only", action="store_true",
                    help="sample only songs with a [MULTI].txt duet reference")
    ap.add_argument("--diarize", default=None, choices=["auto", "yes", "no"],
                    help="pass --diarize to generate.py (run dir keyed by it)")
    args = ap.parse_args()

    songs = scan_library(args.lib)
    if args.duets_only:
        songs = [s for s in songs if s["multi"]]
    print(f"library: {len(songs)} usable song folders in {args.lib}"
          f"{' (duets only)' if args.duets_only else ''}")
    if not songs:
        return 2
    sample = random.Random(args.seed).sample(songs, min(args.n, len(songs)))

    key = (f"n{args.n}-seed{args.seed}"
           + ("-duets" if args.duets_only else "")
           + (f"-diar{args.diarize}" if args.diarize else ""))
    run_dir = os.path.join("eval_runs", key)
    for sub in ("results", "logs", "songs"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)

    if not args.aggregate_only:
        for i, song in enumerate(sample, 1):
            print(f"[{i}/{len(sample)}]")
            run_song(song, run_dir, args.device, args.timeout, args.diarize)

    aggregate(sample, run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
