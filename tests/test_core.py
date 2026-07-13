"""Core (no-ML) tests: timing math, writer/parser round-trip, validator,
syllabifier, and a stats dump of the bundled reference charts.

Run:  python -m tests.test_core      (from the usdx-autochart dir)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import evaluate, usdx_parse, usdx_validate, usdx_writer  # noqa: E402
from pipeline.align import build_words, tokenize_lines  # noqa: E402
from pipeline.assemble import assemble, syllabify_es  # noqa: E402
from pipeline.usdx_writer import Chart, Line, Note  # noqa: E402
import numpy as np  # noqa: E402

# Golden benchmarks live under benchmarks/ at the repo root.
ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks")
DUET_DIR = os.path.join(ROOT, "Carlos Baute y Marta Sánchez - Colgando en tus manos")
REFS = [
    os.path.join(ROOT, "Alejandro Sanz - Corazón partío",
                 "Alejandro Sanz - Corazón partío.txt"),
    os.path.join(DUET_DIR, "Carlos Baute y Marta Sánchez - Colgando en tus manos.txt"),
]
MULTI = os.path.join(DUET_DIR,
                     "Carlos Baute y Marta Sánchez - Colgando en tus manos [MULTI].txt")


def test_timing_matches_usdx():
    # USDX: time = GAP/1000 + Beat*60/(fileBPM*4) = GAP/1000 + Beat*15/fileBPM
    c = Chart("t", "a", bpm=321.0, gap_ms=27300, audio="x.mp3")
    assert abs(c.seconds_per_beat() - 15.0 / 321.0) < 1e-9
    # Corazón partío: first note beat 0 -> exactly GAP
    assert abs(c.beat_to_time(0) - 27.3) < 1e-6
    # beat 4 -> 27.3 + 4*15/321
    assert abs(c.beat_to_time(4) - (27.3 + 4 * 15.0 / 321.0)) < 1e-9
    print("OK timing math")


def test_roundtrip():
    chart = Chart(
        "Title", "Artist", bpm=240.0, gap_ms=11068, audio="a.mp3",
        lines=[
            Line(notes=[Note(2, 1, 18, "Qui"), Note(4, 1, 18, "zá")]),
            Line(break_beat=46, notes=[Note(48, 3, 20, "Con")]),
        ],
    )
    text = usdx_writer.render(chart)
    assert text.endswith("E\n")
    back = usdx_parse.parse(text)
    flat = [n for l in back.lines for n in l.notes]
    assert len(flat) == 3
    assert flat[0].text == "Qui" and flat[0].pitch == 18
    assert back.bpm == 240.0 and back.gap_ms == 11068
    assert usdx_validate.validate(back) == []
    print("OK writer/parser round-trip")


def test_syllabify():
    cases = {
        "coincidencia": ["co", "in", "ci", "den", "cia"],
        "corazón": ["co", "ra", "zón"],
        "manos": ["ma", "nos"],
        "prohibido": ["prohi", "bi", "do"],  # heuristic groups 'pr'
    }
    for word, _expected in cases.items():
        got = syllabify_es(word)
        assert "".join(got) == word, (word, got)
        print(f"  syllabify {word!r} -> {got}")
    print("OK syllabifier (joins back to original)")


def test_reference_charts_parse():
    for ref in REFS:
        if not os.path.isfile(ref):
            print(f"  (missing) {ref}")
            continue
        chart = usdx_parse.read_file(ref)
        notes = [n for l in chart.lines for n in l.notes]
        span = chart.beat_to_time(notes[-1].start_beat) - chart.beat_to_time(notes[0].start_beat)
        problems = usdx_validate.validate(chart)
        print(f"  {os.path.basename(ref)}")
        print(f"     bpm={chart.bpm} gap={chart.gap_ms}ms notes={len(notes)} "
              f"lines={len(chart.lines)} span={span:.1f}s problems={problems}")


def test_parse_keep_tracks():
    if not os.path.isfile(MULTI):
        print(f"  (missing) {MULTI}")
        return
    chart = usdx_parse.read_file(MULTI, keep_tracks=True)
    assert chart.tracks and len(chart.tracks) == 2, "expected 2 P-tracks"
    assert chart.lines == [], "lines should be empty when tracks are kept"
    assert chart.p1 and chart.p2, (chart.p1, chart.p2)
    n1 = sum(len(l.notes) for l in chart.tracks[0])
    n2 = sum(len(l.notes) for l in chart.tracks[1])
    assert n1 > 0 and n2 > 0, (n1, n2)
    # default (flatten) parse must still merge both tracks into one
    flat = usdx_parse.read_file(MULTI)
    assert flat.tracks is None
    assert sum(len(l.notes) for l in flat.lines) == n1 + n2
    print(f"OK keep_tracks: P1={chart.p1!r}({n1}) P2={chart.p2!r}({n2})")


def test_evaluate_duet_identity():
    if not os.path.isfile(MULTI):
        print(f"  (missing) {MULTI}")
        return
    chart = usdx_parse.read_file(MULTI, keep_tracks=True)
    m = evaluate.evaluate_duet(chart, chart)
    assert m["duet"] is True
    assert m["singer_assignment_accuracy"] == 1.0, m["singer_assignment_accuracy"]
    print(f"OK evaluate_duet identity: saa={m['singer_assignment_accuracy']} "
          f"pairing={m['pairing']}")


def test_adlib_tokenize():
    toks = tokenize_lines(["Te amo (te amo) cada día (yeah)"])
    by = {}
    for t in toks:
        by.setdefault(t["text"].lower(), []).append(t)
    assert not by["cada"][0]["adlib"] and not by["día"][0]["adlib"]
    assert not by["te"][0]["adlib"]  # first "Te" is outside parens
    assert any(t["adlib"] for t in by["amo"])  # the bracketed "amo" is an ad-lib
    yeah = by["yeah"][0]
    assert yeah["adlib"] and yeah["key"] == "yeah"  # clean key despite parens
    # unbalanced / nested parens must not crash
    tokenize_lines(["a (b c", "d) e", "((x))", "y)"])
    print("OK adlib tokenize")


def test_adlib_build_words():
    out = build_words(["Hola (backing) mundo"], [
        {"text": "Hola", "start": 0.0, "end": 0.4},
        {"text": "backing", "start": 0.5, "end": 0.9},
        {"text": "mundo", "start": 1.0, "end": 1.4},
    ])
    assert out is not None
    flags = {w["text"].lower(): w["adlib"] for w in out}
    assert flags == {"hola": False, "backing": True, "mundo": False}, flags
    print("OK adlib build_words")


def test_assemble_drops_adlibs():
    n = 200
    pitch = (np.linspace(0, 2.0, n), np.full(n, 220.0), np.full(n, 0.9))
    words = [
        {"text": "Hola", "start": 0.0, "end": 0.4, "line_index": 0, "adlib": False},
        {"text": "backing", "start": 0.5, "end": 0.9, "line_index": 0, "adlib": True},
        {"text": "mundo", "start": 1.0, "end": 1.4, "line_index": 0, "adlib": False},
    ]
    keep = assemble(list(words), pitch, 2.0, "T", "A", "a.mp3", drop_adlibs=False)
    drop = assemble(list(words), pitch, 2.0, "T", "A", "a.mp3", drop_adlibs=True)
    # note texts carry their own leading-space word boundaries; join with ""
    tk = "".join(nt.text for l in keep.lines for nt in l.notes).lower()
    td = "".join(nt.text for l in drop.lines for nt in l.notes).lower()
    assert "backing" in tk and "backing" not in td
    assert "hola" in td and "mundo" in td
    assert all(l.notes for l in drop.lines), "no empty lines after dropping"
    print("OK assemble drops adlibs")


def test_style_guards():
    # 100 fps synthetic pitch track: voice sounds only over the true note spans.
    t = np.linspace(0, 30.0, 3000)
    f0 = np.zeros_like(t)
    conf = np.zeros_like(t)

    def voice(a, b, hz):
        m = (t >= a) & (t < b)
        f0[m] = hz
        conf[m] = 0.9

    voice(3.0, 3.4, 220.0)    # "Voy" truly sung at 3.0s ...
    voice(3.5, 3.9, 220.0)    # "con"
    voice(9.0, 9.4, 220.0)    # "Hoy" (next line)
    voice(9.5, 12.0, 220.0)   # "nooo" held 2.5 s: 220 Hz ...
    f0[(t >= 10.5) & (t < 12.0)] = 246.94  # ... then +2 st (B3) held
    words = [
        {"text": "Voy", "start": 0.1, "end": 3.4, "line_index": 0},  # whisper starts 2.9 s early
        {"text": "con", "start": 3.5, "end": 3.9, "line_index": 0},
        {"text": "Hoy", "start": 9.0, "end": 9.4, "line_index": 1},
        {"text": "no", "start": 9.5, "end": 12.0, "line_index": 1},
    ]
    chart = assemble(words, (t, f0, conf), 30.0, "T", "A", "a.mp3")
    # snap-to-voiced: GAP follows the real 3.0 s onset, not whisper's 0.1 s
    assert chart.gap_ms > 2500, chart.gap_ms
    # held note split into base + '~' continuation at the +2 st change
    texts = [n.text for l in chart.lines for n in l.notes]
    assert "~" in texts, texts
    flat = [n for l in chart.lines for n in l.notes]
    base = next(n for n in flat if n.text.lstrip() == "no")
    holds = [n for n in flat if n.text == "~"]
    assert any(abs(h.pitch - base.pitch) == 2 for h in holds), (base.pitch, [h.pitch for h in holds])
    # minimum duration: nothing shorter than 2 beats unless crammed
    assert all(n.duration >= 1 for n in flat)

    # early first word: two same-line followers agree on a much later time, so
    # the word is re-anchored to abut them (count-in mistimestamp repair)
    voice(1.0, 1.4, 220.0)     # spoken count-in (voiced, fools snapping)
    voice(20.0, 22.0, 220.0)   # the real line
    words2 = [
        {"text": "Sa", "start": 1.0, "end": 1.4, "line_index": 0},
        {"text": "bes", "start": 20.5, "end": 20.9, "line_index": 0},
        {"text": "bien", "start": 21.0, "end": 21.5, "line_index": 0},
    ]
    c2 = assemble(words2, (t, f0, conf), 30.0, "T", "A", "a.mp3")
    assert len(c2.lines) == 1, [len(l.notes) for l in c2.lines]
    assert c2.gap_ms > 18000, c2.gap_ms  # GAP follows the repaired first word

    # gap guard: same line_index but a big hole (no majority) breaks the line
    words3 = [
        {"text": "En", "start": 1.0, "end": 1.4, "line_index": 0},
        {"text": "el", "start": 21.0, "end": 21.4, "line_index": 0},
    ]
    c3 = assemble(words3, (t, f0, conf), 30.0, "T", "A", "a.mp3")
    assert len(c3.lines) == 2, [len(l.notes) for l in c3.lines]
    print("OK style guards (snap-to-voiced GAP, ~ holds, repair, gap guard)")


def test_align_drops_hallucination_runs():
    ww = ([{"text": "intro", "start": 0.0, "end": 0.2}]
          + [{"text": w, "start": 1 + i, "end": 1.4 + i}
             for i, w in enumerate(["hola", "mundo", "feliz"])]
          + [{"text": f"blah{i}", "start": 10 + i, "end": 10.4 + i} for i in range(6)]
          + [{"text": "adios", "start": 20.0, "end": 20.4}]
          + [{"text": f"credit{i}", "start": 30 + i, "end": 30.4 + i} for i in range(6)])
    out = build_words(["hola mundo feliz", "adios"], ww)
    texts = [w["text"] for w in out]
    # out-of-vocab runs of 6 (mid + trailing) dropped; leading intro dropped
    assert texts == ["hola", "mundo", "feliz", "adios"], texts
    # short unmatched runs (<=4) between matches survive
    ww2 = [{"text": w, "start": float(i), "end": i + 0.4}
           for i, w in enumerate(["hola", "extra", "mundo"])]
    out2 = build_words(["hola mundo"], ww2)
    assert [w["text"] for w in out2] == ["hola", "extra", "mundo"]
    # a long unmatched run of *in-vocabulary* words is a repeated chorus: kept
    chorus = ["hola", "mundo", "feliz", "hola", "mundo", "feliz"]
    ww3 = ([{"text": w, "start": 1 + i, "end": 1.4 + i}
            for i, w in enumerate(["hola", "mundo", "feliz"])]
           + [{"text": w, "start": 10 + i, "end": 10.4 + i}
              for i, w in enumerate(chorus)])
    out3 = build_words(["hola mundo feliz"], ww3)
    assert len(out3) == 9, [w["text"] for w in out3]
    # a spoken count-in must not be force-matched to the first lyric words
    ww4 = ([{"text": w, "start": 0.5 * i, "end": 0.5 * i + 0.3}
            for i, w in enumerate(["dos", "tres"])]
           + [{"text": w, "start": 14 + i, "end": 14.4 + i}
              for i, w in enumerate(["sabes", "tu", "muy", "bien"])])
    out4 = build_words(["sabes tu muy bien"], ww4)
    assert [w["text"] for w in out4] == ["sabes", "tu", "muy", "bien"], \
        [(w["text"], w["start"]) for w in out4]
    assert out4[0]["start"] >= 14.0  # timing comes from the real "sabes"
    print("OK align drops hallucination runs")


def test_diarization_split_and_unison():
    # 12 one-word lines; flat pitch so register clustering would NOT split.
    words = [{"text": "la", "start": i * 1.0, "end": i * 1.0 + 0.8,
              "line_index": i} for i in range(12)]
    n = 4000
    pitch = (np.linspace(0, 12.5, n), np.full(n, 220.0), np.full(n, 0.9))

    # lines 0-7 alternate A/B (establishes the split), 8-11 are unison (both)
    diar = [(i * 1.0, i * 1.0 + 0.8, "A" if i % 2 == 0 else "B") for i in range(8)]
    for i in range(8, 12):
        diar += [(i * 1.0, i * 1.0 + 0.9, "A"), (i * 1.0, i * 1.0 + 0.9, "B")]

    both = assemble(list(words), pitch, 13.0, "T", "Ana y Beto", "a.mp3",
                    duet="auto", diarization=diar, unison="both")
    lead = assemble(list(words), pitch, 13.0, "T", "Ana y Beto", "a.mp3",
                    duet="auto", diarization=diar, unison="lead")
    assert both.tracks and both.p1 == "Ana" and both.p2 == "Beto"
    # unison lines duplicated into both tracks
    assert len(both.tracks[0]) == 8 and len(both.tracks[1]) == 8
    # lead policy keeps the unison lines on P1 only
    assert len(lead.tracks[0]) == 8 and len(lead.tracks[1]) == 4

    # without diarization and flat pitch, no split (solo stays solo)
    solo = assemble(list(words), pitch, 13.0, "T", "Ana y Beto", "a.mp3", duet="auto")
    assert solo.tracks is None
    print("OK diarization split + unison policy")


if __name__ == "__main__":
    test_timing_matches_usdx()
    test_roundtrip()
    test_syllabify()
    test_reference_charts_parse()
    test_parse_keep_tracks()
    test_evaluate_duet_identity()
    test_adlib_tokenize()
    test_adlib_build_words()
    test_assemble_drops_adlibs()
    test_style_guards()
    test_align_drops_hallucination_runs()
    test_diarization_split_and_unison()
    print("\nALL CORE TESTS PASSED")
