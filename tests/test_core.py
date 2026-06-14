"""Core (no-ML) tests: timing math, writer/parser round-trip, validator,
syllabifier, and a stats dump of the bundled reference charts.

Run:  python -m tests.test_core      (from the usdx-autochart dir)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import usdx_parse, usdx_validate, usdx_writer  # noqa: E402
from pipeline.assemble import syllabify_es  # noqa: E402
from pipeline.usdx_writer import Chart, Line, Note  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REFS = [
    os.path.join(ROOT, "Alejandro Sanz - Corazón partío",
                 "Alejandro Sanz - Corazón partío.txt"),
    os.path.join(ROOT, "Carlos Baute y Marta Sánchez - Colgando en tus manos",
                 "Carlos Baute y Marta Sánchez - Colgando en tus manos.txt"),
]


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


if __name__ == "__main__":
    test_timing_matches_usdx()
    test_roundtrip()
    test_syllabify()
    test_reference_charts_parse()
    print("\nALL CORE TESTS PASSED")
