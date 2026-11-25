"""Regression tests added after the two reviews of Version 5.

The first review had no findings. The second asked for corrections of
explanation and for worked additions; these tests fix the arithmetic behind
each corrected statement and the behaviour of the added programs.
"""
import copy
import json
import math
import random
import statistics as st
import unittest
from fractions import Fraction
from pathlib import Path

from examples import ch16_bounds, ch16_timing_models, ch23_case, ch23_mix
from swmetro.dist import z_ppf
from swmetro.record import comparable, decide, in_unit, is_unknown, missing

ROOT = Path(__file__).resolve().parent.parent


def load(name):
    return json.loads((ROOT / "data" / name).read_text(encoding="utf-8"))


class CorrectedStatements(unittest.TestCase):
    def test_a_shared_offset_cancels_from_an_interval(self):
        a, b, offset = 100.0, 104.0, 10.0
        self.assertEqual((b + offset) - (a + offset), b - a)
        # A rate error does not cancel: both stamps scaled by 1.001.
        self.assertNotAlmostEqual(1.001 * b - 1.001 * a, b - a, places=6)

    def test_only_a_common_factor_cancels_in_a_ratio(self):
        self.assertEqual(100 / 80, 1.25)
        self.assertEqual((2 * 100) / (2 * 80), 1.25)
        self.assertAlmostEqual((100 + 10) / (80 + 10), 1.2222, places=4)

    def test_variances_are_averaged_before_the_square_root(self):
        n = 200_000
        fs = [(i + 0.5) / n for i in range(n)]
        mean_var = st.fmean(f * (1 - f) for f in fs)
        mean_sd = st.fmean(math.sqrt(f * (1 - f)) for f in fs)
        self.assertAlmostEqual(math.sqrt(mean_var), 1 / math.sqrt(6), places=9)   # 0.408248
        self.assertAlmostEqual(mean_sd, math.pi / 8, places=6)                   # 0.392699
        rng = random.Random(1)
        got = ch16_timing_models.truncation_error(0.5, 20_000, rng)
        self.assertAlmostEqual(got, 0.5, delta=0.01)


class WorkedAdditions(unittest.TestCase):
    def test_request_mix(self):
        rng = random.Random(23)
        before = ch23_mix.record(ch23_mix.BEFORE, rng)
        after = ch23_mix.record(ch23_mix.AFTER, rng)
        for rec, spec in ((before, ch23_mix.BEFORE), (after, ch23_mix.AFTER)):
            for name, n, mean in spec:
                self.assertEqual(len(rec[name]), n)
                self.assertAlmostEqual(st.fmean(rec[name]), mean, places=9)
        self.assertAlmostEqual(ch23_mix.mean(ch23_mix.weighted(before)), 140.0, places=9)
        self.assertAlmostEqual(ch23_mix.mean(ch23_mix.weighted(after)), 414.0, places=9)
        self.assertAlmostEqual(ch23_mix.mean(ch23_mix.weighted(after, before)), 126.0, places=9)
        # Both classes got faster, the workload as a whole got slower.
        std = ch23_mix.p95(ch23_mix.weighted(after, before))
        self.assertLess(std, ch23_mix.p95(ch23_mix.weighted(before)))
        self.assertGreater(ch23_mix.p95(ch23_mix.weighted(after)), ch23_mix.p95(ch23_mix.weighted(before)))
        # Reweighting to its own mix changes nothing.
        self.assertEqual(ch23_mix.p95(ch23_mix.weighted(after, after)), ch23_mix.p95(ch23_mix.weighted(after)))

    def test_margin_decisions(self):
        text = (ROOT / "expected" / "ch23_case.txt").read_text(encoding="utf-8")
        self.assertIn("upper bound, 95 %: 4.8 ms", text)
        self.assertIn("margin 5 ms: slowdown below the margin", text)
        self.assertIn("margin 2 ms: not established", text)
        self.assertAlmostEqual(ch23_case.Z, z_ppf(0.95))

    def test_bounds_from_an_incomplete_capture(self):
        low, high = ch16_bounds.share_bounds(1000, 960, 20)
        # Exact fractions since Version 53; as floats they were 0.02 and 0.06.
        self.assertEqual((low, high), (Fraction(2, 100), Fraction(6, 100)))
        self.assertEqual(ch16_bounds.verdict(low, high, 0.03), "unresolved")
        self.assertEqual(ch16_bounds.verdict(low, high, 0.07), "met")
        self.assertEqual(ch16_bounds.verdict(low, high, 0.01), "missed")
        self.assertEqual(ch16_bounds.share_bounds(1000, 1000, 20), (Fraction(2, 100), Fraction(2, 100)))


class RecordChecks(unittest.TestCase):
    def setUp(self):
        self.rec = load("measurement_record.json")

    def test_unknown_is_not_zero(self):
        self.assertTrue(is_unknown(None))
        self.assertTrue(is_unknown("<a prompt>"))
        self.assertFalse(is_unknown(0))
        self.assertFalse(is_unknown(0.0))
        self.assertEqual(decide(self.rec), ("accept", []))
        self.assertEqual(self.rec["decision"]["outcome"], "accept")
        for section, key in (("capture", "records_lost"), ("result", "u"), ("measurand", "convention")):
            rec = copy.deepcopy(self.rec)
            rec[section][key] = None
            verdict, why = decide(rec)
            self.assertEqual(verdict, "unresolved")
            self.assertIn(f"wants {section}.{key}", why)
        lost = copy.deepcopy(self.rec)
        lost["capture"]["records_lost"] = 3
        self.assertEqual(decide(lost)[0], "unresolved")

    def test_guarded_decision(self):
        for value, want in ((182.1, "accept"), (197.0, "undecided"), (203.0, "undecided"), (212.0, "reject")):
            rec = copy.deepcopy(self.rec)
            rec["result"]["value"] = value
            self.assertEqual(decide(rec)[0], want, value)

    def test_comparison_reports_every_difference(self):
        self.assertEqual(comparable(self.rec, self.rec), ([], []))
        for key in ("quantity", "start_event", "end_event", "convention", "target"):
            other = copy.deepcopy(self.rec)
            other["measurand"][key] = "something else"
            self.assertEqual(comparable(self.rec, other)[0], [key])
        other = copy.deepcopy(self.rec)
        other["measurand"]["unit"] = "s"
        self.assertEqual(comparable(self.rec, other), ([], ["1 s = 1000 ms"]))
        other["result"].update(value=0.1821, u=0.00306)
        value, u = map(float, in_unit(other, "ms"))   # decimals since Version 13
        self.assertAlmostEqual(value, 182.1, places=9)
        self.assertAlmostEqual(u, 3.06, places=9)
        other["measurand"]["unit"] = "MiB"
        self.assertEqual(comparable(self.rec, other)[0], ["unit"])
        other = copy.deepcopy(self.rec)
        other["capture"]["clock"] = "another host"
        self.assertEqual(comparable(self.rec, other)[0], ["clock rate relation"])
        other["capture"]["clock_relations"] = [{
            "clocks": ["another host", self.rec["capture"]["clock"]], "of": "rate",
            "state": "established", "evidence": "both steered by one reference"}]
        self.assertEqual(comparable(self.rec, other)[0], [])

    def test_template_has_the_fields_of_the_record_and_decides_nothing(self):
        blank = load("record_template.json")
        self.assertEqual(decide(blank)[0], "unresolved")
        self.assertGreaterEqual(len(missing(blank)), 13)
        for section in ("measurand", "capture", "result", "decision"):
            absent = set(self.rec[section]) - set(blank[section])
            self.assertEqual(absent, set(), section)
        self.assertIsNone(blank["capture"]["records_lost"])


if __name__ == "__main__":
    unittest.main()
