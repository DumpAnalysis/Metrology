"""Regression tests added after the review of Version 10.

The review found that a displayed record with zero uncertainty could turn a
rejection into an acceptance, because the estimate was rounded to a place
that the record did not state; that two records for different loads compared
as one quantity; that a tolerance of 1e-9 could hide a gap between two
twelve-digit statements; and that the replay passed a count changed by one.
It also asked for an exact check beside the simulation of Chapter 5.

Each test computes what it expects independently of the code under test:
exact fractions, closed forms and a numerical integral, percentiles from
their definition.
"""
import copy
import json
import math
import random
import re
import unittest
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import run_all
from examples import ch10_coordinated_omission, ch22_record_check
from swmetro.record import ALL, comparable, decide, guard
from swmetro.stats import quantile

ROOT = Path(__file__).resolve().parent.parent
DECISIONS = ("accept", "reject", "undecided")
STALE = ["result.u does not follow from the components"]
ROUNDING = ["rounding decides: wants working values"]


def load(name):
    return json.loads((ROOT / "data" / name).read_text(encoding="utf-8"))


def expected(name):
    return (ROOT / "expected" / f"{name}.txt").read_text(encoding="utf-8")


def record(us, stated, value, limit=100.0, dof=15, **result):
    """A complete record with the given components; `result` sets further fields."""
    rec = load("measurement_record.json")
    rec["components"] = [{"name": f"c{i}", "type": "B", "kind": "stipulated", "u": u}
                         for i, u in enumerate(us)]
    rec["result"].update(value=value, u=stated, dof=dof, **result)
    rec["decision"]["limit"] = limit
    return rec


class RoundedEstimate(unittest.TestCase):
    def test_the_two_rows_of_the_review(self):
        for value, outcome in ((99.9996, "accept"), (100.0004, "reject")):
            working = record([0.0], 0.0, value)
            self.assertEqual(decide(working), (outcome, []), value)
            shown = ch22_record_check.displayed(working)
            # With no uncertainty there is no place to round to: the estimate is kept.
            self.assertEqual(shown["result"]["value"], value)
            self.assertEqual(shown["result"]["value_step"], 0.0)
            self.assertEqual(decide(shown), (outcome, []), value)

    def test_a_three_place_display_must_state_its_increment(self):
        # Both estimates of the review print as 100.000 at three places.
        self.assertEqual({round(v, 3) for v in (99.9996, 100.0004)}, {100.0})
        shown = record([0.0], 0.0, 100.0, significant_digits=2, value_step=0.001)
        # Decimals since Version 13, and exactly these two.
        self.assertEqual(guard(shown)[0], (Decimal("99.9995"), Decimal("100.0005")))
        self.assertEqual(decide(shown), ("unresolved", ROUNDING))
        for value, outcome in ((99.999, "accept"), (100.001, "reject")):
            rec = record([0.0], 0.0, value, significant_digits=2, value_step=0.001)
            self.assertEqual(decide(rec)[0], outcome, value)
        # Until Version 55 the third case was an estimate of 99.9995, accepted at equality. No rounding to
        # 0.001 writes that number, and it is refused now. The equality is kept with a limit between two steps.
        rec = record([0.0], 0.0, 99.9995, significant_digits=2, value_step=0.001)
        self.assertEqual(decide(rec), ("invalid", ["result.value is not a multiple of its step"]))
        rec = record([0.0], 0.0, 100.0, limit=100.0005, significant_digits=2, value_step=0.001)
        self.assertEqual(decide(rec), ("accept", []))
        rec = record([0.0], 0.0, 100.001, limit=100.0005, significant_digits=2, value_step=0.001)
        self.assertEqual(decide(rec), ("unresolved", ROUNDING))
        # Without the increment a rounded record decides nothing.
        bare = record([0.0], 0.0, 100.0, significant_digits=2)
        self.assertEqual(decide(bare), ("unresolved", ["wants result.value_step"]))
        for unknown in (None, "", "<increment>"):
            bare["result"]["value_step"] = unknown
            self.assertEqual(decide(bare), ("unresolved", ["wants result.value_step"]), unknown)

    def test_the_increment_is_checked(self):
        for bad in (-0.001, float("nan"), float("inf"), True, "0.001", [0.1]):
            rec = record([0.0], 0.0, 100.0, significant_digits=2, value_step=bad)
            self.assertEqual(decide(rec), ("invalid", ["result.value_step is not a finite number >= 0"]), bad)
        # Working values are not rounded, and a record may not say both things.
        stored = load("measurement_record.json")
        self.assertEqual(stored["result"]["significant_digits"], ALL)
        self.assertNotIn("value_step", stored["result"])
        for fine in (0, 0.0, None):
            self.assertEqual(decide(record([3.0, 4.0], 5.0, 50.0, value_step=fine)), ("accept", []), fine)
        both = record([3.0, 4.0], 5.0, 50.0, value_step=0.1)
        self.assertEqual(decide(both), ("invalid", ["result.value_step is not 0 for working values"]))

    def test_display_states_the_place_it_used(self):
        stored = load("measurement_record.json")
        shown = ch22_record_check.displayed(stored)
        self.assertEqual((shown["result"]["u"], shown["result"]["value_step"]), (3.1, 0.1))
        (lo, hi), _ = guard(ch22_record_check.altered(shown, {("result", "value"): 194.6}))
        self.assertEqual((f"{lo:.2f}", f"{hi:.2f}"), ("199.87", "200.11"))

    def test_display_withholds_a_decision_and_never_changes_one(self):
        rng = random.Random(10)
        seen = {"same": 0, "withheld": 0, "zero": 0}
        for _ in range(3000):
            us = [rng.choice((0.0, 10 ** rng.uniform(-3, 1))) for _ in range(rng.randint(1, 3))]
            u = math.hypot(*us)
            value = 100.0 + rng.uniform(-3, 3) * (u if u else 0.001)
            working = record(us, u, value, dof=rng.choice((3, 15, 15.7327, 60)))
            exact = decide(working)
            self.assertIn(exact[0], DECISIONS)
            got = decide(ch22_record_check.displayed(working))
            if u == 0:
                seen["zero"] += 1
                self.assertEqual(got, exact, (us, value))
            elif got == exact:
                seen["same"] += 1
            else:
                self.assertEqual(got, ("unresolved", ROUNDING), (us, value))
                seen["withheld"] += 1
        self.assertTrue(seen["same"] > 1000 and seen["withheld"] > 50 and seen["zero"] > 300, seen)


class ExactBounds(unittest.TestCase):
    def rounded(self, us, stated, digits=12):
        return record(us, stated, 60.0, significant_digits=digits, value_step=0.0)

    def test_the_twelve_digit_record_of_the_review(self):
        # 1.00000000000 stands for at most 1.000000000005, and 1.00000000050 for at
        # least 1.000000000495: the two cannot both be true, by less than 1e-9.
        upper = Fraction("1.00000000000") + Fraction(1, 2) * Fraction(10) ** -11
        lower = Fraction("1.00000000050") - Fraction(1, 2) * Fraction(10) ** -11
        self.assertEqual((upper, lower), (Fraction("1.000000000005"), Fraction("1.000000000495")))
        self.assertLess(lower - upper, Fraction(1, 10 ** 9))
        self.assertEqual(decide(self.rounded([1.00000000000], 1.00000000050)), ("invalid", STALE))

    def test_the_last_digit_separates_what_is_consistent(self):
        # One component: the stated value may be the component or its neighbour above,
        # whose interval begins where the component's ends.
        self.assertEqual(decide(self.rounded([1.5], 1.5)), ("accept", []))
        self.assertEqual(decide(self.rounded([1.5], 1.50000000001)), ("accept", []))
        self.assertEqual(decide(self.rounded([1.5], 1.50000000002)), ("invalid", STALE))
        self.assertEqual(decide(self.rounded([1.5], 1.49999999998)), ("invalid", STALE))
        # 3 and 4 at twelve digits allow at most 5 + 7e-12, by the exact squares.
        top = (Fraction(3) + Fraction(5, 10 ** 12)) ** 2 + (Fraction(4) + Fraction(5, 10 ** 12)) ** 2
        for stated, ok in (("5.00000000000", True), ("5.00000000001", True), ("5.00000000002", False)):
            low = (Fraction(stated) - Fraction(5, 10 ** 12)) ** 2
            self.assertEqual(low <= top, ok, stated)
            got = decide(self.rounded([3.0, 4.0], float(stated)))
            self.assertEqual(got, ("accept", []) if ok else ("invalid", STALE), stated)

    def test_working_values_allow_only_the_last_bits(self):
        tenths = [0.1] * 10
        by_sum = math.sqrt(sum(x * x for x in tenths))
        self.assertEqual(decide(record(tenths, by_sum, 50.0)), ("accept", []))
        self.assertEqual(decide(record([3.0, 4.0], 5.0 + 1e-13, 50.0)), ("accept", []))
        for off in (1e-10, 1e-9, -1e-9, 1e-6):
            self.assertEqual(decide(record([3.0, 4.0], 5.0 + off, 50.0)), ("invalid", STALE), off)


class Conditions(unittest.TestCase):
    def setUp(self):
        self.a = load("measurement_record.json")
        self.b = copy.deepcopy(self.a)
        self.a["measurand"]["conditions"] = "open loop, 100 requests/s"
        self.b["measurand"]["conditions"] = "open loop, 1000 requests/s"

    def test_the_two_copies_of_the_review(self):
        self.assertEqual(comparable(self.a, self.b), (["conditions"], []))
        self.assertEqual(comparable(self.a, copy.deepcopy(self.a)), ([], []))
        for unknown in (None, "", "<load, versions, hardware>"):
            self.b["measurand"]["conditions"] = unknown
            self.assertEqual(comparable(self.a, self.b), (["conditions unknown"], []), unknown)
            self.assertEqual(comparable(self.b, self.a), (["conditions unknown"], []), unknown)

    def test_the_load_as_the_variable_under_study(self):
        study = ("conditions",)
        self.assertEqual(comparable(self.a, self.b, across=study), ([], ["across conditions"]))
        # Naming the load excuses that difference and no other, and an unknown load is still unknown.
        self.b["measurand"]["start_event"] = "connection accepted"
        self.assertEqual(comparable(self.a, self.b, across=study), (["start_event"], ["across conditions"]))
        self.b["measurand"]["conditions"] = None
        self.assertEqual(comparable(self.a, self.b, across=study)[0], ["start_event", "conditions unknown"])
        same = copy.deepcopy(self.a)
        self.assertEqual(comparable(self.a, same, across=study), ([], []))

    def test_the_block_of_the_transcript(self):
        text = expected("ch22_record_check")
        block = ("against a record at twice the load\n"
                 "  differs in: conditions\n"
                 "  comparable across conditions: yes\n")
        self.assertIn(block, text)
        # The record of other telemetry was taken under the same load.
        stored = load("measurement_record.json")
        reasons, _ = comparable(stored, ch22_record_check.OTHER)
        self.assertEqual(reasons, ["start_event", "convention", "clock rate relation"])


class Percentiles(unittest.TestCase):
    def test_where_the_stalls_first_appear(self):
        xs = ch10_coordinated_omission.closed_loop()
        self.assertEqual((len(xs), xs[-3:]), (5800, [1.0, 1001.0, 1001.0]))
        # Linear interpolation at rank q (n - 1): at p99.98 the rank is 5797.8402, which is
        # 0.8402 of the way from the last 1 ms to the first 1001 ms.
        rank = 0.9998 * 5799
        self.assertAlmostEqual(rank, 5797.8402, places=6)
        self.assertAlmostEqual(quantile(xs, 0.9998), 1 + (rank - 5797) * 1000, places=6)
        self.assertEqual(f"{quantile(xs, 0.9998):.1f}", "841.2")
        self.assertEqual(quantile(xs, 0.99985), 1001.0)
        # Of the percentiles that Transcript 10.1 prints, p99.99 is the first at 1001 ms.
        printed = [quantile(xs, q) for q in (0.5, 0.99, 0.999, 0.9999)]
        self.assertEqual(printed, [1.0, 1.0, 1.0, 1001.0])
        self.assertIn("p99.9 p99.99 mean", expected("ch10_coordinated_omission"))


class ExactRate(unittest.TestCase):
    A, B = 1.2, 2.8

    def test_the_closed_forms(self):
        a, b = self.A, self.B
        mean = 1000 * math.log(b / a) / (b - a)
        sd = math.sqrt(1e6 / (a * b) - mean ** 2)
        lo, hi = 1000 / (b - (b - a) * 0.025), 1000 / (b - (b - a) * 0.975)
        self.assertEqual([f"{x:.2f}" for x in (mean, sd, lo, hi)], ["529.56", "131.09", "362.32", "806.45"])
        # The same by integrating over the uniform duration (midpoint rule).
        n = 200_000
        ts = [a + (b - a) * (i + 0.5) / n for i in range(n)]
        m1 = sum(1000 / t for t in ts) / n
        m2 = sum((1000 / t) ** 2 for t in ts) / n
        self.assertAlmostEqual(m1, mean, places=6)
        self.assertAlmostEqual(math.sqrt(m2 - m1 * m1), sd, places=5)
        # The rate falls as the duration rises, so a share p of rates lies below 1000 / t_p.
        for p, r in ((0.025, lo), (0.975, hi)):
            self.assertAlmostEqual(sum(1000 / t <= r for t in ts) / n, p, places=4)

    def test_the_transcript(self):
        text = expected("ch05_propagation")
        for line in ("MC     R = 530  u = 131", "MC 95 %  [362, 806]",
                     "exact  R = 529.56  u = 131.09", "exact 95 %  [362.32, 806.45]"):
            self.assertIn(line + "\n", text)


class WholeNumbers(unittest.TestCase):
    def test_the_two_substitutions_of_the_review(self):
        t10 = expected("ch10_coordinated_omission")
        self.assertEqual(run_all.compare("exact", t10, t10), [])
        self.assertTrue(run_all.compare("exact", t10, t10.replace("5800", "5799")))
        # Transcript 21.1 now prints counts, so that one event and none are different texts.
        t21 = expected("ch21_psi")
        row = re.search(r"^\s+500\s+\S+\s+\S+\s+(\d+)\s+\S+\s+(\d+)$", t21, re.M)
        self.assertEqual(row.group(1), "1")
        changed = t21[:row.start(1)] + "0" + t21[row.end(1):]
        self.assertTrue(run_all.compare("exact", t21, changed))
        self.assertIn("above crit. 160 of 300", t21)

    def test_whole_numbers_are_identical_and_decimals_have_their_tolerance(self):
        for old, new in (("n 5800", "n 5799"), ("1 of 1000", "0 of 1000"), ("count 0", "count 1"), ("k -3", "k -2")):
            self.assertTrue(run_all.compare("exact", old + "\n", new + "\n"), (old, new))
        self.assertEqual(run_all.compare("exact", "n 5800\n", "n 5800\n"), [])
        self.assertEqual(run_all.compare("exact", "x 0.105\n", "x 0.106\n"), [])
        self.assertTrue(run_all.compare("exact", "x 0.105\n", "x 0.107\n"))
        self.assertEqual(run_all.compare("live", "n 5800\n", "n 12\n"), [])


if __name__ == "__main__":
    unittest.main()
