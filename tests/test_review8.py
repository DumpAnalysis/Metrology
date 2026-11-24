"""Regression tests added after the review of Version 8.

The review found that the tolerance allowed between a stated combined
uncertainty and its components could turn an undecided result into an
acceptance and could refuse a correctly rounded budget; that a coherent
record with correlated components was called invalid; that two records whose
runs were weighted differently compared as one quantity; that a report
rounded a value and its expanded uncertainty to different places above the
units; and that ten false alarms a night is an expectation.

Each test computes what it expects independently of the code under test: a
root sum of squares, a tabulated quantile, a percentile by its definition,
rounding done with the formatting of the standard library, a simulation.
"""
import copy
import json
import math
import random
import re
import statistics as st
import unittest
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

from examples import ch22_record_check
from swmetro import Result
from swmetro.core import decimals, fmt
from swmetro.record import ALL, bounds, combined, comparable, decide, spread

ROOT = Path(__file__).resolve().parent.parent
DECISIONS = ("accept", "reject", "undecided")
STALE = ["result.u does not follow from the components"]
ROUNDING = ["rounding decides: wants working values"]
T10 = 1.812  # Student t, 10 degrees of freedom, one-sided 0.95, as tabulated


def load(name):
    return json.loads((ROOT / "data" / name).read_text(encoding="utf-8"))


def step_of(x, n):
    """The increment of the last of n significant digits of x: the place to which a
    report rounds an estimate whose uncertainty is x."""
    if x == 0:
        return 0.0
    return 10.0 ** (int(f"{x:.15e}".split("e")[1]) - n + 1)


def budget(us, stated, value, digits=ALL, limit=100.0, dof=10, risk=0.05, step=None):
    """A complete record with the given components and stated combined u. A rounded
    record states the increment of its estimate, here that of the last digit of u."""
    rec = load("measurement_record.json")
    rec["components"] = [{"name": f"c{i}", "type": "B", "kind": "stipulated", "u": u}
                         for i, u in enumerate(us)]
    rec["result"].update(value=value, u=stated, dof=dof, significant_digits=digits)
    if type(digits) is int and digits > 0:
        rec["result"]["value_step"] = step_of(stated, digits) if step is None else step
    rec["decision"].update(limit=limit, risk=risk)
    return rec


def exact_decimal(q):
    """A fraction with a power of ten below as the decimal that it is."""
    with localcontext() as ctx:
        ctx.prec = 80
        return Decimal(q.numerator) / Decimal(q.denominator)


def sig(x, n):
    """x rounded to n significant digits by the standard library's formatting."""
    return float(f"{x:.{n - 1}e}")


class WorkingPrecision(unittest.TestCase):
    def test_the_two_records_of_the_review(self):
        # What is at stake, from the table alone: 5.0 leaves the question open, 4.8 would accept.
        self.assertGreater(91.2 + T10 * 5.0, 100.0)
        self.assertLessEqual(91.2 + T10 * 4.8, 100.0)
        for digits in (ALL, 2, 3):
            self.assertEqual(decide(budget([3.0, 4.0], 5.0, 91.2, digits)), ("undecided", []), digits)
            self.assertEqual(decide(budget([3.0, 4.0], 4.8, 91.2, digits)), ("invalid", STALE), digits)
        # Rounded to the nearest 0.1, the components allow 4.93 to 5.07 and no more.
        lo, hi = map(float, spread(budget([3.0, 4.0], 5.0, 91.2, 2)))   # decimals since Version 13
        self.assertAlmostEqual(lo, math.sqrt(2.95 ** 2 + 3.95 ** 2), places=12)
        self.assertAlmostEqual(hi, math.sqrt(3.05 ** 2 + 4.05 ** 2), places=12)
        self.assertEqual((round(lo, 2), round(hi, 2)), (4.93, 5.07))

    def test_the_stated_uncertainty_is_not_used_in_the_guard_band(self):
        # 4.9, 5.0 and 5.1 are all possible roundings of what the components allow.
        for stated in (4.9, 5.0, 5.1):
            self.assertEqual(decide(budget([3.0, 4.0], stated, 91.2, 2)), ("undecided", []), stated)
        # With working values an estimate just inside the limit is accepted, whatever was near it.
        inside = 100.0 - 1.8125 * 5.0
        self.assertEqual(decide(budget([3.0, 4.0], 5.0, inside)), ("accept", []))
        self.assertEqual(decide(budget([3.0, 4.0], 5.0, inside + 0.01)), ("undecided", []))

    def test_a_correctly_rounded_budget_is_not_refused(self):
        working = math.hypot(10.49, 0.50)
        self.assertAlmostEqual(working, 10.5019, places=4)
        self.assertEqual((sig(10.49, 2), sig(0.50, 2), sig(working, 2)), (10.0, 0.5, 11.0))
        self.assertEqual(decide(budget([10.49, 0.50], working, 50.0)), ("accept", []))
        # The three displayed numbers, declared as two significant digits: consistent.
        shown = budget([10.0, 0.50], 11.0, 50.0, digits=2)
        self.assertEqual(decide(shown), ("accept", []))
        # 10 at two digits stands for 9.95 to 10.5 (the review of Version 9).
        lo, hi = map(float, spread(shown))
        self.assertAlmostEqual(lo, math.sqrt(9.95 ** 2 + 0.495 ** 2), places=12)
        self.assertAlmostEqual(hi, math.sqrt(10.5 ** 2 + 0.505 ** 2), places=12)
        # The stated 11 stands for 10.5 to 11.5, so both hold between 10.5 and that upper bound.
        both = tuple(map(float, combined(shown)))
        self.assertAlmostEqual(both[0], 10.5, places=12)
        self.assertAlmostEqual(both[1], hi, places=12)
        # The same three numbers declared as working values contradict each other.
        self.assertEqual(decide(budget([10.0, 0.50], 11.0, 50.0)), ("invalid", STALE))

    def test_rounding_that_could_decide_is_unresolved_and_working_values_resolve_it(self):
        working = math.hypot(10.49, 0.50)
        # From the table: the guarded sum is above the limit at working precision.
        self.assertGreater(82.0 + T10 * working, 100.0)
        self.assertLessEqual(82.0 - T10 * working, 100.0)
        self.assertEqual(decide(budget([10.49, 0.50], working, 82.0)), ("undecided", []))
        # At 82 the rounded record now decides too, because the stated 11 narrows what the
        # components allow (the review of Version 9). At 81 it cannot, and the working values can.
        self.assertEqual(decide(budget([10.0, 0.50], 11.0, 82.0, digits=2)), ("undecided", []))
        self.assertGreater(81.0 + T10 * working, 100.0)
        self.assertEqual(decide(budget([10.49, 0.50], working, 81.0)), ("undecided", []))
        self.assertEqual(decide(budget([10.0, 0.50], 11.0, 81.0, digits=2)), ("unresolved", ROUNDING))
        # Far from the limit the rounded record decides, and decides the same.
        for value, want in ((50.0, "accept"), (95.0, "undecided"), (130.0, "reject")):
            self.assertEqual(decide(budget([10.49, 0.50], working, value))[0], want)
            self.assertEqual(decide(budget([10.0, 0.50], 11.0, value, digits=2))[0], want)

    def test_rounding_withholds_a_decision_and_never_changes_one(self):
        rng = random.Random(8)
        counts = {"same": 0, "withheld": 0}
        for _ in range(3000):
            us = [10 ** rng.uniform(-2, 2) for _ in range(rng.randint(1, 4))]
            u = math.hypot(*us)
            limit = 10 ** rng.uniform(0, 3)
            value = limit + u * rng.uniform(-4, 4)
            dof = rng.choice((2, 5, 10, 15.7327, 40.5, 300))
            exact = decide(budget(us, u, value, ALL, limit, dof))
            self.assertIn(exact[0], DECISIONS)
            n = rng.choice((1, 2, 2, 3, 6, 15))
            shown_u = sig(u, n)
            step = 10.0 ** (math.floor(math.log10(shown_u)) - n + 1)
            # Until Version 55: round(value / step) * step, a float that is seldom a multiple of the step
            # as it prints (13.233500000000001 for a step of 1e-05), and such an estimate is refused now.
            grid = Fraction(repr(step))
            shown_value = exact_decimal(grid * round(Fraction(repr(value)) / grid))
            shown = budget([sig(x, n) for x in us], shown_u, shown_value, n, limit, sig(dof, n), step=step)
            got = decide(shown)
            if got == exact:
                counts["same"] += 1
            else:
                self.assertEqual(got, ("unresolved", ROUNDING), (us, value, limit, dof, n))
                counts["withheld"] += 1
        self.assertGreater(counts["same"], 1500)
        self.assertGreater(counts["withheld"], 100)

    def test_the_values_that_round_to_a_number(self):
        cases = ((0.50, 2, 0.495, 0.505), (11.0, 2, 10.5, 11.5), (3.0, 2, 2.95, 3.05), (120.0, 2, 115.0, 125.0),
                 (182.1, 4, 182.05, 182.15), (0.003, 1, 0.0025, 0.0035), (7.0, 1, 6.5, 7.5))
        for x, n, lo, hi in cases:
            got = bounds(x, n)
            self.assertAlmostEqual(got[0], lo, places=12, msg=(x, n))
            self.assertAlmostEqual(got[1], hi, places=12, msg=(x, n))
        self.assertEqual(bounds(3.0, ALL), (3.0, 3.0))
        self.assertEqual(bounds(0.0, 2), (0.0, 0.0))
        # Every number that rounds to x at n digits lies within the bounds of x.
        rng = random.Random(9)
        for _ in range(20000):
            x = 10 ** rng.uniform(-6, 6)
            n = rng.randint(1, 6)
            lo, hi = bounds(sig(x, n), n)
            self.assertTrue(lo * (1 - 1e-12) <= x <= hi * (1 + 1e-12), (x, n))

    def test_the_declaration_of_digits_is_required_and_checked(self):
        for bad in (0, -1, 18, 2.0, True, "two", "ALL", [2]):
            rec = budget([3.0, 4.0], 5.0, 50.0, bad)
            rec["result"]["value_step"] = 0.1
            got, why = decide(rec)
            self.assertEqual((got, why), ("invalid", ["result.significant_digits is not all or 1 to 17"]), bad)
        for unknown in (None, "", "<all, or a count>"):
            got, why = decide(budget([3.0, 4.0], 5.0, 50.0, unknown))
            self.assertEqual((got, why), ("unresolved", ["wants result.significant_digits"]), unknown)
        self.assertEqual(load("measurement_record.json")["result"]["significant_digits"], ALL)

    def test_the_stored_record_holds_working_values(self):
        rec = load("measurement_record.json")
        lo, hi = map(float, combined(rec))
        self.assertEqual(lo, hi)
        self.assertAlmostEqual(rec["result"]["u"], lo, places=12)
        self.assertEqual(decide(rec), ("accept", []))
        # Rounded for display it is still accepted; brought near the limit it is not decided.
        shown = ch22_record_check.displayed(rec)
        self.assertEqual([c["u"] for c in shown["components"]], [1.7, 2.5, 0.3, 0.29])
        self.assertEqual((shown["result"]["u"], shown["result"]["dof"], shown["result"]["value"]), (3.1, 16.0, 182.1))
        self.assertEqual(decide(shown), ("accept", []))
        near = ch22_record_check.altered(shown, {("result", "value"): 194.6})
        self.assertEqual(decide(near), ("unresolved", ROUNDING))
        above = ch22_record_check.altered(shown, {("result", "value"): 194.8})
        self.assertEqual(decide(above), ("undecided", []))
        exact = ch22_record_check.altered(rec, {("result", "value"): 194.6})
        self.assertEqual(decide(exact), ("accept", []))


class Correlated(unittest.TestCase):
    COV = [{"between": ["c0", "c1"], "r": 0.5, "evidence": "stated for the test"}]

    def record(self, stated, **changes):
        rec = budget([3.0, 4.0], stated, 50.0)
        rec["covariances"] = copy.deepcopy(self.COV)
        for (section, key), value in changes.items():
            if section == "":
                rec[key] = value
            else:
                rec[section][key] = value
        return rec

    def test_a_coherent_correlated_record_is_unresolved(self):
        coherent = math.sqrt(9 + 16 + 2 * 0.5 * 3 * 4)
        self.assertAlmostEqual(coherent, math.sqrt(37), places=12)
        for stated in (coherent, 5.0, 4.8):
            self.assertEqual(decide(self.record(stated)), ("unresolved", ["correlations not supported"]), stated)

    def test_impossible_numbers_still_come_first(self):
        got, why = decide(self.record(-1.0))
        self.assertEqual(got, "invalid")
        self.assertIn("result.u is not a finite number >= 0", why)

    def test_the_sum_of_squares_is_checked_only_where_it_applies(self):
        # A stale 4.8 is not reported while the format or the procedure is another one,
        # or while an input is missing; it is reported once the record is in scope.
        other_format = budget([3.0, 4.0], 4.8, 50.0)
        other_format["format"] = "measurement record, version 3"
        self.assertEqual(decide(other_format), ("unresolved", ["format not supported"]))
        other_rule = budget([3.0, 4.0], 4.8, 50.0)
        other_rule["decision"]["procedure"] = "lower limit, t guard band, v1"
        self.assertEqual(decide(other_rule), ("unresolved", ["procedure not supported"]))
        no_limit = budget([3.0, 4.0], 4.8, 50.0, limit=None)
        self.assertEqual(decide(no_limit), ("unresolved", ["wants decision.limit"]))
        self.assertEqual(decide(budget([3.0, 4.0], 4.8, 50.0)), ("invalid", STALE))
        # And before the things that only withhold a decision.
        lost = budget([3.0, 4.0], 4.8, 50.0)
        lost["capture"]["records_lost"] = 3
        self.assertEqual(decide(lost), ("invalid", STALE))


class Weighting(unittest.TestCase):
    @staticmethod
    def p95(xs):
        """The definition: the smallest value with at least 95 % at or below it."""
        xs = sorted(xs)
        for x in xs:
            if 100 * sum(v <= x for v in xs) >= 95 * len(xs):
                return x

    def test_three_numbers_for_three_questions(self):
        table = {
            "all 200 ms": ((200.0, 50, 200.0, 50), (200, 200, 200)),
            "100 and 300 ms": ((100.0, 50, 300.0, 50), (200, 200, 300)),
            "300 ms runs short": ((100.0, 98, 300.0, 2), (200, 104, 100)),
        }
        text = (ROOT / "expected" / "ch22_runs.txt").read_text(encoding="utf-8")
        for name, ((x, n, y, m), want) in table.items():
            runs = [[x] * n, [y] * m] * 10
            each = [self.p95(r) for r in runs]
            equal = sum(each) / len(each)
            weighted = sum(p * len(r) for p, r in zip(each, runs)) / sum(len(r) for r in runs)
            pooled = self.p95([v for r in runs for v in r])
            self.assertEqual((round(equal), round(weighted), round(pooled)), want, name)
            row = re.search(rf"^{re.escape(name)}\s+(\d+)\s+(\d+)\s+(\d+)$", text, re.M)
            self.assertEqual(tuple(int(g) for g in row.groups()), want, name)
        self.assertEqual((98 * 100 + 2 * 300) / 100, 104)

    def test_records_weighted_differently_are_not_one_quantity(self):
        rec = load("measurement_record.json")
        self.assertEqual(rec["data"]["run_weighting"], "equal")
        self.assertEqual(comparable(rec, copy.deepcopy(rec)), ([], []))
        other = copy.deepcopy(rec)
        other["data"]["run_weighting"] = "by requests"
        self.assertEqual(comparable(rec, other), (["run_weighting"], []))
        self.assertEqual(comparable(other, rec), (["run_weighting"], []))
        for unknown in (None, "", "<equal, by requests ...>"):
            other["data"]["run_weighting"] = unknown
            self.assertEqual(comparable(rec, other), (["run_weighting unknown"], []), unknown)
        bare = {k: v for k, v in rec.items() if k != "data"}
        self.assertEqual(comparable(rec, bare), (["run_weighting unknown"], []))
        # The second record of the example states its weighting, so it differs in three ways only.
        reasons, _ = comparable(rec, ch22_record_check.OTHER)
        self.assertEqual(reasons, ["start_event", "convention", "clock rate relation"])


class Reporting(unittest.TestCase):
    BODY = re.compile(r"^\((-?[\d.]+) ± ([\d.]+)\) (\w+), k = ")

    def test_the_example_of_the_review(self):
        self.assertTrue(Result(1234.0, 120.0, "ms").report().startswith("(1230 ± 240) ms, k = 1.96"))
        self.assertTrue(Result(1.234, 0.120, "s").report().startswith("(1.23 ± 0.24) s, k = 1.96"))
        self.assertTrue(Result(123456.0, 12000.0, "ms").report().startswith("(123000 ± 24000) ms"))
        # Below the tens nothing changed.
        self.assertTrue(Result(182.1, 3.061, "ms", 15.7).report().startswith("(182.1 ± 6.5) ms, k = 2.12"))

    def test_a_change_of_unit_does_not_change_the_digits(self):
        rng = random.Random(22)
        for _ in range(3000):
            u = 10 ** rng.uniform(-1, 4)
            v = u * rng.uniform(2, 500)
            a = self.BODY.match(Result(v, u, "ms").report()).groups()
            b = self.BODY.match(Result(v / 1000, u / 1000, "s").report()).groups()
            self.assertAlmostEqual(float(a[0]), 1000 * float(b[0]), delta=1e-9 * v, msg=(v, u))
            self.assertAlmostEqual(float(a[1]), 1000 * float(b[1]), delta=1e-9 * v, msg=(v, u))
            # Two significant digits of U, and the value at the same place.
            self.assertEqual(float(a[1]), sig(float(a[1]), 2), (v, u))

    def test_two_digits_are_counted_after_rounding(self):
        self.assertEqual([decimals(x) for x in (0.0996, 0.0994, 9.96, 99.6, 0.996)], [2, 3, 0, -1, 1])
        self.assertEqual(fmt(7.5013, 0.00996, "s"), "7.501(10) s")
        self.assertEqual(fmt(12.311, 0.00968, "ms"), "12.3110(97) ms")
        rng = random.Random(16)
        for _ in range(20000):
            u = 10 ** rng.uniform(-9, 9)
            d = decimals(u)
            shown = round(u, d)
            units = round(shown * 10 ** d) if d >= 0 else round(shown / 10 ** -d)
            self.assertTrue(10 <= units <= 99, (u, d, shown))
        text = (ROOT / "expected" / "ch16_trace_measurements.txt").read_text(encoding="utf-8")
        self.assertIn("fitted slope: 7.501(10) s", text)


class FalseAlarms(unittest.TestCase):
    N, RISK = 200, 0.05

    def test_ten_is_an_expectation(self):
        self.assertAlmostEqual(self.N * self.RISK, 10.0, places=12)
        # Each flagged test repeated once, all tests independent.
        confirmed = 1 - (1 - self.RISK ** 2) ** self.N
        self.assertAlmostEqual(confirmed, 0.394, places=3)

    def test_a_shared_baseline_keeps_the_mean_and_changes_the_spread(self):
        rng = random.Random(14)
        z = st.NormalDist().inv_cdf(1 - self.RISK)
        alone, shared = [], []
        for _ in range(3000):
            base = rng.gauss(0, 1)
            a = b = 0
            for _ in range(self.N):
                x = rng.gauss(0, 1)
                a += x > z
                b += (x - base) / math.sqrt(2) > z
            alone.append(a)
            shared.append(b)
        for counts in (alone, shared):
            self.assertAlmostEqual(st.fmean(counts), 10.0, delta=0.6)
        self.assertAlmostEqual(st.pvariance(alone), self.N * self.RISK * (1 - self.RISK), delta=1.0)
        self.assertGreater(st.pvariance(shared), 20 * st.pvariance(alone))
        # Nights without any alarm are next to impossible when the tests are independent.
        self.assertEqual(sum(c == 0 for c in alone), 0)
        self.assertGreater(sum(c == 0 for c in shared), 300)


if __name__ == "__main__":
    unittest.main()
