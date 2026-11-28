"""Regression tests added after the review of Version 11.

The review found that the checker of Chapter 22 could give two verdicts for
one duration written in seconds and in milliseconds, when the guarded value
met the limit exactly: the comparison was made on binary sums of decimal
numbers. It found that an integer beyond 2**53 lost its last unit, and that
the replay raised an exception on a whole number written with an exponent.
It asked what the speedup simulation of Chapter 5 estimates, and for
confidence bounds on the two extreme counts of Chapter 21.

Each test computes what it expects independently of the code under test:
exact fractions of the decimal numbers as written, closed forms of the t
quantile, a second method on the other side of a threshold, moments from
their definitions.
"""
import json
import math
import random
import re
import unittest
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from statistics import NormalDist

import run_all
from examples import ch05_propagation, ch22_record_check
from swmetro import record as checker
from swmetro.dist import t_isf, t_sf, z_ppf
from swmetro.record import ALL, decide, guard

ROOT = Path(__file__).resolve().parent.parent
ROUNDING = ["rounding decides: wants working values"]
FACTOR = ["at the limit: the coverage factor decides"]
# Decimal places by which the numbers of a record in ms move when it is
# written in another unit.
UNITS = {"ms": 0, "s": -3, "us": 3, "ns": 6, "ps": 9, "ks": -6}
SHIFTS = (-9, -6, -3, 3, 6)


def load(name):
    return json.loads((ROOT / "data" / name).read_text(encoding="utf-8"))


def expected(name):
    return (ROOT / "expected" / f"{name}.txt").read_text(encoding="utf-8")


def number(text, shift=0):
    """The decimal number `text`, moved by `shift` decimal places, as the int
    or float that a record in the other unit would hold."""
    d = Decimal(text).scaleb(shift)
    if d == d.to_integral_value() and "." not in text and shift >= 0:
        return int(d)
    return float(str(d))


def record(value, us, limit, dof=15, risk="0.05", digits=ALL, step=None, shift=0, stated=None):
    """A complete record from decimal texts; `shift` writes it in another unit.
    The combined uncertainty is the one that follows, unless `stated`."""
    rec = load("measurement_record.json")
    rec["components"] = [{"name": f"c{i}", "type": "B", "kind": "stipulated", "u": number(u, shift)}
                         for i, u in enumerate(us)]
    total = sum(Fraction(u) ** 2 for u in us)
    if stated is not None:
        stated = Decimal(stated)
    else:
        stated = Decimal(repr(math.sqrt(total)))
        if digits != ALL and stated:
            stated = Decimal(f"{stated:.{digits - 1}e}")
    rec["result"].update(value=number(value, shift), u=float(str(stated.scaleb(shift))), dof=dof,
                         significant_digits=digits)
    rec["result"].pop("value_step", None)
    if step is not None:
        rec["result"]["value_step"] = number(step, shift)
    rec["decision"].update(limit=number(limit, shift), risk=float(risk))
    return rec


class UnitsAtTheLimit(unittest.TestCase):
    def test_the_table_of_the_review(self):
        # Limit 150 ms, no uncertainty, an estimate rounded to the nearest 100 ms.
        rows = (("100", ("accept", [])),            # stands for 50 to 150: all within
                ("200", ("unresolved", ROUNDING)),  # 150 to 250: 150 meets the limit
                ("300", ("reject", [])))            # 250 to 350: all above
        for value, want in rows:
            for unit in ("ms", "s", "us", "ns"):
                rec = record(value, ["0"], "150", digits=2, step="100", shift=UNITS[unit])
                self.assertEqual(decide(rec), want, (value, unit))
        # The record in seconds holds the numbers of the review.
        rec = record("100", ["0"], "150", digits=2, step="100", shift=UNITS["s"])
        self.assertEqual((rec["result"]["value"], rec["result"]["value_step"], rec["decision"]["limit"]),
                         (0.1, 0.1, 0.15))
        # Binary sums of these numbers miss the limit, which is why they are not used.
        self.assertEqual(0.1 + 0.05, 0.15000000000000002)
        self.assertEqual(0.2 - 0.05, 0.15000000000000002)
        self.assertNotEqual(0.1 + 0.05, 0.15)

    def test_the_working_record_with_factor_one(self):
        # One degree of freedom and a risk of 0.25: the t quantile is tan(pi / 4) = 1.
        self.assertAlmostEqual(math.tan(math.pi / 4), 1.0, places=15)
        # The value and its two bounds (since Version 14): all three are 1.
        self.assertEqual(checker._factor(Fraction(1, 4), 1), (1, 1, 1))
        for unit in ("ms", "s", "us", "ns", "ps", "ks"):
            # 100 + 1 x 50 = 150: accepted at equality.
            rec = record("100", ["50"], "150", dof=1, risk="0.25", shift=UNITS[unit])
            self.assertEqual(decide(rec), ("accept", []), unit)
            # 200 - 1 x 50 = 150 is not above the limit: no rejection.
            rec = record("200", ["50"], "150", dof=1, risk="0.25", shift=UNITS[unit])
            self.assertEqual(decide(rec), ("undecided", []), unit)
        rec = record("100", ["50"], "150", dof=1, risk="0.25", shift=UNITS["s"])
        self.assertEqual((rec["result"]["value"], rec["components"][0]["u"], rec["decision"]["limit"]),
                         (0.1, 0.05, 0.15))
        # Two components that combine to 50 exactly: 30 and 40.
        for unit in ("ms", "s"):
            rec = record("100", ["30", "40"], "150", dof=1, risk="0.25", shift=UNITS[unit])
            self.assertEqual(decide(rec), ("accept", []), unit)

    def test_one_unit_in_the_last_place_decides(self):
        cases = (
            ("100.000000000001", ["50"], "150", "undecided"),
            ("99.999999999999", ["50"], "150", "accept"),
            ("100", ["50.000000000001"], "150", "undecided"),
            ("100", ["50"], "149.999999999999", "undecided"),
            ("200.000000000001", ["50"], "150", "reject"),
            ("200", ["49.999999999999"], "150", "reject"),
            ("200", ["50"], "150.000000000001", "undecided"),
        )
        for value, us, limit, want in cases:
            for shift in (0, -3, 3):
                rec = record(value, us, limit, dof=1, risk="0.25", shift=shift)
                self.assertEqual(decide(rec), (want, []), (value, us, limit, shift))

    def test_whole_numbers_beyond_2_53(self):
        big = 2 ** 53
        self.assertEqual(float(big + 1), float(big))   # the float cannot tell them apart
        for value, limit, want in ((big + 1, big, "reject"), (big, big, "accept"), (big - 1, big, "accept"),
                                   (big, big + 1, "accept"), (big + 2, big + 1, "reject"),
                                   (10 ** 30 + 1, 10 ** 30, "reject"), (10 ** 30, 10 ** 30, "accept")):
            rec = record("0", ["0"], "0")
            rec["result"]["value"], rec["decision"]["limit"] = value, limit
            self.assertEqual(decide(rec), (want, []), (value, limit))
            # The record survives its file: JSON keeps the integer.
            self.assertEqual(decide(json.loads(json.dumps(rec))), (want, []), (value, limit))

    def test_any_unit_gives_one_verdict(self):
        rng = random.Random(1911)
        seen = set()
        for trial in range(1500):
            us = [str(Decimal(rng.randrange(0, 2000)).scaleb(-rng.randrange(0, 3)))
                  for _ in range(rng.randrange(1, 4))]
            value = Decimal(rng.randrange(0, 400000)).scaleb(-rng.randrange(0, 4))
            exact_k = rng.random() < 0.5
            dof, risk = (1, "0.25") if exact_k else (rng.choice((1, 2, 4.79, 15, 1e7)), rng.choice(("0.05", "0.25", "0.001")))
            total = sum(Fraction(u) ** 2 for u in us)
            root = Fraction(math.isqrt(total.numerator * total.denominator), total.denominator)
            kind = rng.randrange(4)
            if kind == 0 and root * root == total:   # at the limit exactly
                limit = value + Decimal(root.numerator) / Decimal(root.denominator) * rng.choice((1, -1))
            elif kind == 1:
                limit = value + Decimal(rng.randrange(-3000, 3000)).scaleb(-rng.randrange(0, 3))
            else:
                limit = value + Decimal(rng.randrange(-30, 30))
            digits, step = ALL, None
            if rng.random() < 0.4:
                digits, step = rng.choice((2, 3)), str(Decimal(rng.choice((0, 1, 5, 100))).scaleb(-rng.randrange(0, 3)))
                us = [str(Decimal(f"{Decimal(u):.{digits - 1}e}")) if Decimal(u) else u for u in us]
                dof = 15
            stated = None
            if rng.random() < 0.05:     # a combined uncertainty that does not follow
                stated = str(Decimal(rng.randrange(1, 5000)))
            base = decide(record(str(value), us, str(limit), dof, risk, digits, step, 0, stated))
            seen.add(base[0])
            for shift in SHIFTS:
                moved = decide(record(str(value), us, str(limit), dof, risk, digits, step, shift, stated))
                self.assertEqual(moved, base, (trial, value, us, limit, dof, risk, digits, step, shift))
        self.assertEqual(seen, {"accept", "reject", "undecided", "unresolved", "invalid"})

    def test_against_exact_arithmetic(self):
        rng = random.Random(1912)
        counts = {}
        for trial in range(3000):
            value = Fraction(rng.randrange(0, 4000), 10)
            delta = Fraction(rng.randrange(-40, 41), 10)
            limit = value + delta
            shift = rng.choice((0,) + SHIFTS)
            if trial % 2:
                # No uncertainty, estimate rounded to an increment: only the ends of its interval matter.
                step = Fraction(rng.choice((0, 1, 2, 10, 20, 50)), 10)
                if step:
                    # Since Version 55 the estimate is one that rounding to the increment can have written;
                    # any other is refused as invalid. The draws are those of the earlier versions.
                    value = step * (value // step)
                    limit = value + delta
                lo, hi = value - step / 2, value + step / 2
                want = "accept" if hi <= limit else "reject" if lo > limit else "unresolved"
                rec = record(str(float(value)), ["0"], str(float(limit)), digits=2, step=str(float(step)), shift=shift)
            else:
                # Factor 1: the squares of exact fractions decide, with no square root taken.
                us = [Fraction(rng.randrange(0, 40), 10) for _ in range(rng.randrange(1, 3))]
                total, room = sum(u * u for u in us), limit - value
                if room >= 0 and room * room >= total:
                    want = "accept"
                elif room < 0 and room * room > total:
                    want = "reject"
                else:
                    want = "undecided"
                rec = record(str(float(value)), [str(float(u)) for u in us], str(float(limit)), dof=1, risk="0.25",
                             shift=shift)
            self.assertEqual(decide(rec)[0], want, (trial, value, limit, shift))
            counts[want] = counts.get(want, 0) + 1
        self.assertTrue(all(counts[k] > 100 for k in ("accept", "reject", "undecided", "unresolved")), counts)

    def test_display_bounds_still_describe_the_record(self):
        rec = ch22_record_check.altered(load("measurement_record.json"), {("result", "value"): 194.6})
        (a_lo, a_hi), (r_lo, r_hi) = (tuple(map(float, pair)) for pair in guard(rec))   # decimals since Version 13
        r = rec["result"]
        k = t_isf(rec["decision"]["risk"], r["dof"])
        u = math.sqrt(sum(c["u"] ** 2 for c in rec["components"]))
        # Until Version 71 the two ends of a working record were one number, with the factor as computed.
        # Since Version 72 they are its bounds for the allowance of one part in a million for that factor,
        # the numbers that decide() compares (tests/test_review71.py).
        self.assertAlmostEqual(a_lo, r["value"] + (1 - 1e-6) * k * u, places=9)
        self.assertAlmostEqual(a_hi, r["value"] + (1 + 1e-6) * k * u, places=9)
        self.assertAlmostEqual(r_lo, r["value"] - (1 + 1e-6) * k * u, places=9)
        self.assertAlmostEqual(r_hi, r["value"] - (1 - 1e-6) * k * u, places=9)
        # The values on Figure 22.1, as the review recomputed them, with the factor as computed. Since Version
        # 72 they lie inside the bounds, by the allowance at most: 5.4e-6 of the bands here, and half a unit of
        # the review's last digit.
        self.assertAlmostEqual((a_lo + a_hi) / 2, 199.94976, places=5)
        lo, hi = map(float, guard(ch22_record_check.altered(ch22_record_check.displayed(load("measurement_record.json")),
                                                            {("result", "value"): 194.6}))[0])
        self.assertTrue(0 < 199.86507 - lo < 1.1e-5 and 0 < hi - 200.11094 < 1.1e-5, (lo, hi))


class CoverageFactor(unittest.TestCase):
    """The checker allows one part in a million for the computed factor,
    more towards a risk of one half. The factor must be far inside that."""

    def scaled(self, k, exact, tail):
        return abs(k / exact - 1) * (1 - 2 * tail)

    def test_closed_forms(self):
        inside = float(checker.DELTA) / 100
        for tail in (1e-100, 1e-12, 1e-4, 0.01, 0.05, 0.2, 0.25, 0.3, 0.45, 0.4999, 0.4999999):
            one = 1 / math.tan(math.pi * tail)
            two = (1 - 2 * tail) / math.sqrt(2 * tail * (1 - tail))
            self.assertLess(self.scaled(t_isf(tail, 1), one, tail), inside, tail)
            self.assertLess(self.scaled(t_isf(tail, 2), two, tail), inside, tail)
        for tail in (1e-40, 1e-9, 1e-3, 0.05, 0.25, 0.3):
            a = 4 * tail * (1 - tail)
            four = 2 * math.sqrt(math.cos(math.acos(math.sqrt(a)) / 3) / math.sqrt(a) - 1)
            self.assertLess(self.scaled(t_isf(tail, 4), four, tail), inside, tail)

    def test_next_to_the_centre(self):
        # Near t = 0 the tail is 1/2 - f(0) t (1 - (v + 1) t^2 / (6 v)), with f(0) the density at 0,
        # which for many degrees of freedom v is (1 - 1/(4 v) + 1/(32 v^2)) / sqrt(2 pi).
        for dof in (2e4, 990000.0, 1e6):
            f0 = (1 - 1 / (4 * dof) + 1 / (32 * dof ** 2)) / math.sqrt(2 * math.pi)
            for tail in (0.4999, 0.49999):
                first = (0.5 - tail) / f0
                series = first * (1 + (dof + 1) / (6 * dof) * first ** 2)
                self.assertAlmostEqual(t_isf(tail, dof) / series, 1.0, places=8, msg=(dof, tail))

    def test_many_degrees_of_freedom(self):
        # Until Version 65 two methods met at 10**6: bisection on the incomplete beta function below, and an
        # expansion about the normal quantile above, and the quantile moved by parts in 10**12 between them.
        # From Version 66 the bisection serves every number up to 1e25. The checks stand as they were.
        for tail in (0.4, 0.25, 0.05, 0.001, 1e-9, 1e-30):
            below, above = t_isf(tail, 1e6), t_isf(tail, 1e6 + 1)
            self.assertAlmostEqual(above / below, 1.0, places=8, msg=tail)
            self.assertAlmostEqual(t_sf(below, 1e6 + 1) / t_sf(below, 1e6), 1.0, places=7, msg=tail)
            for dof in (1.5e6, 1e8, 1e12):
                self.assertAlmostEqual(t_sf(t_isf(tail, dof), dof) / tail, 1.0, places=9, msg=(tail, dof))
            self.assertGreater(t_isf(tail, 1e7), -z_ppf(tail))
            self.assertEqual(t_isf(tail, math.inf), -z_ppf(tail))
        # The first term of the expansion: t - z is close to (z^3 + z) / (4 v).
        z = -z_ppf(0.05)
        self.assertAlmostEqual((t_isf(0.05, 1e7) - z) / ((z ** 3 + z) / 4e7), 1.0, places=5)

    def test_the_allowance_withholds_and_never_admits(self):
        dof, risk, u = 10, "0.05", "2"
        k = Fraction(t_isf(float(risk), dof))
        band = k * 2
        for rel, want in ((Fraction(1, 10 ** 5), ("accept", [])), (Fraction(-1, 10 ** 5), ("undecided", [])),
                          (Fraction(-1, 10 ** 8), ("unresolved", FACTOR)), (Fraction(1, 10 ** 8), ("unresolved", FACTOR)),
                          (Fraction(0), ("unresolved", FACTOR))):
            # An estimate that leaves the band, times 1 + rel, below a limit of 200.
            value = Fraction(200) - band * (1 + rel)
            text = f"{Decimal(value.numerator) / Decimal(value.denominator):.15f}"
            rec = record(text, [u], "200", dof=dof, risk=risk)
            self.assertEqual(decide(rec), want, float(rel))
        for rel, want in ((Fraction(1, 10 ** 5), ("reject", [])), (Fraction(-1, 10 ** 5), ("undecided", [])),
                          (Fraction(1, 10 ** 8), ("unresolved", FACTOR))):
            value = Fraction(200) + band * (1 + rel)
            text = f"{Decimal(value.numerator) / Decimal(value.denominator):.15f}"
            self.assertEqual(decide(record(text, [u], "200", dof=dof, risk=risk)), want, float(rel))
        # With no uncertainty there is no band and no factor to allow for.
        self.assertEqual(decide(record("200", ["0"], "200", dof=dof, risk=risk)), ("accept", []))
        self.assertEqual(decide(record("200.000000000001", ["0"], "200", dof=dof, risk=risk)), ("reject", []))

    def test_what_is_not_computed(self):
        # A factor beyond the range is unbounded: nothing is accepted or rejected, unless u is zero.
        self.assertEqual(decide(record("50", ["1"], "100", dof=1e-9)), ("undecided", []))
        self.assertEqual(decide(record("150", ["1"], "100", dof=1e-9)), ("undecided", []))
        self.assertEqual(decide(record("50", ["0"], "100", dof=1e-9)), ("accept", []))
        # Below a risk of 1e-300 a float is too coarse a tail probability.
        for risk in ("1e-310", "5e-324"):
            self.assertEqual(decide(record("50", ["1"], "100", risk=risk)), ("unresolved", ["risk not supported"]))
        self.assertEqual(decide(record("50", ["1"], "100", risk="1e-300", dof=1e7)), ("accept", []))
        # Towards a risk of one half the factor goes to zero. Until Version 13 what was known of
        # it went to zero as well, and this record was unresolved. Since Version 14 the factor is
        # computed from the distance of the risk from one half: here 1e-10 / f(0), with f(0) the
        # density of Student's t for 15 degrees of freedom at zero, about 2.5e-10.
        f0 = math.gamma(8.0) / (math.sqrt(15 * math.pi) * math.gamma(7.5))
        self.assertTrue(99.9999999 + 1e-10 / f0 < 100 < 99.9999999 + 2 * 1e-10 / f0 + 1e-7)
        rec = record("99.9999999", ["1"], "100", risk="0.4999999999")
        self.assertEqual(decide(rec), ("accept", []))
        self.assertEqual(decide(record("99", ["1"], "100", risk="0.4999999999")), ("accept", []))


class ReplayNotation(unittest.TestCase):
    def test_the_token_of_the_review(self):
        text = expected("ch10_coordinated_omission")
        self.assertEqual(len(re.findall(r"\b5800\b", text)), 2)   # 1 until Version 36; the count is printed twice now
        changed = re.sub(r"\b5800\b", "5800e0", text)
        diffs = run_all.compare("exact", text, changed)   # raised ValueError before
        self.assertEqual(len(diffs), 2)
        for diff in diffs:
            self.assertIn("another notation", diff)

    def test_equal_values_written_differently(self):
        for e, a in (("5800", "5800e0"), ("5800", "+5800"), ("5800", "05800"), ("1e3", "1000"), ("1.5e-3", "1.5E-3"),
                     ("1.5e-3", "1.5e-03"), ("+0.0183", "0.0183"), ("0.15", "00.15")):
            self.assertEqual(float(e), float(a))
            for x, y in ((e, a), (a, e)):
                diffs = run_all.compare("exact", f"n {x}\n", f"n {y}\n")
                self.assertEqual(len(diffs), 1, (x, y))
                self.assertIn("another notation", diffs[0], (x, y))
        for e, a in (("5800", "5800.0"), ("1.50", "1.5")):
            diffs = run_all.compare("exact", f"n {e}\n", f"n {a}\n")
            self.assertIn("another place", diffs[0], (e, a))

    def test_what_still_passes_and_what_still_fails(self):
        for e, a in (("5800", "5800"), ("1.2862", "1.2863"), ("-0.00", "0.00"), ("+0.0183", "+0.0184"),
                     ("1.5e-3", "1.6e-3"), ("0", "-0")):
            self.assertEqual(run_all.compare("exact", f"n {e}\n", f"n {a}\n"), [], (e, a))
        for e, a in (("5800", "5799"), ("5800", "-5800"), ("1.2862", "1.2864"), ("+0.0183", "-0.0183"),
                     ("1.5e-3", "1.7e-3")):
            diffs = run_all.compare("exact", f"n {e}\n", f"n {a}\n")
            self.assertEqual(len(diffs), 1, (e, a))
            self.assertNotIn("notation", diffs[0], (e, a))


class Speedup(unittest.TestCase):
    M1, S1, M2, S2 = 842.0, 12.0, 655.0, 15.0

    def test_the_denominator_is_far_from_zero_and_still_reaches_it(self):
        z = self.M2 / self.S2
        self.assertAlmostEqual(z, 43.7, places=1)
        # A normal denominator has a positive density at zero, however small: about 1e-415 of its
        # peak here, which is below what a float can hold. No seeded run will ever visit it.
        ratio = (Decimal(-z * z) / 2).exp()
        self.assertTrue(Decimal("1e-420") < ratio < Decimal("1e-410"))
        self.assertEqual(NormalDist(self.M2, self.S2).pdf(0.0), 0.0)

    def test_lognormal_inputs_keep_the_means_and_standard_deviations(self):
        for m, s in ((self.M1, self.S1), (self.M2, self.S2)):
            v = math.log(1 + (s / m) ** 2)
            mu = math.log(m) - v / 2
            self.assertAlmostEqual(math.exp(mu + v / 2), m, places=9)
            self.assertAlmostEqual(math.exp(mu + v / 2) * math.sqrt(math.expm1(v)), s, places=9)

    def test_the_closed_forms(self):
        # By another route: E[X/Y] = E[X] E[1/Y] and E[(X/Y)^2] = E[X^2] E[1/Y^2], and for a
        # lognormal Y with mean m and coefficient of variation c, E[1/Y] = (1 + c^2) / m and
        # E[1/Y^2] = (1 + c^2)^3 / m^2.
        c1, c2 = (self.S1 / self.M1) ** 2, (self.S2 / self.M2) ** 2
        mean = self.M1 / self.M2 * (1 + c2)
        second = (self.M1 / self.M2) ** 2 * (1 + c1) * (1 + c2) ** 3
        sd = math.sqrt(second - mean ** 2)
        got = ch05_propagation.lognormal_ratio(self.M1, self.S1, self.M2, self.S2)
        self.assertAlmostEqual(got[0], mean, places=12)
        self.assertAlmostEqual(got[1], sd, places=10)
        # The interval, checked through the distribution function of the logarithm.
        sigma = math.sqrt(math.log(1 + c1) + math.log(1 + c2))
        mu = math.log(mean) - sigma ** 2 / 2
        for q, p in ((got[2], 0.025), (got[3], 0.975)):
            self.assertAlmostEqual(0.5 * math.erfc(-(math.log(q) - mu) / (sigma * math.sqrt(2))), p, places=12)
        # The values of the review.
        self.assertEqual([round(x, 5) for x in got], [1.28617, 0.03469, 1.2195, 1.35549])

    def test_the_transcript(self):
        text = expected("ch05_propagation")
        for line in ("MC, independent normal, 200000 draws:", "  sample mean 1.2862  sample sd 0.0348",
                     "  95 % of draws  [1.219, 1.357]", "exact, independent lognormal:",
                     "  mean 1.2862  sd 0.0347", "  central 95 %  [1.219, 1.356]", "r = 0.0  S = 1.2855  u_c = 0.0347"):
            # Until Version 69 the ends of an interval were printed to the nearest digit. Since Version 70 the
            # lower end is rounded downward and the upper one upward, so that the printed interval holds the
            # computed one (tests/test_review69.py). The two read [1.220, 1.356] and [1.220, 1.355].
            self.assertIn(line + "\n", text)


class DriftCounts(unittest.TestCase):
    def test_exact_one_sided_bounds(self):
        # 0 of 1000: the upper 95 % bound p solves (1 - p)^1000 = 0.05.
        upper = 1 - 0.05 ** (1 / 1000)
        self.assertEqual(round(100 * upper, 4), 0.2991)
        self.assertGreater((1 - Fraction(2991, 10 ** 6)) ** 1000, Fraction(1, 20))
        self.assertLess((1 - Fraction(2992, 10 ** 6)) ** 1000, Fraction(1, 20))
        # 300 of 300: the lower 95 % bound p solves p^300 = 0.05.
        lower = 0.05 ** (1 / 300)
        self.assertEqual(round(100 * lower, 4), 99.0064)
        self.assertLess(Fraction(990063, 10 ** 6) ** 300, Fraction(1, 20))
        self.assertGreater(Fraction(990064, 10 ** 6) ** 300, Fraction(1, 20))
        # To the nearest hundredth, as the book printed them until Version 70. The 0.30 lies above its bound, as
        # an upper bound has to, and the 99.01 above a lower one, which it does not hold. Since Version 71 the
        # text has the lower bound downward at three places, 99.006 % (tests/test_review70.py).
        self.assertEqual((f"{100 * upper:.2f}", f"{100 * lower:.2f}"), ("0.30", "99.01"))
        self.assertTrue(100 * upper <= 0.30 and 99.006 <= 100 * lower < 99.01)
        self.assertEqual(math.floor(100_000 * lower) / 1000, 99.006)
        # The rule of three gives nearly the same upper bound.
        self.assertAlmostEqual(3 / 1000, upper, places=4)

    def test_the_counts_are_those_of_the_transcript(self):
        text = expected("ch21_psi")
        self.assertIn("count pairs out of 1000", text)
        row = [line.split() for line in text.splitlines() if line.split()[:1] == ["2000"]][0]
        self.assertEqual(row[3], "0")
        self.assertIn("above crit. 300 of 300", text)


class EarlierRecords(unittest.TestCase):
    def test_another_format_is_named_and_not_decided(self):
        rec = load("measurement_record.json")
        self.assertEqual(rec["format"], "measurement record, version 5")
        self.assertEqual(decide(rec), ("accept", []))
        for old in ("measurement record, version 4", "measurement record, version 3"):
            rec["format"] = old
            self.assertEqual(decide(rec), ("unresolved", ["format not supported"]))

    def test_a_rounded_record_needs_its_increment(self):
        # Converting an older rounded record: the increment has to be recovered, and zero says
        # that the working estimate was kept.
        rec = record("194.6", ["0"], "200", digits=2)
        self.assertEqual(decide(rec), ("unresolved", ["wants result.value_step"]))
        rec["result"]["value_step"] = 0
        self.assertEqual(decide(rec), ("accept", []))
        rec["result"]["value_step"] = 20
        # 194.6 is not a multiple of 20, and since Version 55 that is said first.
        self.assertEqual(decide(rec), ("invalid", ["result.value is not a multiple of its step"]))
        rec["result"]["value"] = 200
        self.assertEqual(decide(rec), ("unresolved", ROUNDING))


if __name__ == "__main__":
    unittest.main()
