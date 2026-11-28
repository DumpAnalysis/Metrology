"""Regression tests after the review of Version 71.

The review found that Listing 22.4 could print a rejected value below its limit, an accepted one above it, and an
end that lay on the limit away from it, and that it read a limit given as a float otherwise than the decision
does. It found that a guard band which is known exactly was rounded to seventeen digits before it was added, so
that a sum above the limit was shown on it. In Listing 16.4 it found a mean exposure that depended on the shares
of two equal durations, and gave a bound below the limit of the model; two probabilities that came out above 1,
one of which also failed where a float could not hold a coefficient; and functions that took a flag for an
exposure and refused a level of four bytes that the package accepts, as the exact interval of Listing 20.2 did.
In the package it found a bound that failed under the decimal settings of the caller.

Beyond the review, helper functions of six other listings took a flag for the number 1 and, under Python 3.12 and
3.13, refused a float of four bytes; the bounds of a combined uncertainty were roots rounded to the nearest; and four
more places of the package and of Listing 22.4 used the decimal context of the caller.

The references of these tests are their own: arithmetic in fractions and whole numbers, square roots from
math.isqrt, and logarithms and exponentials in a decimal context of 80 digits that the tests state themselves.
"""
import contextlib
import copy
import decimal
import io
import math
import random
import struct
import subprocess
import sys
import unittest
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from examples import ch05_propagation as propagation
from examples import ch08_selection as selection
from examples import ch16_silence as silence
from examples import ch16_trace_measurements as trace
from examples import ch18_calibration as calibration
from examples import ch20_judge as judge
from examples import ch22_record_check as check
from examples import ch23_case as case
from examples import ch23_mix as mix
from swmetro import dist, record, stats
from swmetro.record import combined, decide, exact, guard, invalid, missing, spread

try:
    import numpy
except ImportError:                                                        # the package needs no NumPy
    numpy = None

ROOT = Path(__file__).resolve().parent.parent
D = Decimal
F = Fraction
C = decimal.Context(prec=80, rounding=decimal.ROUND_HALF_EVEN, Emax=decimal.MAX_EMAX, Emin=decimal.MIN_EMIN,
                    traps=[decimal.InvalidOperation, decimal.DivisionByZero, decimal.Overflow])
SIGNALS = (decimal.Inexact, decimal.Rounded, decimal.Underflow, decimal.Subnormal, decimal.Overflow,
           decimal.InvalidOperation, decimal.DivisionByZero, decimal.Clamped)
FOUR_BYTES = struct.unpack("f", struct.pack("f", 0.95))[0]                 # the float that 0.95 is in four bytes
TEN = [(30 * i, 30 * i + 30) for i in range(10)]
UNIT = [(i, i + 1) for i in range(10)]


def expected(name):
    return (ROOT / "expected" / f"{name}.txt").read_text(encoding="utf-8")


def printed(function, *args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        function(*args)
    return out.getvalue()


def refusal(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except Exception as error:                                            # the kind and the words of a refusal
        return f"{type(error).__name__}: {error}"
    return None


def dec(x):
    """The number that x is, as a decimal of 80 digits; exact for the floats and fractions used here."""
    x = F(x)
    return C.divide(D(x.numerator), D(x.denominator))


def hostile():
    """A decimal context of three digits, exponents from -10 to 10, rounding away from zero, every trap set."""
    return decimal.Context(prec=3, Emax=10, Emin=-10, rounding=decimal.ROUND_UP, traps=list(SIGNALS))


def least_float(x):
    """The least float that is not below the fraction x."""
    f = float(x)
    return f if F(f) >= x else math.nextafter(f, math.inf)


def text(n, places):
    """The whole number n with `places` digits after the point."""
    whole, part = divmod(abs(n), 10 ** places)
    tail = f".{part:0{places}d}" if places else ""
    return ("-" if n < 0 else "") + f"{whole}{tail}"


def reference(x, places, limit, way):
    """x rounded upward, downward or to the nearest with a half to even, at the fewest places from `places` at
    which the rounded number stands to the limit as x does."""
    x = F(x)
    while True:
        scaled = x * 10 ** places
        low = scaled.numerator // scaled.denominator
        rest = scaled - low
        if way == "up":
            n = low + (rest > 0)
        elif way == "down":
            n = low
        else:
            n = low + (rest > F(1, 2) or (rest == F(1, 2) and low % 2 == 1))
        y = F(n, 10 ** places)
        if limit is None or ((y > limit) == (x > limit) and (y < limit) == (x < limit)):
            return text(n, places)
        places += 1


class Record(unittest.TestCase):
    def setUp(self):
        self.rec = check.load("measurement_record.json")

    def one(self, value, us, dof, risk, limit=None, digits="all", step=None):
        """The published record with the given components, the uncertainty of the result combined from them
        when there is one of them, and otherwise left for the test to state."""
        out = copy.deepcopy(self.rec)
        us = us if isinstance(us, (list, tuple)) else [us]
        parts = []
        for u in us:
            part = copy.deepcopy(out["components"][1])
            part["u"], part["dof"] = u, dof
            parts.append(part)
        out["components"] = parts
        result = out["result"]
        result["value"], result["u"], result["dof"] = value, us[0], dof
        result["significant_digits"] = digits
        if step is not None:
            result["value_step"] = step
        out["decision"]["risk"] = risk
        if limit is not None:
            out["decision"]["limit"] = limit
        return out

    def shown(self, rec):
        return printed(check.near, "a title", rec, rec["result"]["value"]).splitlines()[-1]

    def ends(self, rec):
        line = self.shown(rec)
        self.assertTrue(line.startswith("  estimate + band: "), line)
        return [D(t) for t in line.split(": ")[1].replace(" or more", "").split(" to ")]


class APrintedNumberAndItsLimit(Record):
    """Section "The printed value can cross the decision limit"."""

    def test_the_table_of_the_review(self):
        rows = (("200.004", "200.003", "reject", "200.004"),               # Version 71 printed 200.00
                ("200.006", "200.007", "accept", "200.006"),               # 200.01
                ("200.004", "200.004", "accept", "200.004"))               # 200.00
        for value, limit, status, display in rows:
            rec = self.one(D(value), D(0), D(1), D("0.25"), D(limit))
            self.assertEqual((missing(rec), invalid(rec)), ([], []))
            self.assertEqual(decide(rec), (status, []), value)
            self.assertEqual(self.shown(rec), f"  estimate + band: {display}", value)

    def test_an_end_on_the_limit_stays_on_it(self):
        at = D("200.003")
        self.assertEqual(stats.upward(at, 2, at), "200.003")               # Version 71: 200.01
        self.assertEqual(stats.downward(at, 2, at), "200.003")             # Version 71: 200.00
        self.assertEqual(stats.enclosed(D("199.9"), at, 2, at), ("199.90", "200.003"))
        self.assertEqual(stats.enclosed(at, D("200.1"), 2, at), ("200.003", "200.10"))
        # Without a limit, and away from one, nothing changes.
        self.assertEqual((stats.upward(at, 2), stats.downward(at, 2)), ("200.01", "200.00"))
        self.assertEqual(stats.upward(D("94.9934"), 2, 95), "94.994")
        self.assertEqual(stats.upward(95, 2, 95), "95.00")
        # A rounded record whose upper end is the limit: an estimate of 1886 increments of 0.106, which
        # stands for half an increment either way. It is accepted, and printed up to the limit.
        self.assertEqual(1886 * F("0.106"), F("199.916"))
        rec = self.one(D("199.916"), D(0), D(1), D("0.25"), D("199.969"), 2, D("0.106"))
        self.assertEqual((missing(rec), invalid(rec)), ([], []))
        lo, hi = guard(rec)[0]
        self.assertEqual((F(lo), F(hi)), (F("199.916") - F("0.053"), F("199.969")))
        self.assertEqual(decide(rec), ("accept", []))
        self.assertEqual(self.shown(rec), "  estimate + band: 199.86 to 199.969")    # Version 71: to 199.97

    def test_the_limit_is_read_as_the_decision_reads_it(self):
        # The float 200.1 is a binary number below 200.1. The decision reads the decimal that Python prints.
        self.assertLess(F(200.1), F("200.1"))
        self.assertEqual(exact(200.1), F("200.1"))
        rec = self.one(D("200.1001"), D(0), D(1), D("0.25"), 200.1)
        self.assertEqual(decide(rec), ("reject", []))
        self.assertEqual(self.shown(rec), "  estimate + band: 200.1001")   # Version 71: 200.10
        rec = self.one(D("200.1"), D(0), D(1), D("0.25"), 200.1)
        self.assertEqual(decide(rec), ("accept", []))
        self.assertEqual(self.shown(rec), "  estimate + band: 200.10")
        # The printing functions take a float as the binary number that it is, like every function of the
        # package: against that number 200.10 is above already.
        self.assertEqual(stats.nearest(D("200.1001"), 2, 200.1), "200.10")
        self.assertEqual(stats.nearest(D("200.1001"), 2, exact(200.1)), "200.1001")

    def test_nearest_rounds_a_half_to_even(self):
        cases = ((2.5, 0, "2"), (3.5, 0, "4"), (D("0.125"), 2, "0.12"), (D("0.135"), 2, "0.14"),
                 (D("-0.125"), 2, "-0.12"), (F(1, 3), 3, "0.333"), (F(2, 3), 3, "0.667"), (-7, 1, "-7.0"),
                 (D("199.9497"), 2, "199.95"), (0.125, 2, "0.12"), (0.375, 2, "0.38"))
        for x, places, want in cases:
            self.assertEqual(stats.nearest(x, places), want, x)
            self.assertEqual(reference(x, places, None, "near"), want, x)

    def test_the_three_functions_against_a_reference(self):
        rng = random.Random(71)
        ways = (("up", stats.upward), ("down", stats.downward), ("near", stats.nearest))
        seen = {(-1, -1): 0, (0, 0): 0, (1, 1): 0}
        more = 0
        for _ in range(3000):
            limit = F(rng.randrange(-3000000, 3000000), 10 ** rng.randrange(0, 5))
            step = F(rng.choice((0, 0, 1, -1, 3, -7, 49, -51)), 10 ** rng.randrange(0, 9))
            x = limit + step
            places = rng.randrange(0, 5)
            for way, function in ways:
                got = function(x, places, limit)
                self.assertEqual(got, reference(x, places, limit, way), (x, places, limit, way))
                y = F(got)
                side = (y > limit) - (y < limit), (x > limit) - (x < limit)
                self.assertEqual(side[0], side[1], (x, places, limit, way))
                seen[side] += 1
                given = len(got.partition(".")[2])
                self.assertGreaterEqual(given, places)
                more += given > places
                error = y - x
                bound = F(1, 10 ** given)
                if way == "up":
                    self.assertTrue(0 <= error < bound)
                elif way == "down":
                    self.assertTrue(-bound < error <= 0)
                else:
                    self.assertLessEqual(abs(error), bound / 2)
        self.assertTrue(all(n > 1000 for n in seen.values()) and more > 1000, (seen, more))

    def test_a_number_on_a_limit_that_is_no_decimal(self):
        # A third on a limit of a third has no decimal that lies on the limit: it keeps its places.
        third = F(1, 3)
        self.assertEqual((stats.upward(third, 2, third), stats.downward(third, 2, third)), ("0.34", "0.33"))
        self.assertEqual(stats.nearest(third, 2, third), "0.33")
        # Next to it, the side is kept.
        self.assertEqual(stats.downward(F("0.3334"), 2, third), "0.3334")
        self.assertEqual(stats.upward(F("0.3333"), 2, third), "0.3333")

    def test_the_explanation_of_the_book(self):
        # Chapter 22: "A value of 200.004 ms against a limit of 200.003 ms prints as 200.004 ms".
        self.assertEqual(stats.nearest(D("200.004"), 2, D("200.003")), "200.004")
        self.assertEqual(f"{D('200.004'):.2f}", "200.00")                  # two places, below the limit
        # README: 200.003 on a limit of 200.003.
        self.assertEqual(stats.upward(D("200.003"), 2, D("200.003")), "200.003")


class ABandThatIsKnown(Record):
    """Section "An exactly known guard band loses its deciding digits"."""

    def test_the_record_of_the_review(self):
        rec = self.one(D(199), D("1.00000000000000004"), D(1), D("0.25"), D(200))
        self.assertEqual((missing(rec), invalid(rec)), ([], []))
        self.assertEqual(decide(rec), ("undecided", []))
        want = D("200.00000000000000004")
        self.assertEqual(guard(rec)[0], (want, want))                      # Version 71: 200 and 200
        self.assertEqual(guard(rec)[1], (D("197.99999999999999996"),) * 2)
        self.assertEqual(self.shown(rec), "  estimate + band: 200.00000000000000004")   # Version 71: 200.00

    def test_on_the_limit_and_below_it(self):
        rec = self.one(D(199), D(1), D(1), D("0.25"), D(200))
        self.assertEqual(decide(rec), ("accept", []))
        self.assertEqual(guard(rec)[0], (D(200), D(200)))
        self.assertEqual(self.shown(rec), "  estimate + band: 200.00")
        rec = self.one(D(199), D("0.99999999999999996"), D(1), D("0.25"), D(200))
        self.assertEqual(decide(rec), ("accept", []))
        self.assertEqual(self.shown(rec), "  estimate + band: 199.99999999999999996")
        # A band of more than forty digits is kept as well.
        u = D("1." + "0" * 59 + "7")
        rec = self.one(D(199), u, D(1), D("0.25"), D(200))
        self.assertEqual(guard(rec)[0], (C.add(D(199), u),) * 2)
        self.assertEqual(F(guard(rec)[0][0]), 199 + F(u))
        self.assertEqual(decide(rec), ("undecided", []))

    def two(self, limit):
        """An estimate of 199 with two components of 1 and a factor of exactly 1: a band of the root of 2."""
        rec = self.one(D(199), [D(1), D(1)], D(1), D("0.25"), limit)
        rec["result"]["u"] = D("1.4142135623730950488016887242096980785697")
        self.assertEqual((missing(rec), invalid(rec)), ([], []))
        return rec

    def test_a_root_that_is_no_fraction_is_bounded(self):
        lo, hi = (F(x) - 199 for x in guard(self.two(D(205)))[0])
        self.assertTrue(lo * lo < 2 < hi * hi)
        self.assertEqual(hi - lo, F(1, 10 ** 39))                          # forty digits, one unit apart
        root = math.isqrt(2 * 10 ** 78)                                    # the root of 2 in units of 1e-39
        self.assertEqual((lo, hi), (F(root, 10 ** 39), F(root + 1, 10 ** 39)))
        down, up = (199 - F(x) for x in guard(self.two(D(205)))[1])
        self.assertEqual((down, up), (hi, lo))

    def test_each_bound_stays_on_the_side_of_the_band(self):
        # A limit that the sum exceeds by less than 1e-50, and one that exceeds the sum by as little: forty
        # digits of the root do not tell the two apart, and the bound takes as many as do.
        root = math.isqrt(2 * 10 ** 100)                                   # the root of 2 in units of 1e-50
        below, above = (199 * 10 ** 50 + root + i for i in (0, 1))        # in units of 1e-50
        for units, status in ((below, "undecided"), (above, "accept")):
            limit = F(units, 10 ** 50)
            self.assertEqual((limit - 199) ** 2 < 2, status == "undecided")
            rec = self.two(D(units).scaleb(-50, C))
            self.assertEqual(exact(rec["decision"]["limit"]), limit)
            self.assertEqual(decide(rec), (status, []))
            lo, hi = map(F, guard(rec)[0])
            self.assertTrue(lo < hi and (lo - 199) ** 2 < 2 < (hi - 199) ** 2)
            self.assertEqual((lo > limit, hi > limit), (status == "undecided",) * 2)
            low, top = map(F, self.ends(rec))
            self.assertTrue(low <= lo and hi <= top)
            self.assertEqual((low > limit, top > limit), (status == "undecided",) * 2)

    def test_the_numbers_of_the_book(self):
        # Chapter 22: 199 ms with a standard uncertainty of 1.00000000000000004 ms.
        self.assertEqual(C.add(D(199), D("1.00000000000000004")), D("200.00000000000000004"))
        seventeen = decimal.Context(prec=17).plus(D("1.00000000000000004"))
        self.assertEqual(seventeen, D(1))
        self.assertEqual(f"{C.add(D(199), seventeen):.2f}", "200.00")      # what seventeen digits show


class TheNumbersThatTheDecisionCompares(Record):
    """The four ends of guard() are what decide() compares, and the listing prints them on their sides."""

    def status(self, rec, limit):
        (a_lo, a_hi), (r_lo, r_hi) = (tuple(map(F, pair)) for pair in guard(rec))
        if a_hi <= limit:
            return "accept"
        if r_lo > limit:
            return "reject"
        if a_lo > limit and r_hi <= limit:
            return "undecided"
        return "unresolved"

    def test_the_display_and_the_decision_agree(self):
        rng = random.Random(72)
        counts = {}
        limit = F(200)
        for trial in range(1200):
            dof = rng.choice((D(1), D("2.5"), D("15.7"), D(30)))
            risk = rng.choice((D("0.05"), D("0.25"), D("0.01")))
            u = D(rng.randrange(1, 400)).scaleb(-2, C)
            k = dist.t_isf(float(risk), float(dof))
            band = F(k) * F(u)
            offset = rng.choice((0, 1, -1)) * F(rng.choice((0, 1, 5, 20, 1000, 300000)), 10 ** 7)
            sign = rng.choice((1, -1))
            value = limit - sign * band * (1 + offset)
            digits = rng.choice(("all", "all", 2))
            step = None
            if digits == 2:
                step = D("0.1")
                value = F(round(value * 10), 10)
                u = D(str(float(f"{float(u):.2g}")))
                dof = D(str(float(f"{float(dof):.2g}")))
            else:
                value = F(round(value * 10 ** 12), 10 ** 12)
            rec = self.one(dec(value).normalize(C), u, dof, risk, D(200), digits, step)
            self.assertEqual((missing(rec), invalid(rec)), ([], []), trial)
            status = decide(rec)[0]
            self.assertEqual(status, self.status(rec, limit), trial)
            counts[status] = counts.get(status, 0) + 1
            lo, hi = guard(rec)[0]
            shown = self.ends(rec)
            if lo == hi:                                                   # one value, to the nearest
                self.assertLessEqual(abs(F(shown[0]) - F(lo)), F(1, 200), trial)
            else:                                                          # two bounds, rounded outward
                self.assertTrue(F(shown[0]) <= F(lo) and F(hi) <= F(shown[-1]), trial)
            for end, as_printed in ((lo, shown[0]), (hi, shown[-1])):
                self.assertEqual((F(as_printed) > limit) - (F(as_printed) < limit),
                                 (F(end) > limit) - (F(end) < limit), (trial, end, as_printed))
        self.assertTrue(all(counts.get(s, 0) > 60 for s in ("accept", "reject", "undecided", "unresolved")), counts)

    def test_the_ends_enclose_the_band_of_a_computed_factor(self):
        for value, us, dof, risk in ((D("194.6"), [D("1.7"), D("2.5"), D("0.3")], D("15.7"), D("0.05")),
                                     (D(100), [D(3)], D("2.5"), D("0.01")),
                                     (D(50), [D("0.07"), D("0.11")], D(30), D("0.25"))):
            rec = self.one(value, us, dof, risk, D(200))
            u2 = sum(F(u) ** 2 for u in us)
            rec["result"]["u"] = dec(F(math.isqrt(int(u2 * 10 ** 60)), 10 ** 30)).normalize(C)
            self.assertEqual((missing(rec), invalid(rec)), ([], []))
            k = F(dist.t_isf(float(risk), float(dof)))
            low, high = k * (1 - F(1, 10 ** 6)), k * (1 + F(1, 10 ** 6))   # the allowance of Chapter 22
            (a_lo, a_hi), (r_lo, r_hi) = (tuple(map(F, pair)) for pair in guard(rec))
            at, tiny = F(value), F(1, 10 ** 35)
            for end, factor in ((a_lo - at, low), (at - r_hi, low)):
                self.assertTrue(end ** 2 <= factor ** 2 * u2 < (end + tiny) ** 2, (value, end))
            for end, factor in ((a_hi - at, high), (at - r_lo, high)):
                self.assertTrue((end - tiny) ** 2 < factor ** 2 * u2 <= end ** 2, (value, end))

    def test_the_line_of_the_transcript(self):
        rec = check.altered(self.rec, {("result", "value"): D("194.6")})
        self.assertEqual(self.shown(rec), "  estimate + band: 199.949755 to 199.949767")
        self.assertIn("  estimate + band: 199.949755 to 199.949767\n", expected("ch22_record_check"))
        lo, hi = map(F, guard(rec)[0])
        r = rec["result"]
        k = F(dist.t_isf(0.05, float(r["dof"])))
        u2 = sum(F(c["u"]) ** 2 for c in rec["components"])
        middle = (lo + hi) / 2 - F("194.6")
        self.assertLess(abs(middle ** 2 / (k * k * u2) - 1), F(1, 10 ** 30))    # the sum as it is computed
        self.assertLess(abs((hi - lo) / (2 * middle) - F(1, 10 ** 6)), F(1, 10 ** 30))
        self.assertEqual(f"{float((lo + hi) / 2):.2f}", "199.95")               # the label of Figure 22.1


class RootsRoundedOutward(Record):
    """Beyond the review: the bounds of a combined uncertainty were roots rounded to the nearest."""

    def test_a_root_just_below_a_printed_place(self):
        u = D("2." + "9" * 44)                                             # below 3 by 1e-44
        rec = self.one(D(100), u, D(10), D("0.05"), D(200))
        self.assertEqual((missing(rec), invalid(rec)), ([], []))
        both = combined(rec)
        self.assertEqual(both, (u, u))                                     # Version 71: 3 and 3, forty digits
        self.assertEqual(stats.enclosed(*both, 3), ("2.999", "3.000"))     # Version 71: 3.000 to 3.000
        self.assertEqual(spread(rec), (u, u))

    def test_two_bounds_of_a_root(self):
        rec = self.one(D(100), [D(1), D(1)], D(10), D("0.05"), D(200))
        rec["result"]["u"] = D("1.4142135623730950488016887242096980785697")
        self.assertEqual((missing(rec), invalid(rec)), ([], []))
        lo, hi = map(F, combined(rec))
        root = math.isqrt(2 * 10 ** 78)
        self.assertEqual((lo, hi), (F(root, 10 ** 39), F(root + 1, 10 ** 39)))
        self.assertTrue(lo * lo < 2 < hi * hi)
        self.assertEqual(spread(rec), combined(rec))

    def test_the_transcript_keeps_its_bounds(self):
        lines = expected("ch22_record_check")
        self.assertIn("  components allow: 2.982 to 3.122\n", lines)
        self.assertIn("  with the stated u: 3.050 to 3.122\n", lines)
        shown = check.displayed(self.rec)
        lo, hi = map(F, spread(shown))
        self.assertEqual([str(c["u"]) for c in shown["components"]], ["1.7", "2.5", "0.30", "0.29"])
        # Two significant digits: each component stands for half a unit of its last digit either way.
        low2 = F("1.65") ** 2 + F("2.45") ** 2 + F("0.295") ** 2 + F("0.285") ** 2
        top2 = F("1.75") ** 2 + F("2.55") ** 2 + F("0.305") ** 2 + F("0.295") ** 2
        tiny = F(1, 10 ** 38)
        self.assertTrue(lo ** 2 <= low2 < (lo + tiny) ** 2, lo)
        self.assertTrue((hi - tiny) ** 2 < top2 <= hi ** 2, hi)
        self.assertEqual(stats.enclosed(lo, hi, 3), ("2.982", "3.122"))


class SharesOfAMeanExposure(unittest.TestCase):
    """Section "The exposure of identical durations depends on their shares"."""

    def test_equal_durations_give_the_exposure_of_one(self):
        t = 2.99573227355399
        whole = [(0, t)]
        one = silence.reach(whole, t)
        self.assertEqual(one, 2 * F(t))
        self.assertEqual(F(0.2) + F(0.8) - 1, F(1, 2 ** 54))               # the excess that the review names
        for shares in ([(1, t)], [(0.2, t), (0.8, t)], [(0.1, t), (0.2, t), (0.7, t)],
                       [(F(1, 3), t), (F(2, 3), t)], [(D("0.5"), t), (0.5, t)]):
            self.assertEqual(silence.mean_reach(whole, shares), one, shares)   # Version 71: above it for 0.2, 0.8

    def test_the_bound_of_the_review(self):
        t = 2.99573227355399
        tail = 1 - F(0.95)
        true = C.divide(C.ln(C.divide(D(1), dec(tail))), dec(t))
        self.assertTrue(str(true).startswith("1.00000000000000004646"), true)  # the review's digits, cut off
        want = least_float(F(true))
        self.assertEqual(want, math.nextafter(1.0, 2.0))
        for shares in ([(1, t)], [(0.2, t), (0.8, t)]):
            a = silence.mean_reach([(0, t)], shares)
            bound = stats.quiet_bound(a / 2)
            self.assertEqual(bound, want, shares)                          # Version 71: 1.0 for 0.2 and 0.8
            self.assertEqual(stats.upward(bound, 5), "1.00001")
            self.assertGreaterEqual(F(bound), F(true))

    def test_a_mean_lies_between_its_reaches(self):
        rng = random.Random(73)
        whole = [(0, 30)]
        off = 0
        for _ in range(300):
            weights = [rng.random() for _ in range(rng.randrange(2, 6))]
            ps = [w / sum(weights) for w in weights]                       # floats that sum to one or nearly
            off += sum(map(F, ps)) != 1
            ds = [rng.choice((1, 5, 10, 29.5, 50, 291)) for _ in ps]
            reaches = [silence.reach(whole, d) for d in ds]
            got = silence.mean_reach(whole, list(zip(ps, ds)))
            want = sum(F(p) * a for p, a in zip(ps, reaches)) / sum(F(p) for p in ps)
            self.assertEqual(got, want)
            self.assertTrue(min(reaches) <= got <= max(reaches), (ps, ds))
        self.assertGreater(off, 100)                                       # totals that are not exactly one

    def test_shares_of_several_kinds(self):
        # reach() of a window of 30 s: 40 s for requests of 10 s, and 60 s for requests of 50 s.
        self.assertEqual([silence.reach([(0, 30)], d) for d in (10, 50)], [40, 60])
        mixed = silence.mean_reach([(0, 30)], [(D("0.5"), 10), (0.5, 50)])     # Version 71: a TypeError
        self.assertEqual(mixed, 50)
        self.assertEqual(silence.mean_reach([(0, 30)], [(F(1, 4), 10), (D("0.75"), 50)]), 55)

    def test_what_is_refused(self):
        whole = [(0, 30)]
        for shares in ([(0.5, 10), (0.4, 50)], [(0.5, 10), (0.6, 50)], [], [(1.5, 10), (-0.5, 50)]):
            self.assertEqual(refusal(silence.mean_reach, whole, shares), "ValueError: shares must sum to 1")
        self.assertEqual(refusal(silence.mean_reach, whole, [(True, 10)]), "TypeError: no number: True")
        self.assertEqual(refusal(silence.mean_reach, whole, [("1", 10)]), "TypeError: no number: '1'")
        self.assertEqual(refusal(silence.mean_reach, whole, [(1, True)]), "TypeError: no number: True")


class ProbabilitiesStayProbabilities(unittest.TestCase):
    """Section "Two probability helpers can return more than one"."""

    def gap(self, rate, total, length, top=260):
        """The chance of a gap from the count of the messages: with n of them the n + 1 spacings are those of n
        points placed evenly at random, and the largest reaches `length` with a chance that is a sum over the
        spacings chosen. The Poisson weights are fractions up to exp(-mean), which is a decimal of 80 digits."""
        rate, total, length = F(rate), F(total), F(length)
        mean = rate * total
        weight, acc = F(1), F(0)
        for n in range(top):
            if n:
                weight = weight * mean / n
            some, k = F(0), 1
            while k <= n + 1 and k * length < total:
                some += (-1) ** (k + 1) * math.comb(n + 1, k) * (1 - k * length / total) ** n
                k += 1
            if n == 0:
                some = F(1)
            acc += weight * some
        none = C.exp(dec(-mean))
        # What is left out: the weights from `top` on fall faster than a geometric series.
        rest = weight * mean / top / (1 - mean / top)
        self.assertTrue(top > mean and C.multiply(none, dec(rest)) < D("1e-60"))
        return C.multiply(none, dec(acc))

    def test_the_gap_of_the_review(self):
        true = self.gap(0.4, 20, 1)
        self.assertTrue(str(true).startswith("0.9999999999999999884102175"), true)   # the review's, cut off
        got = silence.gap_somewhere(0.4, 20, 1)
        self.assertEqual(got, float(F(true)))                              # Version 71: 1.0000000000000053
        self.assertEqual(got, 1.0)

    def test_a_rate_whose_coefficients_no_float_holds(self):
        self.assertEqual(silence.gap_somewhere(1e20, 20, 1), 0.0)          # Version 71: an OverflowError
        self.assertEqual(silence.gap_somewhere(1e300, 20, 1), 0.0)
        self.assertEqual(silence.gap_somewhere(0, 20, 1), 1.0)
        self.assertEqual(silence.gap_somewhere(1e-300, 20, 1), 1.0)

    def test_the_gap_is_the_float_nearest_to_the_sum(self):
        for rate, total, length in ((0.2, 300, 30), (0.1, 300, 30), (0.0998577, 300, 30), (1.5, 20, 1),
                                    (3, 20, 1), (0.5, 7, 3), (2, 10, 10), (0.25, 40, 2.5)):
            true = self.gap(rate, total, length)
            got = silence.gap_somewhere(rate, total, length)
            self.assertEqual(got, float(F(true)), (rate, total, length))
            self.assertTrue(0 <= got <= 1)

    def test_every_gap_is_a_probability(self):
        rng = random.Random(74)
        for _ in range(400):
            length = rng.choice((1, 0.5, 3, 30))
            total = length * rng.uniform(1, 20)
            rate = rng.choice((0, 1e-9, 1e-3, 0.01, 0.1, 0.4, 1, 3, 40, 1e6)) / length * rng.uniform(0.5, 2)
            got = silence.gap_somewhere(rate, total, length)
            self.assertTrue(0 <= got <= 1, (rate, total, length, got))

    def test_the_windows_of_the_review(self):
        # Ten unit windows in a row and messages in pairs at one instant: a window is empty unless a request
        # starts in it, and the windows are independent.
        terms = silence.selections(UNIT, 0)
        for rate in (0.0001, 0.01, 1, 3, 10, 40):
            empty = C.exp(dec(-F(rate) / 2))
            true = C.subtract(D(1), C.power(C.subtract(D(1), empty), D(10)))
            got = silence.some_empty(rate, terms)
            self.assertEqual(got, float(F(true)), rate)                    # Version 71: 1.0000000000000038
            self.assertTrue(0 <= got <= 1)
        self.assertEqual(silence.some_empty(0.0001, terms), 1.0)
        self.assertEqual(silence.some_empty(0, terms), 1.0)
        self.assertEqual(silence.some_empty(1e6, terms), 0.0)

    def test_pairs_that_reach_into_the_next_window(self):
        # No closed form here. The sum in floats, term by term, is good to nine digits at these rates.
        terms = silence.selections(TEN, 10)
        for rate in (0.05, 0.1, 0.26, 0.5):
            floats = 0.0
            for sign, a in terms:
                floats += sign * math.exp(-rate / 2 * float(a))
            got = silence.some_empty(rate, terms)
            self.assertAlmostEqual(got, floats, places=9)
            self.assertTrue(0 <= got <= 1)

    def test_the_selections_of_one_reach_are_taken_together(self):
        terms = silence.selections(TEN, 10)
        reaches = {a for _, a in terms}
        self.assertEqual((len(terms), len(reaches)), (1023, 24))
        self.assertEqual(sum(sign for sign, _ in terms), 1)                # the binomial sum of the signs
        # The enclosure takes the terms of one exponent together: it is the one of the 24 net terms, and at
        # forty digits it is narrower than the sum of 1023 enclosures could be.
        net = [(sum(sign for sign, b in terms if b == a), F(a) / 8) for a in sorted(reaches)]
        pairs = [(sign, F(a) / 8) for sign, a in terms]
        self.assertEqual(stats.sum_within(pairs), stats.sum_within(net))
        lo, hi = stats.sum_within(pairs)
        self.assertLess(hi - lo, F(sum(abs(c) for c, _ in net) + 1, 10 ** 39))
        one = stats.poisson_within(0, F(1, 3))
        self.assertEqual(stats.sum_within([(1, F(1, 3)), (2, F(1, 3)), (-3, F(1, 3))]), (0, 0))
        self.assertEqual(stats.sum_within([(F(1, 2), F(1, 3)), (0.5, F(1, 3))]), one)
        self.assertEqual(stats.sum_within([(F(1, 2), 0.5), (0.5, F(1, 2)), (0, D("0.5"))]),
                         stats.poisson_within(0, F(1, 2)))                  # one exponent in three kinds
        # An exponent that is no expected count is refused as it was.
        for bad in (math.inf, -1, math.nan):
            self.assertEqual(refusal(stats.sum_within, [(1, bad)]), "ValueError: need a mean >= 0", bad)
        self.assertEqual(refusal(stats.sum_within, [(1, True)]), "TypeError: no number: True")

    def test_chance(self):
        self.assertEqual(stats.chance([(1, 0)]), 1.0)
        self.assertEqual(stats.chance([]), 0.0)
        self.assertEqual(stats.chance([(1, 3), (-1, 3)]), 0.0)
        self.assertEqual(stats.chance([(F(1, 2), 0), (F(1, 2), 0)]), 1.0)
        self.assertEqual(stats.chance(iter([(1, 0), (-1, 10 ** 30)])), 1.0)
        half = stats.chance([(1, F(math.log(2)))])                         # exp(-x) at the float next to ln 2
        self.assertEqual(half, float(F(C.exp(dec(-F(math.log(2)))))))
        tiny = stats.chance([(1, 800)])                                    # below every float
        self.assertEqual(tiny, 0.0)
        self.assertEqual(stats.chance([(1, 700)]), float(F(C.exp(D(-700)))))


class NumbersAsTheyAre(unittest.TestCase):
    """Section "A one-window search accepts a flag and rejects a valid level"."""

    def bounds(self, level):
        return (stats.quiet_bound(10, level), silence.searched_bound(10, 1, level),
                silence.paired_upper(1, [(0, 30)], 10, level), silence.gap_bound(300, 30, level),
                silence.searched_pairs(TEN, 10, level), judge.exact_interval(2, 20, level))

    def test_a_flag_is_no_exposure(self):
        self.assertEqual(refusal(stats.quiet_bound, True), "TypeError: no number: True")
        self.assertEqual(refusal(silence.searched_bound, True, 1), "TypeError: no number: True")   # V71: a rate
        self.assertEqual(refusal(silence.searched_bound, False, 1), "TypeError: no number: False")
        self.assertEqual(silence.searched_bound(30, 1), stats.quiet_bound(30))
        self.assertEqual(refusal(silence.searched_bound, math.inf, 1), "ValueError: no exposure")
        self.assertEqual(refusal(silence.searched_bound, 0, 1), "ValueError: no exposure")
        self.assertEqual(refusal(silence.searched_bound, math.nan, 1), "ValueError: no exposure")

    def test_a_flag_is_no_level(self):
        calls = ((stats.as_level,), (silence.searched_bound, 10, 1), (silence.paired_upper, 1, [(0, 30)], 10),
                 (silence.paired_upper, 0, [(0, 30)], 10), (silence.gap_bound, 300, 30),
                 (silence.searched_pairs, TEN, 10), (judge.exact_interval, 2, 20), (stats.quiet_bound, 10))
        for function, *args in calls:
            self.assertEqual(refusal(function, *args, True), "TypeError: no number: True", function.__name__)
            for level in (0, 1, 1.5, -0.1, math.nan):
                self.assertEqual(refusal(function, *args, level), "ValueError: need 0 < level < 1",
                                 (function.__name__, level))

    def test_a_level_is_the_number_that_it_is(self):
        self.assertEqual(stats.as_level(0.95), F(0.95))
        self.assertEqual(stats.as_level(D("0.95")), F(19, 20))
        self.assertEqual(stats.as_level(F(19, 20)), F(19, 20))
        self.assertEqual(stats.as_level(FOUR_BYTES), F(FOUR_BYTES))
        self.assertTrue(F(FOUR_BYTES) < F(0.95) < F(19, 20))
        self.assertEqual(FOUR_BYTES, 0.949999988079071044921875)
        # The level in four bytes gives its own bounds, a little below those of the float 0.95.
        narrow, usual = self.bounds(FOUR_BYTES), self.bounds(0.95)
        for a, b in zip(narrow[:5], usual[:5]):
            self.assertLess(a, b)
        tail = 1 - F(FOUR_BYTES)
        true = C.divide(C.ln(C.divide(D(1), dec(tail))), D(10))
        self.assertEqual(narrow[0], least_float(F(true)))
        self.assertEqual(narrow[0], 0.29957320351354405)                   # the value of the review
        self.assertEqual(narrow[1], narrow[0])                             # one window: quiet_bound()

    @unittest.skipIf(numpy is None, "NumPy is not installed")
    def test_a_level_of_four_bytes(self):
        level = numpy.float32(0.95)
        self.assertEqual(float(level), FOUR_BYTES)
        self.assertEqual(stats.as_level(level), F(FOUR_BYTES))
        self.assertEqual(self.bounds(level), self.bounds(FOUR_BYTES))      # Version 71: five TypeErrors
        self.assertTrue(refusal(stats.as_level, numpy.bool_(True)).startswith("TypeError: no number"))

    @unittest.skipIf(numpy is None, "NumPy is not installed")
    def test_times_and_shares_of_other_widths(self):
        f32, i8 = numpy.float32, numpy.int8
        self.assertEqual(silence.reach([(i8(0), i8(30))], f32(10)), 40)
        self.assertEqual(silence.mean_reach([(0, 30)], [(f32(0.5), 10), (f32(0.5), 50)]), 50)
        self.assertEqual(silence.kept_starts([(0, 30)], 10, f32(0.25), "message"), F(55, 4))
        self.assertEqual(silence.searched_bound(f32(10), i8(1)), stats.quiet_bound(10))
        self.assertEqual(silence.gap_somewhere(f32(0.5), i8(20), i8(1)), silence.gap_somewhere(0.5, 20, 1))
        self.assertEqual(silence.gap_terms(f32(0.5), 20, 1), silence.gap_terms(0.5, 20, 1))

    def test_times_rates_and_shares(self):
        flag = "TypeError: no number: True"
        self.assertEqual(refusal(silence.merged, [(True, 2)]), flag)
        self.assertEqual(refusal(silence.merged, [(0, True)]), flag)
        self.assertEqual(refusal(silence.starts, [(0, 30)], True), flag)
        self.assertEqual(refusal(silence.reach, [(0, 30)], True), flag)
        self.assertEqual(refusal(silence.kept_starts, [(0, 30)], 10, True, "message"), flag)
        self.assertEqual(refusal(silence.gap_terms, True, 20, 1), flag)
        self.assertEqual(refusal(silence.gap_terms, 0.5, True, True), flag)
        self.assertEqual(refusal(silence.gap_somewhere, 0.5, 20, True), flag)
        self.assertEqual(refusal(silence.some_empty, True, silence.selections(UNIT, 0)), flag)
        self.assertEqual(refusal(silence.starts, [(0, 30)], math.inf), "ValueError: need delta >= 0")
        self.assertEqual(refusal(silence.starts, [(0, 30)], math.nan), "ValueError: need delta >= 0")
        self.assertEqual(refusal(silence.starts, [(0, 30)], -1), "ValueError: need delta >= 0")
        # Decimals and fractions enter as they are.
        self.assertEqual(silence.reach([(D("0.5"), F(61, 2))], D("10")), 40)
        self.assertEqual(silence.kept_starts([(0, 30)], 10, D("0.25"), "request"), 10)
        self.assertEqual(silence.merged([(0.1, D("0.3")), (F(1, 4), 2)]), [(F(0.1), F(2))])


class NumbersOfTheOtherListings(unittest.TestCase):
    """Beyond the review: helpers of Listings 5.1, 8.4, 16.1, 18.1, 23.1 and 23.2 converted an argument with
    Fraction(). They now take it as native() gives it, through fraction() of the package."""

    FLAG = "TypeError: no number: True"

    def test_fraction(self):
        self.assertEqual(stats.fraction(0.1), F(0.1))
        self.assertEqual(stats.fraction(D("0.1")), F(1, 10))
        self.assertEqual(stats.fraction(F(2, 6)), F(1, 3))
        self.assertEqual(stats.fraction(7), 7)
        self.assertEqual(refusal(stats.fraction, True), self.FLAG)
        self.assertEqual(refusal(stats.fraction, "1"), "TypeError: no number: '1'")
        self.assertEqual(refusal(stats.fraction, None), "TypeError: no number: None")

    def test_a_flag_is_no_number_in_them(self):
        calls = ((propagation.shared_rate, 0, 10 ** 9, True),
                 (selection.summary, [(1.0, True), (2.0, 1)]),
                 (selection.summary, [(True, 1), (2.0, 1)]),
                 (trace.ticks, True),
                 (trace.window, 0, True, (0, 10 ** 6)),
                 (calibration.logit_of, True),
                 (calibration.scaled, True, 2),
                 (calibration.scaled, 2.0, True),
                 (calibration.lean, [True], F(1)),
                 (calibration.lean, [F(1, 2)], True),
                 (calibration.at_one_logit, True, 0.75),
                 (case.pair, True, 1),
                 (case.pair, 1.5, True),
                 (case.exact, 1.5, True),
                 (mix.mean, [(1.0, True), (2.0, 1)]))
        for function, *args in calls:
            self.assertEqual(refusal(function, *args), self.FLAG, (function.__name__, args))   # Version 71: 1

    def test_what_is_no_number_is_refused_as_it_was(self):
        # An infinite logit, and one or a probability that is no number, keep the refusals of at_one_logit().
        beyond, none = "ValueError: T is beyond a float", "ValueError: no T above zero states this p"
        for logit, p, want in ((math.inf, 0.75, beyond), (-math.inf, 0.25, beyond), (D("1e400"), 0.75, beyond),
                               (math.nan, 0.75, none), (40, math.nan, none), (2.0, 1.5, none), (0, 0.75, none),
                               (2.0, 0.5, none)):
            self.assertEqual(refusal(calibration.at_one_logit, logit, p), want, (logit, p))
        # The conversions of the other helpers refuse an infinity and a NaN in the words of Fraction, as before.
        for function, *args in ((calibration.logit_of, math.nan), (case.pair, math.inf, 1),
                                (selection.summary, [(1.0, math.nan)]), (mix.mean, [(1.0, math.inf)])):
            self.assertIn(refusal(function, *args).split(":")[0], ("ValueError", "OverflowError"), function.__name__)

    def test_numbers_enter_as_they_did(self):
        # 100 ms at weight 1 and 400 ms at weight 1/2: a mean of 300 / (3/2), the slower value at the 95th
        # percentile, and a third of the weight above 300 ms.
        self.assertEqual(selection.summary([(100.0, 1), (400.0, F(1, 2))]), (200.0, 400.0, 100 * float(F(1, 3))))
        self.assertEqual(selection.summary([(100.0, 2), (400.0, 1)]), selection.summary([(100.0, 1), (400.0, 0.5)]))
        self.assertEqual((trace.ticks(2.007), trace.ticks(30), trace.ticks(D("0.0005")), trace.ticks(0.0005)),
                         (2007, 30000, F(1, 2), F(1, 2)))
        self.assertEqual((case.pair(1.5, 2.5), case.pair(D("200.3"), 0)), ((1.5, 2.5), (F("200.3"), 0)))
        self.assertEqual(mix.mean([(D("0.1"), F(1)), (D("0.2"), F(1))]), F(3, 20))
        self.assertEqual(mix.mean([(200.0, F(1, 3))] * 3), 200.0)
        self.assertEqual(calibration.scaled(D("3e-324"), D("5e-324")), 0.6)
        self.assertEqual(calibration.scaled(3.0, 2.0), 1.5)
        self.assertEqual(calibration.logit_of(F(1, 2)), 0)
        self.assertEqual(calibration.lean([F(1, 2), F(-1, 3)], F(1)), calibration.lean([0.5, F(-1, 3)], 1))
        self.assertEqual(propagation.shared_rate()[0], 10 ** 9)
        # An end of a window may be given as the text of a decimal, as before, and what is no time is refused
        # as before.
        self.assertEqual((trace.ticks("2.007"), trace.ticks(F(2007, 1000))), (2007, 2007))
        for bad in (math.nan, math.inf, "soon"):
            self.assertTrue(refusal(trace.ticks, bad).startswith("ValueError"), bad)

    @unittest.skipIf(numpy is None, "NumPy is not installed")
    def test_a_float_of_four_bytes_in_them(self):
        f32 = numpy.float32                                                # Version 71: a TypeError in each
        self.assertEqual(selection.summary([(1.0, f32(0.5)), (2.0, 1)]), selection.summary([(1.0, 0.5), (2.0, 1)]))
        self.assertEqual(trace.ticks(f32(0.25)), 250)
        self.assertEqual(calibration.scaled(f32(1.5), 2), 0.75)
        self.assertEqual(calibration.logit_of(f32(0.75)), calibration.logit_of(0.75))
        self.assertEqual(calibration.at_one_logit(f32(2.0), 0.75), calibration.at_one_logit(2.0, 0.75))
        self.assertEqual(case.pair(f32(1.5), 1), (F(3, 2), 1))
        self.assertEqual(case.exact(f32(1.5), f32(0.25)), case.exact(1.5, 0.25))
        self.assertEqual(mix.mean([(1.0, f32(0.5)), (2.0, 1)]), mix.mean([(1.0, 0.5), (2.0, 1)]))
        self.assertEqual(propagation.shared_rate(u=f32(2.0 ** -30)), propagation.shared_rate(u=2.0 ** -30))


class TheSettingsOfTheCaller(Record):
    """Section "The caller's decimal settings can break an ordinary rate bound"."""

    def calls(self):
        rec = self.one(D("194.6"), [D("1.7"), D("2.5"), D("0.3")], D("15.7"), D("0.05"), D(200))
        rec["result"]["u"] = dec(F(math.isqrt(923 * 10 ** 58), 10 ** 30)).normalize(C)   # the root of 9.23
        big = D("1." + "0" * 30 + "1")
        return (
            lambda: stats.quiet_bound(30),
            lambda: stats.quiet_bound(D("30.5"), D("0.95")),
            lambda: stats.rate_limits(3, 30),
            lambda: stats.rate_upper(2, F(61, 2)),
            lambda: judge.exact_interval(2, 20),
            lambda: silence.paired_upper(1, [(0, 30)], 10),
            lambda: silence.searched_pairs(TEN, 10),
            lambda: silence.gap_bound(300, 30),
            lambda: silence.gap_somewhere(0.4, 20, 1),
            lambda: silence.some_empty(0.0001, silence.selections(UNIT, 0)),
            lambda: stats.poisson_within(3, D("2.5")),
            lambda: stats.binomial_within(2, 20, D("0.1")),
            lambda: (decide(rec), guard(rec), combined(rec), spread(rec)),
            lambda: printed(check.near, "t", rec, D("194.6")),
            lambda: check.displayed(self.rec),
            lambda: guard(self.one(D(199), big, D(1), D("0.25"), D(200))),
            lambda: dist._float(D("1e-400"), exact=True),
            lambda: dist._float(big, exact=True),
            lambda: stats.upward(big, 2, D(2)),
            lambda: stats.nearest(D("2.5"), 0),
            lambda: record.rounded(D("2.5"), 0),
            lambda: stats.mean([D("1.5"), D("2.25"), big]),
            lambda: stats.quantile([D("1.5"), D("2.25"), big], 0.5),
            lambda: printed(check.main),
            lambda: calibration.sure([F(1, 3), F(-2, 7), F(5, 9)], F(11, 10), 50),
        )

    def test_the_bound_of_the_review(self):
        true = C.divide(C.ln(C.divide(D(1), dec(1 - F(0.95)))), D(30))
        want = least_float(F(true))
        self.assertEqual(want, 0.09985774245179968)                        # the value of the review
        narrow = decimal.Context(prec=28, Emin=-10, Emax=10, traps=[decimal.Underflow])
        with decimal.localcontext(narrow):
            got = stats.quiet_bound(30)                                    # Version 71: decimal.Underflow
            interval = judge.exact_interval(2, 20)                         # Version 71: decimal.Underflow
        self.assertEqual(got, want)
        self.assertEqual(interval, judge.exact_interval(2, 20))

    def test_a_context_of_three_digits_with_every_trap(self):
        usual = [call() for call in self.calls()]
        with decimal.localcontext(hostile()):
            under = [call() for call in self.calls()]
            self.assertEqual(refusal(lambda: D(1) / D(3)), "Inexact: [<class 'decimal.Inexact'>, "
                             "<class 'decimal.Rounded'>]")               # the context does what it says
        for i, (a, b) in enumerate(zip(usual, under)):
            self.assertEqual(a, b, i)
        self.assertEqual(usual[23], expected("ch22_record_check"))

    def test_the_examples_under_such_a_context(self):
        # The nine examples that take decimal steps or search a limit, each in a process of its own in which
        # the context of three digits is set before anything of the repository is imported.
        names = ("ch03_scales_units", "ch16_silence", "ch16_trace_measurements", "ch17_accuracy",
                 "ch18_calibration", "ch20_judge", "ch22_record_check", "ch23_case", "ch23_mix")
        code = (
            "import contextlib, decimal, importlib, io, pathlib\n"
            "signals = [decimal.Inexact, decimal.Rounded, decimal.Underflow, decimal.Subnormal,\n"
            "           decimal.Overflow, decimal.InvalidOperation, decimal.DivisionByZero, decimal.Clamped]\n"
            "decimal.setcontext(decimal.Context(prec=3, Emax=10, Emin=-10, rounding=decimal.ROUND_UP,\n"
            "                                   traps=signals))\n"
            f"for name in {names!r}:\n"
            "    out = io.StringIO()\n"
            "    with contextlib.redirect_stdout(out):\n"
            "        importlib.import_module('examples.' + name).main()\n"
            "    want = pathlib.Path('expected', name + '.txt').read_text(encoding='utf-8')\n"
            "    print(name, out.getvalue() == want)\n")
        out = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr[-400:])
        self.assertEqual(out.stdout.split(), [word for name in names for word in (name, "True")])

    def test_rounding_as_the_caller_rounds(self):
        # round() of a decimal rounds as the context of the thread says: under one that rounds away from zero
        # the record of the display would hold other numbers.
        away = decimal.Context(prec=28, rounding=decimal.ROUND_UP)
        with decimal.localcontext(away):
            self.assertEqual(round(D("3.041"), 1), D("3.1"))               # what Version 71 relied on
            self.assertEqual(record.rounded(D("3.041"), 1), D("3.0"))
            self.assertEqual(check.two(D("3.041")), D("3.0"))
            shown = check.displayed(self.rec)
        self.assertEqual(shown, check.displayed(self.rec))
        # Half to even, the places kept, and the tens for places below zero.
        for x, places, want in (("2.5", 0, "2"), ("3.5", 0, "4"), ("0.125", 2, "0.12"), ("-0.135", 2, "-0.14"),
                                ("3.049", 1, "3.0"), ("1250", -2, "1.2E+3"), ("1350", -2, "1.4E+3"),
                                ("15.732701986622827", 0, "16"), ("0", 2, "0.00")):
            got = record.rounded(D(x), places)
            self.assertEqual(str(got), want, x)
            self.assertEqual(got, round(D(x), places), x)                  # under the context of the tests
        self.assertEqual((record.rounded(2.675, 2), record.rounded(7, -1)), (round(2.675, 2), round(7, -1)))
        self.assertIsInstance(record.rounded(2.675, 2), float)

    def test_a_default_that_a_program_changed(self):
        # What a context does not state, it takes from decimal.DefaultContext. A program may change that
        # default before it starts its threads, and the contexts of the package state all that they have.
        code = (
            "import decimal\n"
            "d = decimal.DefaultContext\n"
            "d.prec, d.Emax, d.Emin, d.rounding = 3, 10, -10, decimal.ROUND_UP\n"
            "for s in (decimal.Inexact, decimal.Rounded, decimal.Underflow, decimal.Subnormal):\n"
            "    d.traps[s] = True\n"
            "decimal.setcontext(decimal.Context())\n"
            "from fractions import Fraction as F\n"
            "from swmetro import stats\n"
            "from examples import ch18_calibration as c\n"
            "from examples import ch20_judge as j\n"
            "print(stats.quiet_bound(30), j.exact_interval(2, 20),\n"
            "      c.sure([F(1, 3), F(-2, 7), F(5, 9)], F(11, 10), 50))\n")
        out = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr[-400:])
        want = (stats.quiet_bound(30), judge.exact_interval(2, 20),
                calibration.sure([F(1, 3), F(-2, 7), F(5, 9)], F(11, 10), 50))
        self.assertEqual(out.stdout.strip(), " ".join(map(str, want)))

    def test_the_contexts_state_all_that_they_have(self):
        for context in (stats.stated(40), stats.stated(80, decimal.ROUND_FLOOR), *stats._sides(40)):
            self.assertEqual((context.Emax, context.Emin, context.clamp, context.capitals),
                             (decimal.MAX_EMAX, decimal.MIN_EMIN, 0, 1))
            on = {s for s in SIGNALS if context.traps[s]}
            self.assertEqual(on, {decimal.InvalidOperation, decimal.DivisionByZero, decimal.Overflow})
        down, up = stats._sides(40)
        self.assertEqual((down.rounding, up.rounding, down.prec), (decimal.ROUND_FLOOR, decimal.ROUND_CEILING, 40))
        self.assertEqual(stats.stated(40).rounding, decimal.ROUND_HALF_EVEN)


class WhatTheBookPrints(unittest.TestCase):
    """Numbers of the revised text."""

    def test_the_thresholds_of_chapter_15(self):
        # The example of the review: with a blank mean of 10 errors a minute and a standard deviation of 2 the
        # critical value is 13.29 as an indication and 3.29 as an excess, and the mean indication at the
        # detection limit is 16.58. The text gives the two as mu_0 + 1.645 sigma_0 and mu_0 + 3.29 sigma_0, and
        # the listing works with the excess.
        mean, sd = 10, 2
        self.assertEqual((F("1.645") * sd, mean + F("1.645") * sd), (F("3.29"), F("13.29")))
        self.assertEqual((F("3.29") * sd, mean + F("3.29") * sd), (F("6.58"), F("16.58")))
        lines = expected("ch15_detection_limit")
        self.assertIn("critical value  L_C = 3.29 /min\n", lines)
        self.assertIn("detection limit L_D = 6.58 /min\n", lines)

    def test_numbers_that_the_readme_quotes(self):
        self.assertEqual(stats.upward(D("94.9934"), 2, 95), "94.994")
        self.assertEqual(stats.nearest(D("200.004"), 2, D("200.003")), "200.004")
        self.assertEqual(F(0.2) + F(0.8), 1 + F(1, 2 ** 54))


if __name__ == "__main__":
    unittest.main()
