"""Regression tests added after the review of Version 28.

The review found that a line fitted from time stamps of nineteen digits to
elapsed nanoseconds lost its intercept, and one fitted to times of 2**53 + 1
ns apart its slope, although the fit reported no uncertainty for either;
that a result whose value is a fraction could not be reported under Python
3.11, which has no fixed-point format for fractions; that the chance of a
gap given the count belongs to a Poisson process and not to every source of
constant rate; and that the least and the greatest of ten draws, returned
where 99 % was asked for, hold 9/11 of a distribution on average. It
proposed an example of a silence under a reference rate that changes.

Each test computes what it expects independently of the code under test:
a line fitted about the means in fractions, the decimal module for the
rounding of a fraction, a process simulated from its exponential gaps, a
changing rate simulated in time and transformed by a formula written out
here, and seeded draws for what an interval of ranks holds.
"""
import contextlib
import decimal
import io
import math
import random
import sys
import unittest
from fractions import Fraction
from pathlib import Path

from examples import ch16_silence as silence
from swmetro import core, propagate, stats, units
from swmetro.core import Result

ROOT = Path(__file__).resolve().parent.parent
EPOCH = 1_700_000_000_000_000_000                # ns from 1970 to 14 November 2023, 22:13:20 UTC
ELAPSED = [0, 1, 2]                              # three readings of an elapsed clock, ns
STAMPS = [EPOCH + 100 + i for i in ELAPSED]      # the same three moments on a clock that counts from 1970
FEW = "more draws needed"


def expected(name):
    return (ROOT / "expected" / f"{name}.txt").read_text(encoding="utf-8")


def printed(function, *args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        function(*args)
    return out.getvalue()


def exact_line(xs, ys):
    """Least squares about the two means, in fractions: a, b, and the squares of u(a) and u(b)."""
    fx, fy = [Fraction(x) for x in xs], [Fraction(y) for y in ys]
    n = len(fx)
    mx, my = sum(fx) / n, sum(fy) / n
    sxx = sum((x - mx) ** 2 for x in fx)
    b = sum((x - mx) * (y - my) for x, y in zip(fx, fy)) / sxx
    a = my - b * mx
    var = sum((y - a - b * x) ** 2 for x, y in zip(fx, fy)) / (n - 2)
    return a, b, var * (Fraction(1, n) + mx ** 2 / sxx), var / sxx


def version_28_line(xs, ys):
    """The two coefficients as Version 28 returned them: the slope as a float, and the intercept exact only
    where a y was a number that no float holds and a float could not show a thousandth of its uncertainty."""
    a, b, ua2, _ = exact_line(xs, ys)
    held = all(isinstance(y, float) or (abs(y) <= 2 ** 53 and float(y) == y) for y in ys)
    if float(a) != a and not held and 1000 * math.ulp(float(a)) > math.sqrt(ua2):
        return (a.numerator if a.denominator == 1 else a), float(b)
    return float(a), float(b)


class AClockFitInBothDirections(unittest.TestCase):
    def test_the_table_of_the_review(self):
        self.assertEqual(version_28_line(ELAPSED, STAMPS), (EPOCH + 100, 1.0))
        old = version_28_line(STAMPS, ELAPSED)
        self.assertEqual((old, type(old[0])), ((-float(EPOCH), 1.0), float))         # 100 ns of the offset are gone
        self.assertEqual([int(old[0]) + x for x in STAMPS], [100, 101, 102])         # and the readings were 0, 1, 2
        for xs, ys, want in ((ELAPSED, STAMPS, EPOCH + 100), (STAMPS, ELAPSED, -EPOCH - 100)):
            a, b, ua, ub, s = stats.linfit(xs, ys)
            self.assertEqual((a, type(a), b, ua, ub, s), (want, int, 1.0, 0.0, 0.0, 0.0))
            self.assertEqual([a + Fraction(b) * x for x in xs], ys)                   # applied in exact arithmetic
            self.assertEqual(exact_line(xs, ys), (want, 1, 0, 0))

    def test_the_period_of_the_review(self):
        period = 2 ** 53 + 1
        times = [0, period, 2 * period]
        self.assertEqual(version_28_line(ELAPSED, times), (0.0, float(2 ** 53)))      # 1 ns per event short
        a, b, ua, ub, s = stats.linfit(ELAPSED, times)
        self.assertEqual((a, b, type(b), ua, ub, s), (0.0, period, int, 0.0, 0.0, 0.0))
        # A period of half a nanosecond more is a fraction, and it is kept as well.
        halves = [0, Fraction(2 * period + 1, 2), 2 * period + 1]
        self.assertEqual(stats.linfit(ELAPSED, halves)[:2], (0.0, Fraction(2 * period + 1, 2)))

    def test_no_residual_no_rounding(self):
        rng = random.Random(2801)
        kept = 0
        for _ in range(400):
            origin = rng.choice((0, 1000, 2 ** 52, EPOCH, 2 ** 62, -EPOCH, 10 ** 30))
            slope = Fraction(rng.choice((-7, -3, -1, 1, 2, 5, 50_001)), rng.choice((1, 2, 3, 7, 10, 50_000)))
            offset = Fraction(rng.randint(-10 ** 6, 10 ** 6), rng.choice((1, 1, 2, 10)))
            if rng.random() < 0.3:
                offset -= slope * origin                                              # a line through small readings
            xs = [origin + t for t in sorted(rng.sample(range(0, 5000), rng.randint(3, 9)))]
            ys = [slope * x + offset for x in xs]
            ys = [y.numerator if y.denominator == 1 else y for y in ys]
            a, b, ua, ub, s = stats.linfit(xs, ys)
            self.assertEqual((ua, ub, s), (0.0, 0.0, 0.0))
            self.assertEqual((Fraction(a), Fraction(b)), (offset, slope), (xs, ys))   # what the fit found, to the last digit
            self.assertEqual([Fraction(a) + Fraction(b) * x for x in xs], ys)
            for got, want in ((a, offset), (b, slope)):
                self.assertIs(type(got), float if float(want) == want else int if want.denominator == 1 else Fraction)
                kept += type(got) is not float
        self.assertGreater(kept, 300)

    def test_with_residuals_within_a_thousandth_of_the_uncertainty(self):
        rng = random.Random(2802)
        kept = 0
        for _ in range(400):
            origin = rng.choice((EPOCH, 2 ** 62, -EPOCH, 10 ** 30))
            n = rng.randint(3, 10)
            small = sorted(rng.sample(range(0, 4000), n))
            far = [origin + t + rng.choice((0, 0, 0, 1, -1, rng.randint(-40, 40))) for t in small]
            if len(set(far)) < 2 or exact_line(small, far)[3] == 0:
                continue
            for xs, ys in ((small, far), (far, small)):
                a, b, ua2, ub2 = exact_line(xs, ys)
                got = stats.linfit(xs, ys)
                for value, u, want, u2 in ((got[0], got[2], a, ua2), (got[1], got[3], b, ub2)):
                    self.assertAlmostEqual(u / math.sqrt(u2), 1.0, places=12)
                    self.assertLessEqual(1000 * abs(Fraction(value) - want), Fraction(u), (xs, ys))
                    kept += type(value) is not float
        self.assertGreater(kept, 100)
        # The case that the last response left: four stamps on a line but for one nanosecond.
        ys = [EPOCH + 100, EPOCH + 101, EPOCH + 101, EPOCH + 102]
        for xs in ([0, 1, 2, 3], [0.0, 1.0, 2.0, 3.0]):
            a, b, ua, ub, s = stats.linfit(xs, ys)
            self.assertEqual((a - EPOCH, type(a), b), (Fraction(1001, 10), Fraction, 0.6))

    def test_readings_that_floats_hold(self):
        # Decimal readings stored as floats lie on their line to the last bit or two. Versions 28 to 30 gave
        # them the nearest floats. Since Version 31 the coefficients are not rounded where the uncertainty of
        # the fitted line could show the rounding, however the readings are stored, and these are the fractions
        # of the binary numbers that the readings are; float() of them gives the 1.0 and the 2.0 of before.
        xs = [0.1 * k for k in range(10)]
        ys = [1 + 2 * x for x in xs]
        a, b, ua2, ub2 = exact_line(xs, ys)
        got = stats.linfit(xs, ys)
        self.assertTrue(0 < ua2 and 0 < ub2 and got[2] < 1e-15 and got[3] < 1e-15)
        self.assertEqual((got[:2], [type(v) for v in got[2:]]), ((a, b), [float] * 3))
        self.assertEqual((float(got[0]), float(got[1])), (1.0, 2.0))
        rng = random.Random(2803)
        for _ in range(200):
            n = rng.randint(3, 12)
            xs = [rng.choice((rng.uniform(0, 60), rng.randint(0, 60))) for _ in range(n)]
            ys = [2.5 + 0.75 * x + rng.gauss(0, 0.3) for x in xs]
            if len(set(xs)) < 2:
                continue
            a, b, _, _ = exact_line(xs, ys)
            got = stats.linfit(xs, ys)
            self.assertEqual((got[:2], [type(v) for v in got]), ((float(a), float(b)), [float] * 5))
        self.assertEqual([type(v) for v in stats.linfit([1, 2, 3, 4], [2, 4, 5, 8])], [float] * 5)
        self.assertEqual(stats.linfit([0, 1, 2], [1, 3, 5]), (1.0, 2.0, 0.0, 0.0, 0.0))
        # Without residuals nothing is rounded here either: a third, and an intercept of 2**52 + 1/2, which
        # as a float is half a unit away while every reading is one that a float holds.
        self.assertEqual(stats.linfit([0, 3, 6], [0, 1, 2]), (0.0, Fraction(1, 3), 0.0, 0.0, 0.0))
        self.assertEqual(stats.linfit([0.0, 3.0, 6.0], [0.0, 1.0, 2.0])[1], Fraction(1, 3))
        xs, ys = [2 ** 52 + i for i in (1, 2, 3)], [0.5, 1.5, 2.5]
        a, b, ua, ub, s = stats.linfit(xs, ys)
        self.assertEqual((a, type(a), b, ua), (-Fraction(2 ** 53 + 1, 2), Fraction, 1.0, 0.0))
        self.assertEqual(abs(Fraction(float(a)) - a), Fraction(1, 2))
        self.assertEqual([a + Fraction(b) * x for x in xs], ys)

    def test_the_centered_form(self):
        a, b = stats.linfit(STAMPS, ELAPSED)[:2]
        self.assertEqual((a, b), (-(EPOCH + 100), 1.0))
        # The elapsed time is the stamp less the stamp at which the elapsed clock read zero, in whole numbers.
        self.assertEqual([x - (EPOCH + 100) for x in STAMPS], ELAPSED)
        # In floating point even the exact intercept does not help: the product b * x rounds the stamp first.
        self.assertEqual([a + b * x for x in STAMPS], [0.0, 0.0, 0.0])
        self.assertEqual([float(x) - float(EPOCH + 100) for x in STAMPS], [0.0, 0.0, 0.0])
        # With an origin subtracted from the stamps before the fit, the line is an ordinary one.
        got = stats.linfit([x - EPOCH for x in STAMPS], ELAPSED)
        self.assertEqual((got, [type(v) for v in got]), ((-100.0, 1.0, 0.0, 0.0, 0.0), [float] * 5))


class Python311(Fraction):
    """A fraction with the formats of Python 3.11: none but the empty one."""

    def __format__(self, spec):
        if spec:
            raise TypeError("unsupported format string passed to Fraction.__format__")
        return str(self)


def by_decimal(value, places):
    """A fraction with a number of decimal places, rounded half to even by the decimal module."""
    with decimal.localcontext() as ctx:
        ctx.prec = 120
        number = decimal.Decimal(value.numerator) / decimal.Decimal(value.denominator)
        text = format(number.quantize(decimal.Decimal(1).scaleb(-places), rounding=decimal.ROUND_HALF_EVEN), "f")
    # Since Version 63 a value that rounds to zero is shown without a sign, which says nothing there.
    return text.lstrip("-") if not decimal.Decimal(text) else text


class AFractionInAReport(unittest.TestCase):
    MEDIAN = Fraction(2 * EPOCH + 201, 2)

    def test_the_case_of_the_review(self):
        median = stats.quantile([EPOCH + 100, EPOCH + 101], 0.5)
        self.assertEqual((median, type(median)), (self.MEDIAN, Fraction))
        if sys.version_info < (3, 12):
            with self.assertRaises(TypeError):
                format(median, ".3f")                                                # what Version 28 ran into
        for value in (median, Python311(median)):
            result = Result(value, 0, "ns")
            self.assertEqual(result.report(), "(1700000000000000100.5 ± 0) ns, u = 0, p = 0.95")   # Version 62: k = 1.96
            self.assertEqual(str(result), "1700000000000000100.5(0) ns")
            self.assertEqual(result.interval(), (median, median))

    def test_a_third_with_an_uncertainty(self):
        for third in (Fraction(1, 3), Python311(1, 3)):
            result = Result(third, 0.01, "s")
            self.assertEqual((str(result), result.report()), ("0.333(10) s", "(0.333 ± 0.020) s, k = 1.96, p = 0.95"))
            self.assertEqual((str(result), result.report()), (str(Result(1 / 3, 0.01, "s")), Result(1 / 3, 0.01, "s").report()))
            # An uncertainty in the tens, where the value is rounded to the tens or to whole numbers.
            large = Result(1000 * third, 20, "s")
            self.assertEqual((str(large), large.report()), ("333(20) s", "(333 ± 39) s, k = 1.96, p = 0.95"))
            huge = Result(10 ** 6 * third, 2500)
            self.assertEqual((str(huge), huge.report()), ("333300(2500)", "(333300 ± 4900), k = 1.96, p = 0.95"))
            self.assertEqual(str(huge), str(Result(10 ** 6 / 3, 2500)))
            # Without an uncertainty a third has no decimal places to end on, and is shown as what it is.
            alone = Result(third, 0)
            self.assertEqual((str(alone), alone.report()), ("1/3(0)", "(1/3 ± 0), u = 0, p = 0.95"))

    def test_fixed_point_text(self):
        rng = random.Random(2804)
        for _ in range(3000):
            value = Fraction(rng.randint(-10 ** rng.randint(0, 22), 10 ** rng.randint(0, 22)),
                             rng.choice((1, 2, 3, 7, 8, 10, 125, 1000, rng.randint(1, 10 ** 6))))
            for places in (0, 1, 2, 3, 6):
                want = by_decimal(value, places)
                self.assertEqual(core.fixed(value, places), want, (value, places))
                self.assertEqual(core.fixed(Python311(value), places), want)
                if sys.version_info >= (3, 12):
                    python = format(value, f".{places}f")
                    self.assertEqual(python.lstrip("-") if not Fraction(python) else python, want)
        # Ties go to the even digit, as they do for a float that is a tie.
        self.assertEqual([core.fixed(Fraction(k, 2), 0) for k in (1, 3, 5, -1, -3)], ["0", "2", "2", "0", "-2"])   # Version 62: "-0" for -1/2
        self.assertEqual([f"{k / 2:.0f}" for k in (1, 3, 5, -1, -3)], ["0", "2", "2", "-0", "-2"])
        self.assertEqual([core.fixed(Fraction(k, 1000), 2) for k in (125, 135, -5)], ["0.12", "0.14", "0.00"])   # Version 62: "-0.00"
        # A whole number keeps its digits, and a float is formatted as before.
        self.assertEqual(core.fixed(EPOCH + 1, 2), "1700000000000000001.00")
        self.assertEqual(f"{EPOCH + 1:.2f}", "1700000000000000000.00")               # through a float
        self.assertEqual((core.fixed(-7, 0), core.fixed(1, 1)), ("-7", "1.0"))
        with self.assertRaises(TypeError):                                           # until Version 67: "1.0"
            core.fixed(True, 1)
        for _ in range(500):
            x = rng.choice((1e-9, 1.0, 842.0, 1e12)) * rng.uniform(-1.5, 1.5)
            for places in (0, 2, 5):
                want = f"{x:.{places}f}"
                self.assertEqual(core.fixed(x, places), want.lstrip("-") if not float(want) else want)   # see by_decimal()

    def test_a_value_in_full(self):
        self.assertEqual([core.full(Fraction(1, 8)), core.full(Fraction(-1, 8)), core.full(Fraction(7))], ["0.125", "-0.125", "7"])
        self.assertEqual([core.full(Fraction(1, 3)), core.full(Fraction(-22, 7)), core.full(Fraction(1, 6))], ["1/3", "-22/7", "1/6"])
        self.assertEqual(core.full(self.MEDIAN), "1700000000000000100.5")
        # A float taken as a fraction has the decimal expansion of the binary number that it is.
        self.assertEqual(core.full(Fraction(0.1)), format(decimal.Decimal(0.1), "f"))
        self.assertEqual(len(core.full(Fraction(0.1))), 57)
        with decimal.localcontext() as ctx:
            ctx.prec = 200
            self.assertEqual(core.full(Fraction(3, 2 ** 70 * 5 ** 3)), format(decimal.Decimal(3) / decimal.Decimal(2 ** 70 * 5 ** 3), "f"))
        # Floats and whole numbers are shown as before.
        self.assertEqual([core.full(0.1), core.full(99.9996), core.full(EPOCH + 100)], ["0.1", "99.9996", "1700000000000000100"])
        self.assertEqual(str(Result(99.9996, 0, "ms")), "99.9996(0) ms")
        self.assertEqual(str(Result(12.311, 0.0097, "ms")), "12.3110(97) ms")
        self.assertEqual(Result(0.5, 0, "ns").report(), "(0.5 ± 0) ns, u = 0, p = 0.95")

    def test_whole_stamps_keep_their_digits(self):
        # Found while the review was worked: a whole value was formatted through a float.
        stamp = EPOCH + 10_000
        self.assertEqual(f"{float(stamp):.0f}", "1700000000000009984")               # the digits of Version 28
        result = Result(stamp, 300_000, "ns")
        self.assertEqual(str(result), "1700000000000010000(300000) ns")
        self.assertEqual(result.report(), "(1700000000000010000 ± 590000) ns, k = 1.96, p = 0.95")
        self.assertEqual(f"{result.expanded():.0f}", "587989")
        self.assertEqual(str(Result(EPOCH + 123_456, 300_000, "ns")), "1700000000000120000(300000) ns")
        self.assertEqual(str(Result(Fraction(2 * EPOCH + 246_913, 2), 300_000, "ns")), "1700000000000120000(300000) ns")
        # A whole value of ordinary size reads as the float of the same value does, a tie to the tens included.
        for value in (12_345, 12_355, -12_345, 7):
            for u in (300, 4, 0.25):
                whole, as_float = Result(value, u), Result(float(value), u)
                self.assertEqual((str(whole), whole.report()), (str(as_float), as_float.report()), (value, u))
        self.assertEqual((str(Result(12_345, 300)), Result(12_345, 300).report()), ("12340(300)", "(12340 ± 590), k = 1.96, p = 0.95"))

    def test_a_probability_and_a_quantity_that_are_fractions(self):
        result = Result(1.0, 0.1, "s")
        self.assertEqual(result.report(Fraction(19, 20)), result.report(0.95))
        self.assertEqual(result.report(Python311(99, 100)), "(1.00 ± 0.26) s, k = 2.58, p = 0.99")
        self.assertEqual(repr(units.Q(Fraction(1, 3), "s")), "0.333333 s")
        self.assertEqual(repr(units.Q(Python311(1, 3), "s")), "0.333333 s")
        self.assertEqual([repr(units.Q(1.5, "ms")), repr(units.Q(3, "s")), repr(units.Q(1.5, "ms").to("s"))], ["1.5 ms", "3 s", "0.0015 s"])


def rule_of_the_supplement(m, p):
    """JCGM 101:2008, 7.7.2: q = pM if that is a whole number, otherwise the integer part of pM + 1/2; the
    interval from the r-th to the (r + q)-th of the M sorted values, for r from 1 to M - q."""
    pm = Fraction(p) * m
    return int(pm) if pm.denominator == 1 else int(pm + Fraction(1, 2))


class TooFewDrawsForTheProbability(unittest.TestCase):
    def test_ten_draws(self):
        rng = random.Random(2805)
        ys = sorted(rng.gauss(0, 1) for _ in range(10))
        # Version 28: q is cut back to n - 1, and the least and the greatest draw come out for 99 %.
        q = min(rule_of_the_supplement(10, Fraction(99, 100)), 10 - 1)
        self.assertEqual(((10 - q + 1) // 2, (10 - q + 1) // 2 + q), (1, 10))
        for p in (0.99, "0.99", Fraction(99, 100), decimal.Decimal("0.99"), 0.95, 1):
            self.assertEqual(rule_of_the_supplement(10, Fraction(str(p))), 10)       # ten steps, and no r from 1 to 0
            with self.assertRaisesRegex(ValueError, FEW):
                propagate.summarize(ys, p)
        # Nine steps among ten draws are the rule's own interval, from the first draw to the tenth.
        self.assertEqual(rule_of_the_supplement(10, Fraction(9, 10)), 9)
        self.assertEqual(propagate.summarize(ys, 0.9)[2:], (ys[0], ys[-1]))

    def test_the_fewest_draws(self):
        for p, fewest in ((Fraction(1, 2), 2), (Fraction(9, 10), 6), (Fraction(19, 20), 11), (Fraction(99, 100), 51),
                          (Fraction(999, 1000), 501)):
            self.assertEqual(min(m for m in range(2, 600) if rule_of_the_supplement(m, p) <= m - 1), fewest)
            self.assertEqual(propagate.summarize(range(1, fewest + 1), p)[2:], (1, fewest))
            if fewest > 2:
                with self.assertRaisesRegex(ValueError, FEW):
                    propagate.summarize(range(1, fewest), p)
        for m in (2, 10, 1000):
            with self.assertRaisesRegex(ValueError, FEW):
                propagate.summarize(range(1, m + 1), 1)                              # no number of draws shows 100 %
        # The 200 000 draws of Chapter 5 are ten thousand times 1 / (1 - p), which 7.2.2 of the supplement names.
        self.assertEqual(10 ** 4 / (1 - Fraction(19, 20)), 200_000)
        self.assertIn("MC, independent normal, 200000 draws:\n", expected("ch05_propagation"))

    def test_what_an_interval_of_ranks_holds(self):
        rng = random.Random(2806)
        runs = 40_000
        inside, held = 0, 0.0
        for _ in range(runs):
            ten = [rng.random() for _ in range(10)]
            inside += min(ten) < rng.random() < max(ten)                             # an eleventh draw of the same kind
            held += max(ten) - min(ten)                                              # what the range holds of a uniform
        share = Fraction(9, 11)
        self.assertAlmostEqual(inside / runs, share, delta=4.5 * math.sqrt(share * (1 - share) / runs))
        spread = math.sqrt(9 * 2 / (11 ** 2 * 12))                                   # of a beta distribution, 9 and 2
        self.assertAlmostEqual(held / runs, share, delta=4.5 * spread / math.sqrt(runs))
        self.assertEqual(f"{100 * float(share):.2f}", "81.82")
        # A hundred draws at 95 %: the ranks 3 and 98, 95 steps, and 95/101 on average.
        low, high = propagate.summarize(range(1, 101), 0.95)[2:]
        self.assertEqual((low, high), (3, 98))
        runs, held = 4000, 0.0
        for _ in range(runs):
            us = sorted(rng.random() for _ in range(100))
            held += us[high - 1] - us[low - 1]
        spread = math.sqrt(95 * 6 / (101 ** 2 * 102))
        self.assertAlmostEqual(held / runs, 95 / 101, delta=4.5 * spread / math.sqrt(runs))
        # Chapter 5: 190 000 steps among 200 000 draws.
        self.assertEqual(rule_of_the_supplement(200_000, Fraction(19, 20)), 190_000)
        self.assertEqual(f"{100 * 190_000 / 200_001:.3f}", "95.000")


def largest_gap_at_least(n, share):
    """Chance that the largest of the n + 1 gaps left by n uniform points is at least `share` of the whole."""
    out = Fraction(0)
    for k in range(1, n + 2):
        rest = 1 - k * Fraction(share)
        if rest < 0:
            break
        term = math.comb(n + 1, k) * rest ** n
        out += term if k % 2 else -term
    return out


def flat(text):
    return " ".join(text.split())


class TheModelBehindTheGapFigure(unittest.TestCase):
    def test_a_heartbeat_has_the_same_current_and_count(self):
        rng = random.Random(2807)
        for _ in range(2000):
            phase = Fraction(rng.randrange(0, 5000), 1000)                           # the first beat, 0 s to 5 s
            times = [phase + 5 * k for k in range(100) if phase + 5 * k < 300]
            self.assertEqual((len(times), Fraction(len(times), 300)), (60, Fraction(1, 5)))
            gaps = [b - a for a, b in zip([Fraction(0)] + times, times + [Fraction(300)])]
            self.assertEqual((max(gaps), sum(gaps)), (5, 300))
        # The same count and the same Statement Current as the records of Figure 16.5, and no gap of 30 s.
        self.assertEqual(f"{100 * silence.gap_given(60, 300, 30):.2f}", "10.68")

    def test_a_poisson_process_given_its_count(self):
        # Messages with independent exponential gaps at 0.2 msg/s, and of those records the ones with 60.
        rng = random.Random(2808)
        runs, kept, hits = 100_000, 0, {30: 0, 35: 0}
        for _ in range(runs):
            t, times = rng.expovariate(0.2), []
            while t < 300 and len(times) <= 60:
                times.append(t)
                t += rng.expovariate(0.2)
            if len(times) != 60:
                continue
            kept += 1
            largest = max(b - a for a, b in zip([0.0] + times, times + [300.0]))
            for length in hits:
                hits[length] += largest >= length
        at_sixty = math.exp(-60 + 60 * math.log(60) - math.lgamma(61))                # the Poisson chance of 60
        self.assertAlmostEqual(kept / runs, at_sixty, delta=4.5 * math.sqrt(at_sixty * (1 - at_sixty) / runs))
        self.assertGreater(kept, 4500)
        for length, count in hits.items():
            want = float(largest_gap_at_least(60, Fraction(length, 300)))
            self.assertEqual(silence.gap_given(60, 300, length), want)
            self.assertAlmostEqual(count / kept, want, delta=4.5 * math.sqrt(want * (1 - want) / kept), msg=length)

    def test_the_model_is_named(self):
        for function in (silence.gap_given, silence.gap_somewhere):
            self.assertIn("homogeneous Poisson process", flat(function.__doc__))
        self.assertIn("n independent uniform times, put in order", flat(silence.gap_given.__doc__))
        source = (ROOT / "figures" / "make_figures.py").read_text(encoding="utf-8")
        self.assertIn("A homogeneous Poisson process, observed for 300 s, given exactly 60", source)


def expected_count(t):
    """Messages expected up to t seconds at 0.10 msg/s for 150 s and at 0.30 msg/s after that."""
    return t / 10 if t <= 150 else 15 + 3 * (t - 150) / 10


class TheSameSilenceUnderAChangingRate(unittest.TestCase):
    def test_time_in_expected_messages(self):
        self.assertEqual([silence.expected(t) for t in (0, 60, 90, 150, 210, 240, 300)], [0, 6, 9, 15, 33, 42, 60])
        for t in list(range(0, 301, 7)) + [Fraction(1, 3), Fraction(4501, 30), 149.5, 150.25]:
            self.assertEqual(silence.expected(t), expected_count(Fraction(t)), t)
        self.assertEqual((silence.expected(-5), silence.expected(400)), (0, 60))      # no messages outside the record
        self.assertEqual(silence.QUIET, ((60, 90), (210, 240)))
        self.assertEqual([hi - lo for lo, hi in silence.QUIET], [30, 30])

    def test_the_table_of_the_review(self):
        rows = []
        for lo, hi in silence.QUIET:
            spans = Fraction(expected_count(Fraction(hi)) - expected_count(Fraction(lo)))
            empty = (1 - spans / 60) ** 60                                           # each of 60 messages elsewhere
            found = largest_gap_at_least(60, spans / 60)
            self.assertEqual(silence.gap_given(60, 60, spans), float(found))
            rows.append((spans, f"{100 * float(empty):.4g}", f"{100 * float(found):.4g}"))
        self.assertEqual(rows, [(3, "4.607", "98.16"), (9, "0.005823", "0.3551")])
        self.assertEqual(f"{100 * 0.85 ** 60:.3g}", "0.00582")                       # as the review prints it
        # Nine of sixty expected messages are the share that 45 s have of 300 s at a constant rate.
        self.assertEqual(silence.gap_given(60, 60, 9), silence.gap_given(60, 300, 45))

    def test_simulated_records(self):
        # Each of the 60 messages falls into the first half with the chance 15/60 and is uniform in its half.
        rng = random.Random(2809)
        runs, empty, found = 40_000, {60: 0, 210: 0}, {3: 0, 9: 0}
        for _ in range(runs):
            times = sorted(rng.uniform(0, 150) if rng.random() < 0.25 else rng.uniform(150, 300) for _ in range(60))
            for start in empty:
                empty[start] += not any(start <= t < start + 30 for t in times)
            counts = [0.0] + [expected_count(t) for t in times] + [60.0]
            largest = max(b - a for a, b in zip(counts, counts[1:]))
            for span in found:
                found[span] += largest >= span
        for count, want in ((empty[60], 0.95 ** 60), (found[3], float(largest_gap_at_least(60, Fraction(3, 60)))),
                            (found[9], float(largest_gap_at_least(60, Fraction(9, 60))))):
            self.assertAlmostEqual(count / runs, want, delta=4.5 * math.sqrt(want * (1 - want) / runs))
        self.assertLessEqual(empty[210], 12)                                         # 2.3 expected of 40 000
        self.assertGreater(found[9], 20 * max(empty[210], 1))                        # a search is another question

    def test_the_transcript(self):
        out = printed(silence.search)
        piece = ("    45 s or more: 0.36 %\n"
                 "  a rate of 0.10 msg/s for 150 s, then\n  0.30 msg/s for 150 s, 60 counted:\n"
                 "    60 s to 90 s spans 3 expected messages\n"
                 "      chosen in advance, empty: 4.607 %\n      a gap of 3 or more: 98.16 %\n"
                 "    210 s to 240 s spans 9 expected messages\n"
                 "      chosen in advance, empty: 0.005823 %\n      a gap of 9 or more: 0.3551 %\n"
                 "  a second record of 60 messages as\n")      # the line that follows since Version 30
        self.assertIn(piece, out)
        self.assertEqual(printed(silence.changing).count("\n"), 8)
        text = expected("ch16_silence")
        self.assertTrue(text.endswith(out))
        self.assertEqual(len(text.splitlines()), 193)        # 171 in Version 28, 179 in Version 29


class TheFifteenthFigure(unittest.TestCase):
    def test_the_changing_rate_and_the_renumbered_starts(self):
        source = (ROOT / "figures" / "make_figures.py").read_text(encoding="utf-8")
        for piece in ("def changing_rate():", "float(c16s.expected(t))", "for lo, hi in c16s.QUIET:", "c16s.RISE[0][0]",
                      "assert (count(turn), count(total), spans) == (15, 60, [3, 9])",
                      'save(fig, "fig-16-5-gap-given-count.png")', 'save(fig, "fig-16-6-changing-rate.png")',
                      'save(fig, "fig-16-7-shared-starts.png")', "    gap_given_count()\n    changing_rate()\n    shared_starts()\n"):
            self.assertIn(piece, source)
        self.assertEqual(source.count("\n    save(fig, "), 32)         # fourteen in Version 28


if __name__ == "__main__":
    unittest.main()
