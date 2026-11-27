"""Regression tests after the review of Version 69.

The review found three printed bounds that did not enclose what they bounded: the range of Figure 22.1, the upper
limit of the exact interval for one message in 30 s, and the greatest share of requests within 200 ms under a cap of
800 ms. It asked for one convention for printed bounds, and found that the Poisson bound for no event could be
rounded below its value before any printing; that a NaN could still take a place among the values of a rank helper;
that a change with the same mix in both periods printed a zero for the change of the mix where an infinity is taken
from an infinity; that the rectangular sampler returned infinities for limits whose width no float holds; and that a
Fraction with parts of a fixed width could not be the key of a kept quantile.

The references of these tests are their own: logarithms, exponentials and sums of Poisson and binomial terms in
decimals, from the numbers that the inputs are; the Cauchy quantile in closed form; the draws of Python's own
generator; and counting.
"""
import contextlib
import io
import math
import random
import unittest
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from statistics import NormalDist

from examples import ch05_propagation as chapter5
from examples import ch08_sampling as sampling
from examples import ch17_accuracy as accuracy
from examples import ch20_judge as judge
from examples import ch22_record_check as check
from examples import ch23_case as case
from examples import ch23_mix as mix
from swmetro import dist, stats
from swmetro.propagate import monte_carlo, rect, summarize
from swmetro.record import guard

try:
    import numpy as np
except ImportError:                                                       # the package itself needs no NumPy
    np = None

ROOT = Path(__file__).resolve().parent.parent
D = Decimal
F = Fraction
INF = math.inf
NAN = math.nan
FLAGS = (False, True)


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
    except (ValueError, TypeError) as error:
        return f"{type(error).__name__}: {error}"
    return None


def kind(function, *args, **kwargs):
    text = refusal(function, *args, **kwargs)
    return text and text.split(":")[0]


def dec(x):
    """A float, a whole number, a fraction or a decimal as a decimal, in the precision of the context."""
    x = Fraction(x)
    return D(x.numerator) / D(x.denominator)


def down(x, places):
    """x rounded toward -inf to so many places, as text, by the decimal module."""
    with localcontext() as ctx:
        ctx.prec = 400
        return str(dec(x).quantize(D(1).scaleb(-places), rounding=ROUND_FLOOR))


def up(x, places):
    with localcontext() as ctx:
        ctx.prec = 400
        return str(dec(x).quantize(D(1).scaleb(-places), rounding=ROUND_CEILING))


def halved(below, lo, hi):
    """The point at which `below` turns from true to false between lo and hi, in decimals, to 45 digits."""
    lo, hi = D(lo), D(hi)
    while hi - lo > hi * D("1e-45"):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if below(mid) else (lo, mid)
    return lo, hi


def at_most(n, mean):
    """P(X <= n) for a Poisson count, from its terms in decimals."""
    term = total = (-mean).exp()
    for k in range(1, n + 1):
        term = term * mean / k
        total += term
    return total


def poisson_limit(n, tail, lower=False):
    """The expected count at which a count of at most n has the probability `tail`, or, for a lower limit, a count
    of n or more: the limit of Garwood's interval, as a pair of decimals that enclose it."""
    tail = dec(tail)
    if lower:
        return halved(lambda m: 1 - at_most(n - 1, m) < tail, 0, 10 * n + 60)
    return halved(lambda m: at_most(n, m) > tail, 0, 10 * n + 60)


def silence(exposure, level):
    """-ln(1 - level) / exposure in decimals with 420 digits, from the numbers that the two are: the
    complement of the least float above zero differs from 1 in its 324th place, and the bound for it from that
    float in its 324th digit."""
    rest = 1 - Fraction(level)
    with localcontext() as ctx:
        ctx.prec = 420
        return -(D(rest.numerator) / D(rest.denominator)).ln() / dec(exposure)


def binomial_at_most(k, n, p):
    """P(X <= k) of n trials at p, from its terms in decimals."""
    q = 1 - p
    return sum(math.comb(n, i) * p ** i * q ** (n - i) for i in range(k + 1))


def wilson_limits(k, n):
    """The score interval at 95 %, from its two roots, in decimals."""
    z = dec(NormalDist().inv_cdf(0.975))
    p = D(k) / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * (p * (1 - p) / n + z * z / (4 * n * n)).sqrt() / (1 + z * z / n)
    return centre - half, centre + half


class Wrap8(int):
    """A whole number that is no int of Python's own for the Fraction that holds it, as an integer of NumPy is
    not: the hash of such a Fraction stops, at the modular inverse of its denominator under Python 3.11 and
    at the key of its two parts under the later ones."""

    def __hash__(self):
        raise TypeError("unhashable type: 'Wrap8'")

    def __pow__(self, other, modulus=None):
        if modulus is not None:
            raise TypeError("unsupported operand type(s) for pow()")
        return int(self) ** other

    __eq__ = int.__eq__


def part(n, d):
    """A Fraction whose two parts are of a fixed width, as Fraction(numpy.int8(n), numpy.int8(d)) is."""
    f = Fraction(n, d)
    f._numerator, f._denominator = Wrap8(f.numerator), Wrap8(f.denominator)
    return f


class Precision(unittest.TestCase):
    """Every test of this file computes its references with 60 digits."""

    def run(self, result=None):
        with localcontext() as ctx:
            ctx.prec = 60
            return super().run(result)


class OneConventionForABound(Precision):
    """upward(), downward() and enclosed(): what is printed holds what was computed."""

    def test_outward_for_both_signs(self):
        rng = random.Random(6901)
        for _ in range(2000):
            a = Fraction(rng.randint(-10 ** 9, 10 ** 9), rng.randint(1, 10 ** 6))
            b = a + Fraction(rng.randint(0, 10 ** 6), rng.randint(1, 10 ** 6))
            places = rng.randint(0, 6)
            low, top = stats.enclosed(a, b, places)
            self.assertEqual((low, top), (down(a, places), up(b, places)), (a, b, places))
            step = Fraction(1, 10 ** places)
            self.assertTrue(a - step < Fraction(low) <= a and b <= Fraction(top) < b + step)
        self.assertEqual(stats.enclosed(-5.932957, 5.638026, 1), ("-6.0", "5.7"))
        self.assertEqual(stats.enclosed(F(-1, 3), F(1, 3), 2), ("-0.34", "0.34"))
        self.assertEqual(stats.enclosed(0, 0, 2), ("0.00", "0.00"))             # a zero stays a zero
        self.assertEqual(stats.enclosed(D("1.50"), D("1.50"), 1), ("1.5", "1.5"))

    def test_a_bound_is_not_printed_across_a_limit(self):
        # 94.9934 % of an objective of 95 %: to two places upward it would read 95.00 %.
        self.assertEqual(stats.upward(F(949934, 10000), 2), "95.00")
        self.assertEqual(stats.upward(F(949934, 10000), 2, 95), "94.994")
        self.assertEqual(stats.upward(F(9499999, 100000), 2, 95), "94.99999")
        self.assertEqual(stats.upward(95, 2, 95), "95.00")                      # at the limit it is the limit
        self.assertEqual(stats.upward(F(950001, 10000), 2, 95), "95.01")        # beyond it nothing is added
        self.assertEqual(stats.downward(F(2000001, 10000), 2), "200.00")
        self.assertEqual(stats.downward(F(2000001, 10000), 2, 200), "200.0001")
        self.assertEqual(stats.downward(F(-2000001, 10000), 2, -201), "-200.01")
        self.assertEqual(stats.enclosed(D("199.9991"), D("199.9996"), 2, 200), ("199.99", "199.9996"))
        rng = random.Random(6902)
        for _ in range(500):
            limit = Fraction(rng.randint(-10 ** 4, 10 ** 4), 100)
            x = limit - Fraction(1, rng.randint(1, 10 ** 9))
            text = stats.upward(x, 2, limit)
            self.assertTrue(x <= Fraction(text) < limit, (x, limit, text))
            fewer = len(text.split(".")[1]) - 1
            if fewer >= 2:                                                      # one place fewer would reach it
                self.assertGreaterEqual(Fraction(up(x, fewer)), limit)
            y = limit + Fraction(1, rng.randint(1, 10 ** 9))
            self.assertTrue(limit < Fraction(stats.downward(y, 2, limit)) <= y)

    def test_what_is_refused(self):
        for flag in FLAGS:
            self.assertEqual(kind(stats.upward, flag, 2), "TypeError")
            self.assertEqual(kind(stats.upward, 1.5, 2, flag), "TypeError")
            self.assertEqual(kind(stats.enclosed, 1, 2, flag), "ValueError")    # a flag is no number of places
        self.assertEqual(kind(stats.upward, "1.5", 2), "TypeError")
        self.assertIn("holds the one that was computed", " ".join(stats.enclosed.__doc__.split()))


class TheRangeOfFigure22(Precision):
    """Section 1 of the review: the estimate plus its guard band, from a record rounded to two digits."""

    def setUp(self):
        self.rec = check.load("measurement_record.json")
        self.shown = check.displayed(self.rec)

    def ends(self, value, rec=None):
        change = {("result", "value"): value}
        return guard(check.altered(self.shown if rec is None else rec, change))[0]

    def test_the_two_ranges_and_the_point(self):
        lo, hi = self.ends(194.6)
        self.assertEqual((str(lo)[:16], str(hi)[:16]), ("199.865066223853", "200.110939345699"))   # as the review has them
        self.assertTrue(lo < D("200.1105") < hi and D("200.1105") > D("200.11"))   # in the range, above the old label
        text = printed(check.near, "rounded", self.shown, 194.6).splitlines()[-1]
        self.assertEqual(text, "  estimate + band: 199.86 to 200.12")           # Version 69: 199.87 to 200.11
        low, top = (D(t) for t in text.split(": ")[1].split(" to "))
        self.assertTrue(low <= lo and hi <= top and lo - low < D("0.01") and top - hi < D("0.01"))
        lo, hi = self.ends(194.8)
        text = printed(check.near, "rounded", self.shown, 194.8).splitlines()[-1]
        self.assertEqual(text, "  estimate + band: 200.06 to 200.32")           # Version 69: 200.07 to 200.31
        self.assertTrue(D("200.06") <= lo and hi <= D("200.32") and lo > 200)
        lo, hi = self.ends(194.6, self.rec)
        self.assertEqual(lo, hi)                                                # working values: one number
        self.assertEqual(f"{lo:.12f}", "199.949761190600")
        self.assertEqual(printed(check.near, "working", self.rec, 194.6).splitlines()[-1], "  estimate + band: 199.95")
        lines = expected("ch22_record_check")
        for line in ("  estimate + band: 199.86 to 200.12", "  estimate + band: 200.06 to 200.32",
                     "  estimate + band: 199.95"):
            self.assertIn(line + "\n", lines)

    def test_the_same_record_in_seconds(self):
        # Every number of the rounded record with its unit, divided by 1000. Version 69 printed "0.20" for it:
        # the range, 0.199865... to 0.200110... s, crosses the limit of 0.2 s as it does in milliseconds.
        changes = {("measurand", "unit"): "s"}
        for path in (("result", "value"), ("result", "u"), ("decision", "limit"), ("capture", "stored_step"),
                     ("result", "value_step")):
            node = self.shown
            for key in path:
                node = node[key]
            changes[path] = node / 1000 if isinstance(node, float) else D(str(node)).scaleb(-3)
        for i, c in enumerate(self.shown["components"]):
            changes[("components", i, "u")] = D(str(c["u"])).scaleb(-3)
        rec = check.altered(self.shown, changes)
        lo, hi = self.ends(D("0.1946"), rec)
        self.assertTrue(D("0.1998") < lo < D("0.2") < hi < D("0.2002"))
        text = printed(check.near, "in seconds", rec, D("0.1946")).splitlines()[-1].split(": ")[1]
        low, top = (D(t) for t in text.split(" to "))
        self.assertTrue(low <= lo and hi <= top, text)
        self.assertEqual(text, "0.19986 to 0.20012")                            # the second digit of their distance
        self.assertEqual((down(lo, 5), up(hi, 5)), ("0.19986", "0.20012"))

    def test_two_ends_that_differ_are_two_bounds(self):
        # Version 69 printed one number where the ends lay less than 0.005 apart. A narrow range comes from a
        # record with many digits: here the estimate is known to its fourth place and the rest as rounded.
        fine = check.altered(self.shown, {("result", "value_step"): 0.0})
        lo, hi = guard(check.altered(fine, {("result", "value"): 194.6}))[0]
        self.assertLess(lo, hi)
        text = printed(check.near, "fine", fine, 194.6).splitlines()[-1].split(": ")[1]
        self.assertIn(" to ", text)
        low, top = (D(t) for t in text.split(" to "))
        self.assertTrue(low <= lo < hi <= top)

    def test_the_figure_is_labeled_with_the_bounds(self):
        source = (ROOT / "figures" / "make_figures.py").read_text(encoding="utf-8")
        self.assertIn("labels = downward(ends[0], 2), upward(ends[1], 2)", source)
        self.assertIn("lo, hi = map(float, ends)", source)                      # the marks stay where they were computed
        self.assertNotIn('ax.text(x, y + 0.3, f"{x:.2f}"', source)
        self.assertIn("two bounds, however close", " ".join(check.near.__doc__.split()))


class PrintedRateLimits(Precision):
    """Section 2 of the review: the exact interval of a count, and arrival() of Chapter 23."""

    def test_one_message_in_thirty_seconds(self):
        # Garwood's limits for a count of 1: 1 - exp(-m) = 0.025 below, exp(-m) (1 + m) = 0.025 above.
        low = -(1 - D("0.025")).ln() / 30
        top = poisson_limit(1, D("0.025"))[1] / 30
        self.assertEqual((f"{low:.13f}", f"{top:.13f}"), ("0.0008439269328", "0.1857214463646"))   # the review's
        self.assertEqual(poisson_limit(1, D("0.025"), lower=True)[0].quantize(D("1e-30")),
                         (-(1 - D("0.025")).ln()).quantize(D("1e-30")))        # the search against the closed form
        got = stats.rate_limits(1, 30)
        self.assertTrue(dec(got[0]) <= low and top <= dec(got[1]))
        self.assertEqual(stats.enclosed(*got, 5), ("0.00084", "0.18573"))       # Version 69 printed 0.18572
        self.assertEqual((down(low, 5), up(top, 5)), ("0.00084", "0.18573"))
        self.assertIn("  exact, 95 %: 0.00084 to 0.18573\n", expected("ch16_silence"))
        self.assertIn("  normal, 95 %: -0.03200 to 0.09867\n", expected("ch16_silence"))
        # The normal interval: 1/30 less and plus 1.959964 / 30.
        z = dec(NormalDist().inv_cdf(0.975))
        self.assertEqual((down((1 - z) / 30, 5), up((1 + z) / 30, 5)), ("-0.03200", "0.09867"))

    def test_three_requests_in_a_hundred_seconds(self):
        low = poisson_limit(3, D("0.025"), lower=True)[0] / 100
        top = poisson_limit(3, D("0.025"))[1] / 100
        self.assertEqual((f"{low:.10f}", f"{top:.10f}"), ("0.0061867212", "0.0876727307"))        # the review's
        self.assertEqual(case.arrival(3, 100), "0.00618 to 0.08768 req/s at 95 %")   # Version 69: 0.00619 to 0.08767
        self.assertEqual((down(low, 5), up(top, 5)), ("0.00618", "0.08768"))
        self.assertEqual(case.arrival(1, 100), "0.00025 to 0.05572 req/s at 95 %")
        self.assertIn("  one request: 0.00025 to 0.05572 req/s at 95 %\n", expected("ch23_case"))
        low, top = poisson_limit(1, D("0.025"), lower=True)[0] / 100, poisson_limit(1, D("0.025"))[1] / 100
        self.assertEqual((down(low, 5), up(top, 5)), ("0.00025", "0.05572"))

    def test_the_exposure_of_an_arrival_rate(self):
        want = case.arrival(1, 30)
        self.assertEqual(want, "0.00084 to 0.18573 req/s at 95 %")
        for seconds in (D("30"), F(30), 30.0):                                  # Version 69: TypeError for the decimal
            self.assertEqual(case.arrival(1, seconds), want, repr(seconds))
        self.assertEqual(case.arrival(100, 1), "100(10) req/s")
        for seconds in (D("0.5"), F(1, 2), 0.5):
            self.assertEqual(case.arrival(100, seconds), "200(20) req/s", repr(seconds))
        for flag in FLAGS:
            for n in (0, 1, 100):                                               # Version 69: "100(10) req/s" for True
                self.assertEqual(kind(case.arrival, n, flag), "TypeError", (n, flag))
        for bad in (0, -30, INF, NAN, D("0")):
            for n in (0, 1, 100):                                               # before a branch is chosen
                self.assertEqual(refusal(case.arrival, n, bad), "ValueError: no exposure", (n, bad))
        self.assertIn("arrival rate before 19.05(44) req/s\n", expected("ch23_case"))

    def test_limits_on_their_outer_sides(self):
        # At an upper limit a count this low has at most its nominal probability, and just inside the limit more.
        inside = 1 - D("3e-12")
        for n in (1, 2, 3, 7, 20, 60, 150):
            for level in (0.5, 0.9, 0.95, 0.99, 0.999, F(1, 4), D("0.05")):
                for exposure in (1, 30, 0.75, F(1, 3)):
                    tail = 1 - dec(level)
                    top = dec(stats.rate_upper(n, exposure, level)) * dec(exposure)
                    self.assertLessEqual(at_most(n, top), tail, (n, level, exposure))
                    self.assertGreater(at_most(n, top * inside), tail, (n, level, exposure))
                    low, high = (dec(x) * dec(exposure) for x in stats.rate_limits(n, exposure, level))
                    self.assertLessEqual(at_most(n, high), tail / 2)
                    self.assertGreater(at_most(n, high * inside), tail / 2)
                    self.assertLessEqual(1 - at_most(n - 1, low), tail / 2)
                    self.assertGreater(1 - at_most(n - 1, low / inside), tail / 2)
        self.assertEqual(stats.rate_limits(0, 30)[0], 0.0)                      # a zero stays a zero
        self.assertIs(type(stats.rate_limits(0, 30)[0]), float)
        self.assertIn("not above the limit of the model", " ".join(stats.rate_limits.__doc__.split()))
        self.assertIn("1e-12 of itself", " ".join(stats._mean_at.__doc__.split()))

    def test_the_interval_of_the_case(self):
        # The change of the handler percentile with its standard uncertainty: value less and plus 1.959964 u.
        rng = random.Random(23)
        before, after = case.period(rng), case.period(rng)
        d = case.handler_change(before, after)
        z = dec(NormalDist().inv_cdf(0.975))
        low, top = dec(d.value) - z * dec(d.u), dec(d.value) + z * dec(d.u)
        self.assertEqual((str(low)[:9], str(top)[:8]), ("-5.932957", "5.638026"))   # the review's, cut off
        self.assertEqual((down(low, 1), up(top, 1)), ("-6.0", "5.7"))
        self.assertIn("interval, 95 %: [-6.0, 5.7] ms\n", expected("ch23_case"))   # Version 69: [-5.9, 5.6]
        self.assertIn("upper bound, 95 %: 4.8 ms\n", expected("ch23_case"))


class TheBoundForNoEvent(Precision):
    """Section 5 of the review, second part: quiet_bound() before anything is printed."""

    def least_float_not_below(self, got, reference):
        self.assertIs(type(got), float)
        self.assertGreaterEqual(Fraction(got), Fraction(reference))            # compared as the numbers they are
        self.assertLess(Fraction(math.nextafter(got, 0.0)), Fraction(reference))

    def test_the_three_cases_of_the_review(self):
        # 1. A level of 1 - 1e-20 is 1 as a float: the bound is ln(1e20) / 10.
        for level in (1 - D("1e-20"), 1 - F(1, 10 ** 20)):
            reference = D(10) ** 20
            reference = reference.ln() / 10
            self.assertEqual(f"{reference:.12f}", "4.605170185988")
            self.assertEqual(float(level), 1.0)
            for got in (stats.quiet_bound(10, level), stats.rate_upper(0, 10, level)):   # Version 69: math domain error
                self.least_float_not_below(got, reference)
        # One event at that level: exp(-m) (1 + m) = 1e-20.
        one = poisson_limit(1, D("1e-20"))[1] / 10
        self.assertEqual(f"{one:.7f}", "4.9983198")
        got = stats.rate_upper(1, 10, 1 - F(1, 10 ** 20))
        self.assertTrue(one <= dec(got) < one * (1 + D("3e-12")))
        # 2. An exposure of 1e308 s at a level of 1e-16: the bound is positive, and below every float.
        reference = silence(1e308, 1e-16)
        self.assertTrue(0 < reference < dec(5e-324))
        got = stats.quiet_bound(1e308, 1e-16)                                   # Version 69: 0.0
        self.assertEqual(got, 5e-324)
        self.least_float_not_below(got, reference)
        self.assertEqual(stats.upward(got, 5), "0.00001")                       # Version 69: 0.00000
        # 3. The float next to ln 2 as the exposure, at a level of one half.
        t = 0.6931471805599453
        reference = silence(t, 0.5)
        self.assertEqual(str(reference)[:25], "1.00000000000000003345677")       # the review's, cut off
        got = stats.quiet_bound(t, 0.5)                                         # Version 69: 1.0
        self.assertEqual(got, math.nextafter(1.0, 2.0))
        self.least_float_not_below(got, reference)
        self.assertEqual(stats.upward(got, 5), "1.00001")                       # Version 69: 1.00000
        self.assertEqual(up(reference, 5), "1.00001")

    def test_both_neighbors_of_a_printed_step(self):
        # Exposures at which the bound lies next to a step of the fifth place, on either side of it.
        rng = random.Random(6903)
        later = 0
        log20 = D(20).ln()                                                      # -ln(0.05)
        for _ in range(300):
            step = D(rng.randint(1000, 99999)).scaleb(-5)
            t = float(log20 / step)
            for exposure in (math.nextafter(t, 0.0), t, math.nextafter(t, INF)):
                reference = silence(exposure, 0.95)                             # the float 0.95 as the number it is
                got = stats.quiet_bound(exposure)
                self.least_float_not_below(got, reference)
                # What is printed is not below the bound. It is the least number of five places with that
                # property unless such a number lies between the bound and the float next above it: a float
                # is rounded upward once more where it is printed.
                shown = D(stats.upward(got, 5))
                self.assertGreaterEqual(shown, reference)
                least = D(up(reference, 5))
                self.assertIn(shown - least, (0, D("0.00001")), exposure)
                if shown != least:
                    later += 1
                    self.assertTrue(reference <= least < dec(got))
        self.assertLess(later, 150)                                             # of 900: next to a step by design
        for level in (0.5, 0.9, 0.975, 1e-3, 1e-9, 5e-324, math.nextafter(1.0, 0.0), D("0.999999"), F(2, 3)):
            for exposure in (1, 30, 0.1, D("0.3"), F(1, 7), 1e-300, 1e300):
                self.least_float_not_below(stats.quiet_bound(exposure, level), silence(exposure, level))
        self.assertEqual(stats.upward(stats.quiet_bound(30), 5), "0.09986")
        self.assertEqual(stats.upward(stats.quiet_bound(10), 5), "0.29958")

    def test_the_bounds_of_the_table_of_chapter_16(self):
        # No message in a whole second, in the verified time by the stamps and of the events, and two messages
        # in the whole second: upward to three places, as the table of Chapter 16 prints them since Version 70.
        for exposure, want in ((1, "2.996"), (0.65, "4.609"), (0.75, "3.995"), (0.648, "4.624"), (0.749, "4.000")):
            self.assertEqual(up(silence(exposure, 0.95), 3), want)
            self.assertEqual(stats.upward(stats.quiet_bound(exposure), 3), want)
        two = poisson_limit(2, 1 - dec(0.95))[1]
        self.assertEqual(up(two, 3), "6.296")
        self.assertEqual(stats.upward(stats.rate_upper(2, 1.0), 3), "6.296")

    def test_the_docstrings_state_the_contract(self):
        text = " ".join(stats.quiet_bound.__doc__.split())
        self.assertIn("formed to forty digits", text)
        self.assertIn("is not below it", text)
        self.assertIn("the least float above zero", " ".join(stats._rate.__doc__.split()))
        self.assertIn("as a float, 1 - 1e-20 is 1", " ".join(stats._silence.__doc__.split()))


class ANaNBeforeARank(Precision):
    """Section 3 of the review: the same twenty rows in two orders."""

    XS = [200.0] * 18 + [400.0]

    def test_two_orders_of_the_same_rows(self):
        first = sorted([NAN] + self.XS)
        last = sorted(self.XS + [NAN])
        self.assertTrue(first[0] != first[0] and last[-1] != last[-1])          # the sort leaves it where it stood
        # Version 69: 200 for the one and 400 for the other.
        for rows in (first, last, [D("NaN")] + self.XS, self.XS + [D("NaN")]):
            for helper in (stats.ecdf_quantile, stats.order_stat, stats.quantile, stats.quantile_interval):
                self.assertEqual(refusal(helper, rows, 0.95), "ValueError: a NaN has no rank", helper.__name__)
        self.assertEqual(stats.ecdf_quantile(self.XS, 0.95), 400)               # the nineteen: the 19th of them

    def test_a_nan_that_no_rank_selects(self):
        # The median of five is the third: the NaN in the last place is never looked up.
        for nan in (NAN, D("NaN"), D("sNaN")):
            rows = [1, 2, 3, 4, nan]
            self.assertEqual(refusal(stats.ecdf_quantile, rows, 0.5), "ValueError: a NaN has no rank")
            self.assertEqual(refusal(stats.quantile, rows, 0.5), "ValueError: a NaN has no rank")
            self.assertEqual(refusal(stats.ordered, rows), "ValueError: a NaN has no rank")
        self.assertEqual(kind(summarize, [1.0] * 40 + [NAN]), "ValueError")
        self.assertEqual(stats.ecdf_quantile([1, 2, 3, 4, 5], 0.5), 3)

    def test_an_infinity_is_a_value(self):
        rows = self.XS + [INF]
        self.assertEqual(stats.ecdf_quantile(rows, 0.95), 400)                  # the 19th of twenty
        self.assertEqual(stats.ecdf_quantile(rows, 1), INF)
        self.assertEqual(stats.quantile([100, INF], 0), 100)
        clean = [1, 2.5, D("0.1")]
        self.assertIs(stats.ordered(clean), clean)                              # nothing to convert: the same list
        self.assertEqual(stats.mean([1.0, NAN]) != stats.mean([1.0, NAN]), True)   # a sum has no ranks: NaN

    def test_the_twentieth_request_of_the_review(self):
        # Eighteen requests of 100 ms and one of 3000 ms, a limit of 2000 ms and an objective of 95 %.
        rest = [100] * 18 + [3000]
        for last, share, verdict in (((">=", 1000), (F(18, 20), F(19, 20)), "unresolved"),
                                     (None, (F(18, 20), F(19, 20)), "unresolved"),
                                     (NAN, (F(18, 20), F(19, 20)), "unresolved"),
                                     (1200, (F(19, 20), F(19, 20)), "met"),
                                     (INF, (F(18, 20), F(18, 20)), "missed")):
            rec = {"web": rest + [last]}
            got = mix.within(rec, rec, 2000)
            self.assertEqual(got, share, last)
            self.assertEqual(mix.verdict(*got), verdict, last)
        within = sum(x <= 2000 for x in rest)
        self.assertEqual((within, within + 1), (18, 19))                        # by counting

    def test_the_docstrings_say_where(self):
        text = " ".join(stats.ordered.__doc__.split())
        self.assertIn("also where no rank would select it", text)
        self.assertIn("An infinity is a value like any other", text)


class TheSameMixWithAnOpenDuration(Precision):
    """Section 4 of the review, first part: the change of the mix is a difference of two means."""

    @staticmethod
    def parts(stat, before, after):
        """For one class in both periods: the change within the class and the change of the mix in the two
        orders, by subtracting; None where an infinity is taken from an infinity."""
        def less(a, b):
            return None if a == b == INF else a - b
        own_before, own_after = stat(before), stat(after)
        first = (less(own_after, own_before), less(own_after, own_after))        # class values first: via after
        second = (less(own_after, own_before), less(own_before, own_before))     # mix first: via before
        return first, second

    def test_the_table_of_the_review(self):
        mean = lambda xs: sum(xs) / len(xs)                                     # noqa: E731
        p95 = lambda xs: sorted(xs)[-(-95 * len(xs) // 100) - 1]                # noqa: E731
        for stat in (mean, p95):
            first, second = self.parts(stat, [100], [INF])
            self.assertEqual(first, (INF, None))                                # inf less inf
            self.assertEqual(second, (INF, 0))
        for args in ((), (mix.p95, "p95", "distributions")):
            lines = printed(mix.components, {"web": [100]}, {"web": [INF]}, *args).splitlines()
            self.assertEqual(lines[2], "    within +inf, mix not defined")      # Version 69: "within +inf, mix +0"
            self.assertEqual(lines[4], "    within +inf, mix +0")
            self.assertEqual(lines[6], "    within +inf, mix not defined")
            self.assertNotIn("the mix is the same", "\n".join(lines))

    def test_the_lost_duration_of_the_transcript(self):
        # Nineteen of 100 ms and one of x: the mean after is (1900 + x) / 20 at either mix.
        for x in (0, 100, 10 ** 6):
            first, second = self.parts(lambda xs: Fraction(sum(xs), len(xs)), [100] * 20, [100] * 19 + [x])
            self.assertEqual((first[1], second[1]), (0, 0))
            self.assertEqual(first[0], Fraction(1900 + x, 20) - 100)
        first, second = self.parts(lambda xs: sum(xs) / len(xs), [100] * 20, [100] * 19 + [INF])
        self.assertEqual((first[1], second[1]), (None, 0))                      # the first order loses its zero
        lines = expected("ch23_mix").splitlines()[-18:]
        self.assertEqual(lines[:10], ["20 web requests of 100 ms in each",
                                      "period, one duration lost after:",
                                      "mean, 100 -> 95 ms or more: -5 ms or more",
                                      "  open after: 1 of 20 durations",
                                      "  class means first, via 95 ms or more:",
                                      "    within -5 or more, mix +0 if defined",   # Version 69: the two lines of
                                      "  mix first, via 100 ms:",                   # "the mix is the same"
                                      "    within -5 or more, mix +0",
                                      "  average of the two orders:",
                                      "    within -5 or more, mix +0 if defined"])
        # The percentile is the 19th of twenty, 100 ms wherever the lost one stands: its zeros are settled.
        for x in (0, 100, INF):
            self.assertEqual(sorted([100] * 19 + [x])[18], 100)
        self.assertEqual(lines[10:], ["p95, 100 -> 100 ms: +0 ms",
                                      "  open after: 1 of 20 durations",
                                      "  class distributions first, via 100 ms:",
                                      "    within +0, mix +0",
                                      "  mix first, via 100 ms:",
                                      "    within +0, mix +0",
                                      "  average of the two orders:",
                                      "    within +0, mix +0"])

    def test_finite_durations_keep_their_zero(self):
        for before, after in (([100] * 4, [120] * 4), ([100, 200], [300, 100]), ([50], [50])):
            for args in ((), (mix.p95, "p95", "distributions")):
                text = printed(mix.components, {"web": before}, {"web": after}, *args)
                self.assertEqual(text.count(", mix +0\n"), 3, text)
                self.assertNotIn("defined", text)

    def test_two_classes_that_keep_their_shares(self):
        # Half web and half cache in both periods, and one cache duration lost: with x for it the mean after is
        # (100 + 100 + 10 + x) / 4, and the change of the mix is zero for every x that is a number.
        before = {"web": [100, 100], "cache": [10, 10]}
        after = {"web": [100, 100], "cache": [10, None]}
        lines = printed(mix.components, before, after).splitlines()
        self.assertEqual(lines[0], "mean, 55 -> 52.5 ms or more: -2.5 ms or more")
        self.assertEqual(Fraction(100 + 100 + 10 + 0, 4) - 55, Fraction(-5, 2))
        self.assertEqual(lines[-1], "    within -2.5 or more, mix +0 if defined")
        self.assertEqual(lines[4:6], ["  mix first, via 55 ms:", "    within -2.5 or more, mix +0"])


class AnAttainableShare(Precision):
    """Section 5 of the review, first part: 94.93 % was below a share that the chapter itself attains."""

    def test_the_cap_of_800_ms(self):
        # 525 slow requests of 200 ms and 375 of 800 ms: 900 requests with the mean of 450 ms, none above 800.
        self.assertEqual(Fraction(525 * 200 + 375 * 800, 900), 450)
        fast = Fraction(891, 1000)                                              # 99 of 100 fast, nine tenths
        top = fast + Fraction(1, 10) * Fraction(525, 900)
        self.assertEqual(100 * top, Fraction(1424, 15))
        self.assertGreater(100 * top, Fraction("94.93"))                        # what Version 69 printed as the most
        self.assertEqual(up(100 * top, 2), "94.94")
        self.assertEqual(mix.up_to(top), "94.94 %")
        self.assertEqual(mix.most_within(900, 450, 200, 800), 525)
        # 531 of them within the limit reach the objective exactly.
        top = fast + Fraction(1, 10) * Fraction(531, 900)
        self.assertEqual(100 * top, 95)
        self.assertEqual(mix.up_to(top), "95.00 %")
        self.assertEqual(mix.most_within(900, 450, 200, 810), 531)
        lines = expected("ch23_mix")
        for line in ("  within 200 ms: 89.1 % to 99.1 %", "    to 94.94 %, missed", "    to 95.00 %, unresolved"):
            self.assertIn(line + "\n", lines)
        self.assertNotIn("94.93", lines)

    def test_a_share_is_not_printed_as_the_objective_it_misses(self):
        top = Fraction(949934, 1000000)
        self.assertEqual(mix.verdict(Fraction(89, 100), top), "missed")
        self.assertEqual(mix.up_to(top), "94.994 %")
        self.assertEqual(mix.between(Fraction(891, 1000), Fraction(991, 1000), 1), "89.1 % to 99.1 %")
        self.assertEqual(mix.between(Fraction(18, 20), Fraction(19, 20)), "90 % to 95 %")
        self.assertEqual(mix.between(Fraction(9501, 10000), Fraction(9999, 10000)), "95.01 % to 100 %")
        # Every share that the transcript prints with a verdict "missed" is printed below 95.
        for line in expected("ch23_mix").splitlines():
            if line.endswith("missed") and " to " in line and "%" in line:
                self.assertLess(float(line.split("to ")[-1].split(" %")[0]), 95, line)


class OtherPrintedIntervals(Precision):
    """Section 5 of the review, the table: the intervals of other chapters enclose what was computed."""

    def test_the_rate_of_chapter_5(self):
        # T uniform on [1.2, 2.8] s: the central 95 % of 1000 / T is 1000 / 2.76 to 1000 / 1.24.
        low, top = Fraction(1000 * 100, 276), Fraction(1000 * 100, 124)
        self.assertEqual((down(low, 2), up(top, 2)), ("362.31", "806.46"))
        self.assertEqual((f"{float(low):.2f}", f"{float(top):.2f}"), ("362.32", "806.45"))   # to the nearest
        text = expected("ch05_propagation")
        self.assertIn("exact 95 %  [362.31, 806.46]\n", text)
        self.assertIn("MC 95 %  [362, 807]\n", text)
        # The simulation: its two draws at the ranks of the Supplement, in whole numbers on their outer sides.
        draws = [lambda rng: 1000, rect(1.2, 2.8)]
        ys = monte_carlo(chapter5.rate, draws, chapter5.N, 5)
        _, _, lo, hi = summarize(ys)
        self.assertEqual((lo, hi), (ys[4999], ys[194999]))
        self.assertEqual((math.floor(lo), math.ceil(hi)), (362, 807))

    def test_the_accuracy_of_chapter_17(self):
        rows = []
        for n in (500, 2000, 10_000):
            low, top = wilson_limits(round(0.87 * n), n)
            rows.append(f"[{down(100 * low, 1)}, {up(100 * top, 1)}]")
        low, top = wilson_limits(1740, 2000)
        self.assertEqual((str(100 * low)[:9], str(100 * top)[:10]), ("85.454887", "88.4032509"))   # the review's
        self.assertEqual(rows, ["[83.7, 89.7]", "[85.4, 88.5]", "[86.3, 87.7]"])
        lines = expected("ch17_accuracy").splitlines()
        self.assertEqual([line.split(": ")[1].split(" %")[0] for line in lines[1:4]], rows)
        # The same for the profiler of Chapter 8 and the share of Chapter 22: what is printed holds the limits.
        rng = random.Random(8)                                                  # the seed of the listing
        for line in expected("ch08_sampling").splitlines()[1:5]:
            n = int(line.split(":")[0].split("=")[1])
            k = sampling.profile(0.07, n, rng)
            self.assertEqual(f"{100 * k / n:5.2f} %", line.split(": ")[1].split("  ")[0])
            low, top = wilson_limits(k, n)
            ends = line.split("[")[1].rstrip("]").split(",")
            self.assertEqual([e.strip() for e in ends], [down(100 * low, 2), up(100 * top, 2)], line)
        self.assertIn("share <= 200 ms: 0.978, 95 % [0.970, 0.984]\n", expected("ch22_budget"))
        low, top = wilson_limits(1956, 2000)
        self.assertEqual((down(low, 3), up(top, 3)), ("0.970", "0.984"))

    def test_the_ranges_of_chapter_17(self):
        # The least and the greatest probability of a record, each on its outer side.
        self.assertEqual(accuracy.ends(Fraction(502294078, 10 ** 10), Fraction(538538761, 10 ** 10)),
                         "0.050229 to 0.053854")
        self.assertEqual(accuracy.ends(Fraction(438283509, 10 ** 10), Fraction(502294078, 10 ** 10)),
                         "0.043828 to 0.050230")
        self.assertEqual(accuracy.shown(Fraction(1776357, 10 ** 21), math.floor), "1.77e-15")
        self.assertEqual(accuracy.shown(Fraction(1776357, 10 ** 21), math.ceil), "1.78e-15")
        self.assertEqual(accuracy.shown(Fraction(9999, 10 ** 8), math.ceil), "1.00e-04")
        self.assertEqual(accuracy.shown(Fraction(1, 16), math.floor), "0.062500")   # exact: no side
        self.assertEqual(accuracy.shown(Fraction(1, 16), math.ceil), "0.062500")
        self.assertEqual(accuracy.shown(Fraction(1, 3)), "0.333333")            # a value: to the nearest
        for line in expected("ch17_accuracy").splitlines()[24:28]:
            b = int(line.split(":")[0].split("=")[1])
            low, top = accuracy.limits(1722, b, 2000)
            a, c = line.split(": ")[1].split(" to ")
            self.assertTrue(Fraction(a) <= low and top <= Fraction(c), line)
            self.assertLess(Fraction(c) - top, Fraction(1, 10 ** 6), line)      # and by less than its last place

    def test_the_judge_of_chapter_20(self):
        # No pass in 25: the upper limit is 1 - 0.025^(1/25). Forty in 40: the lower one is 0.025^(1/40).
        top = 1 - (D("0.025").ln() / 25).exp()
        low = (D("0.025").ln() / 40).exp()
        self.assertEqual((up(top, 3), down(low, 3)), ("0.138", "0.911"))        # Version 69 printed 0.137 and 0.912
        self.assertEqual((f"{top:.3f}", f"{low:.3f}"), ("0.137", "0.912"))
        # Sixty of 100 after the verdicts are exchanged, with sensitivity 0.90 and specificity 0.85.
        lo, _ = halved(lambda p: 1 - binomial_at_most(59, 100, p) < D("0.025"), 0, 1)
        _, hi = halved(lambda p: binomial_at_most(60, 100, p) > D("0.025"), 0, 1)
        lo, hi = ((p - D("0.15")) / D("0.75") for p in (lo, hi))
        self.assertEqual((down(lo, 3), up(hi, 3)), ("0.462", "0.729"))          # Version 69 printed 0.463
        text = expected("ch20_judge")
        for line in ("  judged rate 0.000 to 0.138,", "  judged rate 0.911 to 1.000,", "  95 % interval 0.462 to 0.729",
                     "  95 % interval 0.000 to 0.223"):
            self.assertIn(line + "\n", text)
        # The model's own band stays as it is written: 0.15 and 0.90 are its numbers.
        self.assertEqual(text.count("  model, 0.15 to 0.90, at 95 %\n"), 2)
        # The conclusions stand: no overlap with the band from 0.15 to 0.90.
        self.assertTrue(top < D("0.15") and low > D("0.90"))

    def test_what_the_prose_states_as_bounds(self):
        # One wrap of a counter of 32 bits at 60 MB a second or less takes at least this long.
        self.assertEqual(down(Fraction(2 ** 32, 60 * 10 ** 6), 4), "71.5827")   # the text had 71.5828
        # Chapter 17: the greatest probabilities with B at 1672 and with six pairs wrong for both.
        self.assertEqual(up(accuracy.limits(1722, 1672, 2000)[1], 4), "0.0465")   # the text had 0.0464 at most
        ps = accuracy.partly((1560, 132, 163, 101), 1722, 1758, 2000)
        self.assertEqual(up(ps[-1], 4), "0.0496")                               # and 0.0495 at most
        ps = accuracy.partly((1560, 142, 163, 95), 1722, 1758, 2000)
        self.assertEqual(down(ps[0], 4), "0.0502")
        # The paired standard uncertainty that scores of 1722 and 1758 of 2000 allow, in points.
        def u(only_a, only_b):
            n = 2000
            return 100 * ((D(only_a + only_b) - D(only_b - only_a) ** 2 / n).sqrt() / n)
        self.assertEqual((down(u(0, 36), 2), up(u(242, 278), 2)), ("0.29", "1.14"))   # the text had 0.30
        # Chapter 16: 564 messages in 30 s, their exact limits and the symmetric interval.
        low, top = poisson_limit(564, D("0.025"), lower=True)[0] / 30, poisson_limit(564, D("0.025"))[1] / 30
        self.assertEqual((down(low, 2), up(top, 2)), ("17.28", "20.42"))
        z, rate, u = dec(NormalDist().inv_cdf(0.975)), D(564) / 30, D(564).sqrt() / 30
        self.assertEqual((down(rate - z * u, 2), up(rate + z * u, 2)), ("17.24", "20.36"))   # 17.25 to 20.35
        # Chapter 8: six of 100 samples.
        low, top = wilson_limits(6, 100)
        self.assertEqual((down(100 * low, 1), up(100 * top, 1)), ("2.7", "12.5"))             # the text had 2.8
        # Chapter 18: 99 positive labels of 100. The upper limit is 0.975 to the power 1/100.
        top = (D("0.975").ln() / 100).exp()
        low, _ = halved(lambda p: 1 - binomial_at_most(98, 100, p) < D("0.025"), 0, 1)
        self.assertEqual((down(low, 4), up(top, 4)), ("0.9455", "0.9998"))                    # the text had 0.9997
        self.assertEqual(tuple(f"{x:.6f}" for x in judge.exact_interval(99, 100)), ("0.945541", "0.999747"))


class LimitsWhoseWidthNoFloatHolds(Precision):
    """Section 6 of the review, first part: the rectangular sampler."""

    def test_the_call_of_the_review(self):
        rng = random.Random(1)
        draw = rect(-1e308, 1e308)
        values = [draw(rng) for _ in range(1000)]                               # Version 69: 1000 infinities
        self.assertTrue(all(-1e308 <= v <= 1e308 for v in values))
        self.assertEqual((f"{min(values):.4e}", f"{max(values):.4e}"), ("-9.9831e+307", "9.9632e+307"))
        rng = random.Random(1)
        own = [(1 - u) * -1e308 + u * 1e308 for u in (rng.random() for _ in range(1000))]
        self.assertEqual(values, own)
        self.assertEqual(1e308 - -1e308, INF)                                   # why Python's formula fails here

    def test_both_signs_and_both_ends(self):
        top = 1.7976931348623157e308
        # The width of the first four pairs is beyond the floats, and that of the last one is not.
        for lo, hi in ((-top, top), (-1e308, top), (-top, 1e300), (-1e300, top), (-top, 5e-324)):
            self.assertEqual(hi - lo == INF, hi > 0.1)
            draw = rect(lo, hi)

            class Fixed:
                def __init__(self, u):
                    self.u = u

                def random(self):
                    return self.u

                def uniform(self, a, b):                                        # Python's own formula
                    return a + (b - a) * self.u

            ends = [draw(Fixed(u)) for u in (0.0, 2.0 ** -53, 0.5, 1 - 2.0 ** -53)]
            self.assertEqual(ends[0], lo)
            self.assertTrue(all(lo <= v <= hi and v == v for v in ends), (lo, hi, ends))
            self.assertEqual(ends, sorted(ends))
            self.assertGreater(ends[-1], hi - abs(hi) * 1e-15 - abs(lo) * 1e-15)
            rng = random.Random(7)
            self.assertTrue(all(lo <= draw(rng) <= hi for _ in range(2000)))

    def test_ordinary_limits_draw_as_before(self):
        for lo, hi in ((1.2, 2.8), (-100, 100), (0, 1e308), (-1e308, 0), (-8e307, 8e307)):
            self.assertLess(float(hi) - float(lo), INF)
            a, b = random.Random(5), random.Random(5)
            draw = rect(lo, hi)
            self.assertEqual([draw(a) for _ in range(200)], [b.uniform(lo, hi) for _ in range(200)])
        self.assertIn("would return an infinity", " ".join(rect.__doc__.split()))


class AFractionAsAKey(Precision):
    """Section 6 of the review, second part: a quantile is kept under the numbers, not under their types."""

    def test_the_quarter_with_one_degree_of_freedom(self):
        # Cauchy: the upper tail is 1/4 at t = 1, and so is the probability between 0 and t.
        self.assertEqual(math.tan(math.pi / 4), 0.9999999999999999)
        with self.assertRaises(TypeError):
            hash(part(1, 4))                                                    # what stopped Version 69
        for order in ((part(1, 4), F(1, 4), 0.25, D("0.25")), (D("0.25"), 0.25, F(1, 4), part(1, 4))):
            dist._t_isf.cache_clear()
            dist._t_mid.cache_clear()
            got = [(dist.t_isf(q, 1), dist.t_mid(q, 1)) for q in order]
            self.assertEqual(len(set(got)), 1, got)
            for value in got[0]:
                self.assertLess(abs(value - 1), 1e-12)
        self.assertEqual(dist.t_isf(part(1, 8), 1), dist.t_isf(0.125, 1))
        self.assertAlmostEqual(dist.t_isf(part(1, 8), 1), 1 + 2 ** 0.5, places=12)   # tan(3 pi / 8)
        self.assertEqual(dist.t_within(part(1, 2), 1), dist.t_mid(0.25, 1))
        self.assertEqual(dist.t_ppf(part(3, 4), 1), dist.t_isf(0.25, 1))

    def test_a_flag_is_still_no_number(self):
        for search in (dist.t_isf, dist.t_mid):
            for order in ((1, True), (True, 1)):
                dist._t_isf.cache_clear()
                dist._t_mid.cache_clear()
                got = dict((type(dof), kind(search, 0.1, dof)) for dof in order)
                self.assertEqual(got, {bool: "TypeError", int: None}, order)
            self.assertEqual(kind(search, True, 4), "TypeError")
        self.assertEqual(dist.t_isf(0.1, 1), dist.t_isf(0.1, 1.0))
        self.assertIn("True is no key for the answer of 1", " ".join(dist.t_mid.__doc__.split()))
        self.assertIn("formed anew from ints", " ".join(dist._float.__doc__.split()))


@unittest.skipIf(np is None, "NumPy is not installed")
class WithTheTypesOfNumPy(Precision):
    """The calls of the review as it made them."""

    def test_the_fraction_of_the_review(self):
        # Whether hash() of this fraction stops depends on what Python has hashed before: since 3.12 it keeps
        # the hashes of fractions by their two parts, and an int8 of 1 is found under the int 1. The stand-in
        # of the test above stops always; here the two calls are made as the review made them.
        q = Fraction(np.int8(1), np.int8(4))
        dist._t_isf.cache_clear()
        dist._t_mid.cache_clear()
        for search in (dist.t_isf, dist.t_mid):                                 # Version 69: TypeError for both
            self.assertEqual(search(q, 1), search(Fraction(1, 4), 1))
            self.assertLess(abs(search(q, 1) - 1), 1e-12)

    def test_other_inputs_of_this_review(self):
        rows = [np.float64("nan")] + [200.0] * 18 + [400.0]
        self.assertEqual(refusal(stats.ecdf_quantile, rows, 0.95), "ValueError: a NaN has no rank")
        self.assertEqual(case.arrival(1, np.float32(30)), "0.00084 to 0.18573 req/s at 95 %")
        self.assertEqual(kind(case.arrival, 100, np.True_), "TypeError")
        rng = random.Random(1)
        draw = rect(np.float64(-1e308), np.float64(1e308))
        self.assertTrue(all(-1e308 <= draw(rng) <= 1e308 for _ in range(1000)))
        self.assertEqual(stats.upward(stats.quiet_bound(np.float64(0.6931471805599453), np.float64(0.5)), 5), "1.00001")
        self.assertEqual(stats.enclosed(np.float32(1.5), np.int8(2), 1), ("1.5", "2.0"))


if __name__ == "__main__":
    unittest.main()
