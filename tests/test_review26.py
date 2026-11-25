"""Regression tests added after the review of Version 26.

The review found that a time stamp of nineteen digits held with no
uncertainty moved by 100 ns when its interval was asked for, when it was
drawn from a sampler without a spread, and when it was averaged with its
own copies; that two fully correlated contributions of 100 000 000 ns and
100 000 001 ns of opposite sign combined to an uncertainty of zero, where
1 ns is left; and that the bound for a search through ten windows raised
an error at a level next to one. It asked for the search for a gap of 30 s
anywhere in a record, which is another search than that of ten windows on
a fixed grid, with a figure.

Each test computes what it expects independently of the code under test:
whole-number arithmetic, exact fractions, closed forms for one and two
degrees of freedom, the largest of the gaps between uniform points summed
over the Poisson count, a calculation to sixty digits, and seeded
simulations.
"""
import contextlib
import decimal
import io
import math
import random
import statistics
import unittest
from fractions import Fraction
from pathlib import Path

from examples import ch05_propagation as chapter5
from examples import ch16_silence as silence
from examples.ch20_judge import exact_interval
from swmetro import Result, core, dist, propagate, stats

ROOT = Path(__file__).resolve().parent.parent
EPOCH = 1_700_000_000_000_000_000                # ns from 1970 to 14 November 2023, 22:13:20 UTC
STAMP = EPOCH + 100                              # not a float: as one it is EPOCH
NEXT_TO_ONE = 0.9999999999999999                 # the largest float below one
TOO_FINE = "too fine for a float of this size"


def expected(name):
    return (ROOT / "expected" / f"{name}.txt").read_text(encoding="utf-8")


def printed(function, *args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        function(*args)
    return out.getvalue()


def duration(t):
    return t[1] - t[0]


def nearest(f, x):
    """Whether x is the float nearest to the square root of the fraction f, with a tie going to the even one."""
    below, above = math.nextafter(x, -math.inf), math.nextafter(x, math.inf)
    upper = (Fraction(x) + Fraction(above)) / 2
    even = (Fraction(x) / Fraction(math.ulp(x))).numerator % 2 == 0
    if not (f < upper ** 2 or (f == upper ** 2 and even)):
        return False
    if x == 0:
        return True
    lower = (Fraction(below) + Fraction(x)) / 2
    return lower ** 2 < f or (lower ** 2 == f and even)


class AnExactValueStaysWhereItIs(unittest.TestCase):
    def test_the_table_of_the_review(self):
        self.assertEqual(int(STAMP + 0.0), EPOCH)            # what a float zero does to the stamp: 100 ns
        self.assertEqual(int(STAMP - 0.0), EPOCH)
        result = Result(STAMP, 0, "ns")
        self.assertEqual(str(result), "1700000000000000100(0) ns")
        for low, high in (result.interval(), result.interval(0.5), core.rect_interval(STAMP, 0),
                          core.rect_interval(STAMP, 0, 1.0), core.rect_interval(STAMP, 5, 0)):
            self.assertEqual((low - EPOCH, high - EPOCH), (100, 100))                # Version 26: (0, 0)
            self.assertEqual((type(low), type(high)), (int, int))
        rng = random.Random(1)
        for sampler in (propagate.normal(STAMP, 0), propagate.rect(STAMP, STAMP)):
            draws = [sampler(rng) for _ in range(5)]
            self.assertEqual([d - EPOCH for d in draws], [100] * 5)                  # Version 26: 0
            self.assertTrue(all(type(d) is int for d in draws))

    def test_the_interval_agrees_with_the_probability(self):
        result = Result(STAMP, 0, "ns")
        self.assertEqual([result.prob_below(t) for t in (EPOCH, STAMP - 1, STAMP, STAMP + 1)], [0.0, 0.0, 1.0, 1.0])
        low, high = result.interval()
        self.assertEqual((result.prob_below(low - 1), result.prob_below(high)), (0.0, 1.0))
        self.assertEqual(result.expanded(), 0.0)
        self.assertEqual(result.report(), "(1700000000000000100 ± 0) ns, k = 1.96, p = 0.95")
        # A float that is held exactly stays the float that it is.
        self.assertEqual(Result(0.1, 0, "s").interval(), (0.1, 0.1))
        self.assertEqual(core.rect_interval(0.1, 0), (0.1, 0.1))

    def test_two_exact_readings_through_the_samplers(self):
        for make in (lambda t: propagate.normal(t, 0), lambda t: propagate.rect(t, t)):
            ys = propagate.monte_carlo(duration, [make(EPOCH), make(STAMP)], 2000, 1)
            self.assertEqual(set(ys), {100})                                         # Version 26: {0.0}
            self.assertEqual(propagate.summarize(ys), (100, 0.0, 100, 100))
        # An exact start and an end that is read to 1.5 ns, with the origin subtracted in whole numbers.
        ys = propagate.monte_carlo(duration, [propagate.normal(0, 0), propagate.normal(STAMP - EPOCH, 1.5)], 40_000, 1)
        mean, sd, _, _ = propagate.summarize(ys)
        self.assertAlmostEqual(mean, 100, delta=0.04)
        self.assertAlmostEqual(sd, 1.5, delta=0.03)
        with self.assertRaisesRegex(ValueError, TOO_FINE):
            propagate.normal(STAMP, 1.5)

    def test_a_sampler_without_a_spread_draws_nothing(self):
        alone = propagate.monte_carlo(lambda x: x[0], [propagate.normal(0, 1)], 500, 7)
        beside = propagate.monte_carlo(lambda x: x[1], [propagate.normal(STAMP, 0), propagate.normal(0, 1),
                                                         propagate.rect(STAMP, STAMP)], 500, 7)
        self.assertEqual(beside, alone)

    def test_three_readings_of_the_same_stamp(self):
        self.assertEqual(int(statistics.fmean([STAMP] * 3)), EPOCH)                  # the mean of Version 26
        result = core.type_a([STAMP] * 3, "ns")
        self.assertEqual((result.value - EPOCH, type(result.value), result.u, result.dof), (100, int, 0.0, 2))
        self.assertEqual(str(result), "1700000000000000100(0) ns")
        self.assertEqual(result.interval(), (STAMP, STAMP))
        # Three draws show an interval of one half; since Version 29 they are refused for 95 %.
        mean, sd, low, high = propagate.summarize([STAMP] * 3, 0.5)
        self.assertEqual((mean - EPOCH, type(mean), sd, low, high), (100, int, 0.0, STAMP, STAMP))
        # Readings that differ are a spread at that size, and the result refuses it as before.
        with self.assertRaisesRegex(ValueError, TOO_FINE):
            core.type_a([STAMP, STAMP, STAMP + 3], "ns")
        relative = core.type_a([t - EPOCH for t in (STAMP, STAMP, STAMP + 3)], "ns")
        self.assertEqual((relative.value, str(relative)), (101, "101.0(10) ns"))

    def test_a_mean_is_rounded_once(self):
        rng = random.Random(2601)
        for _ in range(400):
            n = rng.randint(2, 9)
            scale = rng.choice((1e-9, 1.0, 842.0, 1e12))
            xs = [scale * rng.uniform(0.5, 1.5) for _ in range(n)]
            exact = sum(map(Fraction, xs)) / n
            self.assertEqual(core.type_a(xs).value, float(exact))
            self.assertEqual(propagate.summarize(sorted(xs), 0.5)[0], float(exact))
            self.assertAlmostEqual(core.type_a(xs).value / statistics.fmean(xs), 1.0, places=15)
        self.assertEqual(core.type_a([1, 2, 4]).value, 7 / 3)
        self.assertEqual((core.type_a([1, 2, 3]).value, type(core.type_a([1, 2, 3]).value)), (2, int))
        self.assertEqual(core.type_a([0.1, 0.1, 0.1]).value, 0.1)

    def test_ordinary_results_keep_their_numbers(self):
        result = Result(2346.0, core.u_tri(1.0), "ms")
        k = statistics.NormalDist().inv_cdf(0.975)
        self.assertEqual(str(result), "2346.00(41) ms")
        low, high = result.interval()
        self.assertAlmostEqual(low, 2346 - k * core.u_tri(1.0), places=11)
        self.assertAlmostEqual(high, 2346 + k * core.u_tri(1.0), places=11)
        self.assertEqual(core.rect_interval(100, 0.5, 1.0), (99.5, 100.5))
        self.assertIn("  observed 0.593, corrected 0.592, u 0.024\n", expected("ch20_judge"))
        self.assertIn("MC, independent normal, 200000 draws:\n  sample mean 1.2862  sample sd 0.0348\n"
                      "  95 % of draws  [1.220, 1.356]\n", expected("ch05_propagation"))


def version_26_quantile(xs, q):
    """The interpolated quantile as Version 26 formed it: a float weight times the difference, always added."""
    pos = stats.exact_q(q) * (len(xs) - 1)
    i = int(pos // 1)
    j = min(i + 1, len(xs) - 1)
    return xs[i] + float(pos - i) * (xs[j] - xs[i])


class OtherSummariesOfExactReadings(unittest.TestCase):
    """Found while the review was worked: the interpolated quantile and the fitted line moved the same stamp."""

    def test_a_quantile_with_nothing_to_interpolate(self):
        same = [STAMP] * 5
        self.assertEqual(int(version_26_quantile(same, 0.9)), EPOCH)                 # x + 0.0 * 0: a float
        for q in (0, 0.25, 0.5, 0.9, 0.99, 1):
            self.assertEqual((stats.quantile(same, q) - EPOCH, type(stats.quantile(same, q))), (100, int))
        steps = [STAMP + i for i in range(5)]
        self.assertEqual([stats.quantile(steps, q) - EPOCH for q in (0, 0.25, 0.5, 0.75, 1)], [100, 101, 102, 103, 104])
        self.assertEqual(stats.quantile([1, 2, 3], 0.5), 2)
        self.assertEqual(stats.quantile([1, 2, 4], 0.75), 3.0)                       # interpolated: a float as before

    def test_ordinary_quantiles_keep_their_numbers(self):
        rng = random.Random(2605)
        for _ in range(300):
            xs = sorted(rng.choice((1.0, 842.0, 1e-6)) * rng.uniform(0.5, 1.5) for _ in range(rng.randint(1, 40)))
            if rng.random() < 0.3:
                xs = sorted(xs + xs[:3])                                             # with ties
            for q in (0, 0.01, 0.25, 0.5, 0.9, 0.95, 0.99, 0.999, 1):
                got, was = stats.quantile(xs, q), version_26_quantile(xs, q)
                # Version 27 returned the float of Version 26. Since Version 28 the interpolation is exact
                # and rounded once, which can differ from that float in the last digit.
                self.assertEqual(type(got), float, (xs, q))
                self.assertLessEqual(abs(got - was), math.ulp(was), (xs, q))
        self.assertIn("  observed 0.593, corrected 0.592, u 0.024\n", expected("ch20_judge"))

    def test_a_line_through_readings_of_one_stamp(self):
        a, b, ua, ub, s = stats.linfit([0, 1, 2], [STAMP] * 3)
        self.assertEqual((a - EPOCH, type(a), b, ua, ub, s), (100, int, 0.0, 0.0, 0.0, 0.0))    # Version 26: an intercept of EPOCH
        a, b, ua, ub, s = stats.linfit([0, 1, 2, 3], [STAMP + 3 * i for i in range(4)])
        self.assertEqual((a - EPOCH, b, s), (100, 3.0, 0.0))
        # A whole number that is a float stays a float, and so does every result that is not whole.
        self.assertEqual([type(v) for v in stats.linfit([0, 1, 2], [1, 3, 5])], [float] * 5)
        self.assertEqual(stats.linfit([0, 1, 2], [1, 3, 5])[:2], (1.0, 2.0))
        rng = random.Random(2606)
        for _ in range(200):
            xs = [rng.uniform(0, 100) for _ in range(6)]
            ys = [3 + 0.25 * x + rng.gauss(0, 1) for x in xs]
            fx, fy = [Fraction(x) for x in xs], [Fraction(y) for y in ys]
            mx, my = sum(fx) / 6, sum(fy) / 6
            slope = sum((x - mx) * (y - my) for x, y in zip(fx, fy)) / sum((x - mx) ** 2 for x in fx)
            got = stats.linfit(xs, ys)
            self.assertEqual((got[0], got[1]), (float(my - slope * mx), float(slope)))
            self.assertTrue(all(type(v) is float for v in got))


class ASharedErrorThatDoesNotCancel(unittest.TestCase):
    STAMPS = [EPOCH, EPOCH + 10 ** 9]
    US = [100_000_000, 100_000_001]

    def test_the_case_of_the_review(self):
        y, uc, cu = propagate.lpu(duration, self.STAMPS, self.US, {(0, 1): 1.0})
        self.assertEqual((y, cu), (10 ** 9, [-100_000_000.0, 100_000_001.0]))
        self.assertEqual(uc, 1.0)                                                    # Version 26: 0.0
        # In whole numbers: the two squares and the covariance leave 1.
        a, b = self.US
        self.assertEqual(a * a + b * b - 2 * a * b, 1)
        # In floats, as Version 26 summed them: the square of 100 000 001 is not a float.
        self.assertEqual(float(a) * a + float(b) * b + 2 * 1.0 * -float(a) * float(b), 0.0)
        self.assertNotEqual(int(float(b) * b), b * b)

    def test_one_calibration_variable(self):
        a, b = self.US
        start, end = self.STAMPS

        def model(z):
            return (end + b * z[0]) - (start + a * z[0])
        y, uc, cu = propagate.lpu(model, [0], [1])
        self.assertEqual((y, uc, cu), (10 ** 9, 1.0, [1.0]))
        self.assertEqual(uc, propagate.lpu(duration, self.STAMPS, self.US, {(0, 1): 1.0})[1])
        # Any two fully correlated readings: what is left is the difference of their uncertainties.
        rng = random.Random(2602)
        for _ in range(300):
            a = rng.randint(1, 10 ** rng.randint(1, 17))
            b = a + rng.randint(0, 1000)
            for r, left in ((1.0, b - a), (1, b - a), (Fraction(1), b - a), (-1.0, a + b), (0.0, None)):
                uc = propagate.lpu(duration, self.STAMPS, [a, b], {(0, 1): r})[1]
                if left is None:
                    self.assertTrue(nearest(Fraction(a * a + b * b), uc), (a, b))
                else:
                    self.assertEqual(uc, float(left), (a, b, r))

    def test_the_sizes_of_the_review(self):
        for u in (1e-200, 1e200, 5e-324, 1e-160, 1e160, 1.7e308):
            try:
                uc = propagate.lpu(sum, [0, 0], [u, u])[1]
            except OverflowError:
                self.assertGreater(Fraction(u) ** 2 * 2, Fraction(math.nextafter(math.inf, 0)) ** 2)
                continue
            self.assertTrue(nearest(2 * Fraction(u) ** 2, uc), u)                    # Version 26: 0.0 and inf at the first two
            if u == 5e-324:
                self.assertEqual(uc, 5e-324)                                         # 1.41 of the smallest float is the smallest
            else:
                self.assertAlmostEqual(uc / u, math.sqrt(2), places=14)
        self.assertEqual(f"{propagate.lpu(sum, [0, 0], [1e-200, 1e-200])[1]:.5e}", "1.41421e-200")
        self.assertEqual(f"{propagate.lpu(sum, [0, 0], [1e200, 1e200])[1]:.5e}", "1.41421e+200")

    def test_the_root_is_rounded_once(self):
        rng = random.Random(2603)
        for _ in range(500):
            xs = [rng.uniform(1, 1000), rng.uniform(1, 1000)]
            us = [x * rng.uniform(0.001, 0.05) for x in xs]
            r = rng.choice((0.0, 0.3, -0.7, 1.0, -1.0))
            for f in (chapter5.speedup, duration, lambda x: x[0] * x[1] + x[0]):
                y, uc, cu = propagate.lpu(f, xs, us, {(0, 1): r})
                # Half the change of f over one standard uncertainty either way, in fractions.
                exact = [Fraction(x) for x in xs]
                a = (f([exact[0] + Fraction(us[0]), exact[1]]) - f([exact[0] - Fraction(us[0]), exact[1]])) / 2
                b = (f([exact[0], exact[1] + Fraction(us[1])]) - f([exact[0], exact[1] - Fraction(us[1])])) / 2
                self.assertEqual(cu, [float(a), float(b)])
                # From Version 40 a float r is read as the decimal that it prints as: 0.3 is 3/10.
                self.assertTrue(nearest(a * a + b * b + 2 * Fraction(str(r)) * a * b, uc), (xs, us, r))
                as_floats = math.sqrt(max(0.0, cu[0] * cu[0] + cu[1] * cu[1] + 2 * r * cu[0] * cu[1]))
                if as_floats > 1e-6 * abs(cu[0]):                                    # where the float sum kept its digits
                    self.assertAlmostEqual(uc / as_floats, 1.0, places=9)

    def test_correlations_that_no_quantities_have(self):
        everything_opposed = {(0, 1): -1.0, (0, 2): -1.0, (1, 2): -1.0}
        # Version 26: "math domain error"; until Version 39: "negative variance", for this output only.
        with self.assertRaisesRegex(ValueError, "the correlations cannot hold together"):
            propagate.lpu(sum, [0, 0, 0], [1, 1, 1], everything_opposed)
        with self.assertRaisesRegex(ValueError, "the correlations cannot hold together"):
            propagate.lpu(sum, [0, 0], [1, 1], {(0, 1): -1.5})
        self.assertEqual(propagate.lpu(sum, [0, 0], [1, 1], {(0, 1): -1.0})[1], 0.0)

    def test_the_rate_of_a_clock(self):
        # A rate known to one part in 10**9, on stamps read 10**17 ns and 10**17 + 10**9 ns after the clock was set.
        part = Fraction(1, 10 ** 9)
        start = 10 ** 17
        self.assertEqual([part * t for t in (start, start + 10 ** 9)], self.US)
        self.assertEqual(part * 10 ** 9, 1)                  # on the second between them: 1 ns
        # Version 27 printed five lines from shared_rate(), which rounded the two uncertainties down to whole
        # nanoseconds. Since Version 28 the function returns its results, and clock_rate() prints them.
        out = printed(chapter5.clock_rate)
        self.assertTrue(out.startswith("one constant error of the rate of a\nclock, u = 1e-9; two stamps, 1 s\n"
                                       "apart, 10**17 ns after it was set:\n"
                                       "r = 1.0  D = 1000000000 ns  u_c = 1.0000 ns\n"
                                       "c*u(start) = -100000000.00 ns\nc*u(end)   = +100000001.00 ns\n"
                                       "the rate as the input:  u_c = 1.0000 ns\n"))
        text = expected("ch05_propagation")
        self.assertIn("the factor as the input:  u_c = 0.0000\n" + out + "three stages of 10(1) ms, summed;\n", text)
        self.assertEqual(len(text.splitlines()), 40)         # 19 in Version 26, 24 in Version 27, 33 until 39

    def test_seconds_after_the_subtraction(self):
        self.assertEqual(STAMP / 1e9 - EPOCH / 1e9, 0.0)     # each stamp turned into seconds first
        self.assertEqual((STAMP - EPOCH) / 1e9, 1e-07)       # the whole difference turned into seconds
        self.assertEqual(float(STAMP), float(EPOCH))


def version_26_k(p, dof):
    """The coverage factor as Version 26 found it: the quantile at 1/2 + p/2."""
    return dist.t_ppf(0.5 + p / 2, dof)


def version_26_wilson(k, n, p):
    """The Wilson interval with the normal quantile as Version 26 asked for it, at 1/2 + p/2."""
    z = statistics.NormalDist().inv_cdf(0.5 + p / 2)
    ph = k / n
    den = 1 + z * z / n
    mid = (ph + z * z / (2 * n)) / den
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (mid - half if k else 0.0), (mid + half if k < n else 1.0)


class LevelsNextToOne(unittest.TestCase):
    def test_the_search_of_ten_windows(self):
        self.assertEqual(1 - NEXT_TO_ONE ** (1 / 10), 0.0)   # the subtraction of Version 26, and the logarithm of it failed
        self.assertEqual(f"{silence.searched_bound(30, 10, NEXT_TO_ONE):.5f}", "1.30131")
        for level in (NEXT_TO_ONE, 1 - 1e-12, 0.999, 0.95, 0.5, 0.2, 1e-5, 1e-300):
            for windows in (1, 2, 10, 1000):
                with decimal.localcontext() as context:
                    context.prec = 400                       # enough for 1 - 1e-300
                    each = (decimal.Decimal(level).ln() / windows).exp()
                    want = float(-(1 - each).ln() / 30)
                got = silence.searched_bound(30, windows, level)
                self.assertAlmostEqual(got / want, 1.0, places=12, msg=(level, windows))
            # One window searched is one window chosen.
            self.assertAlmostEqual(silence.searched_bound(30, 1, level) / stats.quiet_bound(30, level), 1.0, places=12)
        self.assertEqual(f"{silence.searched_bound(30, 10):.5f}", "0.17584")

    def test_the_coverage_factor(self):
        # Closed forms for one and two degrees of freedom, from the tail a = (1 - p) / 2.
        for p in (NEXT_TO_ONE, 1 - 1e-12, 1 - 1e-9, 0.999, 0.95, 0.9, 0.5):
            a = Fraction(1 - p) / 2
            one = 1 / math.tan(math.pi * float(a))
            two = float(1 - 2 * a) / math.sqrt(float(2 * a * (1 - a)))
            self.assertAlmostEqual(Result(0.0, 1.0, "", 1).k(p) / one, 1.0, places=9, msg=p)
            self.assertAlmostEqual(Result(0.0, 1.0, "", 2).k(p) / two, 1.0, places=9, msg=p)
            normal = -statistics.NormalDist().inv_cdf(float(a))
            self.assertAlmostEqual(Result(0.0, 1.0).k(p) / normal, 1.0, places=12, msg=p)
            self.assertAlmostEqual(math.erfc(Result(0.0, 1.0).k(p) / math.sqrt(2)) / (1 - p), 1.0, places=6)
        # Version 26 added p / 2 to one half: an error at the level next to one, and four digits at 1 - 1e-12.
        with self.assertRaises(ValueError):
            version_26_k(NEXT_TO_ONE, 1)
        a = Fraction(1 - (1 - 1e-12)) / 2
        self.assertGreater(abs(version_26_k(1 - 1e-12, 1) * math.tan(math.pi * float(a)) - 1), 1e-5)
        # At the usual levels the factor is that of Version 26 to twelve digits, and at 0.95 and 0.99 in every digit.
        for p in (0.5, 0.6827, 0.9, 0.95, 0.9545, 0.99, 0.999):
            for dof in (1, 2, 4.2, 9, 30, 1e7, math.inf):
                self.assertAlmostEqual(Result(0.0, 1.0, "", dof).k(p) / version_26_k(p, dof), 1.0, places=12, msg=(p, dof))
                if p in (0.95, 0.99):
                    self.assertEqual(Result(0.0, 1.0, "", dof).k(p), version_26_k(p, dof), (p, dof))
        # Where the two differ, as at 0.999, the sum 1/2 + p/2 was not a float, and the tail is.
        self.assertNotEqual(Fraction(0.5 + 0.999 / 2), Fraction(1, 2) + Fraction(0.999) / 2)
        self.assertEqual(Fraction((1 - 0.999) / 2), (1 - Fraction(0.999)) / 2)
        self.assertEqual(Result(0.0, 1.0).k(0), 0.0)
        for p in (1, 1.5, -0.5, math.nan):                   # Version 26 gave -0.67 as the factor for p = -0.5
            with self.assertRaisesRegex(ValueError, "need 0 <= p < 1"):
                Result(0.0, 1.0).k(p)

    def test_the_wilson_interval(self):
        with self.assertRaises(statistics.StatisticsError):
            statistics.NormalDist().inv_cdf(0.5 + NEXT_TO_ONE / 2)                   # what Version 26 asked for
        for p in (NEXT_TO_ONE, 1 - 1e-12, 0.999, 0.95, 0.5):
            for k, n in ((3, 40), (0, 25), (40, 40), (17, 20)):
                low, high = stats.wilson(k, n, p)
                self.assertTrue(0 <= low <= k / n <= high <= 1, (p, k, n))
                # The limits are where the score statistic reaches the quantile z with 2 (1 - Phi(z)) = 1 - p.
                for limit in (low, high):
                    if 0 < limit < 1 and limit != k / n:
                        z = abs(k / n - limit) / math.sqrt(limit * (1 - limit) / n)
                        self.assertAlmostEqual(math.erfc(z / math.sqrt(2)) / (1 - p), 1.0, places=5, msg=(p, k, n))
        for k, n in ((3, 40), (0, 25), (40, 40), (17, 20), (1, 3)):
            self.assertEqual(stats.wilson(k, n), version_26_wilson(k, n, 0.95), (k, n))
        wide = [stats.wilson(3, 40, p) for p in (0.95, 0.999, 1 - 1e-12, NEXT_TO_ONE)]
        self.assertEqual(wide, sorted(wide, key=lambda pair: pair[1] - pair[0]))

    def test_the_other_bounds_at_that_level(self):
        ten = [(30 * i, 30 * i + 30) for i in range(10)]
        for bound in (lambda p: stats.quiet_bound(30, p), lambda p: stats.rate_upper(3, 30, p),
                      lambda p: stats.rate_limits(3, 30, p)[1], lambda p: silence.paired_bound(ten[:1], 10, p),
                      lambda p: silence.paired_upper(1, ten[:1], 10, p), lambda p: silence.searched_pairs(ten, 10, p),
                      lambda p: silence.gap_bound(300, 30, p), lambda p: exact_interval(2, 20, p)[1],
                      lambda p: Result(0.0, 1.0, "", 4).k(p)):
            values = [bound(p) for p in (0.5, 0.95, 0.999999, 1 - 1e-12, NEXT_TO_ONE)]
            self.assertTrue(all(math.isfinite(v) and v > 0 for v in values))
            self.assertEqual(values, sorted(values))

    def test_levels_that_the_two_sums_cannot_resolve(self):
        ten = [(30 * i, 30 * i + 30) for i in range(10)]
        for level in (0, 1e-300, 9.99e-10, 1, 1.5):
            with self.assertRaisesRegex(ValueError, "need 1e-9 <= level < 1"):
                silence.gap_bound(300, 30, level)
            with self.assertRaisesRegex(ValueError, "need 1e-9 <= level < 1"):
                silence.searched_pairs(ten, 10, level)
        # At the lowest level allowed the chance at the bound is 1 - level to eleven places, to sixty digits.
        for level in (1e-9, 1e-3, 0.5, 0.95):
            rate = silence.gap_bound(300, 30, level)
            self.assertAlmostEqual(float(gap_to_sixty_digits(rate, 300, 30)), 1 - level, places=11, msg=level)
        self.assertAlmostEqual(float(gap_to_sixty_digits(silence.gap_bound(300, 30, NEXT_TO_ONE), 300, 30))
                               / (1 - NEXT_TO_ONE), 1.0, places=9)


def gap_to_sixty_digits(rate, total, length):
    """The sum of gap_somewhere() in decimal arithmetic of sixty digits, in a context of its own: until
    Version 45 it set the precision of the context that every later test of the run then shared."""
    with decimal.localcontext() as context:
        context.prec = 60
        rate, total, length = decimal.Decimal(rate), decimal.Decimal(total), decimal.Decimal(length)
        out = decimal.Decimal(0)
        for k in range(1, int(total // length) + 1):
            m = rate * (total - k * length)
            lower = m ** (k - 1) if k > 1 else decimal.Decimal(1)
            term = (-k * rate * length).exp() * (m * lower / math.factorial(k) + lower / math.factorial(k - 1))
            out += term if k % 2 else -term
        return out


def largest_gap_at_least(n, share):
    """Chance that the largest of the n + 1 gaps that n uniform points leave on a record is at least `share`
    of the record, as an exact fraction (Whitworth)."""
    out = Fraction(0)
    for k in range(1, n + 2):
        rest = 1 - k * share
        if rest < 0 or (rest == 0 and n > 0):
            break
        out += (-1) ** (k + 1) * math.comb(n + 1, k) * rest ** n
    return out


def gap_by_counts(rate, total, length):
    """A second calculation of gap_somewhere(): the chance for the largest gap given the number of
    messages, summed over the Poisson distribution of that number."""
    mean, share = rate * total, Fraction(length) / Fraction(total)
    out, weight = 0.0, math.exp(-mean)
    for n in range(0, 600):
        out += weight * float(largest_gap_at_least(n, share))
        weight *= mean / (n + 1)
    return out


def simulated_gaps(rate, total, length, runs, rng):
    """Share of simulated records of a Poisson process that hold a gap of `length`, the ends included."""
    found = 0
    for _ in range(runs):
        last, gap = 0.0, False
        t = rng.expovariate(rate)
        while t < total:
            if t - last >= length:
                gap = True
                break
            last = t
            t += rng.expovariate(rate)
        found += gap or total - last >= length
    return found / runs


class AGapAnywhere(unittest.TestCase):
    def test_the_table_of_the_review(self):
        single = silence.searched_bound(30, 10)
        bounds = (stats.quiet_bound(30), single, silence.gap_bound(300, 30))
        self.assertEqual([f"{b:.5f}" for b in bounds], ["0.09986", "0.17584", "0.23854"])
        self.assertEqual(f"{bounds[2]:.10f}", "0.2385368187")
        chances = (math.exp(-30 * single), 1 - (1 - math.exp(-30 * single)) ** 10, silence.gap_somewhere(single, 300, 30))
        self.assertEqual([f"{100 * c:.2f}" for c in chances], ["0.51", "5.00", "22.48"])
        self.assertAlmostEqual(silence.gap_somewhere(bounds[2], 300, 30), 0.05, places=12)

    def test_the_largest_of_the_gaps_between_uniform_points(self):
        # No message: the record is one gap. One message: the larger of two parts of the record.
        self.assertEqual([largest_gap_at_least(0, Fraction(s, 10)) for s in (1, 5, 10)], [1, 1, 1])
        self.assertEqual([largest_gap_at_least(1, Fraction(s, 10)) for s in (1, 5, 6, 9, 10)],
                         [1, 1, Fraction(8, 10), Fraction(2, 10), 0])
        # Two messages, counted on a grid of 400 by 400 midpoints, for a gap of 0.4 of the record: 0.96.
        cells = [((i + 0.5) / 400, (j + 0.5) / 400) for i in range(400) for j in range(400)]
        counted = sum(max(min(a, b), abs(a - b), 1 - max(a, b)) >= 0.4 for a, b in cells) / len(cells)
        self.assertAlmostEqual(counted, float(largest_gap_at_least(2, Fraction(4, 10))), delta=0.004)
        self.assertEqual(largest_gap_at_least(2, Fraction(4, 10)), Fraction(96, 100))
        for rate, total, length in ((0.2385368187, 300, 30), (0.17584, 300, 30), (0.05, 300, 30), (0.1, 100, 30),
                                    (0.3, 45, 30), (1.0, 60, 3), (0.02, 600, 30)):
            self.assertAlmostEqual(silence.gap_somewhere(rate, total, length), gap_by_counts(rate, total, length),
                                   places=11, msg=(rate, total, length))

    def test_to_sixty_digits(self):
        worst = 0.0
        for parts in (1, 1.5, 2, 3, 5, 10, 12.5, 16, 20):
            for i in range(0, 200, 3):
                rate = i * 0.1 / 30
                got = silence.gap_somewhere(rate, 30 * parts, 30)
                worst = max(worst, abs(got - float(gap_to_sixty_digits(rate, 30 * parts, 30))))
        self.assertLess(worst, 1e-12)                        # 6e-14, at twenty gaps

    def test_simulated_records(self):
        rng = random.Random(2604)
        for rate, total, length, runs in ((silence.gap_bound(300, 30), 300, 30, 40_000),
                                          (silence.searched_bound(30, 10), 300, 30, 20_000), (0.1, 100, 30, 20_000)):
            want = silence.gap_somewhere(rate, total, length)
            got = simulated_gaps(rate, total, length, runs, rng)
            self.assertAlmostEqual(got, want, delta=4.5 * math.sqrt(want * (1 - want) / runs), msg=(rate, total))
        self.assertAlmostEqual(silence.gap_somewhere(silence.gap_bound(300, 30), 300, 30), 0.05, places=12)

    def test_a_gap_across_the_boundary_of_two_windows(self):
        messages, grid = (40, 74), (30, 60, 90)
        held = [sum(lo <= t < hi for t in messages) for lo, hi in zip(grid, grid[1:])]
        self.assertEqual(held, [1, 1])                       # the search of the windows finds no empty one
        self.assertFalse(any(42 <= t < 72 for t in messages))
        gaps = [b - a for a, b in zip((grid[0],) + messages, messages + (grid[-1],))]
        self.assertEqual((gaps, max(gaps) >= 30), ([10, 34, 16], True))
        # An empty window of the grid is a gap as well, so the search anywhere succeeds more often at every rate,
        for i in range(1, 80):
            rate = i * 0.01
            fixed = 1 - (1 - math.exp(-30 * rate)) ** 10
            self.assertGreater(silence.gap_somewhere(rate, 300, 30), fixed)
        # and its bound is the highest of the three.
        self.assertTrue(stats.quiet_bound(30) < silence.searched_bound(30, 10) < silence.gap_bound(300, 30))
        # A record of one window has one place for the gap.
        for rate in (0.0, 0.01, 0.1, 1.0):
            self.assertAlmostEqual(silence.gap_somewhere(rate, 30, 30), math.exp(-30 * rate), places=15)
        self.assertAlmostEqual(silence.gap_bound(30, 30), stats.quiet_bound(30), places=12)

    def test_limits_and_refusals(self):
        self.assertEqual(silence.gap_somewhere(0, 300, 30), 1.0)
        chances = [silence.gap_somewhere(i * 0.01, 300, 30) for i in range(0, 150)]
        self.assertEqual(chances, sorted(chances, reverse=True))
        self.assertTrue(all(0 <= c <= 1 for c in chances))
        for call, message in ((lambda: silence.gap_somewhere(0.1, 20, 30), "no room for the gap"),
                              (lambda: silence.gap_somewhere(0.1, 300, 0), "no room for the gap"),
                              (lambda: silence.gap_somewhere(0.1, 630, 30), "need 20 gaps or fewer"),
                              (lambda: silence.gap_somewhere(-0.1, 300, 30), "need rate >= 0"),
                              (lambda: silence.gap_somewhere(math.nan, 300, 30), "need rate >= 0"),
                              (lambda: silence.gap_bound(20, 30), "no room for the gap")):
            with self.assertRaisesRegex(ValueError, message):
                call()

    def test_the_transcript(self):
        out = printed(silence.search)
        piece = ("  at 0.09986 msg/s a search of ten\n  finds an empty window: 40.13 %\n"
                 "  a gap of 30 s anywhere in 300 s:\n    at most 0.23854 msg/s at 95 %\n"
                 "  at 0.17585 msg/s such a gap\n  is found somewhere: 22.48 %\n")
        self.assertIn(piece, out)
        text = expected("ch16_silence")
        self.assertTrue(text.endswith(out))
        self.assertEqual(len(text.splitlines()), 193)        # 162, 166 in Version 27

    def test_the_thirteenth_figure(self):
        source = (ROOT / "figures" / "make_figures.py").read_text(encoding="utf-8")
        for piece in ("GAP_MESSAGES, GAP_FROM = (40, 74), 42", 'save(fig, "fig-16-4-gap-anywhere.png")', "    gap_anywhere()\n",
                      '"shared\\nstarts"', 'save(fig, "fig-16-7-shared-starts.png")'):
            self.assertIn(piece, source)
        self.assertLess(source.index("    gap_anywhere()\n"), source.index("    shared_starts()\n"))
        self.assertEqual(source.count("\n    save(fig, "), 32)         # twelve in Version 26


if __name__ == "__main__":
    unittest.main()
