"""Regression tests added after the review of Version 27.

The review found that the example of a clock rate in Chapter 5 rounded the
uncertainties of its two stamps down to whole nanoseconds before it
propagated them, so that half a second kept 0 ns or 1 ns where it keeps
0.5 ns; that a quantile interpolated between two whole time stamps of
nineteen digits could fall outside both; and that the ranks of the Monte
Carlo coverage interval depended on whether its probability was given as
a float or as a fraction, and were not those of Supplement 1 to the GUM.
It asked for the chance of a gap in a record given the number of its
messages, with a figure, and for the two edges of a band in Figure 16.4
to be labelled.

Each test computes what it expects independently of the code under test:
whole-number and fraction arithmetic, the rule of the supplement written
out as it is stated, closed forms for one and two messages, a sum over
the Poisson count, and seeded simulations.
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
from swmetro import core, propagate, stats, u_rect

ROOT = Path(__file__).resolve().parent.parent
EPOCH = 1_700_000_000_000_000_000                # ns from 1970 to 14 November 2023, 22:13:20 UTC
CLOCK = 10 ** 17                                 # the reading of the clock in Listing 5.1, ns since it was set
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


def version_27_rate(first, last):
    """The clock rate of Version 27: each stamp's uncertainty rounded down to a whole nanosecond."""
    xs = [CLOCK + first, CLOCK + last]
    us = [x // 10 ** 9 for x in xs]
    return propagate.lpu(duration, xs, us, {(0, 1): 1.0})[1]


class HalfASecondKeepsHalfANanosecond(unittest.TestCase):
    ROWS = ((0, 10 ** 9, 1.0, 1.0), (0, 5 * 10 ** 8, 0.0, 0.5), (75 * 10 ** 7, 125 * 10 ** 7, 1.0, 0.5))

    def test_the_table_of_the_review(self):
        for first, last, was, want in self.ROWS:
            self.assertEqual(version_27_rate(first, last), was, (first, last))
            y, uc, cu, one = chapter5.shared_rate(first, last)
            self.assertEqual((y, uc, one), (last - first, want, want), (first, last))
            # One part in 10**9 of the interval itself, in whole numbers.
            self.assertEqual(Fraction(last - first, 10 ** 9), Fraction(want))

    def test_the_same_interval_anywhere_on_the_clock(self):
        rng = random.Random(2701)
        for _ in range(300):
            span = rng.choice((1, 3, 10 ** 3, 5 * 10 ** 8, 10 ** 9, rng.randint(1, 10 ** 12)))
            want = float(Fraction(span, 10 ** 9))
            for first in (0, 1, 999_999_999, 10 ** 9, rng.randint(0, 10 ** 15), -rng.randint(0, 10 ** 15)):
                y, uc, cu, one = chapter5.shared_rate(first, first + span)
                self.assertEqual((y, uc, one), (span, want, want), (first, span))
                # The contributions are the uncertainties of the two stamps themselves, to the last digit of a float.
                self.assertEqual(cu, [-float(Fraction(CLOCK + first, 10 ** 9)), float(Fraction(CLOCK + first + span, 10 ** 9))])
        self.assertEqual(chapter5.shared_rate(5, 5)[1:], (0.0, [-100000000.000000005, 100000000.000000005], 0.0))

    def test_what_one_part_in_a_billion_measures(self):
        self.assertEqual(chapter5.RATE, Fraction(1, 10 ** 9))
        # As a standard uncertainty: 1 ns on a second. As a limit with a rectangular distribution: 1 ns over root 3.
        self.assertEqual(chapter5.shared_rate()[3], 1.0)
        limit = chapter5.shared_rate(u=u_rect(1e-9))
        self.assertAlmostEqual(limit[3], 1 / math.sqrt(3), places=15)
        self.assertEqual(limit[1], limit[3])                 # the two forms agree for any u
        self.assertEqual(f"{limit[3]:.3f}", "0.577")
        self.assertEqual(chapter5.shared_rate(u=1e-9)[1], 1.0)
        self.assertEqual(chapter5.shared_rate(u=Fraction(3, 10 ** 9))[1:], (3.0, [-300000000.0, 300000003.0], 3.0))

    def test_the_transcript(self):
        out = printed(chapter5.clock_rate)
        self.assertEqual(out, "one constant error of the rate of a\nclock, u = 1e-9; two stamps, 1 s\n"
                              "apart, 10**17 ns after it was set:\n"
                              "r = 1.0  D = 1000000000 ns  u_c = 1.0000 ns\n"
                              "c*u(start) = -100000000.00 ns\nc*u(end)   = +100000001.00 ns\n"
                              "the rate as the input:  u_c = 1.0000 ns\n"
                              "half a second, 0.75 s later:\n"
                              "c*u(start) = -100000000.75 ns\nc*u(end)   = +100000001.25 ns\n"
                              "u_c = 0.5000 ns; the rate as the\ninput:  u_c = 0.5000 ns\n"
                              "1e-9 as a limit, rectangular, on\nthe second:  u_c = 0.5774 ns\n")
        text = expected("ch05_propagation")
        self.assertIn("the factor as the input:  u_c = 0.0000\n" + out + "three stages of 10(1) ms, summed;\n", text)
        self.assertEqual(len(text.splitlines()), 40)         # 24 in Version 27, 33 until Version 39
        # The lines of the chapter that the review reproduced are what they were.
        self.assertTrue(text.startswith("r = 0.0  S = 1.2855  u_c = 0.0347\nr = 0.6  S = 1.2855  u_c = 0.0236\n"))
        self.assertTrue(text.endswith("exact  R = 529.56  u = 131.09\nexact 95 %  [362.32, 806.45]\n"))


def version_27_quantile(xs, q):
    """The interpolated quantile as Version 27 formed it: a float weight times the difference."""
    pos = stats.exact_q(q) * (len(xs) - 1)
    i = int(pos // 1)
    j = min(i + 1, len(xs) - 1)
    w = float(pos - i)
    if not w or xs[j] == xs[i]:
        return xs[i]
    return xs[i] + w * (xs[j] - xs[i])


def interpolated(xs, q):
    """The quantile of sorted xs at rank q (n - 1), counted from 0, as an exact fraction."""
    pos = Fraction(q) * (len(xs) - 1)
    i = pos.numerator // pos.denominator
    j = min(i + 1, len(xs) - 1)
    return Fraction(xs[i]) + (pos - i) * (Fraction(xs[j]) - Fraction(xs[i])), xs[i], xs[j]


class AQuantileBetweenTwoStamps(unittest.TestCase):
    QS = (0, Fraction(1, 100), Fraction(1, 10), Fraction(1, 4), Fraction(1, 3), Fraction(1, 2), Fraction(9, 10),
          Fraction(99, 100), 1)

    def test_the_table_of_the_review(self):
        for low, high, was, want in ((99, 101, 0, 100), (129, 131, 256, 130)):
            xs = [EPOCH + low, EPOCH + high]
            self.assertEqual(int(version_27_quantile(xs, 0.5)) - EPOCH, was)         # outside both stamps
            got = stats.quantile(xs, 0.5)
            self.assertEqual((got - EPOCH, type(got)), (want, int))
            self.assertEqual(stats.quantile([low, high], 0.5), want)                 # with the origin subtracted first
        xs = [EPOCH + 99, EPOCH + 101]
        self.assertEqual([stats.quantile(xs, q) - EPOCH for q in (0, 0.5, 1)], [99, 100, 101])
        self.assertEqual([int(version_27_quantile(xs, q)) - EPOCH for q in (0, 0.5, 1)], [99, 0, 101])

    def test_in_order_between_its_neighbors_and_moved_with_the_origin(self):
        rng = random.Random(2702)
        for _ in range(150):
            steps = [rng.choice((0, 1, 2, 3, 99, 257, 10 ** 6)) for _ in range(rng.randint(2, 12))]
            small = [sum(steps[:i + 1]) for i in range(len(steps))]
            for origin in (EPOCH, 2 ** 62, -EPOCH, 10 ** 30):
                xs = [origin + x for x in small]
                last = None
                for q in self.QS:
                    want, a, b = interpolated(small, q)
                    got = stats.quantile(xs, q)
                    self.assertEqual(got - origin, want, (small, q, origin))          # translated by the origin, exactly
                    self.assertTrue(a + origin <= got <= b + origin and xs[0] <= got <= xs[-1])
                    self.assertTrue(last is None or last <= got)
                    self.assertIn(type(got), (int, Fraction) if want.denominator > 1 or float(got) != got else (int, float))
                    last = got
        # Stamps that floats happen to be equal to, 256 ns apart at this size, are whole numbers like the others.
        self.assertEqual(float(EPOCH), EPOCH)
        self.assertEqual(stats.quantile([EPOCH, EPOCH + 4_000_000], Fraction(1, 3)) - EPOCH, Fraction(4_000_000, 3))
        self.assertEqual(stats.quantile([EPOCH + 1, EPOCH + 10 ** 9 + 1], 0.5) - EPOCH, 500_000_001)

    def test_a_fraction_of_a_nanosecond(self):
        xs = [EPOCH + 100, EPOCH + 101]
        got = stats.quantile(xs, 0.5)
        self.assertEqual((got, type(got)), (Fraction(2 * EPOCH + 201, 2), Fraction))
        self.assertEqual(float(got), float(EPOCH))                                   # as a float it is below both
        self.assertEqual(stats.quantile(xs, 0.25) - EPOCH, Fraction(401, 4))
        # Readings that are fractions are kept in the same way; one that is a float is taken as that float.
        self.assertEqual(stats.quantile([Fraction(1, 3), Fraction(2, 3)], Fraction(1, 4)), Fraction(5, 12))
        self.assertEqual(stats.quantile([Fraction(1, 4), Fraction(3, 4)], 0.5), 0.5)
        self.assertEqual(type(stats.quantile([Fraction(1, 4), Fraction(3, 4)], 0.5)), float)

    def test_floats_are_interpolated_exactly_and_rounded_once(self):
        rng = random.Random(2703)
        for _ in range(400):
            xs = sorted(rng.choice((1.0, 842.0, 1e-6, 1e12)) * rng.uniform(-1.5, 1.5) for _ in range(rng.randint(2, 30)))
            last = None
            for q in self.QS:
                want, a, b = interpolated(xs, q)
                got = stats.quantile(xs, q)
                self.assertEqual((got, type(got)), (float(want), float), (xs, q))
                self.assertTrue(a <= got <= b)
                self.assertTrue(last is None or last <= got)
                self.assertLessEqual(abs(got - version_27_quantile(xs, q)), 2 * math.ulp(max(abs(a), abs(b))))
                last = got
        # Ordinary whole numbers are floats as before, in value and in type.
        self.assertEqual([stats.quantile([1, 2, 4], q) for q in (0.25, 0.5, 0.75)], [1.5, 2, 3.0])
        self.assertEqual(type(stats.quantile([1, 3], 0.5)), float)
        self.assertEqual(stats.quantile([1, 2], Fraction(1, 3)), 4 / 3)
        # Where the two values differ much in size, the float weight of Version 27 lost a tenth of the answer.
        wide, q = [-1e16, 3.0], Fraction(2 ** 53 - 1, 2 ** 53)
        exact = 3 - Fraction(10 ** 16 + 3, 2 ** 53)
        self.assertEqual(stats.quantile(wide, q), float(exact))
        self.assertEqual(version_27_quantile(wide, q), 2.0)
        self.assertAlmostEqual(float(exact), 1.8898, places=4)

    def test_the_quantiles_of_chapter_9_are_what_they_were(self):
        for name in ("ch09_histogram", "ch09_dependent_tail", "ch10_coordinated_omission", "ch21_psi"):
            self.assertTrue(expected(name))
        self.assertIn("mean of the p99s", expected("ch09_histogram"))
        rng = random.Random(2704)
        xs = sorted(rng.lognormvariate(3, 0.6) for _ in range(500))
        for q in (0.5, 0.9, 0.95, 0.99, 0.999):
            self.assertEqual(f"{stats.quantile(xs, q):.9f}", f"{version_27_quantile(xs, q):.9f}")


class OtherSummariesBetweenStamps(unittest.TestCase):
    """Found while the review was worked: the mean and the intercept of a fitted line fell outside the stamps
    in the same way."""

    def test_the_mean_of_two_stamps(self):
        xs = [EPOCH + 99, EPOCH + 100]
        self.assertEqual(int(statistics.mean(xs)), EPOCH)                            # Version 27: below both
        got = stats.mean(xs)
        self.assertEqual((got, type(got)), (Fraction(2 * EPOCH + 199, 2), Fraction))
        self.assertTrue(xs[0] <= got <= xs[1])
        mean, sd, low, high = propagate.summarize(xs, 0.5)         # two draws: too few for 95 % since Version 29
        self.assertEqual((mean, low, high), (got, xs[0], xs[1]))
        self.assertAlmostEqual(sd, math.sqrt(0.5), places=15)
        # A result is a value with an uncertainty, and at this size it refuses the pair as before.
        with self.assertRaisesRegex(ValueError, TOO_FINE):
            core.type_a(xs, "ns")
        self.assertEqual(str(core.type_a([x - EPOCH for x in xs], "ns")), "99.50(50) ns")
        rng = random.Random(2705)
        for _ in range(200):
            small = [rng.randint(0, 10 ** 4) for _ in range(rng.randint(1, 9))]
            got = stats.mean([EPOCH + x for x in small])
            self.assertEqual(got - EPOCH, Fraction(sum(small), len(small)))
            self.assertTrue(EPOCH + min(small) <= got <= EPOCH + max(small))

    def test_ordinary_means_are_those_of_the_library(self):
        rng = random.Random(2706)
        for _ in range(300):
            xs = [rng.choice((1.0, 842.0, 1e-9)) * rng.uniform(0.5, 1.5) for _ in range(rng.randint(1, 9))]
            self.assertEqual(stats.mean(xs), statistics.mean(xs))
            self.assertEqual(stats.mean(xs), float(sum(map(Fraction, xs)) / len(xs)))
        self.assertEqual([(stats.mean(v), type(stats.mean(v))) for v in ([1, 2, 3], [1, 2, 4], [0.5, 1])],
                         [(2, int), (7 / 3, float), (0.75, float)])
        self.assertEqual(stats.mean(iter([1, 2, 3])), 2)
        self.assertEqual(stats.mean([Fraction(1, 3), Fraction(2, 3)]), 0.5)
        self.assertEqual(stats.mean([Fraction(1, 3), Fraction(1, 2)]), Fraction(5, 12))

    def test_a_line_through_stamps_that_differ(self):
        ys = [EPOCH + 100, EPOCH + 101, EPOCH + 101, EPOCH + 102]
        a, b, ua, ub, s = stats.linfit([0, 1, 2, 3], ys)
        self.assertEqual((a - EPOCH, type(a), b), (Fraction(1001, 10), Fraction, 0.6))
        self.assertTrue(min(ys) <= a <= max(ys))
        self.assertEqual(int(float(a)), EPOCH)                                       # the float of Version 27: below every y
        self.assertAlmostEqual(ua, math.sqrt(0.07), places=15)
        # With the origin subtracted the same line has an intercept that a float holds.
        self.assertEqual(stats.linfit([0, 1, 2, 3], [y - EPOCH for y in ys])[:2], (100.1, 0.6))

    def test_an_intercept_far_from_its_readings(self):
        # Two clocks read at whole stamps with scatter: the intercept lies 1.7e18 ns from the readings, its
        # uncertainty is of that order, and a float shows it as well as it is known. Until Version 30 the
        # slope was a float as well. It is not rounded since Version 31: a float shows a thousandth of its
        # uncertainty, but times a stamp of nineteen digits it is 37 ns from the fitted line, the readings
        # scatter by 9.8 ns, and the scatter returned would not be that of the line returned.
        rng = random.Random(2707)
        xs = [EPOCH + 10 ** 6 * i + rng.randint(-40, 40) for i in range(12)]
        ys = [x + 500 + rng.randint(-30, 30) for x in xs]
        a, b, ua, ub, s = stats.linfit(xs, ys)
        self.assertEqual([type(v) for v in (a, b, ua, ub, s)], [float, Fraction, float, float, float])
        self.assertGreater(ua, 1000 * math.ulp(a))
        self.assertGreater(ub, 1000 * math.ulp(float(b)))
        fx, fy = [Fraction(x) for x in xs], [Fraction(y) for y in ys]
        mx, my = sum(fx) / 12, sum(fy) / 12
        slope = sum((x - mx) * (y - my) for x, y in zip(fx, fy)) / sum((x - mx) ** 2 for x in fx)
        self.assertEqual((a, b), (float(my - slope * mx), slope))
        self.assertEqual(f"{float(abs(Fraction(float(b)) - b) * EPOCH):.1f}", "36.9")
        rss = sum((y - Fraction(a) - b * x) ** 2 for x, y in zip(fx, fy))
        rounded = sum((y - Fraction(a) - Fraction(float(b)) * x) ** 2 for x, y in zip(fx, fy))
        self.assertEqual((f"{s:.2f}", f"{math.sqrt(rss / 10):.6f}", f"{math.sqrt(rounded / 10):.2f}"), ("9.83", f"{s:.6f}", "41.61"))
        # With an origin subtracted from the stamps, in whole numbers, the five are floats, and the slope,
        # its uncertainty and the scatter are the same.
        near = stats.linfit([x - EPOCH for x in xs], [y - EPOCH for y in ys])
        self.assertEqual([type(v) for v in near], [float] * 5)
        self.assertEqual((near[1], near[3], near[4]), (float(b), ub, s))
        self.assertEqual([type(v) for v in stats.linfit([0, 1, 2], [1, 3, 5])], [float] * 5)


def rule_of_the_supplement(m, p):
    """JCGM 101:2008, 7.7.2, as it is stated: q = pM if that is a whole number, otherwise the integer part of
    pM + 1/2; the interval from the r-th to the (r + q)-th of the M sorted values, with r = (M - q)/2 if that is
    a whole number and otherwise the integer part of (M - q + 1)/2. Ranks from 1."""
    pm = Fraction(p) * m
    q = int(pm) if pm.denominator == 1 else int(pm + Fraction(1, 2))
    half = Fraction(m - q, 2)
    r = int(half) if half.denominator == 1 else int(Fraction(m - q + 1, 2))
    return r, r + q, q


def version_27_ranks(n, p):
    """The two ranks of Version 27, counted from 1: computed in the arithmetic of p."""
    return int(n * (1 - p) / 2) + 1, int(n * (1 + p) / 2)


class TheRanksOfAMonteCarloInterval(unittest.TestCase):
    HUNDRED = list(range(1, 101))

    def test_the_table_of_the_review(self):
        self.assertEqual((version_27_ranks(100, 0.90), version_27_ranks(100, Fraction(9, 10))), ((5, 95), (6, 95)))
        self.assertEqual((version_27_ranks(100, 0.95), version_27_ranks(100, Fraction(19, 20))), ((3, 97), (3, 97)))
        self.assertEqual((rule_of_the_supplement(100, Fraction(9, 10))[:2], rule_of_the_supplement(100, Fraction(19, 20))[:2]),
                         ((5, 95), (3, 98)))
        for p in (0.90, Fraction(9, 10), "0.9", decimal.Decimal("0.90")):
            self.assertEqual(propagate.summarize(self.HUNDRED, p)[2:], (5, 95), p)
        for p in (0.95, Fraction(19, 20), "0.95", decimal.Decimal("0.95")):
            self.assertEqual(propagate.summarize(self.HUNDRED, p)[2:], (3, 98), p)   # 95 ranks apart

    def test_the_rule_of_the_supplement(self):
        rng = random.Random(2708)
        checked = 0
        for _ in range(3000):
            m = rng.choice((2, 3, 10, 99, 100, 101, 1000, rng.randint(2, 400)))
            p = Fraction(rng.randint(0, 1000), 1000)
            r, top, q = rule_of_the_supplement(m, p)
            if m - q < 1:
                continue                                     # the supplement has no interval of q steps among m values
            checked += 1
            got = propagate.summarize(list(range(1, m + 1)), p)[2:]
            self.assertEqual(got, (r, top), (m, p))
            below, above = r - 1, m - top                     # draws left out on either side: equal, or one more above
            self.assertIn(above - below, (0, 1), (m, p))
            self.assertEqual(top - r, q)
        self.assertGreater(checked, 2500)

    def test_however_the_probability_is_written(self):
        for m in (100, 101, 250, 1999):
            ys = list(range(1, m + 1))
            for k in range(0, 1000, 7):
                forms = (k / 1000, Fraction(k, 1000), f"{k / 1000}", decimal.Decimal(k) / 1000)
                got = {propagate.summarize(ys, p)[2:] for p in forms}
                self.assertEqual(len(got), 1, (m, k))
        # Version 27 gave two answers at some of these probabilities for 100 draws.
        split = sum(version_27_ranks(100, k / 1000) != version_27_ranks(100, Fraction(k, 1000)) for k in range(0, 1000, 7))
        self.assertGreater(split, 0)

    def test_more_probability_than_the_draws_can_show(self):
        # Version 28 returned the least and the greatest draw where p * n rounds to n. Since Version 29 such
        # draws are refused as too few, see test_review28.
        for p in (1, 0.999):                                                         # 99.9 rounds to 100 steps
            with self.assertRaisesRegex(ValueError, "more draws needed"):
                propagate.summarize(self.HUNDRED, p)
        self.assertEqual(propagate.summarize(self.HUNDRED, 0.994)[2:], (1, 100))     # 99 steps
        self.assertEqual(propagate.summarize(self.HUNDRED, 0.98)[2:], (1, 99))       # 98 steps, r = 1
        self.assertEqual(propagate.summarize(self.HUNDRED, 0)[2:], (50, 50))
        self.assertEqual(propagate.summarize([7, 7], 0.5), (7, 0.0, 7, 7))
        self.assertEqual(propagate.summarize([1, 2, 3], 0.5)[2:], (1, 3))
        for few in ([7, 7], [1, 2, 3]):
            with self.assertRaisesRegex(ValueError, "more draws needed"):
                propagate.summarize(few, 0.95)
        for p in (-0.1, 1.1):
            with self.assertRaisesRegex(ValueError, r"q must be in \[0, 1\]"):
                propagate.summarize(self.HUNDRED, p)

    def test_the_intervals_of_the_book(self):
        # 200 000 draws at 95 %: 190 000 steps from the 5000th draw to the 195 000th.
        self.assertEqual(rule_of_the_supplement(200_000, Fraction(19, 20)), (5000, 195_000, 190_000))
        self.assertEqual(propagate.summarize(range(1, 200_001))[2:], (5000, 195_000))
        self.assertEqual(version_27_ranks(200_000, 0.95), (5001, 195_000))            # one draw further in at the lower end
        text = expected("ch05_propagation")
        self.assertIn("  95 % of draws  [1.220, 1.356]\n", text)
        self.assertIn("MC 95 %  [362, 806]\n", text)


def largest_gap_at_least(n, share):
    """Chance that the largest of the n + 1 gaps that n uniform points leave on a record is at least `share`
    of the record, as an exact fraction."""
    out = Fraction(0)
    for k in range(1, n + 2):
        rest = 1 - k * share
        if rest < 0 or (rest == 0 and n > 0):
            break
        out += (-1) ** (k + 1) * math.comb(n + 1, k) * rest ** n
    return out


class TheLargestGapGivenTheCount(unittest.TestCase):
    def test_the_numbers_of_the_review(self):
        got = [silence.gap_given(60, 300, length) for length in (30, 35, 40, 45)]
        self.assertEqual([f"{100 * g:.2f}" for g in got], ["10.68", "3.55", "1.14", "0.36"])
        for length, g in zip((30, 35, 40, 45), got):
            self.assertEqual(g, float(largest_gap_at_least(60, Fraction(length, 300))))
        # The observed current of both records of the example: 60 messages in 300 s.
        self.assertEqual(Fraction(60, 300), Fraction(1, 5))

    def test_one_and_two_messages_by_hand(self):
        # No message: the record is the gap. One message: the larger of its two parts, 2 (1 - x) above one half.
        self.assertEqual([silence.gap_given(0, 300, length) for length in (1, 150, 300)], [1.0, 1.0, 1.0])
        self.assertEqual([silence.gap_given(1, 300, length) for length in (30, 150, 200, 270, 300)], [1.0, 1.0, 2 / 3, 0.2, 0.0])
        # Two messages, counted on a grid of 400 by 400 midpoints.
        cells = [((i + 0.5) / 400, (j + 0.5) / 400) for i in range(400) for j in range(400)]
        for share in (0.4, 0.5, 0.7):
            counted = sum(max(min(a, b), abs(a - b), 1 - max(a, b)) >= share for a, b in cells) / len(cells)
            self.assertAlmostEqual(silence.gap_given(2, 1, share), counted, delta=0.004)
        # With n messages the gaps are n + 1 and one of them is at least their mean.
        for n in (1, 5, 60, 600):
            self.assertEqual(silence.gap_given(n, 300, Fraction(300, n + 1)), 1.0)
            self.assertLess(largest_gap_at_least(n, Fraction(301, 300 * (n + 1))), 1)
        self.assertLess(silence.gap_given(5, 300, Fraction(301, 6)), 1.0)

    def test_simulated_records(self):
        rng = random.Random(2709)
        runs, hits = 40_000, {30: 0, 45: 0}
        for _ in range(runs):
            ts = sorted(rng.uniform(0, 300) for _ in range(60))
            largest = max(b - a for a, b in zip([0.0] + ts, ts + [300.0]))
            for length in hits:
                hits[length] += largest >= length
        for length, count in hits.items():
            want = silence.gap_given(60, 300, length)
            self.assertAlmostEqual(count / runs, want, delta=4.5 * math.sqrt(want * (1 - want) / runs), msg=length)

    def test_summed_over_the_count_it_is_the_chance_for_a_rate(self):
        for rate, total, length in ((0.2, 300, 30), (0.2385368187, 300, 30), (0.05, 300, 45), (0.5, 60, 10)):
            mean = rate * total
            out, weight = 0.0, math.exp(-mean)
            for n in range(0, 500):
                out += weight * silence.gap_given(n, total, length)
                weight *= mean / (n + 1)
            self.assertAlmostEqual(out, silence.gap_somewhere(rate, total, length), places=11, msg=(rate, total, length))
        # At the rate of 0.2 msg/s the count is not given, and the chance is another number, and a larger one.
        self.assertGreater(silence.gap_somewhere(0.2, 300, 30), silence.gap_given(60, 300, 30) + 0.02)

    def test_in_fractions_where_floats_would_not_do(self):
        # 200 messages in 300 s and a gap of 2 s: the terms alternate and reach 2e16, the chance is next to 1,
        # and floats, which keep sixteen digits, return a number that is no chance at all.
        share = Fraction(2, 300)
        terms = [math.comb(201, k) * float(1 - k * share) ** 200 for k in range(1, 151)]
        self.assertGreater(max(terms), 1e16)
        as_floats = sum(t if k % 2 else -t for k, t in enumerate(terms, 1))
        exact = largest_gap_at_least(200, share)
        self.assertTrue(Fraction(999, 1000) < exact < 1)
        self.assertEqual(silence.gap_given(200, 300, 2), float(exact))
        self.assertFalse(0 <= as_floats <= 1)
        lengths = [30 + i / 4 for i in range(61)]
        chances = [silence.gap_given(60, 300, length) for length in lengths]
        self.assertEqual(chances, sorted(chances, reverse=True))
        self.assertTrue(all(0 <= c <= 1 for c in chances))

    def test_refusals(self):
        for call, message in ((lambda: silence.gap_given(-1, 300, 30), "need a whole n >= 0"),
                              (lambda: silence.gap_given(2.5, 300, 30), "need a whole n >= 0"),
                              (lambda: silence.gap_given(60, 300, 0), "no room for the gap"),
                              (lambda: silence.gap_given(60, 300, 301), "no room for the gap")):
            with self.assertRaisesRegex(ValueError, message):
                call()
        self.assertEqual(silence.gap_given(60.0, 300, 30), silence.gap_given(60, 300, 30))

    def test_the_transcript(self):
        out = printed(silence.search)
        piece = ("  at 0.17584 msg/s such a gap\n  is found somewhere: 22.48 %\n"
                 "  with 60 messages counted, a gap of\n    30 s or more: 10.68 %\n    35 s or more: 3.55 %\n"
                 "    40 s or more: 1.14 %\n    45 s or more: 0.36 %\n")
        self.assertIn(piece, out)
        # Since Version 29 the eight lines of the changing rate stand between these and the pairs, and since
        # Version 30 the six of the second record.
        # Since Version 31 the eight lines of the matched requests follow those.
        self.assertIn("  two independent records: 6.27 %\nthe same for requests of 10 s that\nwrite two messages:\n", out)
        text = expected("ch16_silence")
        self.assertTrue(text.endswith(out))
        self.assertEqual(len(text.splitlines()), 193)        # 166 in Version 27

    def test_two_stamps_truncated_to_the_millisecond(self):
        # Stamps 30.000 s apart: the times behind them differ by more than 29.999 s and by less than 30.001 s.
        seen = set()
        for i in range(0, 1000, 7):
            for j in range(0, 1000, 11):
                first = Fraction(12_345 * 1000 + i, 10 ** 6)                         # seconds, to the microsecond
                second = Fraction((12_345 + 30_000) * 1000 + j, 10 ** 6)
                stamps = [t.numerator * 1000 // t.denominator for t in (first, second)]
                self.assertEqual(stamps[1] - stamps[0], 30_000)
                seen.add(second - first)
        self.assertTrue(Fraction(29_999, 1000) < min(seen) < 30 < max(seen) < Fraction(30_001, 1000))
        self.assertGreater(74 - 40, Fraction(30_001, 1000))                          # the 34 s of Figure 16.4


class TheFourteenthFigure(unittest.TestCase):
    def test_the_edges_of_the_band_and_the_new_curve(self):
        source = (ROOT / "figures" / "make_figures.py").read_text(encoding="utf-8")
        for piece in ('ax.text(band[0] - 1.2, 1.38, f"{band[0]}", ha="right", va="center")',
                      'ax.text(band[1] + 1.2, 1.38, f"{band[1]}", ha="left", va="center")',
                      "GAP_MESSAGES, GAP_FROM = (40, 74), 42", "GAP_COUNT, GAP_RECORD, GAP_MARKS = 60, 300, (30, 35, 40, 45)",
                      "c16s.gap_given(GAP_COUNT, GAP_RECORD, length)", 'save(fig, "fig-16-4-gap-anywhere.png")',
                      'save(fig, "fig-16-5-gap-given-count.png")', 'save(fig, "fig-16-7-shared-starts.png")'):
            self.assertIn(piece, source)
        self.assertEqual(source.count("\n    save(fig, "), 32)         # thirteen in Version 27
        order = [source.index(f"\n    {name}()\n") for name in ("gap_anywhere", "gap_given_count", "shared_starts")]
        self.assertEqual(order, sorted(order))
        self.assertEqual((42, 42 + 30), (42, 72))


if __name__ == "__main__":
    unittest.main()
