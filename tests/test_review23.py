"""Regression tests added after the review of Version 23.

The review found that the fitted line lost the offset between two clocks:
the slope came from small differences, but the intercept was carried back to
x = 0 through one rounded product of a stamp that counts nanoseconds from
1970, where neighboring floats lie 256 ns apart; that the kappa computed
from a share of reference passes divided by zero where the kappa computed
from counts reports that it is not defined, and returned 1 for 0.947 at a
share of 1e-16; that a request can also enter a sample of durations by its
end, which returns the shares of the source as the start does, and that the
rate drawn from a sample taken by another rule has its 95 % in name only;
and it proposed a worked case in which the shares are estimated from a
sample of 20 requests and the risk is divided between sample and silence.

Each test computes what it expects independently of the code under test:
least squares from whole-number sums and from residuals in exact fractions,
kappa from its definition in exact fractions and from counts of labels, the
start times that a rule admits counted on a grid and in a seeded simulation,
and the binomial limit by bisection on exact sums.
"""
import contextlib
import io
import math
import random
import statistics as st
import unittest
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from examples import ch16_silence as silence
from examples import ch19_float as floats
from examples import ch20_judge as judge
from swmetro import stats

ROOT = Path(__file__).resolve().parent.parent
EPOCH = 1_700_000_000_000_000_000                # ns from 1970 to 14 November 2023, 22:13:20 UTC


def expected(name):
    return (ROOT / "expected" / f"{name}.txt").read_text(encoding="utf-8")


def printed(function, *args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        function(*args)
    return out.getvalue()


def by_whole_sums(xs, ys):
    """Intercept and slope of the least-squares line from sums of whole numbers, as the review writes them:
    D = n Sxx - Sx^2, N = n Sxy - Sx Sy, b = N / D, a = (Sy D - Sx N) / (n D)."""
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    d = n * sum(x * x for x in xs) - sx * sx
    num = n * sum(x * y for x, y in zip(xs, ys)) - sx * sy
    return Fraction(sy * d - sx * num, n * d), Fraction(num, d)


def by_residuals(xs, ys):
    """The same line about the means, in exact fractions, with the squares of u(a), u(b) and s from its residuals."""
    xs, ys = [Fraction(x) for x in xs], [Fraction(y) for y in ys]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    s2 = sum((y - a - b * x) ** 2 for x, y in zip(xs, ys)) / (n - 2)
    return a, b, s2 * (Fraction(1, n) + mx * mx / sxx), s2 / sxx, s2


def version_23_intercept(xs, ys):
    """The intercept as Version 23 returned it: fitted from the first point, carried back in floating point."""
    x0, y0 = xs[0], ys[0]
    dx, dy = [x - x0 for x in xs], [y - y0 for y in ys]
    mx, my = st.fmean(dx), st.fmean(dy)
    sxx = sum((x - mx) ** 2 for x in dx)
    b = sum((x - mx) * (y - my) for x, y in zip(dx, dy)) / sxx
    return y0 + (my - b * mx) - b * x0


class TheOffsetBetweenTwoClocks(unittest.TestCase):
    STAMPS = [EPOCH + t for t in (0, 100, 200, 300)]

    def test_the_table_of_the_review(self):
        self.assertEqual(math.ulp(float(EPOCH)), 256.0)
        for offset, old in ((7, 0.0), (-7, 0.0), (129, 256.0)):
            second = [x + offset for x in self.STAMPS]
            self.assertEqual(version_23_intercept(self.STAMPS, second), old)       # what Version 23 returned
            self.assertEqual(stats.linfit(self.STAMPS, second), (float(offset), 1.0, 0.0, 0.0, 0.0))
            self.assertEqual(by_whole_sums(self.STAMPS, second), (offset, 1))

    def test_a_slope_other_than_one(self):
        first = EPOCH - EPOCH % 3
        xs = [first + t for t in (0, 300, 600, 900)]                               # four multiples of 3, 300 ns apart
        ys = [x // 3 + 7 for x in xs]
        self.assertTrue(all(x % 3 == 0 for x in xs))
        self.assertEqual(by_whole_sums(xs, ys), (7, Fraction(1, 3)))
        self.assertEqual(version_23_intercept(xs, ys), 64.0)
        # Since Version 29 a fit that leaves no residual returns a slope that no float holds as the fraction
        # that it is: the float 1/3, times a stamp of this size, is 31 ns from a third of it.
        self.assertEqual(stats.linfit(xs, ys), (7.0, Fraction(1, 3), 0.0, 0.0, 0.0))
        self.assertEqual(round(abs(Fraction(1 / 3) - Fraction(1, 3)) * xs[0]), 31)
        # A clock that runs fast by 20 parts in a million and is 1 500 ns ahead at 1970.
        xs = [EPOCH + 50_000 * k for k in range(8)]
        ys = [x + x // 50_000 + 1_500 for x in xs]
        a, b = by_whole_sums(xs, ys)
        self.assertEqual((a, b), (1_500, Fraction(50_001, 50_000)))
        got = stats.linfit(xs, ys)
        self.assertEqual(got, (1500.0, Fraction(50_001, 50_000), 0.0, 0.0, 0.0))
        self.assertNotEqual(version_23_intercept(xs, ys), 1500.0)

    def test_whole_stamps_with_scatter(self):
        rng = random.Random(2301)
        for _ in range(300):
            n = rng.randint(3, 12)
            offsets = sorted(rng.sample(range(0, 5000), n))
            for origin in (0, 1_000, 10 ** 12, EPOCH, 2 ** 62, -EPOCH):
                xs = [origin + t for t in offsets]
                ys = [x + rng.randint(-40, 40) + 3 * t // 7 for x, t in zip(xs, offsets)]
                a, b = by_whole_sums(xs, ys)
                a2, b2, ua2, ub2, s2 = by_residuals(xs, ys)
                self.assertEqual((a, b), (a2, b2))                                 # the two routes agree, exactly
                got = stats.linfit(xs, ys)
                # The exact values, rounded once, or since Version 31 not rounded where the line through the
                # rounded values would leave the readings: its residuals are those of the fit within 1e-6.
                for value, exact in zip(got[:2], (a, b)):
                    self.assertIn(value, (float(exact), exact))
                    self.assertTrue(type(value) is float or value == exact)
                if origin in (0, 1_000):
                    self.assertEqual(got[:2], (float(a), float(b)))
                rss = s2 * (n - 2)
                again = sum((y - Fraction(got[0]) - Fraction(got[1]) * x) ** 2 for x, y in zip(xs, ys))
                self.assertLessEqual(again - rss, rss / 10 ** 6)
                # Since Version 25 the scatter is the root of the exact square, rounded once: within one unit
                # in the last place of a root taken of the rounded square.
                self.assertAlmostEqual(got[4] / math.sqrt(s2) if s2 else got[4] + 1, 1.0, places=15)
                for value, want in ((got[2], ua2), (got[3], ub2)):
                    self.assertAlmostEqual(value / math.sqrt(want) if want else value + 1, 1.0, places=12)

    def test_floats_are_fractions_too(self):
        # Seconds since 1970 as floats, 238 ns apart at this size: the readings are what they are, exactly.
        base = 1.7e9
        step = math.ulp(base)
        self.assertEqual(f"{step * 1e9:.0f}", "238")
        xs = [base + k * 4096 * step for k in range(6)]
        ys = [x + 0.25 for x in xs]                                                # a clock a quarter second ahead
        self.assertTrue(all(Fraction(y) - Fraction(x) == Fraction(1, 4) for x, y in zip(xs, ys)))
        self.assertEqual(stats.linfit(xs, ys), (0.25, 1.0, 0.0, 0.0, 0.0))
        rng = random.Random(2302)
        for _ in range(200):
            n = rng.randint(3, 15)
            xs = [rng.uniform(0, 60) for _ in range(n)]
            ys = [2.5 + 0.75 * x + rng.gauss(0, 0.3) for x in xs]
            a, b, ua2, ub2, s2 = by_residuals(xs, ys)
            got = stats.linfit(xs, ys)
            self.assertEqual(got[:2], (float(a), float(b)))
            self.assertAlmostEqual(got[4] / math.sqrt(s2), 1.0, places=15)
            self.assertAlmostEqual(got[2] / math.sqrt(ua2), 1.0, places=12)
            self.assertAlmostEqual(got[3] / math.sqrt(ub2), 1.0, places=12)

    def test_one_answer_under_every_python(self):
        # Version 23 summed floats, and the built-in sum changed in Python 3.12: this intercept was
        # -0.005013876749498519 under 3.11 and -0.00501387674951273 under 3.12 and 3.13.
        rng = random.Random(1)
        ts = [k * 7.5 + rng.gauss(0, 0.05) for k in range(20)]
        got = stats.linfit(range(20), ts)
        a, b, ua2, ub2, s2 = by_residuals(range(20), ts)
        self.assertEqual(got[:2], (float(a), float(b)))
        self.assertEqual(got[:2], (-0.005013876749506785, 7.500587080936322))
        self.assertAlmostEqual(got[3], 0.0019487763773707917, places=18)

    def test_the_uncertainty_of_an_intercept_far_from_the_data(self):
        offsets, ys = (0, 1, 2, 3), (5, 6, 7, 9)
        near = stats.linfit(offsets, ys)
        far = stats.linfit([EPOCH + t for t in offsets], ys)
        self.assertEqual((float(far[1]), far[3], far[4]), (near[1], near[3], near[4]))
        a, b, ua2, _, _ = by_residuals([EPOCH + t for t in offsets], ys)
        self.assertEqual(far[:2], (a, b))                    # Version 31: the line at such stamps is not rounded
        self.assertEqual(float(far[0]), float(a))            # and float() gives what Version 30 returned
        self.assertAlmostEqual(far[2] / math.sqrt(ua2), 1.0, places=12)
        self.assertGreater(far[2], 1e17)                     # 1970 is far from four stamps of 2023, and u(a) says so

    def test_what_is_refused_and_what_is_read(self):
        for xs, ys, message in (([0, 1, 2, 3], [10], "one y for each x"), ([0, 1], [3, 5], "need three points"),
                                ([], [], "need three points"), ([EPOCH] * 3, [1, 2, 3], "xs do not vary"),
                                ([0.1, 0.1, 0.1, 0.1], [1, 2, 3, 4], "xs do not vary")):
            with self.assertRaisesRegex(ValueError, message):
                stats.linfit(xs, ys)
        for bad in (math.nan, math.inf, -math.inf):          # Version 23 returned nan for these
            for xs, ys in (([0, 1, bad], [1, 2, 3]), ([0, 1, 2], [1, bad, 3])):
                with self.assertRaises((ValueError, OverflowError)):
                    stats.linfit(xs, ys)
        with self.assertRaises(TypeError):
            stats.linfit([0, 1, None], [1, 2, 3])
        want = (1.0, 2.0, 0.0, 0.0, 0.0)
        self.assertEqual(stats.linfit((x for x in (0, 1, 2)), (y for y in (1, 3, 5))), want)
        self.assertEqual(stats.linfit([Fraction(0), Fraction(1), Fraction(2)], [Decimal(1), Decimal(3), Decimal(5)]), want)
        self.assertEqual(stats.linfit(range(3), [1, 3, 5.0]), want)
        # Until Version 67 this test gave True for the first y, and Python counted it as 1. A truth value is
        # no reading: the review of Version 67 found a flag taken for a number, and native() refuses it.
        with self.assertRaises(TypeError):
            stats.linfit(range(3), [True, 3, 5.0])

    def test_the_lines_of_the_book_are_what_they_were(self):
        for name, piece in (("ch16_trace_measurements", "E1042 period, fitted slope: 7.501(10) s\n"),
                            ("ch11_leak_rate", "240 min  0.0458  0.0029"),
                            ("ch12_calibration", "ind = 0.8"), ("ch16_timing_models", "phase jitter    0.0037  0.0019\n")):
            self.assertIn(piece, expected(name), name)
        text = " ".join(stats.linfit.__doc__.split())
        self.assertIn("the sums are exact", text)
        self.assertNotIn("first point", text)

    def test_one_nanosecond_converted_before_it_is_subtracted(self):
        self.assertEqual(float(EPOCH), EPOCH)                                      # the day itself is a float
        self.assertEqual(float(EPOCH + 1) - float(EPOCH), 0.0)
        self.assertEqual(float(EPOCH + 129) - float(EPOCH + 128), 256.0)
        self.assertEqual((EPOCH + 1) - EPOCH, (EPOCH + 129) - (EPOCH + 128))       # as whole numbers: 1 ns each
        out = printed(floats.stamps)
        self.assertEqual(out, "f64 ns since 1970: step 256 ns\n  stamps 1 ns apart, as f64:\n"
                              "    +0 and +1 ns: 0 ns\n    +128 and +129 ns: 256 ns\n")
        text = expected("ch19_float")
        self.assertIn("f64 time in 2027: step 238 ns\n" + out, text)
        self.assertEqual(len(text.splitlines()), 16)                 # 12 in Version 23


def kappa_by_definition(share, sens, spec):
    """Observed agreement less chance agreement, over one less chance agreement, in exact fractions."""
    p, se, sp = Fraction(share), Fraction(sens), Fraction(spec)
    agree = p * se + (1 - p) * sp
    passed = p * se + (1 - p) * (1 - sp)
    chance = p * passed + (1 - p) * (1 - passed)
    return (agree - chance) / (1 - chance)


def version_23_kappa_at(share, sens, spec):
    agree = share * sens + (1 - share) * spec
    passed = share * sens + (1 - share) * (1 - spec)
    chance = share * passed + (1 - share) * (1 - passed)
    return (agree - chance) / (1 - chance)


def kappa_of_counts(a, b):
    n = len(a)
    po = Fraction(sum(x == y for x, y in zip(a, b)), n)
    pe = sum(Fraction(a.count(c), n) * Fraction(b.count(c), n) for c in set(a) | set(b))
    return (po - pe) / (1 - pe)


class TwoKappasOneBoundary(unittest.TestCase):
    @staticmethod
    def labels(passes, failures, right_p, right_f):
        ref = [True] * passes + [False] * failures
        ver = [True] * right_p + [False] * (passes - right_p) + [False] * right_f + [True] * (failures - right_f)
        return ref, ver

    def test_the_two_cases_of_the_review(self):
        with self.assertRaises(ZeroDivisionError):
            version_23_kappa_at(0, 0.895, 1)
        with self.assertRaisesRegex(ValueError, "kappa needs two labels"):
            judge.kappa_at(0, 0.895, 1)
        self.assertEqual(version_23_kappa_at(1e-16, 0.9, 1), 1.0)
        want = kappa_by_definition(1e-16, 0.9, 1)
        self.assertEqual(f"{float(want):.6f}", "0.947368")
        self.assertAlmostEqual(judge.kappa_at(1e-16, 0.9, 1), float(want), places=12)
        # In the limit of no reference passes, with a judge that never passes a failure: 2 Se / (1 + Se).
        self.assertAlmostEqual(float(want), 2 * 0.9 / 1.9, places=12)

    def test_no_kappa_where_the_counts_have_none(self):
        # Both give every item the one label: no reference pass and a judge that passes nothing, or the reverse.
        for share, sens, spec, sizes in ((0, 0.895, 1, (0, 40, 0, 40)), (0, 0.0, 1, (0, 7, 0, 7)),
                                         (1, 1, 0.3, (25, 0, 25, 0)), (1, 1, 1, (3, 0, 3, 0))):
            with self.assertRaisesRegex(ValueError, "kappa needs two labels"):
                judge.kappa_at(share, sens, spec)
            with self.assertRaisesRegex(ValueError, "kappa needs two labels"):
                stats.cohen_kappa(*self.labels(*sizes))
        # One of the two with two labels: agreement is what chance gives, kappa is 0, from the share and from counts.
        for share, sens, spec, sizes in ((0, 0.5, 0.8, (0, 40, 0, 32)), (1, 0.9, 0.8, (50, 0, 45, 0))):
            self.assertEqual(judge.kappa_at(share, sens, spec), 0.0)
            self.assertEqual(stats.cohen_kappa(*self.labels(*sizes)), 0.0)

    def test_the_share_formula_against_the_definition(self):
        rng = random.Random(2303)
        shares = [10.0 ** -k for k in (300, 200, 100, 30, 16, 12, 8, 4, 2, 1)] + [0.3, 0.5, 0.7, 0.9, 1 - 1e-8, 1 - 1e-12]
        worst = 0.0
        for share in shares:
            for _ in range(40):
                sens, spec = rng.choice((0.0, 1.0, rng.random())), rng.choice((0.0, 1.0, rng.random()))
                want = float(kappa_by_definition(share, sens, spec))
                got = judge.kappa_at(share, sens, spec)
                worst = max(worst, abs(got - want))
                self.assertAlmostEqual(got, want, places=12, msg=(share, sens, spec))
        self.assertLess(worst, 1e-14)
        # Away from the ends the form of Version 23 gave the same numbers, and those of the transcript.
        for share in (0.05, 0.1, 0.3, 0.5, 0.8):
            for sens, spec in ((0.895, 0.845), (0.6, 0.7), (0.2, 0.3)):
                self.assertAlmostEqual(judge.kappa_at(share, sens, spec), version_23_kappa_at(share, sens, spec), places=12)
        self.assertEqual((f"{judge.kappa_at(0.5, 0.895, 0.845):.3f}", f"{judge.kappa_at(0.1, 0.895, 0.845):.3f}"), ("0.740", "0.470"))
        self.assertIn("Cohen's kappa 0.740\nsens + spec - 1 = 0.740\nkappa with 10 % reference passes: 0.470\n", expected("ch20_judge"))

    def test_kappa_from_counts_in_whole_numbers(self):
        rng = random.Random(2304)
        for _ in range(300):
            n = rng.randint(1, 60)
            alphabet = rng.choice(([True, False], ["pass", "fail", None], [0, 1, 2, 3]))
            a = [rng.choice(alphabet) for _ in range(n)]
            b = [x if rng.random() < 0.7 else rng.choice(alphabet) for x in a]
            if len(set(a) | set(b)) == 1:
                with self.assertRaisesRegex(ValueError, "kappa needs two labels"):
                    stats.cohen_kappa(a, b)
                continue
            self.assertEqual(stats.cohen_kappa(a, b), float(kappa_of_counts(a, b)))
        # A hundred thousand items and one disagreement: one less the chance agreement is 1e-5, and exact here.
        a = [1] * 100_000 + [0]
        b = [1] * 100_001
        self.assertEqual(stats.cohen_kappa(a, b), 0.0)
        b = [1] * 99_999 + [0, 0]
        self.assertEqual(stats.cohen_kappa(a, b), float(kappa_of_counts(a, b)))
        self.assertAlmostEqual(stats.cohen_kappa(a, b), 2 / 3, places=5)
        self.assertEqual(stats.cohen_kappa([1, 0, 1, 1], [1, 0, 0, 1]), 0.5)
        with self.assertRaisesRegex(ValueError, "no items"):
            stats.cohen_kappa([], [])
        with self.assertRaisesRegex(ValueError, "two labels an item"):
            stats.cohen_kappa([1, 1], iter([1]))

    def test_the_two_helpers_give_one_number(self):
        rng = random.Random(2305)
        for _ in range(200):
            passes, failures = rng.randint(1, 300), rng.randint(1, 300)
            right_p, right_f = rng.randint(0, passes), rng.randint(0, failures)
            ref, ver = self.labels(passes, failures, right_p, right_f)
            share = passes / (passes + failures)
            got = judge.kappa_at(share, right_p / passes, right_f / failures)
            self.assertAlmostEqual(got, stats.cohen_kappa(ref, ver), places=12)


class TwoIntervalsOneBoundary(unittest.TestCase):
    """Found while the review was worked: the exact interval of Listing 20.2, which Listing 16.4 now calls with a
    count that its caller gives, and the Wilson interval of the package answered the same edge differently."""

    def test_no_trials_and_counts_that_cannot_be(self):
        for function in (stats.wilson, judge.exact_interval):
            with self.assertRaisesRegex(ValueError, "no trials"):
                function(0, 0)                               # Version 23, exact interval: 0 to 1
            for k, n in ((3, 2), (-1, 5), (math.nan, 3), (math.inf, 3)):
                with self.assertRaisesRegex(ValueError, "need"):
                    function(k, n)                           # Version 23: 1 to 1, 0 to 0, or a math domain error
            for level in (0, 1, 1.5, -0.5):
                with self.assertRaisesRegex(ValueError, "need 0 < "):
                    function(2, 20, level)                   # Version 23: an interval of no width, or turned round
        with self.assertRaisesRegex(ValueError, "need a whole k >= 0"):
            judge.exact_interval(2.5, 20)                    # Version 23: the interval of 2 in 20
        self.assertEqual(judge.exact_interval(2.0, 20), judge.exact_interval(2, 20))
        with self.assertRaisesRegex(ValueError, "no trials"):
            judge.few_outputs(0, 0, 0.9, 0.85)               # Version 23: division by zero
        for k, n in ((0, 0), (21, 20), (-1, 20)):
            with self.assertRaises(ValueError):
                silence.calibrated(k, n)

    def test_the_intervals_are_what_they_were(self):
        self.assertEqual([f"{v:.6f}" for v in judge.exact_interval(2, 20)], ["0.012349", "0.316983"])
        self.assertEqual([f"{v:.4f}" for v in stats.wilson(2, 20)], ["0.0279", "0.3010"])
        low, high = judge.exact_interval(0, 1)
        self.assertEqual(low, 0.0)
        self.assertAlmostEqual(high, 0.975, places=12)
        low, high = judge.exact_interval(1, 1)
        self.assertAlmostEqual(low, 0.025, places=12)
        self.assertEqual(high, 1.0)
        # A binomial count is never below zero, whatever the chance of a success.
        for p in (0.0, 0.3, 1.0):
            self.assertEqual(judge.at_most(-1, 5, p), 0.0)   # Version 23: 1 at p = 0
        self.assertEqual((judge.at_most(0, 5, 0.0), judge.at_most(5, 5, 1.0), judge.at_most(4, 5, 1.0)), (1.0, 1.0, 0.0))
        rng = random.Random(2310)
        for _ in range(200):
            n = rng.randint(1, 40)
            k, p = rng.randint(-1, n), rng.random()
            want = float(sum(math.comb(n, i) * Fraction(p) ** i * (1 - Fraction(p)) ** (n - i) for i in range(k + 1)))
            self.assertAlmostEqual(judge.at_most(k, n, p), want, places=12)


class WhichRequestsEnterTheSample(unittest.TestCase):
    SOURCE = [(0.5, 10), (0.5, 50)]
    WHOLE = [(0, 30)]
    RULES = ("start", "end", "both", "either")

    @staticmethod
    def admits(rule, start, duration, window=30):
        first = 0 <= start < window
        last = 0 <= start + duration < window
        return {"start": first, "end": last, "both": first and last, "either": first or last}[rule]

    def test_the_table_of_the_review(self):
        want = {"start": ((30, 30), (0.5, 0.5), 50.0, "0.11983", "5.00"), "end": ((30, 30), (0.5, 0.5), 50.0, "0.11983", "5.00"),
                "both": ((20, 0), (1.0, 0.0), 40.0, "0.14979", "2.36"), "either": ((40, 60), (0.4, 0.6), 52.0, "0.11522", "5.61")}
        self.assertEqual(tuple(want), self.RULES)
        for rule, (lengths, mix, covered, rate, still) in want.items():
            self.assertEqual(tuple(silence.eligible(30, rule, d) for d in (10, 50)), lengths)
            total = sum(p * length for (p, _), length in zip(self.SOURCE, lengths))
            for (p, _), length, share in zip(self.SOURCE, lengths, mix):
                self.assertAlmostEqual(p * length / total, share, places=12)
            seen = silence.selected(self.SOURCE, 30, rule)
            self.assertEqual([d for _, d in seen], [10, 50])
            for (p, _), share in zip(seen, mix):
                self.assertAlmostEqual(p, share, places=12)
            exposure = sum(p * (30 + min(d, 30)) for p, d in zip(mix, (10, 50)))
            self.assertAlmostEqual(exposure, covered, places=12)
            self.assertAlmostEqual(silence.mean_reach(self.WHOLE, seen), covered, places=12)
            top = 2 * -math.log(0.05) / covered
            self.assertEqual(f"{top:.5f}", rate)
            self.assertEqual(f"{100 * math.exp(-top / 2 * 50):.2f}", still)       # the source covers 50 s of starts

    def test_the_start_times_that_a_rule_admits(self):
        step = 0.01
        for rule in self.RULES:
            for d in (0, 2.5, 10, 29.99, 30, 50, 90):
                grid = [-d - 1 + step * (i + 0.5) for i in range(int((d + 32) / step))]
                counted = step * sum(self.admits(rule, t, d) for t in grid)
                self.assertAlmostEqual(silence.eligible(30, rule, d), counted, delta=0.03, msg=(rule, d))
        with self.assertRaises(KeyError):
            silence.eligible(30, "middle", 10)
        with self.assertRaises(KeyError):
            silence.selected(self.SOURCE, 30, "middle")
        with self.assertRaisesRegex(ValueError, "no request selected"):
            silence.selected([(1, 50)], 30, "both")

    def test_a_request_taken_by_its_end(self):
        rng = random.Random(2306)
        for _ in range(200):
            ds = rng.sample((0, 1, 5, 10, 20, 29, 30, 40, 50, 90, 291), rng.randint(1, 5))
            ws = [rng.randint(1, 9) for _ in ds]
            source = [(w / sum(ws), d) for w, d in zip(ws, ds)]
            window = rng.choice((1, 10, 30, 45, 600))
            by_start, by_end = (silence.selected(source, window, rule) for rule in ("start", "end"))
            self.assertEqual(by_start, by_end)
            for (p, d), (p0, d0) in zip(by_end, source):
                self.assertEqual(d, d0)
                self.assertAlmostEqual(p, p0, places=12)       # Version 23: KeyError 'end'

    def test_a_simulated_sample(self):
        rng = random.Random(2307)
        counts = {rule: {10: 0, 50: 0} for rule in self.RULES}
        for _ in range(40_000):                               # independent windows of 30 s on a stationary source
            t = -60 + rng.expovariate(0.2)
            while t < 30:
                d = rng.choice((10, 50))
                for rule in counts:
                    counts[rule][d] += self.admits(rule, t, d)
                t += rng.expovariate(0.2)
        for rule, share_of_10 in (("start", 0.5), ("end", 0.5), ("both", 1.0), ("either", 0.4)):
            got = counts[rule][10] / (counts[rule][10] + counts[rule][50])
            self.assertAlmostEqual(got, share_of_10, delta=0.006, msg=rule)
            self.assertAlmostEqual(silence.selected(self.SOURCE, 30, rule)[0][0], share_of_10, places=12)

    def test_when_a_rule_returns_the_shares_of_the_source(self):
        rng = random.Random(2308)
        for _ in range(300):
            ds = rng.sample((1, 5, 10, 20, 29, 30, 31, 40, 50, 90), rng.randint(2, 4))
            ws = [rng.randint(1, 9) for _ in ds]
            source = [(w / sum(ws), d) for w, d in zip(ws, ds)]

            def kept(rule):
                seen = silence.selected(source, 30, rule)
                return all(abs(p - p0) < 1e-12 for (p, _), (p0, _) in zip(seen, source))
            self.assertTrue(kept("start") and kept("end"))
            # Either end: the same length for every duration exactly when none of them is shorter than the window.
            self.assertEqual(kept("either"), min(ds) >= 30, ds)
            # Both ends: never, with two durations of which one fits into the window.
            if min(ds) < 30:
                self.assertFalse(kept("both"), ds)
        for d in (30, 31, 90, 291):
            self.assertEqual(silence.eligible(30, "either", d), 60)                # twice the window
        text = " ".join(silence.selected.__doc__.split())
        self.assertNotIn("Only the first", text)
        self.assertIn('It is for "start" and for "end".', text)

    def test_weighting_a_sample_back(self):
        # The sample taken by either end: 0.40 / 40 and 0.60 / 60 are equal, the source has equal shares.
        seen = silence.selected(self.SOURCE, 30, "either")
        back = [p / length for (p, _), length in zip(seen, (40, 60))]
        self.assertAlmostEqual(back[0] / sum(back), 0.5, places=12)
        self.assertAlmostEqual(back[1] / sum(back), 0.5, places=12)
        rng = random.Random(2309)
        for _ in range(200):
            ds = rng.sample((1, 5, 10, 20, 29, 30, 40, 50, 90), rng.randint(1, 4))
            ws = [rng.randint(1, 9) for _ in ds]
            source = [(w / sum(ws), d) for w, d in zip(ws, ds)]
            for rule in self.RULES:
                lengths = [silence.eligible(30, rule, d) for d in ds]
                if not all(lengths):
                    continue
                back = [p / length for (p, _), length in zip(silence.selected(source, 30, rule), lengths)]
                for value, (p, _) in zip(back, source):
                    self.assertAlmostEqual(value / sum(back), p, places=12)
        # Complete pairs: two sources that differ in their long requests give one sample, and no weight tells them apart.
        self.assertEqual(silence.eligible(30, "both", 50), 0)
        self.assertEqual(silence.selected([(0.5, 10), (0.5, 50)], 30, "both"), silence.selected([(0.9, 10), (0.1, 50)], 30, "both"))

    def test_the_transcript(self):
        out = printed(silence.sampling)
        lines = out.splitlines()
        self.assertEqual(len(lines), 26)
        self.assertEqual(lines[:2], ["durations sampled with windows of", "30 s; a request is taken if it"])
        self.assertEqual(lines[8:14], ["  ends in a window:", "    from 30 s and 30 s of start times", "    50 % of 10 s, 50 % of 50 s",
                                       "    starts covered: 50 s", "    0.11983 msg/s at a nominal 95 %",
                                       "    the source is silent at that: 5.00 %"])
        self.assertEqual(lines[20:], ["  starts or ends in it:", "    from 40 s and 60 s of start times", "    40 % of 10 s, 60 % of 50 s",
                                      "    starts covered: 52 s", "    0.11523 msg/s at a nominal 95 %",
                                      "    the source is silent at that: 5.61 %"])
        self.assertNotIn("at most", out)                      # a rate drawn from a sampled mix is not called a bound
        self.assertEqual(out.count("nominal 95 %"), 4)
        text = expected("ch16_silence")
        self.assertIn(printed(silence.durations) + out + printed(silence.calibrated), text)
        self.assertEqual(len(text.splitlines()), 193)                # 118 in Version 23, 140, 152, 162, 166 in Version 27


def at_most(k, n, p):
    """P(X <= k) for a binomial count, summed in exact fractions."""
    p = Fraction(p)
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def upper_limit(k, n, risk):
    """The share at which a count of k or fewer among n still has probability `risk`: bisection on exact sums."""
    if k == n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if at_most(k, n, mid) > risk:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


class TheMixFromTwentyRequests(unittest.TestCase):
    LIMITS = {}

    @classmethod
    def setUpClass(cls):
        cls.LIMITS = {k: upper_limit(k, 20, Fraction(1, 40)) for k in range(21)}

    @staticmethod
    def covered(share):
        return 40 * share + 60 * (1 - share)                  # 30 s + min(30 s, duration), for 10 s and 50 s

    def test_the_numbers_of_the_review(self):
        top = self.LIMITS[2]
        self.assertEqual(f"{top:.6f}", "0.316983")
        self.assertAlmostEqual(judge.exact_interval(2, 20)[1], top, places=12)     # the upper end drawn in Figure 20.1
        self.assertEqual(f"{self.covered(top):.5f}", "53.66035")
        self.assertEqual(f"{-2 * math.log(0.025) / self.covered(top):.5f}", "0.13749")
        self.assertEqual(f"{-2 * math.log(0.05) / self.covered(0.1):.5f}", "0.10330")
        self.assertEqual(f"{-2 * math.log(0.05) / self.covered(1):.5f}", "0.14979")
        self.assertAlmostEqual(silence.mean_reach([(0, 30)], [(top, 10), (1 - top, 50)]), self.covered(top), places=12)

    def test_the_transcript(self):
        out = printed(silence.calibrated)
        self.assertEqual(out, "durations of 10 s and 50 s, the mix\nfrom 20 requests taken by their\n"
                              "start: 2 of 10 s. A silence of 30 s:\n  the 10 % taken as known:\n"
                              "    starts covered: 58 s\n    0.10331 msg/s at a nominal 95 %\n"
                              "  at most 31.699 % of 10 s, at 97.5 %:\n    starts covered: 53.6603 s\n"
                              "    silent at the nominal rate: 6.26 %\n    at most 0.13749 msg/s at 95 %\n"
                              "  no sample, all of 10 s:\n    starts covered: 40 s\n    at most 0.14979 msg/s at 95 %\n")
        self.assertIn(out + printed(silence.first_count), expected("ch16_silence"))
        # At the rate drawn from the 10 % as known, a source at the limit of what the sample allows is silent more
        # often than 5 %.
        plain = -2 * math.log(0.05) / self.covered(0.1)
        self.assertEqual(f"{100 * math.exp(-plain / 2 * self.covered(self.LIMITS[2])):.2f}", "6.26")

    def test_the_share_exceeds_its_limit_in_at_most_one_case_of_forty(self):
        worst = 0
        for share in [i / 200 for i in range(1, 200)] + [self.LIMITS[k] + 1e-9 for k in range(20)]:
            missed = sum(math.comb(20, k) * Fraction(share) ** k * (1 - Fraction(share)) ** (20 - k)
                         for k in range(21) if self.LIMITS[k] < share)
            worst = max(worst, missed)
            self.assertLessEqual(missed, Fraction(1, 40) + Fraction(1, 10 ** 9), share)
        self.assertGreater(worst, Fraction(24, 1000))        # and the limit is reached, just above a limit

    def test_the_bound_holds_in_nineteen_cases_of_twenty(self):
        # A silent window and a bound below the rate, summed over the samples: exact for the sample, and for
        # the window the probability of silence at that rate.
        def failure(share, rate, bound):
            still = math.exp(-rate / 2 * self.covered(share))
            return still * sum(float(math.comb(20, k) * Fraction(share) ** k * (1 - Fraction(share)) ** (20 - k))
                               for k in range(21) if bound(k) < rate)

        def combined(k):
            return -2 * math.log(0.025) / self.covered(self.LIMITS[k])
        for share in (0.02, 0.1, 0.25, 0.3169, 0.5, 0.75, 0.98):
            for k in range(21):
                rate = combined(k) * (1 + 1e-9)                # just above a value that the bound can take
                self.assertLessEqual(failure(share, rate, combined), 0.05, (share, k))
                if self.LIMITS[k] >= share:                     # the sample did not mislead: silence is rare there
                    self.assertLessEqual(math.exp(-rate / 2 * self.covered(share)), 0.025)
        self.assertEqual(f"{combined(2):.5f}", "0.13749")

    def test_other_samples(self):
        # No short request among 20: the share is below 1 - 0.025 ** (1 / 20). All 20 short: no limit below 1.
        self.assertAlmostEqual(self.LIMITS[0], 1 - 0.025 ** (1 / 20), places=12)
        out = printed(silence.calibrated, 0, 20)
        shown = math.ceil(100 * self.LIMITS[0] * 1000) / 1000                   # rounded upward since Version 59
        self.assertIn(f"  at most {shown:.3f} % of 10 s, at 97.5 %:\n", out)
        self.assertIn("  the 0 % taken as known:\n    starts covered: 60 s\n", out)
        out = printed(silence.calibrated, 20, 20)
        self.assertIn("  at most 100.000 % of 10 s, at 97.5 %:\n    starts covered: 40 s\n", out)
        shown = math.ceil(-2 * math.log(0.025) / 40 * 10 ** 5) / 10 ** 5
        self.assertIn(f"    at most {shown:.5f} msg/s at 95 %\n", out)


if __name__ == "__main__":
    unittest.main()
