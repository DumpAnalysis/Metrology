"""Regression tests added after the review of Version 43.

The review found that components() of Listing 13.1 took the length of the first launch for the length of all,
so that launches of unequal length gave a result that depended on their order; that Transcript 17.1 printed two
positive probabilities as 0.0000; and that the text of Chapter 12 called the uncertainty of the references the
floor of two curves that each approach a higher value. It asked for the two ends of the range of the exact
test in closed form, for a case in which the rounding of two scores decides, for a plan of launches within a
budget of time, and for a third host beside the two of Listing 6.3, which Figure 6.4 now draws.

Each test computes what it expects independently of the code under test: the analysis of variance in fractions
from its sums of squares; plans by trying every one; the exact test from binomial coefficients over every table
that two totals allow; the three hosts as a system of inequalities, by trying offsets on a grid.
"""
import contextlib
import io
import itertools
import math
import random
import statistics
import unittest
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
from pathlib import Path

from examples import ch06_exchange as exchange
from examples import ch12_calibration as calibration
from examples import ch13_variance_components as launches
from examples import ch17_accuracy as accuracy

ROOT = Path(__file__).resolve().parent.parent


def expected(name):
    return (ROOT / "expected" / f"{name}.txt").read_text(encoding="utf-8")


def printed(function, *args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        function(*args)
    return out.getvalue()


def refusal(function, *args):
    try:
        function(*args)
    except ValueError as error:
        return str(error)
    return None


class WhatTheCurvesApproach(unittest.TestCase):
    IND = 37.90

    def line(self):
        """The standard uncertainty of the fitted line at the indication, from a fit written out here."""
        rng = random.Random(12)
        xs = [r for r in calibration.REFS for _ in range(calibration.REPEATS)]
        ys = [calibration.probe(x, rng) for x in xs]
        n = len(xs)
        mx, my = math.fsum(xs) / n, math.fsum(ys) / n
        sxx = math.fsum((x - mx) ** 2 for x in xs)
        slope = math.fsum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
        intercept = my - slope * mx
        s = math.sqrt(math.fsum((y - intercept - slope * x) ** 2 for x, y in zip(xs, ys)) / (n - 2))
        x0 = (self.IND - intercept) / slope
        return s / slope, s / slope * math.sqrt(1 / n + (x0 - mx) ** 2 / sxx)

    def test_the_limits(self):
        one, line = self.line()
        self.assertEqual((f"{one:.6f}", f"{line:.6f}"), ("0.152491", "0.028212"))
        xs, fit = calibration.calibrate(random.Random(12))
        for times, want in ((1, "0.03458"), (5, "0.02365")):
            limit = math.sqrt(line ** 2 / times + calibration.U_REF ** 2)
            self.assertEqual(f"{limit:.5f}", want)
            self.assertAlmostEqual(calibration.measure(self.IND, xs, fit, math.inf, times).u, limit, places=12)
            self.assertGreater(limit, calibration.U_REF)                         # the references are below both curves
            us = [calibration.measure(self.IND, xs, fit, m, times).u for m in (1, 10, 100, 10 ** 4, 10 ** 8)]
            self.assertTrue(all(x > y > limit for x, y in zip(us, us[1:])))       # and each curve above its own limit
            self.assertAlmostEqual(us[-1], limit, places=6)

    def test_a_target_of_0_030_ms(self):
        one, line = self.line()

        def u(m, times):
            return math.sqrt(one ** 2 / m + line ** 2 / times + calibration.U_REF ** 2)
        self.assertGreater(math.sqrt(line ** 2 + calibration.U_REF ** 2), 0.030)   # out of reach of 30 readings
        self.assertEqual(next(m for m in range(1, 1000) if u(m, 5) <= 0.030), 69)
        self.assertEqual((f"{u(68, 5):.6f}", f"{u(69, 5):.6f}"), ("0.030019", "0.029936"))
        self.assertEqual((f"{u(68, 5):.3f}", f"{u(69, 5):.3f}"), ("0.030", "0.030"))  # rounded, they look alike
        xs, fit = calibration.calibrate(random.Random(12))
        self.assertGreater(calibration.measure(self.IND, xs, fit, 68, 5).u, 0.030)
        self.assertLessEqual(calibration.measure(self.IND, xs, fit, 69, 5).u, 0.030)
        self.assertGreater(calibration.measure(self.IND, xs, fit, 10 ** 12, 1).u, 0.030)

    def test_the_transcript(self):
        text = expected("ch12_calibration")
        lines = text.splitlines()
        self.assertEqual(len(lines), 13)                                         # 12 in Version 43
        _, line = self.line()
        limits = [math.sqrt(line ** 2 / times + calibration.U_REF ** 2) for times in (1, 5)]
        self.assertEqual(lines[12], f"  limit for many indications:    {limits[0]:.3f}  {limits[1]:.3f}")
        self.assertEqual(lines[12][-12:], "0.035  0.024")
        self.assertEqual(len(lines[12]), len(lines[11]))                         # the columns end together
        self.assertEqual(printed(calibration.main), text)


def version_43_components(data):
    """components() as Version 43 printed it."""
    a, n = len(data), len(data[0])
    means = [statistics.fmean(run) for run in data]
    grand = statistics.fmean(means)
    ms_b = n * statistics.variance(means)
    ms_w = statistics.fmean(statistics.variance(r) for r in data)
    var_b = max(0.0, (ms_b - ms_w) / n)
    return grand, ms_b, ms_w, var_b


def unbalanced(runs):
    """One-way random-effects analysis of variance for runs of any lengths, in fractions: the mean squares
    between and within, the effective size, and the variance between."""
    runs = [[Fraction(x) for x in run] for run in runs]
    total, k = sum(len(run) for run in runs), len(runs)
    grand = sum(sum(run) for run in runs) / total
    means = [sum(run) / len(run) for run in runs]
    between = sum(len(run) * (m - grand) ** 2 for run, m in zip(runs, means)) / (k - 1)
    within = sum(sum((x - m) ** 2 for x in run) for run, m in zip(runs, means)) / (total - k)
    size = (total - Fraction(sum(len(run) ** 2 for run in runs), total)) / (k - 1)
    return between, within, size, (between - within) / size


A, B, C = [9, 11], [11, 11, 13, 13], [13, 13, 13, 15, 15, 15]
MESSAGE = "need two runs or more, all of one length >= 2"


class LaunchesOfUnequalLength(unittest.TestCase):
    def test_what_version_43_returned(self):
        for runs, variance, u in (([A, B, C], "3.2444", "0.8165"), ([C, B, A], "3.7481", "1.4142")):
            grand, ms_b, ms_w, var_b = version_43_components(runs)
            self.assertEqual(f"{var_b:.4f}", variance)
            self.assertEqual(f"{math.sqrt(ms_b / 12):.4f}", u)                   # the formula of the listing for u
            self.assertEqual(grand, 12)

    def test_it_is_refused(self):
        for order in itertools.permutations([A, B, C]):
            self.assertEqual(refusal(launches.components, list(order)), MESSAGE)
        for bad in ([], [[1.0, 2.0]], [[1.0], [2.0]], [[1.0, 2.0], [3.0]], [[], []], [[1.0, 2.0], []]):
            self.assertEqual(refusal(launches.components, bad), MESSAGE, bad)
        self.assertIsNone(refusal(launches.components, [[1.0, 2.0], [2.0, 4.0]]))

    def test_a_balanced_design_is_what_it_was(self):
        data = launches.benchmark(random.Random(13))
        self.assertEqual(launches.components(data), version_43_components(data))
        self.assertEqual(launches.components(data), launches.components(data[::-1]))
        rng = random.Random(3)
        for _ in range(50):
            runs = [[rng.gauss(10, 2) for _ in range(4)] for _ in range(5)]
            self.assertEqual(launches.components(runs), version_43_components(runs))
            between, within, size, variance = unbalanced([[Fraction(x) for x in run] for run in runs])
            self.assertEqual(size, 4)                                            # the common length
            _, ms_b, ms_w, var_b = launches.components(runs)
            self.assertAlmostEqual(ms_b, float(between), places=9)
            self.assertAlmostEqual(ms_w, float(within), places=9)
            self.assertAlmostEqual(var_b, max(0.0, float(variance)), places=9)

    def test_the_analysis_for_unequal_lengths(self):
        for order in itertools.permutations([A, B, C]):
            between, within, size, variance = unbalanced(order)
            self.assertEqual((between, within, size, variance),
                             (Fraction(40, 3), Fraction(4, 3), Fraction(11, 3), Fraction(36, 11)))
        self.assertEqual((f"{40 / 3:.4f}", f"{4 / 3:.4f}", f"{36 / 11:.4f}", f"{36 / 11:.2f}"), ("13.3333", "1.3333", "3.2727", "3.27"))
        # What is averaged: every launch with the same weight, or every iteration.
        self.assertEqual(statistics.fmean(statistics.fmean(run) for run in (A, B, C)), 12)
        self.assertEqual(Fraction(sum(A + B + C), 12), Fraction(38, 3))
        self.assertEqual(f"{38 / 3:.2f}", "12.67")

    def test_the_function_of_the_listing(self):
        want = (Fraction(38, 3), Fraction(40, 3), Fraction(4, 3), Fraction(36, 11))
        for order in itertools.permutations([A, B, C]):
            for now, exact in zip(launches.unbalanced(list(order)), want):
                self.assertAlmostEqual(now, float(exact), places=12)
        rng = random.Random(11)
        for _ in range(200):                                                     # runs of any lengths, against fractions
            runs = [[rng.randrange(0, 40) for _ in range(rng.randrange(1, 7))] for _ in range(rng.randrange(2, 6))]
            if sum(map(len, runs)) == len(runs):
                self.assertIsNotNone(refusal(launches.unbalanced, runs))         # no run with a second iteration
                continue
            between, within, size, variance = unbalanced(runs)
            grand, ms_b, ms_w, var_b = launches.unbalanced(runs)
            self.assertAlmostEqual(grand, sum(map(sum, runs)) / sum(map(len, runs)), places=12)
            self.assertAlmostEqual(ms_b, float(between), places=9)
            self.assertAlmostEqual(ms_w, float(within), places=9)
            self.assertAlmostEqual(var_b, max(0.0, float(variance)), places=9)
        for bad in ([], [[1, 2]], [[1], [2]], [[1, 2], []]):
            self.assertEqual(refusal(launches.unbalanced, bad), "need two runs or more, two iterations in one", bad)
        # For a balanced design it is the analysis of components(), with the mean of all iterations.
        data = launches.benchmark(random.Random(13))
        for now, then in zip(launches.unbalanced(data), launches.components(data)):
            self.assertAlmostEqual(now, then, places=9)

    def test_an_estimate_without_bias(self):
        # Launches with a variance of 4 between them and of 1 within, of lengths 2, 4, 6 and 3: the mean of the
        # estimate over 20 000 sets is the 4 that was put in. With the length of the first launch for all, as
        # in Version 43, it is 4 + (1/2 + 1/4 + 1/6 + 1/3) / 4 - 1/2, the variance of the launch means less
        # the share of the first launch alone.
        rng = random.Random(1344)
        sizes, now, then = (2, 4, 6, 3), [], []
        for _ in range(20_000):
            runs = []
            for n in sizes:
                base = rng.gauss(0, 2)
                runs.append([base + rng.gauss(0, 1) for _ in range(n)])
            _, ms_b, ms_w, _ = launches.unbalanced(runs)
            size = (15 - sum(n * n for n in sizes) / 15) / 3
            now.append((ms_b - ms_w) / size)                                     # before it is cut off at zero
            means = [statistics.fmean(run) for run in runs]
            first = 2 * statistics.variance(means) - statistics.fmean(statistics.variance(r) for r in runs)
            then.append(first / 2)
        self.assertLess(abs(statistics.fmean(now) - 4), 4 * statistics.stdev(now) / math.sqrt(len(now)))
        expected_then = 4 + Fraction(5, 4) / 4 - Fraction(1, 2)
        self.assertEqual(expected_then, Fraction(61, 16))                        # 3.8125
        self.assertLess(abs(statistics.fmean(then) - expected_then), 4 * statistics.stdev(then) / math.sqrt(len(then)))
        self.assertGreater(4 - statistics.fmean(then), 4 * statistics.stdev(then) / math.sqrt(len(then)))

    def test_the_first_lines_of_the_transcript(self):
        lines = expected("ch13_variance_components").splitlines()
        self.assertEqual(lines[:10], ["grand mean, ms            251.75", "repeatability s_w, ms       1.65",
                                      "between-run s_b, ms         2.20", "intraclass correlation      0.64",
                                      "naive u (n = 80), ms        0.30", "u from run means, ms        0.80",
                                      "design, runs x iterations: u, ms", "   8 x 10   0.80", "  40 x 2    0.39", "  80 x 1    0.31"])


def every_plan(var_b, var_w, startup, total):
    """Each balanced design whose launches, with their startup, fit in the time of `total` iterations."""
    plans = []
    for runs in range(1, total + 1):
        for iters in range(1, total + 1):
            if runs * (startup + iters) <= total:
                plans.append((math.sqrt(var_b / runs + var_w / (runs * iters)), runs, iters))
    return plans


class ATimeBudget(unittest.TestCase):
    def fitted(self):
        _, _, ms_w, var_b = launches.components(launches.benchmark(random.Random(13)))
        return var_b, ms_w

    def test_the_table_of_the_review(self):
        var_b, var_w = self.fitted()
        want = (((19, 1), 323, "0.631"), ((17, 3), 323, "0.582"), ((12, 10), 312, "0.654"))
        for (runs, iters), time, u in want:
            self.assertEqual(runs * (16 + iters), time)
            self.assertEqual(f"{math.sqrt(var_b / runs + var_w / (runs * iters)):.3f}", u)
        self.assertGreater(20 * (16 + 1), 323)                                   # a twentieth launch does not fit

    def test_the_best_plan(self):
        var_b, var_w = self.fitted()
        plans = sorted(every_plan(var_b, var_w, 16, 323))
        self.assertEqual(plans[0][1:], (17, 3))
        self.assertLess(plans[0][0], plans[1][0])                                # and no other plan as good
        runs, iters, u = launches.budget(var_b, var_w, 16, 323)
        self.assertEqual((runs, iters), (17, 3))
        self.assertAlmostEqual(u, plans[0][0], places=15)

    def test_other_budgets(self):
        rng = random.Random(16)
        for _ in range(300):
            var_b, var_w = rng.choice((0.0, 0.3, 2.0, 9.0)), rng.choice((0.5, 1.0, 6.0))
            startup, total = rng.randrange(0, 12), rng.randrange(1, 90)
            plans = every_plan(var_b, var_w, startup, total)
            got = launches.budget(var_b, var_w, startup, total)
            if not plans:
                self.assertIsNone(got)
                continue
            self.assertAlmostEqual(got[2], min(plans)[0], places=12)
            self.assertLessEqual(got[0] * (startup + got[1]), total)
            self.assertAlmostEqual(got[2], launches.u_plan(var_b, var_w, got[0], got[1]), places=15)
        self.assertIsNone(launches.budget(1.0, 1.0, 16, 16))                     # no time for a first iteration
        self.assertEqual(launches.budget(0.0, 1.0, 16, 100)[:2], (1, 84))        # nothing between launches: one launch
        self.assertEqual(launches.budget(4.0, 1.0, 0, 50)[:2], (50, 1))          # launches that cost nothing

    def test_the_transcript(self):
        text = expected("ch13_variance_components")
        lines = text.splitlines()
        self.assertEqual(len(lines), 17)                                         # 10 in Version 43
        var_b, var_w = self.fitted()
        rows = [f"{runs:>4} x {iters:<3}{math.sqrt(var_b / runs + var_w / (runs * iters)):>7.3f}"
                for runs, iters in ((19, 1), (17, 3), (12, 10))]
        self.assertEqual(lines[10:14], ["a launch costs 16 iterations, 323 in all:"] + rows)
        self.assertEqual(lines[14:], ["runs of 2, 4 and 6 iterations:", f"  mean of runs {12:.2f} ms, of iterations {38 / 3:.2f} ms",
                                      f"  variance between runs {36 / 11:.2f} ms^2"])
        self.assertEqual(rows, ["  19 x 1    0.631", "  17 x 3    0.582", "  12 x 10   0.654"])
        self.assertEqual(printed(launches.main), text)


def uneven(only_a, only_b):
    """Two-sided exact probability of a division at least as uneven, from binomial coefficients: a fraction."""
    n, low = only_a + only_b, min(only_a, only_b)
    tail = Fraction(sum(math.comb(n, k) for k in range(low + 1)), 2 ** n)
    return min(Fraction(1), 2 * tail)


def every_probability(right_a, right_b, n):
    """The exact probability for each number of items that both models can have right, from the fewest."""
    return [uneven(right_a - both, right_b - both) for both in range(max(0, right_a + right_b - n), min(right_a, right_b) + 1)]


class TheEndsOfTheRange(unittest.TestCase):
    def test_the_table_of_the_review(self):
        want = ((1727, Fraction(1, 16), "0.864711"), (1758, Fraction(1, 34_359_738_368), "0.124738"),
                (1768, Fraction(1, 35_184_372_088_832), "0.046195"))
        for right_b, least, most in want:
            ps = every_probability(1722, right_b, 2000)
            self.assertEqual(min(ps), least)
            self.assertEqual(f"{float(max(ps)):.6f}", most)
            low, high = accuracy.limits(1722, right_b, 2000)
            self.assertEqual(low, least)                                         # a fraction, and exact
            self.assertAlmostEqual(high, float(max(ps)), places=15)
        self.assertEqual((f"{1 / 34_359_738_368:.3g}", f"{1 / 35_184_372_088_832:.3g}"), ("2.91e-11", "2.84e-14"))
        self.assertEqual((2 ** 35, 2 ** 45), (34_359_738_368, 35_184_372_088_832))

    def test_every_pair_of_totals_up_to_31_items(self):
        pairs = 0
        for n in range(1, 32):
            for right_a in range(n + 1):
                for right_b in range(n + 1):
                    ps = every_probability(right_a, right_b, n)
                    lead = abs(right_b - right_a)
                    self.assertTrue(all(x >= y for x, y in zip(ps, ps[1:])))     # more in common, never more probable
                    self.assertEqual(ps[-1], Fraction(1, 2 ** max(0, lead - 1)))
                    low, high = accuracy.limits(right_a, right_b, n)
                    self.assertEqual(low, ps[-1])
                    self.assertAlmostEqual(high, float(ps[0]), places=15)
                    pairs += 1
        self.assertEqual(pairs, 11_439)

    def test_one_item_fewer_in_common(self):
        # One item fewer that both have right is one more for each model alone. With the lead as it is, the
        # division is no more uneven than before.
        rng = random.Random(45)
        for _ in range(400):
            only_a, only_b = rng.randrange(0, 150), rng.randrange(0, 150)
            self.assertGreaterEqual(uneven(only_a + 1, only_b + 1), uneven(only_a, only_b))
        self.assertGreater(uneven(1, 6), uneven(0, 5))
        self.assertEqual((uneven(0, 5), uneven(0, 1), uneven(0, 0), uneven(3, 3)), (Fraction(1, 16), 1, 1, 1))

    def test_never_zero(self):
        low, high = accuracy.limits(0, 2000, 2000)
        self.assertEqual(low, Fraction(1, 2 ** 1999))
        self.assertGreater(low, 0)                                               # where a float has only 0.0
        self.assertEqual(float(low), 0.0)
        self.assertEqual(accuracy.limits(7, 7, 20)[0], 1)
        self.assertEqual(accuracy.limits(7, 8, 20)[0], 1)

    def test_rounding_that_decides(self):
        def stands_for(shown, n=2000):
            """The numbers of right answers that a percentage with one decimal stands for, halves rounded up."""
            return [k for k in range(n + 1)
                    if (Decimal(100 * k) / n).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP) == Decimal(shown)]
        self.assertEqual((stands_for("86.1"), stands_for("88.4")), ([1721, 1722], [1767, 1768]))
        most = {(a, b): float(max(every_probability(a, b, 2000))) for a in stands_for("86.1") for b in stands_for("88.4")}
        self.assertEqual(f"{most[1722, 1767]:.6f}", "0.051495")
        self.assertEqual(f"{most[1722, 1768]:.6f}", "0.046195")
        self.assertEqual(sorted(pair for pair, p in most.items() if p >= 0.05), [(1722, 1767)])
        self.assertAlmostEqual(accuracy.limits(1722, 1767, 2000)[1], most[1722, 1767], places=15)
        self.assertEqual(accuracy.limits(1722, 1767, 2000)[0], Fraction(1, 2 ** 44))

    def test_the_transcript(self):
        text = expected("ch17_accuracy")
        lines = text.splitlines()
        self.assertEqual(len(lines), 43)                                         # 27 in Version 43, 28 in 44
        rows = []
        for right_b in (1727, 1758, 1767, 1768):
            ps = every_probability(1722, right_b, 2000)
            low = float(min(ps))
            low = f"{low:.6f}" if low >= 1e-4 else f"{low:.2e}"                  # ".3g" and ".4f" until Version 44
            # Until Version 69 both ends of a range were printed to the nearest digit. Since Version 70 the least
            # probability is rounded downward and the greatest upward, so that the printed range holds them all
            # (tests/test_review69.py). The upper ends of two rows read 0.124738 and 0.046195.
            top = math.ceil(max(ps) * 10 ** 6) / 10 ** 6
            rows.append(f"  b = {right_b}: {low} to {top:.6f}")
        self.assertEqual(lines[23:28], ["two-sided p that 1722 and b allow:"] + rows)
        self.assertEqual(rows, ["  b = 1727: 0.062500 to 0.864711", "  b = 1758: 2.91e-11 to 0.124739",
                                "  b = 1767: 5.68e-14 to 0.051495", "  b = 1768: 2.84e-14 to 0.046196"])
        self.assertNotIn("0.0000", "".join(lines[23:28]))                         # two rows of Version 43 began so
        self.assertEqual(printed(accuracy.main), text)


def fits(stamps, theta):
    """Whether the four events of an exchange with exact stamps can follow one another, if the clock of the
    host that replies is `theta` ahead of the clock of the host that sends."""
    a1, b2, b3, a4 = stamps
    return a1 <= b2 - theta <= b3 - theta <= a4


GRID = [Fraction(k, 4) for k in range(-40, 41)]                                 # offsets from -10 to 10 ms


def hosts(a_to_c):
    """Every pair of offsets on the grid, of B from A and of C from A, that all four exchanges allow."""
    good = []
    for of_b in GRID:
        if not all(fits(stamps, of_b) for stamps in exchange.EXCHANGES):
            continue
        for of_c in GRID:
            if fits(exchange.B_TO_C, of_c - of_b) and fits(a_to_c, of_c):
                good.append((of_b, of_c))
    return good


class AThirdHost(unittest.TestCase):
    def test_the_numbers_of_the_review(self):
        a_b = exchange.both([exchange.offset(*stamps) for stamps in exchange.EXCHANGES])
        b_c, direct = exchange.offset(*exchange.B_TO_C), exchange.offset(*exchange.A_TO_C)
        via_b = exchange.chain(a_b, b_c)
        joint = exchange.both([direct, via_b])
        self.assertEqual((a_b, b_c, direct), ((-6, Fraction(-7, 2)), (1, 2), (-3, 1)))
        self.assertEqual((via_b, joint), ((-5, Fraction(-3, 2)), (-3, Fraction(-3, 2))))
        self.assertEqual([exchange.gap(300, 299, span) for span in (direct, via_b, joint)],
                         [(-2, 2), (Fraction(1, 2), 4), (Fraction(1, 2), 2)])

    def test_as_a_system_of_inequalities(self):
        good = hosts(exchange.A_TO_C)
        of_c = sorted({c for _, c in good})
        self.assertEqual((of_c[0], of_c[-1]), (-3, Fraction(-3, 2)))             # the clock of C from that of A
        times = [(299 - c) - 300 for c in of_c]                                  # from the event on A to that on C
        self.assertEqual((min(times), max(times)), (Fraction(1, 2), 2))
        of_b = sorted({b for b, _ in good})
        self.assertEqual((of_b[0], of_b[-1]), (-5, Fraction(-7, 2)))             # C narrows B as well: -6 to -3.5 before
        alone = [c for c in GRID if fits(exchange.A_TO_C, c)]
        self.assertEqual((alone[0], alone[-1]), (-3, 1))
        self.assertTrue(min((299 - c) - 300 for c in alone) < 0 < max((299 - c) - 300 for c in alone))

    def test_a_contradiction(self):
        other = exchange.A_TO_C[:3] + (62,)
        self.assertEqual(hosts(other), [])                                       # no offsets fit all four exchanges
        direct = exchange.offset(*other)                                         # though each fits by itself
        a_b = exchange.both([exchange.offset(*stamps) for stamps in exchange.EXCHANGES])
        via_b = exchange.chain(a_b, exchange.offset(*exchange.B_TO_C))
        self.assertEqual((direct, via_b), ((0, 1), (-5, Fraction(-3, 2))))
        self.assertIsNone(exchange.both([direct, via_b]))
        self.assertEqual(direct[0] - via_b[1], Fraction(3, 2))                   # they miss each other by 1.5 ms
        self.assertEqual(exchange.gap(300, 299, direct), (-2, -1))               # and give the two events opposite orders
        out = printed(exchange.third, other).splitlines()
        self.assertEqual(out[2], "both        no offset fits")

    def test_added_end_to_end(self):
        self.assertEqual(exchange.chain((-6, -3.5), (1, 2)), (-5, Fraction(-3, 2)))
        self.assertEqual(exchange.chain((0.1, 0.2), (0.2, 0.3)), (Fraction(3, 10), Fraction(1, 2)))   # as printed
        self.assertNotEqual(0.1 + 0.2, 0.3)                                      # where floats give 0.30000000000000004
        origin = 1_700_000_000_000_000_000                                       # C counts from 1970, in ms here
        a_b = exchange.both([exchange.offset(*stamps) for stamps in exchange.EXCHANGES])
        a1, c2, c3, a4 = exchange.A_TO_C
        b1, d2, d3, b4 = exchange.B_TO_C
        direct = exchange.offset(a1, c2 + origin, c3 + origin, a4)
        via_b = exchange.chain(a_b, exchange.offset(b1, d2 + origin, d3 + origin, b4))
        joint = exchange.both([direct, via_b])
        self.assertEqual((joint[0] - origin, joint[1] - origin), (-3, Fraction(-3, 2)))
        self.assertEqual(exchange.gap(300, 299 + origin, joint), (Fraction(1, 2), 2))

    def test_the_transcript(self):
        text = expected("ch06_exchange")
        lines = text.splitlines()
        self.assertEqual(len(lines), 39)                                         # 21 in Version 43, 29 in 44
        rows = []
        for a_to_c in (exchange.A_TO_C, exchange.A_TO_C[:3] + (62,)):
            alone = [c for c in GRID if fits(a_to_c, c)]
            through = sorted({c for b in GRID if all(fits(s, b) for s in exchange.EXCHANGES)
                              for c in GRID if fits(exchange.B_TO_C, c - b)})
            together = sorted({c for _, c in hosts(a_to_c)})
            for name, cs in (("direct", alone), ("via B", through), ("both", together)):
                if not cs:
                    rows.append(f"{name:<12}no offset fits")
                    continue
                lo, hi = float(cs[0]), float(cs[-1])
                rows.append(f"{name:<12}{lo:>7.2f}{hi:>7.2f}{-1 - hi:>+8.2f}{-1 - lo:>+8.2f}")
        self.assertEqual(lines[21:29], ["C: offset from A and C after A by, ms"] + rows[:3]
                         + ["the last stamp of A at 62 ms for 65:"] + rows[3:])
        self.assertEqual(rows[2], "both          -3.00  -1.50   +0.50   +2.00")
        self.assertEqual(printed(exchange.main), text)


class TheTwentyFifthFigure(unittest.TestCase):
    def test_the_script(self):
        source = (ROOT / "figures" / "make_figures.py").read_text(encoding="utf-8")
        for piece in ("def third_host():", "c6x.chain(a_b, b_c)", "c6x.offset(*c6x.B_TO_C), c6x.offset(*c6x.A_TO_C)",
                      "c6x.gap(300, 299, span)", 'save(fig, "fig-06-4-third-host.png")',
                      "    calibration_age()\n    third_host()\n    three_host_bound()\n    stamps_at_fault()\n    aliasing()\n"):
            self.assertIn(piece, source)
        self.assertEqual(source.count("\n    save(fig, "), 32)                   # twenty-four in Version 43

    def test_what_it_shows(self):
        # The three intervals of its right half lie within the axis, and the labels of its arrows are these.
        for lo, hi in ((-2, 2), (0.5, 4), (0.5, 2)):
            self.assertTrue(-2.6 < lo < hi < 4.5)
        said = [" to ".join(f"{float(x):+g}" for x in span) for span in ((-6, -3.5), (1, 2), (-3, 1))]
        self.assertEqual(said, ["-6 to -3.5", "+1 to +2", "-3 to +1"])


if __name__ == "__main__":
    unittest.main()
