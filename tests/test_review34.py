"""Regression tests added after the review of Version 34.

The review found that a count sampled in groups lost its last group when that group was smaller than the others,
in the estimate and in the chance of an empty record alike; that enough() of Listing 9.2 never returned for
arguments that have no answer; that Listing 10.1 interpolates its percentiles, where the text gave every example
of its chapter to the inverse of the empirical distribution; and that a weighted percentile is unmoved by a
factor on the weights only when the factor is applied exactly. It asked for the number of units that an empty
record still allows, and for the standard uncertainty of a sampled count whose units are of unequal size.

Each test computes what it expects independently of the code under test: every outcome of a small sampling
design written out in fractions, powers of fractions searched one by one, a counter on the random draws, the
random stream of a transcript drawn again, and percentiles from their definitions.
"""
import contextlib
import inspect
import io
import itertools
import math
import random
import re
import statistics
import unittest
from fractions import Fraction
from pathlib import Path

from examples import ch08_sampling as sampling
from examples import ch09_dependent_tail as dependent
from examples import ch10_censoring, ch22_budget, ch22_runs, ch23_case, ch23_mix
from examples import ch10_coordinated_omission as omission
from swmetro import stats

ROOT = Path(__file__).resolve().parent.parent


def expected(name):
    return (ROOT / "expected" / f"{name}.txt").read_text(encoding="utf-8")


def printed(function, *args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        function(*args)
    return out.getvalue()


def every_outcome(sizes, rate):
    """Mean and variance of the estimate kept / rate, and the chance of an empty record, with every outcome of
    one independent decision for each unit written out."""
    rate = Fraction(rate)
    mean = square = empty = Fraction(0)
    for outcome in itertools.product((0, 1), repeat=len(sizes)):
        chance = math.prod(rate if kept else 1 - rate for kept in outcome)
        estimate = sum(m for m, kept in zip(sizes, outcome) if kept) / rate
        mean += chance * estimate
        square += chance * estimate ** 2
        empty += chance * (not any(outcome))
    return mean, square - mean ** 2, empty


def in_units(n, group):
    """The units of n events in groups: whole groups, and what is left over as one unit more."""
    sizes = [group] * (n // group)
    if n % group:
        sizes.append(n % group)
    return sizes


class Counted(random.Random):
    """A generator that counts its draws."""
    draws = 0

    def random(self):
        self.draws += 1
        return super().random()


class ALastSmallerUnit(unittest.TestCase):
    def test_the_table_of_the_review(self):
        # With every event kept the count is the count. Version 34: 350 for 355, and 0 for 5.
        self.assertEqual(sampling.sampled_count(355, 1, random.Random(8), group=10), 355)
        self.assertEqual(sampling.sampled_count(5, 1, random.Random(8), group=10), 5)
        self.assertEqual(sampling.none_kept(5, 1, group=10), 0)                  # Version 34: 1, a certain silence
        ninety_nine = Fraction(99, 100)
        for n, before, now in ((5, "100.00", "99.00"), (350, "70.34", "70.34"), (355, "70.34", "69.64")):
            self.assertEqual(f"{100 * float(ninety_nine ** (n // 10)):.2f}", before)            # the last group left out
            self.assertEqual(f"{100 * float(ninety_nine ** len(in_units(n, 10))):.2f}", now)
            self.assertEqual(f"{100 * sampling.none_kept(n, 0.01, group=10):.2f}", now)
            self.assertEqual(sampling.none_kept(n, Fraction(1, 100), 10), ninety_nine ** len(in_units(n, 10)))

    def test_full_retention_recovers_the_count(self):
        rng = random.Random(3401)
        for _ in range(400):
            n, group = rng.randint(0, 500), rng.randint(1, 40)
            for rate in (1, 1.0, Fraction(1)):
                self.assertEqual(sampling.sampled_count(n, rate, rng, group), n, (n, group))
            sizes = [rng.randint(1, 30) for _ in range(rng.randint(0, 12))]
            self.assertEqual(sampling.sampled_count(sum(sizes), 1, rng, sizes), sum(sizes))
            self.assertEqual(sampling.none_kept(n, 1, group), 0 if n else 1)
            self.assertEqual(sampling.none_kept(n, 0, group), 1)

    def test_both_functions_count_the_same_units(self):
        rng = Counted(3402)
        quarter = Fraction(1, 4)
        for _ in range(300):
            n, group = rng.randint(0, 300), rng.randint(1, 25)
            want = in_units(n, group)
            self.assertEqual(sampling.units(n, group), want)
            self.assertEqual(len(want), -(-n // group))
            rng.draws = 0
            estimate = sampling.sampled_count(n, 0.3, rng, group)
            self.assertEqual(rng.draws, len(want))                               # one decision for every unit
            self.assertEqual(sampling.none_kept(n, quarter, group), (1 - quarter) ** len(want))
            # What is kept is a sum of whole units.
            sums = {0}
            for m in set(want):
                sums |= {s + k * m for s in sums for k in range(1, want.count(m) + 1)}
            self.assertIn(round(estimate * 0.3), sums)
        self.assertEqual(sampling.units(0), [])
        self.assertEqual(sampling.units(7), [1] * 7)
        self.assertEqual(sampling.units(9, (4, 5)), [4, 5])

    def test_every_outcome_with_a_remainder(self):
        # Thirteen events in fives are units of 5, 5 and 3. At a rate of one in four the estimate is unbiased, its
        # variance is 3 * (25 + 25 + 9), and the record is empty with the chance (3/4) ** 3.
        mean, variance, empty = every_outcome([5, 5, 3], Fraction(1, 4))
        self.assertEqual((mean, variance, empty), (13, 177, Fraction(27, 64)))
        self.assertAlmostEqual(sampling.spread(13, 0.25, 5) ** 2, 177, places=9)
        self.assertEqual(sampling.none_kept(13, Fraction(1, 4), 5), Fraction(27, 64))
        for n in range(0, 11):
            for group in range(1, 7):
                for rate in (Fraction(1, 3), Fraction(4, 5)):
                    mean, variance, empty = every_outcome(in_units(n, group), rate)
                    self.assertEqual(mean, n)
                    self.assertAlmostEqual(sampling.spread(n, float(rate), group) ** 2, float(variance), places=9)
                    self.assertEqual(sampling.none_kept(n, rate, group), empty)

    def test_simulated_decisions_with_a_remainder(self):
        rng = random.Random(3403)
        sizes = in_units(355, 10)                                                # 35 tens and a five
        model = math.sqrt((1 - 0.2) / 0.2 * sum(m * m for m in sizes))
        estimates = [sampling.sampled_count(355, 0.2, rng, 10) for _ in range(4000)]
        self.assertAlmostEqual(statistics.fmean(estimates), 355, delta=4.5 * model / math.sqrt(4000))
        self.assertAlmostEqual(statistics.stdev(estimates) / model, 1.0, delta=0.08)
        self.assertAlmostEqual(sampling.spread(355, 0.2, 10), model, places=9)
        self.assertTrue(any(round(e * 0.2) % 10 == 5 for e in estimates))        # the five is kept in some runs

    def test_the_draws_of_version_34(self):
        # Where the group divides the count, the estimates are those of Version 34, draw for draw.
        for n, rate, group in ((350, 0.01, 1), (350, 0.1, 1), (350, 0.01, 10), (120, 0.25, 8)):
            rng_a, rng_b = random.Random(34), random.Random(34)
            now = [sampling.sampled_count(n, rate, rng_a, group) for _ in range(60)]
            then = [group * sum(rng_b.random() < rate for _ in range(n // group)) / rate for _ in range(60)]
            self.assertEqual(now, then)
            self.assertEqual(sampling.spread(n, rate, group), math.sqrt(group * n * (1 - rate) / rate))

    def test_what_is_refused(self):
        rng = random.Random(1)
        for group in (0, -1, 2.5, "ten", None, [5, 0], [5, 4], [], [2.5, 7.5]):
            with self.assertRaises(ValueError, msg=repr(group)):
                sampling.sampled_count(10, 0.5, rng, group)
            with self.assertRaises(ValueError, msg=repr(group)):
                sampling.none_kept(10, 0.5, group)
        for n in (-1, 2.5):
            with self.assertRaises(ValueError):
                sampling.sampled_count(n, 0.5, rng)
        for rate in (0, -0.1, 1.5):
            with self.assertRaises(ValueError):                                  # nothing can be divided by a rate of 0
                sampling.sampled_count(10, rate, rng)
        for rate in (-0.1, 1.5):
            with self.assertRaises(ValueError):
                sampling.none_kept(10, rate)


class UnitsOfUnequalSize(unittest.TestCase):
    def test_the_table_of_the_review(self):
        even, uneven = [10] * 35, [180] + [5] * 34
        self.assertEqual(sampling.UNEVEN, uneven)
        for sizes, want in ((even, "588.64"), (uneven, "1814.32")):
            self.assertEqual((sum(sizes), len(sizes)), (350, 35))
            self.assertEqual(f"{math.sqrt(99 * sum(m * m for m in sizes)):.2f}", want)          # (1 - p) / p is 99
            self.assertEqual(f"{sampling.spread(350, 0.01, sizes):.2f}", want)
            self.assertEqual(f"{100 * sampling.none_kept(350, 0.01, sizes):.2f}", "70.34")
            self.assertEqual(sampling.none_kept(350, Fraction(1, 100), sizes), Fraction(99, 100) ** 35)
        self.assertEqual(sampling.spread(350, 0.01, even), sampling.spread(350, 0.01, 10))

    def test_every_outcome_of_a_small_design(self):
        # Units of 6, 1 and 1 at a rate of one in three: variance 2 * (36 + 1 + 1), against 2 * 3 * 64 / 9 for three
        # units of 8/3 each, which no design has, and 2 * 8 for eight single events.
        mean, variance, empty = every_outcome([6, 1, 1], Fraction(1, 3))
        self.assertEqual((mean, variance, empty), (8, 76, Fraction(8, 27)))
        self.assertAlmostEqual(sampling.spread(8, 1 / 3, [6, 1, 1]) ** 2, 76, places=9)
        self.assertEqual(sampling.none_kept(8, Fraction(1, 3), [6, 1, 1]), Fraction(8, 27))
        self.assertEqual(every_outcome([1] * 8, Fraction(1, 3))[1], 16)
        rng = random.Random(3404)
        for _ in range(40):
            sizes = [rng.randint(1, 9) for _ in range(rng.randint(1, 8))]
            rate = Fraction(rng.randint(1, 9), 10)
            mean, variance, empty = every_outcome(sizes, rate)
            self.assertEqual(mean, sum(sizes))
            self.assertEqual(variance, (1 - rate) / rate * sum(m * m for m in sizes))
            self.assertAlmostEqual(sampling.spread(sum(sizes), float(rate), sizes) ** 2 / float(variance), 1, places=9)
            self.assertEqual(sampling.none_kept(sum(sizes), rate, sizes), empty)

    def test_the_transcript(self):
        text = expected("ch08_sampling")
        lines = text.splitlines()
        self.assertEqual(len(lines), 25)                                         # 14 in Version 34, 20 in Version 35
        self.assertEqual(printed(sampling.main), text)
        # The three lines for the uneven units, from the random stream drawn again: the generator of the listing
        # has by then made 2000 runs of 350, 350 and 35 decisions.
        rng = random.Random(8)
        for decisions in (350, 350, 35):
            for _ in range(2000 * decisions):
                rng.random()
        sizes = [180] + [5] * 34
        runs = [sum(m for m in sizes if rng.random() < 0.01) / 0.01 for _ in range(2000)]
        model = math.sqrt(99 * sum(m * m for m in sizes))
        self.assertEqual(lines[14:17], [
            "one trace of 180, 34 of 5,",
            f"rate 0.01: mean {statistics.fmean(runs):.0f}, s {statistics.stdev(runs):.0f}, model {model:.0f}",
            f"  runs with no error kept: {100 * runs.count(0) / 2000:.1f} %, model {100 * 0.99 ** 35:.1f} %"])
        self.assertEqual(lines[15], "rate 0.01: mean 382, s 2020, model 1814")
        # The large trace is kept in about one run in a hundred, and the standard deviation follows that number.
        large = sum(e >= 18000 for e in runs)
        self.assertAlmostEqual(large, 20, delta=4.5 * math.sqrt(2000 * 0.01 * 0.99))
        self.assertAlmostEqual(statistics.stdev(runs) ** 2 / model ** 2, large / 20, delta=0.05)


class WhatAnEmptyRecordAllows(unittest.TestCase):
    def test_the_table_of_the_review(self):
        self.assertEqual((sampling.most_units(0.01), sampling.most_units(0.1)), (298, 28))
        self.assertEqual((10 * sampling.most_units(0.01), 10 * sampling.most_units(0.1)), (2980, 280))
        ninety_nine, nine = Fraction(99, 100), Fraction(9, 10)
        self.assertEqual([f"{100 * float(ninety_nine ** g):.4f}" for g in (298, 299)], ["5.0037", "4.9536"])
        self.assertTrue(nine ** 28 >= Fraction(1, 20) > nine ** 29)

    def test_the_definition(self):
        # The largest number of units for which an empty record keeps a chance of `miss` or more, searched one
        # unit at a time.
        rng = random.Random(3405)
        cases = [(Fraction(1, 2), Fraction(1, 4)), (Fraction(1, 2), Fraction(1, 8)), (Fraction(1, 10), Fraction(81, 100)),
                 (Fraction(2, 3), Fraction(1, 9)), (Fraction(99, 100), Fraction(1, 20)), (Fraction(1), Fraction(1, 2))]
        cases += [(Fraction(rng.randint(1, 60), 100), Fraction(rng.randint(1, 99), 100)) for _ in range(300)]
        for rate, miss in cases:
            g = 0
            while (1 - rate) ** (g + 1) >= miss:
                g += 1
            self.assertEqual(sampling.most_units(rate, miss), g, (rate, miss))
        self.assertEqual(sampling.most_units(Fraction(1, 2), Fraction(1, 4)), 2)  # the chance is reached exactly
        self.assertEqual(sampling.most_units(0.5, 0.25), 2)
        self.assertEqual(sampling.most_units(1), 0)                              # with every event kept, an empty record is none

    def test_the_powers_of_chapter_9(self):
        # One power less than the number of values whose largest bounds a percentile: 28 and 29, 298 and 299.
        for rate in (0.1, 0.05, 0.01, 0.001):
            self.assertEqual(sampling.most_units(rate), dependent.enough(1 - stats.exact_q(rate), 0.05) - 1)
        self.assertEqual(dependent.enough(0.9, 0.05), 29)

    def test_what_is_refused(self):
        with self.assertRaises(ValueError):
            sampling.most_units(0)                                               # a collector that keeps nothing
        for miss in (0, 1):
            with self.assertRaises(ValueError):
                sampling.most_units(0.01, miss)
        for rate in (-0.1, 1.5):
            with self.assertRaises(ValueError):
                sampling.most_units(rate)

    def test_the_transcript(self):
        self.assertEqual(expected("ch08_sampling").splitlines()[21:24], [
            "an empty record: at 95 %, at most", "  rate 0.01: 298 units", "  rate 0.10:  28 units"])


class ARunOfEventsThatAllHappen(unittest.TestCase):
    def test_against_a_search_one_by_one(self):
        rng = random.Random(3406)
        cases = [(Fraction(1, 2), Fraction(1, 2)), (Fraction(1, 2), Fraction(1, 4)), (Fraction(1, 2), Fraction(1, 5)),
                 (Fraction(9, 10), Fraction(81, 100)), (Fraction(9, 10), Fraction(1, 20)), (Fraction(0), Fraction(0)),
                 (Fraction(0), Fraction(1, 3)), (Fraction(1), Fraction(1)), (Fraction(1, 3), Fraction(1))]
        cases += [(Fraction(rng.randint(1, 97), 100), Fraction(rng.randint(1, 100), 1000)) for _ in range(400)]
        for q, miss in cases:
            n = 1
            while q ** n > miss:
                n += 1
            self.assertEqual(stats.run_length(q, miss), n, (q, miss))
            self.assertEqual(dependent.enough(q, miss), n)
        self.assertEqual(stats.run_length(0.9, 0.05), 29)                        # floats are read as their decimals
        self.assertEqual(stats.run_length("0.9", "0.05"), 29)

    def test_no_finite_number(self):
        # Version 34 never returned for these.
        for q, miss in ((1, 0.05), (0.9, 0), (1, 0), (Fraction(1), Fraction(999, 1000)), (Fraction(1, 10 ** 9), 0)):
            with self.assertRaises(ValueError, msg=(q, miss)):
                dependent.enough(q, miss)
            with self.assertRaises(ValueError, msg=(q, miss)):
                stats.run_length(q, miss)
        for q, miss in ((1.5, 0.05), (-0.1, 0.05), (0.9, 1.5), (0.9, -0.1)):
            with self.assertRaises(ValueError, msg=(q, miss)):
                stats.run_length(q, miss)

    def test_a_long_run(self):
        # The answer is bracketed exactly, and it agrees with the logarithms to the unit that they can settle.
        for q, miss in ((Fraction(9999, 10000), Fraction(1, 40)), (Fraction(99999, 100000), Fraction(1, 20)),
                        (Fraction(999, 1000), Fraction(1, 10 ** 30))):
            n = stats.run_length(q, miss)
            self.assertTrue(q ** n <= miss < q ** (n - 1))
            self.assertLessEqual(abs(n - math.log(miss) / math.log(q)), 1)

    def test_the_table_of_chapter_9_stands(self):
        text = expected("ch09_dependent_tail")
        self.assertEqual(len(text.splitlines()), 21)
        self.assertEqual(text.splitlines()[-4:], ["  p90       29     36", "  p95       59     72",
                                                 "  p99      299    368", "  p99.9   2995   3688"])
        self.assertEqual(printed(dependent.upper_ends).splitlines(), text.splitlines()[-8:])


def interpolated(xs, q):
    """The convention of quantile(): the rank q * (n - 1), counted from 0, between two sorted values."""
    position = Fraction(q) * (len(xs) - 1)
    k = position.numerator // position.denominator
    if k + 1 >= len(xs):
        return Fraction(xs[-1])
    return Fraction(xs[k]) + (position - k) * (Fraction(xs[k + 1]) - Fraction(xs[k]))


def inverse(xs, q):
    """The smallest value with at least a share q of the values at or below it. Every value is tried."""
    for x in sorted(set(xs)):
        if sum(y <= x for y in xs) >= Fraction(q) * len(xs):
            return x
    raise AssertionError("no value has the share")


class TheConventionOfListing101(unittest.TestCase):
    def setUp(self):
        self.measured = omission.closed_loop()
        self.filled = omission.back_fill(self.measured)
        self.intended = omission.open_loop()

    def test_the_table_of_the_review(self):
        rows = ((self.filled, Fraction(99, 100), "701.100", 701), (self.intended, Fraction(999, 1000), "974.009", 974),
                (self.measured, Fraction(9997, 10000), "261.300", 1001))
        for xs, q, by_interpolation, by_rank in rows:
            self.assertEqual(f"{float(interpolated(xs, q)):.3f}", by_interpolation)
            self.assertEqual(f"{stats.quantile(xs, float(q)):.3f}", by_interpolation)
            self.assertEqual(inverse(xs, q), by_rank)
            self.assertEqual(stats.ecdf_quantile(xs, float(q)), by_rank)

    def test_an_objective_of_500_ms(self):
        xs = self.measured
        self.assertEqual((len(xs), xs.count(1.0), xs.count(1001.0)), (5800, 5798, 2))
        self.assertEqual(f"{100 * sum(x <= 500 for x in xs) / len(xs):.4f}", "99.9655")
        self.assertLess(Fraction(5798, 5800), Fraction(9997, 10000))             # fewer than 99.97 % within 500 ms
        self.assertGreater(inverse(xs, Fraction(9997, 10000)), 500)
        self.assertLess(interpolated(xs, Fraction(9997, 10000)), 500)
        position = Fraction(9997, 10000) * 5799                                  # 26.03 % of the way from 1 to 1001 ms
        self.assertEqual(f"{100 * float(position - 5797):.2f}", "26.03")

    def test_the_printed_table_shows_no_difference(self):
        # To the whole millisecond of Transcript 10.1 the two conventions agree in all twelve cells.
        for xs in (self.measured, self.filled, self.intended):
            for q in (Fraction(1, 2), Fraction(99, 100), Fraction(999, 1000), Fraction(9999, 10000)):
                self.assertEqual(f"{float(interpolated(xs, q)):.0f}", f"{inverse(xs, q):.0f}")

    def test_the_transcript(self):
        text = expected("ch10_coordinated_omission")
        lines = text.splitlines()
        self.assertEqual(lines[:4], ["latency, ms     n    p50    p99  p99.9 p99.99 mean",
                                     "as measured  5800      1      1      1   1001  1.3",
                                     "back-filled  6000      1    701    971   1001 17.8",
                                     "intended     6000      1    731    974   1001 19.7"])
        want = float(interpolated(self.measured, Fraction(9997, 10000)))
        self.assertEqual(lines[4:6], [f"p99.97 as measured: interpolated {want:.1f} ms,",
                                     f"  inverse of the empirical CDF {inverse(self.measured, Fraction(9997, 10000)):.0f} ms"])
        self.assertEqual(lines[4], "p99.97 as measured: interpolated 261.3 ms,")
        self.assertEqual(printed(omission.main), text)

    def test_which_examples_interpolate(self):
        # Listing 10.1 interpolates. Listing 10.2 and the objectives of Chapters 22 and 23 take the inverse of
        # the empirical distribution, with or without weights.
        interpolating = re.compile(r"(?<![\w.])quantile\(")
        self.assertTrue(interpolating.search(inspect.getsource(omission)))
        for module in (ch10_censoring, ch22_budget, ch22_runs, ch23_case, ch23_mix):
            source = inspect.getsource(module)
            self.assertIsNone(interpolating.search(source), module.__name__)
            self.assertTrue(re.search(r"\b(ecdf|weighted)_quantile\(", source), module.__name__)


class AFactorOnTheWeights(unittest.TestCase):
    def test_the_example_of_the_review(self):
        pairs = [(100, 19), (5000, 1)]
        exact = [(x, w * Fraction(7, 10)) for x, w in pairs]
        rounded = [(x, w * 0.7) for x, w in pairs]
        self.assertEqual(stats.weighted_quantile(pairs, 0.95), 100)
        self.assertEqual(stats.weighted_quantile(exact, 0.95), 100)
        self.assertEqual(stats.weighted_quantile(rounded, 0.95), 5000)
        # The function is right about the weights that it receives: 19 * 0.7 as a float is below nineteen times
        # the float 0.7, and the share of the first value is below 95 %.
        heavy, light = Fraction(19 * 0.7), Fraction(1 * 0.7)
        self.assertLess(heavy, 19 * light)
        self.assertLess(heavy / (heavy + light), Fraction(19, 20))
        self.assertEqual((repr(19 * 0.7), repr(1 * 0.7)), ("13.299999999999999", "0.7"))

    def test_factors_that_are_applied_exactly(self):
        rng = random.Random(3407)
        for _ in range(200):
            pairs = [(rng.randint(1, 50), rng.randint(1, 20)) for _ in range(rng.randint(1, 12))]
            q = Fraction(rng.randint(1, 20), 20)
            want = inverse([x for x, w in pairs for _ in range(w)], q)
            for factor in (Fraction(7, 10), Fraction(1, 3), 0.5, 4.0, 2.0 ** -30):   # powers of two are exact in floats
                scaled = [(x, w * factor) for x, w in pairs]
                self.assertEqual(stats.weighted_quantile(scaled, q), want, (pairs, q, factor))


if __name__ == "__main__":
    unittest.main()
