"""Regression tests added after the review of Version 33.

The review found that the weighted 95th percentile of Listing 8.4 summed its weights as floats, so that twenty
equal weights of 4/3 moved it from 100 ms to 5000 ms; that the number of values which a finite upper end of an
interval for a percentile needs was stated for equal tails only; and that the standard uncertainty of a count
estimated from a sampled trace, and the meaning of an empty record, depend on whether one retention decision is
made for each event or for each group of events.

Each test computes what it expects independently of the code under test: the weighted empirical distribution from
its definition in fractions, the method of Version 33 rebuilt beside it, powers of fractions, binomial sums, every
outcome of a small sampling design written out, and seeded simulations.
"""
import contextlib
import io
import itertools
import math
import random
import statistics
import sys
import unittest
from fractions import Fraction
from pathlib import Path

from examples import ch08_sampling as sampling
from examples import ch08_selection as selection
from examples import ch09_dependent_tail as dependent
from examples import ch23_mix as mix
from swmetro import stats

ROOT = Path(__file__).resolve().parent.parent


def expected(name):
    return (ROOT / "expected" / f"{name}.txt").read_text(encoding="utf-8")


def printed(function, *args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        function(*args)
    return out.getvalue()


def flat(text):
    return " ".join(text.split())


def by_definition(pairs, q):
    """The smallest value with at least a share q of the weight at or below it. Every value is tried."""
    pairs = [(x, Fraction(w)) for x, w in pairs]
    total = sum(w for _, w in pairs)
    for x in sorted({x for x, _ in pairs}):
        if sum(w for y, w in pairs if y <= x) >= Fraction(q) * total:
            return x
    raise AssertionError("no value has the share")


def version_33(pairs, plain=False):
    """The p95 of summary() as Version 33 had it: the weights added to a float, and compared with their total.
    For float weights the total is the sum as Python 3.12 and later form it, which is the exactly rounded one
    in these cases; `plain` gives the total of Python 3.11, added one weight at a time."""
    pairs = sorted(pairs)
    weights = [w for _, w in pairs]
    if all(isinstance(w, float) for w in weights):
        total = math.fsum(weights)
        if plain:
            total = 0.0
            for w in weights:
                total += w
        elif sys.version_info >= (3, 12):
            assert sum(weights) == total
    else:
        total = sum(weights)
    acc = 0.0
    for x, w in pairs:
        acc += w
        if 20 * acc >= 19 * total:
            break
    return x


NINETEEN_AND_ONE = [100] * 19 + [5000]


class EqualWeightsDoNotMoveAPercentile(unittest.TestCase):
    def test_the_table_of_the_review(self):
        table = ((1, 100), (4, 100), (4 / 3, 5000), (Fraction(4, 3), 5000))
        for weight, was in table:
            pairs = [(x, weight) for x in NINETEEN_AND_ONE]
            self.assertEqual(version_33(pairs), was, weight)
            self.assertEqual(by_definition(pairs, Fraction(19, 20)), 100)
            self.assertEqual(selection.summary(pairs)[1], 100, weight)
            self.assertEqual(stats.weighted_quantile(pairs, 0.95), 100, weight)
        # The fault with float weights needed the sum of Python 3.12: against the plain total of 3.11, which
        # is rounded as the running sum is, the comparison came out right. With fractions it failed on both.
        floats = [(x, 4 / 3) for x in NINETEEN_AND_ONE]
        self.assertEqual((version_33(floats), version_33(floats, plain=True)), (5000, 100))
        # One request in twenty is slow: the p95 is 100 ms, and anything above 95 % is the slow request.
        for weight in (1, 4 / 3, Fraction(4, 3), 0.1, 1e-300, 1e300, Fraction(1, 7)):
            pairs = [(x, weight) for x in NINETEEN_AND_ONE]
            self.assertEqual([stats.weighted_quantile(pairs, q) for q in (0, 0.5, 0.95, "0.9500001", 1)], [100, 100, 100, 5000, 5000])
            self.assertEqual(stats.weighted_quantile(pairs, 0.95), stats.ecdf_quantile(NINETEEN_AND_ONE, 0.95))

    def test_every_slow_request_and_three_quarters_of_the_fast(self):
        # 114 fast requests kept with probability 3/4 and 8 slow ones kept always: weighted counts 152 and 8,
        # and 152 of 160 is 95 % exactly.
        keep = Fraction(3, 4)
        pairs = [(100, 1 / keep)] * 114 + [(5000, 1)] * 8
        self.assertEqual((sum(w for x, w in pairs if x == 100), sum(w for x, w in pairs if x == 5000)), (152, 8))
        self.assertEqual(version_33(pairs), 5000)
        self.assertEqual((selection.summary(pairs)[1], by_definition(pairs, Fraction(19, 20))), (100, 100))
        whole = [(100, 4)] * 114 + [(5000, 3)] * 8                               # the same proportions in whole numbers
        self.assertEqual((selection.summary(whole)[1], version_33(whole)), (100, 100))
        # The reciprocal of the float 0.75 is a float below 4/3, and with it the fast requests weigh a hair
        # less than 95 %. The listing forms its weights from exact probabilities for this reason.
        self.assertLess(Fraction(1 / 0.75), Fraction(4, 3))
        rounded = [(100, 1 / 0.75)] * 114 + [(5000, 1)] * 8
        self.assertEqual((selection.summary(rounded)[1], by_definition(rounded, Fraction(19, 20))), (5000, 5000))
        self.assertIn("1 / 0.75 as a float is already below 4 / 3", flat(stats.weighted_quantile.__doc__))

    def test_one_factor_on_every_weight(self):
        rng = random.Random(3301)
        for _ in range(300):
            n = rng.randint(1, 25)
            pairs = [(rng.choice((10, 20, 20, 50, 100, 400)) + rng.randint(0, 3), Fraction(rng.randint(0, 12), rng.choice((1, 2, 3, 7))))
                     for _ in range(n)]
            if not sum(w for _, w in pairs):
                continue
            for q in (Fraction(1, 2), Fraction(19, 20), Fraction(rng.randint(0, 40), 40),
                      sum(w for _, w in pairs[:rng.randint(1, n)]) / sum(w for _, w in pairs)):
                if q > 1:
                    continue
                want = by_definition(pairs, q)
                self.assertEqual(stats.weighted_quantile(pairs, q), want, (pairs, q))
                for factor in (3, Fraction(4, 3), Fraction(1, 1000), 0.25, 2.0 ** 40):
                    scaled = [(x, w * Fraction(factor)) for x, w in pairs]
                    self.assertEqual(stats.weighted_quantile(scaled, q), want, (factor, q))
                self.assertEqual(stats.weighted_quantile(iter(pairs), q), want)
            # Whole weights are repetitions, and the convention is that of ecdf_quantile().
            whole = [(x, w.numerator) for x, w in pairs]
            if sum(w for _, w in whole):
                repeated = sorted(x for x, w in whole for _ in range(w))
                for q in (0.5, 0.9, 0.95, 1):
                    self.assertEqual(stats.weighted_quantile(whole, q), stats.ecdf_quantile(repeated, q))

    def test_the_listing_keeps_exact_probabilities(self):
        self.assertEqual((selection.KEEP, type(selection.KEEP)), (Fraction(1, 50), Fraction))
        rng = random.Random(8)
        rows = selection.requests(40_000, rng)
        sample = selection.kept(rows, rng)
        self.assertEqual(len(sample), 908)
        self.assertEqual({(type(p), p) for _, p in sample}, {(Fraction, Fraction(1, 50)), (Fraction, Fraction(1))})
        self.assertTrue(all(x > selection.SLOW for x, p in sample if p == 1))
        weighted = selection.summary([(x, 1 / p) for x, p in sample])
        self.assertEqual([type(v) for v in weighted], [float, float, float])
        self.assertEqual([f"{v:.1f}" for v in weighted], ["96.4", "214.9", "1.3"])
        self.assertEqual(weighted[1], by_definition([(x, 1 / p) for x, p in sample], Fraction(19, 20)))
        self.assertEqual(weighted[1], version_33([(x, 1 / float(p)) for x, p in sample]))     # 50 and 1: the printed p95 stands
        text = expected("ch08_selection")
        self.assertEqual((len(text.splitlines()), printed(selection.main)), (9, text))

    def test_the_percentile_of_chapter_23(self):
        pairs = [(x, 4 / 3) for x in NINETEEN_AND_ONE]
        self.assertEqual(mix.p95(pairs), 100)                                    # Version 33: 5000
        self.assertEqual(mix.p95([(x, Fraction(4, 3)) for x in NINETEEN_AND_ONE]), 100)
        self.assertEqual(mix.p95([(100.0, 1.0)] * 19 + [(5000.0, 1.0)]), 100.0)
        with self.assertRaisesRegex(ValueError, "no requests"):
            mix.p95([])
        with self.assertRaisesRegex(ValueError, "no requests"):
            mix.p95([(1.0, 0)])
        self.assertEqual(printed(mix.main), expected("ch23_mix"))

    def test_what_is_refused(self):
        for pairs in ([], [(1, 0)], [(1, 0), (2, 0.0)], [(1, -1), (2, 3)], [(1, 2), (2, -1)], [(1, -2)]):
            with self.assertRaisesRegex(ValueError, "need weights >= 0, not all zero"):
                stats.weighted_quantile(pairs, 0.5)
        for q in (-0.1, 1.5):
            with self.assertRaises(ValueError):
                stats.weighted_quantile([(1, 1)], q)
        # Until Version 50 this was 3, a value of no weight. The quantile of rank zero is the least of the values
        # that are there, as ecdf_quantile() has it, and a weight of zero says that the value is not.
        self.assertEqual(stats.weighted_quantile([(3, 0), (5, 2), (9, 0)], 0), 5)
        self.assertEqual(stats.weighted_quantile([(3, 0), (5, 2), (9, 0)], "0.000001"), 5)
        self.assertEqual(stats.weighted_quantile([(3, 0), (5, 2), (9, 0)], 1), 5)


def fewest(q, miss):
    """The smallest n with q ** n <= miss, and a check that n - 1 is not enough."""
    q, miss = Fraction(q), Fraction(miss)
    n = math.ceil(math.log(float(miss)) / math.log(float(q)))
    while q ** n > miss:
        n += 1
    while n > 1 and q ** (n - 1) <= miss:
        n -= 1
    return n


def below(n, q, r):
    """P(B < r) for B binomial with n and q, a sum of fractions."""
    q = Fraction(q)
    return sum(Fraction(math.comb(n, i)) * q ** i * (1 - q) ** (n - i) for i in range(r))


class TheLargestValueAsAnUpperBound(unittest.TestCase):
    TABLE = (("p90", Fraction(9, 10), 29, 36), ("p95", Fraction(19, 20), 59, 72),
             ("p99", Fraction(99, 100), 299, 368), ("p99.9", Fraction(999, 1000), 2995, 3688))

    def test_the_table_of_the_review(self):
        for name, q, one_sided, equal_tails in self.TABLE:
            self.assertEqual((fewest(q, Fraction(1, 20)), fewest(q, Fraction(1, 40))), (one_sided, equal_tails), name)
            self.assertEqual((dependent.enough(float(q), 0.05), dependent.enough(float(q), 0.025)), (one_sided, equal_tails))
            # The second column is where the interval of Chapter 2 first has an upper end.
            self.assertEqual([stats.quantile_ranks(n, q)[1] for n in (equal_tails - 1, equal_tails)], [None, equal_tails])
            self.assertLessEqual(q ** one_sided, Fraction(1, 20))
            self.assertGreater(q ** (one_sided - 1), Fraction(1, 20))
        nine = Fraction(9, 10)
        self.assertEqual([f"{100 * float(nine ** n):.4f}" for n in (29, 35, 36)], ["4.7101", "2.5032", "2.2528"])
        self.assertEqual(f"{100 * float(1 - Fraction(999, 1000) ** 500):.2f}", "39.36")

    def test_two_ranks_fixed_in_advance(self):
        # The 21st and the 29th of 29 values, for the p90: tails of 0.1550 % and 4.7101 %.
        q = Fraction(9, 10)
        low, high = below(29, q, 21), 1 - below(29, q, 29)
        self.assertEqual([f"{100 * float(v):.4f}" for v in (low, high, 1 - low - high)], ["0.1550", "4.7101", "95.1349"])
        self.assertEqual(high, q ** 29)
        # With equal tails the same 29 values have no upper end, and the lower end is taken further in.
        lower = max(r for r in range(1, 30) if below(29, q, r) <= Fraction(1, 40))
        self.assertEqual((lower, stats.quantile_ranks(29, q)[:2]), (23, (23, None)))

    def test_the_largest_of_29(self):
        # A seeded check of what the bound means: the largest of 29 draws lies above the p90 of the
        # distribution in 1 - 0.9 ** 29 of the samples.
        rng = random.Random(3302)
        true, runs = math.log(10), 20_000                                        # the p90 of the unit exponential
        above = sum(max(rng.expovariate(1.0) for _ in range(29)) > true for _ in range(runs))
        want = 1 - 0.9 ** 29
        self.assertAlmostEqual(above / runs, want, delta=4.5 * math.sqrt(want * (1 - want) / runs))

    def test_the_transcript(self):
        text = expected("ch09_dependent_tail")
        lines = text.splitlines()
        self.assertEqual(len(lines), 21)                          # 13 in Version 33
        self.assertEqual(lines[11][:34], "p99.9 from 500 independent values:")
        self.assertTrue(lines[12].endswith("upper bound none"))
        self.assertEqual(lines[13:], ["  the largest of them is above the", "  p99.9 in 39.36 % of samples",
                                      "values for an upper end at 95 %,", "one-sided and with equal tails:",
                                      "  p90       29     36", "  p95       59     72", "  p99      299    368", "  p99.9   2995   3688"])
        self.assertEqual(printed(dependent.main), text)
        with self.assertRaises(ValueError):
            dependent.enough(1.5, 0.05)


def every_outcome(n, group, rate):
    """Mean and variance of the estimate kept / rate, and the chance of an empty record, with every outcome of
    the n // group independent decisions written out."""
    rate = Fraction(rate)
    decisions = n // group
    mean = square = empty = Fraction(0)
    for outcome in itertools.product((0, 1), repeat=decisions):
        chance = math.prod(rate if kept else 1 - rate for kept in outcome)
        estimate = group * sum(outcome) / rate
        mean += chance * estimate
        square += chance * estimate ** 2
        empty += chance * (not any(outcome))
    return mean, square - mean ** 2, empty


class AnEmptyRecordUnderSampling(unittest.TestCase):
    def test_the_table_of_the_review(self):
        for group, sd, none in ((1, "186.15", "2.97"), (10, "588.64", "70.34")):
            self.assertEqual(f"{math.sqrt(group * 350 * (1 - 0.01) / 0.01):.2f}", sd)
            self.assertEqual(f"{100 * float(Fraction(99, 100) ** (350 // group)):.2f}", none)
            self.assertEqual(f"{100 * sampling.none_kept(350, 0.01, group):.2f}", none)
        self.assertEqual(sampling.none_kept(350, Fraction(1, 100), 10), Fraction(99, 100) ** 35)
        self.assertEqual((sampling.none_kept(350, 0), sampling.none_kept(350, 1), sampling.none_kept(0, 0.5)), (1, 0, 1))

    def test_every_outcome_of_a_small_design(self):
        # Twelve events at a rate of one in four, decided singly, in pairs, in threes and in sixes: the
        # estimate is unbiased in each, its variance is group * n * (1 - p) / p, and an empty record has the
        # chance (1 - p) ** (n / group).
        rate = Fraction(1, 4)
        for group in (1, 2, 3, 6):
            mean, variance, empty = every_outcome(12, group, rate)
            self.assertEqual((mean, variance, empty), (12, group * 12 * (1 - rate) / rate, (1 - rate) ** (12 // group)))
            self.assertEqual(sampling.none_kept(12, rate, group), empty)

    def test_simulated_decisions(self):
        rng = random.Random(3303)
        for group in (1, 10):
            estimates = [sampling.sampled_count(350, 0.01, rng, group) for _ in range(4000)]
            model = math.sqrt(group * 350 * 0.99 / 0.01)
            self.assertAlmostEqual(statistics.fmean(estimates), 350, delta=4.5 * model / math.sqrt(4000))
            self.assertAlmostEqual(statistics.stdev(estimates) / model, 1.0, delta=0.12)
            none = sum(e == 0 for e in estimates) / 4000
            want = 0.99 ** (350 // group)
            self.assertAlmostEqual(none, want, delta=4.5 * math.sqrt(want * (1 - want) / 4000))
            self.assertTrue(all(e * 0.01 % group == 0 or abs(e * 0.01 / group - round(e * 0.01 / group)) < 1e-9 for e in estimates))

    def test_the_transcript(self):
        text = expected("ch08_sampling")
        lines = text.splitlines()
        self.assertEqual(len(lines), 25)                          # 11 in Version 33, 14 in 34, 20 in 35
        self.assertEqual(lines[6:14], ["sampled trace, 350 true errors",
                                     "rate 0.01: mean 342, s 183, model 186", "  runs with no error kept: 3.2 %, model 3.0 %",
                                     "rate 0.10: mean 349, s 56, model 56", "  runs with no error kept: 0.0 %, model 0.0 %",
                                     "one decision for 10 errors,",
                                     "rate 0.01: mean 352, s 598, model 589", "  runs with no error kept: 70.5 %, model 70.3 %"])
        self.assertEqual(printed(sampling.main), text)
        # With one decision for each event the draws are those of Version 33.
        rng_a, rng_b = random.Random(8), random.Random(8)
        self.assertEqual([sampling.sampled_count(350, 0.01, rng_a) for _ in range(50)],
                         [sum(rng_b.random() < 0.01 for _ in range(350)) / 0.01 for _ in range(50)])

    def test_the_sixteenth_figure(self):
        source = (ROOT / "figures" / "make_figures.py").read_text(encoding="utf-8")
        for piece in ("from examples import ch08_sampling as c8s", "EMPTY_ERRORS, EMPTY_RATE = 350, 0.01",
                      "c8s.none_kept(EMPTY_ERRORS, rate, group)", 'save(fig, "fig-08-3-empty-record.png")',     # Figure 8.2 until Version 37
                      "    aliasing()\n    counter_readings()\n    empty_record()\n    omission()\n"):
            self.assertIn(piece, source)
        self.assertEqual(source.count("\n    save(fig, "), 32)         # fifteen in Version 33, sixteen until 35


if __name__ == "__main__":
    unittest.main()
