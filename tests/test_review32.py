"""Regression tests added after the review of Version 32.

The review found that the order-statistic interval for a quantile summed floating-point binomial probabilities to
one tail and took the other from 1, so that a coverage which two ranks attain exactly could lose one limit or
both, and a very high coverage could be given a rank too early; that two functions of Chapter 20 passed twice over
an argument and gave another answer for an iterator than for a list; and that the comparison of Chapter 17 ignored
one list given without the other. It proposed wording for the passage on which the two models have the same total
from different answers, and an example with passages of different lengths.

Each test computes what it expects independently of the code under test: binomial tails as sums of fractions with
math.comb, ranks from their definition, the method of Version 32 rebuilt beside them, a seeded simulation of
coverage, and counts taken from the rows of the listing.
"""
import contextlib
import io
import math
import random
import re
import unittest
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

from examples import ch02_targets as targets
from examples import ch09_dependent_tail as dependent
from examples import ch17_accuracy as chapter17
from examples import ch20_error_bars as bars
from swmetro import core, stats

ROOT = Path(__file__).resolve().parent.parent


def expected(name):
    return (ROOT / "expected" / f"{name}.txt").read_text(encoding="utf-8")


def printed(function, *args, **kwargs):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        function(*args, **kwargs)
    return out.getvalue()


def below(n, q, r):
    """P(B < r) for B binomial with n and q: the chance that the r-th of n sorted draws lies above the
    q-quantile. A sum of fractions."""
    q = Fraction(q)
    return sum(Fraction(math.comb(n, i)) * q ** i * (1 - q) ** (n - i) for i in range(r))


def ranks_by_definition(n, q, p):
    """The largest rank with P(B < r) <= a, the smallest with P(B >= r) <= a, None where there is none, and the
    coverage of the two."""
    a = (1 - Fraction(p)) / 2
    lows = [r for r in range(1, n + 1) if below(n, q, r) <= a]
    highs = [r for r in range(1, n + 1) if 1 - below(n, q, r) <= a]
    lo, hi = (max(lows) if lows else None), (min(highs) if highs else None)
    cover = 1 - (below(n, q, lo) if lo else 0) - (1 - below(n, q, hi) if hi else 0)
    return lo, hi, cover


def version_32(xs, q, p=0.95):
    """quantile_interval() as Version 32 had it: floats summed to the lower tail, the upper taken from 1."""
    n, a = len(xs), (1 - p) / 2
    logq, log1q = math.log(q), math.log1p(-q)
    cdf, acc = [], 0.0
    lg = math.lgamma
    for k in range(n + 1):
        ln = lg(n + 1) - lg(k + 1) - lg(n - k + 1)
        acc += math.exp(ln + k * logq + (n - k) * log1q)
        cdf.append(acc)
    lo = [r for r in range(1, n + 1) if cdf[r - 1] <= a]
    hi = [r for r in range(1, n + 1) if 1 - cdf[r - 1] <= a]
    return (xs[lo[-1] - 1] if lo else None, xs[hi[0] - 1] if hi else None)


class TheRanksOfAQuantileInterval(unittest.TestCase):
    def test_the_table_of_the_review(self):
        table = ((3, 0.75, (None, 3), (1, 3)), (8, 0.99, (1, 8), (1, 8)),
                 (8, 127 / 128, (None, None), (1, 8)), (8, 0.995, (None, None), (None, None)))
        for n, p, was, want in table:
            xs = range(1, n + 1)
            self.assertEqual(version_32(xs, 0.5, p), was, (n, p))
            self.assertEqual(ranks_by_definition(n, Fraction(1, 2), stats.exact_q(p))[:2], want)
            self.assertEqual(stats.quantile_interval(xs, 0.5, p), want, (n, p))
        # The range of eight readings misses the median only if all eight fall on one side of it.
        self.assertEqual(stats.quantile_ranks(8, 0.5, 127 / 128), (1, 8, 1 - 2 / 256))
        self.assertEqual(stats.quantile_ranks(8, 0.5, 0.995), (None, None, 1.0))

    def test_a_coverage_that_two_ranks_attain(self):
        # For the median of n readings the pair of ranks k and n + 1 - k covers with 1 less twice the chance
        # of fewer than k below. Asked for exactly that, the pair is returned; asked for a little more, the
        # next pair inward is not, and at the outermost pair there is no limit.
        lost = 0
        for n in range(2, 41):
            for k in range(1, n // 2 + 1):
                tail = below(n, Fraction(1, 2), k)
                p = 1 - 2 * tail
                self.assertEqual(stats.quantile_ranks(n, Fraction(1, 2), p), (k, n + 1 - k, float(p)), (n, k))
                more = p + Fraction(1, 10 ** 30)
                want = (k - 1, n + 2 - k) if k > 1 else (None, None)
                self.assertEqual(stats.quantile_ranks(n, Fraction(1, 2), more)[:2], want, (n, k))
                lost += version_32(range(1, n + 1), 0.5, float(p)) != (k, n + 1 - k)
        self.assertGreater(lost, 100)                         # of 400 such requests, Version 32 lost a limit in these

    def test_ranks_from_their_definition(self):
        rng = random.Random(3201)
        seen = set()
        for _ in range(400):
            n = rng.randint(0, 45)
            q = Fraction(rng.choice((1, 1, 2, 3, 5, 9, 19, 99)), rng.choice((2, 4, 10, 20, 100)))
            if not 0 < q < 1:
                continue
            p = rng.choice((Fraction(19, 20), Fraction(9, 10), Fraction(1, 2), Fraction(999, 1000), Fraction(0), Fraction(1),
                            1 - 2 * below(max(n, 1), q, rng.randint(0, max(n, 1)))))
            if not 0 <= p <= 1:
                continue
            lo, hi, cover = ranks_by_definition(n, q, p)
            got = stats.quantile_ranks(n, q, p)
            self.assertEqual(got, (lo, hi, float(cover)), (n, q, p))
            self.assertGreaterEqual(cover, p)
            # The same question for the draws above the quantile gives the same pair from the other end.
            mirror = stats.quantile_ranks(n, 1 - q, p)
            self.assertEqual(mirror[:2], (hi and n + 1 - hi, lo and n + 1 - lo))
            self.assertAlmostEqual(mirror[2], got[2], places=15)
            xs = [rng.random() for _ in range(n)]
            xs.sort()
            self.assertEqual(stats.quantile_interval(xs, q, p), (xs[lo - 1] if lo else None, xs[hi - 1] if hi else None))
            seen.add((lo is None, hi is None, 2 * q > 1))
        # Every case of a missing limit that can occur, on both sides of 1/2. A quantile below 1/2 cannot
        # lack only its upper limit, nor one above 1/2 only its lower: the far end of the sample is the less
        # likely to be passed.
        self.assertEqual(seen, {(False, False, False), (False, False, True), (True, True, False), (True, True, True),
                                (True, False, False), (False, True, True)})

    def test_a_limit_given_too_early(self):
        # 5000 readings and a coverage of 1 - 1e-12 for the median. Version 32 took the upper tail from 1 and
        # gave the ranks 2248 and 2746, which miss more often than was asked.
        n, p = 5000, 1 - 1e-12
        self.assertEqual(stats.exact_q(p), 1 - Fraction(1, 10 ** 12))
        self.assertEqual(version_32(range(1, n + 1), 0.5, p), (2248, 2746))
        lo, hi, cover = stats.quantile_ranks(n, 0.5, p)
        self.assertEqual((lo, hi), (2248, 2753))
        whole = 2 ** n
        cumulative, sums = 0, [0]
        for i in range(n + 1):
            cumulative += math.comb(n, i)
            sums.append(cumulative)                           # sums[r] = whole * P(B < r)

        def miss(low, high):
            return Fraction(sums[low] + whole - sums[high], whole)
        self.assertEqual((f"{float(miss(2248, 2746)):.4g}", f"{float(miss(2248, 2753)):.4g}"), ("2.277e-12", "8.821e-13"))
        half = Fraction(1, 2 * 10 ** 12)
        self.assertLessEqual(Fraction(sums[2248], whole), half)
        self.assertGreater(Fraction(sums[2249], whole), half)                    # 2248 is the largest lower rank
        self.assertLessEqual(Fraction(whole - sums[2753], whole), half)
        self.assertGreater(Fraction(whole - sums[2752], whole), half)            # 2753 is the smallest upper rank
        self.assertEqual(cover, float(1 - miss(2248, 2753)))

    def test_the_latencies_of_chapter_2(self):
        xs = targets.record(400, random.Random(2))
        lo, hi, cover = stats.quantile_ranks(400, 0.9)
        self.assertEqual((lo, hi, f"{100 * cover:.4f}"), (348, 372, "95.4773"))
        self.assertEqual(ranks_by_definition(400, Fraction(9, 10), Fraction(19, 20))[:2], (348, 372))
        self.assertEqual(stats.quantile_interval(xs, 0.9), (xs[347], xs[371]))
        self.assertEqual(stats.quantile_interval(xs, 0.9), version_32(xs, 0.9))            # the published interval stands
        text = expected("ch02_targets")
        self.assertEqual(text.splitlines()[2:4], ["p90 of the process, 95 % [158.0, 174.4]",
                                                  "  ranks 348 and 372, coverage 95.48 %"])
        self.assertEqual((len(text.splitlines()), printed(targets.main)), (7, text))             # 6 lines in Version 32
        # The text: fewer than 36 values give the p90 no upper end at this level, since 0.9 ** 35 is above 2.5 %.
        self.assertEqual([stats.quantile_ranks(n, 0.9)[1] for n in (30, 35, 36, 37)], [None, None, 36, 37])
        self.assertTrue(Fraction(9, 10) ** 36 <= Fraction(1, 40) < Fraction(9, 10) ** 35)
        # Chapter 9: 500 values bound the p99.9 from below and not from above, as before.
        self.assertEqual(stats.quantile_ranks(500, 0.999)[:2], (498, None))
        self.assertIn("p99.9 from 500 independent values:\n", printed(dependent.main))
        self.assertEqual(printed(dependent.main), expected("ch09_dependent_tail"))

    def test_simulated_coverage(self):
        # 30 draws from a continuous distribution, the 0.3-quantile, 90 % asked: the pair of ranks covers the
        # quantile as often as the coverage that is returned with it says.
        lo, hi, cover = stats.quantile_ranks(30, 0.3, 0.9)
        self.assertEqual((lo, hi, cover), tuple(float(v) if i == 2 else v for i, v in enumerate(ranks_by_definition(30, Fraction(3, 10), Fraction(9, 10)))))
        rng = random.Random(3202)
        true, runs, hits = -math.log(0.7), 20_000, 0                             # the 0.3-quantile of the unit exponential
        for _ in range(runs):
            xs = sorted(rng.expovariate(1.0) for _ in range(30))
            hits += xs[lo - 1] <= true <= xs[hi - 1]
        self.assertGreater(cover, 0.9)
        self.assertAlmostEqual(hits / runs, cover, delta=4.5 * math.sqrt(cover * (1 - cover) / runs))

    def test_what_is_refused(self):
        for q in (0, 1, 0.0, 1.0):
            with self.assertRaisesRegex(ValueError, "need 0 < q < 1"):
                stats.quantile_ranks(10, q)
        for args in ((10, 1.5), (10, -0.1), (10, 0.5, 1.5), (10, 0.5, -0.1), (-1, 0.5), (2.5, 0.5)):
            with self.assertRaises(ValueError):
                stats.quantile_ranks(*args)
        self.assertEqual(stats.quantile_ranks(10.0, "0.5", "0.95"), stats.quantile_ranks(10, 0.5))
        self.assertEqual(stats.quantile_interval([], 0.5), (None, None))


def outcome(function, *args):
    """What a call returns and prints, or the kind of error that it raises."""
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            result = function(*args)
    except Exception as error:                               # noqa: BLE001 - the kind is what is compared
        return type(error).__name__
    return result, out.getvalue()


class ABenchmarkReadOnce(unittest.TestCase):
    def setUp(self):
        self.rows = bars.evaluate(random.Random(20))

    def test_the_table_of_the_review(self):
        want = printed(bars.exchanged, self.rows)
        self.assertIn("  by question\n    for A 9, for B 19, none 272\n    B ahead 0.04358, two-sided 0.08716\n", want)
        self.assertEqual(printed(bars.exchanged, iter(self.rows)), want)        # Version 32: for A 0, for B 0, none 0
        self.assertEqual(printed(bars.exchanged, (row for row in self.rows)), want)
        self.assertEqual(printed(bars.exchanged, tuple(self.rows)), want)
        self.assertTrue(expected("ch20_error_bars").endswith(want))
        self.assertEqual(printed(bars.main), expected("ch20_error_bars"))

    def test_four_differences(self):
        # Three for A and one for B, each of one answer. B ahead: one or more of the four for B, 15/16. A
        # ahead: one or none for B, 5/16. Two-sided: twice the smaller.
        diffs = [-1, -1, -1, 1]
        want = ([3, 1, 0], 15 / 16, 10 / 16)
        self.assertEqual(bars.tails(diffs), want)
        self.assertEqual(bars.tails(iter(diffs)), want)                          # Version 32: [0, 0, 0] and 1.0
        self.assertEqual(bars.tails(d for d in diffs), want)
        self.assertEqual(sorted(bars.by_passage(iter(self.rows))), sorted(bars.by_passage(self.rows)))

    def test_an_iterator_gives_the_same_answer_or_none(self):
        # Every function of the package and of the listings in which an argument that may be a sequence is
        # passed over more than once: an iterator gives what the list gives, or is refused. None gives
        # another answer without saying so.
        rng = random.Random(3203)
        xs = sorted(rng.gauss(100, 5) for _ in range(40))
        net = bars.by_passage(self.rows)
        fit = (1.0, 2.0, 0.0, 0.0, 0.1)
        # Six of eight cases on the side of their logit of 2: the loss has its minimum at T = 2 / ln 3. (Until
        # Version 35 this test had four pairs whose minimum lies below the search interval, at T = 0.16, and
        # fit_temperature() returned the end of the interval for them, 0.2, without saying so.)
        pairs = [(2.0, 1), (-2.0, 0)] * 3 + [(2.0, 0), (-2.0, 1)]
        from examples import ch10_coordinated_omission as c10, ch12_calibration as c12, ch13_variance_components as c13
        from examples import ch15_control_chart as c15, ch18_calibration as c18, ch22_budget as c22
        cases = [(bars.tails, [net]), (bars.exchanged, [self.rows]), (bars.by_passage, [self.rows]),
                 (bars.naive_se, [xs]), (bars.clustered_se, [xs, list(range(20)) * 2]),
                 (stats.exchange_exact, [net]), (stats.mean, [xs]), (stats.quantile, [xs, 0.5]),
                 (stats.linfit, [list(range(40)), xs]), (stats.cohen_kappa, [[1, 0, 1, 1], [1, 0, 0, 1]]),
                 (stats.bootstrap, [xs, max, 20]), (stats.quantile_interval, [xs, 0.5]),
                 (core.type_a, [xs]), (core.welch_satterthwaite, [[1.0, 2.0], [3, 4]]),
                 (c10.row, ["x", xs]), (c12.measure, [5.0, xs, fit]), (c13.components, [[xs[:5], xs[5:10], xs[10:15]]]),
                 (c15.xmr, [xs]), (c18.fit_temperature, [pairs]), (c22.from_raw, [xs])]
        same = refused = 0
        for function, args in cases:
            with_lists = outcome(function, *args)
            self.assertNotIsInstance(with_lists, str, function.__name__)
            with_iterators = outcome(function, *(iter(a) if isinstance(a, list) else a for a in args))
            if isinstance(with_iterators, str):
                self.assertEqual(with_iterators, "TypeError", function.__name__)
                refused += 1
            else:
                one, two = (re.sub(r" at 0x[0-9a-f]+", "", repr(x)) for x in (with_iterators, with_lists))
                self.assertEqual(one, two, function.__name__)
                same += 1
        # Until Version 36 the fit of the temperature passed over its data at every step of a search and
        # refused an iterator; it now reads the margins once: (9, 11) became (10, 10). Since Version 55
        # from_raw() of Listing 22.1 reads its durations into a list where it checks them, and an iterator
        # gives what the list gives: (11, 9). The budget it returns is an object, compared without its address.
        # Since Version 68 type_a() reads its indications into a list where it takes each as the number that
        # it is, see native(), and an iterator gives what the list gives: (12, 8).
        self.assertEqual((same, refused), (12, 8))


class OneListOfTwo(unittest.TestCase):
    def test_one_list_is_refused(self):
        a, b = chapter17.two_models(2000, random.Random(17))
        for kwargs in ({"b": [0]}, {"a": [1]}, {"a": a}, {"b": b}):             # Version 32: the demonstration; TypeError
            out = io.StringIO()
            with contextlib.redirect_stdout(out), self.assertRaisesRegex(ValueError, "both lists, or none"):
                chapter17.compare(**kwargs)
            self.assertEqual(out.getvalue(), "")
        with self.assertRaisesRegex(ValueError, "both lists, or none"):
            chapter17.compare([1])
        self.assertEqual(printed(chapter17.compare), printed(chapter17.compare, a, b))
        self.assertEqual(printed(chapter17.compare, b=b, a=a), printed(chapter17.compare))
        self.assertIn(printed(chapter17.compare), expected("ch17_accuracy"))    # a second pair follows it since Version 42
        self.assertEqual(len(expected("ch17_accuracy").splitlines()), 43)       # 12 until Version 41, 23 in 42, 27 in 43, 28 in 44


class WhatAPassageScoreConceals(unittest.TestCase):
    def setUp(self):
        self.by = defaultdict(list)
        for passage, a, b in bars.evaluate(random.Random(20)):
            self.by[passage].append((a, b))

    def test_the_passage_with_one_gained_and_one_lost(self):
        changed = [c for c, pairs in self.by.items() if any(a != b for a, b in pairs)]
        totals = [c for c, pairs in self.by.items() if sum(b - a for a, b in pairs)]
        self.assertEqual((len(self.by), len(changed), len(totals)), (60, 21, 20))
        self.assertEqual(sorted(set(changed) - set(totals)), [48])
        self.assertEqual(([a for a, _ in self.by[48]], [b for _, b in self.by[48]]), ([1, 1, 0, 0, 1], [1, 1, 1, 0, 0]))
        # 28 answers differ; the passages count 26 of them, 14 by one and 6 by two, and two cancel.
        discordant = sum(a != b for pairs in self.by.values() for a, b in pairs)
        by_passage = [abs(sum(b - a for a, b in pairs)) for pairs in self.by.values()]
        self.assertEqual((discordant, sum(by_passage), by_passage.count(1), by_passage.count(2)), (28, 26, 14, 6))
        self.assertEqual(discordant - sum(by_passage), 2)

    def test_two_passages_of_different_length(self):
        # The example of the text: (right, questions) for a passage of ten questions and one of two.
        model_a, model_b = ((4, 10), (2, 2)), ((6, 10), (1, 2))

        def by_question(model):
            return Fraction(sum(right for right, _ in model), sum(n for _, n in model))

        def by_passage(model):
            return sum(Fraction(right, n) for right, n in model) / len(model)
        points = [f"{100 * float(v):.2f}" for v in (by_question(model_a), by_question(model_b), by_question(model_b) - by_question(model_a),
                                                    by_passage(model_a), by_passage(model_b), by_passage(model_b) - by_passage(model_a))]
        self.assertEqual(points, ["50.00", "58.33", "8.33", "70.00", "55.00", "-15.00"])
        each = [100 * (Fraction(b[0], b[1]) - Fraction(a[0], a[1])) for a, b in zip(model_a, model_b)]
        self.assertEqual(each, [20, -50])
        self.assertEqual((sum(d > 0 for d in each), sum(d < 0 for d in each)), (1, 1))            # one passage won by each
        # With passages of one length the two weightings are one number, as in the listing.
        rows = bars.evaluate(random.Random(20))
        for column in (1, 2):
            pooled = Fraction(sum(r[column] for r in rows), len(rows))
            means = [Fraction(sum(pair[column - 1] for pair in pairs), len(pairs)) for pairs in self.by.values()]
            self.assertEqual(sum(means) / len(means), pooled)


if __name__ == "__main__":
    unittest.main()
