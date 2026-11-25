"""Regression tests added after the review of Version 60.

The review found that a reference mix in which two classes name one iterator counted the second class as empty,
so that the result depended on the order of the classes; that an iterator under two observed classes was read as
empty under the second; that Result.prob_below() gave two probabilities, on two sides of a decision, for one
limit written as a whole number and as a float, and raised for other mixtures of numeric types, as interval(),
expanded(), report() and k() did; that a bound of Listing 23.1 formed by subtracting or adding two floats could
be rounded across a margin before it was compared with it; that the lines which report the bound by ranks
raised an exception for a trace too short for one of its ends; and that the first line of Transcript 14.1 left
open what happens to a reading of exactly 204.94 ms. It proposed an exercise with the same requests reported by
two configurations.

Each test computes what it expects independently of the code under test: weights and shares by counting, the
normal distribution by the statistics module, Student's distribution by integrating its density, sums and
differences in fractions, and the bootstrap by its own draws from the random module.
"""
import contextlib
import io
import itertools
import math
import random
import statistics
import unittest
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from statistics import NormalDist

from examples import ch14_decisions as decisions
from examples import ch23_case as case
from examples import ch23_mix as mix
from swmetro import Result, stats

ROOT = Path(__file__).resolve().parent.parent
D = Decimal
KINDS = (int, float, Decimal, Fraction)


def expected(name):
    return (ROOT / "expected" / f"{name}.txt").read_text(encoding="utf-8")


def printed(function, *args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        function(*args)
    return out.getvalue()


def student_cdf(t, dof, steps=4000):
    """P(T <= t) for t >= 0, by Simpson's rule on the density of Student's distribution."""
    c = math.gamma((dof + 1) / 2) / (math.sqrt(dof * math.pi) * math.gamma(dof / 2))
    h = t / steps
    ys = [c * (1 + (i * h) ** 2 / dof) ** (-(dof + 1) / 2) for i in range(steps + 1)]
    return 0.5 + h / 3 * (ys[0] + ys[-1] + 4 * sum(ys[1:-1:2]) + 2 * sum(ys[2:-1:2]))


class OneStreamTwoClasses(unittest.TestCase):
    REC = {"fast": [100] * 18, "slow": [500] * 2}

    def facts(self, reference):
        pairs = mix.weighted(dict(self.REC), reference())
        share = mix.within(dict(self.REC), reference(), 200)
        return sum(w for _, w in pairs), mix.p95(pairs), share, mix.verdict(*share)

    def test_the_rows_of_the_review(self):
        # Ten requests of each class at the reference: half of the weight is at 100 ms and half at 500 ms.
        # The 95th percentile is 500 ms and ten of twenty requests are within 200 ms.
        want = (20, 500, (Fraction(1, 2),) * 2, "missed")
        for order in (("fast", "slow"), ("slow", "fast")):
            forms = {"separate ranges": lambda: {name: range(10) for name in order},
                     "one iterator": lambda: dict.fromkeys(order, iter(range(10))),     # Version 60: 10, and met
                     "one generator": lambda: dict.fromkeys(order, (0 for _ in range(10))),
                     "one list": lambda: dict.fromkeys(order, [0] * 10),
                     "separate iterators": lambda: {name: iter(range(10)) for name in order}}
            for name, reference in forms.items():
                self.assertEqual(self.facts(reference), want, (order, name))

    def test_one_count_for_every_name(self):
        stream = iter(range(7))
        held, sizes = mix.both(dict(self.REC), {"a": stream, "b": stream, "c": stream, "d": range(3)})
        self.assertEqual({name: len(xs) for name, xs in sizes.items()}, {"a": 7, "b": 7, "c": 7, "d": 3})
        # An object that the record holds and that two classes of the reference name as well.
        slow = iter([500, 500])
        rec = {"fast": [100] * 18, "slow": slow}
        held, sizes = mix.both(rec, {"fast": range(18), "slow": slow, "late": slow})
        self.assertEqual({name: len(xs) for name, xs in sizes.items()}, {"fast": 18, "slow": 2, "late": 2})
        self.assertEqual(held["slow"], [500, 500])
        # The copy of Version 59's review still counts all twenty.
        rec = {"fast": [100] * 18, "slow": iter([500, 500])}
        self.assertEqual(mix.within(rec, rec.copy(), 200), (Fraction(9, 10),) * 2)

    def test_the_transcript_is_what_it_was(self):
        self.assertEqual(printed(mix.main), expected("ch23_mix"))
        self.assertEqual(len(expected("ch23_mix").splitlines()), 108)


class OneStreamUnderTwoObservedClasses(unittest.TestCase):
    def test_it_is_refused_before_it_is_read(self):
        slow = iter([500, 500])
        rec = {"fast": [100] * 18, "slow_a": slow, "slow_b": slow}
        for call in (lambda: mix.classes(rec), lambda: mix.weighted(rec), lambda: mix.weighted(rec, rec),
                     lambda: mix.within(rec, rec, 200), lambda: mix.both(rec, {"fast": range(1)})):
            with self.assertRaises(ValueError):                                # Version 60: a total of 18, and met
                call()
        self.assertEqual(list(slow), [500, 500])                               # nothing was taken from it
        gen = (x for x in (500, 500))
        with self.assertRaises(ValueError):
            mix.classes({"a": gen, "b": gen})

    def test_what_is_not_such_a_stream(self):
        # Two iterators over equal values are two streams, and a list can be read twice.
        rec = {"fast": [100] * 18, "slow_a": iter([500]), "slow_b": iter([500])}
        self.assertEqual(sum(w for _, w in mix.weighted(rec)), 20)
        both = [500, 500]
        rec = {"fast": [100] * 18, "slow_a": both, "slow_b": both}
        self.assertEqual({name: len(xs) for name, xs in mix.classes(rec).items()},
                         {"fast": 18, "slow_a": 2, "slow_b": 2})
        self.assertEqual(mix.classes({"a": (), "b": (), "c": range(0)}), {"a": [], "b": [], "c": []})
        # Each class is still held to the check.
        with self.assertRaises(ValueError):
            mix.classes({"fast": [100], "odd": iter([False])})


class OneLimitOneProbability(unittest.TestCase):
    def test_the_time_stamp_of_the_review(self):
        limit = 10 ** 18
        r = Result(limit - 246_720, 150_000.0, "ns")
        want = NormalDist().cdf(246_720 / 150_000)
        self.assertEqual(f"{want:.9f}", "0.949994469")
        self.assertNotEqual(int(float(limit - 246_720)), limit - 246_720)      # a float does not hold the value
        for form in (limit, float(limit), D(limit), Fraction(limit), D("1E18"), 1e18):
            got = r.prob_below(form)
            self.assertAlmostEqual(got, want, places=14, msg=repr(form))       # Version 60, float: 0.950038462
            self.assertLess(got, 0.95, repr(form))                             # undecided by the rule of Chapter 14
            self.assertEqual(got, r.prob_below(limit))
        # The float that Version 60 subtracted lies 64 ns below the value.
        self.assertEqual(int(float(limit - 246_720)) - (limit - 246_720), -64)
        self.assertGreater(NormalDist().cdf(246_784 / 150_000), 0.95)

    def test_the_sixty_four_combinations(self):
        normal = NormalDist().cdf(5 / 3)
        self.assertEqual(f"{normal:.9f}", "0.952209648")
        ten = student_cdf(5 / 3, 10)
        for dof, want in ((math.inf, normal), (10, ten)):
            seen = set()
            for value, u, limit in itertools.product(KINDS, repeat=3):         # Version 60: 19 of the 64 raised
                got = Result(value(195), u(3), "ms", dof).prob_below(limit(200))
                self.assertAlmostEqual(got, want, places=10, msg=(dof, value, u, limit))
                seen.add(got)
            self.assertEqual(len(seen), 1, dof)                                # one number, whatever the types

    def test_numbers_that_are_equal_give_one_result(self):
        rng = random.Random(60)
        for _ in range(300):
            value = Fraction(rng.randint(-10 ** 6, 10 ** 6), 2 ** rng.randint(0, 20))
            u = Fraction(rng.randint(1, 10 ** 5), 2 ** rng.randint(0, 12))
            limit = Fraction(rng.randint(-10 ** 6, 10 ** 6), 2 ** rng.randint(0, 20))
            dof = rng.choice((math.inf, 3, 25))
            want = Result(value, u, dof=dof).prob_below(limit)
            for a, b, c in itertools.product((float, Decimal, Fraction), repeat=3):
                def as_(kind, x):
                    return kind(x.numerator) / kind(x.denominator) if kind is not Fraction else x
                got = Result(as_(a, value), as_(b, u), dof=dof).prob_below(as_(c, limit))
                self.assertEqual(got, want, (value, u, limit, a, b, c))
            z = float((limit - value) / u)
            if dof == math.inf:
                self.assertAlmostEqual(want, NormalDist().cdf(z), places=12)

    def test_limits_beyond_the_range_of_a_float(self):
        r = Result(195, 3)
        for big in (10 ** 400, Fraction(10 ** 400), D("1E400"), math.inf, D("Infinity")):
            self.assertEqual(r.prob_below(big), 1.0, repr(big)[:20])           # Version 60: OverflowError for the first
        for small in (-10 ** 400, Fraction(-10 ** 400), D("-1E400"), -math.inf, D("-Infinity")):
            self.assertEqual(r.prob_below(small), 0.0, repr(small)[:20])
        # A ratio beyond the range of a float, of numbers that are within it.
        fine = Result(0, 1e-10)
        self.assertEqual((fine.prob_below(1e308), fine.prob_below(-1e308)), (1.0, 0.0))

    def test_what_was_defined_stays(self):
        for bad in (math.nan, D("NaN")):
            with self.assertRaises(ValueError):
                Result(195, 3).prob_below(bad)
        for bad in (None, "200", [200]):
            with self.assertRaises(TypeError):
                Result(195, 3).prob_below(bad)
        point = Result(195, 0)
        self.assertEqual([point.prob_below(x) for x in (195, D("195"), 194.999, Fraction(1951, 10), -math.inf)],
                         [1.0, 1.0, 0.0, 1.0, 0.0])
        self.assertEqual(Result(D("0.3"), 0).prob_below(0.3), 0.0)             # the float lies below three tenths


class TheLimitsOfAnInterval(unittest.TestCase):
    STAMP = 10 ** 18 - 246_720

    def test_no_coverage_and_no_uncertainty(self):
        r = Result(self.STAMP, 150_000.0, "ns")
        self.assertEqual(r.interval(0), (self.STAMP, self.STAMP))              # Version 60: both 64 ns below
        self.assertTrue(all(type(x) is int for x in r.interval(0)))
        self.assertEqual(Result(self.STAMP, 0, "ns").interval(), (self.STAMP, self.STAMP))
        self.assertEqual(Result(D("195.1"), 3).interval(0), (D("195.1"), D("195.1")))

    def test_a_value_that_a_float_does_not_hold(self):
        r = Result(self.STAMP, 150_000.0, "ns")
        w = Fraction(r.expanded())
        self.assertAlmostEqual(float(w), 150_000 * NormalDist().inv_cdf(0.975), places=4)
        low, high = r.interval()
        self.assertEqual((low, high), (self.STAMP - w, self.STAMP + w))
        self.assertEqual((type(low), type(high)), (Fraction, Fraction))
        self.assertEqual((low + high) / 2, self.STAMP)                         # the middle is the value as stored
        self.assertNotEqual(Fraction(float(low)), low)                         # a float holds neither limit
        self.assertNotEqual(Fraction(float(self.STAMP) - float(w)), low)       # what Version 60 returned
        low, high = Result(D("195.1"), 3).interval()
        self.assertEqual((low + high) / 2, Fraction(1951, 10))
        self.assertEqual(high - low, 2 * Fraction(Result(D("195.1"), 3).expanded()))

    def test_a_value_that_a_float_holds(self):
        want = Result(195.0, 3.0).interval()
        self.assertEqual([type(x) for x in want], [float, float])
        for value, u in itertools.product(KINDS, repeat=2):                    # Version 60: a decimal of either raised
            got = Result(value(195), u(3)).interval()
            self.assertEqual((got, [type(x) for x in got]), (want, [float, float]), (value, u))
        rng = random.Random(61)
        for _ in range(500):                                                   # floats: the float arithmetic it was
            value, u, p = rng.gauss(0, 100), rng.uniform(0.01, 5), rng.choice((0.5, 0.9, 0.95, 0.99))
            r = Result(value, u)
            w = r.k(p) * u
            self.assertEqual((r.expanded(p), r.interval(p)), (w, (value - w, value + w)))

    def test_an_uncertainty_and_a_level_of_any_type(self):
        plain = Result(195, 3.0, "ms")
        for u in KINDS:
            r = Result(195, u(3), "ms")
            self.assertEqual((r.expanded(), r.report(), str(r)), (plain.expanded(), plain.report(), str(plain)), u)
        self.assertEqual(plain.report(), "(195.0 ± 5.9) ms, k = 1.96, p = 0.95")
        r = Result(195, 3)
        self.assertEqual(r.k(D("0.1")), r.k(Fraction(1, 10)))                  # Version 60: TypeError for the decimal
        self.assertEqual(r.k(D("0.95")), r.k(Fraction(95, 100)))
        self.assertAlmostEqual(r.k(D("0.95")), NormalDist().inv_cdf(0.975), places=9)
        self.assertEqual(r.k(D("0")), 0.0)
        self.assertEqual(r.report(D("0.9")), "(195.0 ± 4.9), k = 1.64, p = 0.9")
        with self.assertRaises(ValueError):
            r.k(D("1"))


def periods():
    rng = random.Random(23)
    return case.period(rng), case.period(rng)


class ABoundDoesNotCrossItsMargin(unittest.TestCase):
    def test_the_two_rank_bounds_of_the_review(self):
        before = [(0, 0.1 + (i - 63) * 0.001) for i in range(72)]
        after = [(0, 1.3 - (71 - i) * 0.001) for i in range(72)]
        low, high, top = case.rank_bound(before, after)
        self.assertEqual((low, high), (0.1, 1.3))                              # the 64th and the 72nd
        exact = Fraction(1.3) - Fraction(0.1)
        self.assertTrue(1.3 - 0.1 < Fraction(12, 10) < exact)                  # the float lies below, the numbers above
        self.assertEqual(top, exact)
        self.assertIs(case.within(top, D("1.20")), False)                      # Version 60: True
        self.assertEqual(stats.upward(top, 2), "1.21")                         # Version 60: 1.20
        after = [(0, 5.1 - (71 - i) * 0.001) for i in range(72)]
        low, high, top = case.rank_bound(before, after)
        exact = Fraction(5.1) - Fraction(0.1)
        self.assertTrue(exact < 5 == 5.1 - 0.1)                                # here the float lies above
        self.assertEqual(top, exact)
        self.assertIs(case.within(top, 5), True)                               # Version 60: False

    def test_the_bootstrap_and_the_sum(self):
        d = case.handler_change([(0, 0.1)] * 72, [(0, 1.3)] * 72)
        exact = Fraction(1.3) - Fraction(0.1)
        self.assertEqual((d.value, d.u), (exact, 0.0))
        upper = case.upper_bound(d)
        self.assertEqual(upper, exact)
        self.assertIs(case.within(upper, D("1.20")), False)                    # Version 60: True
        # A centre of 0.2 ms and a half-width of 1.0 ms.
        u = 1.0 / case.Z
        self.assertEqual(case.Z * u, 1.0)
        upper = case.upper_bound(Result(0.2, u, "ms"))
        self.assertEqual(upper, Fraction(0.2) + 1)
        self.assertTrue(0.2 + 1.0 < Fraction(12, 10) < upper)
        self.assertIs(case.within(upper, D("1.20")), False)                    # Version 60: True
        self.assertEqual(stats.upward(upper, 2), "1.21")
        # Exact readings were exact already, and a bound equal to its margin is still not below it.
        d = case.handler_change([(0, D("100.0"))] * 20, [(0, D("100.3"))] * 20)
        self.assertEqual((d.value, case.upper_bound(d)), (Fraction(3, 10), Fraction(3, 10)))
        self.assertIs(case.within(case.upper_bound(d), D("0.3")), False)

    def test_what_exact_returns(self):
        for x, y, sign, want, kind in ((0.5, 0.25, 1, 0.75, float), (0.5, 0.25, -1, 0.25, float),
                                       (1.5, 1.5, -1, 0.0, float), (1, 2, 1, 3, Fraction),
                                       (D("0.1"), D("0.2"), 1, Fraction(3, 10), Fraction),
                                       (0.5, D("0.25"), 1, Fraction(3, 4), Fraction),
                                       (0.1, 0.2, 1, Fraction(0.1) + Fraction(0.2), Fraction),
                                       (1.3, 0.1, -1, Fraction(1.3) - Fraction(0.1), Fraction)):
            got = case.exact(x, y, sign)
            self.assertEqual((got, type(got)), (want, kind), (x, y, sign))
        rng = random.Random(62)
        kept = 0
        for _ in range(3000):
            x, y = rng.uniform(-200, 200), rng.uniform(-200, 200) * 10 ** rng.randint(-6, 0)
            sign = rng.choice((1, -1))
            got = case.exact(x, y, sign)
            want = Fraction(x) + sign * Fraction(y)
            self.assertEqual(got, want)                                        # never rounded
            fits = Fraction(float(want)) == want
            self.assertIs(type(got), float if fits else Fraction)
            kept += fits
        self.assertTrue(0 < kept < 3000)

    def test_the_two_periods_of_the_book(self):
        before, after = periods()
        d = case.handler_change(before, after)
        upper = case.upper_bound(d)
        low, high, ranks = case.rank_bound(before, after)
        # The three numbers are the floats that they were: here a float holds each of them exactly.
        self.assertEqual((d.value, upper, ranks), (-0.1474653726260442, 4.7078728237108995, 8.651982212040906))
        self.assertEqual([type(x) for x in (d.value, upper, ranks)], [float, float, float])
        hb, ha = [h for _, h in before], [h for _, h in after]
        self.assertEqual(Fraction(d.value), Fraction(case.p95(ha)) - Fraction(case.p95(hb)))
        self.assertEqual(Fraction(upper), Fraction(d.value) + Fraction(case.Z * d.u))
        self.assertEqual(Fraction(ranks), Fraction(high) - Fraction(low))
        # The duration that a configuration records is still the sum that it forms, two floats as floats.
        self.assertEqual(case.recorded([(0.1, 0.2)], {"start": "accept"}), [0.1 + 0.2])
        self.assertEqual(case.recorded(after, case.NEW), [q + h for q, h in after])


class AShortTraceThroughTheReport(unittest.TestCase):
    @staticmethod
    def rows(n, seed):
        rng = random.Random(seed)
        return [(0.0, rng.uniform(90, 110)) for _ in range(n)]

    def test_which_end_is_missing(self):
        # At 2.5 % a side: a lower end from 2 requests, since 1/20 is above 1/40 and its square is below it,
        # and an upper end from 72.
        self.assertTrue(Fraction(1, 20) ** 2 <= Fraction(1, 40) < Fraction(1, 20))
        self.assertTrue(Fraction(19, 20) ** 72 <= Fraction(1, 40) < Fraction(19, 20) ** 71)
        self.assertEqual((stats.run_length(0.05, 0.025), stats.run_length(0.95, 0.025)), (2, 72))
        for nb, na, lower, upper in ((1, 72, False, True), (20, 71, True, False), (72, 20, True, False),
                                     (2, 72, True, True), (1, 20, False, False), (20, 20, True, False)):
            before, after = self.rows(nb, 1), self.rows(na, 2)
            low, high, top = case.rank_bound(before, after)                    # Version 60: None for all three
            self.assertEqual((low is not None, high is not None, top is not None),
                             (lower, upper, lower and upper), (nb, na))
            if lower:
                self.assertEqual(low, sorted(h for _, h in before)[stats.quantile_ranks(nb, 0.95)[0] - 1])
            if upper:
                self.assertEqual(high, max(h for _, h in after))

    def test_the_lines_that_are_printed(self):
        cases = {(1, 72): "  no lower end with fewer than 2 before\n",
                 (20, 71): "  no upper end with fewer than 72 after\n",
                 (72, 20): "  no upper end with fewer than 72 after\n",
                 (1, 20): "  no lower end with fewer than 2 before\n  no upper end with fewer than 72 after\n"}
        for (nb, na), tail in cases.items():                                   # Version 60: TypeError in the formatter
            text = printed(case.ranked, self.rows(nb, 1), self.rows(na, 2))
            self.assertEqual(text, f"the p95 by ranks:\n  requests: {nb} before, {na} after\n" + tail)
        before, after = self.rows(2, 1), self.rows(72, 2)
        low, high = min(h for _, h in before), max(h for _, h in after)
        text = printed(case.ranked, before, after)
        self.assertEqual(text.splitlines()[1:4], [f"  before, at least {stats.downward(low, 2)} ms at 97.5 %",
                                                  f"  after, at most {stats.upward(high, 2)} ms at 97.5 %",
                                                  f"  change at most {stats.upward(Fraction(high) - Fraction(low), 2)}"
                                                  " ms at 95 %"])
        self.assertEqual(len(text.splitlines()), 6)

    def test_the_transcript(self):
        text = expected("ch23_case")
        self.assertEqual(printed(case.main), text)
        self.assertIn("the p95 by ranks:\n  before, at least 177.04 ms at 97.5 %\n"
                      "  after, at most 185.70 ms at 97.5 %\n  change at most 8.66 ms at 95 %\n"
                      "margin 5 ms: not established\nmargin 2 ms: not established\n", text)
        self.assertIn("  requests for 95 %: 59\nthe p95 by ranks:\n  requests: 20 before, 20 after\n"
                      "  no upper end with fewer than 72 after\n", text)
        self.assertNotIn("None", text)
        self.assertEqual(len(text.splitlines()), 41)


class TheSameRequestsReportedTwice(unittest.TestCase):
    ONE = [100 + i for i in range(100)]
    TWO = [101 + i for i in range(100)]

    @staticmethod
    def p95(xs):
        xs = sorted(xs)
        return xs[math.ceil(Fraction(95, 100) * len(xs)) - 1]

    def test_as_two_periods(self):
        self.assertEqual((self.p95(self.ONE), self.p95(self.TWO)), (194, 195))
        # The bootstrap of the listing, by its own draws: 300 resamples of each report with the seed 23.
        spreads = []
        for xs in (self.ONE, self.TWO):
            rng = random.Random(23)
            reps = [self.p95(rng.choices(xs, k=100)) for _ in range(300)]
            spreads.append(statistics.stdev(reps))
        u = math.hypot(*spreads)
        upper = 1 + NormalDist().inv_cdf(0.95) * u
        self.assertEqual((f"{u:.4f}", f"{upper:.4f}"), ("3.3322", "6.4810"))
        d = case.handler_change([(0, x) for x in self.ONE], [(0, x) for x in self.TWO])
        self.assertEqual(d.value, 1)
        self.assertAlmostEqual(d.u, u, places=12)
        self.assertAlmostEqual(float(case.upper_bound(d)), upper, places=9)
        self.assertEqual((str(d), stats.upward(case.upper_bound(d), 2)), ("1.0(33) ms", "6.49"))

    def test_as_pairs(self):
        rng = random.Random(23)
        changes = []
        for _ in range(300):
            ids = rng.choices(range(100), k=100)
            changes.append(self.p95(self.TWO[i] for i in ids) - self.p95(self.ONE[i] for i in ids))
        self.assertEqual(set(changes), {1})
        self.assertEqual(statistics.stdev(changes), 0)
        self.assertEqual(printed(case.side_by_side), "100 requests reported twice, 1 ms apart:\n"
                         "  as two periods: 1.0(33) ms\n    at most 6.49 ms at 95 %\n"
                         "  as pairs, one draw for both: 1(0) ms\n")
        self.assertTrue(expected("ch23_case").endswith(printed(case.side_by_side)))
        # Another offset is carried through, and the spread of the pairs is still none.
        text = printed(case.side_by_side, 60, D("2.5"))
        self.assertEqual(text.splitlines()[0], "60 requests reported twice, 2.5 ms apart:")
        self.assertEqual(text.splitlines()[-1], "  as pairs, one draw for both: 2.5(0) ms")

    def test_the_change_of_the_percentile_is_not_the_percentile_of_the_changes(self):
        first, second = (100, 200), (200, 100)                                 # two requests that change places
        self.assertEqual(self.p95(second) - self.p95(first), 0)
        self.assertEqual(self.p95(b - a for a, b in zip(first, second)), 100)


class TheRuleInHundredths(unittest.TestCase):
    def test_the_line_and_the_cutoffs(self):
        z = NormalDist().inv_cdf(0.95)
        accept, reject = 200 - 3 * z, 200 + 3 * z
        self.assertEqual((f"{accept:.9f}", f"{reject:.9f}"), ("195.065439119", "204.934560881"))
        text = expected("ch14_decisions")
        self.assertEqual(text.splitlines()[0], "accept <= 195.06, reject >= 204.94 ms")
        self.assertEqual(printed(decisions.conformity), "".join(text.splitlines(True)[:9]))

        def decision(y):
            p = NormalDist().cdf((200 - y) / 3)
            return "accept" if p >= 0.95 else "reject" if p < 0.05 else "undecided"
        # The whole rule for readings in hundredths, as the review writes it.
        ranges = {"accept": [], "undecided": [], "reject": []}
        for hundredths in range(18000, 22001):
            ranges[decision(hundredths / 100)].append(hundredths)
        self.assertEqual((max(ranges["accept"]), min(ranges["undecided"]), max(ranges["undecided"]),
                          min(ranges["reject"])), (19506, 19507, 20493, 20494))
        # Finer readings turn on the cutoffs themselves.
        self.assertEqual([decision(y) for y in (195.065, 195.066, 204.934, 204.935)],
                         ["accept", "undecided", "undecided", "reject"])
        self.assertEqual([decision(y) for y in (195.08, 204.92)], ["undecided", "undecided"])


if __name__ == "__main__":
    unittest.main()
