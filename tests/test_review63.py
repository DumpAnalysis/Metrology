"""Regression tests added after the review of Version 63.

The review found that both() of Listing 23.2 asked a class of the reference for its length after the record had
been read, so that a view of a stream which counts what is left of it was counted as empty, and a record that
misses its objective was reported as met; that the functions which pass over their periods more than once read a
stream at the first pass and found it empty at the second; that the weighted mean of Listings 8.4 and 23.2 added
rounded products, so that three durations of 200 ms could have a mean below or above 200 ms; that components()
compared two percentiles after it had made floats of them; that the durations of the record are at their mean
only within the rounding of floats; that the scores of Listing 18.1 took a temperature below zero, an empty
record, a bin count of zero and a decimal label or logit, lost the residual of a confident case on one side, and
overflowed at logits of 1e308; that prob_below() returned the probability of a capped distance for a finite limit
far out; and that rect_interval() refused widths which its exact arithmetic can hold, while the three standard
deviations of the Type B evaluations treated a width beyond a float differently by its type. It proposed an
exercise of two cases for the calibration error and asked for the exact instants between which adjacent floats
of nanoseconds are 256 ns apart.

Each test computes what it expects independently of the code under test: shares and percentiles by counting and
sorting, means from fractions, scores from their definitions, Student's tail far out from its density, in
decimals of fifty digits, and dates from whole seconds.
"""
import contextlib
import datetime
import io
import math
import random
import unittest
from collections import deque
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

from examples import ch08_selection as selection
from examples import ch18_calibration as cal
from examples import ch20_judge as judge
from examples import ch23_mix as mix
from swmetro import Result, core, dist, rect_interval, u_rect, u_resolution, u_tri

ROOT = Path(__file__).resolve().parent.parent
D = Decimal


def expected(name):
    return (ROOT / "expected" / f"{name}.txt").read_text(encoding="utf-8")


def printed(function, *args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        function(*args)
    return out.getvalue()


def p95(xs):
    """Inverse empirical CDF: the least value with at least 95 % of the values at or below it."""
    xs = sorted(xs)
    for x in xs:
        if 100 * sum(y <= x for y in xs) >= 95 * len(xs):
            return x


def far_tail(exponent, dof):
    """Upper tail of Student's distribution at t = 10 ** exponent, far out: the density there is
    c * (t * t / dof) ** (-(dof + 1) / 2), and its integral from t on is c * sqrt(dof) * (t * t / dof) ** (-dof / 2)
    / dof. The power is taken in decimals of 50 digits."""
    c = math.gamma((dof + 1) / 2) / (math.sqrt(math.pi * dof) * math.gamma(dof / 2))
    with localcontext() as ctx:
        ctx.prec = 50
        power = ((D(10) ** (2 * exponent)) / D(dof)).ln() * (-D(dof) / 2)
        return float(D(c) * D(dof).sqrt() / D(dof) * power.exp())


def views(shared_cursor):
    """A record and a reference whose slow class is two views of one source of two rows. A view tells its
    length from the rows that are left. With `shared_cursor` both views hand out one cursor, as in the review;
    without it each call of iter() opens a cursor of its own on the same rows."""
    pending = deque([500, 500])

    def drain():
        while pending:
            yield pending.popleft()

    cursor = drain()

    class View:
        def __iter__(self):
            return cursor if shared_cursor else drain()

        def __len__(self):
            return len(pending)

    return {"fast": [100] * 18, "slow": View()}, {"fast": range(18), "slow": View()}


def small():
    """Two periods of 18 fast and 2 slow requests, as lists."""
    return ({"fast": [100.0] * 18, "slow": [500.0] * 2}, {"fast": [90.0] * 18, "slow": [450.0] * 2})


KINDS = (iter, tuple, deque, lambda xs: (x for x in xs))


def as_streams(periods, kind):
    return tuple({name: kind(xs) for name, xs in rec.items()} for rec in periods)


def chapter():
    rng = random.Random(23)
    before = mix.record(mix.BEFORE, rng)
    return before, mix.record(mix.AFTER, rng)


class TwoViewsOfOneStream(unittest.TestCase):
    """Eighteen fast requests of 100 ms and two slow ones of 500 ms: 90 % are within 200 ms."""

    def test_the_views_of_the_review(self):
        for shared_cursor in (True, False):
            pairs = mix.weighted(*views(shared_cursor))
            self.assertEqual(sum(w for _, w in pairs), 20)                     # Version 63: 18
            self.assertEqual(mix.p95(pairs), p95([100] * 18 + [500] * 2))      # Version 63: 100
            self.assertEqual(mix.p95(pairs), 500)
            low, high = mix.within(*views(shared_cursor), 200)
            self.assertEqual((low, high), (Fraction(18, 20), Fraction(18, 20)))
            self.assertEqual(mix.verdict(low, high), "missed")                 # Version 63: met

    def test_a_length_is_taken_before_the_record_is_read(self):
        rec, ref = views(False)
        self.assertEqual(len(ref["slow"]), 2)
        held, sizes = mix.both(rec, ref)
        self.assertEqual(len(ref["slow"]), 0)                                  # the record has drained the rows
        self.assertEqual({k: len(v) for k, v in sizes.items()}, {"fast": 18, "slow": 2})
        self.assertEqual(held["slow"], [500, 500])

    def test_what_was_counted_before_is_counted_as_before(self):
        slow = iter([500, 500])
        pairs = mix.weighted({"fast": [100] * 18, "slow": slow}, {"fast": range(18), "slow": slow})
        self.assertEqual(sum(w for _, w in pairs), 20)                         # one cursor, record and reference
        rows = deque([500, 500])
        pairs = mix.weighted({"fast": [100] * 18, "slow": rows}, {"fast": range(18), "slow": rows})
        self.assertEqual(sum(w for _, w in pairs), 20)                         # one sized object in both
        pairs = mix.weighted({"fast": [100] * 18, "slow": [500, 500]}, {"fast": range(18), "slow": iter("ab")})
        self.assertEqual(sum(w for _, w in pairs), 20)                         # a stream of its own is read
        twice = iter(range(3))
        low, high = mix.within({"a": [100], "b": [500]}, {"a": twice, "b": twice}, 200)
        self.assertEqual((low, high), (Fraction(1, 2), Fraction(1, 2)))        # read once, kept for two names


class APeriodIsReadOnce(unittest.TestCase):
    FUNCTIONS = ("components", "new_class", "one_mix", "lost_class")

    def test_the_decomposition_of_two_small_periods(self):
        # 18 * 100 + 2 * 500 = 2800 and 18 * 90 + 2 * 450 = 2520 over 20 requests: 140 and 126 ms. The mix
        # is the same in both periods, so each order has all of the change within the classes.
        want = ("mean, 140 -> 126 ms: -14 ms\n"
                "  class means first, via 126 ms:\n    within -14, mix +0\n"
                "  mix first, via 140 ms:\n    within -14, mix +0\n"
                "  average of the two orders:\n    within -14, mix +0\n")
        self.assertEqual(printed(mix.components, *small()), want)
        for kind in KINDS:                                                     # Version 63: "empty reference mix"
            self.assertEqual(printed(mix.components, *as_streams(small(), kind)), want)

    def test_every_report_of_streams_is_the_report_of_lists(self):
        for name in self.FUNCTIONS:
            want = printed(getattr(mix, name), *small())
            for kind in KINDS:
                self.assertEqual(printed(getattr(mix, name), *as_streams(small(), kind)), want, name)
        want = printed(mix.components, *small(), mix.p95, "p95", "distributions")
        for kind in KINDS:
            got = printed(mix.components, *as_streams(small(), kind), mix.p95, "p95", "distributions")
            self.assertEqual(got, want)

    def test_the_reports_of_the_chapter_from_streams(self):
        # The transcript is the report of the lists. Each function, given the same two periods as streams,
        # prints lines that stand in it.
        transcript = expected("ch23_mix")
        for name in self.FUNCTIONS:
            for kind in (iter, tuple):
                got = printed(getattr(mix, name), *as_streams(chapter(), kind))
                self.assertIn(got, transcript, name)
                self.assertGreater(got.count("\n"), 5)

    def test_a_stream_in_two_periods(self):
        slow = iter([500.0, 500.0])
        before = {"fast": [100.0] * 18, "slow": slow}
        after = {"fast": [90.0] * 18, "slow": slow}
        for name in self.FUNCTIONS:
            with self.assertRaisesRegex(ValueError, "stream in two periods"):
                getattr(mix, name)(before, after)
        self.assertEqual(list(slow), [500.0, 500.0])                           # refused before it was read
        rec, ref = views(True)
        with self.assertRaisesRegex(ValueError, "stream in two periods"):
            mix.periods(rec, {"slow": ref["slow"]})
        both = [500.0, 500.0]                                                  # a list can be read again
        out = printed(mix.components, {"fast": [100.0] * 18, "slow": both}, {"fast": [90.0] * 18, "slow": both})
        self.assertIn("mean, 140 -> 131 ms: -9 ms\n", out)                     # (18 * 90 + 1000) / 20 = 131
        one = iter([1.0])
        with self.assertRaisesRegex(ValueError, "stream in two classes"):      # as before, within one period
            mix.periods({"a": one, "b": one}, {"a": [1.0]})

    def test_periods_are_lists_held_to_the_rules_of_a_class(self):
        before, after = mix.periods(*as_streams(small(), iter))
        self.assertEqual((before, after), small())
        with self.assertRaisesRegex(ValueError, "a time below zero"):
            mix.periods({"a": iter([1.0, -1.0])}, {"a": [1.0]})

    def test_a_class_that_was_added(self):
        self.assertEqual(mix.added({"fast": iter([1.0]), "bulk": iter([5.0])}, {"fast": range(1)}), ["bulk"])
        # Version 63 took an empty stream for a class with observations: a stream is true whatever it holds.
        self.assertEqual(mix.added({"fast": iter([1.0]), "bulk": iter([])}, {"fast": range(1)}), [])
        self.assertEqual(mix.added({"fast": [1.0], "bulk": [5.0]}, {"fast": [0], "bulk": iter([])}), ["bulk"])


class AMeanIsOneDivision(unittest.TestCase):
    def test_three_durations_of_200_ms(self):
        for weight in (Fraction(1, 3), Fraction(5, 3), 0.1, 3, D("0.7")):
            pairs = [(200.0, weight)] * 3
            self.assertEqual(mix.mean(pairs), 200.0)                           # Version 63: 199.99999999999997
            self.assertEqual(selection.summary(pairs)[0], 200.0)               # and 200.00000000000003
        # The sums of Version 63: each product a float, rounded before it is added.
        third = sum(200.0 * w for w in [Fraction(1, 3)] * 3) / 1
        self.assertEqual((third, sum(200.0 * w for w in [Fraction(5, 3)] * 3) / 5),
                         (199.99999999999997, 200.00000000000003))

    def test_the_mean_is_the_exact_one_rounded_once(self):
        rng = random.Random(63)
        differ = 0
        for _ in range(200):
            xs = [rng.uniform(0, 1000) for _ in range(rng.randint(1, 30))]
            ws = [Fraction(rng.randint(1, 9), rng.randint(1, 9)) for _ in xs]
            exact = sum(Fraction(x) * w for x, w in zip(xs, ws)) / sum(ws)
            pairs = list(zip(xs, ws))
            self.assertEqual(mix.mean(pairs), float(exact))
            self.assertEqual(selection.summary(pairs)[0], float(exact))
            differ += sum(x * w for x, w in pairs) / sum(ws) != float(exact)
        self.assertGreater(differ, 20)                                         # the sum of floats often differs

    def test_a_mean_within_the_range_of_floats(self):
        self.assertEqual(mix.mean([(1e308, Fraction(2))]), 1e308)              # Version 63: inf
        self.assertEqual(selection.summary([(1e308, 2)])[0], 1e308)
        self.assertEqual(mix.mean([(1e308, 3), (1.5e308, 5)]), float((3 * Fraction(1e308) + 5 * Fraction(1.5e308)) / 8))

    def test_the_type_of_a_mean(self):
        for pairs, want in (([(100, 1), (200, 1)], Fraction(150)), ([(D("100.5"), 1), (200, 1)], Fraction(601, 4)),
                            ([(Fraction(1, 3), 1), (1, 2)], Fraction(7, 9))):
            got = mix.mean(pairs)
            self.assertEqual((got, type(got)), (want, Fraction))
        got = mix.mean([(100.5, 1), (200, 1)])
        self.assertEqual((got, type(got)), (150.25, float))
        with self.assertRaisesRegex(ValueError, "no requests"):
            mix.mean([])

    def test_a_fill_and_an_unknown_duration(self):
        self.assertEqual(mix.mean([(math.inf, 1), (5.0, 1)]), math.inf)
        self.assertEqual(mix.mean([(-math.inf, Fraction(1, 3)), (5, 1)]), -math.inf)
        self.assertTrue(math.isnan(mix.mean([(math.inf, 1), (-math.inf, 1)])))
        self.assertTrue(math.isnan(mix.mean([(math.nan, 1), (5.0, 1)])))
        rec = {"fast": [100.0] * 18}
        low, top = (mix.mean(mix.weighted(rec, {"fast": range(18), "slow": range(2)}, fill=x))
                    for x in (-math.inf, math.inf))
        self.assertEqual((low, top), (-math.inf, math.inf))

    def test_the_summary_of_a_selection(self):
        rng = random.Random(8)
        pairs = [(rng.uniform(1, 400), 1 / rng.choice((1.0, 0.5, 0.02))) for _ in range(300)]
        mean, p, slow = selection.summary(pairs)
        total = sum(Fraction(w) for _, w in pairs)
        self.assertEqual(mean, float(sum(Fraction(x) * Fraction(w) for x, w in pairs) / total))
        acc = 0
        for x, w in sorted(pairs):
            acc += Fraction(w)
            if acc * 20 >= total * 19:
                break
        self.assertEqual(p, x)
        over = sum(Fraction(w) for x, w in pairs if x > selection.SLOW)
        self.assertEqual(slow, 100 * float(over / total))


class TwoEndsAreComparedAsTheyAre(unittest.TestCase):
    def test_the_case_of_the_review(self):
        # 94 readings of 100 ms and five of 100.000000000000001 ms in one class, then 99 of 100 ms in that
        # class and one request of a new class. At the mix of the second period the first has no duration
        # for the new class. With it below everything the 95th of 100 weights is the last reading of 100 ms,
        # and with it above everything it is a reading of 100.000000000000001 ms: the two are not equal.
        longer = D("100.000000000000001")
        before = {"common": [100] * 94 + [longer] * 5}
        after = {"common": [100] * 99, "new": [100]}
        ends = [mix.p95(mix.weighted(before, after, fill=x)) for x in (-math.inf, math.inf)]
        self.assertEqual(ends, [100, longer])
        self.assertNotEqual(*ends)
        self.assertEqual(float(ends[0]), float(ends[1]))                       # as floats they were equal
        got = printed(mix.components, before, after, mix.p95, "p95", "distributions")
        self.assertEqual(got, "p95, 100 -> 100 ms: +0 ms\n"
                              "  class distributions first, via 100 ms:\n    within +0, mix +0\n"
                              "  mix first: not determined,\n    no new requests before\n")

    def test_what_is_determined_stays_determined(self):
        transcript = expected("ch23_mix")
        before, after = chapter()
        for args in ((), (mix.p95, "p95", "distributions")):
            self.assertIn(printed(mix.components, before, after, *args), transcript)
        self.assertIn(printed(mix.one_bulk), transcript)


class TheMeanOfTheSlowClass(unittest.TestCase):
    def test_the_generated_mean(self):
        _, after = chapter()
        slow = after["slow"]
        over = sum(map(Fraction, slow)) / 900 - 450
        self.assertEqual(over, Fraction(95, 1266637395197952))                 # the figure of the review
        self.assertEqual(f"{float(over):.2e}", "7.50e-14")
        self.assertEqual(f"{sum(slow) / 900:.10f}", "450.0000000000")
        self.assertIn("within the rounding of floats", mix.durations.__doc__)
        self.assertNotIn("exactly", mix.durations.__doc__)

    def test_one_request_more_or_less(self):
        # With the bound of 810 ms and one duration recovered, x, the other 899 have the sum 900 m - x, and
        # k of them within 200 ms need 200 k + 810 (899 - k) >= 900 m - x.
        def most(x, m):
            k = (810 * 899 - (900 * m - x)) // 610
            return int(k) + (x <= 200)
        _, after = chapter()
        generated = sum(map(Fraction, after["slow"])) / 900
        for x in (110, 720):
            self.assertEqual((most(x, Fraction(450)), most(x, generated)), (531, 530))
            self.assertEqual(mix.with_one(x, 450), 531)
            self.assertEqual(mix.with_one(x, generated), 530)
        self.assertEqual((most(100, Fraction(450)), mix.with_one(100, 450)), (530, 530))


class AScoreAtATemperatureAboveZero(unittest.TestCase):
    ONE = [(2, 0)]

    def scores(self):
        return (cal.nll, cal.brier, cal.ece, cal.bins_of)

    def test_one_confident_error(self):
        # A logit of 2 with the label 0: the stated probability of the label 1 is 1 / (1 + exp(-2)).
        p = 1 / (1 + math.exp(-2))
        self.assertAlmostEqual(cal.nll(self.ONE), -math.log(1 - p), places=12)
        self.assertAlmostEqual(cal.brier(self.ONE), p * p, places=12)
        self.assertAlmostEqual(cal.ece(self.ONE), p, places=12)
        self.assertEqual(f"{-math.log(1 - p):.3f} {-math.log(p):.3f}", "2.127 0.127")   # at T = 1, and at T = -1

    def test_a_temperature_is_a_finite_number_above_zero(self):
        for temp in (-1, 0, -0.0, -1e-300, D("-2"), Fraction(-1, 3)):          # Version 63 at -1: NLL 0.127
            for score in self.scores():
                with self.assertRaisesRegex(ValueError, "need T > 0"):
                    score(self.ONE, temp)
        for temp in (math.nan, math.inf, D("NaN"), D("Infinity"), 10 ** 400):
            for score in self.scores():
                with self.assertRaisesRegex(ValueError, "a T is missing or too large"):
                    score(self.ONE, temp)
        for temp in (None, "2", True, 2 + 0j):
            for score in self.scores():
                with self.assertRaisesRegex(ValueError, "no number"):
                    score(self.ONE, temp)
        for temp in (2, 2.0, D("2"), Fraction(2)):
            for score in self.scores():
                self.assertEqual(score(self.ONE, temp), score([(1, 0)]))

    def test_no_cases(self):
        for score in self.scores() + (cal.fit_temperature, cal.checked):
            with self.assertRaisesRegex(ValueError, "no cases"):               # Version 63: four other errors
                score([])
            with self.assertRaisesRegex(ValueError, "no cases"):
                score(iter(()))

    def test_a_count_of_bins(self):
        for bins in (0, -3, 2.5, math.nan, math.inf, None, "10"):
            for score in (cal.ece, cal.bins_of):
                with self.assertRaisesRegex(ValueError, "need a whole bin count >= 1"):
                    score(self.ONE, 1.0, bins)
        self.assertEqual(cal.bins_of(self.ONE, 1.0, 10.0), cal.bins_of(self.ONE))
        self.assertEqual(len(cal.bins_of(self.ONE, 1.0, 1)), 1)
        p = 1 / (1 + math.exp(-2))
        self.assertAlmostEqual(cal.ece(self.ONE, 1.0, 1), p, places=12)

    def test_a_decimal_label_and_a_decimal_logit(self):
        plain = [(2.0, 1), (-1.0, 0), (0.5, 1)]
        for other in ([(D("2"), D("1")), (D("-1"), D("0")), (D("0.5"), D("1"))],
                      [(Fraction(2), Fraction(1)), (Fraction(-1), Fraction(0)), (Fraction(1, 2), True)],
                      [(2, 1.0), (-1, 0.0), (0.5, 1.0)]):
            for score in self.scores() + (cal.checked,):                       # Version 63: TypeError
                self.assertEqual(score(other), score(plain))
        self.assertEqual(cal.checked(plain), plain)
        self.assertEqual([type(v) for row in cal.checked([(D("2"), D("1"))]) for v in row], [float, int])
        for label in (2, -1, 0.5, None, "1", math.nan, D("NaN"), D("0.5")):
            with self.assertRaisesRegex(ValueError, "a label is 0 or 1"):
                cal.checked([(2.0, label)])
        for logit in (None, "2", True):
            with self.assertRaisesRegex(ValueError, "no number"):
                cal.checked([(logit, 1)])
        with self.assertRaisesRegex(ValueError, "a logit is missing or too large"):
            cal.checked([(10 ** 400, 1)])
        self.assertEqual(cal.checked([(2, 1 + 0j), (1, 0j)]), [(2.0, 1), (1.0, 0)])

    def test_decimal_ends_of_a_search_and_a_decimal_logit(self):
        data = [(2, 1), (1, 0), (0, 1)]
        want = cal.fit_temperature(data)
        for lo, hi in ((D("0.2"), D("10")), (Fraction(1, 5), 10), (0.2, D("10"))):
            self.assertAlmostEqual(cal.fit_temperature(data, lo, hi), want, places=9)   # Version 63: TypeError
        self.assertEqual(cal.at_one_logit(D("40"), 0.99), cal.at_one_logit(40, 0.99))
        self.assertEqual(cal.temperatures(D("40"), 99, 100), cal.temperatures(40, 99, 100))
        self.assertEqual(cal.temperatures(Fraction(-40), 1, 100), cal.temperatures(40, 99, 100))


class AResidualOnItsOwnSide(unittest.TestCase):
    def test_a_confident_case_under_either_name_of_its_label(self):
        # At a logit of 40 the stated probability differs from 1 by exp(-40) / (1 + exp(-40)), 4.2e-18.
        r = math.exp(-40) / (1 + math.exp(-40))
        self.assertEqual(f"{r:.5e} {r * r:.5e}", "4.24835e-18 1.80485e-35")    # the figures of the review
        for data in ([(40, 1)], [(-40, 0)]):
            self.assertAlmostEqual(cal.brier(data) / (r * r), 1, places=12)    # Version 63: 0 for the first
            self.assertAlmostEqual(cal.ece(data) / r, 1, places=12)
            self.assertAlmostEqual(cal.nll(data) / r, 1, places=12)
        self.assertEqual(cal.brier([(40, 1)]), cal.brier([(-40, 0)]))
        self.assertEqual(cal.ece([(40, 1)]), cal.ece([(-40, 0)]))

    def test_the_brier_score_does_not_know_which_label_is_called_1(self):
        rng = random.Random(18)
        data = [(rng.gauss(0, 12), rng.randint(0, 1)) for _ in range(500)]
        other = [(-z, 1 - y) for z, y in data]
        self.assertEqual(cal.brier(data), cal.brier(other))
        self.assertEqual(cal.nll(data), cal.nll(other))
        want = sum((1 / (1 + math.exp(-z)) - y) ** 2 for z, y in data) / 500
        self.assertAlmostEqual(cal.brier(data), want, places=12)

    def test_the_residuals_of_a_bin(self):
        rng = random.Random(63)
        data = [(rng.gauss(0, 2), rng.randint(0, 1)) for _ in range(300)]
        tot = cal.bins_of(data)
        self.assertEqual(sum(b[0] for b in tot), 300)
        for n, s, k, r in tot:
            self.assertAlmostEqual(r, s - k, places=10)
        self.assertAlmostEqual(cal.ece(data), sum(abs(s - k) for _, s, k, _ in tot) / 300, places=12)


class LogitsAtTheEndOfTheFloats(unittest.TestCase):
    def test_a_mean_loss_of_1e308(self):
        self.assertEqual(cal.nll([(1e308, 0)] * 2), 1e308)                     # Version 63: inf
        self.assertEqual(cal.nll([(-1e308, 1)] * 3), 1e308)
        self.assertEqual(cal.nll([(1e308, 0), (1e308, 1)]), 5e307)
        top = math.nextafter(math.inf, 0)
        self.assertEqual(cal.nll([(top, 0)] * 3), top)
        self.assertEqual(cal.brier([(1e308, 0)] * 2), 1.0)
        self.assertEqual(cal.ece([(1e308, 0)] * 2), 1.0)

    def test_a_quotient_beyond_a_float(self):
        for score in (cal.nll, cal.brier, cal.ece, cal.bins_of):
            with self.assertRaisesRegex(ValueError, "logit / T too large"):
                score([(1e308, 0)], 0.5)
            self.assertIsNotNone(score([(1e308, 0)], 2))

    def test_a_fit_beside_two_certain_cases(self):
        # Two cases with margins of 2 and -1: the slope of the loss is zero where
        # 2 / (1 + exp(2 b)) = 1 / (1 + exp(-b)), b being 1 / T. A case that is right at 1e308 adds nothing.
        lo, hi = 0.1, 10.0
        for _ in range(200):
            b = (lo + hi) / 2
            if 2 / (1 + math.exp(2 * b)) > 1 / (1 + math.exp(-b)):
                lo = b
            else:
                hi = b
        want = 2 / (lo + hi)
        self.assertEqual(f"{want:.5f}", "2.38312")                             # the figure of the review
        self.assertAlmostEqual(cal.fit_temperature([(2, 1), (1, 0)]), want, places=9)
        beside = [(2, 1), (1, 0), (1e308, 1), (1e308, 1)]                      # Version 63: OverflowError
        self.assertAlmostEqual(cal.fit_temperature(beside), want, places=9)
        self.assertAlmostEqual(cal.fit_temperature(beside + [(-1e308, 0)] * 3), want, places=9)

    def test_the_sign_of_a_sum_beyond_a_float(self):
        ms = [1e308, 1e308, -1e308, -1e308, -1e308]
        self.assertGreater(cal.slope(ms, 1.0), 0)
        self.assertEqual(cal.slope(ms, 1.0), 3 * Fraction(1e308))
        with self.assertRaisesRegex(ValueError, "the loss falls as T rises"):
            cal.fit_temperature([(1e308, 1), (1e308, 1), (1e308, 0), (1e308, 0), (1e308, 0)])
        with self.assertRaisesRegex(ValueError, "the loss falls as T rises"):
            cal.fit_temperature([(1e308, 1), (1e308, 0), (1.0, 0), (1.0, 1)])  # the margins sum to zero exactly
        self.assertAlmostEqual(cal.slope([2.0, -1.0], 0.5), -2 / (1 + math.e) + 1 / (1 + math.exp(-0.5)), places=12)

    def test_the_ends_of_a_search(self):
        data = [(2, 1), (1, 0), (0, 1)]
        for lo in (5e-324, 0, -1, math.nan):
            with self.assertRaisesRegex(ValueError, "need 0 < lo < hi < inf"):
                cal.fit_temperature(data, lo, 10)
        self.assertAlmostEqual(cal.fit_temperature(data, 1e-300, 10), cal.fit_temperature(data), places=9)


class TwoCasesAndTheNameOfALabel(unittest.TestCase):
    Z = math.log(11 / 9)                                                       # the logit of 0.55

    def test_the_exercise(self):
        # Stated 0.50 with the label 0 and 0.55 with the label 1: both in the bin from 0.5 to 0.6, whose gap
        # is |0.50 + 0.55 - 1| / 2. With the other label called 1 the cases are stated at 0.50 and 0.45, with
        # the labels 1 and 0, and stand in two bins, with the gaps 0.50 and 0.45.
        first = [(0.0, 0), (self.Z, 1)]
        other = [(0.0, 1), (-self.Z, 0)]
        self.assertAlmostEqual(cal.ece(first), abs(0.50 + 0.55 - 1) / 2, places=12)
        self.assertAlmostEqual(cal.ece(other), (0.50 + 0.45) / 2, places=12)
        self.assertEqual(f"{cal.ece(first):.3f} {cal.ece(other):.3f}", "0.025 0.475")
        brier = (0.50 ** 2 + 0.45 ** 2) / 2
        loss = -(math.log(0.50) + math.log(0.55)) / 2
        for data in (first, other):
            self.assertAlmostEqual(cal.brier(data), brier, places=12)
            self.assertAlmostEqual(cal.nll(data), loss, places=12)
        self.assertEqual(f"{brier:.5f} {loss:.4f}", "0.22625 0.6455")
        self.assertEqual(f"{0.475 / 0.025:.0f}", "19")
        for copies in (2, 100, 1000):                                          # more cases change nothing
            self.assertAlmostEqual(cal.ece(first * copies), 0.025, places=12)
            self.assertAlmostEqual(cal.ece(other * copies), 0.475, places=12)

    def test_where_the_cases_stand(self):
        first = cal.bins_of([(0.0, 0), (self.Z, 1)])
        self.assertEqual([b[0] for b in first], [0, 0, 0, 0, 0, 2, 0, 0, 0, 0])
        other = cal.bins_of([(0.0, 1), (-self.Z, 0)])
        self.assertEqual([b[0] for b in other], [0, 0, 0, 0, 1, 1, 0, 0, 0, 0])
        self.assertEqual([b[0] for b in cal.bins_of([(0.0, 0)], 1.0, 2)], [0, 1])   # one half is on an edge
        self.assertEqual([b[0] for b in cal.bins_of([(800.0, 1)], 1.0, 4)], [0, 0, 0, 1])   # 1 is in the last

    def test_a_probability_is_in_the_bin_where_the_float_lies(self):
        # A logit whose stated probability is the float 0.3, 0.6 or 0.7: each of the three floats is below
        # the decimal that it prints as, and lies in the bin below that edge. Version 63 multiplied the float
        # by ten, which rounds to the whole number, and put it in the bin above.
        found = 0
        for edge, below in ((0.3, 2), (0.6, 5), (0.7, 6)):
            self.assertLess(Fraction(edge), Fraction(below + 1, 10))
            z = math.log(edge / (1 - edge))
            near = [z]
            for _ in range(40):
                near += [math.nextafter(near[-2] if len(near) > 1 else z, math.inf),
                         math.nextafter(near[-1] if len(near) > 1 else z, -math.inf)]
            near = sorted(set(near) | {math.nextafter(z, math.inf), math.nextafter(z, -math.inf)})
            for logit in near:
                tot = cal.bins_of([(logit, 1)])
                b = next(i for i, row in enumerate(tot) if row[0])
                p = tot[b][1]
                self.assertEqual(b, min(int(Fraction(p) * 10), 9))
                if p == edge:
                    found += 1
                    self.assertEqual(b, below)
                    self.assertEqual(int(p * 10), below + 1)
        self.assertGreater(found, 0)
        rng = random.Random(63)
        for _ in range(2000):
            bins = rng.randint(1, 40)
            tot = cal.bins_of([(rng.gauss(0, 3), 0)], 1.0, bins)
            b = next(i for i, row in enumerate(tot) if row[0])
            self.assertEqual(b, min(int(Fraction(tot[b][1]) * bins), bins - 1))


class AFitOfTheWholeModel(unittest.TestCase):
    def test_all_labels_positive(self):
        data = [(40.0, 1)] * 100
        with self.assertRaisesRegex(ValueError, "no finite T: the loss falls as T falls"):
            cal.fit_temperature(data)
        # A fit confined to the interval of the search would have its least loss at the lower end, 0.2.
        losses = [cal.nll(data, t) for t in (0.2, 0.5, 1, 2, 5, 10)]
        self.assertEqual(losses, sorted(losses))
        self.assertLess(losses[0], losses[1])
        self.assertIn("bound the search and not the model", " ".join(cal.fit_temperature.__doc__.split()))

    def test_the_limits_before_they_are_rounded(self):
        low, high = judge.exact_interval(99, 100)
        self.assertEqual((f"{low:.12f}", f"{high:.12f}"), ("0.945540614608", "0.999746853967"))
        self.assertEqual((f"{low:.4f}", f"{high:.4f}"), ("0.9455", "0.9997"))

        def temperature(p):
            return 40 / math.log(p / (1 - p))
        self.assertEqual((f"{temperature(low):.3f}", f"{temperature(high):.3f}"), ("14.014", "4.830"))
        self.assertEqual((f"{temperature(0.9455):.4f}", f"{temperature(0.9997):.4f}"), ("14.0178", "4.9313"))
        ends = cal.temperatures(40, 99, 100)
        self.assertAlmostEqual(ends[0], temperature(high), places=9)
        self.assertAlmostEqual(ends[1], temperature(low), places=9)


class AFiniteLimitFarOut(unittest.TestCase):
    def test_the_rows_of_the_review(self):
        rows = ((1, 1, -1e200, "3.1831e-201"), (1, 0.005, 1e200, "0.95082813"),
                (1e-300, 0.005, 1e308, "0.99955155"), (1e-300, 0.001, 1e308, "0.87720826"))
        for u, dof, limit, model in rows:
            got = Result(0, u, dof=dof).prob_below(limit)
            self.assertEqual(f"{got:.4e}" if got < 1e-9 else f"{got:.8f}", model)
        # What Version 63 returned for the second row, at the cap of the distance: below 0.95, where the
        # probability is above it.
        capped = 1 - far_tail(153, 0.005) * 0.005 ** (-0.005 / 2)
        self.assertEqual(f"{capped:.4f}", "0.9144")

    def test_the_tail_from_its_density(self):
        for dof in (0.001, 0.005, 0.01, 0.5, 1, 2, 3.5):
            for exponent in (160, 200, 308, 400, 608):
                t = Fraction(10) ** exponent
                want = far_tail(exponent, dof)
                got = dist.t_sf(t, dof)
                if want > 1e-300:
                    self.assertAlmostEqual(got / want, 1, places=9, msg=(dof, exponent))
                else:
                    self.assertLess(got, 1e-300)
                self.assertAlmostEqual(dist.t_cdf(t, dof), 1 - want, places=12)
                self.assertAlmostEqual(dist.t_cdf(-t, dof) + dist.t_cdf(t, dof), 1, places=12)

    def test_one_and_two_degrees_of_freedom(self):
        # Closed forms: 1 / (pi t) far out with one degree of freedom, and 1 / (2 t^2) with two.
        for exponent in (155, 200, 300):
            want = math.exp(-math.log(math.pi) - exponent * math.log(10))
            self.assertAlmostEqual(dist.t_sf(10.0 ** exponent, 1) / want, 1, places=9)
            self.assertAlmostEqual(dist.t_sf(10 ** exponent, 1) / want, 1, places=9)
        self.assertAlmostEqual(dist.t_sf(1e155, 2) / 5e-311, 1, places=6)
        self.assertEqual(dist.t_sf(10 ** 400, 1), 0.0)                         # 3.2e-401 is below every float

    def test_the_tail_falls_across_the_join(self):
        for dof in (0.001, 0.01, 0.5, 1, 2):
            join = 1e153 * math.sqrt(dof)
            ts = [join / 1e3, join / 2, math.nextafter(join, 0), join, math.nextafter(join, math.inf), join * 2,
                  join * 1e3, Fraction(join) * 10 ** 200]
            tails = [dist.t_sf(t, dof) for t in ts]
            for near, far in zip(tails, tails[1:]):
                self.assertGreaterEqual(near * (1 + 1e-12), far, dof)
            self.assertGreater(tails[0], tails[-1])
            self.assertAlmostEqual(tails[2] / tails[4], 1, places=9)

    def test_a_limit_is_one_limit_in_every_type(self):
        r = Result(0, 1, dof=1)
        want = r.prob_below(-1e200)
        for limit in (-10 ** 200, Fraction(-10 ** 200), D("-1e200")):
            self.assertAlmostEqual(r.prob_below(limit) / want, 1, places=9)
        self.assertEqual(r.prob_below(10 ** 18), r.prob_below(1e18))
        self.assertGreater(r.prob_below(-10 ** 300), r.prob_below(-math.inf))   # 3.2e-301 against 0
        self.assertLess(Result(0, 1, dof=0.001).prob_below(10 ** 400), 1.0)    # not the probability of infinity
        self.assertEqual(r.prob_below(10 ** 400), 1.0)                         # a tail of 3.2e-401 is no float
        self.assertEqual(Result(0, 1).prob_below(10 ** 400), 1.0)

    def test_where_no_factor_is_needed(self):
        # A level whose tail is below the least normal float is refused where a factor has to be found.
        p = 1 - Fraction(1, 10 ** 330)
        exact = Result(5, 0)
        with self.assertRaisesRegex(ValueError, "p is too near 1"):
            exact.k(p)
        self.assertEqual((exact.interval(p), exact.expanded(p)), ((5, 5), 0.0))
        self.assertTrue(exact.report(p).startswith("(5 ± 0), u = 0, p = 0.9999"))
        for method in ("k", "interval", "expanded", "report"):
            with self.assertRaisesRegex(ValueError, "p is too near 1"):
                getattr(Result(5, 1), method)(p)


class RectangularHelpers(unittest.TestCase):
    ORIGIN = 10 ** 18

    def test_no_step_of_a_float_limits_an_exact_width(self):
        for half, p in ((150_050, 0.5), (150_050, 1), (1, 1), (1, Fraction(1, 3)), (D("0.001"), D("0.5"))):
            low, high = rect_interval(self.ORIGIN, half, p)                    # Version 63 refused all but one
            w = Fraction(p) * Fraction(half)
            self.assertEqual((low, high), (self.ORIGIN - w, self.ORIGIN + w))
            near = rect_interval(0, half, p)
            self.assertEqual((near[0] + self.ORIGIN, near[1] + self.ORIGIN), (low, high))

    def test_a_width_beyond_a_float(self):
        wide = 2 * 10 ** 308
        for f, root, digits in ((u_rect, 3, "1.15470e+308"), (u_tri, 6, "8.16497e+307"),
                                (u_resolution, 12, "5.77350e+307")):
            got = [f(kind(wide)) for kind in (int, Fraction, D)]              # Version 63: an error, or inf
            self.assertEqual(len(set(got)), 1)
            self.assertEqual(f"{got[0]:.5e}", digits)
            self.assertAlmostEqual(got[0] / (2 * (1e308 / math.sqrt(root))), 1, places=14)
            self.assertEqual(got[0], float(Fraction(wide) / Fraction(math.sqrt(root))))
        for kind in (int, Fraction, D):
            low, high = rect_interval(0, kind(wide), Fraction(1, 2))
            self.assertEqual((low, high), (-10 ** 308, 10 ** 308))
            for f in (u_rect, u_tri, u_resolution):
                with self.assertRaises(OverflowError):                         # the answer itself is beyond a float
                    f(kind(10 ** 310))

    def test_floats_as_before(self):
        rng = random.Random(4)
        for _ in range(3000):
            w = rng.choice((rng.uniform(0, 1e6), 10 ** rng.uniform(-300, 300), 0.0, 5e-324))
            self.assertEqual(u_rect(w), w / math.sqrt(3))
            self.assertEqual(u_tri(w), w / math.sqrt(6))
            self.assertEqual(u_resolution(w), w / math.sqrt(12))
        self.assertEqual(u_rect(3), 3 / math.sqrt(3))
        for bad in (-1, math.nan, math.inf, -0.5, D("Infinity")):
            for f in (u_rect, u_tri, u_resolution, lambda w: rect_interval(0, w)):
                with self.assertRaisesRegex(ValueError, "need a width >= 0"):
                    f(bad)

    def test_a_center_is_a_number(self):
        for half, p in ((0, 1), (1, 0), (1, 1), (0, 0)):
            for center in (math.nan, math.inf, -math.inf, D("NaN"), D("-Infinity")):
                with self.assertRaisesRegex(ValueError, "center is no number"):    # Version 63: (nan, nan)
                    rect_interval(center, half, p)
            for center in ("5", None):
                with self.assertRaises(TypeError):
                    rect_interval(center, half, p)
        self.assertEqual(rect_interval(5, 0, 1), (5, 5))
        self.assertEqual(rect_interval(0.1, 2, 0), (0.1, 0.1))
        self.assertEqual(rect_interval(10 ** 400, 1, 1), (10 ** 400 - 1, 10 ** 400 + 1))


class TheYearsOf256ns(unittest.TestCase):
    EPOCH = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)

    def instant(self, ns):
        seconds, rest = divmod(ns, 10 ** 9)
        at = self.EPOCH + datetime.timedelta(seconds=seconds)
        return f"{at:%H:%M:%S}.{rest:09d} UTC on {at.day} {at:%B %Y}"

    def test_the_two_instants(self):
        self.assertEqual(self.instant(2 ** 60), "23:58:24.606846976 UTC on 14 July 2006")
        self.assertEqual(self.instant(2 ** 61), "23:56:49.213693952 UTC on 25 January 2043")

    def test_the_step_on_either_side(self):
        self.assertEqual(math.ulp(float(2 ** 60)), 256)
        self.assertEqual(math.ulp(math.nextafter(float(2 ** 60), 0)), 128)
        self.assertEqual(math.ulp(math.nextafter(float(2 ** 61), 0)), 256)
        self.assertEqual(math.ulp(float(2 ** 61)), 512)
        midnight = datetime.datetime(2006, 7, 14, tzinfo=datetime.timezone.utc)
        ns = int((midnight - self.EPOCH).total_seconds()) * 10 ** 9
        self.assertLess(ns, 2 ** 60)
        self.assertEqual(math.ulp(float(ns)), 128)                             # midnight of that day is before it
        self.assertEqual(math.ulp(float(1_700_000_000 * 10 ** 9)), 256)        # a time stamp of 2023


if __name__ == "__main__":
    unittest.main()
