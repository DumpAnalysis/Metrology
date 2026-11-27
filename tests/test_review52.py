"""Regression tests added after the consolidated review of Version 52.

The review found that a count of zero was said to give no estimate, where the estimate is zero and only its
plug-in uncertainty fails; that the failure shares of Listing 16.3 were divided out to floats before they met
their limit; that completeness() of Listing 16.1 called the widest step a gap where no number was missing,
counted backward over repeated and restarted numbers and stopped on a single line, and that time_delta() let
a second start overwrite the first; that comparable() let any defining field be named as the variable under
study; that the first bucket of two histogram helpers began at zero, that a NaN passed for an observation and
that the sketch took any alpha; that ulp32() was wrong at zero and below the normal range; that confidence
limits next to zero or one were rounded away; that the compact form of an uncertainty was rounded twice; that
eight functions took arguments with no meaning; that read() took the bare words NaN and Infinity, and that a
stored outcome was never compared with the recomputed one; and that Chapters 14 and 22 treated equality at the
rejection boundary differently.

Each test computes what it expects independently of the code under test: tails of the Poisson and binomial
distributions from sums of terms formed with lgamma(), limits from closed forms, digits from the decimal
module, counts from sets of numbers, spacings from the bits of a binary32 value, and verdicts from products
of whole numbers.
"""
import contextlib
import copy
import io
import json
import math
import random
import struct
import unittest
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from statistics import NormalDist
from unittest import mock

from examples import ch09_histogram as h9
from examples import ch14_decisions as c14
from examples import ch16_bounds as bounds
from examples import ch16_silence as silence
from examples import ch16_trace_measurements as trace
from examples import ch19_float as f19
from examples import ch20_judge as judge
from examples import ch21_psi as drift
from examples import ch22_budget as budget
from examples import ch22_record_check as check
from examples import ch23_case as case
from examples import ch23_mix as mix
from swmetro import Result, core, rect_interval, u_rect, u_resolution, u_tri, welch_satterthwaite
from swmetro import record as R
from swmetro import stats

ROOT = Path(__file__).resolve().parent.parent
NEXT_TO_ONE = math.nextafter(1.0, 0.0)
STORED = (ROOT / "data" / "measurement_record.json").read_text(encoding="utf-8")


def expected(name):
    return (ROOT / "expected" / f"{name}.txt").read_text(encoding="utf-8")


def printed(function, *args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        function(*args)
    return out.getvalue()


def poisson_term(j, mean):
    return math.exp(j * math.log(mean) - mean - math.lgamma(j + 1))


def poisson_low(n, mean):
    """P(X <= n), each term from its logarithm."""
    return math.fsum(poisson_term(j, mean) for j in range(n + 1))


def poisson_high(n, mean):
    """P(X > n), summed until the terms no longer count."""
    terms, j = [], n + 1
    while True:
        terms.append(poisson_term(j, mean))
        if j > mean and terms[-1] < 1e-30 * max(terms):
            return math.fsum(terms)
        j += 1


def binom_term(j, n, p):
    log = math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1)
    return math.exp(log + j * math.log(p) + (n - j) * math.log1p(-p))


def binom_low(k, n, p):
    return math.fsum(binom_term(j, n, p) for j in range(k + 1))


def binom_high(k, n, p):
    return math.fsum(binom_term(j, n, p) for j in range(k + 1, n + 1))


class ZeroCount(unittest.TestCase):
    """02: at a count of zero the estimate is zero; what is reported is the one-sided upper bound."""

    def test_the_bound_of_thirty_silent_seconds(self):
        self.assertAlmostEqual(stats.quiet_bound(30), -math.log(0.05) / 30, places=15)
        self.assertEqual(f"{-math.log(0.05) / 30:.5f}", "0.09986")
        self.assertIn("  at most 0.09986 msg/s at 95 %\n", expected("ch16_trace_measurements"))

    def test_no_result_is_formed_and_none_is_called_missing(self):
        with self.assertRaisesRegex(ValueError, "a bound"):
            trace.current(0, 30)
        for text in (trace.current.__doc__, case.arrival.__doc__):
            self.assertNotIn("no estimate", " ".join(text.split()))
            self.assertIn("upper bound", " ".join(text.split()))
        self.assertIn("estimate of zero", " ".join(trace.current.__doc__.split()))
        self.assertIn("the estimate is zero", " ".join(case.arrival.__doc__.split()))

    def test_the_case_states_the_bound_for_its_window(self):
        top = -math.log(0.05) / case.WINDOW
        self.assertEqual(case.arrival(0), f"at most {top:.5f} req/s at 95 %")
        # Without exposure there is no finite bound to state.
        for seconds in (0, -1, math.nan):
            with self.assertRaisesRegex(ValueError, "no exposure"):
                case.arrival(0, seconds)
            with self.assertRaisesRegex(ValueError, "no exposure"):
                stats.quiet_bound(seconds)


class ExactShares(unittest.TestCase):
    """03: a failure share meets its limit as an exact fraction."""

    @staticmethod
    def by_products(total, captured, failures, num, den):
        """The verdict for a limit of num/den, from products of whole numbers alone."""
        missing = total - captured
        if (failures + missing) * den <= num * total:
            return "met"
        if failures * den > num * total:
            return "missed"
        return "unresolved"

    def test_the_two_rows_of_the_review(self):
        self.assertEqual(bounds.verdict(*bounds.share_bounds(10, 10, 1), Fraction(1, 10)), "met")
        big = bounds.share_bounds(10 ** 18, 10 ** 18, 10 ** 16 + 1)
        self.assertEqual(bounds.verdict(*big, 0.01), "missed")
        # Divided out, the first share lies above the fraction and the second is the float 0.01 itself.
        self.assertGreater(Fraction(1 / 10), Fraction(1, 10))
        self.assertEqual((10 ** 16 + 1) / 10 ** 18, 0.01)

    def test_equality_and_one_count_either_side(self):
        for total in (10, 100, 1000, 10 ** 6, 10 ** 18):
            for num, den, text in ((1, 10, "0.1"), (1, 100, "0.01"), (3, 100, "0.03"), (7, 100, "0.07")):
                if num * total % den:
                    continue
                at = num * total // den
                for limit in (float(text), text, Fraction(num, den), Decimal(text)):
                    self.assertEqual(bounds.verdict(*bounds.share_bounds(total, total, at), limit), "met", (total, limit))
                    self.assertEqual(bounds.verdict(*bounds.share_bounds(total, total, at - 1), limit), "met")
                    self.assertEqual(bounds.verdict(*bounds.share_bounds(total, total, at + 1), limit), "missed", (total, limit))

    def test_a_capture_with_missing_outcomes(self):
        low, high = bounds.share_bounds(1000, 960, 20)
        self.assertEqual((low, high), (Fraction(20, 1000), Fraction(60, 1000)))
        self.assertTrue(all(isinstance(x, Fraction) for x in (low, high)))
        for limit, want in ((0.06, "met"), (0.0599, "unresolved"), (0.02, "unresolved"), (0.0199, "missed"), (0, "missed"),
                            (1, "met")):
            self.assertEqual(bounds.verdict(low, high, limit), want, limit)

    def test_every_verdict_is_that_of_the_whole_numbers(self):
        rng = random.Random(5203)
        seen = set()
        for _ in range(4000):
            total = rng.choice((7, 10, 64, 1000, 10 ** 9, 10 ** 18))
            captured = rng.randint(max(0, total - rng.choice((0, 0, 1, total // 20 + 1))), total)
            den = rng.choice((10, 100, 1000))
            num = rng.randint(0, den)
            near = num * total // den
            failures = min(captured, max(0, near + rng.randint(-2, 2) - rng.choice((0, total - captured))))
            want = self.by_products(total, captured, failures, num, den)
            got = bounds.verdict(*bounds.share_bounds(total, captured, failures), float(Fraction(num, den)))
            self.assertEqual(got, want, (total, captured, failures, num, den))
            seen.add(want)
        self.assertEqual(seen, {"met", "missed", "unresolved"})

    def test_the_limit_is_read_as_a_probability(self):
        for limit in (-0.01, 1.01, math.nan):
            with self.assertRaises(ValueError):
                bounds.verdict(Fraction(1, 50), Fraction(3, 50), limit)

    def test_the_transcript_is_unchanged(self):
        self.assertEqual(printed(bounds.main), expected("ch16_bounds"))
        self.assertIn("failure share: 2 % to 6 %\n", expected("ch16_bounds"))


def rows_of(marks):
    return [(n, ms, "db", "query done") for n, ms in marks]


def by_sets(marks, counter=None):
    """Lost, sent and the span of the widest run of missing numbers, from the set of numbers assigned."""
    seen = [n for n, _ in marks]
    first, after = counter if counter else (seen[0], seen[-1] + 1)
    assigned = set(range(first, after))
    widest, gap = 0, None
    for (p, t0), (q, t1) in zip(marks, marks[1:]):
        absent = len(set(range(p + 1, q)))
        if absent > widest:
            widest, gap = absent, t1 - t0
    return len(assigned - set(seen)), len(assigned), gap


class TraceEvidence(unittest.TestCase):
    """04: sequence numbers, gaps, the edges of the capture, and repeated starts."""

    def test_the_four_rows_of_the_review(self):
        self.assertEqual(trace.completeness(rows_of(((1, 0), (2, 10), (3, 20)))), (0, 3, None))
        for marks in (((1, 0), (1, 10)), ((7, 0), (8, 10), (1, 20)), ((1, 0), (3, 10), (2, 20))):
            with self.assertRaisesRegex(ValueError, "numbers do not rise"):
                trace.completeness(rows_of(marks))
        self.assertEqual(trace.completeness(rows_of(((5, 0),))), (0, 1, None))
        # What the counter establishes for the single line: it is one of 799 that were assigned.
        self.assertEqual(trace.completeness(rows_of(((5, 0),)), (1, 800)), (798, 799, None))

    def test_the_edges_are_counted_with_the_counter_alone(self):
        marks = ((3, 0), (4, 10), (6, 45))
        self.assertEqual(trace.completeness(rows_of(marks)), (1, 4, 35))
        self.assertEqual(trace.completeness(rows_of(marks), (1, 10)), (6, 9, 35))
        self.assertEqual(trace.completeness(rows_of(marks), (3, 7)), (1, 4, 35))
        # Lines that the counter never assigned belong to another stream.
        for counter in ((4, 10), (1, 6), (7, 20)):
            with self.assertRaisesRegex(ValueError, "beyond the counter"):
                trace.completeness(rows_of(marks), counter)
        # No line: with the counter all that was assigned is lost, and without it there is nothing to count.
        self.assertEqual(trace.completeness([], (1, 5)), (4, 4, None))
        with self.assertRaisesRegex(ValueError, "no line, no counter"):
            trace.completeness([])

    def test_every_count_is_that_of_the_sets(self):
        rng = random.Random(5204)
        for _ in range(600):
            n, ms, marks = rng.randint(1, 50), 0, []
            for _ in range(rng.randint(1, 12)):
                marks.append((n, ms))
                n += rng.choice((1, 1, 1, 2, 3, 7))
                ms += rng.randint(0, 40)
            first, last = marks[0][0], marks[-1][0]
            counter = (first - rng.randint(0, 3), last + 1 + rng.randint(0, 3))
            self.assertEqual(trace.completeness(rows_of(marks)), by_sets(marks), marks)
            self.assertEqual(trace.completeness(rows_of(marks), counter), by_sets(marks, counter), marks)

    def test_the_fixture(self):
        lines = trace.LOG.read_text(encoding="utf-8").splitlines()
        rows = [trace.parse(x) for x in lines]
        numbers = [int(x.split()[0]) for x in lines]
        self.assertEqual((numbers[0], numbers[-1], len(numbers)), (1, 799, 759))
        self.assertEqual(trace.COUNTER, (1, 800))
        # The counter was at 800 when the capture stopped: 799 numbers were assigned, and 40 are not in the file.
        self.assertEqual(trace.completeness(rows, trace.COUNTER)[:2], (len(set(range(1, 800)) - set(numbers)), 799))
        self.assertEqual(trace.completeness(rows, trace.COUNTER), by_sets([(r[0], r[1]) for r in rows], (1, 800)))
        self.assertEqual(trace.completeness(rows, trace.COUNTER), (40, 799, 2088))
        text = expected("ch16_trace_measurements")
        self.assertIn("lost 40 of 799 messages (5.0 %)\nsequence gap spans 2088 ms\n", text)
        self.assertEqual(printed(trace.main), text)

    def test_one_start_and_one_end(self):
        def line(n, ms, message):
            return (n, ms, "svc", message)
        pair = ("request 17 start", "request 17 end")
        self.assertEqual(str(trace.time_delta([line(1, 0, pair[0]), line(2, 100, pair[1])])), "100.00(41) ms")
        self.assertAlmostEqual(trace.time_delta([line(1, 0, pair[0]), line(2, 100, pair[1])]).u, 1 / math.sqrt(6), places=15)
        again = [line(1, 0, pair[0]), line(2, 100, pair[1]), line(3, 150, pair[0])]     # Version 52: -50.00(41) ms
        cases = (again, [line(1, 0, pair[0]), line(2, 50, pair[0]), line(3, 100, pair[1])], [line(1, 0, pair[0])],
                 [line(1, 100, pair[1])], [line(1, 0, pair[1]), line(2, 100, pair[0])], [],
                 [line(1, 0, pair[0]), line(2, 100, pair[1]), line(3, 150, pair[1])])
        for rows in cases:
            with self.assertRaisesRegex(ValueError, "no single pair"):
                trace.time_delta(rows)
        with self.assertRaisesRegex(ValueError, "stamps go back"):
            trace.time_delta([line(1, 100, pair[0]), line(2, 40, pair[1])])
        # Other messages are not looked at, and another pair can be named.
        mixed = [line(1, 0, "request 170 start"), line(2, 5, pair[0]), line(3, 9, "request 1 end"), line(4, 25, pair[1])]
        self.assertEqual(trace.time_delta(mixed).value, 20)
        self.assertEqual(trace.time_delta(mixed, ("request 170 start", "request 1 end")).value, 9)

    def test_the_fixture_has_one_pair(self):
        lines = trace.LOG.read_text(encoding="utf-8").splitlines()
        own = [x for x in lines if "request 17 " in x]
        self.assertEqual([x.split(None, 4)[4] for x in own], ["request 17 start", "request 17 end"])
        stamps = [trace.stamp(x.split()[2]) for x in own]
        rows = [trace.parse(x) for x in lines]
        self.assertEqual(trace.time_delta(rows).value, stamps[1] - stamps[0])
        self.assertEqual(stamps[1] - stamps[0], 2346)

    def test_a_trace_without_the_pair_is_still_read_for_its_silence(self):
        span = tuple(trace.stamp(x) for x in trace.CAPTURE)
        out = printed(silence.models, [], span, (1, 1))
        self.assertIn("  no request to take the time\n  between a start and an end from\n", out)


class HandTrace(unittest.TestCase):
    """The five lines that Chapter 16 prints for the reader to work by hand, and what the listing makes of them."""
    LINES = ("000041 2204 09:15:00.100 svc  request 9 start",
             "000042 3120 09:15:00.250 db   query 3310 done",
             "000044 2204 09:15:00.400 svc  request 9 start",
             "000045 2204 09:15:00.700 svc  request 9 end",
             "000046 3120 09:15:00.900 db   query 3311 done")
    CAPTURE = ("09:15:00.000", "09:15:01.000")
    COUNTER = (41, 48)

    def setUp(self):
        self.rows = [trace.parse(x) for x in self.LINES]
        self.span = tuple(trace.stamp(x) for x in self.CAPTURE)

    def test_the_loss(self):
        # The counter assigned 41 to 47: seven numbers, five lines. Number 43 is missing between two lines, 150 ms
        # apart, and number 47 after the last line, which the lines alone do not show.
        assigned = set(range(41, 48))
        self.assertEqual(sorted(assigned - {r[0] for r in self.rows}), [43, 47])
        self.assertEqual(trace.completeness(self.rows, self.COUNTER), (2, 7, 400 - 250))
        self.assertEqual(trace.completeness(self.rows), (1, 6, 150))

    def test_the_duration(self):
        # Two starts and one end: 600 ms from the first, 300 ms from the second, and no line says which.
        starts = [r[1] for r in self.rows if r[3] == "request 9 start"]
        ends = [r[1] for r in self.rows if r[3] == "request 9 end"]
        self.assertEqual([ends[0] - s for s in starts], [600, 300])
        with self.assertRaisesRegex(ValueError, "no single pair"):
            trace.time_delta(self.rows, ("request 9 start", "request 9 end"))

    def test_the_rate_statement(self):
        # Verified: from the start of the capture to the line at 250 ms, since the counter stood at the number of
        # the first line, and from 400 ms to 900 ms. After 900 ms a number is missing.
        found = trace.stretches(self.rows, 0, 1, self.span, self.COUNTER)
        self.assertEqual(found, [(0, 250), (400, 900)])
        seconds = trace.verified(self.rows, 0, 1, self.span, self.COUNTER)
        self.assertEqual(seconds, 0.75)
        self.assertEqual(trace.count(self.rows, "gc", 0, 1, self.span), 0)
        self.assertEqual(f"{-math.log(0.05) / 0.75:.2f}", "3.99")
        self.assertAlmostEqual(stats.quiet_bound(seconds), -math.log(0.05) / 0.75, places=14)
        # The whole second would give 3.00, and without the counter only the two runs of lines are verified.
        self.assertEqual(f"{-math.log(0.05) / 1:.2f}", "3.00")
        self.assertEqual(trace.verified(self.rows, 0, 1, self.span), 0.65)
        self.assertEqual(f"{-math.log(0.05) / 0.65:.2f}", "4.61")
        self.assertAlmostEqual(stats.quiet_bound(0.65), -math.log(0.05) / 0.65, places=14)
        # The database count: two lines, and up to two more among the lost.
        self.assertEqual(trace.count(self.rows, "db", 0, 1, self.span), 2)


class Comparable(unittest.TestCase):
    """05: only the conditions can be the variable under study."""

    def setUp(self):
        self.rec = R.read(STORED)

    def changed(self, section, key, value):
        other = copy.deepcopy(self.rec)
        other[section][key] = value
        return other

    def test_the_bypass_of_the_review(self):
        other = self.changed("data", "run_weighting", "by requests")
        self.assertEqual(R.comparable(self.rec, other), (["run_weighting"], []))
        with self.assertRaisesRegex(ValueError, "only conditions vary"):
            R.comparable(self.rec, other, across=("run_weighting",))                # Version 52: ([], ["across run_weighting"])

    def test_what_defines_the_quantity_cannot_be_studied_away(self):
        fields = (("measurand", "quantity"), ("measurand", "convention"), ("measurand", "target"),
                  ("measurand", "start_event"), ("measurand", "end_event"), ("data", "run_weighting"))
        for section, key in fields:
            other = self.changed(section, key, "something else")
            self.assertEqual(R.comparable(self.rec, other), ([key], []), key)
            for across in ((key,), ("conditions", key), [key], ("unit",), ("clock",), ("nothing",)):
                with self.assertRaisesRegex(ValueError, "only conditions vary"):
                    R.comparable(self.rec, other, across=across)
            # With the conditions under study the difference is still a reason.
            self.assertEqual(R.comparable(self.rec, other, across=("conditions",)), ([key], []), key)

    def test_the_conditions_can(self):
        other = self.changed("measurand", "conditions", "twice the test load")
        self.assertEqual(R.comparable(self.rec, other), (["conditions"], []))
        self.assertEqual(R.comparable(self.rec, other, across=("conditions",)), ([], ["across conditions"]))
        self.assertEqual(R.comparable(self.rec, other, across=["conditions"]), ([], ["across conditions"]))
        self.assertEqual(R.comparable(self.rec, self.rec, across=()), ([], []))
        self.assertEqual(R.VARIED, ("conditions",))


class Histograms(unittest.TestCase):
    """06: the first bucket, values that are no observations, and the accuracy of the sketch."""

    @staticmethod
    def p95(xs):
        return sorted(xs)[math.ceil(Fraction(95, 100) * len(xs)) - 1]

    def test_the_first_bucket_has_no_lower_end(self):
        for xs in ([0.0] * 20, [-0.9] * 20, [-3.0] * 19 + [49.0]):
            lo, hi = budget.bracket(xs, [50, 100])
            self.assertEqual((lo, hi), (-math.inf, 50.0))
            self.assertTrue(lo < self.p95(xs) <= hi)                              # Version 52: (0.0, 50.0), p95 outside
            self.assertEqual(budget.verdict(lo, hi), "met")
        self.assertEqual(budget.bracket([60.0] * 20, [50, 100]), (50.0, 100.0))
        self.assertEqual(budget.bracket([500.0] * 20, [50, 100]), (100.0, math.inf))

    def test_the_bracket_holds_the_quantile(self):
        rng = random.Random(5206)
        for _ in range(500):
            xs = [round(rng.uniform(-20, 260), rng.choice((0, 1))) for _ in range(rng.randint(1, 60))]
            cuts = sorted(rng.sample(range(-10, 250), 4))
            lo, hi = budget.bracket(xs, cuts)
            q = self.p95(xs)
            self.assertTrue(lo < q <= hi, (xs, cuts))
            self.assertEqual(lo == -math.inf, q <= cuts[0])
            self.assertIn(lo, [-math.inf] + cuts)

    def test_the_transcript_names_no_first_bucket(self):
        text = expected("ch22_budget")
        self.assertEqual(printed(budget.main), text)
        self.assertNotIn("(0, ", text)
        self.assertNotIn("(-inf", text)

    def test_only_observations_enter_a_bucket_quantile(self):
        # Since Version 57 a NaN is refused by known() of the package, before the sign is asked for.
        for xs, why in (([math.nan], "a duration is missing"), ([0.1, math.nan], "a duration is missing"),
                        ([-0.001], "need values >= 0"), ([0.2, -5], "need values >= 0")):
            with self.assertRaisesRegex(ValueError, why):
                h9.bucket_quantile(xs, 0.95)                                      # Version 52: 0.00475 for [nan]
        self.assertTrue(math.isnan(h9.bucket_quantile([], 0.95)))
        # Twenty zeros lie in the first bucket, which begins at 0: rank 19 of its 20, interpolated.
        self.assertAlmostEqual(h9.bucket_quantile([0.0] * 20, 0.95, [50, 100]), 50 * 19 / 20, places=12)
        # A value beyond the last bound lies in the open-ended bucket. Until Version 56 an infinity was
        # counted there too; since Version 57 it is no observed duration, and known() of the package refuses it.
        self.assertEqual(h9.bucket_quantile([0.0, 1e9], 1, [50, 100]), 100)
        with self.assertRaisesRegex(ValueError, "a duration is missing or too large"):
            h9.bucket_quantile([0.0, math.inf], 1, [50, 100])

    def test_the_convention_at_zero(self):
        self.assertEqual(h9.bucket_quantile([1], 0, [0.25, 0.5, 1, 2]), 0.5)
        self.assertEqual(h9.bucket_quantile([0.1], 0, [0.25, 0.5, 1, 2]), 0.0)
        self.assertIn("For q = 0 the result is the lower end of the first bucket that holds a value",
                      " ".join(h9.bucket_quantile.__doc__.split()))

    def test_the_sketch_takes_an_accuracy(self):
        for alpha in (-0.1, 0, 1, 1.5, math.nan, math.inf):
            with self.assertRaisesRegex(ValueError, "need 0 < alpha < 1"):
                h9.LogSketch(alpha)
        for alpha in (1e-20, 1e-17):
            with self.assertRaisesRegex(ValueError, "alpha too small"):
                h9.LogSketch(alpha)                                               # Version 52: ZeroDivisionError on add()
        self.assertEqual((1 + 1e-17) / (1 - 1e-17), 1.0)

    def test_the_sketch_keeps_it(self):
        # The lower order statistic at rank floor(q * (n - 1)), from a sorted list, within alpha. The bounds are formed
        # from the same logarithm as the indexes. With the logarithm from alpha and the bounds from powers of the
        # rounded g, an alpha of 1e-9 was missed 270-fold while the change was made; no version had that.
        rng = random.Random(5209)
        for alpha in (0.3, 0.01, 1e-4, 1e-6, 1e-9, 1e-12):
            for _ in range(20):
                xs = [rng.lognormvariate(0, 3) for _ in range(40)]
                sk = h9.sketch_of(xs, alpha)
                for q in (0, 0.25, 0.5, 0.95, 1):
                    want = sorted(xs)[math.floor(q * (len(xs) - 1))]
                    self.assertLessEqual(abs(sk.quantile(q) - want), 1.001 * alpha * want, (alpha, q))

    def test_the_transcript_of_chapter_nine_is_unchanged(self):
        self.assertEqual(printed(h9.compare) + printed(h9.averaging), expected("ch09_histogram"))


def above32(bits):
    """The binary32 value with these bits and the next one above it."""
    def value(b):
        return struct.unpack("<f", struct.pack("<I", b))[0]
    return value(bits), value(bits + 1)


class Binary32(unittest.TestCase):
    """07: the spacing of binary32 values, from their bits."""

    def test_zero_and_the_subnormal_range(self):
        least = above32(0)[1]
        self.assertEqual(least, 2.0 ** -149)
        for bits in (0, 1, 2, 12345, 0x007FFFFF, 0x00800000):                    # zero, subnormals, the least normal value
            x, up = above32(bits)
            self.assertEqual(f19.ulp32(x), up - x, hex(bits))
            self.assertEqual(up - x, least, hex(bits))
            self.assertEqual(f19.ulp32(-x), least)
        self.assertEqual(above32(0x00800000)[0], 2.0 ** -126)
        # Version 52 gave 2**-24 at zero and 2**-172 at the least subnormal value.

    def test_the_normal_range(self):
        rng = random.Random(5207)
        cases = [0x3F800000, 0x4B800000, 0x7F7FFFFE, 0x00FFFFFF, 0x01000000] + [rng.randrange(0x00800000, 0x7F7FFFFF)
                                                                               for _ in range(2000)]
        for bits in cases:
            x, up = above32(bits)
            self.assertEqual(f19.ulp32(x), up - x, hex(bits))
            self.assertEqual(f19.ulp32(-x), up - x, hex(bits))
            between = x + (up - x) / 4                                            # no binary32 value: the step of its neighbors
            self.assertEqual(f19.ulp32(between), up - x, hex(bits))
        self.assertEqual((f19.ulp32(1.0), f19.ulp32(16777216.0)), (2.0 ** -23, 2.0))

    def test_what_has_no_spacing(self):
        largest = above32(0x7F7FFFFE)[1]
        self.assertEqual(f19.ulp32(largest), 2.0 ** 104)
        for x in (math.inf, -math.inf, math.nan, 2.0 ** 128, -1e39, 1e300):
            with self.assertRaisesRegex(ValueError, "no binary32 value"):
                f19.ulp32(x)

    def test_the_reference_sum_is_named_as_the_text_names_it(self):
        self.assertIn("against their exactly rounded sum", " ".join(f19.order.__doc__.split()))
        self.assertEqual(printed(f19.main), expected("ch19_float"))


class RareTails(unittest.TestCase):
    """08: a limit next to zero, or a level next to zero or one, keeps its digits."""

    def test_the_five_values_of_the_review(self):
        a = (1 - NEXT_TO_ONE) / 2
        self.assertEqual(1 - a, 1.0)                                              # why 1 less the other tail fails
        low = stats.rate_limits(1, 1, NEXT_TO_ONE)[0]
        # Since Version 70 a lower limit is moved downward by 1e-12 of itself, so that it is not above the limit
        # of the model (tests/test_review69.py): the tolerance was 1e-12 until then.
        self.assertTrue(math.isclose(low, -math.log1p(-a), rel_tol=3e-12), low)   # Version 52: 3.11e-61
        self.assertLessEqual(low, -math.log1p(-a))
        low = judge.exact_interval(1, 10, NEXT_TO_ONE)[0]
        self.assertTrue(math.isclose(low, -math.expm1(math.log1p(-a) / 10), rel_tol=1e-12), low)   # 1.08e-17
        k = Result(0, 1).k(1e-18)
        self.assertTrue(math.isclose(k, math.sqrt(2 * math.pi) * 5e-19, rel_tol=1e-12), k)        # 0.0
        top = stats.rate_upper(0, 1, 1e-18)
        self.assertTrue(math.isclose(top, 1e-18, rel_tol=1e-12), top)                             # 3.11e-61
        z = math.sqrt(2 * math.pi) * 5e-19
        low, high = stats.wilson(0, 10, 1e-18)
        self.assertEqual(low, 0.0)
        self.assertTrue(math.isclose(high, z * z / (10 + z * z), rel_tol=1e-12), high)            # 0.0

    def test_poisson_limits_solve_their_equations(self):
        for n in (1, 2, 5, 40):
            for level in (0.5, 0.9, 0.95, 1 - 1e-9, NEXT_TO_ONE):
                a = (1 - level) / 2
                low, high = stats.rate_limits(n, 1.0, level)
                self.assertTrue(math.isclose(poisson_high(n - 1, low), a, rel_tol=1e-9), (n, level))
                self.assertTrue(math.isclose(poisson_low(n, high), a, rel_tol=1e-9), (n, level))
                scaled = stats.rate_limits(n, 2.5, level)
                self.assertTrue(math.isclose(scaled[0], low / 2.5, rel_tol=1e-15))
                self.assertTrue(math.isclose(scaled[1], high / 2.5, rel_tol=1e-15))
        self.assertEqual(stats.rate_limits(0, 1.0, NEXT_TO_ONE)[0], 0.0)

    def test_the_first_event_has_closed_forms(self):
        # P(X >= 1) = 1 - exp(-m) = a, and P(X = 0) = exp(-m) = a.
        # The tail is taken from the level as a float holds it: 1 - 2e-10 is not that number, and its tail is
        # 1.00000008e-10.
        for level in (0.5, 0.998, 1 - 2e-10, NEXT_TO_ONE):
            a = (1 - level) / 2
            low, high = stats.rate_limits(1, 1, level)
            self.assertTrue(math.isclose(low, -math.log1p(-a), rel_tol=1e-11), a)
            self.assertTrue(math.isclose(stats.rate_limits(0, 1, level)[1], -math.log(a), rel_tol=1e-11), a)
        self.assertNotEqual((1 - (1 - 2e-10)) / 2, 1e-10)

    def test_upper_bounds_at_every_level(self):
        for level in (1e-300, 1e-18, 1e-6, 0.3, 0.49, 0.95, NEXT_TO_ONE):
            top = stats.rate_upper(0, 1, level)
            self.assertTrue(math.isclose(top, -math.log1p(-level), rel_tol=1e-12), level)
            # Without an event the bound is the closed form itself, to the last digit.
            self.assertEqual(top, stats.quiet_bound(1, level), level)
            self.assertEqual(stats.rate_upper(0, 30, level), stats.quiet_bound(30, level), level)
        for n in (0, 1, 3, 20):
            for level in (1e-18, 1e-6, 0.3, 0.49):
                top = stats.rate_upper(n, 1, level)
                self.assertTrue(math.isclose(poisson_high(n, top), level, rel_tol=1e-9), (n, level))
            for level in (0.5, 0.9, 0.95, 1 - 1e-12):
                top = stats.rate_upper(n, 4, level)
                self.assertTrue(math.isclose(poisson_low(n, 4 * top), 1 - level, rel_tol=1e-9), (n, level))

    def test_both_poisson_tails(self):
        for n, mean in ((0, 1e-30), (0, 0.3), (3, 2.5), (10, 40.0), (60, 40.0), (2, 1e-9)):
            low, high = stats.poisson_tails(n, mean)
            self.assertTrue(math.isclose(low, poisson_low(n, mean), rel_tol=1e-12), (n, mean))
            self.assertTrue(math.isclose(high, poisson_high(n, mean), rel_tol=1e-12), (n, mean))
            self.assertEqual(stats.poisson_at_most(n, mean), low)
        self.assertTrue(math.isclose(stats.poisson_tails(0, 1e-30)[1], 1e-30, rel_tol=1e-12))
        self.assertEqual(stats.poisson_tails(-1, 2.0), (0.0, 1.0))
        self.assertEqual(stats.poisson_tails(3, 0), (1.0, 0.0))

    def test_binomial_limits_solve_their_equations(self):
        for k, n in ((1, 10), (3, 20), (17, 20), (500, 1000)):
            for level in (0.9, 0.95, 1 - 1e-9):
                a = (1 - level) / 2
                low, high = judge.exact_interval(k, n, level)
                self.assertTrue(math.isclose(binom_high(k - 1, n, low), a, rel_tol=1e-9), (k, n, level))
                self.assertTrue(math.isclose(binom_low(k, n, high), a, rel_tol=1e-9), (k, n, level))
        for k, n in ((1, 10), (3, 20), (2, 1000)):
            a = (1 - NEXT_TO_ONE) / 2
            low = judge.exact_interval(k, n, NEXT_TO_ONE)[0]
            self.assertTrue(math.isclose(binom_high(k - 1, n, low), a, rel_tol=1e-9), (k, n))

    def test_the_ends_have_closed_forms(self):
        # No success: (1 - p)**n = a. All successes: p**n = a. One success, lower limit: 1 - (1 - p)**n = a.
        for n in (1, 10, 1000):
            for a in (0.025, 1e-9, 1e-300, (1 - NEXT_TO_ONE) / 2):
                level = 1 - 2 * a
                if (1 - level) / 2 != a:
                    continue
                low, high = judge.exact_interval(0, n, level)
                self.assertEqual(low, 0.0)
                self.assertTrue(math.isclose(high, -math.expm1(math.log(a) / n), rel_tol=1e-12), (n, a))
                low, high = judge.exact_interval(n, n, level)
                self.assertEqual(high, 1.0)
                self.assertTrue(math.isclose(low, math.exp(math.log(a) / n), rel_tol=1e-12), (n, a))
                if n > 1:
                    low = judge.exact_interval(1, n, level)[0]
                    self.assertTrue(math.isclose(low, -math.expm1(math.log1p(-a) / n), rel_tol=1e-12), (n, a))

    def test_both_binomial_tails(self):
        for k, n, p in ((0, 10, 1e-30), (2, 20, 0.15), (7, 40, 0.5), (39, 40, 0.999), (0, 1, 0.25), (500, 1000, 0.45)):
            low, high = judge.tails(k, n, p)
            self.assertTrue(math.isclose(low, binom_low(k, n, p), rel_tol=1e-11), (k, n, p))
            self.assertTrue(math.isclose(high, binom_high(k, n, p), rel_tol=1e-11), (k, n, p))
            self.assertEqual(judge.at_most(k, n, p), low)
        self.assertTrue(math.isclose(judge.tails(0, 10, 1e-30)[1], 1e-29, rel_tol=1e-12))
        self.assertEqual((judge.tails(-1, 5, 0.5), judge.tails(5, 5, 0.5)), ((0.0, 1.0), (1.0, 0.0)))
        self.assertEqual((judge.tails(2, 5, 0), judge.tails(2, 5, 1)), ((1.0, 0.0), (0.0, 1.0)))

    def test_coverage_factors_below_one_half(self):
        # One degree of freedom: P(|T| < k) = (2 / pi) atan(k). Two: k / sqrt(2 + k * k). Many: the normal distribution.
        for p in (1e-18, 1e-9, 1e-3, 0.1, 0.3, 0.49, 0.5, 0.9):
            self.assertTrue(math.isclose(Result(0, 1, dof=1).k(p), math.tan(math.pi * p / 2), rel_tol=1e-9), p)
            self.assertTrue(math.isclose(Result(0, 1, dof=2).k(p), p * math.sqrt(2 / (1 - p * p)), rel_tol=1e-9), p)
        for p in (1e-3, 0.1, 0.3, 0.49, 0.5, 0.9, 0.95):
            self.assertTrue(math.isclose(Result(0, 1).k(p), NormalDist().inv_cdf(0.5 + p / 2), rel_tol=1e-9), p)
        for p in (1e-18, 1e-12, 1e-6):
            self.assertTrue(math.isclose(Result(0, 1).k(p), math.sqrt(2 * math.pi) * p / 2, rel_tol=1e-9), p)
        for dof in (1, 2, 7.5, math.inf):
            self.assertEqual(Result(0, 1, dof=dof).k(0), 0.0)
            self.assertEqual(Result(3, 2, dof=dof).interval(0), (3, 3))

    def test_wilson_at_ordinary_levels_is_what_it_was(self):
        z = NormalDist().inv_cdf(0.975)
        for k, n in ((0, 10), (3, 20), (10, 10), (870, 1000)):
            ph = k / n
            mid = (ph + z * z / (2 * n)) / (1 + z * z / n)
            half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
            low, high = stats.wilson(k, n)
            self.assertAlmostEqual(low, mid - half if k else 0.0, places=14)
            self.assertAlmostEqual(high, mid + half if k < n else 1.0, places=14)

    def test_the_transcripts_are_unchanged(self):
        self.assertEqual(printed(judge.main), expected("ch20_judge"))


def two_digits(u):
    """The decimal places and the two significant digits of a float, rounded once from the number that it is."""
    with localcontext() as ctx:
        ctx.prec = 2000
        d = Decimal(u)
        exp = d.adjusted()
        q = d.quantize(Decimal(1).scaleb(exp - 1), rounding=ROUND_HALF_EVEN)
        if q.adjusted() > exp:                       # 0.0996 is shown as 0.10
            exp += 1
            q = d.quantize(Decimal(1).scaleb(exp - 1), rounding=ROUND_HALF_EVEN)
        return 1 - exp, int(q.scaleb(1 - exp))


def to_place(value, places):
    with localcontext() as ctx:
        ctx.prec = 2000
        return Decimal(value).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_EVEN)


class CompactUncertainty(unittest.TestCase):
    """09: the digits in parentheses are those of the float, rounded once."""

    def test_the_four_cases_of_the_review(self):
        self.assertEqual(str(Result(1, 1.6500000000000001)), "1.0(17)")           # Version 52: 1.0(16)
        self.assertEqual(str(Result(0, 0.165)), "0.00(17)")                       # 0.00(16)
        self.assertEqual(str(Result(0, 0.155)), "0.00(15)")                       # 0.00(16)
        self.assertEqual((f"{0.165:.2f}", f"{0.155:.2f}", f"{1.6500000000000001:.1f}"), ("0.17", "0.15", "1.7"))
        text = str(Result(0, 1e-310))                                             # OverflowError
        self.assertEqual((text[:4], text[-4:], len(text)), ("0.00", "(10)", 2 + 311 + 4))

    def test_the_two_neighbors_of_a_half_step(self):
        # A decimal half-step is no float. Of the two floats next to it one lies below and one above, and their
        # digits differ by one. The float that Python stores for the decimal is one of the two.
        for text in ("1.65", "0.165", "0.155", "2.55e-7", "34.5", "0.0125", "8.45e-5", "0.125", "62.5", "7.35e9"):
            half = Fraction(text)
            stored = float(text)
            if Fraction(stored) == half:           # a half-step that is a float: the tie goes to the even digit
                digits = two_digits(stored)[1]
                self.assertEqual(digits % 2, 0, text)
                below, above = math.nextafter(stored, 0), math.nextafter(stored, math.inf)
            elif Fraction(stored) < half:
                below, above = stored, math.nextafter(stored, math.inf)
            else:
                below, above = math.nextafter(stored, 0), stored
            self.assertTrue(Fraction(below) < half < Fraction(above), text)
            low, high = two_digits(below), two_digits(above)
            self.assertEqual((high[0], high[1] - low[1]), (low[0], 1), text)
            for u, (places, digits) in ((below, low), (above, high)):
                self.assertEqual(core.decimals(u), places, u)
                shown = digits * 10 ** max(0, -places)     # in units of the last place shown, which may be the tens
                self.assertTrue(core.fmt(0, u).endswith(f"({shown})"), (u, core.fmt(0, u)))

    def test_the_digits_are_those_of_the_decimal_module(self):
        rng = random.Random(5209)
        us = [k / 1000 * 10.0 ** e for k in range(100, 1000, 5) for e in (-9, -3, -1, 0, 1)]
        us += [rng.uniform(1, 10) * 10.0 ** rng.randint(-12, 2) for _ in range(3000)]
        us += [0.0996, 0.995, 9.95, 0.0995, 9.949999, 1e-300, 5e-324, 0.125, 0.135, 0.145]
        for u in us:
            places, digits = two_digits(u)
            self.assertEqual(core.decimals(u), places, u)
            if places > 0:
                self.assertEqual(core.fmt(0, u), f"{0:.{places}f}({digits})", u)
                # ... and those that a format of the same float shows.
                self.assertEqual(int(f"{u:.{places}f}".replace(".", "")), digits, u)

    def test_tens_and_above(self):
        rng = random.Random(5210)
        cases = [(1250.0, 450.0), (1250.0, 4500.0), (1350, 4500), (1234.0, 120.0), (125.0, 20.0), (-1250.0, 4500.0),
                 (Fraction(2501, 2), 4500.0), (15.0, 99.6), (0, 995), (10 ** 19 + 1, 1.5e3)]
        cases += [(rng.uniform(-1e6, 1e6), rng.uniform(1, 10) * 10.0 ** rng.randint(1, 5)) for _ in range(2000)]
        for value, u in cases:
            places, digits = two_digits(u)
            self.assertLessEqual(places, 0)
            self.assertEqual(core.decimals(u), places, u)
            v = int(to_place(Fraction(value).numerator, places)) if Fraction(value).denominator == 1 else None
            if v is None:
                v = int(to_place(Decimal(Fraction(value).numerator) / Decimal(Fraction(value).denominator), places))
            self.assertEqual(core.fmt(value, u), f"{v}({digits * 10 ** -places})", (value, u))
        self.assertEqual(core.fmt(1250.0, 4500.0), "1200(4500)")
        self.assertEqual(core.fmt(1350, 4500), "1400(4500)")

    def test_what_did_not_change(self):
        self.assertEqual(core.fmt(12.311, 0.0097, "ms"), "12.3110(97) ms")
        self.assertEqual(core.fmt(99.9996, 0, "ms"), "99.9996(0) ms")
        self.assertEqual(str(Result(5, 0.0996)), "5.00(10)")
        self.assertEqual(core.decimals(0), 3)


class Arguments(unittest.TestCase):
    """10: arguments with no interpretation as a measurement are refused, and the degenerate ones are kept."""

    def refused(self, call, text):
        with self.assertRaisesRegex(ValueError, text):
            call()

    def test_the_eight_calls_of_the_review(self):
        self.refused(lambda: Result(math.nan, 0).prob_below(200), "value is no number")           # Version 52: 0.0
        self.refused(lambda: rect_interval(0, 1, 1.2), "need 0 <= p <= 1")                        # (-1.2, 1.2)
        self.refused(lambda: u_tri(-1), "need a width >= 0")                                      # -0.408
        self.refused(lambda: stats.poisson_at_most(0, -1), "need a mean >= 0")                    # 1.0
        self.refused(lambda: welch_satterthwaite([-1, 1], [1, 1]), "u must be >= 0")              # 4
        self.refused(lambda: drift.psi([-1, 2], [-1, 2]), "need a whole n >= 0")                  # 0.0
        self.refused(lambda: Result(42, 0, dof=-1).interval(2), "dof must be > 0")                # (42, 42)
        self.refused(lambda: Result(42, 0).interval(2), "need 0 <= p < 1")                        # (42, 42)
        self.refused(lambda: judge.at_most(0, 1, 1.2), "need 0 <= p <= 1")                        # 0.0

    def test_their_neighbors(self):
        for value in (math.nan, math.inf, -math.inf):
            self.refused(lambda: Result(value, 1), "value is no number")
        for dof in (0, -1, math.nan):
            self.refused(lambda: Result(1, 1, dof=dof), "dof must be > 0")
        for p in (-0.1, 1, 1.5, math.nan):
            self.refused(lambda: Result(1, 0).interval(p), "need 0 <= p < 1")
            self.refused(lambda: Result(1, 1).k(p), "need 0 <= p < 1")
        for p in (-0.1, 1.0000001, math.nan):
            self.refused(lambda: rect_interval(0, 1, p), "need 0 <= p <= 1")
            self.refused(lambda: judge.at_most(0, 1, p), "need 0 <= p <= 1")
            self.refused(lambda: judge.tails(0, 1, p), "need 0 <= p <= 1")
        for width in (-1, -1e-300, math.nan, math.inf):
            for f in (u_rect, u_tri, u_resolution, lambda w: rect_interval(0, w)):
                self.refused(lambda: f(width), "need a width >= 0")
        for mean in (-1, -1e-300, math.nan, math.inf):
            self.refused(lambda: stats.poisson_at_most(3, mean), "need a mean >= 0")
            self.refused(lambda: stats.poisson_tails(-1, mean), "need a mean >= 0")
        for us in ([-1, 1], [1, math.nan], [math.inf, 1]):
            self.refused(lambda: welch_satterthwaite(us, [1, 1]), "u must be >= 0")
        for ref, cur in (([0.5, 2], [1.5, 1]), ([1, 2], [1, -2]), ([1, math.nan], [1, 2]), ([1, 2], [1, 2.5])):
            self.refused(lambda: drift.psi(ref, cur), "need a whole n >= 0")                      # 0.344 for the first

    def test_the_degenerate_cases_stay(self):
        self.assertEqual(Result(42, 0).interval(0.95), (42, 42))
        self.assertEqual(Result(42, 0).prob_below(42), 1.0)
        self.assertEqual(Result(42, 0, dof=math.inf).k(0), 0.0)
        self.assertEqual((rect_interval(5, 0), rect_interval(5, 2, 0), rect_interval(0, 1, 1)), ((5, 5), (5, 5), (-1, 1)))
        self.assertEqual((u_rect(0), u_tri(0), u_resolution(0)), (0, 0, 0))
        self.assertAlmostEqual(u_tri(1), 1 / math.sqrt(6), places=15)
        self.assertEqual(stats.poisson_at_most(3, 0), 1.0)
        self.assertEqual(welch_satterthwaite([0, 0], [1, 1]), math.inf)
        self.assertAlmostEqual(welch_satterthwaite([1, 1], [1, 1]), 2, places=12)
        self.assertEqual(drift.psi([0, 5], [0, 5]), 0.0)
        self.assertEqual(drift.psi([1.0, 2.0], [1, 2]), 0.0)
        self.assertGreater(drift.psi([10, 0], [0, 10]), 1)
        self.assertEqual((judge.at_most(0, 1, 0), judge.at_most(0, 1, 1), judge.at_most(1, 1, 1)), (1.0, 0.0, 1.0))


class NamesAndTokens(unittest.TestCase):
    """11: what is not JSON is not read, a name is text, and a stale stored outcome is reported."""

    def test_the_bare_words_are_refused(self):
        self.assertIn('"clock": "one host, monotonic"', STORED)
        for token in ("NaN", "Infinity", "-Infinity"):
            text = STORED.replace('"one host, monotonic"', token)
            self.assertTrue(isinstance(json.loads(text)["capture"]["clock"], float))             # the default reader takes it
            with self.assertRaisesRegex(ValueError, f"{token} is not JSON"):
                R.read(text)                                                      # Version 52: read, valid, accepted
            with self.assertRaisesRegex(ValueError, "is not JSON"):
                R.read(STORED.replace('"limit": 200.0', f'"limit": {token}'))
        self.assertEqual(R.decide(R.read(STORED)), ("accept", []))

    def test_a_name_is_text(self):
        rec = R.read(STORED)
        fields = [("measurand", key) for key in ("quantity", "convention", "target", "conditions", "start_event",
                                                 "end_event", "unit")] + [("data", "run_weighting"), ("capture", "clock")]
        for section, key in fields:
            for value in (math.nan, 7, 0, True, ["one host"], {"name": "x"}, Decimal("1.5")):
                other = copy.deepcopy(rec)
                other[section][key] = value
                self.assertEqual(R.invalid(other), [f"{section}.{key} is not text"], (key, value))
                self.assertEqual(R.decide(other), ("invalid", [f"{section}.{key} is not text"]))
            # Unknown is not invalid: it is asked for, where the decision needs it.
            for value in (None, "", "<which clock>"):
                other = copy.deepcopy(rec)
                other[section][key] = value
                self.assertEqual(R.invalid(other), [])
                self.assertEqual(R.decide(other)[0] == "unresolved", f"{section}.{key}" in R.missing(other))
        self.assertEqual(R.invalid(rec), [])

    def test_a_stale_outcome_is_reported_beside_the_decision(self):
        rec = R.read(STORED)
        self.assertEqual((rec["decision"]["outcome"], R.decide(rec)[0], R.stale(rec)), ("accept", "accept", None))
        edited = copy.deepcopy(rec)
        edited["decision"]["outcome"] = "reject"
        self.assertEqual((R.invalid(edited), R.decide(edited), R.stale(edited)), ([], ("accept", []), "reject"))
        lossy = copy.deepcopy(rec)
        lossy["capture"]["records_lost"] = None
        self.assertEqual((R.decide(lossy)[0], R.stale(lossy)), ("unresolved", "accept"))
        for unknown in (None, "", "<outcome>"):
            blank = copy.deepcopy(rec)
            blank["decision"]["outcome"] = unknown
            self.assertIsNone(R.stale(blank))
        self.assertIsNone(R.stale(check.load("record_template.json")))
        self.assertIsNone(R.stale(check.OTHER))

    def test_the_transcript_names_each_stale_outcome(self):
        text = expected("ch22_record_check")
        self.assertEqual(printed(check.main), text)
        self.assertIn("outcome edited\n  decision inputs missing: 0\n  status: accept\n  stored outcome, stale: reject\n", text)
        blocks = text.split("\n")
        stale = [i for i, line in enumerate(blocks) if line.startswith("  stored outcome, stale:")]
        self.assertEqual(len(stale), 8)
        self.assertEqual(len(blocks) - 1, 67)
        # Everywhere but in the new block the stored word is the one that the generator wrote.
        self.assertEqual(sorted(blocks[i] for i in stale), ["  stored outcome, stale: accept"] * 7
                         + ["  stored outcome, stale: reject"])
        self.assertTrue(text.startswith("stored record\n  decision inputs missing: 0\n  status: accept\nloss count unknown\n"))


def tie(value, u="1", dof=1, limit="200", risk="0.25"):
    """A record of one component, with these numbers as they are written."""
    rec = R.read(STORED)
    rec["components"] = [{"name": "one", "type": "A", "kind": "stipulated", "u": Decimal(u), "dof": dof, "source": "set"}]
    rec["result"].update(value=Decimal(value), u=Decimal(u), dof=dof)
    rec["decision"].update(limit=Decimal(limit), risk=Decimal(risk), outcome=None)
    return rec


class RejectionBoundary(unittest.TestCase):
    """12: one rule at the boundary. A guarded value at the limit is accepted from below and not rejected from above."""

    def test_the_tie_of_the_review_in_the_record(self):
        # Student t, one degree of freedom, risk 1/4: the quantile at 3/4 is tan(pi/4) = 1.
        self.assertAlmostEqual(math.tan(math.pi / 4), 1.0, places=15)
        self.assertEqual(R.decide(tie("201")), ("undecided", []))
        self.assertEqual(R.decide(tie("201.000001")), ("reject", []))
        self.assertEqual(R.decide(tie("200.999999")), ("undecided", []))
        self.assertEqual(R.decide(tie("199")), ("accept", []))
        self.assertEqual(R.decide(tie("199.000001")), ("undecided", []))
        # With no uncertainty the rule is the objective itself: at most the limit.
        self.assertEqual(R.decide(tie("200", u="0")), ("accept", []))
        self.assertEqual(R.decide(tie("200.000001", u="0")), ("reject", []))

    def test_the_rule_of_chapter_fourteen_at_the_same_tie(self):
        # A measured 203 ms with u = 3 ms against 200 ms: the probability of conformity is phi(-1). With exactly that
        # risk the guarded value is at the limit. Version 52 rejected there.
        at = NormalDist().cdf(-1.0)
        with mock.patch.object(c14, "RISK", at):
            out = printed(c14.conformity).splitlines()
        rows = {int(line.split()[0]): line.split()[-1] for line in out[2:]}
        self.assertEqual((c14.LIMIT - 203) / c14.U, -1.0)
        self.assertEqual(rows[203], "undecided")
        self.assertEqual(rows[206], "reject")
        self.assertEqual(rows[188], "accept")

    def test_the_transcript_of_chapter_fourteen_is_unchanged(self):
        self.assertEqual(printed(c14.main), expected("ch14_decisions"))
        self.assertIn("accept if value + k u <= limit", R.EXPECTS[1][2])
        self.assertIn("reject if value - k u > limit", " ".join(R.decide.__doc__.split()))


class BoundaryTable(unittest.TestCase):
    """The three rows of the table that Chapter 24 gains."""

    def test_an_exact_count_at_its_limit(self):
        self.assertEqual(math.ceil(Fraction(95, 100) * 1000), 950)
        self.assertEqual(bounds.quantile_verdict(1000, 0.95, 950, 0), "met")
        self.assertEqual(bounds.quantile_verdict(1000, 0.95, 949, 0), "missed")
        self.assertEqual(bounds.quantile_verdict(1000, 0.95, 949, 1), "unresolved")

    def test_zero_events_at_positive_exposure(self):
        self.assertEqual(f"{stats.quiet_bound(30):.3f}", "0.100")
        self.assertEqual(f"{-math.log(0.05) / 30:.3f}", "0.100")

    def test_an_upper_bound_equal_to_what_the_objective_needs(self):
        # Of the 900 slow requests 531 must be within 200 ms: 891 + 0.1 * 531 / 900 of a thousand parts is 95 %.
        fast = Fraction(891, 1000)
        self.assertEqual(fast + Fraction(1, 10) * Fraction(531, 900), Fraction(95, 100))
        self.assertEqual(mix.with_one(720, 450), 531)
        self.assertEqual(mix.verdict(fast, fast + Fraction(1, 10) * Fraction(531, 900)), "unresolved")
        self.assertEqual(mix.verdict(fast, fast + Fraction(1, 10) * Fraction(530, 900)), "missed")


class Release(unittest.TestCase):
    def test_the_figures_are_those_of_version_52(self):
        source = (ROOT / "figures" / "make_figures.py").read_text(encoding="utf-8")
        self.assertEqual(source.count("\n    save(fig, "), 32)

    def test_the_record_names_this_release(self):
        rec = json.loads(STORED)
        # The version of the release moves on; test_release.py holds the current one.
        self.assertRegex(rec["sources"]["release"], r"^book Version \d+, November 2026$")
        self.assertGreaterEqual(int(rec["sources"]["release"].split()[2].rstrip(",")), 53)
        self.assertEqual(rec["decision"]["outcome"], "accept")


if __name__ == "__main__":
    unittest.main()
