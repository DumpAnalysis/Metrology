"""Regression tests added after the review of Version 53.

The review found that the bound from the verified time of the five lines of Chapter 16 was stated without the
condition that the verified time be selected independently of the arrivals, and gave a bound that does without
it; that bracket() of Listing 22.1 counted a missing duration as one beyond every bound; that completeness() of
Listing 16.1 took a counter that runs back and readings that are not whole, and gave a negative time where the
stamps go back; that comparable() found two records with the same invalid name comparable; that two sketches
with one rounded factor and two logarithms were merged, and that a value next to the largest float could not
be reported; that paired_upper() and kept_starts() of Listing 16.4 subtracted a probability from one; and that
ulp32(), prob_below(), Result.k() and wilson() each had an endpoint that their words did not cover.

Each test computes what it expects independently of the code under test: bounds from the equations that define
them, solved here by halving; probabilities from sums of terms formed with lgamma(); counts from sets of
numbers; spacings from the bits of a binary32 value; and one simulation of a recorder that loses only the
messages of the component in question.
"""
import contextlib
import copy
import io
import math
import random
import struct
import sys
import unittest
from fractions import Fraction
from pathlib import Path

from examples import ch09_histogram as h9
from examples import ch14_decisions as c14
from examples import ch16_bounds as bounds
from examples import ch16_silence as silence
from examples import ch16_trace_measurements as trace
from examples import ch19_float as f19
from examples import ch22_budget as budget
from examples import ch22_record_check as check
from examples import ch23_mix as mix
from swmetro import Result, dist, stats
from swmetro import record as R

ROOT = Path(__file__).resolve().parent.parent
STORED = (ROOT / "data" / "measurement_record.json").read_text(encoding="utf-8")
LEAST = math.ldexp(1.0, -1074)
NEXT_TO_ONE = math.nextafter(1.0, 0.0)


def expected(name):
    return (ROOT / "expected" / f"{name}.txt").read_text(encoding="utf-8")


def printed(function, *args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        function(*args)
    return out.getvalue()


def poisson_term(j, mean):
    if mean == 0:
        return float(j == 0)
    return math.exp(j * math.log(mean) - mean - math.lgamma(j + 1))


def poisson_low(n, mean):
    return math.fsum(poisson_term(j, mean) for j in range(n + 1))


def poisson_high(n, mean):
    terms, j = [], n + 1
    while True:
        terms.append(poisson_term(j, mean))
        if j > mean and terms[-1] <= 1e-30 * max(terms):
            return math.fsum(terms)
        j += 1


def root(f, lo, hi):
    """The x between lo and hi at which a falling f passes zero, by halving."""
    for _ in range(200):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


class FiveLines(unittest.TestCase):
    """The bound from verified time, its condition, and the bound that does without the condition."""
    LINES = ("000041 2204 09:15:00.100 svc  request 9 start",
             "000042 3120 09:15:00.250 db   query 3310 done",
             "000044 2204 09:15:00.400 svc  request 9 start",
             "000045 2204 09:15:00.700 svc  request 9 end",
             "000046 3120 09:15:00.900 db   query 3311 done")
    COUNTER = (41, 48)

    def setUp(self):
        self.rows = [trace.parse(x) for x in self.LINES]
        self.span = (trace.stamp("09:15:00.000"), trace.stamp("09:15:01.000"))

    def test_the_table_of_the_review(self):
        self.assertEqual(trace.completeness(self.rows), (1, 6, 150))
        self.assertEqual(trace.completeness(self.rows, self.COUNTER), (2, 7, 150))
        self.assertEqual(trace.verified(self.rows, 0, 1, self.span), 0.65)
        self.assertEqual(trace.verified(self.rows, 0, 1, self.span, self.COUNTER), 0.75)
        self.assertEqual((f"{-math.log(0.05) / 0.65:.5f}", f"{-math.log(0.05) / 0.75:.5f}"), ("4.60882", "3.99431"))
        self.assertEqual(f"{stats.quiet_bound(0.65):.5f}", "4.60882")
        self.assertEqual(f"{stats.quiet_bound(0.75):.5f}", "3.99431")

    def test_the_bound_that_counts_the_lost_as_its_own(self):
        # Seven numbers were assigned in the second and five lines are there: at most two messages of a
        # component that wrote none of the lines. Two or fewer in one second have probability 5 % at the rate u
        # with exp(-u) (1 + u + u*u/2) = 0.05.
        lost, sent, _ = trace.completeness(self.rows, self.COUNTER)
        self.assertEqual((lost, sent - len(self.rows)), (2, 2))
        u = root(lambda x: math.exp(-x) * (1 + x + x * x / 2) - 0.05, 0.0, 50.0)
        self.assertEqual(f"{u:.5f}", "6.29579")
        self.assertAlmostEqual(stats.rate_upper(lost, 1.0), u, places=10)
        self.assertEqual(f"{stats.rate_upper(2, 1.0):.2f}", "6.30")
        # The bound grows with the count, so the bound for two covers one and none as well.
        tops = [stats.rate_upper(n, 1.0) for n in range(4)]
        self.assertEqual(tops, sorted(tops))
        self.assertEqual([f"{x:.5f}" for x in tops[:3]], ["2.99573", "4.74386", "6.29579"])
        one = root(lambda x: math.exp(-x) * (1 + x) - 0.05, 0.0, 50.0)
        self.assertAlmostEqual(tops[1], one, places=10)

    def test_twenty_heartbeats_in_closed_form(self):
        # An interval is verified exactly if it holds none of the messages: probability exp(-1) at one a second.
        p = math.exp(-1)
        few = sum(math.comb(20, k) * p ** k * (1 - p) ** (20 - k) for k in range(3))
        self.assertEqual(f"{100 * (1 - few):.2f}", "99.20")
        self.assertEqual(f"{p:.3f}", "0.368")
        # With three verified seconds the bound is below the true rate, and with two it is above.
        self.assertEqual((f"{-math.log(0.05):.5f}", f"{-math.log(0.05) / 3:.3f}"), ("2.99573", "0.999"))
        self.assertLess(stats.quiet_bound(3), 1.0)
        self.assertGreater(stats.quiet_bound(2), 1.0)

    def test_twenty_heartbeats_through_the_listing(self):
        # A recorder keeps a heartbeat every second, numbers every message and loses those of one component,
        # which arrive at one a second. The functions of Listing 16.1 verify exactly the seconds in which
        # that component wrote nothing, it has no line in them, and the bound from them is below its rate
        # in nearly every record.
        rng = random.Random(5301)
        span, runs, below = (0, 20_000), 3000, 0
        for _ in range(runs):
            seq, rows, quiet = 0, [], 0
            for second in range(21):
                seq += 1
                rows.append((seq, 1000 * second, "beat", "alive"))
                if second < 20:
                    lost, t = 0, rng.expovariate(1.0)
                    while t < 1:
                        lost, t = lost + 1, t + rng.expovariate(1.0)
                    seq += lost
                    quiet += lost == 0
            seconds = trace.verified(rows, 0, 20, span)
            self.assertEqual(seconds, quiet)
            self.assertEqual(trace.count(rows, "svc", 0, 20, span), 0)
            if seconds and stats.quiet_bound(seconds) < 1.0:
                below += 1
                self.assertGreaterEqual(quiet, 3)
        # 99.20 % of the records; three standard deviations of the simulated share are 0.5 points.
        self.assertAlmostEqual(below / runs, 0.9920, delta=0.006)

    def test_a_recorder_that_loses_other_messages(self):
        # The same recorder, losing three in ten of the messages of another component and none of the one in
        # question: its lines in verified time, set against that time, give a bound that holds in at least
        # 95 % of the records (in 99.9 % of these: a line of the component can end a verified stretch, and it
        # never shortens one). None of its messages is lost, so none of them makes a stretch unverified.
        rng = random.Random(5306)
        span, runs, holds = (0, 20_000), 3000, 0
        for _ in range(runs):
            events = [(1000 * second, "beat") for second in range(21)]
            for comp, rate in (("svc", 1.0), ("db", 3.0)):
                t = rng.expovariate(rate)
                while t < 20:
                    events.append((1000 * t, comp))
                    t += rng.expovariate(rate)
            events.sort()
            rows = [(seq, ms, comp, "m") for seq, (ms, comp) in enumerate(events, 1)
                    if comp != "db" or rng.random() >= 0.3]
            by_number = {r[0]: r[2] for r in rows}
            parts = trace.runs(rows, 0, 20, span)
            seconds = sum(y - x for x, y, _ in parts) / 1000
            own = sum(by_number[n] == "svc" for _, _, numbers in parts for n in numbers)
            self.assertEqual(sum(r[2] == "svc" for r in rows), sum(c == "svc" for _, c in events))
            if seconds == 0 or stats.rate_upper(own, seconds) >= 1.0:
                holds += 1
        self.assertGreaterEqual(holds / runs, 0.95)


class MissingDuration(unittest.TestCase):
    """1: a duration that is missing is in no bucket."""

    @staticmethod
    def p95(xs):
        return sorted(xs)[math.ceil(Fraction(95, 100) * len(xs)) - 1]

    def test_the_three_rows_of_the_review(self):
        known = [100.0] * 18 + [250.0]
        for missing, rank19, want in ((100.0, 100.0, "met"), (250.0, 250.0, "missed")):
            xs = known + [missing]
            self.assertEqual(self.p95(xs), rank19)
            lo, hi = budget.bracket(xs, budget.FITTED)
            self.assertTrue(lo < rank19 <= hi)
            self.assertEqual(budget.verdict(lo, hi), want)
        with self.assertRaisesRegex(ValueError, "a duration is missing"):
            budget.bracket(known + [math.nan], budget.FITTED)                    # Version 53: (200, 250], "missed"
        # Eighteen of twenty are within 200 ms, one is beyond, and one could be either: rank 19 decides.
        self.assertEqual(bounds.quantile_verdict(20, 0.95, 18, 1), "unresolved")
        self.assertEqual(bounds.quantile_verdict(20, 0.95, 19, 0), "met")
        self.assertEqual(bounds.quantile_verdict(20, 0.95, 18, 0), "missed")

    def test_no_missing_duration_is_counted(self):
        for xs in ([math.nan] * 20, [math.nan], [100.0, math.nan, 100.0], [float("nan"), -0.9]):
            for edges in (budget.FITTED, budget.COARSE, [50, 100]):
                with self.assertRaisesRegex(ValueError, "a duration is missing"):
                    budget.bracket(xs, edges)                                     # (500, inf) for twenty of them
        # Until Version 56 an infinite duration was counted beyond every bound, (500, inf) here. The
        # entry of the listing is one for the percentile, the budget and the buckets since Version 55, and
        # it takes finite numbers only since Version 56 (review of Version 55): a budget cannot take an
        # infinity, and a request that never completed belongs to the counts of Chapter 16.
        with self.assertRaisesRegex(ValueError, "a duration is missing"):
            budget.bracket([100.0] * 18 + [math.inf] * 2, budget.FITTED)
        self.assertEqual(budget.bracket([100.0] * 18 + [600.0] * 2, budget.FITTED), (500.0, math.inf))

    def test_the_bounds_rise(self):
        for edges in ([200, 50, 100], [50, 50, 100], [50, 100, 100], [math.nan, 100], [50, math.nan], [-math.inf, 50],
                      [100, 50]):
            with self.assertRaisesRegex(ValueError, "bounds do not rise"):
                budget.bracket([100.0] * 20, edges)                               # (-inf, 200.0) for the first
        self.assertEqual(budget.bracket([100.0] * 20, [50, 100, math.inf]), (50.0, 100.0))
        self.assertEqual(budget.bracket([100.0] * 20, []), (-math.inf, math.inf))

    def test_zero_and_negative_indications_stay(self):
        for xs in ([0.0] * 20, [-0.9] * 20, [-3.0] * 19 + [49.0]):
            self.assertEqual(budget.bracket(xs, [50, 100]), (-math.inf, 50.0))
        rng = random.Random(5302)
        for _ in range(300):
            xs = [round(rng.uniform(-20, 260), 1) for _ in range(rng.randint(1, 40))]
            cuts = sorted(rng.sample(range(-10, 250), 4))
            lo, hi = budget.bracket(xs, cuts)
            self.assertTrue(lo < self.p95(xs) <= hi, (xs, cuts))

    def test_the_transcript_is_unchanged(self):
        self.assertEqual(printed(budget.main), expected("ch22_budget"))


def rows_of(marks):
    return [(n, ms, "db", "query done") for n, ms in marks]


class CounterAndClock(unittest.TestCase):
    """2: what the counter establishes and what the clock establishes are kept apart."""

    def test_the_three_cases_of_the_review(self):
        with self.assertRaisesRegex(ValueError, "counter runs back"):
            trace.completeness([], (48, 41))                                      # Version 53: (-7, -7, None)
        self.assertEqual(trace.completeness([], (41, 48)), (7, 7, None))
        lost, sent, gap = trace.completeness(rows_of(((41, 250), (43, 100))))
        self.assertEqual((lost, sent), (1, 3))
        self.assertTrue(math.isnan(gap))                                          # -150
        with self.assertRaisesRegex(ValueError, "need a whole reading"):
            trace.completeness(rows_of(((41, 0), (42, 10))), (41, 44.5))          # (1.5, 3.5, None)

    def test_readings_and_numbers_are_whole(self):
        marks = ((41, 0), (42, 10))
        for counter in ((41.5, 44), (41, math.inf), (math.nan, 44), ("41", "x"), (None, 44)):
            with self.assertRaisesRegex(ValueError, "need a whole reading"):
                trace.completeness(rows_of(marks), counter)
        for bad in (42.5, math.nan, math.inf, "x"):
            with self.assertRaisesRegex(ValueError, "need a whole number"):
                trace.completeness(rows_of(((41, 0), (bad, 10))))
        # A whole number may be written as a float, and a label may be below zero.
        self.assertEqual(trace.completeness(rows_of(marks), (41.0, 44.0)), (1, 3, None))
        self.assertEqual(trace.completeness(rows_of(((-3, 0), (-2, 5), (0, 30)))), (1, 4, 25))
        self.assertEqual(trace.completeness(rows_of(((-3, 0), (-2, 5), (0, 30))), (-5, 2)), (4, 7, 25))

    def test_the_counter_does_not_run_back(self):
        for counter in ((48, 41), (1, 0), (0, -1)):
            with self.assertRaisesRegex(ValueError, "counter runs back"):
                trace.completeness([], counter)
        # A counter that did not move assigned nothing, and a line is then beyond it.
        self.assertEqual(trace.completeness([], (41, 41)), (0, 0, None))
        with self.assertRaisesRegex(ValueError, "counter runs back|beyond the counter"):
            trace.completeness(rows_of(((41, 0),)), (41, 41))
        with self.assertRaisesRegex(ValueError, "counter runs back|beyond the counter"):
            trace.completeness(rows_of(((41, 0), (42, 5))), (44, 41))

    def test_stamps_that_go_back_leave_the_counts(self):
        # The counts come from the numbers. The time of the gap comes from the stamps, and where any stamp lies
        # before the one before it, the stamps are no evidence, as for runs().
        cases = ((((41, 250), (43, 100)), (1, 3)),
                 (((41, 0), (42, 50), (43, 40), (46, 90)), (2, 6)),
                 (((41, 90), (42, 50), (45, 60)), (2, 5)))
        for marks, counts in cases:
            lost, sent, gap = trace.completeness(rows_of(marks))
            self.assertEqual((lost, sent), counts)
            self.assertTrue(math.isnan(gap), marks)
            numbers = [n for n, _ in marks]
            self.assertEqual(lost, len(set(range(numbers[0], numbers[-1] + 1)) - set(numbers)))
            span = (0, 1000)
            self.assertEqual(trace.runs(rows_of(marks), 0, 1, span), [])
        # Two lines with one stamp do not go back, and where no number is missing there is no gap to time.
        self.assertEqual(trace.completeness(rows_of(((41, 40), (43, 40)))), (1, 3, 0))
        self.assertEqual(trace.completeness(rows_of(((41, 90), (42, 50)))), (0, 2, None))

    def test_the_fixture_and_the_five_lines_are_as_they_were(self):
        lines = trace.LOG.read_text(encoding="utf-8").splitlines()
        rows = [trace.parse(x) for x in lines]
        self.assertEqual(trace.completeness(rows, trace.COUNTER), (40, 799, 2088))
        self.assertEqual(printed(trace.main), expected("ch16_trace_measurements"))


class Identity(unittest.TestCase):
    """3: two records with the same invalid name do not describe one quantity."""
    FIELDS = (("measurand", "quantity"), ("measurand", "convention"), ("measurand", "target"),
              ("measurand", "conditions"), ("measurand", "start_event"), ("measurand", "end_event"),
              ("data", "run_weighting"))

    def setUp(self):
        self.rec = R.read(STORED)

    def both(self, section, key, value):
        a, b = copy.deepcopy(self.rec), copy.deepcopy(self.rec)
        a[section][key] = b[section][key] = value
        return a, b

    def test_the_case_of_the_review(self):
        a, b = self.both("capture", "clock", 7)
        self.assertEqual(R.invalid(a), ["capture.clock is not text"])
        self.assertEqual(R.comparable(a, b), (["clock is not text"], []))        # Version 53: ([], [])
        for key in ("start_event", "quantity"):
            a, b = self.both("measurand", key, 7)
            self.assertEqual(R.comparable(a, b), ([f"{key} is not text"], []))
        a, b = self.both("data", "run_weighting", 7)
        self.assertEqual(R.comparable(a, b), (["run_weighting is not text"], []))

    def test_every_name_that_the_comparison_reads(self):
        for section, key in self.FIELDS + (("capture", "clock"),):
            name = "clock" if key == "clock" else key
            for value in (7, 0, True, 1.5, ["test load"], {"a": 1}, math.nan):
                a, b = self.both(section, key, value)
                self.assertEqual(R.comparable(a, b)[0], [f"{name} is not text"], (key, value))
                # One valid and one invalid name are no match either.
                self.assertEqual(R.comparable(a, self.rec)[0], [f"{name} is not text"], (key, value))
                self.assertEqual(R.comparable(self.rec, b)[0], [f"{name} is not text"], (key, value))
            # Unknown is unknown, as before.
            for value in (None, "", "<which>"):
                a, b = self.both(section, key, value)
                self.assertEqual(R.comparable(a, b)[0], [f"{name} unknown"], (key, value))
        a, b = self.both("measurand", "unit", 7)
        self.assertEqual(R.comparable(a, b)[0], ["unit"])

    def test_the_comparisons_of_the_listing_stay(self):
        self.assertEqual(R.comparable(self.rec, self.rec), ([], []))
        heavier = copy.deepcopy(self.rec)
        heavier["measurand"]["conditions"] = "twice the test load"
        self.assertEqual(R.comparable(self.rec, heavier), (["conditions"], []))
        self.assertEqual(R.comparable(self.rec, heavier, across=("conditions",)), ([], ["across conditions"]))
        # A record with only the fields that a comparison reads is still compared.
        self.assertEqual(R.comparable(self.rec, check.OTHER),
                         (["start_event", "convention", "clock rate relation"], ["1 s = 1000 ms"]))
        # The conditions under study must still be names.
        a, b = self.both("measurand", "conditions", 7)
        self.assertEqual(R.comparable(a, b, across=("conditions",)), (["conditions is not text"], []))
        self.assertEqual(printed(check.main), expected("ch22_record_check"))


class Sketches(unittest.TestCase):
    """4: one bucket map for a merge, and a value for every bucket."""

    def test_two_logarithms_for_one_rounded_factor(self):
        a, b = h9.LogSketch(8e-17), h9.LogSketch(1e-16)
        self.assertEqual(a.g, b.g)
        self.assertNotEqual(a.lg, b.lg)
        b.add(2.0)
        self.assertAlmostEqual(b.quantile(0.5), 2.0, places=12)
        with self.assertRaisesRegex(ValueError, "different accuracy"):
            a.merge(b)                                                            # Version 53: merged, and 1.7411 read back
        with self.assertRaisesRegex(ValueError, "different accuracy"):
            h9.LogSketch(0.01).merge(h9.LogSketch(0.02))

    def test_a_merge_is_the_sketch_of_all_the_values(self):
        rng = random.Random(5304)
        xs = [rng.lognormvariate(0, 2) for _ in range(300)]
        whole, left, right = h9.sketch_of(xs), h9.sketch_of(xs[:120]), h9.sketch_of(xs[120:])
        left.merge(right)
        self.assertEqual((left.n, left.counts), (whole.n, whole.counts))
        for q in (0, 0.5, 0.99, 1):
            self.assertEqual(left.quantile(q), whole.quantile(q))

    def test_values_next_to_the_largest_float(self):
        top = sys.float_info.max
        for alpha in (0.3, 0.01, 1e-6):
            for x in (1e308, 1.79e308, top, math.nextafter(top, 0), 8.99e307, 1e300):
                sk = h9.LogSketch(alpha)
                sk.add(x)
                got = sk.quantile(0.5)                                            # Version 53: inf, or OverflowError
                self.assertTrue(math.isfinite(got), (alpha, x))
                self.assertLessEqual(abs(got / x - 1), alpha * (1 + 1e-9), (alpha, x))

    def test_every_positive_float_has_its_value(self):
        rng = random.Random(5305)
        for alpha in (0.05, 1e-3, 1e-9):
            for _ in range(400):
                x = math.ldexp(rng.uniform(0.5, 1), rng.randint(-1000, 1024))
                sk = h9.LogSketch(alpha)
                sk.add(x)
                self.assertLessEqual(abs(sk.quantile(0) / x - 1), alpha * (1 + 1e-6), (alpha, x))
        # Until Version 55 the sketch took the floats below the normal ones, and this test held that 1, 3
        # and 40 of the least float came back as themselves. Sixty of them came back as fifty-nine, 1.67 %
        # off at an accuracy of 1 % (review of Version 54): down there the floats lie further apart than
        # the accuracy, and the sketch now refuses them.
        for x in (LEAST, 3 * LEAST, 40 * LEAST, 60 * LEAST, math.nextafter(2.2250738585072014e-308, 0)):
            with self.assertRaisesRegex(ValueError, "need 2..-1022 <= x < inf"):
                h9.LogSketch(0.01).add(x)
        sk = h9.LogSketch(0.01)
        sk.add(2.2250738585072014e-308)                  # the least normal float has neighbors within 1 %
        self.assertLessEqual(abs(sk.quantile(0.5) / 2.2250738585072014e-308 - 1), 0.01)
        for x in (0, -1.0):
            with self.assertRaisesRegex(ValueError, "need 2..-1022 <= x < inf"):
                h9.LogSketch(0.01).add(x)
        for x in (math.inf, math.nan, -math.inf):        # since Version 57 refused by known() of the package
            with self.assertRaisesRegex(ValueError, "a duration is missing or too large"):
                h9.LogSketch(0.01).add(x)

    def test_the_reported_value_is_the_one_of_the_text(self):
        # 2 g**i / (g + 1) is g**i (1 - alpha), since (g + 1) / 2 is 1 / (1 - alpha).
        for alpha in (0.3, 0.01, 1e-4):
            g = Fraction(1 + Fraction(str(alpha))) / (1 - Fraction(str(alpha)))
            self.assertEqual(2 / (g + 1), 1 - Fraction(str(alpha)))
            sk = h9.LogSketch(alpha)
            for x in (0.003, 1.0, 17.5, 1e6):
                sk.counts, sk.n = {}, 0
                sk.add(x)
                (i,) = sk.counts
                self.assertTrue(math.isclose(sk.quantile(0.5), float(2 * g ** i / (g + 1)), rel_tol=1e-12), (alpha, x))

    def test_the_transcript_is_unchanged(self):
        self.assertEqual(printed(h9.compare) + printed(h9.averaging), expected("ch09_histogram"))


def pair_tails(n, r, two, one):
    """P(at most n messages) and P(more than n) when requests arrive at rate r and bring two messages from
    start times of length `two` and one from those of length `one`: sums of terms formed with lgamma()."""
    low = math.fsum(poisson_term(j, r * two) * poisson_low(n - 2 * j, r * one) for j in range(n // 2 + 1))
    high = math.fsum(poisson_term(j, r * two) * (poisson_high(n - 2 * j, r * one) if one else 0.0)
                     for j in range(n // 2 + 1))
    return low, high + (poisson_high(n // 2, r * two) if two else 0.0)


class PairedModel(unittest.TestCase):
    """5: the paired bound from the tail that is next to zero, and the share kept without a subtraction from one."""

    def test_the_cases_of_the_review(self):
        second = [(0, 1)]
        for level in (0.95, 1e-18):
            self.assertEqual(silence.paired_upper(0, second, 0, level), 2 * stats.quiet_bound(1, level))
            self.assertEqual(silence.paired_upper(0, second, 0, level), silence.paired_bound(second, 0, level))
        self.assertEqual(f"{silence.paired_upper(0, second, 0, 0.95):.5f}", "5.99146")
        got = silence.paired_upper(0, second, 0, 1e-18)
        self.assertTrue(math.isclose(got, 2e-18, rel_tol=1e-12), got)            # Version 53: 6.2230e-61
        kept = silence.kept_starts(second, 0, 1e-18, "message")
        self.assertTrue(math.isclose(kept, 1e-18 * (2 - 1e-18), rel_tol=1e-15), kept)   # 0.0
        self.assertEqual(1 - (1 - 1e-18) ** 2, 0.0)

    def test_no_time_between_the_two_messages(self):
        # A window of 30 s and no delay: the two intervals of start times are one, and every request brings both
        # messages. The bound for messages is twice that for single ones, 0.200 against 0.100.
        window = [(0, 30)]
        self.assertEqual(silence.by_messages(window, 0), (30, 0))
        self.assertEqual(silence.reach(window, 0), 30)
        self.assertEqual(silence.paired_bound(window, 0), 2 * stats.quiet_bound(30))
        self.assertEqual((f"{silence.paired_bound(window, 0):.3f}", f"{stats.quiet_bound(30):.3f}"), ("0.200", "0.100"))
        self.assertEqual(f"{2 * -math.log(0.05) / 30:.3f}", "0.200")
        # The request of the fixture, 2.346 s long: 32.346 s of start times.
        self.assertAlmostEqual(silence.reach(window, 2.346), 32.346, places=12)
        # Until Version 70 reach() rounded the length to a float, 32.346. Since Version 71 it returns the fraction
        # that the length is (tests/test_review70.py), which that float is not, and plain() rounds it.
        self.assertEqual(silence.reach(window, Fraction(2346, 1000)), Fraction(16173, 500))
        self.assertEqual(silence.plain(silence.reach(window, Fraction(2346, 1000))), 32.346)

    def test_pairs_that_coincide_are_requests_counted_twice(self):
        # With no delay every request brings both messages: at most n messages are at most n // 2 requests.
        for n in range(1, 7):
            for level in (1e-18, 1e-6, 0.3, 0.5, 0.95, NEXT_TO_ONE):
                got = silence.paired_upper(n, [(0, 1)], 0, level)
                want = 2 * stats.rate_upper(n // 2, 1, level)
                self.assertTrue(math.isclose(got, want, rel_tol=1e-9), (n, level, got, want))

    def test_pairs_that_never_meet_are_single_messages(self):
        # With a delay beyond the window a request brings one message, from start times twice as long.
        self.assertEqual(silence.by_messages([(0, 1)], 5), (0, 2))
        for n in range(1, 6):
            for level in (1e-18, 1e-6, 0.3, 0.95):
                got = silence.paired_upper(n, [(0, 1)], 5, level)
                want = 2 * stats.rate_upper(n, 2, level)
                self.assertTrue(math.isclose(got, want, rel_tol=1e-9), (n, level, got, want))

    def test_the_bound_solves_its_equation(self):
        for spans, delta in (([(0, 30)], 10), ([(0, 30)], 2.346), ([(0, 10), (20, 30)], 4)):
            two, one = silence.by_messages(spans, delta)
            self.assertTrue(two > 0 and one > 0)
            for n in (1, 2, 5, 12):
                for level in (1e-18, 1e-6, 0.3, 0.5, 0.9, 0.95, 1 - 1e-9):
                    r = silence.paired_upper(n, spans, delta, level) / 2
                    low, high = pair_tails(n, r, two, one)
                    if level < 0.5:
                        self.assertTrue(math.isclose(high, level, rel_tol=1e-8), (n, level, high))
                    else:
                        self.assertTrue(math.isclose(low, 1 - level, rel_tol=1e-8), (n, level, low))

    def test_the_share_kept(self):
        whole = [(0, 30)]
        two, one = silence.by_messages(whole, 10)
        for keep in (1e-18, 1e-9, 0.1, 0.5, 0.9, 1):
            p = Fraction(keep)
            by_message = float(p * one + (1 - (1 - p) ** 2) * two)
            by_request = float(p * one + p * two)
            self.assertTrue(math.isclose(silence.kept_starts(whole, 10, keep, "message"), by_message, rel_tol=1e-15))
            self.assertTrue(math.isclose(silence.kept_starts(whole, 10, keep, "request"), by_request, rel_tol=1e-15))
        for keep in (0, -0.1, 1.1, math.nan):
            with self.assertRaisesRegex(ValueError, "need 0 < keep <= 1"):
                silence.kept_starts(whole, 10, keep, "message")

    def test_the_transcript_is_unchanged(self):
        self.assertEqual(printed(silence.main), expected("ch16_silence"))


class Endpoints(unittest.TestCase):
    """6: the words of four functions and what they do at an end of their range."""

    def test_the_largest_binary32_value(self):
        top = struct.unpack("<f", struct.pack("<I", 0x7F7FFFFF))[0]
        below = struct.unpack("<f", struct.pack("<I", 0x7F7FFFFE))[0]
        self.assertEqual((top, f19.MAX32), ((2 - 2.0 ** -23) * 2.0 ** 127, top))
        self.assertEqual(f19.ulp32(top), top - below)
        self.assertEqual(f19.ulp32(-top), top - below)
        for x in (math.nextafter(top, math.inf), -math.nextafter(top, math.inf), 3.5e38, math.nextafter(2.0 ** 128, 0),
                  2.0 ** 128, math.inf, math.nan):
            with self.assertRaisesRegex(ValueError, "no binary32 value"):
                f19.ulp32(x)                                                      # Version 53: 2**104 below 2**128
        self.assertEqual((f19.ulp32(0.0), f19.ulp32(1.0)), (2.0 ** -149, 2.0 ** -23))
        self.assertEqual(printed(f19.main), expected("ch19_float"))

    def test_a_limit_that_is_no_number(self):
        for u in (0, 1, 3.5):
            for dof in (2, math.inf):
                with self.assertRaisesRegex(ValueError, "limit is no number"):
                    Result(0, u, dof=dof).prob_below(math.nan)                    # Version 53: 0.0 for u = 0, NaN otherwise
                # An infinite limit is a number. With few degrees of freedom the tail beyond 1e153 standard
                # uncertainties is overstated by design, at 2.5e-307, and is never understated.
                self.assertEqual(Result(0, u, dof=dof).prob_below(math.inf), 1.0)
                self.assertLess(Result(0, u, dof=dof).prob_below(-math.inf), 1e-300)
        self.assertEqual(Result(0, 0).prob_below(-math.inf), 0.0)
        self.assertEqual(Result(0, 1).prob_below(-math.inf), 0.0)
        self.assertEqual((Result(0, 0).prob_below(0), Result(0, 0).prob_below(-1e-300)), (1.0, 0.0))
        self.assertAlmostEqual(Result(0, 1).prob_below(0), 0.5, places=15)

    def test_the_least_probability(self):
        # Next to zero the factor is proportional to p: p pi / 2 for one degree of freedom, p sqrt(2) for two, and
        # p sqrt(2 pi) / 2 for the normal distribution. In units of the least float, the nearest whole number.
        consts = ((1, math.pi / 2), (2, math.sqrt(2)), (math.inf, math.sqrt(2 * math.pi) / 2))
        for units in (1, 2, 3, 7, 1000):
            p = units * LEAST
            for dof, c in consts:
                got = Result(0, 1, dof=dof).k(p)                                  # Version 53: ValueError for one unit
                self.assertEqual(got, round(units * c) * LEAST, (units, dof))
        self.assertEqual(LEAST / 2, 0.0)
        self.assertEqual(dist.t_within(LEAST, 5) > 0, True)

    def test_the_factor_is_continuous_where_the_route_changes(self):
        for dof in (1, 2, 7.5, 1e7, math.inf):
            for p in (1e-300, 1e-250, 1e-18, 0.1, 0.3, 0.49):
                self.assertEqual(dist.t_within(p, dof), dist.t_mid(p / 2, dof))
            lower, upper = math.nextafter(1e-300, 0), 1e-300
            a, b = dist.t_within(lower, dof), dist.t_within(upper, dof)
            self.assertTrue(math.isclose(a, b, rel_tol=1e-12), (dof, a, b))
            for p in (1e-305, 1e-310, 4e-320):
                c = dist.t_within(1e-290, dof) / 1e-290
                self.assertTrue(math.isclose(dist.t_within(p, dof), c * p, rel_tol=1e-3), (dof, p))

    def test_wilson_at_the_least_probability(self):
        self.assertEqual(stats.wilson(0, 10, LEAST), (0.0, 0.0))                  # Version 53: ValueError
        self.assertEqual(stats.wilson(10, 10, LEAST), (1.0, 1.0))
        low, high = stats.wilson(5, 10, LEAST)
        self.assertEqual((low, high), (0.5, 0.5))
        low, high = stats.wilson(5, 10, 1e-300)
        self.assertTrue(low <= 0.5 <= high)
        # At an ordinary level nothing moved.
        low, high = stats.wilson(8, 10)
        self.assertEqual((f"{low:.4f}", f"{high:.4f}"), ("0.4902", "0.9433"))


class BoundaryRows(unittest.TestCase):
    """7 and 8: the rule for a range of attainable counts, the two bucket widths, and the tie of Chapter 14."""

    @staticmethod
    def share(k):
        return Fraction(891, 1000) + Fraction(1, 10) * Fraction(k, 900)

    def by_rule(self, low, high, need=531):
        if low >= need:
            return "met"
        if high < need:
            return "missed"
        return "unresolved"

    def test_both_ends_of_the_range(self):
        self.assertEqual(self.share(531), Fraction(95, 100))
        for low, high in ((531, 531), (0, 531), (530, 531), (0, 530), (531, 900), (532, 600), (0, 0), (467, 467)):
            self.assertEqual(mix.verdict(self.share(low), self.share(high)), self.by_rule(low, high), (low, high))
        # The upper end alone does not decide: 531 attainable is "met" or "unresolved" by the lower end.
        self.assertEqual({mix.verdict(self.share(low), self.share(531)) for low in (0, 530, 531)}, {"met", "unresolved"})
        # In Figure 23.4 none can be within the limit, so a count below 531 can be attained.
        self.assertEqual(mix.with_one(720, 450), 531)

    def test_the_two_bucket_widths_of_chapter_nine(self):
        rows = {line.split()[0]: line.split() for line in expected("ch09_histogram").splitlines()[1:5]}
        widths = {}
        for q in ("0.99", "0.999"):
            exact, bucketed = float(rows[q][2]), float(rows[q][3])
            hi = min(b for b in h9.BOUNDS if exact <= b)
            lo = max(b for b in h9.BOUNDS if b < exact)
            self.assertTrue(lo < bucketed <= hi)
            widths[q] = (hi - lo, round(100 * (bucketed - exact) / exact))
        self.assertEqual(widths, {"0.99": (0.25, 15), "0.999": (0.5, 27)})

    def test_exactly_at_the_risk_is_undecided(self):
        from statistics import NormalDist
        from unittest import mock
        at = NormalDist().cdf(-1.0)
        with mock.patch.object(c14, "RISK", at):
            rows = {int(x.split()[0]): x.split()[-1] for x in printed(c14.conformity).splitlines()[2:]}
        self.assertEqual((rows[203], rows[206], rows[188]), ("undecided", "reject", "accept"))


class Release(unittest.TestCase):
    def test_the_figures_are_those_of_version_53(self):
        source = (ROOT / "figures" / "make_figures.py").read_text(encoding="utf-8")
        self.assertEqual(source.count("\n    save(fig, "), 32)


if __name__ == "__main__":
    unittest.main()
