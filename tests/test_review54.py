"""Regression tests added after the review of Version 54.

The review found that p95() and from_raw() of Listing 22.1 sorted and resampled a record with a missing
duration, so that the percentile followed the order of the rows; that the five lines of Chapter 16 have three
bounds and not two, each with a condition of its own, and that the third comes from a function that Listing
16.1 does not hold; that time_delta() of Listing 16.1 pairs two events by their text alone; that window()
multiplied float seconds and could count a line on the wrong side of an end; that runs() and verified() took
numbers and readings that completeness() refuses; that a rounded record could state an uncertainty with more
digits than it declares, and that bounds() did not mirror its interval below zero; that a clock-rate relation
could name a clock by a number; that read() took a text that is JSON and no object; that bracket() turned the
bounds of its buckets into floats, and that bucket_quantile() of Listing 9.1 did not examine its bounds; and
that the sketch of Listing 9.1 stated for all positive floats a guarantee that holds for real numbers.

Each test computes what it expects independently of the code under test: percentiles by sorting complete
records, bounds from the equations that define them, solved here by halving, windows from whole milliseconds,
the contract of a counter from a predicate written here, the digits of a number from its decimal text, the
values that round to a number by rounding every point of a fine grid, and the error of the sketch in exact
fractions of the floats that go in and come out.
"""
import bisect
import contextlib
import copy
import io
import math
import random
import re
import sys
import unittest
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Context, Decimal
from fractions import Fraction
from pathlib import Path

from examples import ch09_histogram as h9
from examples import ch16_bounds as bounds16
from examples import ch16_trace_measurements as trace
from examples import ch22_budget as budget
from examples import ch22_record_check as check
from swmetro import record as R
from swmetro import stats

ROOT = Path(__file__).resolve().parent.parent
NAN = math.nan


def halve(f, lo, hi):
    """The zero of a falling function between lo and hi."""
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return (lo + hi) / 2


def upper(n, seconds, level=0.05):
    """Rate at which n or fewer events in the time have the probability `level`."""
    def excess(mean):
        return sum(math.exp(-mean) * mean ** k / math.factorial(k) for k in range(n + 1)) - level
    return halve(excess, 0.0, 100.0) / seconds


def printed(function, *args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        function(*args)
    return out.getvalue()


def stored():
    return R.read((ROOT / "data" / "measurement_record.json").read_text(encoding="utf-8"))


class MissingDurationOnTheRawRoute(unittest.TestCase):
    KNOWN = [100.0] * 18 + [250.0]

    def orders(self):
        return ([NAN] + self.KNOWN, [100.0] * 18 + [NAN, 250.0], self.KNOWN + [NAN])

    def test_what_a_sort_makes_of_the_three_orders(self):
        # A NaN has no rank, and a sort puts it where the order of the input leads it. Rank 19 of the
        # sorted list, for the NaN first, between the eighteen and the one, and last: the three answers of
        # Version 54. (Until Version 56 this comment said that a sort leaves a NaN where it met it.)
        got = [sorted(xs)[18] for xs in self.orders()]
        self.assertEqual((got[0], got[2]), (100.0, 250.0))
        self.assertTrue(math.isnan(got[1]))

    def test_the_percentile_of_the_twenty_depends_on_the_unknown(self):
        # Rank 19 of 20. If the unknown duration was short the percentile is 100 ms, if long it is 250 ms.
        self.assertEqual(sorted(self.KNOWN + [1.0])[18], 100.0)
        self.assertEqual(sorted(self.KNOWN + [9999.0])[18], 250.0)
        # Eighteen within the objective, one unknown: the count route of Listing 16.3 says so.
        self.assertEqual(bounds16.quantile_verdict(20, 0.95, 18, 1), "unresolved")

    def test_every_order_is_refused_before_anything_is_sorted(self):
        # Version 54 returned 100, NaN and 250 for these three orders of one record.
        for xs in self.orders():
            for function in (budget.p95, budget.from_raw, budget.known):
                with self.assertRaisesRegex(ValueError, "a duration is missing"):
                    function(xs)
            with self.assertRaisesRegex(ValueError, "a duration is missing"):
                budget.bracket(xs, budget.FITTED)
            with self.assertRaisesRegex(ValueError, "a duration is missing"):
                budget.p95(iter(xs))

    def test_the_budget_of_two_thousand(self):
        # Version 54: 99.1 ms with the NaN at 1, 249.1 ms with it at 1900, an error of the standard
        # deviation elsewhere.
        base = [100.0] * 1899 + [250.0] * 100
        self.assertEqual(sorted(base + [0.0])[1899], 100.0)       # rank 1900 of 2000 is the request in doubt
        self.assertEqual(sorted(base + [999.0])[1899], 250.0)
        for at in (0, 1, 500, 1899, 1900, 1999):
            ms = base[:at] + [NAN] + base[at:]
            self.assertEqual(len(ms), 2000)
            with self.assertRaisesRegex(ValueError, "a duration is missing"):
                budget.from_raw(ms)

    def test_a_complete_record_is_read_as_before(self):
        xs = self.KNOWN + [100.0]
        self.assertEqual(budget.p95(xs), 100.0)
        self.assertEqual(budget.p95(x for x in xs), 100.0)
        self.assertEqual(budget.known(x for x in xs), xs)
        text = printed(budget.main)
        self.assertEqual(text, (ROOT / "expected" / "ch22_budget.txt").read_text(encoding="utf-8"))
        self.assertIn("result (182.1 ± 6.5) ms, k = 2.12, p = 0.95\n", text)      # the line of Version 54

    def test_one_word_for_a_record_that_does_not_say(self):
        self.assertEqual(budget.verdict(100, 250), "unresolved")
        self.assertEqual(budget.verdict(100, 200), "met")
        self.assertEqual(budget.verdict(200, 250), "missed")
        source = (ROOT / "examples" / "ch22_budget.py").read_text(encoding="utf-8")
        self.assertNotIn("undecided", source)
        text = (ROOT / "expected" / "ch22_budget.txt").read_text(encoding="utf-8")
        self.assertIn("  p95 of the record in (100, 250] ms: unresolved\n", text)
        self.assertIn("  p95 of the record in (100, 200] ms: met\n", text)
        self.assertNotIn("undecided", text)


class ThreeBoundsOfTheFiveLines(unittest.TestCase):
    LINES = ("000041 2204 09:15:00.100 svc  request 9 start",
             "000042 3120 09:15:00.250 db   query 3310 done",
             "000044 2204 09:15:00.400 svc  request 9 start",
             "000045 2204 09:15:00.700 svc  request 9 end",
             "000046 3120 09:15:00.900 db   query 3311 done")
    SPAN = (trace.stamp("09:15:00.000"), trace.stamp("09:15:01.000"))

    def test_the_three_bounds_and_their_functions(self):
        rows = [trace.parse(x) for x in self.LINES]
        self.assertEqual(trace.completeness(rows, (41, 48))[:2], (2, 7))
        self.assertEqual(trace.verified(rows, 0, 1, self.SPAN, (41, 48)), 0.75)
        self.assertEqual(trace.verified(rows, 0, 1, self.SPAN), 0.65)
        # Every message of the component recorded: none in the whole second.
        self.assertAlmostEqual(stats.quiet_bound(1.0), upper(0, 1.0), places=9)
        self.assertAlmostEqual(stats.quiet_bound(1.0), math.log(20), places=12)
        self.assertEqual(f"{stats.quiet_bound(1.0):.2f}", "3.00")
        # The verified time, selected independently of its arrivals.
        self.assertAlmostEqual(stats.quiet_bound(0.75), upper(0, 0.75), places=9)
        self.assertEqual(f"{stats.quiet_bound(0.75):.2f}", "3.99")
        self.assertEqual(f"{stats.quiet_bound(0.65):.2f}", "4.61")
        # Two of seven missing, both counted as its own: two or fewer in the whole second.
        self.assertAlmostEqual(stats.rate_upper(2, 1.0), upper(2, 1.0), places=9)
        self.assertEqual(f"{stats.rate_upper(2, 1.0):.2f}", "6.30")
        self.assertEqual(f"{upper(2, 1.0):.5f}", "6.29579")

    def test_the_third_is_not_a_function_of_listing_16_1(self):
        # The book says where it comes from: rate_upper() of swmetro.stats, which Listing 16.4 imports.
        self.assertFalse(hasattr(trace, "rate_upper"))
        self.assertTrue(hasattr(trace, "quiet_bound"))
        from examples import ch16_silence as silence
        self.assertIs(silence.rate_upper, stats.rate_upper)

    def test_the_bounds_of_figure_16_3(self):
        self.assertEqual(f"{stats.quiet_bound(20):.5f}", f"{upper(0, 20):.5f}")
        self.assertEqual(f"{upper(0, 20):.5f}", "0.14979")
        self.assertEqual(f"{stats.rate_upper(1, 20):.5f}", f"{upper(1, 20):.5f}")
        self.assertEqual(f"{upper(1, 20):.5f}", "0.23719")

    def test_a_fixed_outage_leaves_the_bound_valid(self):
        # A recorder deaf from 0.40 s to 0.65 s of every second, whatever is written then: the 0.75 s
        # outside are an exposure fixed in advance, although messages of the component are lost. At the
        # rate of the bound the remaining time is silent 5 % of the time, as the bound says.
        rate = math.log(20) / 0.75
        rng, silent, lost, trials = random.Random(5401), 0, 0, 40000
        for _ in range(trials):
            t, seen, dropped = rng.expovariate(rate), 0, 0
            while t < 1.0:
                if 0.40 <= t < 0.65:
                    dropped += 1
                else:
                    seen += 1
                t += rng.expovariate(rate)
            silent += seen == 0
            lost += dropped > 0
        self.assertAlmostEqual(silent / trials, 0.05, delta=0.004)
        self.assertGreater(lost / trials, 0.5)              # its messages were lost in most of the seconds


class OneRequestForTwoEvents(unittest.TestCase):
    def test_two_workers_with_one_number(self):
        a = trace.stamp("09:15:00.100")
        rows = [(1, a, "A", "request 17 start"), (2, a + 600, "B", "request 17 end")]
        # Without a scope the text is all there is, and the caller answers for it.
        self.assertEqual(trace.time_delta(rows).value, 600)
        for worker in ("A", "B"):
            with self.assertRaisesRegex(ValueError, "no single pair"):
                trace.time_delta(rows, comp=worker)

    def test_the_scope_separates_two_requests(self):
        rows = [(1, 100, "A", "request 17 start"), (2, 300, "B", "request 17 start"),
                (3, 450, "B", "request 17 end"), (4, 700, "A", "request 17 end")]
        with self.assertRaisesRegex(ValueError, "no single pair"):
            trace.time_delta(rows)
        self.assertEqual(trace.time_delta(rows, comp="A").value, 600)
        self.assertEqual(trace.time_delta(rows, comp="B").value, 150)
        self.assertAlmostEqual(trace.time_delta(rows, comp="B").u, 1 / math.sqrt(6), places=12)

    def test_the_fixture_is_read_as_before(self):
        text = (ROOT / "data" / "app_trace.log").read_text(encoding="utf-8")
        rows = [trace.parse(x) for x in text.splitlines()]
        self.assertEqual({c for _, _, c, m in rows if m in trace.PAIR}, {"svc"})
        self.assertEqual(trace.time_delta(rows).value, 2346)
        self.assertEqual(trace.time_delta(rows, comp="svc").value, 2346)
        with self.assertRaisesRegex(ValueError, "no single pair"):
            trace.time_delta(rows, comp="db")


class EndsOfAWindow(unittest.TestCase):
    SPAN = (0, 3000)

    def test_the_line_at_2007_ms(self):
        self.assertEqual(repr(1000 * 2.007), "2007.0000000000002")
        rows = [(1, 2007, "svc", "x")]
        for end in (2.007, Fraction(2007, 1000), Decimal("2.007"), "2.007"):
            got = (trace.count(rows, "svc", 0, end, self.SPAN), trace.count(rows, "svc", end, 3, self.SPAN))
            self.assertEqual(got, (0, 1), end)                # Version 54 with the float: (1, 0)

    def test_every_millisecond_of_the_capture(self):
        # A line stamped k ms belongs to [k ms, 3 s) and not to [0, k ms), for the float k / 1000.
        wrong = 0
        for k in range(1, 3000):
            rows = [(1, k, "svc", "x")]
            self.assertEqual(trace.count(rows, "svc", 0, k / 1000, self.SPAN), 0, k)
            self.assertEqual(trace.count(rows, "svc", k / 1000, 3, self.SPAN), 1, k)
            wrong += 1000 * (k / 1000) > k
        self.assertEqual(wrong, 11)                            # ends that the product of floats put past the line

    def test_ticks(self):
        self.assertEqual(trace.ticks(2.007), 2007)
        self.assertIs(type(trace.ticks(2.007)), int)
        self.assertIs(type(trace.ticks(30)), int)
        self.assertEqual(trace.ticks(Decimal("0.0005")), Fraction(1, 2))
        self.assertEqual(trace.ticks(0.0005), Fraction(1, 2))
        for bad in (math.nan, math.inf, "soon"):
            with self.assertRaises(ValueError):
                trace.ticks(bad)
            with self.assertRaises(ValueError):
                trace.window(0, bad, self.SPAN)

    def test_verified_seconds_are_floats(self):
        rows = [(1, 100, "db", "m"), (2, 900, "db", "m")]
        for lo, hi, want in ((0, 3, 0.8), (0, 0.5, 0.4), (0.2505, 0.7505, 0.5), (Fraction(1, 8), 1, 0.775)):
            got = trace.verified(rows, lo, hi, self.SPAN)
            self.assertIs(type(got), float)
            self.assertEqual(got, want, (lo, hi))
        with self.assertRaisesRegex(ValueError, "outside the capture"):
            trace.window(0, 3.0005, self.SPAN)

    def test_one_day(self):
        late, early = trace.stamp("23:59:59.900"), trace.stamp("00:00:00.200")
        self.assertEqual((late, early), (86_399_900, 200))
        rows = [(1, late, "svc", "request 17 start"), (2, early, "svc", "request 17 end")]
        with self.assertRaisesRegex(ValueError, "stamps go back"):
            trace.time_delta(rows)
        self.assertIn("one day", " ".join(trace.stamp.__doc__.split()))


def keeps_the_contract(rows, counter):
    """The contract of numbers and readings, written out apart from the listing."""
    numbers = [r[0] for r in rows]
    readings = list(counter) if counter else []
    if any(x != int(x) for x in numbers + readings):
        return False
    if counter and counter[1] < counter[0]:
        return False
    if counter and numbers:
        return counter[0] <= min(numbers) and max(numbers) < counter[1]
    return True


class OneContractForCountsAndExposure(unittest.TestCase):
    SPAN = (0, 1000)

    def refused(self, function, *args):
        try:
            function(*args)
        except ValueError:
            return True
        return False

    def test_the_rows_of_the_review(self):
        rows = [(41.5, 100, "db", "m"), (42.5, 900, "db", "m")]
        self.assertTrue(self.refused(trace.completeness, rows, (41, 44)))
        for function in (trace.runs, trace.stretches, trace.verified):          # Version 54: 0.8 s verified
            with self.assertRaisesRegex(ValueError, "whole number"):
                function(rows, 0, 1, self.SPAN, (41, 44))
        whole = [(41, 100, "db", "m"), (42, 900, "db", "m")]
        with self.assertRaisesRegex(ValueError, "whole reading"):               # Version 54: 0.9 s
            trace.verified(whole, 0, 1, self.SPAN, (41, 43.5))
        with self.assertRaisesRegex(ValueError, "counter runs back"):           # Version 54: 0.0 s
            trace.verified(whole, 0, 1, self.SPAN, (48, 41))
        with self.assertRaisesRegex(ValueError, "beyond the counter"):
            trace.verified(whole, 0, 1, self.SPAN, (42, 48))
        self.assertEqual(trace.verified(whole, 0, 1, self.SPAN, (41, 43)), 1.0)
        self.assertEqual(trace.verified(whole, 0, 1, self.SPAN, (41, 44)), 0.9)
        self.assertEqual(trace.verified(whole, 0, 1, self.SPAN), 0.8)

    def test_both_refuse_what_the_contract_refuses(self):
        rng, seen = random.Random(5403), {True: 0, False: 0}
        for _ in range(3000):
            first = rng.randrange(-5, 60)
            numbers = sorted(rng.sample(range(first, first + 12), rng.randrange(0, 6)))
            stamps = sorted(rng.sample(range(1, 1000), len(numbers)))
            rows = [(n, t, "db", "m") for n, t in zip(numbers, stamps)]
            counter = (first, first + 12) if rng.random() < 0.7 else None
            kind = rng.randrange(6)
            if kind == 1 and rows:
                i = rng.randrange(len(rows))
                rows[i] = (rows[i][0] + 0.5,) + rows[i][1:]
            elif kind == 2 and counter:
                counter = (counter[0], counter[1] + 0.25)
            elif kind == 3 and counter:
                counter = (counter[1] + 1, counter[0])
            elif kind == 4 and counter and rows:
                counter = (rows[-1][0] + 1, counter[1] + 20) if rng.random() < 0.5 else (counter[0], rows[-1][0])
            if not (rows or counter):
                continue                                       # no line and no counter: nothing to count
            want = keeps_the_contract(rows, counter)
            self.assertEqual(not self.refused(trace.completeness, rows, counter), want, (rows, counter))
            self.assertEqual(not self.refused(trace.verified, rows, 0, 1, self.SPAN, counter), want, (rows, counter))
            self.assertEqual(not self.refused(trace.runs, rows, 0, 1, self.SPAN, counter), want, (rows, counter))
            seen[want] += 1
        self.assertGreater(min(seen.values()), 700, seen)

    def test_what_still_differs_is_said(self):
        # Numbers that do not rise: no count can be given, and no time is verified. The first is refused
        # and the second is an answer, as the two docstrings say.
        rows = [(5, 100, "db", "m"), (6, 400, "db", "m"), (3, 450, "db", "m"), (4, 500, "db", "m")]
        with self.assertRaisesRegex(ValueError, "numbers do not rise"):
            trace.completeness(rows)
        self.assertEqual(trace.verified(rows, 0, 1, self.SPAN), 0.0)
        self.assertEqual(trace.numbered(rows), ([5, 6, 3, 4], 3, 7))
        self.assertEqual(trace.numbered([], (41, 48)), ([], 41, 48))


def digits_needed(text):
    """Significant digits of a decimal as it is written, trailing zeros apart."""
    mantissa = text.lower().split("e")[0].replace("-", "").replace(".", "").lstrip("0")
    return len(mantissa.rstrip("0"))


class WhatRoundingCanHaveWritten(unittest.TestCase):
    def shown(self):
        return check.displayed(stored())

    def test_the_record_of_the_review(self):
        rec = self.shown()
        self.assertEqual(rec["result"]["significant_digits"], 2)
        self.assertEqual(R.invalid(rec), [])
        self.assertEqual(R.decide(rec)[0], "accept")
        rec["components"] = [{"name": "one", "type": "A", "kind": "stipulated", "u": 1.234, "dof": 15,
                              "source": "set"}]
        rec["result"]["u"], rec["result"]["dof"] = 1.234, 15
        why = ["result.u is not rounded to 2 digits", "components[0].u is not rounded to 2 digits"]
        self.assertEqual(R.invalid(rec), why)
        self.assertEqual(R.decide(rec), ("invalid", why))                       # Version 54: accept
        rec["components"][0]["u"] = rec["result"]["u"] = 1.2
        self.assertEqual(R.invalid(rec), [])

    def test_digits_as_they_are_written(self):
        cases = (("1.2", True), ("1.20", True), ("1.200000", True), ("0.30", True), ("0.0012", True),
                 ("1.2E+3", True), ("1200", True), ("12", True), ("1", True), ("0", True), ("0.0", True),
                 ("1.23", False), ("123", False), ("0.00123", False), ("1.201", False), ("12.5", False),
                 ("1.2000000000000000000000000000000001", False), ("1.23E+3", False), ("1230", False))
        for text, fine in cases:
            self.assertEqual(digits_needed(text) <= 2, fine, text)
            rec = self.shown()
            rec["components"] = [{"name": "one", "type": "B", "kind": "stipulated", "u": Decimal(text)}]
            rec["result"]["u"] = Decimal(text)
            said = [w for w in R.invalid(rec) if "rounded" in w]
            self.assertEqual(said == [], fine, (text, said))
            if not fine:
                self.assertEqual(R.decide(rec)[0], "invalid", text)

    def test_the_degrees_of_freedom_and_other_precisions(self):
        for dof, n, fine in ((15, 2, True), (16.0, 2, True), (150, 2, True), (15.7, 2, False), (157, 2, False),
                             (15.7, 3, True), (15.7327, 6, True), (15.7327, 5, False), (7, 1, True), (15, 1, False)):
            rec = self.shown()
            rec["result"]["significant_digits"] = n
            rec["result"]["dof"] = dof
            said = [w for w in R.invalid(rec) if "dof" in w]
            want = [] if fine else [f"result.dof is not rounded to {n} digits"]
            self.assertEqual(said, want, (dof, n))

    def test_random_numbers_rounded_and_not(self):
        rng, seen = random.Random(5404), {True: 0, False: 0}
        for _ in range(2000):
            n = rng.randrange(1, 7)
            u = 10 ** rng.uniform(-3, 3)
            more = rng.randrange(0, 3)
            text = f"{u:.{n - 1 + more}e}"
            fine = digits_needed(text) <= n
            rec = self.shown()
            rec["result"]["significant_digits"] = n
            rec["components"] = [{"name": "one", "type": "B", "kind": "stipulated", "u": float(text)}]
            rec["result"]["u"] = Decimal(text)
            rec["result"]["dof"] = 7
            said = [w for w in R.invalid(rec) if "rounded" in w]
            self.assertEqual(len(said), 0 if fine else 2, (text, n, said))
            seen[fine] += 1
        self.assertGreater(min(seen.values()), 500, seen)

    def test_a_working_record_states_its_digits_in_full(self):
        rec = stored()
        self.assertEqual(rec["result"]["significant_digits"], R.ALL)
        self.assertEqual(R.invalid(rec), [])
        self.assertEqual(R.decide(rec), ("accept", []))

    def test_the_estimate_and_its_increment(self):
        rec = self.shown()
        self.assertEqual((rec["result"]["value"], rec["result"]["value_step"]), (Decimal("182.1"), 0.1))
        for value, step, fine in (("182.1", "0.1", True), ("182.137", "0.1", False), ("182.137", "0", True),
                                  ("180", "20", True), ("182", "20", False), ("0.3", 0.1, True),
                                  (0.30000000000000004, 0.1, False), ("-182.1", "0.1", True),
                                  ("182.15", "0.05", True), ("182.13", "0.05", False)):
            rec = self.shown()
            rec["result"]["value"] = Decimal(value) if isinstance(value, str) else value
            rec["result"]["value_step"] = Decimal(step) if isinstance(step, str) else step
            want = [] if fine else ["result.value is not a multiple of its step"]
            self.assertEqual(R.invalid(rec), want, (value, step))
            self.assertEqual(Fraction(str(value)) % Fraction(str(step)) == 0 if Fraction(str(step)) else True,
                             fine, (value, step))

    def test_values_that_round_to_ten_and_to_minus_ten(self):
        self.assertEqual(R.bounds(10, 2), (Fraction("9.95"), Fraction("10.5")))
        self.assertEqual(R.bounds(-10, 2), (Fraction("-10.5"), Fraction("-9.95")))          # Version 54: -9.5
        # Every point of a grid of thousandths, rounded to two digits both ways of breaking a tie.
        for sign in (1, -1):
            for rule in (ROUND_HALF_EVEN, ROUND_HALF_UP):
                two = Context(prec=2, rounding=rule)
                hits = [Fraction(k, 1000) for k in range(-12000, 12001)
                        if two.create_decimal(Decimal(k) / 1000) == 10 * sign]
                lo, hi = R.bounds(10 * sign, 2)
                self.assertGreaterEqual(min(hits), lo)
                self.assertLessEqual(max(hits), hi)
                self.assertLessEqual(min(hits) - lo, Fraction(1, 1000))
                self.assertLessEqual(hi - max(hits), Fraction(1, 1000))

    def test_the_interval_below_zero_is_the_mirror(self):
        for x in (10, 1, 100, "0.1", "0.001", "9.6", "1.5", 1000, "99", "0.10"):
            for n in (1, 2, 3, 6):
                x_dec = Decimal(str(x))
                lo, hi = R.bounds(x_dec, n)
                self.assertEqual(R.bounds(-x_dec, n), (-hi, -lo), (x, n))
                self.assertTrue(lo <= Fraction(x_dec) <= hi)
        self.assertEqual(R.bounds(-10, R.ALL), (-10, -10))
        self.assertEqual(R.bounds(Decimal("-9.6"), 2), (Fraction("-9.65"), Fraction("-9.55")))


class NamesInARelationAndRecordsThatAreNone(unittest.TestCase):
    def pair(self, clocks):
        a, b = stored(), stored()
        a["capture"]["clock"], b["capture"]["clock"] = "7", "host-B"
        if clocks is not None:
            a["capture"]["clock_relations"] = [{"clocks": clocks, "of": "rate", "state": "established"}]
        return a, b

    def test_a_number_names_no_clock(self):
        self.assertEqual(R.comparable(*self.pair(None)), (["clock rate relation"], []))
        self.assertEqual(R.comparable(*self.pair(["7", "host-B"])), ([], []))
        self.assertEqual(R.comparable(*self.pair(["host-B", "7"])), ([], []))
        for clocks in ([7, "host-B"], ["host-B", 7], [7.0, "host-B"], [Decimal(7), "host-B"], [True, "host-B"],
                       [None, "host-B"], [["7"], "host-B"], ["7"], ["7", "host-B", "host-C"], [], "7host-B",
                       {"7": "host-B"}, ["7", "host-A"]):
            got = R.comparable(*self.pair(clocks))                               # Version 54 took the first
            self.assertEqual(got, (["clock rate relation"], []), clocks)

    def test_the_relation_may_stand_in_either_record(self):
        a, b = self.pair(None)
        b["capture"]["clock_relations"] = [{"clocks": [7, "host-B"], "of": "rate", "state": "established"},
                                           {"clocks": ["host-B", "7"], "of": "rate", "state": "established"}]
        self.assertEqual(R.comparable(a, b), ([], []))
        b["capture"]["clock_relations"].pop()
        self.assertEqual(R.comparable(a, b), (["clock rate relation"], []))

    def test_a_record_is_an_object(self):
        for text in ("[]", "null", "7", '"x"', "true", "1.5", '[{"format": 5}]'):
            with self.assertRaisesRegex(ValueError, "a record is an object"):
                R.read(text)                                                     # Version 54 returned the value
        self.assertEqual(R.read("{}"), {})
        self.assertEqual(R.read('{"format": 5}'), {"format": 5})

    def test_a_value_that_is_no_record_is_invalid(self):
        for value in ([], None, 7, "x", True, [{"format": 5}]):
            self.assertEqual(R.invalid(value), ["the record is not an object"], value)
            self.assertEqual(R.decide(value), ("invalid", ["the record is not an object"]), value)
            self.assertEqual(R.missing(value), R.missing({}), value)             # Version 54: AttributeError
            self.assertIsNone(R.stale(value))
            reasons, notes = R.comparable(value, stored())
            self.assertIn("quantity unknown", reasons)
            self.assertIn("clock unknown", reasons)
        self.assertEqual(R.decide({})[0], "unresolved")


class EdgesAsTheyWereGiven(unittest.TestCase):
    def test_an_edge_next_to_the_objective(self):
        below, above = Decimal("199.99999999999999999"), Decimal("200.00000000000000001")
        self.assertEqual((float(below), float(above)), (200.0, 200.0))
        # Twenty readings of 200 ms lie above the first edge: the bucket straddles the objective.
        self.assertTrue(below < 200.0 < 250)
        got = budget.bracket([200.0] * 20, [below, 250])
        self.assertEqual(got, (below, 250))
        self.assertIs(got[0], below)
        self.assertEqual(budget.verdict(*got), "unresolved")                     # Version 54: missed
        # Twenty readings at the second edge are above the objective, and the counts do not say so.
        got = budget.bracket([above] * 20, [100, above, 250])
        self.assertEqual(got, (100, above))
        self.assertEqual(budget.verdict(*got), "unresolved")                     # Version 54: met
        self.assertTrue(all(x > budget.LIMIT for x in [above] * 20))

    def test_edges_of_other_kinds(self):
        third = Fraction(1, 3)
        self.assertEqual(budget.bracket([third] * 20, [third, 1]), (-math.inf, third))
        self.assertEqual(budget.bracket([150.0] * 20, budget.COARSE), (100, 250))
        self.assertEqual(budget.bracket([150.0] * 20, budget.FITTED), (100, 200))
        self.assertEqual(budget.bracket([600.0] * 20, budget.FITTED), (500, math.inf))
        for bad in ([100, 100], [250, 100], [100, NAN], [NAN, 100]):
            with self.assertRaisesRegex(ValueError, "bounds do not rise"):
                budget.bracket([150.0] * 20, bad)

    def test_the_interpolation_helper_examines_its_bounds(self):
        # Version 54: 50.0, 0.5 and an IndexError for the first three.
        for bad in ([100, 2], [1, NAN], [], [1, 1], [-1, 5], [1, math.inf], [NAN], [5, 1, 2]):
            with self.assertRaisesRegex(ValueError, "no rising bounds >= 0"):
                h9.bucket_quantile([1, 1], 0.5, bad)
            with self.assertRaisesRegex(ValueError, "no rising bounds >= 0"):
                h9.bucket_quantile([], 0.5, bad)                                # a schema is wrong with no value too
        self.assertEqual(h9.bucket_quantile([1, 1], 0.5, [2, 100]), 1.0)      # rank 1 of 2 in [0, 2]
        # What a search that presumes order makes of the first two: both values in the first bucket, and
        # the middle of [0, 100] and of [0, 1].
        self.assertEqual((bisect.bisect_left([100, 2], 1), bisect.bisect_left([1, NAN], 1)), (0, 0))
        self.assertEqual((0 + 100 * Fraction(1, 2), 0 + 1 * Fraction(1, 2)), (50, Fraction(1, 2)))
        self.assertEqual(h9.bucket_quantile([0, 0, 1], 0.5, [0, 1, 2]), 0.0)  # a bound at zero is a bucket of zeros
        self.assertTrue(math.isnan(h9.bucket_quantile([], 0.5)))
        self.assertEqual(h9.BOUNDS, sorted(h9.BOUNDS))


class TheSketchInFloats(unittest.TestCase):
    EXTRA = Fraction(1, 10 ** 12)

    def single(self, x, alpha):
        sk = h9.LogSketch(alpha)
        sk.add(x)
        return sk.quantile(0.5)

    def error(self, x, alpha):
        return abs(Fraction(self.single(x, alpha)) / Fraction(x) - 1)

    def test_the_guarantee_that_the_floats_keep(self):
        rng = random.Random(5405)
        lo, hi = math.log(sys.float_info.min), math.log(sys.float_info.max)
        worst = 0
        for alpha in (0.5, 0.01, 1e-6, 1e-9, 1e-12, 1e-16):
            sk = h9.LogSketch(alpha)
            for _ in range(400):
                x = math.exp(rng.uniform(lo, hi))
                near = [x]
                with contextlib.suppress(OverflowError):
                    edge = math.exp(math.ceil(math.log(x) / sk.lg) * sk.lg)   # next to the end of a bucket
                    near += [edge, math.nextafter(edge, math.inf), math.nextafter(edge, 0)]
                for y in near:
                    if sys.float_info.min <= y < math.inf:
                        excess = self.error(y, alpha) - Fraction(repr(alpha))
                        self.assertLessEqual(excess, self.EXTRA, (alpha, y))
                        worst = max(worst, excess)
        # The guarantee of real numbers is not kept: what the floats add is near one part in 1e13.
        self.assertGreater(worst, Fraction(1, 10 ** 14))
        self.assertLess(worst, Fraction(1, 10 ** 12) / 2)
        for x in (sys.float_info.min, sys.float_info.max, 1.0, math.nextafter(1.0, 0)):
            for alpha in (0.01, 1e-9, 1e-15):
                self.assertLessEqual(self.error(x, alpha), Fraction(repr(alpha)) + self.EXTRA, (x, alpha))

    def test_an_accuracy_finer_than_the_floats(self):
        # Near 100 two neighboring doubles differ by 1.4 parts in 1e16: no arithmetic in doubles keeps 1e-16.
        self.assertEqual(math.ulp(100.0), 2.0 ** -46)
        self.assertEqual(f"{math.ulp(100.0) / 100:.1e}", "1.4e-16")
        got = self.error(100.0, 1e-16)
        self.assertLessEqual(got, Fraction(1, 10 ** 16) + self.EXTRA)
        self.assertLessEqual(self.error(100.0, 0.01), Fraction(1, 100) + self.EXTRA)

    def test_below_the_normal_doubles(self):
        least = math.ldexp(1.0, -1074)
        self.assertEqual(h9.LEAST, sys.float_info.min)
        self.assertEqual(f"{h9.LEAST:.1e}", "2.2e-308")
        # Sixty of the least step: the doubles on either side are 1/60 away, 1.67 %.
        self.assertEqual(math.nextafter(60 * least, math.inf), 61 * least)
        self.assertEqual(f"{100 / 60:.2f}", "1.67")
        for x in (60 * least, least, math.nextafter(h9.LEAST, 0), 0.0, -1.0):
            with self.assertRaisesRegex(ValueError, "need 2..-1022 <= x < inf"):   # Version 54: 59 for the first
                h9.LogSketch(0.01).add(x)
        for x in (math.inf, NAN):                        # since Version 57 refused by known() of the package
            with self.assertRaisesRegex(ValueError, "a duration is missing or too large"):
                h9.LogSketch(0.01).add(x)
        self.assertLessEqual(self.error(h9.LEAST, 0.01), Fraction(1, 100) + self.EXTRA)


class PrefixesAndRelease(unittest.TestCase):
    def test_binary_against_decimal_prefixes(self):
        excess = [100 * (Fraction(1024, 1000) ** k - 1) for k in range(1, 7)]
        shown = [f"{float(e):.2f}" for e in excess]
        self.assertEqual(shown, ["2.40", "4.86", "7.37", "9.95", "12.59", "15.29"])
        self.assertTrue(all(2 <= e <= 10 for e in excess[:4]))                    # kibi to tebi
        self.assertTrue(all(e > 10 for e in excess[4:]))                          # pebi and exbi

    def test_the_release(self):
        # The record and the README name one version of the book, 55 or a later one.
        rec = stored()
        version = re.fullmatch(r"book Version (\d+), \w+ \d{4}", rec["sources"]["release"])
        self.assertIsNotNone(version, rec["sources"]["release"])
        self.assertGreaterEqual(int(version.group(1)), 55)
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn(f"Book Version {version.group(1)},", readme)
        # "This is the release for Version N of the book" until Version 59; no release is published.
        self.assertIn(f"These are the files for Version {version.group(1)} of the book.", readme)


if __name__ == "__main__":
    unittest.main()
