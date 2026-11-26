"""Regression tests added after the review of Version 58.

The review found that recorded() of Listing 23.1, as Version 58 had changed it, made a float of an exact sum, so
that twenty requests of exactly 200.3 ms fell on a decimal bucket bound of 200.3 ms from handler entry and above
it from accept; that the change between two periods was made a float in the same way, so that a slowdown equal to
a margin was reported as below it; that a bootstrap of twenty equal durations gives a bound of 0 ms and says
nothing of a tail that the trace did not reach; that ten upper bounds of Transcript 16.4, and others, were
printed rounded to the nearest and so below the bound; that a reference mix of iterators raised TypeError; that
elapsed times below zero were added and counted; that an infinite limit left a bounded request open; that the
bounds of the histogram were not held to the range of a float; and that a negative step could pass as zero.

Each test computes what it expects independently of the code under test: sums and differences in fractions,
shares by counting, chances from powers of fractions, and the printed form of a bound from the decimal module.
"""
import contextlib
import io
import math
import random
import unittest
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal, Inexact, localcontext
from fractions import Fraction
from pathlib import Path

from examples import ch09_dependent_tail as tail
from examples import ch09_histogram as h9
from examples import ch11_leak_rate as leak
from examples import ch16_silence as silence
from examples import ch18_calibration as cal
from examples import ch20_judge as judge
from examples import ch22_budget as budget
from examples import ch23_case as case
from examples import ch23_mix as mix
from swmetro import stats

ROOT = Path(__file__).resolve().parent.parent
NAN = math.nan
D = Decimal


def expected(name):
    return (ROOT / "expected" / f"{name}.txt").read_text(encoding="utf-8")


def printed(function, *args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        function(*args)
    return out.getvalue()


def up(x, places):
    """x rounded toward +inf at so many places, by the decimal module: a float as the binary number it is. Since
    Version 62 the upper bound of Listing 23.1 is a fraction with a power of two below the line: its decimal ends."""
    if isinstance(x, Fraction):
        with localcontext() as context:
            context.prec = 4000
            context.traps[Inexact] = True
            x = Decimal(x.numerator) / Decimal(x.denominator)
    return str(Decimal(x).quantize(Decimal(1).scaleb(-places), rounding=ROUND_CEILING))


def down(x, places):
    return str(Decimal(x).quantize(Decimal(1).scaleb(-places), rounding=ROUND_FLOOR))


class OneDurationOneBucket(unittest.TestCase):
    ROWS = [(D("0.0"), D("200.3"))] * 20
    EDGES = [D("100"), D("200.3"), D("250")]

    def test_the_twenty_requests_of_the_review(self):
        # No wait: from either start the duration is 200.3 ms, on the inclusive upper bound of the second
        # bucket. Rank 19 of its 20 lies 19/20 of the way from 100 to 200.3.
        want = 100 + Fraction("100.3") * Fraction(19, 20)
        self.assertEqual(want, Fraction("195.285"))
        for start in ("handler", "accept"):
            config = {"start": start, "bounds": self.EDGES}
            xs = case.recorded(self.ROWS, config)
            self.assertTrue(all(Fraction(x) == Fraction("200.3") for x in xs), start)
            self.assertEqual(sum(x <= D("200.3") for x in xs), 20, start)        # Version 58: 0 from accept
            self.assertAlmostEqual(case.reported(self.ROWS, config), float(want), places=9)   # Version 58: 247.515
            rec = {"all": xs}
            low, high = mix.within(rec, rec, D("200.3"))
            self.assertEqual((low, high, mix.verdict(low, high)), (1, 1, "met"), start)
            self.assertEqual(case.p95(xs), D("200.3"))

    def test_a_decimal_tenth_stays_on_its_own_bound(self):
        # Made a float, 401 of the thousand tenths from 0.1 to 100.0 lie above the decimal that they stand for.
        above = [n for n in range(1, 1001) if Fraction(float(Fraction(n, 10))) > Fraction(n, 10)]
        self.assertEqual(len(above), 401)
        for n in range(1, 1001):
            edge = D(n) / 10
            (total,) = case.recorded([(D("0.0"), edge)], {"start": "accept"})
            self.assertEqual(total, Fraction(n, 10))
            self.assertLessEqual(total, edge)
            self.assertEqual(h9.bucket_quantile([total], 1, [edge, 200]), float(edge))

    def test_what_is_added_and_how(self):
        accept = {"start": "accept"}
        for q, h, want, kind in ((D("2.5"), 100.0, Fraction(205, 2), Fraction), (D("0.1"), D("0.2"), Fraction(3, 10), Fraction),
                                 (Fraction(1, 3), 1, Fraction(4, 3), Fraction), (2, 3, 5, Fraction),
                                 (0.1, D("0.2"), Fraction(0.1) + Fraction(1, 5), Fraction), (0.1, 0.2, 0.1 + 0.2, float)):
            (got,) = case.recorded([(q, h)], accept)
            self.assertEqual((got, type(got)), (want, kind), (q, h))
        self.assertEqual(case.pair(1.5, 2.5), (1.5, 2.5))
        self.assertEqual([type(x) for x in case.pair(1.5, 2)], [Fraction, Fraction])
        self.assertNotEqual(Fraction(3, 10), 0.3)                               # which is why the sum is not made a float

    def test_the_two_periods_of_the_book_are_as_they_were(self):
        rng = random.Random(23)
        before, after = case.period(rng), case.period(rng)
        for rows in (before, after):
            self.assertEqual(case.recorded(rows, case.NEW), [q + h for q, h in rows])
            self.assertTrue(all(type(x) is float for x in case.recorded(rows, case.NEW)))


class AMarginThatIsReached(unittest.TestCase):
    BEFORE = [(0, D("100.0"))] * 20
    AFTER = [(0, D("100.3"))] * 20

    def test_the_change_of_the_review(self):
        d = case.handler_change(self.BEFORE, self.AFTER)
        self.assertEqual((d.value, d.u), (Fraction(3, 10), 0.0))                # Version 58: the float 0.3
        self.assertLess(0.3, Fraction(3, 10))                                   # which lies below three tenths
        upper = case.upper_bound(d)
        self.assertEqual(upper, Fraction(3, 10))
        # A bound equal to the margin is not below it. Version 58: True for the first.
        for margin, below in ((D("0.3"), False), (Fraction(3, 10), False), (D("0.30000000000000000001"), True),
                              (D("0.29999999999999999999"), False), (D("0.31"), True), (D("0.29"), False), (1, True), (0, False)):
            self.assertIs(case.within(upper, margin), below, margin)
        # A margin written as the float 0.3 is the binary number below three tenths.
        self.assertIs(case.within(upper, 0.3), False)

    def test_a_bound_with_a_half_width(self):
        rng = random.Random(58)
        before = [(0, D(rng.randint(900, 1300)) / 10) for _ in range(40)]
        after = [(q, h + D("0.3")) for q, h in before]
        d = case.handler_change(before, after)
        self.assertEqual(d.value, Fraction(3, 10))
        self.assertGreater(d.u, 0)
        upper = case.upper_bound(d)
        # Version 61 and before: Fraction(case.Z * d.u), the half-width as the float product. The review of
        # Version 61 showed a product that rounds below a margin which the bound exceeds: now the exact product.
        self.assertEqual(upper, Fraction(3, 10) + Fraction(case.Z) * Fraction(d.u))
        self.assertIs(case.within(upper, upper), False)
        self.assertIs(case.within(upper, upper + Fraction(1, 10 ** 30)), True)
        self.assertIs(case.within(upper, upper - Fraction(1, 10 ** 30)), False)

    def test_floats_are_subtracted_as_floats(self):
        before = [(0.0, 100.0)] * 20
        after = [(0.0, 100.3)] * 20
        d = case.handler_change(before, after)
        self.assertEqual((d.value, type(d.value)), (100.3 - 100.0, float))
        self.assertEqual(case.upper_bound(d), d.value + case.Z * d.u)
        # The float 100.3 lies below 100.3, and so does the change below 0.3: the readings are what was stored.
        self.assertLess(Fraction(100.3) - 100, Fraction(3, 10))
        self.assertIs(case.within(case.upper_bound(d), D("0.3")), True)


class ATailThatTheTraceDidNotReach(unittest.TestCase):
    def test_what_the_twenty_requests_show(self):
        rows = [(0.0, 100.0)] * 20
        d = case.handler_change(rows, rows)
        self.assertEqual((d.value, d.u, case.upper_bound(d)), (0.0, 0.0, 0.0))
        self.assertIs(case.within(case.upper_bound(d), 5.0), True)             # of the observed p95s, and of no more
        # The populations of the review: every request of 100 ms, and then one in ten of 200 ms.
        after = sorted([100] * 9 + [200])
        self.assertEqual(after[math.ceil(Fraction(95, 100) * 10) - 1], 200)

    def test_the_three_numbers(self):
        miss = Fraction(9, 10) ** 20
        reach = 1 - Fraction(19, 20) ** 20
        self.assertEqual((f"{float(100 * miss):.2f}", f"{float(100 * reach):.2f}"), ("12.16", "64.15"))
        self.assertGreater(miss, Fraction(5, 100))                              # more than the 5 % of the bound
        first = next(n for n in range(1, 200) if 1 - Fraction(19, 20) ** n >= Fraction(95, 100))
        self.assertEqual(first, 59)
        self.assertEqual(stats.run_length(0.95, 0.05), 59)
        text = expected("ch23_case")
        # The five lines of Version 59. Version 60 added a line on the bound by ranks; since Version 61 three lines
        # say which end is missing, and the transcript goes on with the requests that are reported twice.
        lines = ("20 requests of 100 ms, twice:\n  difference 0.0(0) ms\n"
                 "  no slow one among them: 12.16 %\n  the largest above the p95: 64.15 %\n"
                 "  requests for 95 %: 59\nthe p95 by ranks:\n  requests: 20 before, 20 after\n"
                 "  no upper end with fewer than 72 after\n")
        self.assertIn(lines + "100 requests reported twice, 1 ms apart:\n", text)
        self.assertEqual(printed(case.short_trace), lines)


class BoundsArePrintedOnTheirSafeSide(unittest.TestCase):
    def test_the_two_functions(self):
        for x, places, most, least in ((0.185230463, 5, "0.18524", "0.18523"), (D("0.18523"), 5, "0.18523", "0.18523"),
                                       (Fraction(1, 3), 3, "0.334", "0.333"), (-5.93, 1, "-5.9", "-6.0"),
                                       (4.7078728237108995, 1, "4.8", "4.7"), (100.0, 3, "100.000", "100.000"),
                                       (7, 2, "7.00", "7.00"), (3e-9, 8, "0.00000001", "0.00000000"),
                                       (D("12.190000000000000000001"), 2, "12.20", "12.19"), (0, 5, "0.00000", "0.00000")):
            self.assertEqual((stats.upward(x, places), stats.downward(x, places)), (most, least), x)
        rng = random.Random(58)
        for _ in range(2000):
            x, places = rng.uniform(-50, 50) * 10 ** rng.randint(-6, 3), rng.randint(0, 7)
            self.assertEqual(Decimal(stats.upward(x, places)), Decimal(up(x, places)))
            self.assertEqual(Decimal(stats.downward(x, places)), Decimal(down(x, places)))
            self.assertLessEqual(Fraction(stats.downward(x, places)), Fraction(x))
            self.assertGreaterEqual(Fraction(stats.upward(x, places)), Fraction(x))
        # A float is the binary number that it is: 0.1 lies above one tenth, and 0.3 below three tenths.
        self.assertEqual((stats.upward(0.1, 1), stats.upward(D("0.1"), 1), stats.upward(0.3, 1)), ("0.2", "0.1", "0.3"))

    def rates_of_the_silence_example(self):
        seen, kept = [], silence.most

        def spy(rate):
            seen.append(rate)
            return kept(rate)
        silence.most = spy
        try:
            text = printed(silence.main)
        finally:
            silence.most = kept
        return seen, text

    def test_the_twenty_nine_rates_of_transcript_16_4(self):
        seen, text = self.rates_of_the_silence_example()
        self.assertEqual(text, expected("ch16_silence"))
        self.assertEqual(len(seen), 29)
        lines = [line.strip() for line in text.splitlines() if line.strip().startswith("at most") and "msg/s" in line]
        self.assertEqual(lines, [f"at most {up(rate, 5)} msg/s at 95 %" for rate in seen])
        for rate, line in zip(seen, lines):
            self.assertGreaterEqual(Fraction(line.split()[2]), Fraction(rate))   # Version 58: below it for ten
        moved = sorted(f"{rate:.5f}" for rate in seen if f"{rate:.5f}" != up(rate, 5))
        self.assertEqual(moved, ["0.07489", "0.17584", "0.18523", "0.20565", "0.20565", "0.23719", "0.26313",
                                 "0.29957", "0.39943", "0.43574"])
        self.assertEqual(sorted(f"{rate:.9f}" for rate in seen if f"{rate:.5f}" != up(rate, 5))[2], "0.185230463")

    def test_the_share_and_the_named_rates(self):
        top = 100 * judge.exact_interval(2, 20, 0.95)[1]
        self.assertEqual((f"{top:.7f}", f"{top:.3f}", up(top, 3)), ("31.6982714", "31.698", "31.699"))
        text = expected("ch16_silence")
        self.assertIn("  at most 31.699 % of 10 s, at 97.5 %:\n", text)
        # A rate that is named where a chance is taken at it is printed as the bound that it is.
        for line in ("    0.11523 msg/s at a nominal 95 %", "    0.10331 msg/s at a nominal 95 %",
                     "    silent at the nominal rate: 6.26 %", "    silent at the first bound: 6.42 %",
                     "  at the second bound such a gap", "  at the single-message search bound"):
            self.assertIn(line + "\n", text)
        self.assertEqual(silence.up(0.39943097), "0.39944 msg/s")
        self.assertEqual(silence.most(0.5), "at most 0.50000 msg/s at 95 %")
        self.assertEqual(silence.nominal(-math.log(0.05) / 30), "0.09986 msg/s at a nominal 95 %")

    def test_the_other_transcripts(self):
        half = -math.log(0.05) / 30
        self.assertIn(f"  at most {up(half, 5)} msg/s at 95 %\n", expected("ch16_trace_measurements"))
        # Listing 23.1: no arrival in 10 s and in 100 s, and the bound on the change.
        self.assertEqual(case.arrival(0, 10.0), "at most 0.29958 req/s at 95 %")         # Version 58: 0.29957
        self.assertLess(0.29957, -math.log(0.05) / 10)
        self.assertEqual(case.arrival(0), f"at most {up(-math.log(0.05) / 100, 5)} req/s at 95 %")
        rng = random.Random(23)
        before, after = case.period(rng), case.period(rng)
        upper = case.upper_bound(case.handler_change(before, after))
        self.assertEqual((f"{float(upper):.1f}", up(upper, 1)), ("4.7", "4.8"))            # a fraction since Version 62
        self.assertIn("upper bound, 95 %: 4.8 ms\n", expected("ch23_case"))               # Version 58: 4.7
        self.assertIs(case.within(upper, 5.0), True)                                      # decided on the bound itself

    def test_chapters_9_11_and_18(self):
        # Listing 9.2: the lower bound for the percentile is an order statistic of 500 values.
        calls, kept = [], tail.quantile_interval

        def spy(xs, q, p=0.95):
            calls.append(kept(xs, q, p))
            return calls[-1]
        tail.quantile_interval = spy
        try:
            text = printed(tail.main)
        finally:
            tail.quantile_interval = kept
        low = calls[-1][0]
        self.assertEqual((f"{low:.1f}", down(low, 1)), ("113.2", "113.1"))
        self.assertIn("  lower bound 113.1 ms, upper bound none\n", text)                # Version 58: 113.2
        self.assertEqual(text, expected("ch09_dependent_tail"))
        # Chapter 11: the upper limit of a slope.
        ts, ms = leak.readings(240, random.Random(11))
        rows = expected("ch11_leak_rate").splitlines()[1:]
        for row, w in zip(rows, (20, 60, 120, 240)):
            fit = stats.linfit(ts[:w + 1], ms[:w + 1])
            upper = fit[1] + 1.645 * fit[3]
            self.assertEqual(row.split()[4], up(upper, 4), w)
            self.assertGreaterEqual(Fraction(row.split()[4]), Fraction(upper))
        # Chapter 18: one-sided ends of a temperature.
        self.assertEqual(cal.worded((0.0, 12.190118245514029)), "T up to 12.20")          # Version 58: 12.19
        self.assertEqual(cal.worded((64.76349379652108, math.inf)), "T from 64.76, no upper end")
        # Version 58 to 62 left an interval with two ends rounded to the nearest, 14.01. The review of Version
        # 62 showed a temperature inside the interval and beyond that print: both ends are now outward.
        self.assertEqual(cal.worded((4.830164851560481, 14.013935816351987)), "4.83 to 14.02")
        self.assertIn("  100 positive: T up to 12.20\n", expected("ch18_calibration"))


class TheReferenceAndTheRecord(unittest.TestCase):
    REC = {"fast": [100] * 18, "slow": [500, 500]}

    def test_a_reference_mix_is_counted_once(self):
        for form in (list, tuple, iter, lambda xs: (x for x in xs)):
            ref = {"fast": form(range(18)), "slow": form(range(2))}
            self.assertEqual(len(mix.weighted(self.REC, ref)), 20)              # Version 58: TypeError for iterators
            ref = {"fast": form(range(18)), "slow": form(range(2))}
            self.assertEqual(mix.within(self.REC, ref, 200), (Fraction(9, 10), Fraction(9, 10)))
        # Its entries stand for requests and are not examined.
        ref = {"fast": ["a"] * 9, "slow": [None, False, NAN]}
        low, high = mix.within(self.REC, ref, 200)
        self.assertEqual((low, high), (Fraction(9, 12), Fraction(9, 12)))
        self.assertEqual(sum(w for _, w in mix.weighted(self.REC, ref)), 12)

    def test_a_record_of_iterators_as_its_own_mix(self):
        rec = {"fast": iter([100] * 18), "slow": iter([500, 500])}
        self.assertEqual(mix.within(rec, rec, 200), (Fraction(9, 10), Fraction(9, 10)))   # Version 58: TypeError
        rec = {"fast": iter([100] * 18), "slow": iter([500, 500])}
        pairs = mix.weighted(rec, rec)
        self.assertEqual((len(pairs), mix.p95(pairs)), (20, 500))
        held, ref = mix.both(self.REC, self.REC)
        # Version 59 returned the held record as its own mix. Since Version 60 a mix is the counts of its classes.
        self.assertEqual({name: len(xs) for name, xs in ref.items()}, {"fast": 18, "slow": 2})
        self.assertIs(held["slow"], self.REC["slow"])

    def test_every_class_of_a_record_is_held_to_the_check(self):
        ref = {"fast": [1] * 20}
        rec = {"fast": [100] * 20, "odd": [False]}
        for call in (lambda: mix.weighted(rec, ref), lambda: mix.within(rec, ref, 200)):
            with self.assertRaisesRegex(ValueError, "no number: False"):
                call()
        # A class of durations that the reference does not have carries no weight in it.
        rec = {"fast": [100] * 20, "new": [900, 950]}
        self.assertEqual(len(mix.weighted(rec, ref)), 20)
        self.assertEqual(mix.within(rec, ref, 200), (1, 1))
        self.assertIn("also one without weight in a mix", " ".join(mix.classes.__doc__.split()))


class ElapsedTimes(unittest.TestCase):
    EDGES = {"bounds": [50, 100, 200]}

    def test_a_time_below_zero_is_not_added(self):
        for rows in ([(30, -5)] * 20, [(-5, 30)] * 20, [(30.0, -0.001)] * 20, [(D("-1"), D("30"))] * 20):
            with self.assertRaisesRegex(ValueError, "a time below zero"):       # Version 58: 47.5 for the first two
                case.reported(rows, dict(self.EDGES, start="accept"))
        with self.assertRaisesRegex(ValueError, "a time below zero"):
            case.reported([(30, -5)] * 20, dict(self.EDGES, start="handler"))
        # From handler entry the queueing delay is not read, whatever stands there.
        self.assertEqual(case.reported([(-5, 30)] * 20, dict(self.EDGES, start="handler")), 0 + 50 * 19 / 20)
        self.assertEqual(case.recorded([(0, 0)], {"start": "accept"}), [0])
        with self.assertRaisesRegex(ValueError, "a time below zero"):
            case.p95([100.0] * 19 + [-5])                                       # Version 58: 100.0
        self.assertEqual(case.elapsed(iter([0, 1.5, D("2")])), [0, 1.5, D("2")])

    def test_a_duration_below_zero_meets_no_objective(self):
        for bad in (-5, -0.001, D("-5"), Fraction(-1, 3), (">", -5), (">=", D("-0.1"))):
            rec = {"all": [5] * 19 + [bad]}
            for call in (lambda: mix.within(rec, rec, 0), lambda: mix.weighted(rec), lambda: mix.observed(rec["all"])):
                with self.assertRaisesRegex(ValueError, "a time below zero", msg=repr(bad)):   # Version 58: [1, 1], met
                    call()
        with self.assertRaises(ValueError):
            mix.observed([5, -math.inf])
        zero = {"all": [0] * 20}
        self.assertEqual(mix.within(zero, zero, 0), (1, 1))
        # What stands for no duration at all is not below zero, and stays.
        kept = [0, None, NAN, math.inf, (">", 0), (">=", 300)]
        self.assertIs(mix.observed(kept), kept)


class AnInfiniteLimit(unittest.TestCase):
    def test_every_duration_is_within_it(self):
        for entry in (1, (">=", 1000), (">", 5), None, NAN, math.inf, D("Infinity")):
            rec = {"all": [entry]}
            low, high = mix.within(rec, rec, math.inf)
            self.assertEqual((low, high, mix.verdict(low, high)), (1, 1, "met"), entry)   # Version 58: [0, 1] for bounds
            self.assertIs(mix.side(entry, math.inf), True)

    def test_a_finite_limit_is_read_as_before(self):
        for entry, limit, want in (((">=", 1000), 999, False), ((">=", 1000), 1000, None), ((">", 1000), 1000, False),
                                   ((">", 5), 10 ** 300, None), (None, 200, None), (math.inf, 10 ** 300, False),
                                   (150, 200, True), (250, 200, False)):
            self.assertIs(mix.side(entry, limit), want, (entry, limit))


class TheSchemaOfTheHistogram(unittest.TestCase):
    def test_a_bound_is_a_number_that_a_float_holds(self):
        for bad, why in (([1, D("1E400")], "a bound is missing or too large"), ([1, 10 ** 400], "a bound is missing or too large"),
                         ([1, 2 ** 1024], "a bound is missing or too large"), ([False, 5], "no number: False"),
                         ([True, 5], "no number: True"), ([1, Fraction(10 ** 400, 3)], "a bound is missing or too large")):
            with self.assertRaisesRegex(ValueError, why, msg=str(bad)[:30]):    # Version 58: inf, OverflowError, 2.5, 3.0
                h9.bucket_quantile([2], 0.5, bad)
        for bad in ([1, NAN], [1, math.inf], [5, 1], [], [-1, 5]):
            with self.assertRaisesRegex(ValueError, "no rising bounds >= 0"):
                h9.bucket_quantile([2], 0.5, bad)
        # The schema is refused before a value is looked at.
        with self.assertRaisesRegex(ValueError, "no number: False"):
            h9.bucket_quantile([NAN], 0.5, [False, 5])
        # The largest float is a bound: two lies halfway up the bucket from 1 to it.
        top = 2.0 ** 1023 * (2 - 2.0 ** -52)
        self.assertEqual(h9.bucket_quantile([2], 0.5, [1, top]), 1 + (top - 1) * 0.5)
        self.assertEqual(h9.bucket_quantile([2], 0.5, [1, D("250")]), 1 + 249 * 0.5)

    def test_what_is_returned_for_the_open_ended_bucket(self):
        # The last bound as it was given, in its own type: the result is a float only where a bucket is interpolated.
        for last in (250, 250.0, D("250"), Fraction(250)):
            got = h9.bucket_quantile([300] * 20, 0.95, [100, last])
            self.assertEqual((got, type(got)), (last, type(last)))
        self.assertIs(type(h9.bucket_quantile([150] * 20, 0.95, [D("100"), D("250")])), float)


class TheStepOfAnImportedRecord(unittest.TestCase):
    def test_the_sign_is_read_before_the_step_is_made_a_float(self):
        xs = [D("182.1")] * 20
        self.assertEqual(float(D("-1E-400")), 0.0)                              # which is how it passed in Version 58
        for bad in (D("-1E-400"), -1, D("-0.1"), Fraction(-1, 10 ** 400), -5e-324):
            with self.assertRaisesRegex(ValueError, "need a width >= 0"):
                budget.from_raw(xs, bad)
        for step, want in ((0, math.sqrt(2.5 ** 2 + 0.09)), (D("1E-400"), math.sqrt(2.5 ** 2 + 0.09)),
                           (D("0.1"), math.sqrt(2.5 ** 2 + 0.09 + 0.01 / 12))):
            self.assertAlmostEqual(budget.from_raw(xs, step)[1].u, want, places=12)


if __name__ == "__main__":
    unittest.main()
