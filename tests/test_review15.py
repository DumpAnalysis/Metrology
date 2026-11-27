"""Regression tests added after the review of Version 15.

The review found that a window with no messages was given a rate of exactly
zero with an uncertainty of zero, where a count of zero supports only an
upper bound; that the windows of Listing 16.1 were measured from the first
message and not from the start of the capture; that the helper for a judge
with known error rates clipped each end of an interval separately, and so
returned [0, 0] and [1, 1] for samples that no true pass rate fits; and that
the exact interval overflowed for a large sample. It also suggested a worked
case of a silent trace and an illustration for Chapter 20.

Each test computes what it expects independently of the code under test: the
Poisson probability of silence, seeded simulations of a process and of its
thinning, counts taken straight from the lines of the trace, closed forms of
the exact interval, an inversion of the exact test by enumeration over true
pass rates, a second route through the incomplete beta function, and the
binomial tail in 50-digit decimal arithmetic.
"""
import math
import random
import unittest
from decimal import Decimal, localcontext
from pathlib import Path

from examples import ch16_trace_measurements as trace
from examples import ch20_judge, ch23_case
from swmetro import Result
from swmetro.dist import betainc
from swmetro.stats import quiet_bound

ROOT = Path(__file__).resolve().parent.parent
START = (12 * 3600 + 26 * 60) * 1000          # 12:26:00.000 in ms since midnight
SPAN = (START, START + 60_000)
COUNTER = (1, 800)                            # the tracer's next sequence number at start and stop


def expected(name):
    return (ROOT / "expected" / f"{name}.txt").read_text(encoding="utf-8")


def fixture():
    text = (ROOT / "data" / "app_trace.log").read_text(encoding="utf-8")
    return [trace.parse(line) for line in text.splitlines()]


def reference(k, n, p):
    """P(X <= k) for a binomial count, in 50-digit decimal arithmetic: the terms from the first one
    by their ratios, with the float p taken as the binary number it is."""
    with localcontext() as ctx:
        ctx.prec = 50
        p = Decimal(p)
        term, odds, total = (1 - p) ** n, p / (1 - p), Decimal(0)
        for i in range(k + 1):
            total += term
            term = term * (n - i) / (i + 1) * odds
        return float(total)


def direct(k, n, p):
    """The same as the plain sum of the binomial probabilities, for small n."""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


class CountOfZero(unittest.TestCase):
    def test_silence_gives_a_bound_and_no_estimate(self):
        # The plug-in uncertainty sqrt(N) / T is 0 at N = 0; Version 15 printed 0.0(0) msg/s.
        self.assertEqual(str(Result(0 / 30, math.sqrt(0) / 30, "msg/s")), "0.0(0) msg/s")
        with self.assertRaises(ValueError):
            trace.current(0, 30)
        # The bound is the rate at which 30 s of silence still has probability 5 %.
        top = quiet_bound(30)
        self.assertAlmostEqual(math.exp(-top * 30), 0.05, places=14)
        self.assertEqual(f"{top:.5f}", "0.09986")
        # A count above zero keeps its estimate.
        self.assertEqual(str(trace.current(564, 30)), "18.80(79) msg/s")
        self.assertEqual(str(trace.current(1, 30)), "0.033(33) msg/s")

    def test_the_figures_of_the_review(self):
        for seconds, want in ((30, "0.09986"), (60, "0.04993"), (40, "0.07489"), (10, "0.29957"), (100, "0.02996")):
            self.assertEqual(f"{-math.log(0.05) / seconds:.5f}", want)
            self.assertEqual(f"{quiet_bound(seconds):.5f}", want, seconds)

    def test_the_bound_is_one_sided_at_its_level(self):
        # Any rate above the bound makes the silence less probable than 1 - level, any rate below more.
        for level in (0.5, 0.9, 0.95, 0.99):
            for exposure in (0.25, 10, 60, 3600):
                top = quiet_bound(exposure, level)
                self.assertAlmostEqual(math.exp(-top * exposure), 1 - level, places=12)
                self.assertLess(math.exp(-1.01 * top * exposure), 1 - level)
                self.assertGreater(math.exp(-0.99 * top * exposure), 1 - level)
        self.assertAlmostEqual(quiet_bound(30, 0.9), math.log(10) / 30, places=14)

    def test_a_simulated_process_is_silent_that_often(self):
        # Events at the bounding rate: the first one falls beyond the window in 5 % of the windows.
        rng = random.Random(16)
        top = quiet_bound(30)
        silent = sum(rng.expovariate(top) > 30 for _ in range(40_000)) / 40_000
        self.assertAlmostEqual(silent, 0.05, delta=0.004)

    def test_exposure_is_live_time_multiplied_by_retention(self):
        # A source of constant rate, a recorder live for 40 s of 60 s at fixed times, each message
        # kept with probability 1/4: silence has the probability of 10 s of full observation.
        rng = random.Random(1609)
        rate, trials, silent = 0.2, 40_000, 0
        for _ in range(trials):
            t, kept = rng.expovariate(rate), 0
            while t < 60 and not kept:
                kept = (t < 25 or t >= 45) and rng.random() < 0.25
                t += rng.expovariate(rate)
            silent += not kept
        self.assertAlmostEqual(silent / trials, math.exp(-rate * 40 * 0.25), delta=0.008)
        for live, keep, want in ((60, 1.0, "0.04993"), (40, 1.0, "0.07489"), (40, 0.25, "0.29957")):
            self.assertEqual(f"{quiet_bound(live * keep):.5f}", want)

    def test_what_is_refused(self):
        for exposure in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                quiet_bound(exposure)
        for level in (0, 1, -0.5, 1.5):
            with self.assertRaises(ValueError):
                quiet_bound(30, level)


class ExplicitStart(unittest.TestCase):
    # The capture of the review: 60 s, heartbeats at 10 s and 55 s, one database message at 35 s.
    ROWS = [(1, 10_000, "pool", "heartbeat"), (2, 35_000, "db", "query 1 done"), (3, 55_000, "pool", "heartbeat")]

    def test_the_capture_of_the_review(self):
        span = (0, 60_000)
        halves = [trace.count(self.ROWS, "db", lo, hi, span) for lo, hi in ((0, 30), (30, 60))]
        self.assertEqual(halves, [0, 1])
        # Measured from the first message, as Version 15 did, the message moves into the first half.
        first = self.ROWS[0][1]
        moved = [sum(c == "db" and lo <= (ms - first) / 1000 < hi for _, ms, c, _ in self.ROWS)
                 for lo, hi in ((0, 30), (30, 60))]
        self.assertEqual(moved, [1, 0])

    def test_the_fixture_gives_the_same_counts_from_either_origin(self):
        rows = fixture()
        self.assertEqual(rows[0][1] - START, 22)
        self.assertEqual(tuple(trace.stamp(x) for x in trace.CAPTURE), SPAN)
        lines = (ROOT / "data" / "app_trace.log").read_text(encoding="utf-8").splitlines()
        for lo, hi, want in ((0, 30, 564), (30, 60, 127)):
            # Straight from the text of the lines: the seconds field, and the component.
            direct = sum(lo <= float(x.split()[2].split(":")[2]) < hi and x.split()[3] == "db" for x in lines)
            self.assertEqual(direct, want)
            self.assertEqual(trace.count(rows, "db", lo, hi, SPAN), want)
        self.assertEqual(trace.count(rows, "svc", 0, 30, SPAN), 2)
        self.assertEqual(trace.count(rows, "svc", 30, 60, SPAN), 0)

    def test_a_record_without_rows_is_a_quiet_window(self):
        # Since Version 17 the span alone verifies nothing: the counter must not have moved.
        self.assertEqual(trace.count([], "db", 0, 60, SPAN), 0)
        self.assertEqual(trace.verified([], 0, 60, SPAN, (1, 1)), 60.0)
        self.assertEqual(f"{quiet_bound(trace.verified([], 0, 60, SPAN, (1, 1))):.5f}", "0.04993")

    def test_a_window_that_leaves_the_capture_is_refused(self):
        for lo, hi in ((-1, 30), (30, 61), (0, 60.001), (30, 30), (40, 20), (60, 90)):
            with self.assertRaises(ValueError):
                trace.count([], "db", lo, hi, SPAN)
            with self.assertRaises(ValueError):
                trace.verified([], lo, hi, SPAN)
        self.assertEqual(trace.window(0, 60, SPAN), SPAN)

    def test_verified_time_leaves_out_the_stretch_of_a_loss(self):
        rows = fixture()
        # One loss: sequence numbers 431 to 470 are missing, between 20.099 s and 22.187 s.
        jumps = [(a, b) for a, b in zip(rows, rows[1:]) if b[0] - a[0] > 1]
        self.assertEqual(len(jumps), 1)
        (before, after), = jumps
        self.assertEqual((before[0], after[0], after[1] - before[1]), (430, 471, 2088))
        self.assertEqual(trace.COUNTER, COUNTER)
        self.assertEqual(trace.verified(rows, 0, 30, SPAN, COUNTER), 30 - 2.088)
        self.assertEqual(trace.verified(rows, 30, 60, SPAN, COUNTER), 30.0)
        self.assertEqual(trace.verified(rows, 0, 60, SPAN, COUNTER), 60 - 2.088)
        # A stretch that crosses the edge of a window counts as far as it lies inside.
        self.assertAlmostEqual(trace.verified(rows, 21, 30, SPAN, COUNTER), 9 - 1.187, places=9)
        self.assertAlmostEqual(trace.verified(rows, 0, 21, SPAN, COUNTER), 21 - 0.901, places=9)
        self.assertEqual(trace.verified(rows, 20.5, 21.5, SPAN, COUNTER), 0.0)
        self.assertEqual(trace.verified(rows, 40, 50, SPAN, COUNTER), 10.0)

    def test_the_transcript(self):
        text = expected("ch16_trace_measurements")
        # Since Version 17 the bound names its model and its level, and the rows on a silent
        # capture are in the transcript of Listing 16.4.
        for line in ("db current,  0-30 s: 18.80(79) msg/s", "db current, 30-60 s: 4.23(38) msg/s",
                     "svc, 30-60 s: 0 messages, 30 s verified", "  if messages arrive singly:",
                     "  at most 0.09986 msg/s at 95 %", "verified time, 0-30 s: 27.912 s",
                     "  messages were db: 20.13(82) msg/s"):
            self.assertIn(line + "\n", text)
        self.assertNotIn("0.0(0)", text)
        text = expected("ch16_silence")
        for line in ("  60 s, 100 %: 60 s,\n    at most 0.04993 msg/s at 95 %", "  40 s, 100 %: 40 s,\n    at most 0.07490 msg/s at 95 %",
                     "  40 s,  25 %: 10 s,\n    at most 0.29958 msg/s at 95 %"):
            self.assertIn(line + "\n", text)


class QuietCase(unittest.TestCase):
    def test_an_empty_window_bounds_the_arrival_rate(self):
        self.assertEqual(ch23_case.WINDOW, 100.0)
        self.assertEqual(ch23_case.arrival(0), "at most 0.02996 req/s at 95 %")
        self.assertEqual(ch23_case.arrival(0), f"at most {-math.log(0.05) / 100:.5f} req/s at 95 %")
        self.assertEqual(ch23_case.arrival(1905), "19.05(44) req/s")
        text = expected("ch23_case")
        for line in ("empty window: no observations -> no observations", "  its arrival rate at most 0.02996 req/s at 95 %",
                     "arrival rate before 19.05(44) req/s", "arrival rate after  19.55(44) req/s"):
            self.assertIn(line + "\n", text)


class JudgeModel(unittest.TestCase):
    SENS, SPEC = 0.90, 0.85

    def fits(self, k, n, theta):
        """True if the exact two-sided test at 5 % keeps this true pass rate."""
        q = 0.15 + 0.75 * theta
        at_or_below = direct(k, n, q)
        at_or_above = 1 - direct(k - 1, n, q) if k else 1.0
        return min(at_or_below, at_or_above) >= 0.025

    def test_the_three_samples_of_the_review(self):
        self.assertEqual((ch20_judge.SENS, ch20_judge.SPEC), (self.SENS, self.SPEC))
        rows = ((0, 25, ("0.00", "13.72"), None), (2, 20, ("1.23", "31.70"), ("0.00", "22.26")),
                (40, 40, ("91.19", "100.00"), None))
        for k, n, seen_want, ends_want in rows:
            y, seen, ends = ch20_judge.few_outputs(k, n, self.SENS, self.SPEC)
            self.assertAlmostEqual(y, (k / n - 0.15) / 0.75, places=12)
            self.assertEqual(tuple(f"{100 * x:.2f}" for x in seen), seen_want, (k, n))
            if ends_want is None:
                self.assertIsNone(ends, (k, n))
            else:
                self.assertEqual(tuple(f"{100 * x:.2f}" for x in ends), ends_want)
        # Closed forms of the two ends that decide.
        self.assertAlmostEqual(ch20_judge.exact_interval(0, 25)[1], 1 - 0.025 ** (1 / 25), places=12)
        self.assertAlmostEqual(ch20_judge.exact_interval(40, 40)[0], 0.025 ** (1 / 40), places=12)
        self.assertLess(1 - 0.025 ** (1 / 25), 0.15)
        self.assertGreater(0.025 ** (1 / 40), 0.90)
        # What Version 15 returned for the outer two: each end clipped by itself.
        for k, n, clipped in ((0, 25, [0.0, 0.0]), (40, 40, [1.0, 1.0])):
            ends = [ch20_judge.feasible(ch20_judge.corrected([p, self.SENS, self.SPEC]))[0]
                    for p in ch20_judge.exact_interval(k, n)]
            self.assertEqual(ends, clipped)

    def test_no_overlap_means_that_no_true_rate_fits(self):
        grid = [i / 200 for i in range(201)]
        self.assertFalse(any(self.fits(0, 25, theta) for theta in grid))
        self.assertFalse(any(self.fits(40, 40, theta) for theta in grid))
        self.assertTrue(self.fits(2, 20, 0.04))
        # Such samples are unusual and not impossible: at the nearest rate the model allows.
        self.assertEqual(f"{0.85 ** 25:.4f}", "0.0172")
        self.assertEqual(f"{0.90 ** 40:.4f}", "0.0148")
        self.assertAlmostEqual(ch20_judge.at_most(0, 25, 0.15), 0.85 ** 25, places=14)

    def test_the_interval_is_the_set_of_rates_that_fit(self):
        # The rates of a fine grid that the exact test keeps, against the ends that the listing returns.
        grid = [i / 400 for i in range(401)]
        for k, n in ((2, 20), (5, 30), (12, 25), (30, 40), (39, 40), (0, 22), (0, 23), (1, 25)):
            kept = [theta for theta in grid if self.fits(k, n, theta)]
            ends = ch20_judge.few_outputs(k, n, self.SENS, self.SPEC)[2]
            if ends is None:
                self.assertEqual(kept, [], (k, n))
                continue
            self.assertTrue(kept, (k, n))
            self.assertTrue(kept[0] - 1 / 400 <= ends[0] <= kept[0] + 1e-9, (k, n, ends))
            self.assertTrue(kept[-1] - 1e-9 <= ends[1] <= kept[-1] + 1 / 400, (k, n, ends))
            # No rate between the ends is rejected: the set is an interval.
            self.assertEqual(len(kept), round(400 * (kept[-1] - kept[0])) + 1, (k, n))
        # The band is reached or missed by a hair between 22 and 23 outputs without a pass.
        self.assertGreater(1 - 0.025 ** (1 / 22), 0.15)
        self.assertLess(1 - 0.025 ** (1 / 23), 0.15)
        self.assertIsNotNone(ch20_judge.few_outputs(0, 22, self.SENS, self.SPEC)[2])
        self.assertIsNone(ch20_judge.few_outputs(0, 23, self.SENS, self.SPEC)[2])

    def test_a_large_sample_does_not_overflow(self):
        # The direct sum of Version 15 fails on its first term.
        with self.assertRaises(OverflowError):
            math.comb(2000, 1000) * 0.5 ** 1000
        low, high = ch20_judge.exact_interval(1000, 2000)
        self.assertEqual((f"{100 * low:.3f}", f"{100 * high:.3f}"), ("47.785", "52.215"))
        self.assertAlmostEqual(low + high, 1.0, places=12)          # the sample is symmetric
        # The ends in 50-digit arithmetic: each tail is 2.5 % there.
        self.assertAlmostEqual(1 - reference(999, 2000, low), 0.025, places=12)
        self.assertAlmostEqual(reference(1000, 2000, high), 0.025, places=12)
        # And through the incomplete beta function: P(X >= k) = I_p(k, n - k + 1).
        self.assertAlmostEqual(betainc(1000, 1001, low), 0.025, places=10)
        self.assertAlmostEqual(betainc(1001, 1000, high), 0.975, places=10)

    def test_the_tail_from_ratios_is_the_binomial_sum(self):
        for k, n, p in ((2, 20, 0.18), (0, 25, 0.15), (7, 20, 0.5), (19, 20, 0.95), (3, 7, 1e-9), (150, 400, 0.4)):
            self.assertAlmostEqual(ch20_judge.at_most(k, n, p), direct(k, n, p), places=14, msg=(k, n, p))
        for k, n, p in ((1000, 2000, 0.49), (900, 2000, 0.47), (24_900, 50_000, 0.5), (10, 50_000, 0.0002),
                        (2, 1_000_000, 3e-6), (640, 2000, 0.37)):
            want = reference(k, n, p)
            self.assertAlmostEqual(ch20_judge.at_most(k, n, p) / want, 1.0, places=12, msg=(k, n))
        # A count and its complement share the whole probability. (With p = 1/2 or 1/4 the
        # complement 1 - p is exact as a float; with other p the sum would test the rounding of 1 - p.)
        for k, n, p in ((1000, 2000, 0.5), (24_900, 50_000, 0.5), (480, 2000, 0.25), (12_700, 50_000, 0.25)):
            both = ch20_judge.at_most(k, n, p) + ch20_judge.at_most(n - k - 1, n, 1 - p)
            self.assertAlmostEqual(both, 1.0, places=13, msg=(k, n))
        self.assertEqual((ch20_judge.at_most(3, 20, 0.0), ch20_judge.at_most(3, 20, 1.0)), (1.0, 0.0))
        self.assertEqual((ch20_judge.at_most(20, 20, 1.0), ch20_judge.at_most(25, 20, 0.3)), (1.0, 1.0))

    def test_the_transcript(self):
        text = expected("ch20_judge")
        for line in ("2 of 20 judged passes:", "  95 % interval 0.000 to 0.223", "0 of 25 judged passes:",
                     "  corrected -0.2000, below 0", "  judged rate 0.000 to 0.138,", "40 of 40 judged passes:",
                     "  corrected +1.1333, above 1", "  judged rate 0.911 to 1.000,", "  no overlap with the judge",
                     "  model, 0.15 to 0.90, at 95 %"):
            # Until Version 69 the ends of an interval were printed to the nearest digit. Since Version 70 the
            # lower end is rounded downward and the upper one upward, so that the printed interval holds the
            # computed one (tests/test_review69.py). The two read 0.000 to 0.137 and 0.912 to 1.000.
            self.assertIn(line + "\n", text)
        self.assertEqual(text.count("  no overlap with the judge\n"), 2)

    def test_the_figure_is_drawn_from_the_listing(self):
        source = (ROOT / "figures" / "make_figures.py").read_text(encoding="utf-8")
        for piece in ("from examples import ch20_judge as c20", "c20.exact_interval(k, n)", "1 - c20.SPEC, c20.SENS",
                      'save(fig, "fig-20-1-judge-model.png")', "samples = ((40, 40), (2, 20), (0, 25))"):
            self.assertIn(piece, source)
        self.assertEqual(source.count("\n    save(fig, "), 32)         # seven until Version 16


if __name__ == "__main__":
    unittest.main()
