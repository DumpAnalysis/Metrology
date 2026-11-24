"""Regression tests added after the review of Version 17.

The review found that the bound for messages that come in pairs took the
window plus the duration of a request as its exposure for every duration,
which is too much when a request is longer than the window and can span it
without leaving a message; that the bounds for a silence were printed
whatever the count; that time stamps running backwards made verified() count
more time than the capture holds; and that the comparison at a reference mix
described one class alone, or failed, when a class of the mix had no
observations.

Each test computes what it expects independently of the code under test:
start times on a grid that are tried one by one, seeded simulations of the
requests, the lines of the trace themselves, means from the class means and
their shares, and every number of missing requests that could have met a
threshold.
"""
import contextlib
import io
import math
import random
import unittest
from fractions import Fraction
from pathlib import Path

from examples import ch16_silence as silence
from examples import ch16_trace_measurements as trace
from examples import ch23_mix as mix
from swmetro.stats import quiet_bound, rate_upper

ROOT = Path(__file__).resolve().parent.parent
START = (12 * 3600 + 26 * 60) * 1000
SPAN = (START, START + 60_000)
COUNTER = (1, 800)


def expected(name):
    return (ROOT / "expected" / f"{name}.txt").read_text(encoding="utf-8")


def fixture():
    text = (ROOT / "data" / "app_trace.log").read_text(encoding="utf-8")
    return [trace.parse(line) for line in text.splitlines()]


def printed(function, *args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        function(*args)
    return out.getvalue()


def seen(start, stretches, delta):
    """True if a request that starts here leaves its start or its end in one of the stretches."""
    return any(x <= t < y for t in (start, start + delta) for x, y in stretches)


def covered(stretches, delta, step=0.001):
    """Length of the start times that leave a message, by trying them one by one."""
    low = min(x for x, _ in stretches) - delta - 1
    high = max(y for _, y in stretches) + 1
    count = sum(seen(low + step * i, stretches, delta) for i in range(int((high - low) / step)))
    return count * step


def periods():
    rng = random.Random(23)
    return mix.record(mix.BEFORE, rng), mix.record(mix.AFTER, rng)


class LongRequests(unittest.TestCase):
    def test_the_request_of_the_review(self):
        whole = [(0, 30)]
        # A request that starts 15 s before the window and ends 45 s after its start leaves neither.
        self.assertFalse(seen(-15, whole, 60))
        self.assertTrue(seen(-45, whole, 60) and seen(10, whole, 60))
        self.assertEqual(silence.starts(whole, 60), [(-60, -30), (0, 30)])
        self.assertEqual(silence.reach(whole, 60), 60)
        self.assertEqual(f"{silence.paired_bound(whole, 60):.5f}", "0.09986")
        self.assertEqual(f"{2 * math.log(20) / 60:.5f}", "0.09986")
        # Version 17 took 30 + 60 s. At its bound the window is silent 13.57 % of the time.
        old = 2 * quiet_bound(30 + 60)
        self.assertEqual(f"{old:.5f}", "0.06657")
        self.assertEqual(f"{100 * math.exp(-old / 2 * 60):.2f}", "13.57")
        # A request as long as the window or longer shows each message as if it came alone.
        for delta in (30, 31, 60, 1e6):
            self.assertAlmostEqual(silence.paired_bound(whole, delta), quiet_bound(30), places=15)

    def test_the_union_for_one_window(self):
        for window in (30, 7.5, 100):
            for delta in (0, 0.5, 2.346, window / 2, window, 1.5 * window, 60, 250):
                stretches = [(0, window)]
                self.assertAlmostEqual(silence.reach(stretches, delta), window + min(window, delta), places=9)
                if window + delta < 200:
                    self.assertAlmostEqual(covered(stretches, delta), window + min(window, delta), delta=0.01)
        # The short request of the fixture is unchanged: 32.346 s, and 0.18523 msg/s.
        self.assertAlmostEqual(silence.reach([(0, 30)], 2.346), 32.346, places=12)
        self.assertEqual(f"{silence.paired_bound([(0, 30)], 2.346):.5f}", "0.18523")

    def test_a_simulated_long_request_process(self):
        rng = random.Random(1718)
        for rate, want in ((silence.paired_bound([(0, 30)], 60), 0.05), (2 * quiet_bound(90), 0.1357)):
            trials, silent = 30_000, 0
            for _ in range(trials):
                t, hit = -150 + rng.expovariate(rate / 2), False       # requests from 150 s before the window
                while t < 30 and not hit:
                    hit = seen(t, [(0, 30)], 60)
                    t += rng.expovariate(rate / 2)
                silent += not hit
            self.assertAlmostEqual(silent / trials, want, delta=0.008, msg=rate)

    def test_stretches_keep_their_spacing(self):
        # The first half minute of the fixture: a gap of 2.088 s, shorter than the request.
        found = trace.stretches(fixture(), 0, 30, SPAN, COUNTER)
        self.assertEqual(found, [(0, 20_099), (22_187, 30_000)])
        live = [(x / 1000, y / 1000) for x, y in found]
        self.assertAlmostEqual(sum(y - x for x, y in live), 27.912, places=9)
        # No request can hide in that gap: one of its two messages lies outside it.
        self.assertAlmostEqual(silence.reach(live, 2.346), 32.346, places=9)
        self.assertAlmostEqual(covered(live, 2.346), 32.346, delta=0.01)
        self.assertNotAlmostEqual(silence.reach(live, 2.346), 27.912 + 2.346, places=3)
        # Gaps longer than the request, and requests longer than the gaps.
        for stretches, delta in (([(0, 10), (20, 30)], 2.346), ([(0, 10), (20, 30)], 15), ([(0, 5), (6, 7), (40, 60)], 3),
                                 ([(0, 10), (20, 30)], 10), ([(0, 10), (20, 30)], 20), ([(0, 10), (20, 30)], 45)):
            self.assertAlmostEqual(silence.reach(stretches, delta), covered(stretches, delta), delta=0.02, msg=(stretches, delta))
        self.assertAlmostEqual(silence.reach([(0, 10), (20, 30)], 2.346), 20 + 2 * 2.346, places=9)
        self.assertEqual(silence.starts([(0, 10), (20, 30)], 15), [(-15, -5), (0, 15), (20, 30)])
        self.assertEqual(silence.starts([(0, 10), (20, 30)], 20), [(-20, -10), (0, 10), (20, 30)])
        self.assertEqual(silence.starts([(0, 10), (20, 30)], 10), [(-10, 30)])        # the copies fill the gaps

    def test_what_is_refused(self):
        with self.assertRaises(ValueError):
            silence.paired_bound([(0, 30)], -1)
        for stretches in ([(30, 30)], [(5, 0)], [(0, 30), (40, 40)]):
            with self.assertRaises(ValueError):
                silence.paired_bound(stretches, 2)
        with self.assertRaises(ValueError):
            silence.paired_bound([], 2)                      # no stretch, no exposure

    def test_the_transcript(self):
        text = expected("ch16_silence")
        for line in ("    starts covered: 32.346 s\n    at most 0.18523 msg/s at 95 %", "requests of 60 s, window of 30 s:",
                     "  starts covered: 60 s\n    at most 0.09986 msg/s at 95 %", "  0 to 20.099 and 22.187 to 30 s:",
                     "  27.912 s, and for pairs 2.346 s", "  apart, starts covered: 32.346 s", "    silent at 0.09986 msg/s: 19.9 %"):
            self.assertIn(line + "\n", text)

    def test_the_figure_has_both_requests(self):
        source = (ROOT / "figures" / "make_figures.py").read_text(encoding="utf-8")
        for piece in ("from examples import ch16_silence as c16s", "for ax, d in zip(axes, (c16.time_delta(rows).value / 1000, 60.0)):",
                      '("either", c16s.starts(whole, d))', "mfc=BLACK", 'mfc="white"'):
            self.assertIn(piece, source)
        self.assertEqual(source.count("\n    save(fig, "), 32)


class TimingEvidence(unittest.TestCase):
    @staticmethod
    def lines(*seconds):
        return [(i + 1, 1000 * t, "x", "") for i, t in enumerate(seconds)]

    def test_a_backward_step_of_the_clock_verifies_nothing(self):
        # Version 17 added the spans from 1 to 26 s and from 6 to 29 s: 48 s of a capture of 30 s.
        back = self.lines(1, 26, 6, 29)
        self.assertEqual((26 - 1) + (29 - 6), 48)
        self.assertEqual(trace.stretches(back, 0, 30, (0, 30_000)), [])
        self.assertEqual(trace.verified(back, 0, 30, (0, 30_000)), 0.0)
        self.assertEqual(trace.verified(back, 0, 30, (0, 30_000), (1, 5)), 0.0)
        # The same four lines in the order of their stamps are evidence from the first to the last.
        self.assertEqual(trace.stretches(self.lines(1, 6, 26, 29), 0, 30, (0, 30_000)), [(1000, 29_000)])
        self.assertEqual(trace.verified(self.lines(1, 6, 26, 29), 0, 30, (0, 30_000)), 28.0)
        self.assertEqual(trace.verified(self.lines(1, 6, 26, 29), 0, 30, (0, 30_000), (1, 5)), 30.0)
        # Equal stamps, which truncation produces, are compatible.
        self.assertEqual(trace.verified(self.lines(1, 6, 6, 29), 0, 30, (0, 30_000)), 28.0)
        # A line stamped outside the capture contradicts the readings taken at its ends.
        self.assertEqual(trace.verified(self.lines(1, 6, 26, 31), 0, 30, (0, 30_000), (1, 5)), 0.0)

    def test_verified_time_never_exceeds_the_window(self):
        rng = random.Random(17)
        for _ in range(400):
            stamps = [rng.uniform(0, 30) for _ in range(rng.randrange(0, 9))]
            if rng.random() < 0.5:
                stamps.sort()
            rows, seq = [], 0
            for t in stamps:
                seq += rng.choice((1, 1, 1, 3))
                rows.append((seq, round(1000 * t), "x", ""))
            counter = rng.choice((None, (1, seq + 1), (1, seq + 4)))
            lo = rng.choice((0, 5, 12))
            hi = rng.choice((20, 30))
            got = trace.verified(rows, lo, hi, (0, 30_000), counter)
            self.assertTrue(0 <= got <= hi - lo, (rows, counter, lo, hi, got))
            found = trace.stretches(rows, lo, hi, (0, 30_000), counter)
            self.assertTrue(all(x < y for x, y in found))
            self.assertTrue(all(a[1] < b[0] for a, b in zip(found, found[1:])), found)        # disjoint, in order

    def test_the_fixture_is_verified_as_before(self):
        rows = fixture()
        self.assertTrue(all(a[1] <= b[1] for a, b in zip(rows, rows[1:])))
        self.assertEqual(trace.stretches(rows, 30, 60, SPAN, COUNTER), [(30_000, 60_000)])
        self.assertEqual(trace.verified(rows, 0, 30, SPAN, COUNTER), 27.912)
        self.assertEqual(trace.verified(rows, 30, 60, SPAN, COUNTER), 30.0)
        text = expected("ch16_silence")
        self.assertIn("four consecutive lines stamped 1, 26,\n6 and 29 s: exposure unknown, no bound\n", text)
        self.assertEqual(expected("ch16_trace_measurements"),
                         (ROOT / "expected" / "ch16_trace_measurements.txt").read_text(encoding="utf-8"))


class GuardedBounds(unittest.TestCase):
    def test_one_message_is_not_a_silence(self):
        # One heartbeat of the second half minute turned into a service message.
        rows, changed, done = fixture(), [], False
        for seq, ms, comp, text in rows:
            if not done and comp == "pool" and ms - START >= 30_000:
                comp, text, done = "svc", "request 18 start", True
            changed.append((seq, ms, comp, text))
        self.assertTrue(done)
        out = printed(silence.models, changed, SPAN)
        self.assertIn("svc, 30-60 s: 1 message, 30 s verified\n", out)
        self.assertIn("    at most 0.15813 msg/s at 95 %\n", out)
        self.assertEqual(f"{rate_upper(1, 30):.5f}", "0.15813")
        # Since Version 19 the bound for pairs with one message is computed (Version 18 said that
        # it needed a calculation of its own).
        self.assertIn("    at most 0.21000 msg/s at 95 %\n", out)
        for wrong in ("0.09986", "0.18523", "silent at"):       # Version 17 printed the bounds of a silence
            self.assertNotIn(wrong, out)

    def test_without_verified_time_there_is_no_bound(self):
        rows = fixture()
        moved = [(seq, ms - 5000 if seq == 700 else ms, comp, text) for seq, ms, comp, text in rows]
        out = printed(silence.models, moved, SPAN)
        self.assertEqual(out, "svc, 30-60 s: 0 messages, 0 s verified\n  exposure unknown, no bound\n")

    def test_the_silent_half_minute_is_as_printed(self):
        out = printed(silence.models, fixture(), SPAN)
        self.assertTrue(expected("ch16_silence").startswith(out))
        for line in ("svc, 30-60 s: 0 messages, 30 s verified", "    at most 0.09986 msg/s at 95 %", "    at most 0.18523 msg/s at 95 %"):
            self.assertIn(line + "\n", out)


class MissingClass(unittest.TestCase):
    def test_the_printed_comparison_is_unchanged(self):
        before, after = periods()
        full = mix.weighted(after, before)
        self.assertEqual((round(float(mix.mean(mix.weighted(before)))), round(float(mix.mean(full))), round(mix.p95(full))),
                         (140, 126, 432))
        self.assertEqual(mix.absent(after, before), {})
        text = expected("ch23_mix")
        for line in ("mean, as observed     140 ->  414", "mean, after, at the mix of before   126",
                     "p95, after, at the mix of before   432"):
            self.assertIn(line + "\n", text)

    def test_a_class_without_observations_is_named(self):
        before, after = periods()
        for part in ({"fast": after["fast"]}, {"fast": after["fast"], "slow": []}):
            self.assertEqual(mix.absent(part, before), {"slow": Fraction(1, 10)})
            with self.assertRaises(ValueError) as caught:              # Version 17: fast alone, or ZeroDivisionError
                mix.weighted(part, before)
            self.assertEqual(str(caught.exception), "no observations for slow")
            alone = mix.weighted(part, before, partial=True)
            self.assertEqual(sum(w for _, w in alone), 900)            # nine tenths of the reference
            self.assertAlmostEqual(float(mix.mean(alone)), 90.0, places=9)
            self.assertEqual(f"{mix.p95(alone):.2f}", "150.45")
        # A class without weight in the reference is not missed.
        self.assertEqual(mix.absent({"fast": after["fast"]}, {"fast": before["fast"], "slow": []}), {})
        # Without a reference mix nothing is reweighted and nothing can be absent.
        self.assertEqual(len(mix.weighted({"fast": after["fast"]})), 100)

    def test_two_completions_give_opposite_conclusions(self):
        before, after = periods()
        part = {"fast": after["fast"]}
        self.assertAlmostEqual(sum(after["fast"]) / 100, 90.0, places=9)
        self.assertEqual(f"{max(after['fast']):.2f}", "216.25")
        for slow, mean, change, p95 in ((450.0, 126, -14, 450), (900.0, 171, 31, 900)):
            full = mix.weighted(dict(part, slow=[slow] * 900), before)
            self.assertAlmostEqual(float(mix.mean(full)), 0.9 * 90 + 0.1 * slow, places=9)       # from the class means
            self.assertEqual((round(float(mix.mean(full))), round(float(mix.mean(full))) - 140, mix.p95(full)), (mean, change, p95))
        # A kept class mean of 450 ms fixes the mean and leaves the percentile open.
        half = mix.weighted(dict(part, slow=[300.0] * 450 + [600.0] * 450), before)
        self.assertAlmostEqual(float(mix.mean(half)), 126.0, places=9)
        self.assertEqual(mix.p95(half), 300.0)
        # The fast class fills the first 90 % of the ranks, and the lower half of the slow class
        # brings the cumulative share to exactly 95 %.
        self.assertEqual(Fraction(9, 10) + Fraction(1, 10) * Fraction(1, 2), Fraction(95, 100))

    def test_bounds_on_a_share_decide_or_do_not(self):
        before, after = periods()
        part = {"fast": after["fast"]}
        self.assertEqual(sum(x <= 200 for x in after["fast"]), 99)
        low, high = mix.within(part, before, 200)
        self.assertEqual((low, high), (Fraction(891, 1000), Fraction(991, 1000)))
        self.assertEqual(mix.verdict(low, high), "unresolved")
        # Every number of the 100 slow requests of the reference that could have been within 200 ms.
        outcomes = {Fraction(891 + k, 1000) >= Fraction(95, 100) for k in range(101)}
        self.assertEqual(outcomes, {True, False})
        # With complete records the two bounds meet, and the share is that of the reweighted sample.
        low, high = mix.within(after, before, 432)
        self.assertEqual(low, high)
        pairs = mix.weighted(after, before)
        self.assertEqual(low, sum(w for x, w in pairs if x <= 432) / sum(w for _, w in pairs))
        self.assertEqual(mix.verdict(low, high), "met" if mix.p95(pairs) <= 432 else "missed")
        # The rule can settle a decision: 98 % known within and 2 % missing; and it can settle it the other way.
        self.assertEqual(mix.verdict(Fraction(98, 100), Fraction(1)), "met")
        self.assertEqual(mix.verdict(Fraction(50, 100), Fraction(60, 100)), "missed")
        self.assertEqual(mix.verdict(Fraction(95, 100), Fraction(1)), "met")              # 95 % within is enough
        self.assertEqual(mix.verdict(Fraction(94, 100), Fraction(95, 100)), "unresolved")

    def test_the_transcript(self):
        text = expected("ch23_mix")
        # The case as the listing prints it, under every supported Python: fractions take a format
        # only from Python 3.12 on, and the listing must not rely on it.
        before, after = periods()
        self.assertIn(printed(mix.lost_class, before, after), text)
        for line in ("after, no records of the slow\nclass, 10 % of the mix of before:", "  mean and p95 at that mix:\n    not determined",
                     "  fast class alone:\n    mean 90, p95 150", "  all slow requests at 450 ms:\n    mean 126, p95 450",
                     "  all slow requests at 900 ms:\n    mean 171, p95 900", "  half at 300 ms, half at 600 ms:\n    mean 126, p95 300",
                     "  within 200 ms: 89.1 % to 99.1 %", "  p95 at most 200 ms: unresolved", "  98 % known within, 2 % missing: met"):
            self.assertIn(line + "\n", text)


if __name__ == "__main__":
    unittest.main()
