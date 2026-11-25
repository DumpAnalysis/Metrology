"""Regression tests added after the review of Version 18.

The review found that the count of service messages was taken over the whole
half minute while the exposure was the verified part of it, so that a message
recorded in a gap entered a bound for time in which it did not lie; that an
empty trace with verified time failed where the duration of a request was
looked up; and that a class which appears only after a deployment made the
comparison at the earlier mix fail, although that mix gives it no weight. It
worked the bound for one message of a paired source and the decomposition of
the change of a mean into a change within classes and a change of mix.

Each test computes what it expects independently of the code under test: the
closed form of the probability of at most one message, seeded simulations of
a paired source, start times tried one by one for the messages they
contribute, and means from the class means and their shares.
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


def messages(start, stretches, delta):
    """How many of its two messages a request that starts here leaves in the stretches."""
    return sum(x <= t < y for t in (start, start + delta) for x, y in stretches)


def simulated(rate, stretches, delta, at_most, rng, trials=30_000):
    """Share of trials in which requests at this message rate leave at most so many messages."""
    low = min(x for x, _ in stretches) - delta - 5
    high = max(y for _, y in stretches)
    few = 0
    for _ in range(trials):
        t, count = low + rng.expovariate(rate / 2), 0
        while t < high and count <= at_most:
            count += messages(t, stretches, delta)
            t += rng.expovariate(rate / 2)
        few += count <= at_most
    return few / trials


def periods():
    rng = random.Random(23)
    return mix.record(mix.BEFORE, rng), mix.record(mix.AFTER, rng)


class CountInVerifiedTime(unittest.TestCase):
    HALF = (0, 60_000)

    def test_the_two_records_of_the_review(self):
        for at, within in ((45, 0), (35, 1)):
            rows = silence.gap(at)
            found = trace.stretches(rows, 30, 60, self.HALF)
            self.assertEqual(found, [(30_000, 40_000), (50_000, 60_000)], at)
            self.assertEqual(trace.verified(rows, 30, 60, self.HALF), 20.0)
            self.assertEqual(trace.count(rows, "svc", 30, 60, self.HALF), 1)        # the record holds one message
            self.assertEqual(silence.inside(rows, "svc", trace.runs(rows, 30, 60, self.HALF)), within)
            # The same by going through the lines.
            direct = sum(c == "svc" and (30 <= ms / 1000 < 40 or 50 <= ms / 1000 < 60) for _, ms, c, _ in rows)
            self.assertEqual(direct, within)
        # Twenty verified seconds without a message, and with one.
        self.assertEqual(f"{quiet_bound(20):.5f}", "0.14979")
        self.assertEqual(f"{rate_upper(1, 20):.5f}", "0.23719")
        out45 = printed(silence.models, silence.gap(45), self.HALF, None)
        out35 = printed(silence.models, silence.gap(35), self.HALF, None)
        self.assertIn("svc, 30-60 s: 1 message, 20 s verified\n  in the verified time: 0\n", out45)
        self.assertIn("    at most 0.14979 msg/s at 95 %\n", out45)
        self.assertNotIn("0.23719", out45)                       # Version 18 printed it for both records
        self.assertIn("svc, 30-60 s: 1 message, 20 s verified\n  in the verified time: 1\n", out35)
        self.assertIn("    at most 0.23720 msg/s at 95 %\n", out35)

    def test_a_count_belongs_to_its_stretches(self):
        # Version 19 counted by time stamp and took a line stamped at an end of a stretch as part of it. The next
        # review showed that the stamp does not settle that (tests/test_review19.py): the count goes by the lines of
        # the runs, and from stamps alone it is a range that is open at the ends of the stretches.
        stretches = [(30_000, 40_000), (50_000, 60_000)]
        for second, want in ((30, (0, 1)), (39.999, (1, 1)), (40, (0, 1)), (40.001, (0, 0)), (45, (0, 0)), (49.999, (0, 0)),
                             (50, (0, 1)), (59.999, (1, 1)), (10, (0, 0))):
            rows = [(1, round(1000 * second), "svc", "x"), (2, round(1000 * second), "db", "x")]
            self.assertEqual(silence.stamped(rows, "svc", stretches, 30, 60, (0, 60_000)), want, second)   # two lines, one stamp
        # The fixture: both service messages lie in the verified part of the first half minute.
        rows = fixture()
        first = trace.runs(rows, 0, 30, SPAN, COUNTER)
        second = trace.runs(rows, 30, 60, SPAN, COUNTER)
        self.assertEqual((silence.inside(rows, "svc", first), silence.inside(rows, "svc", second)), (2, 0))
        self.assertEqual(silence.inside(rows, "db", first), trace.count(rows, "db", 0, 30, SPAN))
        self.assertEqual(silence.inside(rows, "db", first + second), 564 + 127)

    def test_an_empty_trace_with_verified_time_has_its_bound(self):
        # Version 18 printed the verified 30 s and then failed on the request that is not there.
        out = printed(silence.models, [], SPAN, (1, 1))
        self.assertEqual(out.splitlines()[:4], ["svc, 30-60 s: 0 messages, 30 s verified", "  in the verified time: 0",
                                                "  messages arriving singly:", "    at most 0.09986 msg/s at 95 %"])
        self.assertIn("no request to take the time", out)
        self.assertNotIn("starts covered", out)
        # Without verified time there is still no bound.
        self.assertEqual(printed(silence.models, [], SPAN, None), "svc, 30-60 s: 0 messages, 0 s verified\n  exposure unknown, no bound\n")

    def test_the_bound_for_single_messages_comes_first(self):
        out = printed(silence.models, fixture(), SPAN)
        lines = out.splitlines()
        self.assertLess(lines.index("    at most 0.09986 msg/s at 95 %"), lines.index("  requests arriving, each with a"))
        self.assertTrue(expected("ch16_silence").startswith(out))

    def test_the_transcript(self):
        text = expected("ch16_silence")
        for piece in ("svc, 30-60 s: 0 messages, 30 s verified\n  in the verified time: 0\n",
                      "at 45 s\nsvc, 30-60 s: 1 message, 20 s verified\n  in the verified time: 0\n  messages arriving singly:\n"
                      "    at most 0.14979 msg/s at 95 %\n",
                      "at 35 s\nsvc, 30-60 s: 1 message, 20 s verified\n  in the verified time: 1\n  messages arriving singly:\n"
                      "    at most 0.23720 msg/s at 95 %\n"):
            self.assertIn(piece, text)

    def test_the_figure_is_drawn_from_the_two_records(self):
        source = (ROOT / "figures" / "make_figures.py").read_text(encoding="utf-8")
        for piece in ("rows = c16s.gap(at)", "c16.runs(rows, 30, 60, (0, 60_000))", 'save(fig, "fig-16-3-counted-where.png")'):
            self.assertIn(piece, source)
        self.assertEqual(source.count("\n    save(fig, "), 32)


class OneMessageOfAPair(unittest.TestCase):
    def test_the_figures_of_the_review(self):
        whole = [(0, 30)]
        self.assertEqual(f"{silence.paired_upper(1, whole, 2.346):.5f}", "0.21000")
        self.assertEqual(f"{silence.paired_upper(1, whole, 60):.5f}", "0.15813")
        # A request longer than the window shows each message as if it came alone.
        self.assertAlmostEqual(silence.paired_upper(1, whole, 60), rate_upper(1, 30), places=12)
        self.assertAlmostEqual(silence.paired_upper(1, whole, 30), rate_upper(1, 30), places=12)

    def test_the_closed_form_for_at_most_one_message(self):
        # P(at most one message) = exp(-r (T + m)) (1 + 2 r m), with m = min(T, d) and r half the message rate.
        for window, delta in ((30, 2.346), (30, 60), (30, 0.01), (10, 9.99), (100, 3), (7.5, 7.5)):
            m = min(window, delta)
            r = silence.paired_upper(1, [(0, window)], delta) / 2
            self.assertAlmostEqual(math.exp(-r * (window + m)) * (1 + 2 * r * m), 0.05, places=12, msg=(window, delta))
        # Messages that fall together: at most one message means no request at all.
        self.assertAlmostEqual(silence.paired_upper(1, [(0, 30)], 0), 2 * quiet_bound(30), places=12)

    def test_no_message_is_the_bound_of_the_figure(self):
        for stretches, delta in (([(0, 30)], 2.346), ([(0, 30)], 60), ([(0, 10), (20, 30)], 2.346), ([(0, 10), (20, 30)], 15)):
            self.assertAlmostEqual(silence.paired_upper(0, stretches, delta), silence.paired_bound(stretches, delta), places=12)
        bounds = [silence.paired_upper(n, [(0, 30)], 2.346) for n in range(6)]
        self.assertEqual(bounds, sorted(bounds))                 # more messages, a higher bound
        self.assertEqual(f"{bounds[0]:.5f}", "0.18523")

    def test_the_two_kinds_of_start_time(self):
        # Start times that bring one message and two, tried one by one, against the lengths used.
        step = 0.002
        for stretches, delta in (([(0, 30)], 2.346), ([(0, 30)], 60), ([(0, 10), (20, 30)], 2.346), ([(0, 10), (20, 30)], 15),
                                 ([(0, 10), (20, 30)], 20), ([(0, 4), (5, 30)], 2.346)):
            low = min(x for x, _ in stretches) - delta - 1
            counts = [messages(low + step * i, stretches, delta) for i in range(int((31 + delta) / step))]
            seen = sum(y - x for x, y in stretches)
            union = silence.reach(stretches, delta)
            self.assertAlmostEqual(counts.count(2) * step, 2 * seen - union, delta=0.02, msg=(stretches, delta))
            self.assertAlmostEqual(counts.count(1) * step, 2 * (union - seen), delta=0.02, msg=(stretches, delta))

    def test_a_simulated_paired_source(self):
        rng = random.Random(1819)
        cases = ((1, [(0, 30)], 2.346), (1, [(0, 30)], 60), (2, [(0, 30)], 2.346), (3, [(0, 30)], 2.346), (1, [(0, 10), (20, 30)], 2.346),
                 (2, [(0, 10), (20, 30)], 15))
        for n, stretches, delta in cases:
            rate = silence.paired_upper(n, stretches, delta)
            self.assertAlmostEqual(simulated(rate, stretches, delta, n, rng), 0.05, delta=0.008, msg=(n, stretches, delta))
        # At the bound for messages that arrive singly, one message of a pair is not unusual.
        share = simulated(rate_upper(1, 30), [(0, 30)], 2.346, 1, rng)
        m, r = 2.346, rate_upper(1, 30) / 2
        self.assertAlmostEqual(share, math.exp(-r * (30 + m)) * (1 + 2 * r * m), delta=0.008)
        self.assertGreater(share, 0.10)

    def test_the_transcript(self):
        text = expected("ch16_silence")
        self.assertIn("one message in 30 s, from pairs\n  2.346 s apart:\n    at most 0.21000 msg/s at 95 %\n"
                      "  60 s apart:\n    at most 0.15813 msg/s at 95 %\n", text)


class NewClass(unittest.TestCase):
    def test_a_class_that_before_did_not_have(self):
        before, after = periods()
        more = dict(after, bulk=[1000.0] * 100)
        self.assertEqual(mix.added(more, before), ["bulk"])
        self.assertEqual(mix.added(after, before), [])
        self.assertEqual(mix.absent(more, before), {})
        # At the mix of before the new class has no weight: the result is that of the record without it.
        # (Version 18 raised KeyError for both.)
        self.assertEqual(sorted(mix.weighted(more, before)), sorted(mix.weighted(after, before)))
        self.assertEqual(mix.within(more, before, 200), mix.within(after, before, 200))
        pairs = mix.weighted(more, before)
        self.assertEqual((f"{mix.mean(pairs):.2f}", f"{mix.p95(pairs):.2f}"), ("126.00", "432.40"))
        self.assertEqual(mix.within(more, before, 200), (Fraction(892, 1000),) * 2)

    def test_the_new_class_stays_in_the_workload_as_observed(self):
        before, after = periods()
        more = dict(after, bulk=[1000.0] * 100)
        rows = ((after, "414.00", "673.92", "10.80"), (more, "467.27", "1000.00", "9.82"))
        for rec, mean, p95, share in rows:
            pairs = mix.weighted(rec)
            self.assertEqual(len(pairs), sum(len(xs) for xs in rec.values()))
            flat = sorted(x for xs in rec.values() for x in xs)
            self.assertEqual((f"{mix.mean(pairs):.2f}", f"{mix.p95(pairs):.2f}"), (mean, p95))
            self.assertEqual(f"{100 * sum(x <= 200 for x in flat) / len(flat):.2f}", share)
            self.assertAlmostEqual(float(mix.mean(pairs)), sum(flat) / len(flat), places=9)
        # The mean with the bulk class from the two means and the counts.
        self.assertEqual(f"{(414 * 1000 + 1000 * 100) / 1100:.2f}", "467.27")

    def test_an_empty_reference_mix_is_refused(self):
        before, after = periods()
        for empty in ({}, {"fast": [], "slow": []}):             # Version 18 took {} for the mix as observed
            with self.assertRaises(ValueError) as caught:
                mix.weighted(after, empty)
            self.assertEqual(str(caught.exception), "empty reference mix")
            with self.assertRaises(ValueError):
                mix.within(after, empty, 200)
        self.assertEqual(len(mix.weighted(after)), 1000)         # no mix: as observed
        self.assertEqual(len(mix.weighted(after, None)), 1000)
        # A class with no weight in the reference and no observation is not missed either.
        self.assertEqual(mix.absent({"fast": after["fast"], "slow": after["slow"]}, dict(before, bulk=[])), {})

    def test_the_transcript(self):
        text = expected("ch23_mix")
        for line in ("after, with a new class, bulk:", "workload         mean      p95  <=200 ms", "after          414.00   673.92  10.80 %",
                     "with bulk      467.27  1000.00   9.82 %", "at old mix     126.00   432.40  89.20 %"):
            self.assertIn(line + "\n", text)
        before, after = periods()
        self.assertIn(printed(mix.new_class, before, after), text)        # the end of the transcript until Version 20


class ComponentsOfTheChange(unittest.TestCase):
    def test_the_table_of_the_review(self):
        # From the class means and the shares alone.
        old_mix, new_mix = (0.9, 0.1), (0.1, 0.9)
        old_means, new_means = (100, 500), (90, 450)

        def at(means, shares):
            return sum(m * s for m, s in zip(means, shares))
        start, end = at(old_means, old_mix), at(new_means, new_mix)
        self.assertEqual((round(start), round(end)), (140, 414))
        means_first = (at(new_means, old_mix) - start, end - at(new_means, old_mix))
        mix_first = (end - at(old_means, new_mix), at(old_means, new_mix) - start)
        self.assertEqual(tuple(round(x) for x in means_first), (-14, 288))
        self.assertEqual(tuple(round(x) for x in mix_first), (-46, 320))
        self.assertEqual(tuple(round((a + b) / 2) for a, b in zip(means_first, mix_first)), (-30, 304))
        for pair in (means_first, mix_first):
            self.assertAlmostEqual(sum(pair), 274, places=9)
        # The listing, from the records.
        before, after = periods()
        def overall(rec, at):                        # the listing reweights the durations (since Version 20)
            return float(mix.mean(mix.weighted(rec, at)))
        self.assertAlmostEqual(overall(before, before), 140, places=9)
        self.assertAlmostEqual(overall(after, after), 414, places=9)
        self.assertAlmostEqual(overall(after, before), 126, places=9)
        self.assertAlmostEqual(overall(before, after), 460, places=9)
        out = printed(mix.components, before, after)                 # each order now names the value it passes through
        self.assertEqual(out, "mean, 140 -> 414 ms: +274 ms\n  class means first, via 126 ms:\n    within -14, mix +288\n"
                              "  mix first, via 460 ms:\n    within -46, mix +320\n  average of the two orders:\n"
                              "    within -30, mix +304\n")
        self.assertIn(out, expected("ch23_mix"))


if __name__ == "__main__":
    unittest.main()
