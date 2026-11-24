"""Regression tests added after the review of Version 19.

The review found that a line was assigned to the verified time by its time
stamp, so that a line isolated between two sequence gaps was counted when it
bore the stamp of the line that ends a verified run; that the bound for a
paired source measured the time shared by two overlapping stretches twice;
that a pooled percentile can be recomputed at both mixes and its change
divided in the two orders, as the change of the mean is; and that the
two-order calculation failed on a class which only the later record has,
where one order is determined and can be reported.

Each test computes what it expects independently of the code under test: the
runs of a record from a table of its sequence numbers, the times of the
verified stretches with the loop of Version 19, unions of stretches on a grid,
the probability of at most one message in closed form and by seeded
simulation, percentiles at a mix from observations replicated in proportion to
their weights, and means from the class means and their shares.
"""
import contextlib
import io
import math
import random
import unittest
from pathlib import Path

from examples import ch16_silence as silence
from examples import ch16_trace_measurements as trace
from examples import ch23_mix as mix
from swmetro.stats import quiet_bound, rate_upper

ROOT = Path(__file__).resolve().parent.parent
START = (12 * 3600 + 26 * 60) * 1000
SPAN = (START, START + 60_000)
COUNTER = (1, 800)
HALF = (0, 60_000)
REVIEW = ((1, 30), (2, 40), (5, 40), (9, 50), (10, 60))      # sequence number, second


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


def lines(marks, own=None):
    return [(n, round(1000 * t), "svc" if n == own else "pool", "x") for n, t in marks]


def stretches_v19(rows, lo, hi, span, counter=None):
    """The times of the verified stretches as Version 19 computed them, from pairs of neighbouring lines."""
    a, b = trace.window(lo, hi, span)
    marks = [(r[0], r[1]) for r in rows]
    if counter:
        marks.insert(0, (counter[0] - 1, span[0]))
        marks.append((counter[1], span[1]))
    out = []
    for p, q in zip(marks, marks[1:]):
        if q[1] < p[1]:
            return []
        x, y = max(p[1], a), min(q[1], b)
        if q[0] - p[0] != 1 or x >= y:
            continue
        x, y = x - span[0], y - span[0]
        if out and out[-1][1] == x:
            x = out.pop()[0]
        out.append((x, y))
    return out


def members(rows, a, b):
    """Sequence numbers of the lines in the verified time of [a, b) ms, found line by line: a line belongs if it lies
    in the window and the run of consecutive numbers around it spans time inside the window."""
    at = {seq: ms for seq, ms, _, _ in rows}
    out = set()
    for seq, ms in at.items():
        first = last = seq
        while first - 1 in at:
            first -= 1
        while last + 1 in at:
            last += 1
        if a <= ms < b and max(at[first], a) < min(at[last], b):
            out.add(seq)
    return out


def record(rng, size=40):
    """A record with lost lines and with lines that share a stamp, on a capture of 60 s."""
    seq, ms, rows = 0, 0, []
    for _ in range(size):
        seq += rng.choice((1, 1, 1, 1, 2, 5))
        ms += rng.choice((0, 0, 500, 1000, 1500, 4000))
        if ms > 60_000:
            break
        rows.append((seq, ms, rng.choice(("svc", "pool", "db")), "x"))
    return rows


def union(spans, step=0.01):
    """Length of the union of the stretches, counted on a grid."""
    low, high = min(x for x, _ in spans), max(y for _, y in spans)
    cells = int(round((high - low) / step))
    return step * sum(any(x <= low + step * (i + 0.5) < y for x, y in spans) for i in range(cells))


def messages(start, spans, delta):
    """How many of its two messages a request that starts here leaves in the observed time."""
    return sum(any(x <= t < y for x, y in spans) for t in (start, start + delta))


def simulated(rate, spans, delta, at_most, rng, trials=30_000):
    """Share of trials in which requests at this message rate leave at most so many messages."""
    low = min(x for x, _ in spans) - delta
    high = max(y for _, y in spans)
    few = 0
    for _ in range(trials):
        t, count = low + rng.expovariate(rate / 2), 0
        while t < high and count <= at_most:
            count += messages(t, spans, delta)
            t += rng.expovariate(rate / 2)
        few += count <= at_most
    return few / trials


def periods():
    rng = random.Random(23)
    return mix.record(mix.BEFORE, rng), mix.record(mix.AFTER, rng)


def replicated_p95(rec, shares):
    """The 95th percentile at the given class shares, by the inverse empirical distribution of the observations
    replicated in proportion to their weights. Shares are in tenths, and every class has 100 or 900 observations."""
    pooled = []
    for name, tenths in shares.items():
        copies = tenths * 900 // len(rec[name])              # equal total weight per tenth of the mix
        pooled += rec[name] * copies
    pooled.sort()
    return pooled[math.ceil(0.95 * len(pooled)) - 1]


class LinesOfTheVerifiedTime(unittest.TestCase):
    def test_the_rows_of_the_review(self):
        found = trace.runs(lines(REVIEW), 30, 60, HALF)
        self.assertEqual(found, [(30_000, 40_000, [1, 2]), (50_000, 60_000, [9])])
        self.assertEqual(trace.stretches(lines(REVIEW), 30, 60, HALF), [(30_000, 40_000), (50_000, 60_000)])
        self.assertEqual(trace.verified(lines(REVIEW), 30, 60, HALF), 20.0)
        # The record holds the service line in the window twice and outside it once; the verified time holds it once.
        for own, in_record, in_verified in ((2, 1, 1), (5, 1, 0), (10, 0, 0)):
            rows = lines(REVIEW, own)
            self.assertEqual(trace.count(rows, "svc", 30, 60, HALF), in_record, own)
            self.assertEqual(silence.inside(rows, "svc", found), in_verified, own)
            self.assertEqual(silence.inside(rows, "svc", trace.runs(rows, 30, 60, HALF)), in_verified, own)
            self.assertEqual(rows, silence.edge(own))
        # Version 19 counted by the stamp, with closed ends, and took line 5 for line 2.
        spans = trace.stretches(lines(REVIEW), 30, 60, HALF)
        by_stamp = [sum(c == "svc" and 30_000 <= ms < 60_000 and any(x <= ms <= y for x, y in spans)
                        for _, ms, c, _ in lines(REVIEW, own)) for own in (2, 5, 10)]
        self.assertEqual(by_stamp, [1, 1, 0])

    def test_the_bounds_follow_the_count(self):
        # Line 5 is in the record and not in the 20 verified seconds; line 2 is in both.
        for own, count, bound in ((5, 0, quiet_bound(20)), (2, 1, rate_upper(1, 20))):
            out = printed(silence.models, lines(REVIEW, own), HALF, None)
            self.assertIn(f"svc, 30-60 s: 1 message, 20 s verified\n  in the verified time: {count}\n", out)
            self.assertIn(f"    at most {bound:.5f} msg/s at 95 %\n", out)
        out = printed(silence.models, lines(REVIEW, 10), HALF, None)
        self.assertIn("svc, 30-60 s: 0 messages, 20 s verified\n  in the verified time: 0\n", out)

    def test_stamps_alone_leave_the_ends_open(self):
        # Since Version 21 a line that is alone with its stamp at an end of a run is settled by the stamp
        # (tests/test_review20.py); lines 2 and 5 share theirs, and stay open.
        found = trace.runs(lines(REVIEW), 30, 60, HALF)
        for own, want in ((1, (1, 1)), (2, (0, 1)), (5, (0, 1)), (9, (1, 1)), (10, (0, 0))):
            self.assertEqual(silence.stamped(lines(REVIEW, own), "svc", found, 30, 60, HALF), want, own)
        self.assertEqual(silence.stamped(lines(((3, 35),), 3), "svc", found, 30, 60, HALF), (1, 1))        # inside a run
        self.assertEqual(silence.stamped(lines(((3, 45),), 3), "svc", found, 30, 60, HALF), (0, 0))        # in the gap
        # The range always holds the count by sequence number.
        rng = random.Random(1920)
        open_ends = 0
        for _ in range(600):
            rows = record(rng)
            lo = rng.choice((0, 10, 20, 30))
            hi = lo + rng.choice((10, 20, 30))
            found = trace.runs(rows, lo, hi, HALF)
            for comp in ("svc", "db", "pool"):
                least, most = silence.stamped(rows, comp, found, lo, hi, HALF)
                count = silence.inside(rows, comp, found)
                self.assertLessEqual(least, count)
                self.assertLessEqual(count, most)
                open_ends += least < most
        self.assertGreater(open_ends, 300)
        # Two runs that meet at one stamp, with lost lines and an isolated line between them: the stretch of time is
        # one, and the stamp still does not say to which side of the loss a line belongs.
        rows = lines(((1, 30), (2, 40), (5, 40), (8, 40), (9, 50)), 5)
        found = trace.runs(rows, 30, 60, HALF)
        self.assertEqual(found, [(30_000, 40_000, [1, 2]), (40_000, 50_000, [8, 9])])
        self.assertEqual(trace.stretches(rows, 30, 60, HALF), [(30_000, 50_000)])
        self.assertEqual((silence.inside(rows, "svc", found), silence.stamped(rows, "svc", found, 30, 60, HALF)), (0, (0, 1)))
        # A capture that does not begin at zero: the times of the runs are in ms after its start.
        late = (5000, 65_000)
        shifted = [(n, ms + 5000, c, m) for n, ms, c, m in lines(REVIEW, 2)]
        self.assertEqual(silence.stamped(shifted, "svc", trace.runs(shifted, 30, 60, late), 30, 60, late), (0, 1))

    def test_lines_that_share_a_stamp(self):
        # Two lines of one stamp between two gaps: no time is verified, and neither line is counted.
        rows = lines(((1, 30), (2, 40), (5, 40), (6, 40), (9, 50), (10, 60)), 5)
        self.assertEqual(trace.runs(rows, 30, 60, HALF), [(30_000, 40_000, [1, 2]), (50_000, 60_000, [9])])
        # A run that begins and ends with lines of one stamp keeps them all.
        rows = lines(((1, 30), (2, 40), (5, 40), (6, 40), (7, 45), (8, 45), (11, 45), (12, 50)))
        self.assertEqual(trace.runs(rows, 30, 60, HALF),
                         [(30_000, 40_000, [1, 2]), (40_000, 45_000, [5, 6, 7, 8]), (45_000, 50_000, [11, 12])])
        self.assertEqual(trace.stretches(rows, 30, 60, HALF), [(30_000, 50_000)])       # the times touch
        # Inside a run, too.
        rows = lines(((1, 30), (2, 35), (3, 35), (4, 40)), 3)
        self.assertEqual(trace.runs(rows, 30, 60, HALF), [(30_000, 40_000, [1, 2, 3, 4])])
        self.assertEqual(silence.inside(rows, "svc", trace.runs(rows, 30, 60, HALF)), 1)

    def test_the_window_decides_by_the_stamp(self):
        rows = lines(((1, 20), (2, 30), (3, 35), (4, 40), (5, 50)))
        self.assertEqual(trace.runs(rows, 30, 40, HALF), [(30_000, 40_000, [2, 3])])    # 30 s is in, 40 s is out
        self.assertEqual(trace.runs(rows, 25, 45, HALF), [(25_000, 45_000, [2, 3, 4])])
        self.assertEqual(trace.runs(rows, 31, 34, HALF), [(31_000, 34_000, [])])        # verified, and no line in it
        self.assertEqual(trace.runs(rows, 50, 60, HALF), [])
        # A line stamped where the window opens, with its run before the window: no verified time, no count.
        rows = lines(((1, 30), (2, 40), (5, 40), (6, 45)), 2)
        self.assertEqual(trace.runs(rows, 40, 60, HALF), [(40_000, 45_000, [5, 6])])
        self.assertEqual(silence.inside(rows, "svc", trace.runs(rows, 40, 60, HALF)), 0)
        self.assertEqual(trace.count(rows, "svc", 40, 60, HALF), 1)

    def test_counter_readings_are_not_lines(self):
        self.assertEqual(trace.runs([], 0, 60, HALF, (1, 1)), [(0, 60_000, [])])
        self.assertEqual(trace.runs([], 0, 60, HALF, (1, 41)), [])
        self.assertEqual(trace.runs([], 0, 60, HALF), [])
        rows = lines(((1, 0), (2, 20), (3, 60)))
        self.assertEqual(trace.runs(rows, 0, 60, HALF, (1, 4)), [(0, 60_000, [1, 2])])  # the line at 60 s is outside
        self.assertEqual(trace.runs(rows, 0, 60, HALF, (1, 9)), [(0, 60_000, [1, 2])])  # and so is the lost end
        self.assertEqual(trace.runs(rows, 0, 60, HALF, (0, 4)), [(0, 60_000, [1, 2])])  # a line was lost at the start
        rows = lines(((3, 10), (4, 20)))
        self.assertEqual(trace.runs(rows, 0, 60, HALF, (1, 5)), [(10_000, 60_000, [3, 4])])
        self.assertEqual(trace.runs(rows, 0, 60, HALF, (3, 9)), [(0, 20_000, [3, 4])])
        # Stamps that contradict the order verify nothing.
        back = lines(((1, 1), (2, 26), (3, 6), (4, 29)))
        self.assertEqual(trace.runs(back, 0, 30, (0, 30_000)), [])

    def test_numbers_that_do_not_rise_verify_nothing(self):
        # Lines are told by their numbers, so that the numbers have to name one line each. After a restart of the
        # counter a second line 2, alone between two losses, would be taken for the line 2 that ends a run.
        restart = lines(((1, 30), (2, 40), (2, 41), (9, 50), (10, 60)))
        again = lines(((1, 30), (2, 35), (1, 40), (2, 45)))
        reordered = lines(((5, 30), (6, 40), (3, 45), (4, 50)))
        for rows in (restart, again, reordered):
            self.assertEqual(trace.runs(rows, 30, 60, HALF), [])
            self.assertEqual(trace.stretches(rows, 30, 60, HALF), [])
            self.assertEqual(trace.verified(rows, 30, 60, HALF), 0.0)
            self.assertEqual(printed(silence.models, rows, HALF, None).splitlines()[1], "  exposure unknown, no bound")
        self.assertEqual(stretches_v19(again, 30, 60, HALF), [(30_000, 35_000), (40_000, 45_000)])     # Version 19
        # Counter readings that contradict the numbers of the lines.
        rows = lines(((3, 10), (4, 20)))
        self.assertEqual(trace.runs(rows, 0, 60, HALF, (3, 5)), [(0, 60_000, [3, 4])])
        # Until Version 55 these three verified nothing, and completeness() refused them: the two now
        # share one check of the numbers and of the readings (review of Version 54).
        for counter in ((4, 9), (1, 4)):        # a line numbered below the first reading, and one at the last
            with self.assertRaisesRegex(ValueError, "beyond the counter"):
                trace.runs(rows, 0, 60, HALF, counter)
        with self.assertRaisesRegex(ValueError, "counter runs back"):
            trace.runs([], 0, 60, HALF, (5, 3))
        # Numbers that rise with gaps are the ordinary case.
        self.assertEqual(trace.runs(lines(((1, 30), (2, 40), (7, 41), (8, 50))), 30, 60, HALF),
                         [(30_000, 40_000, [1, 2]), (41_000, 50_000, [7, 8])])

    def test_random_records_against_a_line_by_line_reference(self):
        rng = random.Random(1916)
        seen = 0
        for _ in range(1500):
            rows = record(rng)
            lo = rng.choice((0, 5, 10, 20, 30))
            hi = min(60, lo + rng.choice((5, 10, 20, 30, 60)))
            found = trace.runs(rows, lo, hi, HALF)
            got = [seq for _, _, own in found for seq in own]
            self.assertEqual(len(got), len(set(got)))                                   # a line belongs to one part
            self.assertEqual(set(got), members(rows, 1000 * lo, 1000 * hi))
            for x, y, own in found:
                self.assertLess(x, y)
                self.assertEqual(own, list(range(own[0], own[-1] + 1)) if own else [])
            for comp in ("svc", "db"):
                want = sum(c == comp and seq in set(got) for seq, _, c, _ in rows)
                self.assertEqual(silence.inside(rows, comp, found), want)
            seen += len(got)
            # The times are those of Version 19, with and without counter readings.
            for counter in (None, (rows[0][0], rows[-1][0] + 1), (rows[0][0] - 2, rows[-1][0] + 3)):
                want = stretches_v19(rows, lo, hi, HALF, counter)
                self.assertEqual(trace.stretches(rows, lo, hi, HALF, counter), want)
                parts = trace.runs(rows, lo, hi, HALF, counter)
                self.assertAlmostEqual(sum(y - x for x, y, _ in parts), sum(y - x for x, y in want))
                self.assertEqual(trace.verified(rows, lo, hi, HALF, counter), sum(y - x for x, y in want) / 1000)
        self.assertGreater(seen, 10_000)

    def test_the_fixture(self):
        rows = fixture()
        first = trace.runs(rows, 0, 30, SPAN, COUNTER)
        second = trace.runs(rows, 30, 60, SPAN, COUNTER)
        last = max(seq for seq, ms, _, _ in rows if ms < START + 30_000)        # the last line of the first half
        self.assertEqual(last, 638)
        self.assertEqual([(x, y, own[0], own[-1]) for x, y, own in first], [(0, 20_099, 1, 430), (22_187, 30_000, 471, last)])
        self.assertEqual([(x, y, own[0], own[-1]) for x, y, own in second], [(30_000, 60_000, last + 1, 799)])
        numbers = [r[0] for r in rows]
        self.assertEqual(sorted(seq for _, _, own in first + second for seq in own), numbers)   # every line, once
        self.assertEqual(trace.stretches(rows, 0, 30, SPAN, COUNTER), [(0, 20_099), (22_187, 30_000)])
        self.assertEqual((silence.inside(rows, "svc", first), silence.inside(rows, "svc", second)), (2, 0))
        self.assertEqual((silence.inside(rows, "db", first), silence.inside(rows, "db", second)), (564, 127))

    def test_one_service_line_whatever_its_second(self):
        # Version 19 marked every line of the given second as the service's, and gap(40) had two.
        for at in (30, 31, 35, 40, 41, 45, 49, 50, 55, 60):
            rows = silence.gap(at)
            self.assertEqual([c for _, _, c, _ in rows].count("svc"), 1, at)
            self.assertEqual(len(rows), 5)
            self.assertEqual([ms for _, ms, _, _ in rows], sorted(ms for _, ms, _, _ in rows))
            self.assertEqual(trace.stretches(rows, 30, 60, HALF), [(30_000, 40_000), (50_000, 60_000)], at)
            found = trace.runs(rows, 30, 60, HALF)
            want = 30 <= at <= 40 or 50 <= at < 60          # written after the other line of its second, and in its run
            self.assertEqual(silence.inside(rows, "svc", found), int(want), at)
        self.assertEqual(silence.gap(45), lines(((1, 30), (2, 40), (6, 45), (10, 50), (11, 60)), 6))
        self.assertEqual(silence.gap(35), lines(((1, 30), (2, 35), (3, 40), (7, 50), (8, 60)), 2))

    def test_the_transcript(self):
        text = expected("ch16_silence")
        self.assertIn("lines 1, 2, 5, 9 and 10, stamped\n30, 40, 40, 50 and 60 s:\n  verified time has lines 1 to 2 and 9\n"
                      "  the line of the service, counted\n  by number, and by stamp alone:\n    line 2: 1, and 0 to 1\n"
                      "    line 5: 0, and 0 to 1\n    line 10: 0, and 0\n", text)
        self.assertIn(printed(silence.counted), text)
        # What Version 19 printed is still printed, line for line.
        old = ("svc, 30-60 s: 0 messages, 30 s verified\n  in the verified time: 0\n",
               "at 45 s\nsvc, 30-60 s: 1 message, 20 s verified\n  in the verified time: 0\n  messages arriving singly:\n"
               "    at most 0.14979 msg/s at 95 %\n",
               "at 35 s\nsvc, 30-60 s: 1 message, 20 s verified\n  in the verified time: 1\n  messages arriving singly:\n"
               "    at most 0.23719 msg/s at 95 %\n",
               "first half minute, verified from\n  0 to 20.099 and 22.187 to 30 s:\n  27.912 s, and for pairs 2.346 s\n"
               "  apart, starts covered: 32.346 s\n")
        for piece in old:
            self.assertIn(piece, text)
        self.assertEqual(len(text.splitlines()), 193)            # 82 in Version 20, 88, 101, 118, 140, 152, 162, 166 in Version 27
        self.assertIn("svc, 30-60 s: 0 messages, 30 s verified\n", expected("ch16_trace_measurements"))

    def test_the_figure_fills_the_end_of_a_run(self):
        # The rule of the figure, applied here: an end is filled where a line of the run is stamped.
        for at in (45, 35):
            rows = silence.gap(at)
            stamp = {seq: ms for seq, ms, _, _ in rows}
            ends = {}
            for lo, hi, own in trace.runs(rows, 30, 60, HALF):
                ends[lo // 1000] = stamp[own[0]] == lo
                ends[hi // 1000] = stamp[own[-1]] == hi
            self.assertEqual(ends, {30: True, 40: True, 50: True, 60: False}, at)
        source = (ROOT / "figures" / "make_figures.py").read_text(encoding="utf-8")
        for piece in ("for lo, hi, own in c16.runs(rows, 30, 60, (0, 60_000)):", "for end, seq in ((lo, own[0]), (hi, own[-1])):",
                      'fill = BLACK if stamp[seq] == end else "white"'):
            self.assertIn(piece, source)
        self.assertEqual(source.count("\n    save(fig, "), 32)


class OverlappingStretches(unittest.TestCase):
    def test_the_figures_of_the_review(self):
        self.assertEqual(f"{silence.paired_upper(1, [(0, 30)], 10):.5f}", "0.20565")
        self.assertEqual(f"{silence.paired_upper(1, [(0, 20), (10, 30)], 10):.5f}", "0.20565")
        # What Version 19 returned for the second description, and what that rate would mean.
        seen, covered = 20 + 20, 40                                   # the shared ten seconds counted twice
        two, one = 2 * seen - covered, 2 * (covered - seen)
        self.assertEqual((two, one), (40, 0))
        wrong = 2 * -math.log(0.05) / two                             # no request may start in 40 s
        self.assertEqual(f"{wrong:.5f}", "0.14979")
        r = wrong / 2
        self.assertEqual(f"{100 * math.exp(-r * 40) * (1 + 2 * r * 10):.2f}", "12.49")
        # At the bound, at most one message has a probability of 5 %.
        r = silence.paired_upper(1, [(0, 20), (10, 30)], 10) / 2
        self.assertAlmostEqual(math.exp(-r * 40) * (1 + 2 * r * 10), 0.05, places=12)

    def test_a_description_does_not_change_the_bound(self):
        same = ([(0, 30)], [(0, 20), (10, 30)], [(10, 30), (0, 20)], [(0, 15), (15, 30)], [(0, 30), (0, 30)],
                [(0, 30), (5, 6)], [(0, 10), (5, 20), (15, 30)], [(20, 30), (0, 12), (11, 21), (3, 4)])
        for delta in (0, 2.346, 10, 29, 60):
            for n in (0, 1, 2, 5):
                want = silence.paired_upper(n, [(0, 30)], delta)
                for spans in same:
                    self.assertAlmostEqual(silence.paired_upper(n, spans, delta), want, places=12, msg=(n, spans, delta))
            for spans in same:
                self.assertAlmostEqual(silence.paired_bound(spans, delta), silence.paired_bound([(0, 30)], delta), places=12)
                self.assertEqual(silence.starts(spans, delta), silence.starts([(0, 30)], delta))

    def test_merged(self):
        self.assertEqual(silence.merged([(10, 30), (0, 20)]), [(0, 30)])
        self.assertEqual(silence.merged([(0, 10), (10, 20), (25, 30)]), [(0, 20), (25, 30)])
        self.assertEqual(silence.merged([(0, 30), (5, 6), (40, 50), (45, 46)]), [(0, 30), (40, 50)])
        self.assertEqual(silence.merged([]), [])
        with self.assertRaisesRegex(ValueError, "empty stretch"):
            silence.merged([(0, 10), (5, 5)])
        with self.assertRaisesRegex(ValueError, "empty stretch"):
            silence.paired_upper(1, [(0, 10), (12, 11)], 2)
        with self.assertRaisesRegex(ValueError, "need delta >= 0"):
            silence.starts([(0, 10)], -1)
        spans = [(10, 30), (0, 20)]
        silence.paired_upper(1, spans, 10)
        self.assertEqual(spans, [(10, 30), (0, 20)])                 # the caller's list is left as it was

    def test_random_stretches_against_a_grid(self):
        rng = random.Random(1919)
        for _ in range(40):
            spans = []
            for _ in range(rng.randint(1, 5)):
                x = rng.randint(0, 50)
                spans.append((x, x + rng.randint(1, 20)))
            delta = rng.choice((0.5, 2, 7, 15, 40))
            seen = sum(y - x for x, y in silence.merged(spans))
            self.assertAlmostEqual(seen, union(spans), delta=0.02, msg=spans)
            covered = silence.reach(spans, delta)
            moved = spans + [(x - delta, y - delta) for x, y in spans]
            self.assertAlmostEqual(covered, union(moved), delta=0.03, msg=(spans, delta))
            # The two kinds of start time, tried one by one.
            step = 0.01
            low = min(x for x, _ in spans) - delta - 1
            counts = [messages(low + step * (i + 0.5), spans, delta) for i in range(int((max(y for _, y in spans) - low + 1) / step))]
            self.assertAlmostEqual(counts.count(2) * step, 2 * seen - covered, delta=0.05, msg=(spans, delta))
            self.assertAlmostEqual(counts.count(1) * step, 2 * (covered - seen), delta=0.1, msg=(spans, delta))

    def test_a_simulated_paired_source(self):
        rng = random.Random(1920)
        cases = ((1, [(0, 20), (10, 30)], 10), (1, [(0, 20), (10, 30)], 2.346), (2, [(0, 20), (5, 12), (10, 30)], 10),
                 (0, [(0, 20), (10, 30)], 10), (1, [(0, 12), (8, 20), (30, 40)], 15))
        for n, spans, delta in cases:
            rate = silence.paired_upper(n, spans, delta)
            self.assertAlmostEqual(simulated(rate, spans, delta, n, rng), 0.05, delta=0.008, msg=(n, spans, delta))
        # At the rate that Version 19 returned, one message or none is not unusual.
        self.assertAlmostEqual(simulated(0.14979, [(0, 20), (10, 30)], 10, 1, rng), 0.1249, delta=0.008)

    def test_the_callers_of_the_chapter_are_unchanged(self):
        text = expected("ch16_silence")
        for piece in ("    starts covered: 32.346 s\n    at most 0.18523 msg/s at 95 %\n",
                      "  2.346 s apart:\n    at most 0.21000 msg/s at 95 %\n  60 s apart:\n    at most 0.15813 msg/s at 95 %\n"):
            self.assertIn(piece, text)
        # The window of the review, whole and as two stretches that overlap.
        self.assertIn("  10 s apart:\n    at most 0.20565 msg/s at 95 %\n  10 s apart, the 30 s given as\n"
                      "  0 to 20 s and 10 to 30 s:\n    at most 0.20565 msg/s at 95 %\n", text)
        self.assertIn(printed(silence.pair_count), text)


class PercentileAtBothMixes(unittest.TestCase):
    OLD, NEW = {"fast": 9, "slow": 1}, {"fast": 1, "slow": 9}

    def test_the_four_cells_of_the_review(self):
        before, after = periods()
        cells = [[replicated_p95(rec, shares) for shares in (self.OLD, self.NEW)] for rec in (before, after)]
        self.assertEqual([[f"{x:.2f}" for x in row] for row in cells], [["469.09", "805.85"], ["432.40", "673.92"]])
        # The listing reweights instead of replicating, and reads the same observations.
        for rec, row in zip((before, after), cells):
            for at, want in zip((before, after), row):
                self.assertEqual(mix.p95(mix.weighted(rec, at)), want)
        self.assertEqual(mix.p95(mix.weighted(before)), cells[0][0])
        self.assertEqual(mix.p95(mix.weighted(after)), cells[1][1])

    def test_the_two_orders(self):
        before, after = periods()
        (a, b), (c, d) = [[replicated_p95(rec, shares) for shares in (self.OLD, self.NEW)] for rec in (before, after)]
        durations_first, mix_first = (c - a, d - c), (d - b, b - a)
        self.assertEqual([f"{x:+.2f}" for x in durations_first], ["-36.69", "+241.52"])
        self.assertEqual([f"{x:+.2f}" for x in mix_first], ["-131.93", "+336.76"])
        average = [(x + y) / 2 for x, y in zip(durations_first, mix_first)]
        self.assertEqual([f"{x:+.2f}" for x in average], ["-84.31", "+289.14"])
        for pair in (durations_first, mix_first, average):
            self.assertAlmostEqual(sum(pair), d - a, places=9)
        self.assertEqual(f"{d - a:+.2f}", "+204.83")
        self.assertTrue(all(pair[1] > 0 for pair in (durations_first, mix_first)))      # the mix raises p95 in both
        out = printed(mix.components, before, after, mix.p95, "p95", "distributions")
        self.assertEqual(out, "p95, 469 -> 674 ms: +205 ms\n  class distributions first, via 432 ms:\n"
                              "    within -37, mix +242\n  mix first, via 806 ms:\n    within -132, mix +337\n"
                              "  average of the two orders:\n    within -84, mix +289\n")
        self.assertIn(out, expected("ch23_mix"))

    def test_the_class_percentiles_fall(self):
        before, after = periods()

        def nearest_rank(xs):
            return sorted(xs)[math.ceil(0.95 * len(xs)) - 1]
        got = {name: (nearest_rank(before[name]), nearest_rank(after[name])) for name in before}
        self.assertEqual({name: tuple(f"{x:.2f}" for x in pair) for name, pair in got.items()},
                         {"fast": ("158.27", "150.45"), "slow": ("805.85", "683.34")})
        for name, (b, a) in got.items():
            self.assertLess(a, b)
            self.assertEqual((mix.class_p95(before[name]), mix.class_p95(after[name])), (b, a))
        text = expected("ch23_mix")
        self.assertTrue(text.startswith("            before             after\nclass     n  mean  p95      n  mean  p95\n"
                                        "fast    900   100  158    100    90  150\n"
                                        "slow    100   500  806    900   450  683\n"))
        self.assertIn("mean of the class p95s, after     417\n", text)
        self.assertEqual(round((got["fast"][1] + got["slow"][1]) / 2), 417)

    def test_the_mean_is_the_mean_of_version_19(self):
        before, after = periods()
        out = printed(mix.components, before, after)
        self.assertEqual([line for line in out.splitlines() if "within" in line],
                         ["    within -14, mix +288", "    within -46, mix +320", "    within -30, mix +304"])
        # The values passed through, from the class means and the shares.
        self.assertIn("  class means first, via 126 ms:\n", out)        # 0.9 * 90 + 0.1 * 450
        self.assertIn("  mix first, via 460 ms:\n", out)                # 0.1 * 100 + 0.9 * 500
        self.assertEqual((round(0.9 * 90 + 0.1 * 450), round(0.1 * 100 + 0.9 * 500)), (126, 460))


class ClassThatOnlyOneRecordHas(unittest.TestCase):
    def test_the_path_of_the_review(self):
        before, after = periods()
        more = dict(after, bulk=[1000.0] * 100)
        # From the class means and the counts.
        start = (900 * 100 + 100 * 500) / 1000
        at_old_mix = (900 * 90 + 100 * 450) / 1000
        end = (100 * 90 + 900 * 450 + 100 * 1000) / 1100
        self.assertEqual((f"{start:.2f}", f"{at_old_mix:.2f}", f"{end:.2f}"), ("140.00", "126.00", "467.27"))
        self.assertEqual((f"{at_old_mix - start:+.2f}", f"{end - at_old_mix:+.2f}"), ("-14.00", "+341.27"))
        out = printed(mix.components, before, more)                   # Version 19: KeyError 'bulk'
        self.assertEqual(out, "mean, 140 -> 467 ms: +327 ms\n  class means first, via 126 ms:\n    within -14, mix +341\n"
                              "  mix first: not determined,\n    no bulk requests before\n")
        self.assertIn(out, expected("ch23_mix"))                     # the end of the transcript in Version 20
        self.assertAlmostEqual(float(mix.mean(mix.weighted(more, before))), at_old_mix, places=9)
        self.assertAlmostEqual(float(mix.mean(mix.weighted(more))), end, places=9)
        with self.assertRaisesRegex(ValueError, "no observations for bulk"):
            mix.weighted(before, more)                                # the order that is not determined
        # The percentile has the same one path.
        out = printed(mix.components, before, more, mix.p95, "p95", "distributions")
        self.assertEqual(out, "p95, 469 -> 1000 ms: +531 ms\n  class distributions first, via 432 ms:\n"
                              "    within -37, mix +568\n  mix first: not determined,\n    no bulk requests before\n")

    def test_a_class_without_later_records(self):
        before, after = periods()
        part = {"fast": after["fast"]}
        out = printed(mix.components, before, part)
        self.assertEqual(out, "mean, 140 -> 90 ms: -50 ms\n  class means first: not determined,\n    no slow requests after\n"
                              "  mix first, via 100 ms:\n    within -10, mix -40\n")

    def test_neither_order(self):
        out = printed(mix.components, {"a": [1.0, 3.0], "b": [10.0]}, {"b": [20.0], "c": [5.0, 7.0]})
        self.assertEqual(out, "mean, 5 -> 11 ms: +6 ms\n  class means first: not determined,\n    no a requests after\n"
                              "  mix first: not determined,\n    no c requests before\n")

    def test_no_requests_have_no_statistic(self):
        # Version 19 returned None for the percentile of nothing and divided by zero for its mean.
        before, after = periods()
        for call in (lambda: mix.mean([]), lambda: mix.p95([]), lambda: mix.class_p95([]),
                     lambda: mix.mean(mix.weighted({}, before, partial=True)),
                     lambda: mix.p95(mix.weighted({"fast": []}, before, partial=True)),
                     lambda: printed(mix.components, {}, after), lambda: printed(mix.components, before, {"fast": []})):
            with self.assertRaisesRegex(ValueError, "no requests"):
                call()
        self.assertEqual(mix.p95([(7.0, 1)]), 7.0)
        self.assertEqual(mix.mean([(7.0, 1), (9.0, 3)]), 8.5)

    def test_both_orders_when_the_classes_agree(self):
        # Exact halves: two classes of two requests each, and a change of durations only, then of mix only.
        before = {"a": [10.0, 10.0], "b": [30.0, 30.0]}
        out = printed(mix.components, before, {"a": [20.0, 20.0], "b": [40.0, 40.0]})
        self.assertEqual(out.splitlines()[-1], "    within +10, mix +0")
        out = printed(mix.components, before, {"a": [10.0], "b": [30.0, 30.0, 30.0]})
        self.assertEqual(out.splitlines()[-1], "    within +0, mix +5")


if __name__ == "__main__":
    unittest.main()
