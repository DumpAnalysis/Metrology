"""Regression tests added after the review of Version 20.

The review found that the bound for a silent paired source changed when its
one window was supplied as an iterator, because the function read its
argument twice; that the range which time stamps alone leave for a count is
wider than the stamps require where a single line bears the stamp at which a
run ends; that a percentile can be determined at a mix with a class that has
no durations, where the two-order calculation declared the order open; and
that the same two records, compared at one mix for both, give the later
period the higher 95th percentile at a mix of 4.1 % slow requests although
the percentile of each class fell.

Each test computes what it expects independently of the code under test:
closed forms, counts by sequence number on random records, bounds on
cumulative shares in exact fractions, completions of the missing class tried
one by one, and percentiles from observations replicated in proportion to
their weights.
"""
import contextlib
import io
import math
import random
import statistics as st
import unittest
from fractions import Fraction
from pathlib import Path

from examples import ch10_coordinated_omission as omission
from examples import ch16_silence as silence
from examples import ch16_trace_measurements as trace
from examples import ch21_psi as drift
from examples import ch23_mix as mix
from swmetro import core, propagate, stats
from swmetro.stats import quiet_bound

ROOT = Path(__file__).resolve().parent.parent
START = (12 * 3600 + 26 * 60) * 1000
SPAN = (START, START + 60_000)
HALF = (0, 60_000)
TIED = ((1, 30), (2, 40), (5, 40), (9, 50), (10, 60))        # the lines of the book: 2 and 5 share a stamp
LONE = ((1, 30), (2, 40), (5, 50), (6, 60))                  # the lines of the review: no stamp twice
BEFORE = {"common": [80.0] * 92 + [100.0] * 6 + [180.0]}
AFTER = {"common": [70.0] * 92 + [90.0] * 6 + [160.0], "bulk": [500.0]}


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


def once(items):
    """The items as a generator, which can be read through one time."""
    return (x for x in items)


def record(rng, ties, size=40):
    """A record with lost lines on a capture of 60 s; with `ties`, some lines share a stamp."""
    seq, ms, rows = 0, 0, []
    for _ in range(size):
        seq += rng.choice((1, 1, 1, 1, 2, 5))
        ms += rng.choice((0, 0, 500, 1000, 1500, 4000) if ties else (250, 500, 1000, 1500, 4000))
        if ms > 60_000:
            break
        rows.append((seq, ms, rng.choice(("svc", "pool", "db")), "x"))
    return rows


def periods():
    rng = random.Random(23)
    return mix.record(mix.BEFORE, rng), mix.record(mix.AFTER, rng)


def replicated_p95(rec, fast, slow):
    """The 95th percentile at a mix of fast : slow, from observations replicated so that every class has copies in
    proportion to its weight; inverse empirical distribution of the pooled values."""
    weights = {"fast": Fraction(fast, len(rec["fast"])), "slow": Fraction(slow, len(rec["slow"]))}
    scale = math.lcm(*(w.denominator for w in weights.values()))
    copies = {name: int(w * scale) for name, w in weights.items()}
    shrink = math.gcd(*copies.values())
    pooled = sorted(x for name, xs in rec.items() for x in xs for _ in range(copies[name] // shrink))
    return pooled[math.ceil(Fraction(95, 100) * len(pooled)) - 1]


def shares(xs, weight, limit):
    """Shares of the whole mix that one class puts strictly below a limit, and at or below it."""
    below = Fraction(sum(x < limit for x in xs), len(xs)) * weight
    upto = Fraction(sum(x <= limit for x in xs), len(xs)) * weight
    return below, upto


def settled_p95(xs, weight, gap):
    """The criterion of the review, in exact fractions: the duration, if there is one, below which at most 95 % can
    lie and at or below which at least 95 % must, when a share `gap` of the mix has no durations."""
    need = Fraction(95, 100)
    hits = []
    for limit in sorted(set(xs)):
        below, upto = shares(xs, weight, limit)
        if below + gap < need <= upto:
            hits.append(limit)
    assert len(hits) <= 1
    return hits[0] if hits else None


class OneShotArguments(unittest.TestCase):
    def test_the_figures_of_the_review(self):
        closed = 2 * -math.log(0.05) / 40                    # a request of 10 s leaves a message from -10 s to 30 s
        self.assertEqual(f"{closed:.5f}", "0.14979")
        for supply in (list, tuple, iter, once):
            window = [(0, 30)]
            self.assertAlmostEqual(silence.paired_bound(supply(window), 10), closed, places=12, msg=supply)
            self.assertAlmostEqual(silence.paired_upper(0, supply(window), 10), closed, places=12, msg=supply)
            self.assertEqual(silence.starts(supply(window), 10), [(-10, 30)])
            self.assertEqual(silence.reach(supply(window), 10), 40)
        # What Version 20 returned for the iterator: the window without its copy moved back, 30 s of start times.
        self.assertEqual(f"{2 * -math.log(0.05) / 30:.5f}", "0.19972")

    def test_any_iterable_of_stretches(self):
        rng = random.Random(2016)
        for _ in range(60):
            spans = []
            for _ in range(rng.randint(1, 4)):
                x = rng.randint(0, 50)
                spans.append((x, x + rng.randint(1, 20)))
            delta = rng.choice((0, 0.5, 2.346, 15, 60))
            want = silence.starts(spans, delta)
            covered = sum(y - x for x, y in want)
            for supply in (tuple, iter, once):
                self.assertEqual(silence.starts(supply(spans), delta), want)
                self.assertEqual(silence.reach(supply(spans), delta), covered)
                self.assertAlmostEqual(silence.paired_bound(supply(spans), delta), 2 * quiet_bound(covered), places=12)
                for n in (0, 1, 3):
                    self.assertEqual(silence.paired_upper(n, supply(spans), delta), silence.paired_upper(n, spans, delta))
                self.assertEqual(silence.merged(supply(spans)), silence.merged(spans))

    def test_functions_that_read_an_argument_twice(self):
        # Found by going through every function of the listings and of the package for an argument that is read
        # more than once. With a generator each of these returned another number in Version 20, and none failed.
        step = omission.INTERVAL / omission.MS            # in ms; the listing counts nanoseconds since Version 46
        xs = [1.0, 2.5 * step, 0.5 * step, 4.0 * step]
        want = sorted(xs + [1.5 * step, 0.5 * step] + [3.0 * step, 2.0 * step, 1.0 * step])   # what a stalled tester missed
        self.assertEqual(omission.back_fill(xs), want)
        self.assertEqual(omission.back_fill(once(xs)), want)
        self.assertEqual(len(want), 9)

        ref, cur = [10, 20, 30], [12, 18, 30]
        bins = len(ref)                                 # Version 21 took ten, whatever the lists (tests/test_review21.py)
        p = [(r + 0.5) / (60 + 0.5 * bins) for r in ref]
        q = [(c + 0.5) / (60 + 0.5 * bins) for c in cur]
        index = sum((b - a) * math.log(b / a) for a, b in zip(p, q))
        for first, second in ((ref, cur), (once(ref), cur), (ref, once(cur)), (once(ref), once(cur))):
            self.assertAlmostEqual(drift.psi(first, second), index, places=12)
        self.assertGreater(index, 0)

        us, dofs = [1.0, 2.0], [5, 7]
        dof = (1 + 4) ** 2 / (1 / 5 + 16 / 7)                                       # Welch-Satterthwaite, by hand
        for first, second in ((us, dofs), (once(us), dofs), (us, once(dofs)), (once(us), once(dofs))):
            self.assertAlmostEqual(core.welch_satterthwaite(first, second), dof, places=12)

        ts, ys = [1, 2, 3, 4], [2.1, 3.9, 6.2, 7.8]
        slope = (sum(t * y for t, y in zip(ts, ys)) - 4 * 2.5 * st.fmean(ys)) / (sum(t * t for t in ts) - 4 * 2.5 ** 2)
        for first, second in ((ts, ys), (ts, once(ys)), (once(ts), ys), (once(ts), once(ys))):
            fit = stats.linfit(first, second)
            self.assertAlmostEqual(fit[1], slope, places=12)
            self.assertEqual(fit, stats.linfit(ts, ys))
        self.assertAlmostEqual(slope, 1.94, places=12)

        samplers = [propagate.normal(1, 0.1), propagate.normal(2, 0.1)]
        self.assertEqual(propagate.monte_carlo(sum, once(samplers), n=50), propagate.monte_carlo(sum, samplers, n=50))
        self.assertGreater(min(propagate.monte_carlo(sum, once(samplers), n=50)), 2)

        pairs = [(1.0, 1), (3.0, 3)]
        self.assertEqual(mix.mean(once(pairs)), 2.5)
        self.assertEqual(mix.p95(once(pairs)), 3.0)

        rows = lines(TIED, 2)
        found = trace.runs(rows, 30, 60, HALF)
        self.assertEqual(silence.inside(rows, "pool", once(found)), 2)              # lines 1 and 9
        self.assertEqual(silence.inside(once(rows), "pool", found), 2)
        self.assertEqual(silence.stamped(once(rows), "svc", once(found), 30, 60, HALF), (0, 1))
        self.assertEqual(trace.runs(once(rows), 30, 60, HALF), found)
        self.assertEqual(trace.verified(once(rows), 30, 60, HALF), 20.0)
        self.assertEqual(trace.count(once(rows), "pool", 30, 60, HALF), 3)

    def test_the_printing_functions_take_a_generator(self):
        rows = fixture()
        self.assertEqual(printed(silence.models, once(rows), SPAN), printed(silence.models, rows, SPAN))
        self.assertEqual(printed(silence.spacing, once(rows), SPAN), printed(silence.spacing, rows, SPAN))
        self.assertTrue(expected("ch16_silence").startswith(printed(silence.models, once(rows), SPAN)))
        self.assertIn("svc, 30-60 s: 0 messages, 30 s verified\n", printed(silence.models, once(rows), SPAN))

    def test_what_is_left_fails_and_does_not_answer(self):
        # Functions that ask an argument for its length, or for an item by position, stop on a generator.
        for call in (lambda: core.type_a(once([1.0, 2.0, 4.0])),
                     lambda: stats.bootstrap(once([1.0, 2.0, 3.0]), st.fmean, 10)):
            with self.assertRaises((TypeError, AttributeError, IndexError)):
                call()
        # completeness() stopped too, until Version 53: it keeps its lines in a list since it reads them twice. Numbers 1, 2, 5,
        # 9, 10: five of ten are missing, and the widest step, from 5 to 9, spans the stamps 40 s and 50 s.
        self.assertEqual(trace.completeness(once(lines(TIED))), (5, 10, 10000))
        # Two that stopped in Version 21 keep their arguments in lists since they check that the arguments pair up.
        self.assertEqual(stats.cohen_kappa([1, 0, 1, 1], once([1, 0, 0, 1])), 0.5)
        self.assertEqual(propagate.lpu(lambda x: x[0] * x[1], once([2.0, 3.0]), [0.1, 0.2]),
                         propagate.lpu(lambda x: x[0] * x[1], [2.0, 3.0], [0.1, 0.2]))


class StampsAlone(unittest.TestCase):
    def test_the_rows_of_the_review(self):
        found = trace.runs(lines(LONE), 30, 60, HALF)
        self.assertEqual(found, [(30_000, 40_000, [1, 2]), (50_000, 60_000, [5])])
        rows = lines(LONE, 2)
        self.assertEqual(silence.inside(rows, "svc", found), 1)
        self.assertEqual(silence.stamped(rows, "svc", found, 30, 60, HALF), (1, 1))  # Version 20: (0, 1)
        self.assertEqual(rows, silence.edge(2, silence.LONE))
        # Each line in turn: the stamps settle every one, and the line at 60 s is outside the window.
        for own, count in ((1, 1), (2, 1), (5, 1), (6, 0)):
            rows = lines(LONE, own)
            self.assertEqual(silence.inside(rows, "svc", found), count, own)
            self.assertEqual(silence.stamped(rows, "svc", found, 30, 60, HALF), (count, count), own)

    def test_two_lines_of_one_stamp_stay_open(self):
        self.assertEqual(silence.TIED, TIED)
        found = trace.runs(lines(TIED), 30, 60, HALF)
        for own, count, want in ((1, 1, (1, 1)), (2, 1, (0, 1)), (5, 0, (0, 1)), (9, 1, (1, 1)), (10, 0, (0, 0))):
            rows = lines(TIED, own)
            self.assertEqual(silence.inside(rows, "svc", found), count, own)
            self.assertEqual(silence.stamped(rows, "svc", found, 30, 60, HALF), want, own)
        # The two records with the service's line as 2 and as 5 have the same stamps and components, and counts of
        # 1 and 0: no rule that sees the stamps alone can give less than the range from 0 to 1.
        stamps = [sorted((ms, c) for _, ms, c, _ in lines(TIED, own)) for own in (2, 5)]
        self.assertEqual(stamps[0], stamps[1])
        # Both lines of the stamp from the service: one of them ends the run, whichever it is.
        both = [(n, ms, "svc" if n in (2, 5) else "pool", m) for n, ms, _, m in lines(TIED)]
        self.assertEqual((silence.inside(both, "svc", found), silence.stamped(both, "svc", found, 30, 60, HALF)), (1, (1, 2)))

    def test_where_two_runs_meet(self):
        # Two lines at the stamp at which one run ends and the next begins: both are lines of runs.
        rows = lines(((1, 30), (2, 40), (5, 40), (6, 50)), 5)
        found = trace.runs(rows, 30, 60, HALF)
        self.assertEqual(found, [(30_000, 40_000, [1, 2]), (40_000, 50_000, [5, 6])])
        self.assertEqual((silence.inside(rows, "svc", found), silence.stamped(rows, "svc", found, 30, 60, HALF)), (1, (1, 1)))
        # A third line of that stamp, alone between the two losses: any one of the three may be it.
        for own, count in ((2, 1), (5, 0), (8, 1)):
            rows = lines(((1, 30), (2, 40), (5, 40), (8, 40), (9, 50)), own)
            found = trace.runs(rows, 30, 60, HALF)
            self.assertEqual(found, [(30_000, 40_000, [1, 2]), (40_000, 50_000, [8, 9])])
            self.assertEqual(silence.inside(rows, "svc", found), count)
            self.assertEqual(silence.stamped(rows, "svc", found, 30, 60, HALF), (0, 1))

    def test_at_the_edges_of_the_window(self):
        # A run that began before the window: the line stamped where the window opens is a line of the run.
        rows = lines(((1, 20), (2, 30), (3, 40)), 2)
        found = trace.runs(rows, 30, 60, HALF)
        self.assertEqual(found, [(30_000, 40_000, [2, 3])])
        self.assertEqual((silence.inside(rows, "svc", found), silence.stamped(rows, "svc", found, 30, 60, HALF)), (1, (1, 1)))
        # A line alone before a loss, stamped like the first line of the run that follows.
        for own, count in ((1, 0), (5, 1)):
            rows = lines(((1, 30), (5, 30), (6, 40)), own)
            found = trace.runs(rows, 30, 60, HALF)
            self.assertEqual(found, [(30_000, 40_000, [5, 6])])
            self.assertEqual((silence.inside(rows, "svc", found), silence.stamped(rows, "svc", found, 30, 60, HALF)), (count, (0, 1)))
        # Verified by counter readings from the start of the capture: the first line is a line of the run.
        rows = lines(((1, 0), (2, 10)), 1)
        found = trace.runs(rows, 0, 60, HALF, (1, 3))
        self.assertEqual(found, [(0, 60_000, [1, 2])])
        self.assertEqual((silence.inside(rows, "svc", found), silence.stamped(rows, "svc", found, 0, 60, HALF)), (1, (1, 1)))
        # A line stamped where the window closes is not in it, whatever ends there.
        rows = lines(((1, 30), (2, 60)), 2)
        found = trace.runs(rows, 30, 60, HALF)
        self.assertEqual((silence.inside(rows, "svc", found), silence.stamped(rows, "svc", found, 30, 60, HALF)), (0, (0, 0)))
        # A window that leaves the capture is refused, as everywhere.
        with self.assertRaisesRegex(ValueError, "outside the capture"):
            silence.stamped(rows, "svc", found, 30, 90, HALF)

    def test_distinct_stamps_give_the_count(self):
        rng = random.Random(2020)
        counted = 0
        for _ in range(800):
            rows = record(rng, ties=False)
            self.assertEqual(len({ms for _, ms, _, _ in rows}), len(rows))
            lo = rng.choice((0, 5, 10, 20, 30))
            hi = min(60, lo + rng.choice((5, 10, 20, 30, 60)))
            for counter in (None, (rows[0][0], rows[-1][0] + 1), (rows[0][0] - 2, rows[-1][0] + 3)):
                found = trace.runs(rows, lo, hi, HALF, counter)
                for comp in ("svc", "db", "pool"):
                    count = silence.inside(rows, comp, found)
                    self.assertEqual(silence.stamped(rows, comp, found, lo, hi, HALF), (count, count))
                    counted += count
        self.assertGreater(counted, 10_000)

    def test_the_range_holds_the_count_and_is_no_wider_than_before(self):
        rng = random.Random(2021)
        narrower = open_ends = 0
        for _ in range(1500):
            rows = record(rng, ties=True)
            lo = rng.choice((0, 5, 10, 20, 30))
            hi = min(60, lo + rng.choice((5, 10, 20, 30, 60)))
            for counter in (None, (rows[0][0], rows[-1][0] + 1)):
                found = trace.runs(rows, lo, hi, HALF, counter)
                held = [r for r in rows if 1000 * lo <= r[1] < 1000 * hi]
                for comp in ("svc", "db", "pool"):
                    least, most = silence.stamped(rows, comp, found, lo, hi, HALF)
                    count = silence.inside(rows, comp, found)
                    self.assertLessEqual(least, count)
                    self.assertLessEqual(count, most)
                    # The rule of Version 20: inside a run for certain, at an end in doubt.
                    old_least = sum(c == comp and any(x < ms < y for x, y, _ in found) for _, ms, c, _ in held)
                    old_most = sum(c == comp and any(x <= ms <= y for x, y, _ in found) for _, ms, c, _ in held)
                    self.assertLessEqual(old_least, least)
                    self.assertEqual(most, old_most)
                    narrower += least > old_least
                    open_ends += least < most
        self.assertGreater(narrower, 2000)
        self.assertGreater(open_ends, 500)                    # ties remain, and with them the doubt

    def test_the_fixture(self):
        rows = fixture()
        for lo, hi in ((0, 30), (30, 60)):
            found = trace.runs(rows, lo, hi, SPAN, trace.COUNTER)
            for comp in ("svc", "db"):
                count = silence.inside(rows, comp, found)
                least, most = silence.stamped(rows, comp, found, lo, hi, SPAN)
                self.assertLessEqual(least, count)
                self.assertLessEqual(count, most)
            self.assertEqual(silence.stamped(rows, "svc", found, lo, hi, SPAN), ((2, 2), (0, 0))[lo == 30])

    def test_the_transcript(self):
        text = expected("ch16_silence")
        self.assertIn("lines 1, 2, 5, 9 and 10, stamped\n30, 40, 40, 50 and 60 s:\n  verified time has lines 1 to 2 and 9\n"
                      "  the line of the service, counted\n  by number, and by stamp alone:\n    line 2: 1, and 0 to 1\n"
                      "    line 5: 0, and 0 to 1\n    line 10: 0, and 0\n"
                      "lines 1, 2, 5 and 6, stamped\n30, 40, 50 and 60 s:\n  verified time has lines 1 to 2 and 5\n"
                      "  the line of the service, counted\n  by number, and by stamp alone:\n    line 2: 1, and 1\n", text)
        self.assertIn(printed(silence.counted), text)
        self.assertEqual(len(text.splitlines()), 193)            # 88 in Version 21, 101, 118, 140, 152, 162, 166 in Version 27


class PercentileWithAMissingClass(unittest.TestCase):
    def test_the_records_of_the_review(self):
        common = BEFORE["common"]
        weight, gap = Fraction(99, 100), Fraction(1, 100)     # the later mix: 99 common requests and one bulk request
        below, upto = shares(common, weight, 100.0)
        self.assertEqual((below + gap, upto), (Fraction(93, 100), Fraction(98, 100)))
        self.assertEqual(settled_p95(common, weight, gap), 100.0)
        # The later percentile is 90 ms at either mix, by rank.
        self.assertEqual(sorted(AFTER["common"] + AFTER["bulk"])[94], 90.0)          # rank 95 of 100
        self.assertEqual(sorted(AFTER["common"])[math.ceil(Fraction(95, 100) * 99) - 1], 90.0)
        self.assertEqual(sorted(common)[math.ceil(Fraction(95, 100) * 99) - 1], 100.0)
        out = printed(mix.components, BEFORE, AFTER, mix.p95, "p95", "distributions")
        self.assertEqual(out, "p95, 100 -> 90 ms: -10 ms\n  class distributions first, via 90 ms:\n    within -10, mix +0\n"
                              "  mix first, via 100 ms:\n    within -10, mix +0\n  average of the two orders:\n"
                              "    within -10, mix +0\n")                           # Version 20: mix first not determined

    def test_the_mean_stays_open(self):
        start, at_old_mix, end = Fraction(8140, 99), Fraction(7140, 99), Fraction(7640, 100)
        self.assertEqual(sum(BEFORE["common"]), 8140)
        self.assertEqual((sum(AFTER["common"]), sum(AFTER["common"]) + 500), (7140, 7640))
        out = printed(mix.components, BEFORE, AFTER)
        self.assertEqual(out, f"mean, {float(start):.0f} -> {float(end):.0f} ms: {float(end - start):+.0f} ms\n"
                              f"  class means first, via {float(at_old_mix):.0f} ms:\n"
                              f"    within {float(at_old_mix - start):+.0f}, mix {float(end - at_old_mix):+.0f}\n"
                              "  mix first: not determined,\n    no bulk requests before\n")
        self.assertEqual(out.splitlines()[0], "mean, 82 -> 76 ms: -6 ms")
        self.assertEqual(out.splitlines()[2], "    within -10, mix +4")
        low = mix.mean(mix.weighted(BEFORE, AFTER, fill=-math.inf))
        top = mix.mean(mix.weighted(BEFORE, AFTER, fill=math.inf))
        self.assertEqual((low, top), (-math.inf, math.inf))
        # Completions with a bulk request of 0 ms and of 1000 ms move the mean by ten milliseconds.
        means = [float(mix.mean(mix.weighted(dict(BEFORE, bulk=[d]), AFTER))) for d in (0.0, 1000.0)]
        self.assertAlmostEqual(means[1] - means[0], 10.0, places=9)

    def test_every_completion_gives_the_same_percentile(self):
        for d in (0.0, 0.001, 79.9, 80.0, 99.99, 100.0, 100.01, 180.0, 500.0, 1e9):
            done = dict(BEFORE, bulk=[d])
            self.assertEqual(mix.p95(mix.weighted(done, AFTER)), 100.0, d)
        self.assertEqual(mix.p95(mix.weighted(BEFORE, AFTER, fill=-math.inf)), 100.0)
        self.assertEqual(mix.p95(mix.weighted(BEFORE, AFTER, fill=math.inf)), 100.0)
        # With more weight on the missing class the same durations no longer settle it.
        heavy = {"common": range(90), "bulk": range(10)}
        ends = [mix.p95(mix.weighted(BEFORE, heavy, fill=x)) for x in (-math.inf, math.inf)]
        self.assertEqual(ends, [100.0, math.inf])             # 10 % missing: the percentile may lie in the missing class
        self.assertIsNone(settled_p95(BEFORE["common"], Fraction(90, 100), Fraction(10, 100)))

    def test_the_criterion_of_the_review_on_random_records(self):
        rng = random.Random(2023)
        settled = left_open = 0
        for _ in range(600):
            n = rng.choice((20, 50, 99, 100))
            xs = [float(rng.choice((50, 60, 70, 80, 90, 100, 150, 400))) for _ in range(n)]
            missing = rng.choice((1, 2, 3, 5, 10))            # requests of the missing class in a mix of 100
            ref = {"common": range(100 - missing), "bulk": range(missing)}
            rec = {"common": xs}
            low = mix.p95(mix.weighted(rec, ref, fill=-math.inf))
            top = mix.p95(mix.weighted(rec, ref, fill=math.inf))
            want = settled_p95(xs, Fraction(100 - missing, 100), Fraction(missing, 100))
            self.assertEqual(low == top, want is not None, (xs, missing))
            if want is not None:
                self.assertEqual(low, want)
                settled += 1
            else:
                self.assertLess(low, top)
                left_open += 1
            # Whatever the missing durations were, the percentile lies between the two.
            for d in (0.0, 65.0, 95.0, 1000.0):
                done = mix.p95(mix.weighted(dict(rec, bulk=[d] * 3), ref))
                self.assertLessEqual(low, done)
                self.assertLessEqual(done, top)
            out = printed(mix.components, rec, dict(rec, bulk=[500.0] * 2), mix.p95, "p95", "distributions")
            self.assertIn("class distributions first, via", out)
        self.assertGreater(settled, 100)
        self.assertGreater(left_open, 100)

    def test_a_class_at_a_given_duration(self):
        before, after = periods()
        more = dict(after, bulk=[1000.0] * 100)
        pairs = mix.weighted(before, more, fill=7.0)
        self.assertEqual(pairs[-1], (7.0, Fraction(100)))
        self.assertEqual(sum(w for _, w in pairs), 1100)       # the whole weight of the mix
        self.assertEqual(len(pairs), 1001)
        self.assertEqual(mix.weighted(before, before, fill=7.0), mix.weighted(before, before))
        self.assertEqual(mix.weighted(before, fill=7.0), mix.weighted(before))
        with self.assertRaisesRegex(ValueError, "no observations for bulk"):
            mix.weighted(before, more)
        with self.assertRaisesRegex(ValueError, "empty reference mix"):
            mix.weighted(before, {}, fill=7.0)
        alone = mix.weighted(before, more, partial=True)
        self.assertEqual(sum(w for _, w in alone), 1000)

    def test_the_examples_of_the_book_are_as_they_were(self):
        before, after = periods()
        more = dict(after, bulk=[1000.0] * 100)
        # A hundred bulk requests in 1100: the percentile of the first period at that mix is open as well.
        ends = [mix.p95(mix.weighted(before, more, fill=x)) for x in (-math.inf, math.inf)]
        self.assertLess(ends[0], ends[1])
        self.assertEqual(ends[1], math.inf)
        out = printed(mix.components, before, more, mix.p95, "p95", "distributions")
        self.assertTrue(out.endswith("  mix first: not determined,\n    no bulk requests before\n"))
        out = printed(mix.components, before, more)
        self.assertEqual(out, "mean, 140 -> 467 ms: +327 ms\n  class means first, via 126 ms:\n    within -14, mix +341\n"
                              "  mix first: not determined,\n    no bulk requests before\n")
        out = printed(mix.components, before, after)
        self.assertEqual([line for line in out.splitlines() if "within" in line],
                         ["    within -14, mix +288", "    within -46, mix +320", "    within -30, mix +304"])

    def test_the_transcript(self):
        text = expected("ch23_mix")
        out = printed(mix.one_bulk)
        self.assertTrue(text.endswith(out))
        self.assertTrue(out.startswith("99 common requests in each period,\none bulk request of 500 ms after:\nmean, 82 -> 76 ms: -6 ms\n"))
        self.assertIn("p95, 100 -> 90 ms: -10 ms\n", out)
        self.assertIn("  mix first, via 100 ms:\n    within -10, mix +0\n", out)
        self.assertEqual(len(text.splitlines()), 103)            # 69 in Version 21, 73 until 48, 84 in 49, 96 in 50, 100 in 51


class BothPeriodsAtOneMix(unittest.TestCase):
    def test_the_figures_of_the_review(self):
        before, after = periods()
        cells = {slow: (replicated_p95(before, 1000 - slow, slow), replicated_p95(after, 1000 - slow, slow)) for slow in (40, 41)}
        self.assertEqual({k: tuple(f"{x:.2f}" for x in v) for k, v in cells.items()},
                         {40: ("190.98", "159.62"), 41: ("194.92", "216.25")})
        for slow, (b, a) in cells.items():
            self.assertEqual((mix.at_share(before, slow), mix.at_share(after, slow)), (b, a))
        self.assertLess(cells[40][1], cells[40][0])           # at 4.0 % the second period is the lower
        self.assertGreater(cells[41][1], cells[41][0])        # at 4.1 % the higher
        # And the percentile of each class fell.
        for name in before:
            self.assertLess(mix.class_p95(after[name]), mix.class_p95(before[name]))

    def test_where_the_percentile_lies_in_the_fast_class(self):
        before, after = periods()
        fast_b, fast_a = sorted(before["fast"]), sorted(after["fast"])
        self.assertEqual((len(fast_b), len(fast_a)), (900, 100))
        self.assertEqual(mix.at_share(after, 41), fast_a[-1])          # the largest of 100
        self.assertEqual(mix.at_share(after, 40), fast_a[-2])          # the next largest
        self.assertEqual(mix.at_share(before, 41), fast_b[891])        # the 892nd of 900
        self.assertEqual((f"{fast_a[-2]:.2f}", f"{fast_a[-1]:.2f}"), ("159.62", "216.25"))
        # From the weights: 99 fast requests of the second period carry 99 x 0.959 % = 94.94 %, short of 95 %.
        self.assertLess(Fraction(99, 100) * Fraction(959, 1000), Fraction(95, 100))
        self.assertGreaterEqual(Fraction(99, 100) * Fraction(960, 1000), Fraction(95, 100))
        self.assertEqual(sum(x <= fast_a[-1] for x in after["slow"]), 12)   # and twelve slow requests lie below the largest

    def test_where_the_order_turns(self):
        before, after = periods()
        higher = [k for k in range(3800, 4801, 5) if mix.at_share(after, k, 100_000) > mix.at_share(before, k, 100_000)]
        self.assertEqual(higher, list(range(higher[0], higher[-1] + 1, 5)))          # one stretch of mixes
        self.assertEqual((higher[0], higher[-1]), (4070, 4680))
        self.assertEqual((f"{mix.at_share(before, 47):.2f}", f"{mix.at_share(after, 47):.2f}"), ("220.15", "216.25"))
        self.assertEqual(mix.at_share(before, 41), mix.at_share(before, 4100, 100_000))
        # At the mixes that the periods had, the second period has the lower percentile.
        self.assertEqual([f"{mix.at_share(rec, slow):.2f}" for slow in (100, 900) for rec in (before, after)],
                         ["469.09", "432.40", "805.85", "673.92"])
        self.assertEqual(mix.at_share(before, 100), mix.p95(mix.weighted(before)))
        self.assertEqual(mix.at_share(after, 900), mix.p95(mix.weighted(after)))

    def test_the_transcript(self):
        before, after = periods()
        out = printed(mix.one_mix, before, after)
        self.assertTrue(out.startswith("p95 of both periods at one mix,\nby its share of slow requests:\n  4.0 %: 190.98 -> 159.62 ms\n"
                                       "  4.1 %: 194.92 -> 216.25 ms\n  4.7 %: 220.15 -> 216.25 ms\nthe two largest of the 100 fast\n"
                                       "requests after: 159.62, 216.25 ms\n"))        # all of it in Version 21
        self.assertIn(out, expected("ch23_mix"))

    def test_the_figure_is_drawn_from_the_listing(self):
        source = (ROOT / "figures" / "make_figures.py").read_text(encoding="utf-8")
        for piece in ("ys = [c23.at_share(rec, k, 100_000) for k in parts]", "y = c23.at_share(rec, 41)",
                      'save(fig, "fig-23-1-one-mix.png")', "before = c23.record(c23.BEFORE, rng)"):
            self.assertIn(piece, source)
        self.assertEqual(source.count("\n    save(fig, "), 32)


if __name__ == "__main__":
    unittest.main()
