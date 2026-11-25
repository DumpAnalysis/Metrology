"""Regression tests added after the review of Version 25.

The review found two calculations that gave another answer when the same
short interval was moved to a clock origin of nineteen digits: the law of
propagation, which formed its trial points in floating point and returned an
uncertainty of zero for a duration of 100 ns between two stamps read to
1 ns, and the start times of paired messages, which shifted the stretches by
a float duration and returned a negative length. It asked for the search
through ten windows to be worked for requests that write two messages, whose
windows are not independent; proposed a worked case in which a factor shared
by two durations is propagated as two correlated inputs and as one; and
found one double rounding left in the root of a fraction, at the smallest
float.

Each test computes what it expects independently of the code under test:
closed forms, exact fractions, a second calculation of the search over
disjoint start times, the float neighbors of a root, and seeded simulations.
"""
import contextlib
import io
import itertools
import math
import random
import unittest
from fractions import Fraction
from pathlib import Path

from examples import ch05_propagation as chapter5
from examples import ch16_silence as silence
from swmetro import Result, core, propagate, stats

ROOT = Path(__file__).resolve().parent.parent
EPOCH = 1_700_000_000_000_000_000                # ns from 1970 to 14 November 2023, 22:13:20 UTC
TINY = math.ulp(0.0)                              # the smallest float above zero, 5e-324


def expected(name):
    return (ROOT / "expected" / f"{name}.txt").read_text(encoding="utf-8")


def printed(function, *args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        function(*args)
    return out.getvalue()


def duration(t):
    return t[1] - t[0]


def version_25_lpu(f, xs, us):
    """The contributions as Version 25 formed them: trial points in the arithmetic of the inputs."""
    cu = []
    for i, u in enumerate(us):
        hi, lo = list(xs), list(xs)
        hi[i] += u
        lo[i] -= u
        cu.append((f(hi) - f(lo)) / 2)
    return math.sqrt(sum(c * c for c in cu)), cu


class AnIntervalAtAClockOrigin(unittest.TestCase):
    def test_the_table_of_the_review(self):
        stamps = [EPOCH, EPOCH + 100]
        self.assertEqual(version_25_lpu(duration, stamps, [1.0, 1.0]), (0.0, [0.0, 0.0]))   # what Version 25 returned
        for xs, us in ((stamps, [1.0, 1.0]), (stamps, [1, 1]), ([0, 100], [1.0, 1.0])):
            y, uc, cu = propagate.lpu(duration, xs, us)
            self.assertEqual((y, cu), (100, [-1.0, 1.0]))
            self.assertEqual(uc, math.sqrt(2))
        self.assertEqual(f"{math.sqrt(2):.5f}", "1.41421")

    def test_a_fraction_of_a_nanosecond_and_stamps_in_seconds(self):
        y, uc, cu = propagate.lpu(duration, [EPOCH, EPOCH + 100], [0.5, 0.25])
        self.assertEqual((y, cu), (100, [-0.5, 0.25]))
        self.assertAlmostEqual(uc, math.hypot(0.5, 0.25), places=15)
        # Seconds since 1970 as floats, 238 ns apart at this size, read to 1 ns.
        start = 1.7e9
        end = start + 4096 * math.ulp(start)
        y, uc, cu = propagate.lpu(duration, [start, end], [1e-9, 1e-9])
        self.assertEqual(cu, [-1e-9, 1e-9])
        self.assertEqual(version_25_lpu(duration, [start, end], [1e-9, 1e-9])[0], 0.0)

    def test_the_origin_does_not_change_the_contributions(self):
        rng = random.Random(2501)
        for _ in range(200):
            length = rng.randint(1, 10 ** 6)
            us = [rng.choice((1, 0.5, 2.5, 1e-3, 40.0)), rng.choice((1, 0.5, 2.5, 1e-3, 40.0))]
            near = propagate.lpu(duration, [0, length], us)
            for origin in (1_000, 10 ** 12, EPOCH, 2 ** 62, -EPOCH):
                far = propagate.lpu(duration, [origin, origin + length], us)
                self.assertEqual(far, near, (origin, length, us))
            self.assertEqual(near[2], [-us[0], us[1]])

    def test_the_trial_points_are_fractions(self):
        seen = []

        def spy(x):
            seen.append([type(v) for v in x])
            return x[0] * x[1]
        y, uc, cu = propagate.lpu(spy, [3, 0.5], [0.25, 0.125])
        self.assertEqual(seen[0], [int, float])               # the estimates as they were given
        self.assertTrue(all(kinds == [Fraction, Fraction] for kinds in seen[1:]))
        self.assertEqual((y, cu), (1.5, [0.125, 0.375]))      # a product: the other factor times u, exactly
        for bad in (math.nan, math.inf):
            with self.assertRaises((ValueError, OverflowError)):
                propagate.lpu(duration, [0, bad], [1, 1])
        with self.assertRaisesRegex(ValueError, "one u for each x"):
            propagate.lpu(duration, [0, 1], [1])

    def test_ordinary_results_keep_their_numbers(self):
        rng = random.Random(2502)
        for _ in range(300):
            xs = [rng.uniform(1, 1000), rng.uniform(1, 1000)]
            us = [x * rng.uniform(0.001, 0.05) for x in xs]
            for f in (chapter5.speedup, lambda x: x[0] * x[1] + x[0], lambda x: math.log(x[0]) - math.sqrt(x[1])):
                uc, cu = version_25_lpu(f, xs, us)
                got = propagate.lpu(f, xs, us)
                self.assertAlmostEqual(got[1] / uc, 1.0, places=9)
                for a, b in zip(got[2], cu):
                    self.assertAlmostEqual(a / b, 1.0, places=9)
        text = expected("ch05_propagation")
        self.assertTrue(text.startswith("r = 0.0  S = 1.2855  u_c = 0.0347\nr = 0.6  S = 1.2855  u_c = 0.0236\n"
                                        "c*u(T_old) = +0.0183  c*u(T_new) = -0.0295\n"))
        self.assertIn("LPU    R = 500  u_c = 122\n", text)
        self.assertIn("  observed 0.593, corrected 0.592, u 0.024\n", expected("ch20_judge"))


def version_25_by_messages(spans, delta):
    """The two lengths as Version 25 formed them: the stretches moved back in the arithmetic of their ends."""
    def merged(spans):
        out = []
        for x, y in sorted(spans):
            if out and x <= out[-1][1]:
                x, top = out.pop()
                y = max(y, top)
            out.append((x, y))
        return out
    spans = merged(spans)
    seen = sum(y - x for x, y in spans)
    back = [(x - delta, y - delta) for x, y in spans]
    union = sum(y - x for x, y in merged(spans + back))
    return 2 * seen - union, 2 * (union - seen)


class ExposureAtAClockOrigin(unittest.TestCase):
    def test_the_case_of_the_review(self):
        far = [(EPOCH, EPOCH + 300)]
        self.assertEqual(version_25_by_messages(far, 10.0), (344.0, -88.0))        # what Version 25 returned
        for spans, delta in ((far, 10.0), ([(0, 300)], 10.0), (far, 10)):
            self.assertEqual(silence.by_messages(spans, delta), (290, 20))
        # Counted: both messages inside for starts in [0, 290), exactly one for [-10, 0) and [290, 300).
        both = sum(0 <= t < 300 and 0 <= t + 10 < 300 for t in range(-20, 320))
        one = sum((0 <= t < 300) != (0 <= t + 10 < 300) for t in range(-20, 320))
        self.assertEqual((both, one), (290, 20))

    def test_the_whole_window_translated(self):
        rng = random.Random(2503)
        for _ in range(300):
            cuts = sorted(rng.sample(range(0, 2000), 2 * rng.randint(1, 3)))
            spans = list(zip(cuts[::2], cuts[1::2]))
            delta = rng.choice((0, 1, 10, 2.5, 10.0, 0.125, 137, 2346.0, 5000))
            near = silence.by_messages(spans, delta)
            self.assertTrue(near[0] >= 0 and near[1] >= 0, (spans, delta))
            self.assertEqual(near[0] + near[1], silence.reach(spans, delta))
            shares = [(0.5, delta), (0.5, 3 * delta + 1)]
            for origin in (1_000, 10 ** 12, EPOCH, 2 ** 62, -EPOCH):
                moved = [(origin + x, origin + y) for x, y in spans]
                self.assertEqual(silence.by_messages(moved, delta), near, (origin, spans, delta))
                self.assertEqual(silence.reach(moved, delta), silence.reach(spans, delta))
                self.assertEqual(silence.mean_reach(moved, shares), silence.mean_reach(spans, shares))
                self.assertEqual(silence.paired_bound(moved, delta), silence.paired_bound(spans, delta))
                for unit in ("message", "request"):
                    self.assertEqual(silence.kept_starts(moved, delta, 0.25, unit), silence.kept_starts(spans, delta, 0.25, unit))
        whole, far = [(0, 300)], [(EPOCH, EPOCH + 300)]
        self.assertEqual(silence.paired_upper(1, far, 10.0), silence.paired_upper(1, whole, 10.0))

    def test_exact_times(self):
        found = silence.starts([(EPOCH, EPOCH + 300)], 2.5)
        self.assertEqual(found, [(Fraction(2 * EPOCH - 5, 2), EPOCH + 300)])
        self.assertTrue(all(type(t) is Fraction for pair in found for t in pair))
        self.assertEqual(silence.reach([(EPOCH, EPOCH + 300)], 2.5), 302.5)
        self.assertEqual(silence.by_messages([(EPOCH, EPOCH + 300)], 2.5), (297.5, 5))
        # A length that is whole comes back as an int, and any other as the nearest float.
        self.assertEqual([type(v) for v in (silence.reach([(0, 30)], 10), silence.reach([(0, 30)], 10.0), silence.reach([(0, 30)], 2.346))],
                         [int, int, float])
        self.assertEqual((silence.plain(Fraction(6, 2)), silence.plain(Fraction(1, 3))), (3, 1 / 3))
        self.assertEqual(silence.mean_reach([(0, 30)], [(Fraction(1, 3), 10), (Fraction(2, 3), 40)]), Fraction(160, 3))
        # Windows given as floats are the floats that they are: at this size, 256 ns long.
        self.assertEqual(silence.by_messages([(float(EPOCH), float(EPOCH + 300))], 10), (246, 20))
        # Stretches from an iterator, read once.
        self.assertEqual(silence.by_messages(iter([(0, 20), (10, 30)]), 10), (20, 20))
        self.assertEqual(silence.merged([(10, 30), (0, 20)]), [(0, 30)])
        with self.assertRaisesRegex(ValueError, "empty stretch"):
            silence.merged([(EPOCH, EPOCH)])

    def test_the_lines_of_the_book_are_what_they_were(self):
        text = expected("ch16_silence")
        for piece in ("    starts covered: 32.346 s\n    at most 0.18524 msg/s at 95 %\n", "requests of 60 s, window of 30 s:\n  starts covered: 60 s\n",
                      "  27.912 s, and for pairs 2.346 s\n  apart, starts covered: 32.346 s\n", "  for each message: 6.875 s\n",
                      "    from 40 s and 60 s of start times\n"):
            self.assertIn(piece, text)


def some_empty_by_segments(rate, n, length, delta):
    """A second calculation, for windows that follow one another and requests no longer than a window. The start
    times fall into disjoint segments: those from which both messages reach one window, and those from which
    the start reaches one window and the end the next. Their counts are independent, and the chance that every
    window holds a message follows window by window."""
    r = rate / 2
    q = math.exp(-r * delta)                                 # no request across an edge between two windows
    a = math.exp(-r * (length - delta))                      # no request with both messages in one window
    clear, crossed = q, 1 - q                                # before the first window: the segment that ends in it
    for _ in range(n):
        clear, crossed = q * (crossed + clear * (1 - a)), (clear + crossed) * (1 - q)
    return 1 - (clear + crossed)


class TenWindowsSearched(unittest.TestCase):
    TEN = [(30 * i, 30 * i + 30) for i in range(10)]

    def test_the_table_of_the_review(self):
        singles = (silence.quiet_bound(30), silence.searched_bound(30, 10), silence.quiet_bound(300))
        self.assertEqual([f"{v:.5f}" for v in singles], ["0.09986", "0.17584", "0.00999"])
        pairs = (silence.paired_bound(self.TEN[:1], 10), silence.searched_pairs(self.TEN, 10), silence.paired_bound(self.TEN, 10))
        self.assertEqual([f"{v:.5f}" for v in pairs], ["0.14979", "0.26313", "0.01933"])
        self.assertAlmostEqual(pairs[0], 2 * -math.log(0.05) / 40, places=14)     # 40 s of start times for one window
        self.assertAlmostEqual(pairs[2], 2 * -math.log(0.05) / 310, places=14)    # 310 s for the whole record
        terms = silence.selections(self.TEN, 10)
        self.assertEqual(f"{100 * silence.some_empty(singles[1], terms):.2f}", "25.17")
        self.assertAlmostEqual(silence.some_empty(pairs[1], terms), 0.05, places=12)

    def test_a_selection_of_k_windows_in_c_runs(self):
        terms = silence.selections(self.TEN, 10)
        self.assertEqual(len(terms), 1023)
        want = []
        for k in range(1, 11):
            for some in itertools.combinations(range(10), k):
                runs = sum(1 for i in some if i - 1 not in some)
                want.append(((-1) ** (k + 1), 30 * k + 10 * runs))
        self.assertEqual(terms, want)
        self.assertEqual(sum(sign for sign, _ in terms), 1)   # at a rate of zero every window is empty

    def test_a_second_calculation(self):
        for delta in (10, 2.346, 30, 0):
            for n in (1, 2, 3, 10):
                terms = silence.selections(self.TEN[:n], delta)
                for rate in (0.001, 0.05, 0.17584, 0.26313, 1.0):
                    want = some_empty_by_segments(rate, n, 30, delta)
                    self.assertAlmostEqual(silence.some_empty(rate, terms), want, places=11, msg=(delta, n, rate))
        # Messages that come two at a time, at once: the windows are independent again, at half the rate.
        terms = silence.selections(self.TEN, 0)
        for rate in (0.05, 0.3):
            self.assertAlmostEqual(silence.some_empty(rate, terms), 1 - (1 - math.exp(-rate / 2 * 30)) ** 10, places=11)
        # One window searched is one window chosen.
        for delta in (2.346, 10, 60):
            self.assertAlmostEqual(silence.searched_pairs(self.TEN[:1], delta), silence.paired_bound(self.TEN[:1], delta), places=12)
        # More windows to look through, or longer requests up to a window, weaken the bound further.
        bounds = [silence.searched_pairs(self.TEN[:n], 10) for n in (1, 2, 5, 10)]
        self.assertEqual(bounds, sorted(bounds))
        self.assertGreater(silence.searched_pairs(self.TEN, 10), silence.searched_bound(30, 10))

    def test_simulated_searches(self):
        rng = random.Random(2504)
        for rate, delta, n, share, runs in ((0.26313, 10, 10, 0.05, 40_000), (0.17584, 10, 10, 0.2517, 20_000),
                                            (0.2, 45, 3, None, 20_000)):
            found = 0
            for _ in range(runs):
                hit = [False] * n
                t = -delta + rng.expovariate(rate / 2)
                while t < 30 * n:
                    for at in (t, t + delta):
                        if 0 <= at < 30 * n:
                            hit[int(at // 30)] = True
                    t += rng.expovariate(rate / 2)
                found += not all(hit)
            want = share
            if want is None:                                  # requests longer than a window: the sum of the listing
                want = silence.some_empty(rate, silence.selections(self.TEN[:n], delta))
            self.assertAlmostEqual(found / runs, want, delta=4.5 * math.sqrt(want * (1 - want) / runs), msg=(rate, delta, n))

    def test_refusals_and_the_transcript(self):
        with self.assertRaisesRegex(ValueError, "need 1 to 16 windows"):
            silence.selections([], 10)
        with self.assertRaisesRegex(ValueError, "need 1 to 16 windows"):
            silence.selections([(30 * i, 30 * i + 30) for i in range(17)], 10)
        for level in (0, 1, 1.5):
            with self.assertRaisesRegex(ValueError, "need 1e-9 <= level < 1"):     # "need 0 < level < 1" in Version 26
                silence.searched_pairs(self.TEN, 10, level)
        with self.assertRaisesRegex(ValueError, "need delta >= 0"):
            silence.searched_pairs(self.TEN, -1)
        self.assertEqual(silence.selections(iter(self.TEN[:2]), 10), [(1, 40), (1, 40), (-1, 70)])
        out = printed(silence.search)
        self.assertTrue(out.endswith("the same for requests of 10 s that\nwrite two messages:\n  chosen in advance:\n"
                                     "    at most 0.14979 msg/s at 95 %\n  found by a search of ten:\n"
                                     "    at most 0.26314 msg/s at 95 %\n  all ten empty:\n    at most 0.01933 msg/s at 95 %\n"
                                     "  at 0.17585 msg/s a search of\n  ten finds an empty window: 25.17 %\n"))
        text = expected("ch16_silence")
        self.assertTrue(text.endswith(out))
        self.assertEqual(len(text.splitlines()), 193)        # 152 in Version 25, 162, 166 in Version 27


class AFactorSharedByTwoDurations(unittest.TestCase):
    def test_the_numbers_of_the_review(self):
        y, uc, cu = propagate.lpu(chapter5.speedup, [100, 80], [10, 8], {(0, 1): 1.0})
        self.assertEqual(y, 1.25)
        self.assertEqual((f"{uc:.8f}", f"{cu[0]:+.7f}", f"{cu[1]:+.7f}"), ("0.00126263", "+0.1250000", "-0.1262626"))
        # Half the change over the whole standard uncertainty, by hand.
        self.assertAlmostEqual(cu[0], (110 / 80 - 90 / 80) / 2, places=15)
        self.assertAlmostEqual(cu[1], (100 / 88 - 100 / 72) / 2, places=15)
        # The derivatives: 1 / 80 and -100 / 80 ** 2, each times its u. They cancel.
        analytic = (10 / 80, -100 / 80 ** 2 * 8)
        self.assertEqual(analytic, (0.125, -0.125))
        self.assertEqual(analytic[0] ** 2 + analytic[1] ** 2 + 2 * analytic[0] * analytic[1], 0.0)

    def test_the_factor_as_the_input(self):
        def model(g):
            return chapter5.speedup([100 * g[0], 80 * g[0]])
        for factor, u in ((1.0, 0.1), (1.0, 0.5), (0.97, 0.03), (3, 1)):
            y, uc, cu = propagate.lpu(model, [factor], [u])
            self.assertEqual((uc, cu), (0.0, [0.0]), (factor, u))                  # exactly: the factor cancels
            self.assertAlmostEqual(y, 1.25, places=15)
        for g in (Fraction(11, 10), Fraction(9, 10), Fraction(1, 3)):
            self.assertEqual(model([g]), Fraction(5, 4))

    def test_the_transcript(self):
        out = printed(chapter5.shared_factor)
        self.assertEqual(out, "a factor of 1.0(1) on both durations:\nr = 1.0  S = 1.2500  u_c = 0.0013\n"
                              "c*u(T_old) = +0.1250  c*u(T_new) = -0.1263\nthe factor as the input:  u_c = 0.0000\n")
        text = expected("ch05_propagation")
        self.assertIn("c*u(T_old) = +0.0183  c*u(T_new) = -0.0295\n" + out + "one constant error of the rate of a\n", text)
        self.assertEqual(len(text.splitlines()), 40)         # 15 in Version 25, 19 in 26, 24 in 27, 33 until 39


def nearest(f, x):
    """Whether x is the float nearest to the square root of the fraction f, with a tie going to the even one."""
    below, above = math.nextafter(x, -math.inf), math.nextafter(x, math.inf)
    upper = (Fraction(x) + Fraction(above)) / 2
    even = (Fraction(x) / Fraction(math.ulp(x))).numerator % 2 == 0
    if not (f < upper ** 2 or (f == upper ** 2 and even)):
        return False
    if x == 0:
        return True
    lower = (Fraction(below) + Fraction(x)) / 2
    return lower ** 2 < f or (lower ** 2 == f and even)


class ARootAtTheSmallestFloat(unittest.TestCase):
    def test_the_case_of_the_review(self):
        xs, ys = [0, 0, 1, 1 + 2 ** -26], [TINY, 0, 0, 0]
        fx, fy = [Fraction(x) for x in xs], [Fraction(y) for y in ys]
        mx, my = sum(fx) / 4, sum(fy) / 4
        sxx = sum((x - mx) ** 2 for x in fx)
        b = sum((x - mx) * (y - my) for x, y in zip(fx, fy)) / sxx
        a = my - b * mx
        s2 = sum((y - a - b * x) ** 2 for x, y in zip(fx, fy)) / 2
        self.assertEqual(s2 / Fraction(TINY) ** 2 - Fraction(1, 4), Fraction(1, 72_057_595_111_669_772))
        got = stats.linfit(xs, ys)
        self.assertEqual(got[4], TINY)                       # Version 25: 0.0. Just above half the smallest float.
        self.assertEqual(stats._root(s2), TINY)
        # Since Version 31 the coefficients are not rounded where the line through the floats would lie further
        # from the fitted line than a thousandth of its uncertainty, however the readings are stored; here the
        # floats would be 0.0 and -0.0.
        self.assertEqual(got[:2], (a, b))
        self.assertEqual((float(a), float(b)), (0.0, -0.0))

    def test_ties_on_the_smallest_grid(self):
        tiny = Fraction(TINY)
        for m in range(60):
            square = (Fraction(2 * m + 1, 2) * tiny) ** 2    # the root is half-way between two floats
            even = m if m % 2 == 0 else m + 1
            self.assertEqual(stats._root(square), even * TINY, m)
            self.assertEqual(stats._root(square + Fraction(1, 10 ** 700)), (m + 1) * TINY, m)
            self.assertEqual(stats._root(square - Fraction(1, 10 ** 700)), m * TINY, m)
        self.assertEqual(stats._root(tiny ** 2 / 5), 0.0)
        self.assertEqual(stats._root(tiny ** 2), TINY)

    def test_one_rounding_at_every_size(self):
        rng = random.Random(2505)
        checked = 0
        for _ in range(6000):
            e = rng.choice((-2160, -2150, -2148, -2147, -2146, -2140, -2100, -2060, -2046, -2044, -2040, -1500, -3, 0, 1, 700, 2040))
            f = Fraction(rng.randint(1, 2 ** rng.randint(1, 120)), rng.randint(1, 2 ** rng.randint(1, 60))) * Fraction(2) ** e
            try:
                x = stats._root(f)
            except OverflowError:
                self.assertGreater(f, Fraction(2) ** 2047)
                continue
            checked += 1
            self.assertTrue(nearest(f, x), (f, x))
        self.assertGreater(checked, 5000)
        self.assertEqual((stats._root(Fraction(0)), stats._root(Fraction(9, 4)), stats._root(Fraction(10) ** 600)), (0.0, 1.5, 1e300))
        self.assertEqual(stats._root(Fraction(2)), math.sqrt(2))
        largest = Fraction(math.nextafter(math.inf, 0))
        self.assertEqual(stats._root(largest ** 2), float(largest))
        with self.assertRaises(OverflowError):
            stats._root(largest ** 2 * 2)

    def test_the_other_two_results_of_the_fit(self):
        # The intercept and the slope are fractions turned into floats: one rounding there as well, at any size.
        rng = random.Random(2506)
        for _ in range(4000):
            e = rng.choice((-1080, -1076, -1075, -1074, -1073, -1070, -1040, -1023, -1022, -1021, -500, 0, 900))
            f = Fraction(rng.randint(1, 2 ** rng.randint(1, 120)), rng.randint(1, 2 ** rng.randint(1, 60))) * Fraction(2) ** e
            x = float(f)
            below, above = math.nextafter(x, -math.inf), math.nextafter(x, math.inf)
            self.assertLessEqual(f, (Fraction(x) + Fraction(above)) / 2)
            if x > 0:
                self.assertGreaterEqual(f, (Fraction(below) + Fraction(x)) / 2)


class SpreadsThatAFloatCannotShow(unittest.TestCase):
    """Found while the review was worked: three more places where an uncertainty met a value of nineteen digits
    in floating point."""

    def test_a_simulation_at_the_origin(self):
        with self.assertRaisesRegex(ValueError, "too fine for a float of this size"):
            propagate.normal(EPOCH, 1)                       # Version 25: every draw the same float, mean 0 and sd 0
        with self.assertRaisesRegex(ValueError, "too fine for a float of this size"):
            propagate.rect(EPOCH - 1, EPOCH + 1)
        # With the origin subtracted first, in whole numbers, the draws vary as they should.
        stamps = [EPOCH, EPOCH + 100]
        draws = [propagate.normal(t - stamps[0], 1) for t in stamps]
        ys = propagate.monte_carlo(duration, draws, 40_000, 1)
        mean, sd, _, _ = propagate.summarize(ys)
        self.assertAlmostEqual(mean, 100, delta=0.03)
        self.assertAlmostEqual(sd, math.sqrt(2), delta=0.02)
        self.assertEqual(propagate.lpu(duration, stamps, [1.0, 1.0])[1], math.sqrt(2))

    def test_a_result_and_an_interval(self):
        with self.assertRaisesRegex(ValueError, "too fine for a float of this size"):
            Result(EPOCH + 100, 1.4, "ns")                   # Version 25 printed 1700000000000000000.0(14) ns
        with self.assertRaisesRegex(ValueError, "too fine for a float of this size"):
            core.rect_interval(EPOCH, 0.5)
        delta = Result((EPOCH + 100) - EPOCH, 1.4, "ns")
        self.assertEqual(str(delta), "100.0(14) ns")
        low, high = delta.interval()
        self.assertTrue(low < 100 < high)
        self.assertEqual(core.rect_interval(100, 0.5, 1.0), (99.5, 100.5))
        self.assertEqual(Result(EPOCH, 0, "ns").u, 0)        # an exact value has no spread to lose

    def test_where_the_line_is_drawn(self):
        step = math.ulp(1.0)
        core.resolved(1.0, 1000 * step)
        core.resolved(0.0, 1e-300)
        core.resolved(EPOCH, 256_000)
        core.resolved(EPOCH, 0)
        for at, spread in ((1.0, 999 * step), (EPOCH, 255_999), (1.7e9, 1e-9), (-EPOCH, 1.0)):
            with self.assertRaisesRegex(ValueError, "too fine for a float of this size"):
                core.resolved(at, spread)
        # The results of the book are far from that line.
        self.assertEqual(str(Result(2346.0, core.u_tri(1.0), "ms")), "2346.00(41) ms")
        propagate.normal(842, 12)
        propagate.rect(1.2, 2.8)


class TheTwelfthFigure(unittest.TestCase):
    def test_it_is_drawn_from_the_listing(self):
        source = (ROOT / "figures" / "make_figures.py").read_text(encoding="utf-8")
        # The seventh figure of the chapter since Version 29: the gap anywhere has the fourth, the gap given the count the
        # fifth, and the changing rate the sixth.
        for piece in ("c16s.starts(w, 10)", 'save(fig, "fig-16-7-shared-starts.png")', "    shared_starts()\n"):
            self.assertIn(piece, source)
        self.assertEqual(source.count("\n    save(fig, "), 32)         # eleven until Version 25
        first, second = (silence.starts([w], 10) for w in ((0, 30), (30, 60)))
        self.assertEqual((first, second), ([(-10, 30)], [(20, 60)]))
        # A request that starts at 25 s writes at 25 s and at 35 s: one message in each window.
        self.assertTrue(max(first[0][0], second[0][0]) <= 25 < min(first[0][1], second[0][1]))
        self.assertEqual(min(first[0][1], second[0][1]) - max(first[0][0], second[0][0]), 10)


if __name__ == "__main__":
    unittest.main()
