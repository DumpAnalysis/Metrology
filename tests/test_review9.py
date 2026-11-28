"""Regression tests added after the review of Version 9.

The review checked the five figures and the passages beside them, and found
two refinements to the rounded record: a number at a power of ten was read
with too wide an interval below it, and the decision used what the
components allow without the narrower range that the stated combined
uncertainty leaves.

Each test computes what it expects independently of the code under test:
rounding by the formatting of the standard library, a Student t quantile by
numerical integration of its density, percentiles and sums from their
definitions.
"""
import copy
import json
import math
import random
import re
import statistics as st
import unittest
from pathlib import Path

import run_all
from examples import ch08_aliasing, ch10_coordinated_omission, ch15_control_chart
from examples import ch18_calibration, ch21_psi, ch22_record_check
from swmetro.record import ALL, bounds, combined, decide, guard, spread
from swmetro.stats import quantile

ROOT = Path(__file__).resolve().parent.parent
STALE = ["result.u does not follow from the components"]
ROUNDING = ["rounding decides: wants working values"]


def load(name):
    return json.loads((ROOT / "data" / name).read_text(encoding="utf-8"))


def expected(name):
    return (ROOT / "expected" / f"{name}.txt").read_text(encoding="utf-8")


def step_of(x, n):
    """The increment of the last of n significant digits of x: the place to which a
    report rounds an estimate whose uncertainty is x."""
    if x == 0:
        return 0.0
    return 10.0 ** (int(f"{x:.15e}".split("e")[1]) - n + 1)


def budget(us, stated, value, digits=ALL, limit=100.0, dof=10, risk=0.05, step=None):
    """A complete record with the given components and stated combined u. A rounded
    record states the increment of its estimate, here that of the last digit of u."""
    rec = load("measurement_record.json")
    rec["components"] = [{"name": f"c{i}", "type": "B", "kind": "stipulated", "u": u}
                         for i, u in enumerate(us)]
    rec["result"].update(value=value, u=stated, dof=dof, significant_digits=digits)
    if type(digits) is int and digits > 0:
        rec["result"]["value_step"] = step_of(stated, digits) if step is None else step
    rec["decision"].update(limit=limit, risk=risk)
    return rec


def sig(x, n):
    """x rounded to n significant digits by the standard library's formatting."""
    return float(f"{x:.{n - 1}e}")


def t_upper(tail, dof, steps=4000):
    """Student t quantile with the given upper tail, by Simpson's rule on the
    density and bisection. Independent of swmetro.dist."""
    c = math.exp(math.lgamma((dof + 1) / 2) - math.lgamma(dof / 2)) / math.sqrt(dof * math.pi)

    def area(t):
        h = t / steps
        f = [c * (1 + (i * h) ** 2 / dof) ** (-(dof + 1) / 2) for i in range(steps + 1)]
        return h / 3 * (f[0] + f[-1] + 4 * sum(f[1:-1:2]) + 2 * sum(f[2:-1:2]))

    lo, hi = 0.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if 0.5 - area(mid) > tail else (lo, mid)
    return (lo + hi) / 2


class RoundingBounds(unittest.TestCase):
    def test_the_quantile_used_for_the_expected_values(self):
        # Tabulated one-sided 0.95 values: 1.812 (10), 1.753 (15), 1.746 (16).
        for dof, table in ((10, 1.812), (15, 1.753), (16, 1.746)):
            self.assertAlmostEqual(t_upper(0.05, dof), table, places=3)

    def test_a_power_of_ten_is_not_symmetric(self):
        for x, n, lo, hi in ((10.0, 2, 9.95, 10.5), (10, 2, 9.95, 10.5), (100.0, 2, 99.5, 105.0),
                             (1.0, 1, 0.95, 1.5), (10.0, 1, 9.5, 15.0), (0.1, 2, 0.0995, 0.105),
                             (1e-06, 1, 9.5e-07, 1.5e-06), (0.001, 3, 0.0009995, 0.001005)):
            got = bounds(x, n)
            self.assertAlmostEqual(got[0] / lo, 1.0, places=12, msg=(x, n))
            self.assertAlmostEqual(got[1] / hi, 1.0, places=12, msg=(x, n))
        # On either side of ten the steps differ: 9.9 and 11 are its neighbours at two digits.
        self.assertAlmostEqual(bounds(9.9, 2)[1], 9.95, places=12)
        self.assertAlmostEqual(bounds(11.0, 2)[0], 10.5, places=12)
        self.assertEqual((sig(9.94, 2), sig(9.96, 2), sig(10.4, 2), sig(10.6, 2)), (9.9, 10.0, 10.0, 11.0))

    def test_the_bounds_are_the_midpoints_to_the_neighbours(self):
        rng = random.Random(10)
        for _ in range(4000):
            n = rng.randint(1, 5)
            x = sig(10 ** rng.uniform(-4, 4), n)
            if rng.random() < 0.3:
                x = sig(10.0 ** rng.randint(-4, 4), n)
            lo, hi = bounds(x, n)
            gap = (hi - lo) * 1e-6
            # Just inside the bounds a number rounds to x, just outside it does not.
            self.assertEqual(sig(lo + gap, n), x, (x, n))
            self.assertEqual(sig(hi - gap, n), x, (x, n))
            self.assertNotEqual(sig(lo - gap, n), x, (x, n))
            self.assertNotEqual(sig(hi + gap, n), x, (x, n))

    def test_the_record_of_the_review(self):
        # One component shown as 9.6, the combined uncertainty shown as 10, two digits.
        # 9.6 stands for 9.55 to 9.65, and 10 for 9.95 to 10.5: they cannot both be true.
        self.assertEqual(decide(budget([9.6], 10.0, 60.0, 2)), ("invalid", STALE))
        self.assertIsNone(combined(budget([9.6], 10.0, 60.0, 2)))

    def test_on_both_sides_of_ten(self):
        # Neighbours meet at their common midpoint (9.95, 10.5), which either may claim.
        valid = (([9.9], 9.9), ([10.0], 10.0), ([9.9], 10.0), ([10.0], 11.0), ([11.0], 11.0),
                 ([10.0, 0.50], 11.0))
        for us, stated in valid:
            self.assertEqual(decide(budget(us, stated, 60.0, 2)), ("accept", []), (us, stated))
        stale = (([9.6], 10.0), ([9.4], 10.0), ([9.8], 10.0), ([9.9], 11.0), ([12.0], 10.0), ([10.0], 12.0))
        for us, stated in stale:
            self.assertEqual(decide(budget(us, stated, 60.0, 2)), ("invalid", STALE), (us, stated))
        # The earlier example keeps its place: 10.49 and 0.50 combine to 10.5019.
        self.assertAlmostEqual(math.hypot(10.49, 0.50), 10.5019, places=4)
        self.assertEqual(decide(budget([10.49, 0.50], math.hypot(10.49, 0.50), 60.0)), ("accept", []))


class WholeRecord(unittest.TestCase):
    def setUp(self):
        rec = load("measurement_record.json")
        self.rec = rec
        self.shown = ch22_record_check.displayed(rec)

    def near(self, rec, value):
        return ch22_record_check.altered(rec, {("result", "value"): value})

    def test_the_interval_that_both_statements_leave(self):
        self.assertEqual([c["u"] for c in self.shown["components"]], [1.7, 2.5, 0.3, 0.29])
        lo = math.sqrt(1.65 ** 2 + 2.45 ** 2 + 0.295 ** 2 + 0.285 ** 2)
        hi = math.sqrt(1.75 ** 2 + 2.55 ** 2 + 0.305 ** 2 + 0.295 ** 2)
        self.assertEqual((round(lo, 5), round(hi, 5)), (2.98216, 3.12171))
        got = tuple(map(float, spread(self.shown)))   # decimals since Version 13
        self.assertAlmostEqual(got[0], lo, places=12)
        self.assertAlmostEqual(got[1], hi, places=12)
        # The stated 3.1 stands for 3.05 to 3.15: both hold from 3.05 to the upper bound.
        both = tuple(map(float, combined(self.shown)))
        self.assertAlmostEqual(both[0], 3.05, places=12)
        self.assertAlmostEqual(both[1], hi, places=12)

    def test_the_table_of_the_review(self):
        hi = math.sqrt(1.75 ** 2 + 2.55 ** 2 + 0.305 ** 2 + 0.295 ** 2)
        lo_alone = math.sqrt(1.65 ** 2 + 2.45 ** 2 + 0.295 ** 2 + 0.285 ** 2)
        k_small, k_large = t_upper(0.05, 16.5), t_upper(0.05, 15.5)
        rows = ((194.6, (199.75, 200.11), (199.87, 200.11), ("unresolved", ROUNDING)),
                (194.8, (199.95, 200.31), (200.07, 200.31), ("undecided", [])))
        for value, alone, both, outcome in rows:
            top = value + 0.05 + k_large * hi
            self.assertEqual((round(value - 0.05 + k_small * lo_alone, 2), round(top, 2)), alone)
            self.assertEqual((round(value - 0.05 + k_small * 3.05, 2), round(top, 2)), both)
            rec = self.near(self.shown, value)
            (a_lo, a_hi), (r_lo, r_hi) = (tuple(map(float, pair)) for pair in guard(rec))
            # Since Version 72 the four ends are the numbers that decide() compares with the limit: a computed
            # coverage factor stands one part in a million lower at a lower bound of the band and as much higher
            # at an upper one, the allowance of Chapter 22. Until Version 71 the ends held the factor as it was
            # computed, and the display could show both below a limit at which decide() withheld a decision
            # (tests/test_review71.py). The comparison was to five places and is now to nine.
            less, more = 1 - 1e-6, 1 + 1e-6
            self.assertAlmostEqual(a_lo, value - 0.05 + less * k_small * 3.05, places=9)
            self.assertAlmostEqual(a_hi, value + 0.05 + more * k_large * hi, places=9)
            self.assertAlmostEqual(r_lo, value - 0.05 - more * k_large * hi, places=9)
            self.assertAlmostEqual(r_hi, value + 0.05 - less * k_small * 3.05, places=9)
            self.assertEqual(decide(rec), outcome, value)
        # At 194.8 every value the record allows gives one outcome.
        self.assertGreater(194.75 + k_small * 3.05, 200.0)
        self.assertLessEqual(194.85 - k_small * 3.05, 200.0)

    def test_the_working_values_and_the_figure(self):
        rec = self.near(self.rec, 194.6)
        a_lo, a_hi = map(float, guard(rec)[0])
        k = t_upper(0.05, self.rec["result"]["dof"])
        band = k * self.rec["result"]["u"]
        # Until Version 71 the two ends were one number, the sum with the factor as computed, 199.94976.
        # Since Version 72 they are its two bounds for the allowance of the factor, a millionth of the band
        # on either side: the numbers that decide() compares (tests/test_review71.py).
        self.assertAlmostEqual(a_lo, 194.6 + (1 - 1e-6) * band, places=9)
        self.assertAlmostEqual(a_hi, 194.6 + (1 + 1e-6) * band, places=9)
        self.assertEqual(f"{(a_lo + a_hi) / 2:.5f}", "199.94976")
        self.assertEqual((f"{a_lo:.6f}", f"{a_hi:.6f}"), ("199.949756", "199.949767"))
        a_lo = (a_lo + a_hi) / 2            # the mark of Figure 22.1
        self.assertEqual(decide(rec), ("accept", []))
        # What Figure 22.1 draws, and what Transcript 22.4 prints.
        lo, hi = guard(self.near(self.shown, 194.6))[0]
        self.assertEqual((f"{a_lo:.2f}", f"{lo:.2f}", f"{hi:.2f}"), ("199.95", "199.87", "200.11"))
        text = expected("ch22_record_check")
        # Until Version 69 the two ranges of the estimate plus its band read 199.87 to 200.11 and 200.07 to
        # 200.31, each end to the nearest hundredth. Since Version 70 the lower end is rounded downward and the
        # upper one upward: 200.1105 lies in the first range and above 200.11 (tests/test_review69.py).
        self.assertTrue(199.86 < lo < 199.87 and 200.11 < hi < 200.12)
        for line in ("  components allow: 2.982 to 3.122", "  with the stated u: 3.050 to 3.122",
                     "  estimate + band: 199.86 to 200.12", "  estimate + band: 200.06 to 200.32",
                     "  estimate + band: 199.949755 to 199.949767"):      # until Version 71: 199.95
            self.assertIn(line + "\n", text)
        source = (ROOT / "figures" / "make_figures.py").read_text(encoding="utf-8")
        self.assertIn("from swmetro.record import guard", source)

    def test_a_working_record_always_has_one_number(self):
        # The name is that of Version 10, when the two ends of a working record were one number. Since Version
        # 72 that holds where the band is known exactly (tests/test_review71.py). With a computed factor the
        # ends are two bounds, two millionths of the band apart, and nothing else separates them: the rounding
        # of a rounded record moves them a ten thousand times further.
        k = t_upper(0.05, self.rec["result"]["dof"])
        width = 2e-6 * k * self.rec["result"]["u"]
        for value in (150.0, 194.6, 194.65, 199.0, 250.0):
            (a_lo, a_hi), (r_lo, r_hi) = guard(self.near(self.rec, value))
            self.assertAlmostEqual(float(a_hi - a_lo), width, places=12)
            self.assertAlmostEqual(float(r_hi - r_lo), width, places=12)
            self.assertTrue(a_lo < a_hi and r_lo < r_hi and a_lo + r_hi == a_hi + r_lo)
            self.assertIn(decide(self.near(self.rec, value))[0], ("accept", "reject", "undecided"))


class BesideTheFigures(unittest.TestCase):
    def test_figure_8_1_the_chance_of_seeing_a_spike(self):
        # A spike of 2 s that may begin anywhere in a 15 s interval: exactly 2/15.
        grid = 150000
        seen = sum(i * 15 / grid + 2.0 > 15 for i in range(grid)) / grid
        self.assertAlmostEqual(seen, 2 / 15, places=4)
        self.assertEqual(f"{100 * 2 / 15:.1f}", "13.3")
        share = ch08_aliasing.spikes_seen(2.0, ch08_aliasing.SCRAPE, 20_000, random.Random(8))
        self.assertEqual(f"{100 * share:.1f}", "13.4")
        self.assertIn("2 s spikes seen: 13.4 %", expected("ch08_aliasing"))
        # Seventeen scrapes fit one slow wave while the signal makes fifteen cycles.
        self.assertEqual(16 * ch08_aliasing.SCRAPE / ch08_aliasing.PERIOD, 15)

    def test_figure_10_1_where_the_two_stalls_reappear(self):
        xs = ch10_coordinated_omission.closed_loop()
        self.assertEqual((len(xs), sum(x == 1001.0 for x in xs), sum(x == 1.0 for x in xs)), (5800, 2, 5798))
        # Linear interpolation at rank q (n - 1): 5793.2 for p99.9 and 5798.4 for p99.99.
        self.assertLess(0.999 * 5799, 5797)
        self.assertGreater(0.9999 * 5799, 5798)
        self.assertEqual((quantile(xs, 0.999), quantile(xs, 0.9999)), (1.0, 1001.0))
        plotted = (0.5, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999)
        self.assertEqual({quantile(xs, q) for q in plotted}, {1.0})
        row = re.search(r"^as measured\s+5800\s+1\s+1\s+1\s+1001\s+1\.3$", expected("ch10_coordinated_omission"), re.M)
        self.assertIsNotNone(row)

    def test_figure_15_1_the_moving_range_and_the_upper_sum(self):
        xs = ch15_control_chart.series(random.Random(15))
        base = xs[:30]
        diffs = [b - a for a, b in zip(base, base[1:])]
        sigma = sum(abs(d) for d in diffs) / len(diffs) / 1.128
        self.assertEqual(f"{sigma:.2f}", "1.80")
        self.assertAlmostEqual(1.128, 2 / math.sqrt(math.pi), places=3)
        self.assertEqual(f"{sum(diffs) / len(diffs):.3f}", "-0.058")
        centre, got = ch15_control_chart.xmr(base)
        self.assertAlmostEqual(got, sigma, places=12)
        lo, hi = centre - 3 * sigma, centre + 3 * sigma
        self.assertEqual((f"{lo:.2f}", f"{hi:.2f}"), ("94.41", "105.21"))
        self.assertTrue(all(lo <= x <= hi for x in xs[30:]))
        upper = lower = 0.0
        first_up = first_down = None
        for i, x in enumerate(xs[30:], 1):
            z = (x - centre) / sigma
            upper, lower = max(0.0, upper + z - 0.5), max(0.0, lower - z - 0.5)
            first_up = first_up or (i if upper > 5 else None)
            first_down = first_down or (i if lower > 5 else None)
        self.assertEqual((first_up, first_down), (10, None))

    def test_figure_18_1_the_weighting_of_the_score(self):
        rng = random.Random(18)
        cal = ch18_calibration.make(2000, 2.0, rng)
        test = ch18_calibration.make(5000, 2.0, rng)
        temp = ch18_calibration.fit_temperature(cal)
        self.assertEqual(f"{temp:.2f}", "2.04")
        bins = [b for b in ch18_calibration.bins_of(test, 1.0) if b[0]]
        self.assertEqual(len(bins), 10)
        # Version 64: a bin has a fourth entry, the sum of its residuals, which is not used here.
        gaps = [abs(s / n - k / n) for n, s, k, _ in bins]
        weighted = sum(b[0] * g for b, g in zip(bins, gaps)) / sum(b[0] for b in bins)
        self.assertEqual((f"{weighted:.3f}", f"{sum(gaps) / len(gaps):.3f}"), ("0.102", "0.099"))
        self.assertAlmostEqual(ch18_calibration.ece(test), weighted, places=12)
        self.assertEqual(len([b for b in ch18_calibration.bins_of(test, 1.0, 5) if b[0]]), 5)

    def test_figure_21_1_medians_of_300_and_means_of_1000(self):
        rng = random.Random(21)
        xs = ch21_psi.simulate(100, None, rng)
        self.assertEqual(len(xs), 300)
        got = [f"{quantile(xs, q):.3f}" for q in (0.05, 0.5, 0.95)]
        self.assertEqual(got, ["0.061", "0.152", "0.333"])
        self.assertEqual(sum(x > 0.1 for x in xs), 233)
        # The transcript: 1000 pairs for each size, drawn one size after another.
        rng = random.Random(21)
        counts = [sum(x > 0.1 for x in ch21_psi.simulate(n, None, rng, 1000)) for n in (100, 200, 500, 2000)]
        self.assertEqual(counts, [803, 343, 1, 0])
        # Since the review of Version 10 the transcript prints the counts themselves.
        text = expected("ch21_psi")
        for n, count in zip((100, 200, 500, 2000), counts):
            self.assertIsNotNone(re.search(rf"^\s*{n}\s+\S+\s+\S+\s+{count}\s", text, re.M), n)
        self.assertIn("   100  0.171", text)


class Replay(unittest.TestCase):
    def test_a_number_printed_to_another_place_is_a_difference(self):
        self.assertEqual(run_all.compare("exact", "share 0.1 %\n", "share 0.1 %\n"), [])
        for old, new in (("share 0 %", "share 0.1 %"), ("share 0 %", "share 0.0 %"), ("u 1.50", "u 1.5"),
                         ("n 10", "n 10.0")):
            self.assertTrue(run_all.compare("exact", old + "\n", new + "\n"), (old, new))
        # The tolerance of one unit in the last place is as it was.
        self.assertEqual(run_all.compare("exact", "x 0.105\n", "x 0.106\n"), [])
        self.assertTrue(run_all.compare("exact", "x 0.105\n", "x 0.107\n"))
        # A live example is compared by its form only.
        self.assertEqual(run_all.compare("live", "t 12.5 ms\n", "t 7 ms\n"), [])


if __name__ == "__main__":
    unittest.main()
