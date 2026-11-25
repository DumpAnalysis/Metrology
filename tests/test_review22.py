"""Regression tests added after the review of Version 22.

The review found that a line fitted to time stamps which count nanoseconds
from 1970 had a wrong slope, and for stamps one nanosecond apart no slope at
all, because the fit took a floating-point mean of the stamps before it
subtracted; that Cohen's kappa and the sum of sensitivity and specificity
less one agree in the judge example because its calibration set is balanced,
and that kappa is undefined, and was a division by zero, when both raters
give every item one label; and that the distribution of request durations
which the bound for a silence takes as known depends on the rule by which
requests entered the sample it was learned from.

Each test computes what it expects independently of the code under test:
least squares in exact fractions, kappa from counts of labels, the lengths of
the start times that a selection rule admits tried on a grid, and seeded
simulations of a source of requests.
"""
import contextlib
import io
import math
import random
import unittest
from fractions import Fraction
from pathlib import Path

from examples import ch16_silence as silence
from examples import ch20_error_bars as bars
from examples import ch20_judge as judge
from swmetro import stats

ROOT = Path(__file__).resolve().parent.parent
EPOCH = 1_700_000_000_000_000_000                # ns from 1970 to a day in November 2023


def expected(name):
    return (ROOT / "expected" / f"{name}.txt").read_text(encoding="utf-8")


def printed(function, *args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        function(*args)
    return out.getvalue()


def exact_fit(xs, ys):
    """Least squares in exact fractions: intercept at x = 0, slope, and the squares of u(a), u(b) and s."""
    xs, ys = [Fraction(x) for x in xs], [Fraction(y) for y in ys]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    s2 = sum((y - a - b * x) ** 2 for x, y in zip(xs, ys)) / (n - 2)
    return a, b, s2 * (Fraction(1, n) + mx * mx / sxx), s2 / sxx, s2


def float_mean_fit(xs, ys):
    """The slope as Version 22 computed it: about a floating-point mean of the abscissas as they come."""
    n = len(xs)
    mx, my = math.fsum(xs) / n, math.fsum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx


def close(got, want, rel=1e-9):
    want = float(want)
    return abs(got - want) <= rel * max(abs(want), 1e-300)


class LineAtAnEpochOrigin(unittest.TestCase):
    def test_the_table_of_the_review(self):
        self.assertEqual(math.ulp(float(EPOCH)), 256.0)      # neighbouring floats at this origin, in ns
        for offsets, ys, slope, old in (((0, 100, 200, 300), (0, 100, 200, 300), 1.0, 0.390625),
                                        ((0, 1000, 2000, 3000), (10, 12, 14, 16), 0.002, 0.001953125)):
            stamps = [EPOCH + t for t in offsets]
            self.assertEqual(float_mean_fit(stamps, ys), old)                 # what Version 22 returned
            for xs in (offsets, stamps):
                a, b, ua, ub, s = stats.linfit(xs, ys)
                self.assertAlmostEqual(b, slope, places=15)
                self.assertAlmostEqual(ub, 0.0, places=12)
                self.assertAlmostEqual(s, 0.0, places=12)
        # Responses that are stamps of the same origin: elapsed time against elapsed time.
        stamps = [EPOCH + t for t in (0, 100, 200, 300)]
        self.assertEqual(stats.linfit(stamps, stamps)[1:], (1.0, 0.0, 0.0, 0.0))

    def test_stamps_one_nanosecond_apart(self):
        offsets, ys = (0, 1, 2, 3), (5, 6, 7, 9)
        near = stats.linfit(offsets, ys)
        far = stats.linfit([EPOCH + t for t in offsets], ys)              # Version 22: division by zero
        self.assertEqual((float(far[1]), far[3], far[4]), (near[1], near[3], near[4]))   # slope, its uncertainty, the scatter
        self.assertGreater(far[2], 1e17)                     # the intercept at 1970 is far from the data, and says so
        _, b, _, ub2, s2 = exact_fit(offsets, ys)
        self.assertEqual(b, Fraction(13, 10))
        # Since Version 31 the slope at such stamps is not rounded: the float 1.3 times a stamp of nineteen
        # digits is 75 ns from 13/10 of it, and the readings scatter by 0.39.
        self.assertEqual((far[1], type(far[1])), (b, Fraction))
        self.assertGreater(abs(Fraction(1.3) - b) * EPOCH, 75)
        self.assertTrue(close(far[1], b) and close(far[3], math.sqrt(ub2)) and close(far[4], math.sqrt(s2)))

    def test_the_origin_does_not_change_the_fit(self):
        rng = random.Random(2204)
        for _ in range(300):
            n = rng.randint(3, 12)
            offsets = sorted(rng.sample(range(0, 5000), n))
            ys = [rng.randint(-50, 50) + 3 * t // 7 for t in offsets]
            _, b, _, ub2, s2 = exact_fit(offsets, ys)
            for origin in (0, 1_000, 10 ** 12, EPOCH, 2 ** 62, -EPOCH):
                for lift in (0, EPOCH):                      # the responses at an origin of their own
                    xs = [origin + t for t in offsets]
                    a, got, ua, ub, s = stats.linfit(xs, [lift + y for y in ys])
                    self.assertTrue(close(got, b), (origin, lift, got, float(b)))
                    self.assertTrue(close(ub, math.sqrt(ub2)), (origin, lift))
                    self.assertTrue(close(s, math.sqrt(s2)), (origin, lift))

    def test_the_intercept_is_the_one_at_zero(self):
        rng = random.Random(2205)
        for _ in range(200):
            n = rng.randint(3, 10)
            offsets = sorted(rng.sample(range(0, 400), n))
            ys = [rng.randint(-20, 20) + 2 * t for t in offsets]
            for origin in (0, 37, 5_000, EPOCH):
                xs = [origin + t for t in offsets]
                a, b, ua2, _, _ = exact_fit(xs, ys)
                got = stats.linfit(xs, ys)
                self.assertTrue(close(got[0], a, 1e-6) or abs(got[0] - float(a)) < 1e-6, (origin, got[0], float(a)))
                self.assertTrue(close(got[2], math.sqrt(ua2), 1e-6), origin)
        # The same line read from two origins: the intercepts differ by the slope times the shift.
        xs, ys = [3, 5, 8, 9], [10.0, 13.5, 20.5, 22.0]
        near, moved = stats.linfit(xs, ys), stats.linfit([x + 1000 for x in xs], ys)
        self.assertAlmostEqual(moved[0], near[0] - 1000 * near[1], places=9)
        self.assertAlmostEqual(moved[1], near[1], places=12)

    def test_floats_as_before(self):
        rng = random.Random(2206)
        for _ in range(200):
            n = rng.randint(3, 15)
            xs = [rng.uniform(0, 60) for _ in range(n)]
            ys = [2.5 + 0.75 * x + rng.gauss(0, 0.3) for x in xs]
            a, b, ua2, ub2, s2 = exact_fit(xs, ys)
            got = stats.linfit(xs, ys)
            for value, want in zip(got, (a, b, math.sqrt(ua2), math.sqrt(ub2), math.sqrt(s2))):
                self.assertTrue(close(value, want, 1e-8), (value, float(want)))
        for xs, ys, message in (([0, 1, 2, 3], [10], "one y for each x"), ([0, 1], [3, 5], "need three points"),
                                ([EPOCH] * 3, [1, 2, 3], "xs do not vary")):
            with self.assertRaisesRegex(ValueError, message):
                stats.linfit(xs, ys)
        self.assertEqual(stats.linfit((x for x in (0, 1, 2)), (y for y in (1, 3, 5))), (1.0, 2.0, 0.0, 0.0, 0.0))

    def test_the_lines_of_the_book_are_what_they_were(self):
        for name, piece in (("ch16_trace_measurements", "E1042 period, fitted slope: 7.501(10) s\n"),
                            ("ch16_trace_measurements", "E1042 period, mean of gaps: 7.503(42) s\n")):
            self.assertIn(piece, expected(name))
        for name in ("ch11_leak_rate", "ch12_calibration", "ch16_timing_models"):
            self.assertTrue(expected(name).strip())


class KappaAndTheSharesOfTheLabels(unittest.TestCase):
    SENS, SPEC = 0.895, 0.845                        # the calibration of Transcript 20.2

    @staticmethod
    def labels(passes, failures, sens, spec):
        """Reference labels and verdicts with exactly these hit rates."""
        right_p, right_f = round(passes * sens), round(failures * spec)
        ref = [True] * passes + [False] * failures
        ver = [True] * right_p + [False] * (passes - right_p) + [False] * right_f + [True] * (failures - right_f)
        return ref, ver

    def test_why_the_transcript_prints_0_740_twice(self):
        agree = (self.SENS + self.SPEC) / 2                  # 200 reference passes and 200 failures
        self.assertEqual((f"{agree:.3f}", f"{(agree - 0.5) / (1 - 0.5):.3f}"), ("0.870", "0.740"))
        self.assertEqual(f"{self.SENS + self.SPEC - 1:.3f}", "0.740")
        ref, ver = self.labels(200, 200, self.SENS, self.SPEC)
        self.assertEqual((sum(a and b for a, b in zip(ref, ver)), sum(not (a or b) for a, b in zip(ref, ver))), (179, 169))
        self.assertAlmostEqual(stats.cohen_kappa(ref, ver), self.SENS + self.SPEC - 1, places=12)
        self.assertAlmostEqual(judge.kappa_at(0.5, self.SENS, self.SPEC), self.SENS + self.SPEC - 1, places=12)
        # With as many passes as failures the chance agreement is one half whatever the second rater says.
        rng = random.Random(2220)
        for _ in range(50):
            sens, spec = rng.random(), rng.random()
            self.assertAlmostEqual(judge.kappa_at(0.5, sens, spec), sens + spec - 1, places=12)
        text = expected("ch20_judge")
        self.assertIn("Cohen's kappa 0.740\nsens + spec - 1 = 0.740\nkappa with 10 % reference passes: 0.470\n", text)
        self.assertEqual(len(text.splitlines()), 29)

    def test_one_reference_pass_in_ten(self):
        self.assertEqual(f"{judge.kappa_at(0.1, self.SENS, self.SPEC):.5f}", "0.47034")
        # From the counts of 2000 items: 200 passes, of which 179 judged right, 1800 failures, 1521 judged right.
        ref, ver = self.labels(200, 1800, self.SENS, self.SPEC)
        self.assertEqual((sum(ref), sum(a and b for a, b in zip(ref, ver)), sum(not (a or b) for a, b in zip(ref, ver))), (200, 179, 1521))
        n = 2000
        agree = (179 + 1521) / n
        passed = (179 + 1800 - 1521) / n
        chance = (200 / n) * passed + (1800 / n) * (1 - passed)
        self.assertEqual((f"{agree:.4f}", f"{passed:.4f}", f"{chance:.4f}"), ("0.8500", "0.2290", "0.7168"))
        self.assertAlmostEqual(stats.cohen_kappa(ref, ver), (agree - chance) / (1 - chance), places=12)
        self.assertAlmostEqual(stats.cohen_kappa(ref, ver), judge.kappa_at(0.1, self.SENS, self.SPEC), places=12)
        # The error rates are those of the balanced set, and so is their sum less one.
        self.assertAlmostEqual(179 / 200 + 1521 / 1800 - 1, 0.740, places=12)
        # Kappa falls toward either end of the range of shares, at fixed error rates.
        values = [judge.kappa_at(s, self.SENS, self.SPEC) for s in (0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99)]
        self.assertEqual(max(values), values[3])
        self.assertLess(values[0], 0.15)

    def test_one_label_has_no_kappa(self):
        for labels in ([1] * 5, [True], ["pass"] * 40):
            with self.assertRaisesRegex(ValueError, "kappa needs two labels"):
                stats.cohen_kappa(labels, list(labels))      # Version 22: division by zero
        with self.assertRaisesRegex(ValueError, "no items"):
            stats.cohen_kappa([], [])
        with self.assertRaisesRegex(ValueError, "two labels an item"):
            stats.cohen_kappa([1, 1], [1])
        # One rater with one label and the other with two: agreement is what chance gives, and kappa is 0.
        self.assertEqual(stats.cohen_kappa([1, 1, 1, 1], [1, 0, 1, 0]), 0.0)
        # The two raters with different single labels: no agreement, none expected, kappa is 0.
        self.assertEqual(stats.cohen_kappa([1, 1, 1], [0, 0, 0]), 0.0)
        self.assertEqual(stats.cohen_kappa([1, 0, 1, 1], [1, 0, 0, 1]), 0.5)

    def test_other_results_that_are_not_defined(self):
        # Found by calling the functions of the package with inputs at the edge: a division by zero for an input
        # that can occur. Each says what is missing.
        with self.assertRaisesRegex(ValueError, "no trials"):
            stats.wilson(0, 0)
        low, high = stats.wilson(0, 5)
        self.assertEqual(low, 0.0)                           # Versions 22 to 24 returned 2.8e-17, and this test let it pass
        self.assertTrue(0.3 < high < 0.5)
        with self.assertRaisesRegex(ValueError, "need two clusters"):
            bars.clustered_se([1.0, 0.0, 1.0], ["p", "p", "p"])
        self.assertGreater(bars.clustered_se([1.0, 0.0, 1.0, 1.0], ["p", "p", "q", "q"]), 0)


class HowARequestEntersTheSample(unittest.TestCase):
    SOURCE = [(0.5, 10), (0.5, 50)]
    WHOLE = [(0, 30)]

    @staticmethod
    def admits(rule, start, duration, window=30):
        first = 0 <= start < window
        last = 0 <= start + duration < window
        return {"start": first, "both": first and last, "either": first or last}[rule]

    def test_the_table_of_the_review(self):
        want = {"start": ((0.5, 0.5), 50.0, "0.11983", "5.00"), "both": ((1.0, 0.0), 40.0, "0.14979", "2.36"),
                "either": ((0.4, 0.6), 52.0, "0.11522", "5.61")}
        for rule, (mix, covered, bound, still) in want.items():
            seen = silence.selected(self.SOURCE, 30, rule)
            self.assertEqual([d for _, d in seen], [10, 50])
            for (p, _), share in zip(seen, mix):
                self.assertAlmostEqual(p, share, places=12)
            exposure = sum(p * (30 + min(d, 30)) for p, d in zip(mix, (10, 50)))
            self.assertAlmostEqual(exposure, covered, places=12)
            self.assertAlmostEqual(silence.mean_reach(self.WHOLE, seen), covered, places=12)
            rate = 2 * -math.log(0.05) / covered
            self.assertEqual(f"{rate:.5f}", bound)
            self.assertEqual(f"{100 * math.exp(-rate / 2 * 50):.2f}", still)    # the source covers 50 s of starts

    def test_the_start_times_that_a_rule_admits(self):
        step = 0.01
        for rule in ("start", "both", "either"):
            lengths = {}
            for d in (0, 2.5, 10, 29.99, 30, 50, 90):
                grid = [-d - 1 + step * (i + 0.5) for i in range(int((d + 32) / step))]
                lengths[d] = step * sum(self.admits(rule, t, d) for t in grid)
            for d, length in lengths.items():
                want = {"start": 30, "both": max(0, 30 - d), "either": 30 + min(d, 30)}[rule]
                self.assertAlmostEqual(length, want, delta=0.03, msg=(rule, d))
        # A sample shows a duration in proportion to its share times that length.
        rng = random.Random(2216)
        for _ in range(100):
            ds = rng.sample((1, 5, 10, 20, 29, 40, 50, 90), rng.randint(1, 4))
            ws = [rng.randint(1, 9) for _ in ds]
            source = [(w / sum(ws), d) for w, d in zip(ws, ds)]
            for rule, room in (("start", lambda d: 30), ("both", lambda d: max(0, 30 - d)), ("either", lambda d: 30 + min(d, 30))):
                weights = [p * room(d) for p, d in source]
                if not sum(weights):
                    with self.assertRaisesRegex(ValueError, "no request selected"):
                        silence.selected(source, 30, rule)
                    continue
                seen = silence.selected(source, 30, rule)
                for (p, d), w, (_, d0) in zip(seen, weights, source):
                    self.assertEqual(d, d0)
                    self.assertAlmostEqual(p, w / sum(weights), places=12)
                self.assertAlmostEqual(sum(p for p, _ in seen), 1.0, places=12)
                if rule == "start":
                    self.assertEqual([round(p, 12) for p, _ in seen], [round(p, 12) for p, _ in source])

    def test_a_simulated_sample(self):
        rng = random.Random(2217)
        counts = {rule: {10: 0, 50: 0} for rule in ("start", "both", "either")}
        for _ in range(40_000):                               # independent windows of 30 s on a stationary source
            t = -60 + rng.expovariate(0.2)
            while t < 30:
                d = rng.choice((10, 50))
                for rule in counts:
                    counts[rule][d] += self.admits(rule, t, d)
                t += rng.expovariate(0.2)
        for rule, share_of_10 in (("start", 0.5), ("both", 1.0), ("either", 0.4)):
            got = counts[rule][10] / (counts[rule][10] + counts[rule][50])
            self.assertAlmostEqual(got, share_of_10, delta=0.006, msg=rule)
            self.assertAlmostEqual(silence.selected(self.SOURCE, 30, rule)[0][0], share_of_10, places=12)

    def test_rules_and_their_limits(self):
        with self.assertRaises(KeyError):
            silence.selected(self.SOURCE, 30, "middle")              # "end" is a rule of its own since Version 24
        with self.assertRaisesRegex(ValueError, "no request selected"):
            silence.selected([(1, 50)], 30, "both")
        for rule in ("start", "either"):
            self.assertEqual(silence.selected([(1, 50)], 30, rule), [(1.0, 50)])
        self.assertEqual(silence.selected(((p, d) for p, d in self.SOURCE), 30, "start"), [(0.5, 10), (0.5, 50)])

    def test_the_transcript(self):
        out = printed(silence.sampling)
        # Since Version 24 the rate drawn from a sampled mix is printed as nominal, with the start times that
        # select each duration above the mix, and a request taken by its end stands between the first two rules.
        for rule, lengths, mix, covered, rate, still in (
                ("starts in a window", "30 s and 30 s", "50 % of 10 s, 50 % of 50 s", "50", "0.11983", "5.00"),
                ("starts and ends in it", "20 s and 0 s", "100 % of 10 s, 0 % of 50 s", "40", "0.14979", "2.36"),
                ("starts or ends in it", "40 s and 60 s", "40 % of 10 s, 60 % of 50 s", "52", "0.11523", "5.61")):
            # 0.11522 until Version 58: the rate is 0.115220..., and a bound is printed rounded upward.
            self.assertIn(f"  {rule}:\n    from {lengths} of start times\n    {mix}\n    starts covered: {covered} s\n"
                          f"    {rate} msg/s at a nominal 95 %\n    the source is silent at that: {still} %\n", out)
        self.assertTrue(out.startswith("durations sampled with windows of\n30 s; a request is taken if it\n  starts in a window:\n"))
        text = expected("ch16_silence")
        self.assertIn(out, text)
        self.assertIn(printed(silence.durations) + out, text)   # after the three sources of one mean duration
        self.assertEqual(len(text.splitlines()), 193)           # 118 in Version 23, 140, 152, 162, 166 in Version 27


class TwoWordings(unittest.TestCase):
    def test_the_mean_duration_can_give_too_low_a_bound(self):
        text = " ".join(silence.mean_reach.__doc__.split())
        self.assertIn("can give too low a bound", text)
        self.assertNotIn("would give", text)
        whole = [(0, 30)]

        def both(shares):
            mean = sum(p * d for p, d in shares)
            return silence.mean_reach(whole, shares), silence.reach(whole, mean)
        # Every duration within the window, or every duration at least as long: the same exposure.
        for shares in ([(0.5, 5), (0.5, 25)], [(0.2, 0), (0.8, 30)], [(0.5, 30), (0.5, 90)], [(0.3, 40), (0.7, 291)]):
            got, at_mean = both(shares)
            self.assertAlmostEqual(got, at_mean, places=12, msg=shares)
        # Durations on both sides of the window: strictly less, and the bound for the mean duration is too low.
        for shares in ([(0.5, 10), (0.5, 50)], [(0.9, 1), (0.1, 291)], [(0.99, 29), (0.01, 31)]):
            got, at_mean = both(shares)
            self.assertLess(got, at_mean, shares)

    def test_the_rate_in_the_sentence_with_8_24(self):
        low = 2 * -math.log(0.05) / 60
        self.assertEqual((f"{low:.5f}", f"{100 * math.exp(-low / 2 * 50):.2f}"), ("0.09986", "8.24"))
        self.assertEqual(f"{100 * math.exp(-0.100 / 2 * 50):.2f}", "8.21")              # the rounded rate, taken literally
        self.assertIn("  at the first bound the second\n  stays silent: 8.24 %\n", expected("ch16_silence"))


if __name__ == "__main__":
    unittest.main()
