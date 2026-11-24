"""Regression tests added after the review of Version 24.

The review found that the book let a request taken by its end start only
before the window, where the calculation also takes one that starts inside
it, and that the narrower reading gives another mix; that kappa is zero, and
not undefined, at either end of the range of shares while the judge still
gives both labels; that the Wilson interval for no success, or for no
failure, stopped just short of 0, or of 1; that a fractional number of
trials, a fractional number of successes and a negative number of events
still gave an interval or a bound; that the uncertainties of the fitted line
went through a floating-point square, which is lost at extreme scales where
its root is not; and it proposed a worked case in which a recorder keeps one
message in four, by the message or by the request.

Each test computes what it expects independently of the code under test:
start times counted on a grid, seeded simulations of the sources and of the
recorders, closed forms for the limits, and exact fractions.
"""
import contextlib
import io
import math
import random
import unittest
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

from examples import ch16_bounds as bounds
from examples import ch16_silence as silence
from examples import ch16_trace_measurements as trace
from examples import ch20_judge as judge
from examples import ch23_case as case
from swmetro import core, dist, stats

ROOT = Path(__file__).resolve().parent.parent
Z95 = dist.z_ppf(0.975)                           # the 97.5 % point of the standard normal distribution, 1.95996...


def expected(name):
    return (ROOT / "expected" / f"{name}.txt").read_text(encoding="utf-8")


def printed(function, *args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        function(*args)
    return out.getvalue()


class WhereACompletionMayStart(unittest.TestCase):
    SOURCE = [(0.5, 10), (0.5, 50)]

    @staticmethod
    def length(admits, duration, step=0.01):
        """Start times that a rule admits, counted on a grid from before the window to its end."""
        grid = [-duration - 1 + step * (i + 0.5) for i in range(int((duration + 32) / step))]
        return step * sum(admits(t, duration) for t in grid)

    @staticmethod
    def any_start(t, d):
        return 0 <= t + d < 30                               # every end in the window

    @staticmethod
    def started_before(t, d):
        return 0 <= t + d < 30 and t < 0                     # and the start strictly before it opens

    def test_the_two_readings_of_the_review(self):
        want = {self.any_start: ((30, 30), (0.5, 0.5), 50.0, "0.11983", "5.00"),
                self.started_before: ((10, 30), (0.25, 0.75), 55.0, "0.10894", "6.57")}
        for admits, (lengths, mix, covered, rate, still) in want.items():
            counted = [self.length(admits, d) for d in (10, 50)]
            for got, length in zip(counted, lengths):
                self.assertAlmostEqual(got, length, delta=0.03)
            total = sum(p * length for (p, _), length in zip(self.SOURCE, lengths))
            shares = [p * length / total for (p, _), length in zip(self.SOURCE, lengths)]
            for got, share in zip(shares, mix):
                self.assertAlmostEqual(got, share, places=12)
            exposure = sum(share * (30 + min(d, 30)) for share, d in zip(shares, (10, 50)))
            self.assertAlmostEqual(exposure, covered, places=12)
            top = 2 * -math.log(0.05) / exposure
            self.assertEqual((f"{top:.5f}", f"{100 * math.exp(-top / 2 * 50):.2f}"), (rate, still))

    def test_the_listing_takes_a_request_wherever_it_starts(self):
        for d in (0, 2.346, 10, 29.99, 30, 50, 90):
            self.assertAlmostEqual(silence.eligible(30, "end", d), self.length(self.any_start, d), delta=0.03, msg=d)
            narrow = self.length(self.started_before, d)
            self.assertAlmostEqual(narrow, min(30, d), delta=0.03, msg=d)         # the other rule: min(window, d)
            # What the narrower rule leaves out: the starts inside the window, 30 s less the duration of them.
            self.assertAlmostEqual(silence.eligible(30, "end", d) - narrow, max(0, 30 - d), delta=0.05, msg=d)
        # The end message of the request of the fixture: starts from -2.346 s to 27.654 s, as in Figure 16.1.
        self.assertTrue(self.any_start(-2.346, 2.346) and self.any_start(27.653, 2.346) and self.any_start(5, 2.346))
        self.assertFalse(self.any_start(27.654, 2.346) or self.any_start(-2.347, 2.346))
        self.assertEqual([(float(a), float(b)) for a, b in silence.starts([(0, 30)], 2.346)], [(-2.346, 30.0)])
        seen = silence.selected(self.SOURCE, 30, "end")
        self.assertEqual([(round(p, 12), d) for p, d in seen], [(0.5, 10), (0.5, 50)])
        text = " ".join(silence.eligible.__doc__.split())
        self.assertIn("wherever its start falls", text)
        self.assertIn("min(window, d)", text)

    def test_a_simulated_sample(self):
        rng = random.Random(2401)
        counts = {name: {10: 0, 50: 0} for name in ("any", "before")}
        for _ in range(40_000):                               # independent windows of 30 s on a stationary source
            t = -60 + rng.expovariate(0.2)
            while t < 30:
                d = rng.choice((10, 50))
                counts["any"][d] += self.any_start(t, d)
                counts["before"][d] += self.started_before(t, d)
                t += rng.expovariate(0.2)
        for name, share_of_10 in (("any", 0.5), ("before", 0.25)):
            got = counts[name][10] / (counts[name][10] + counts[name][50])
            self.assertAlmostEqual(got, share_of_10, delta=0.006, msg=name)


class KappaAtTheEndsOfTheRange(unittest.TestCase):
    def test_zero_while_the_judge_gives_both_labels(self):
        for share in (0, 1):
            self.assertEqual(judge.kappa_at(share, 0.895, 0.845), 0.0)             # the rates of Transcript 20.2
        # The same from counts: no reference pass among 200, the judge passes 31 of them.
        ref = [False] * 200
        ver = [True] * 31 + [False] * 169
        self.assertEqual(stats.cohen_kappa(ref, ver), 0.0)
        ref = [True] * 200
        ver = [True] * 179 + [False] * 21
        self.assertEqual(stats.cohen_kappa(ref, ver), 0.0)

    def test_undefined_only_where_both_give_one_label(self):
        for share, sens, spec in ((0, 0.895, 1), (1, 1, 0.845)):
            with self.assertRaisesRegex(ValueError, "kappa needs two labels"):
                judge.kappa_at(share, sens, spec)
        for labels in ([False] * 200, [True] * 200):
            with self.assertRaisesRegex(ValueError, "kappa needs two labels"):
                stats.cohen_kappa(labels, list(labels))
        # Towards an end kappa goes to 0 with these rates: no jump there.
        values = [judge.kappa_at(10.0 ** -k, 0.895, 0.845) for k in (1, 2, 4, 8, 12)]
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertLess(values[-1], 1e-10)
        self.assertGreater(values[-1], 0)


def wilson_version_24(k, n):
    ph = k / n
    den = 1 + Z95 * Z95 / n
    mid = (ph + Z95 * Z95 / (2 * n)) / den
    half = Z95 * math.sqrt(ph * (1 - ph) / n + Z95 * Z95 / (4 * n * n)) / den
    return mid - half, mid + half


class ZeroAndOneInTheirWilsonIntervals(unittest.TestCase):
    def test_the_table_of_the_review(self):
        old = wilson_version_24(0, 40)
        self.assertEqual(f"{old[0]:.4e}", "6.9389e-18")
        self.assertEqual(repr(wilson_version_24(40, 40)[1]), "0.9999999999999999")
        low, high = stats.wilson(0, 40)
        self.assertEqual((low, f"{high:.8f}"), (0.0, "0.08762160"))
        self.assertIs(type(low), float)
        low, high = stats.wilson(40, 40)
        self.assertEqual((f"{low:.8f}", high), ("0.91237840", 1.0))
        # The other limit in closed form: z^2 / (n + z^2) from 0, and n / (n + z^2) from 1.
        self.assertAlmostEqual(stats.wilson(0, 40)[1], Z95 ** 2 / (40 + Z95 ** 2), places=15)
        self.assertAlmostEqual(stats.wilson(40, 40)[0], 40 / (40 + Z95 ** 2), places=15)

    def test_an_interval_holds_the_value_it_has_to_hold(self):
        for n in list(range(1, 300)) + [10 ** 4, 10 ** 6, 10 ** 9]:
            low, _ = stats.wilson(0, n)
            _, high = stats.wilson(n, n)
            self.assertEqual((low, high), (0.0, 1.0), n)
            self.assertTrue(low <= 0 <= stats.wilson(0, n)[1] and stats.wilson(n, n)[0] <= 1 <= high)
        for level in (0.5, 0.9, 0.99, 0.999999):
            self.assertEqual((stats.wilson(0, 7, level)[0], stats.wilson(7, 7, level)[1]), (0.0, 1.0))

    def test_between_the_ends_nothing_changed(self):
        for n in (1, 2, 5, 40, 1000):
            for k in range(n + 1):
                got = stats.wilson(k, n)
                old = wilson_version_24(k, n)
                self.assertAlmostEqual(got[0], old[0], places=15)
                self.assertAlmostEqual(got[1], old[1], places=15)
                if 0 < k:
                    self.assertEqual(got[0], old[0])
                    self.assertGreater(got[0], 0)
                if k < n:
                    self.assertEqual(got[1], old[1])
                    self.assertLess(got[1], 1)
                mirror = stats.wilson(n - k, n)
                self.assertAlmostEqual(got[0], 1 - mirror[1], places=14)
        self.assertEqual([f"{v:.4f}" for v in stats.wilson(2, 20)], ["0.0279", "0.3010"])
        # The exact interval of Listing 20.2 had its ends already.
        self.assertEqual((judge.exact_interval(0, 40)[0], judge.exact_interval(40, 40)[1]), (0.0, 1.0))


class CountsThatAreNotCounts(unittest.TestCase):
    def test_what_a_count_is(self):
        for good, want in ((0, 0), (3, 3), (3.0, 3), (True, 1), (Fraction(4), 4), (Decimal("5"), 5), (10 ** 30, 10 ** 30)):
            got = stats.as_count(good)
            self.assertEqual((got, type(got)), (want, int))
        for bad in (-1, 0.5, 2.5, -0.0001, math.nan, math.inf, -math.inf, "3", None, [3], Fraction(7, 2), 1e-9):
            with self.assertRaisesRegex(ValueError, "need a whole n >= 0"):
                stats.as_count(bad)
        with self.assertRaisesRegex(ValueError, "need a whole windows >= 1"):
            stats.as_count(0, "windows", 1)
        self.assertEqual(stats.as_count(1, "windows", 1), 1)

    def test_the_table_of_the_review(self):
        with self.assertRaisesRegex(ValueError, "need a whole n >= 0"):
            judge.exact_interval(2, 2.5)                     # Version 24: 0.11784 to 0.98992
        with self.assertRaisesRegex(ValueError, "need a whole k >= 0"):
            stats.wilson(0.5, 5)                             # Version 24: 0.01053 to 0.53706
        with self.assertRaisesRegex(ValueError, "need a whole n >= 0"):
            stats.wilson(2, 5.5)
        with self.assertRaisesRegex(ValueError, "need a whole n >= 0"):
            stats.rate_upper(-1, 30)                         # Version 24: 0 msg/s
        # A true count of zero in 30 s supports 0.09986 msg/s, and no less.
        self.assertEqual(f"{stats.rate_upper(0, 30):.5f}", "0.09986")
        self.assertAlmostEqual(stats.rate_upper(0, 30), -math.log(0.05) / 30, places=15)
        for n in (-1, 1.5, 0.5, math.nan):
            for helper in (stats.rate_upper, stats.rate_limits):
                with self.assertRaisesRegex(ValueError, "need a whole n >= 0"):
                    helper(n, 30)                            # Version 24, rate_limits(-1, 30): -1e-62 to 0
        # Counts that are whole in value are taken, whatever their type.
        self.assertEqual(stats.rate_upper(1.0, 30), stats.rate_upper(1, 30))
        self.assertEqual(stats.wilson(2.0, 20.0), stats.wilson(2, 20))
        self.assertEqual(judge.exact_interval(2.0, 20.0), judge.exact_interval(2, 20))

    def test_two_readings_of_a_counter_after_a_reset(self):
        before, after = 1200, 40                             # the counter restarted between the readings
        with self.assertRaises(ValueError):
            stats.rate_upper(after - before, 30)
        with self.assertRaises(ValueError):
            trace.current(after - before, 30)
        with self.assertRaises(ValueError):
            case.arrival(after - before)
        with self.assertRaises(ValueError):
            silence.paired_upper(after - before, [(0, 30)], 10)                    # Version 24: 1.6e-62 msg/s

    def test_the_other_helpers_that_take_a_count(self):
        whole = [(0, 30)]
        for n in (-1, 1.5):
            with self.assertRaisesRegex(ValueError, "need a whole n >= 0"):
                silence.paired_upper(n, whole, 10)
            with self.assertRaisesRegex(ValueError, "need a whole n >= 0"):
                trace.current(n, 30)                         # Version 24: "no messages: a bound", and 0.050(41) msg/s
            with self.assertRaisesRegex(ValueError, "need a whole n >= 0"):
                case.arrival(n)                              # Version 24: a math domain error, and limits
        for windows in (0, -3, 2.5):
            with self.assertRaisesRegex(ValueError, "need a whole windows >= 1"):
                silence.searched_bound(30, windows)          # Version 24: a division by zero, a domain error, 0.12989
        with self.assertRaisesRegex(ValueError, "need a whole n >= 0"):
            judge.at_most(2, 2.5, 0.3)                       # Version 24: 0.98992
        with self.assertRaisesRegex(ValueError, "need a whole n >= 0"):
            judge.few_outputs(2, 2.5, 0.9, 0.85)
        for n in (0, 2.5, -4):
            with self.assertRaisesRegex(ValueError, "need a whole n >= 1"):
                judge.binom_u(0.5, n)                        # Version 24: a division by zero, and 0.316
        self.assertAlmostEqual(judge.binom_u(0.5, 100), 0.05, places=15)
        # What they returned for counts keeps its value.
        self.assertEqual(f"{silence.paired_upper(1, whole, 10):.5f}", "0.20565")
        self.assertAlmostEqual(silence.paired_upper(0, whole, 10), silence.paired_bound(whole, 10), places=12)
        self.assertEqual(f"{silence.searched_bound(30, 10):.5f}", "0.17584")
        self.assertEqual(str(trace.current(1, 30)), "0.033(33) msg/s")
        self.assertEqual(case.arrival(0, 100), f"at most {-math.log(0.05) / 100:.5f} req/s at 95 %")
        with self.assertRaisesRegex(ValueError, "no messages: a bound"):
            trace.current(0, 30)

    def test_levels_and_exposures(self):
        whole = [(0, 30)]
        for level in (0, 1, 1.5, -1):
            with self.assertRaisesRegex(ValueError, "need 0 < level < 1"):
                silence.paired_upper(0, whole, 10, level)    # Version 24 at 1.5: no answer, the search did not end
            with self.assertRaisesRegex(ValueError, "need 0 < level < 1"):
                silence.searched_bound(30, 10, level)        # Version 24 at 0: -0.0 msg/s
        with self.assertRaisesRegex(ValueError, "no exposure"):
            silence.paired_upper(1, [], 10)                  # Version 24: division by zero
        for seconds in (0, -30):
            with self.assertRaisesRegex(ValueError, "no exposure"):
                silence.searched_bound(seconds, 10)
            with self.assertRaisesRegex(ValueError, "no exposure"):
                trace.current(3, seconds)
            with self.assertRaisesRegex(ValueError, "no exposure"):
                case.arrival(3, seconds)

    def test_parts_and_their_wholes(self):
        # More outcomes captured than requests counted: Version 24 returned 2 % to -2 %, and with that an
        # objective of at most 1 % was "met".
        low, high = 20 / 1000, (20 + 1000 - 1040) / 1000
        self.assertEqual(bounds.verdict(low, high, 0.01), "met")
        for counts in ((1000, 1040, 20), (1000, 960, 980)):
            with self.assertRaisesRegex(ValueError, "a part above its whole"):
                bounds.share_bounds(*counts)
        for counts, name in (((0, 0, 0), "total >= 1"), ((1000, 960.5, 20), "captured >= 0"), ((1000, 960, -3), "failures >= 0"),
                             ((1000.5, 960, 20), "total >= 1")):
            with self.assertRaisesRegex(ValueError, f"need a whole {name}"):
                bounds.share_bounds(*counts)
        # Exact fractions since Version 53; as floats they were (0.02, 0.06) and (0.0, 0.0).
        self.assertEqual(bounds.share_bounds(1000, 960, 20), (Fraction(2, 100), Fraction(6, 100)))
        self.assertEqual(bounds.share_bounds(1000, 1000, 0), (0, 0))
        with self.assertRaisesRegex(ValueError, "a part above its whole"):
            bounds.rank_bounds(1000, 1040, 0.95)             # Version 24: ranks 990 and 950, the lower above the upper
        for total, captured in ((1000, -5), (1000, 960.5), (1000.5, 960), (0, 0)):
            with self.assertRaisesRegex(ValueError, "need a whole"):
                bounds.rank_bounds(total, captured, 0.95)
        self.assertEqual(bounds.rank_bounds(1000, 960, 0.95), (910, 950))
        with self.assertRaisesRegex(ValueError, "a part above its whole"):
            bounds.quantile_verdict(1000, 0.95, 990, 40)     # Version 24: "met"
        for within, unknown in ((-5, 10), (940.5, 10), (940, -1)):
            with self.assertRaisesRegex(ValueError, "need a whole"):
                bounds.quantile_verdict(1000, 0.95, within, unknown)
        self.assertEqual([bounds.quantile_verdict(1000, 0.95, w, 12) for w in (950, 940, 930)], ["met", "unresolved", "missed"])


class RootsOfExactSquares(unittest.TestCase):
    def test_the_two_cases_of_the_review(self):
        for h in (1e-200, 1e160, 1.0, -1e-200, 1e-170, 1e150):
            a, b, ua, ub, s = stats.linfit([0, 1, 2], [0, h, 0])                   # Version 24: s = 0, and OverflowError
            size = abs(h)
            self.assertAlmostEqual(a / h, 1 / 3, places=15)
            self.assertEqual(b, 0.0)
            self.assertAlmostEqual(s / size, math.sqrt(2 / 3), places=15)
            self.assertAlmostEqual(ua / size, math.sqrt(2 / 3 * 5 / 6), places=15)
            self.assertAlmostEqual(ub / size, math.sqrt(1 / 3), places=15)

    def test_every_scale_a_float_has(self):
        for k in range(-300, 301, 10):
            h = 10.0 ** k
            a, b, ua, ub, s = stats.linfit([0, 1, 2], [0, h, 0])
            self.assertAlmostEqual(s / h, math.sqrt(2 / 3), places=14, msg=k)
            self.assertAlmostEqual(ub / h, math.sqrt(1 / 3), places=14, msg=k)
        # A power of two changes no digit of the fit.
        rng = random.Random(2402)
        xs = [rng.uniform(0, 60) for _ in range(9)]
        ys = [2.5 + 0.75 * x + rng.gauss(0, 0.3) for x in xs]
        base = stats.linfit(xs, ys)
        for e in (-900, -300, 10, 300, 900):
            got = stats.linfit(xs, [math.ldexp(y, e) for y in ys])
            self.assertEqual(got, tuple(math.ldexp(v, e) for v in base), e)
        # Beyond the largest float there is no answer, and the fit says so.
        self.assertAlmostEqual(stats.linfit([0, 1, 2], [0, 1e308, -1e308])[4] / 1e308, math.sqrt(1.5), places=15)
        with self.assertRaises(OverflowError):
            stats.linfit([0, 1, 2], [0, 1.7e308, -1.7e308])                        # s would be 2.08e308

    def test_the_root_of_a_fraction(self):
        self.assertEqual((stats._root(Fraction(0)), stats._root(Fraction(9, 4)), stats._root(Fraction(1, 4))), (0.0, 1.5, 0.5))
        self.assertEqual((stats._root(Fraction(10) ** 600), stats._root(Fraction(1, 10 ** 600))), (1e300, 1e-300))
        self.assertEqual(stats._root(Fraction(1, 10 ** 800)), 0.0)
        with self.assertRaises(OverflowError):
            stats._root(Fraction(10) ** 700)
        # One rounding: the float nearest to the root, here against the root to 80 digits.
        rng = random.Random(2403)
        with localcontext() as ctx:
            ctx.prec = 80
            for _ in range(3000):
                f = Fraction(rng.randint(1, 10 ** rng.randint(1, 60)), rng.randint(1, 10 ** rng.randint(1, 60)))
                want = float((Decimal(f.numerator) / Decimal(f.denominator)).sqrt())
                self.assertEqual(stats._root(f), want, f)
                whole = rng.randint(1, 10 ** 15)
                self.assertEqual(stats._root(Fraction(whole * whole, 49)), whole / 7)

    def test_ordinary_fits_keep_their_numbers(self):
        rng = random.Random(2404)
        for _ in range(300):
            n = rng.randint(3, 15)
            xs = [Fraction(rng.uniform(0, 60)) for _ in range(n)]
            ys = [Fraction(2.5 + 0.75 * float(x) + rng.gauss(0, 0.3)) for x in xs]
            mx, my = sum(xs) / n, sum(ys) / n
            sxx = sum((x - mx) ** 2 for x in xs)
            b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
            a = my - b * mx
            s2 = sum((y - a - b * x) ** 2 for x, y in zip(xs, ys)) / (n - 2)
            got = stats.linfit(xs, ys)
            self.assertEqual(got[:2], (float(a), float(b)))
            for value, square in zip(got[2:], (s2 * (Fraction(1, n) + mx * mx / sxx), s2 / sxx, s2)):
                self.assertAlmostEqual(value / math.sqrt(square), 1.0, places=15)
        for name, piece in (("ch16_trace_measurements", "E1042 period, fitted slope: 7.501(10) s\n"),
                            ("ch11_leak_rate", "240 min  0.0458  0.0029"), ("ch16_timing_models", "phase jitter    0.0037  0.0019\n")):
            self.assertIn(piece, expected(name), name)
        self.assertIn("roots of exact squares", " ".join(stats.linfit.__doc__.split()))

    def test_combined_uncertainties_at_the_same_scales(self):
        self.assertAlmostEqual(core.combine(1e-200, 1e-200) / 1e-200, math.sqrt(2), places=15)   # Version 24: 0
        self.assertAlmostEqual(core.combine(3e200, 4e200) / 1e200, 5.0, places=15)               # Version 24: inf
        for scale in (1e-100, 1.0, 1e100):
            got = core.welch_satterthwaite([scale, scale], [5, 5])                               # Version 24: inf, 10, error
            self.assertAlmostEqual(got, 10.0, places=12)
        self.assertEqual((core.combine(), core.combine(0, 0), core.combine(3, 4), core.combine(-3, 4)), (0.0, 0.0, 5.0, 5.0))
        self.assertEqual((core.welch_satterthwaite([], []), core.welch_satterthwaite([0, 0], [5, 5])), (math.inf, math.inf))
        self.assertEqual(core.welch_satterthwaite([1.0, 2.0], [math.inf, math.inf]), math.inf)
        # At ordinary sizes the scaling, a power of two, changes no digit of the combination.
        rng = random.Random(2405)
        for _ in range(20_000):
            us = [rng.lognormvariate(0, 3) for _ in range(rng.randint(0, 6))]
            self.assertEqual(core.combine(*us), math.sqrt(sum(u * u for u in us)))
        for _ in range(2000):
            us = [rng.lognormvariate(0, 2) for _ in range(rng.randint(1, 6))]
            dofs = [rng.choice((1, 2, 5, 9, 30, math.inf)) for _ in us]
            den = sum(u ** 4 / d for u, d in zip(us, dofs) if math.isfinite(d))
            want = math.sqrt(sum(u * u for u in us)) ** 4 / den if den else math.inf
            got = core.welch_satterthwaite(us, dofs)
            if math.isinf(want):
                self.assertEqual(got, want)
            else:
                self.assertAlmostEqual(got / want, 1.0, places=12)
        bud = core.Budget("ms").add("a", "A", 1.0, 5).add("b", "A", 2.0, 7).add("c", "B", 0.5)
        self.assertAlmostEqual(bud.dof(), (1 + 4 + 0.25) ** 2 / (1 / 5 + 16 / 7), places=12)


class OneMessageInFourKept(unittest.TestCase):
    WHOLE = [(0, 30)]
    SINGLE = -math.log(0.05) / 7.5                           # the bound for single messages, 0.39943 msg/s

    def test_the_table_of_the_review(self):
        both_kept = 1 - (3 / 4) ** 2
        self.assertEqual(Fraction(both_kept), Fraction(7, 16))
        rows = (("single", 30 * 0.25, 7.5, "0.39943", "5.00"),
                ("message", (20 * 0.25 + 20 * both_kept) / 2, 6.875, "0.43574", "6.42"),
                ("request", 40 * 0.25 / 2, 5.0, "0.59915", "13.57"))
        for unit, exposure, want, bound, still in rows:
            self.assertEqual(exposure, want)
            self.assertEqual(f"{-math.log(0.05) / exposure:.5f}", bound)
            self.assertEqual(f"{100 * math.exp(-self.SINGLE * exposure):.2f}", still)
            if unit != "single":
                self.assertAlmostEqual(silence.kept_starts(self.WHOLE, 10, 0.25, unit) / 2, want, places=12)
        self.assertEqual(f"{self.SINGLE:.5f}", "0.39943")

    def test_start_times_that_bring_two_messages_and_one(self):
        step = 0.01
        for window, d in ((30, 10), (30, 2.346), (30, 50), (30, 30), (12, 5)):
            grid = [-d - 1 + step * (i + 0.5) for i in range(int((d + window + 2) / step))]
            inside = [(0 <= t < window) + (0 <= t + d < window) for t in grid]
            two, one = silence.by_messages([(0, window)], d)
            self.assertAlmostEqual(two, step * inside.count(2), delta=0.03, msg=(window, d))
            self.assertAlmostEqual(one, step * inside.count(1), delta=0.03, msg=(window, d))
            self.assertAlmostEqual(two + one, silence.reach([(0, window)], d), places=9)
        self.assertEqual(silence.by_messages(self.WHOLE, 10), (20, 20))
        # Two stretches, and time that they share counted once.
        self.assertEqual(silence.by_messages([(0, 20), (10, 30)], 10), (20, 20))

    def test_the_weights(self):
        for d in (2.346, 10, 30, 50):
            reach = silence.reach(self.WHOLE, d)
            for unit in ("message", "request"):
                self.assertAlmostEqual(silence.kept_starts(self.WHOLE, d, 1, unit), reach, places=12)   # everything kept
            two, one = silence.by_messages(self.WHOLE, d)
            for keep in (0.01, 0.25, 0.5, 0.9):
                by_message = silence.kept_starts(self.WHOLE, d, keep, "message")
                by_request = silence.kept_starts(self.WHOLE, d, keep, "request")
                self.assertAlmostEqual(by_message, keep * one + (2 * keep - keep * keep) * two, places=12)
                self.assertAlmostEqual(by_request, keep * (one + two), places=12)
                self.assertGreaterEqual(by_message, by_request)
                # No request shows both messages: one decision or two, the same.
                if not two:
                    self.assertEqual(by_message, by_request)
        for keep in (0, -0.25, 1.5, math.nan):
            with self.assertRaisesRegex(ValueError, "need 0 < keep <= 1"):
                silence.kept_starts(self.WHOLE, 10, keep, "message")
        with self.assertRaises(KeyError):
            silence.kept_starts(self.WHOLE, 10, 0.25, "trace")

    def test_three_simulated_recorders(self):
        rng = random.Random(2406)
        rate, windows = self.SINGLE, 60_000
        silent = {"single": 0, "message": 0, "request": 0}
        for _ in range(windows):
            # Single messages at the rate, each kept with probability 1/4.
            t, kept = rng.expovariate(rate), 0
            while t < 30 and not kept:
                kept = rng.random() < 0.25
                t += rng.expovariate(rate)
            silent["single"] += not kept
            # Requests of 10 s at half the rate, from 10 s before the window.
            starts, t = [], -10 + rng.expovariate(rate / 2)
            while t < 30:
                starts.append(t)
                t += rng.expovariate(rate / 2)
            seen = {"message": False, "request": False}
            for t in starts:
                ends = [x for x in (t, t + 10) if 0 <= x < 30]
                whole_request = rng.random() < 0.25
                seen["request"] |= whole_request and bool(ends)
                seen["message"] |= any(rng.random() < 0.25 for _ in ends)
            for unit in seen:
                silent[unit] += not seen[unit]
        for unit, share in (("single", 0.05), ("message", 0.0642), ("request", 0.1357)):
            self.assertAlmostEqual(silent[unit] / windows, share, delta=0.004, msg=unit)

    def test_the_transcript(self):
        out = printed(silence.retained)
        self.assertEqual(out, "silence of 30 s, one message in\nfour kept; effective exposure:\n"
                              "  single messages: 7.5 s\n    at most 0.39943 msg/s at 95 %\n"
                              "  pairs 10 s apart, a decision\n  for each message: 6.875 s\n"
                              "    at most 0.43574 msg/s at 95 %\n    silent at 0.39943 msg/s: 6.42 %\n"
                              "  pairs 10 s apart, a decision\n  for both messages: 5 s\n"
                              "    at most 0.59915 msg/s at 95 %\n    silent at 0.39943 msg/s: 13.57 %\n")
        text = expected("ch16_silence")
        self.assertIn(out + printed(silence.search), text)
        self.assertTrue(text.endswith(out + printed(silence.search)))
        self.assertEqual(len(text.splitlines()), 193)        # 140 in Version 24, 152, 162, 166 in Version 27


if __name__ == "__main__":
    unittest.main()
