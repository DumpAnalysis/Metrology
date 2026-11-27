"""Regression tests added after the review of Version 21.

The review found that the population stability index took its shares over
ten bins whatever the number of bins it was given, so that two samples with
a third of their observations in each of three bins showed drift; that the
effective degrees of freedom and the fitted line were computed from as many
pairs as the shorter of two lists held, so that a degree of freedom left out
counted as infinitely many and one response gave a slope of zero with no
uncertainty; that the caption of Figure 23.1 gave the plateau of the second
period no right end; and that the rule for a class without durations covers
only statistics that do not fall when a duration rises. It worked two
additions: what one request decides at the marked mix, and a silence when
requests differ in duration.

Each test computes what it expects independently of the code under test: the
index from its definition, the Welch-Satterthwaite formula and the least-
squares line by hand, percentiles from cumulative weights in exact
fractions, and the probability of a silence in closed form and by a seeded
simulation.
"""
import contextlib
import io
import math
import random
import statistics as st
import unittest
from fractions import Fraction
from pathlib import Path

from examples import ch16_silence as silence
from examples import ch20_error_bars as bars
from examples import ch21_psi as drift
from examples import ch23_mix as mix
from swmetro import Budget, core, propagate, stats
from swmetro.dist import t_ppf
from swmetro.stats import quiet_bound

ROOT = Path(__file__).resolve().parent.parent


def expected(name):
    return (ROOT / "expected" / f"{name}.txt").read_text(encoding="utf-8")


def printed(function, *args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        function(*args)
    return out.getvalue()


def once(items):
    return (x for x in items)


def index_of(ref, cur, bins):
    """The index from its definition, with half a count for each bin and `bins` halves in each total."""
    n, m = sum(ref), sum(cur)
    p = [(r + 0.5) / (n + 0.5 * bins) for r in ref]
    q = [(c + 0.5) / (m + 0.5 * bins) for c in cur]
    return sum((b - a) * math.log(b / a) for a, b in zip(p, q))


def periods():
    rng = random.Random(23)
    return mix.record(mix.BEFORE, rng), mix.record(mix.AFTER, rng)


def exact_p95(classes):
    """The 95th percentile of a mixture, {name: (durations, share)}, by cumulative weights in exact fractions: the
    smallest duration at or below which at least 95 % of the weight lies."""
    pairs = sorted((x, Fraction(share) / len(xs)) for xs, share in classes.values() for x in xs)
    assert sum(w for _, w in pairs) == 1
    acc = Fraction(0)
    for x, w in pairs:
        acc += w
        if acc >= Fraction(95, 100):
            return x


def silent(rate, window, shares, rng, trials=30_000):
    """Share of trials in which a window stays empty: requests start as a Poisson process at half the message rate,
    each with a duration drawn from the shares, and write a start message and, a duration later, an end message."""
    longest = max(d for _, d in shares)
    durations = [d for _, d in shares]
    weights = [p for p, _ in shares]
    empty = 0
    for _ in range(trials):
        t, seen = -longest + rng.expovariate(rate / 2), False
        while t < window and not seen:
            d = rng.choices(durations, weights)[0]
            seen = 0 <= t < window or 0 <= t + d < window
            t += rng.expovariate(rate / 2)
        empty += not seen
    return empty / trials


class DriftInItsOwnBins(unittest.TestCase):
    def test_the_figures_of_the_review(self):
        ref, cur = [1, 1, 1], [10, 10, 10]
        self.assertAlmostEqual(drift.psi(ref, cur), 0.0, places=15)
        # What Version 21 returned: half a count in ten bins added to totals of three bins.
        self.assertEqual(f"{index_of(ref, cur, 10):.5f}", "0.15863")
        self.assertEqual((f"{1.5 / 8:.4f}", f"{10.5 / 35:.4f}"), ("0.1875", "0.3000"))
        self.assertEqual((Fraction(15, 45), Fraction(105, 315)), (Fraction(1, 3), Fraction(1, 3)))
        # The lists of the earlier test: the value with three bins, and the one with ten that the test then expected.
        ref, cur = [10, 20, 30], [12, 18, 30]
        self.assertEqual(f"{drift.psi(ref, cur):.8f}", "0.00900838")
        self.assertEqual(f"{index_of(ref, cur, 3):.8f}", "0.00900838")
        self.assertEqual(f"{index_of(ref, cur, 10):.8f}", "0.00852331")

    def test_the_shares_of_each_sample_sum_to_one(self):
        rng = random.Random(2121)
        for _ in range(300):
            bins = rng.randint(1, 25)
            ref = [rng.randint(0, 40) for _ in range(bins)]
            cur = [rng.randint(0, 400) for _ in range(bins)]
            if not (sum(ref) and sum(cur)):
                continue
            self.assertAlmostEqual(drift.psi(ref, cur), index_of(ref, cur, bins), places=12)
            self.assertAlmostEqual(drift.psi(ref, cur), drift.psi(cur, ref), places=12)      # the index is symmetric
            self.assertGreaterEqual(drift.psi(ref, cur), 0)
            self.assertAlmostEqual(drift.psi(ref, ref), 0.0, places=15)
            self.assertAlmostEqual(sum((r + 0.5) / (sum(ref) + 0.5 * bins) for r in ref), 1.0, places=12)
            # The same count in every bin is the same share in every bin, whatever the size of the sample.
            self.assertAlmostEqual(drift.psi([3] * bins, [70] * bins), 0.0, places=15)
        # With unequal shares the half counts pull a small sample further toward equal shares than a large one, and
        # the same proportions in samples of different size give an index above zero: a property of the convention.
        self.assertGreater(drift.psi([1, 2], [10, 20]), 0.005)
        self.assertAlmostEqual(drift.psi([1, 2], [10, 20]), index_of([1, 2], [10, 20], 2), places=15)

    def test_ten_bins_are_what_they_were(self):
        rng = random.Random(21)
        for _ in range(50):
            ref, cur = drift.counts(200, None, rng), drift.counts(200, drift.shifted(0.2), rng)
            self.assertEqual(len(ref), 10)
            self.assertAlmostEqual(drift.psi(ref, cur), index_of(ref, cur, 10), places=14)
        text = expected("ch21_psi")
        self.assertIn("   100  0.171  0.180     803  0.338       49\n", text)
        self.assertIn("  5000  median PSI 0.041, above crit. 300 of 300\n", text)
        self.assertEqual(len(text.splitlines()), 10)

    def test_the_two_samples_need_the_same_bins(self):
        for ref, cur in (([1, 1, 1], [10, 10]), ([1, 1], [10, 10, 10]), ([], []), ([], [1])):
            with self.assertRaisesRegex(ValueError, "need the same bins"):
                drift.psi(ref, cur)
        for ref, cur in (([0, 0, 0], [1, 2, 3]), ([1, 2, 3], [0, 0, 0])):
            with self.assertRaisesRegex(ValueError, "empty sample"):
                drift.psi(ref, cur)
        self.assertAlmostEqual(drift.psi(once([1, 1, 1]), once([10, 10, 10])), 0.0, places=15)
        # A bin that is empty in one sample still gives a finite term.
        self.assertTrue(math.isfinite(drift.psi([5, 0, 5], [0, 9, 1])))


class CompletePairs(unittest.TestCase):
    def test_the_decision_of_the_review(self):
        us = [1.0, 2.0]
        u_c = math.sqrt(5)
        both = (1 + 4) ** 2 / (1 / 5 + 16 / 7)
        self.assertEqual(f"{both:.5f}", "10.05747")
        self.assertAlmostEqual(core.welch_satterthwaite(us, [5, 7]), both, places=12)
        self.assertEqual(f"{96.10 + t_ppf(0.95, both) * u_c:.5f}", "100.15044")           # above the limit of 100 ms
        with self.assertRaisesRegex(ValueError, "one dof for each u"):
            core.welch_satterthwaite(us, [5])
        # What Version 21 made of the missing entry: no term for the second uncertainty, as if it had infinitely many.
        one = (1 + 4) ** 2 / (1 / 5)
        self.assertEqual((one, f"{96.10 + t_ppf(0.95, one) * u_c:.5f}"), (125.0, "99.80547"))
        # Infinitely many degrees of freedom are a statement, and have to be made.
        self.assertAlmostEqual(core.welch_satterthwaite(us, [5, math.inf]), 125.0, places=12)
        self.assertEqual(core.welch_satterthwaite(us, [math.inf, math.inf]), math.inf)

    def test_every_uncertainty_has_its_degrees_of_freedom(self):
        for us, dofs in (([1.0, 2.0], [5]), ([1.0], [5, 7]), ([1.0, 2.0], []), ([], [3])):
            with self.assertRaisesRegex(ValueError, "one dof for each u"):
                core.welch_satterthwaite(us, dofs)
            with self.assertRaisesRegex(ValueError, "one dof for each u"):
                core.welch_satterthwaite(once(us), once(dofs))
        for dofs in ([5, 0], [5, -2], [5, math.nan]):
            with self.assertRaisesRegex(ValueError, "dof must be > 0"):
                core.welch_satterthwaite([1.0, 2.0], dofs)
        self.assertEqual(core.welch_satterthwaite([], []), math.inf)
        self.assertEqual(core.welch_satterthwaite([0.0, 2.0], [3, 7]), 7.0)                # a component of nothing
        # The budget supplies complete pairs, and gives what it gave.
        bud = Budget("ms")
        bud.add("a", "A", 1.0, 5).add("b", "A", 2.0, 7).add("c", "B", 0.5)
        want = (1 + 4 + 0.25) ** 2 / (1 / 5 + 16 / 7)
        self.assertAlmostEqual(bud.dof(), want, places=12)

    def test_a_line_needs_pairs(self):
        times = [0, 1, 2, 3]
        with self.assertRaisesRegex(ValueError, "one y for each x"):
            stats.linfit(times, [10])                       # Version 21: slope 0, uncertainty 0
        with self.assertRaisesRegex(ValueError, "one y for each x"):
            stats.linfit([0], [10, 11, 12, 13])
        with self.assertRaisesRegex(ValueError, "one y for each x"):
            stats.linfit(once(times), once([10, 11, 12]))
        for xs, ys in (([], []), ([1], [2]), ([0, 1], [3, 5])):
            with self.assertRaisesRegex(ValueError, "need three points"):
                stats.linfit(xs, ys)
        for xs in ([2, 2, 2], [0.1, 0.1, 0.1, 0.1]):
            with self.assertRaisesRegex(ValueError, "xs do not vary"):
                stats.linfit(xs, [1, 2, 3, 4][:len(xs)])
        # Three points on a line, and three off it.
        a, b, ua, ub, s = stats.linfit([0, 1, 2], [1, 3, 5])
        self.assertEqual((a, b, ua, ub, s), (1.0, 2.0, 0.0, 0.0, 0.0))
        a, b, ua, ub, s = stats.linfit([0, 1, 2], [0, 1, 0])
        self.assertAlmostEqual(b, 0.0, places=15)
        self.assertAlmostEqual(a, 1 / 3, places=15)
        self.assertAlmostEqual(s, math.sqrt((1 / 9 + 4 / 9 + 1 / 9) / 1), places=15)       # one degree of freedom
        self.assertAlmostEqual(ub, s / math.sqrt(2), places=15)
        # The slope of the earlier test, by hand.
        ys = [2.1, 3.9, 6.2, 7.8]
        slope = (sum(t * y for t, y in zip(times, ys)) - 4 * 1.5 * st.fmean(ys)) / (sum(t * t for t in times) - 4 * 1.5 ** 2)
        self.assertAlmostEqual(stats.linfit(times, ys)[1], slope, places=12)

    def test_other_functions_that_pair_two_lists(self):
        # Found by going through every pairing of two arguments. Each returned a number from the shorter list.
        a, b = [1, 0, 1, 1], [1, 0, 0, 1]
        agree, chance = 3 / 4, (3 / 4) * (2 / 4) + (1 / 4) * (2 / 4)
        self.assertEqual(stats.cohen_kappa(a, b), (agree - chance) / (1 - chance))
        self.assertEqual(stats.cohen_kappa(once(a), once(b)), 0.5)
        for first, second in ((a, b[:2]), (a[:2], b), (a, [])):
            with self.assertRaisesRegex(ValueError, "two labels an item"):
                stats.cohen_kappa(first, second)             # Version 21: 1/3 for the first of these

        values, clusters = [1.0, 0.0, 1.0, 1.0, 0.0, 0.0], ["p", "p", "q", "q", "r", "r"]
        mean = 0.5
        sums = [1.0 + 0.0 - 2 * mean, 1.0 + 1.0 - 2 * mean, 0.0 + 0.0 - 2 * mean]
        want = math.sqrt(sum(t * t for t in sums) * 3 / 2) / 6
        self.assertAlmostEqual(bars.clustered_se(values, clusters), want, places=15)
        self.assertAlmostEqual(bars.clustered_se(once(values), once(clusters)), want, places=15)
        for vs, cs in ((values, clusters[:4]), (values[:4], clusters)):
            with self.assertRaisesRegex(ValueError, "one cluster a value"):
                bars.clustered_se(vs, cs)

        def product(x):
            return x[0] * x[1]
        y, u, cu = propagate.lpu(product, [2.0, 3.0], [0.1, 0.2])
        self.assertAlmostEqual(y, 6.0, places=15)
        self.assertAlmostEqual(u, math.hypot(3.0 * 0.1, 2.0 * 0.2), places=12)
        for us in ([0.1], [0.1, 0.2, 0.3], []):
            with self.assertRaisesRegex(ValueError, "one u for each x"):
                propagate.lpu(product, [2.0, 3.0], us)       # Version 21, one u: the second input taken as exact
        self.assertEqual(propagate.lpu(product, [2.0, 3.0], [0.1, 0.0])[1], propagate.lpu(product, [2.0, 3.0], [0.1, 0])[1])

    def test_the_callers_of_the_book_are_unchanged(self):
        for name, piece in (("ch16_trace_measurements", "E1042 period, fitted slope: 7.501(10) s\n"),
                            ("ch20_judge", "Cohen's kappa"), ("ch20_error_bars", "B - A, unpaired u"),
                            ("ch22_budget", "combined u_c")):
            self.assertIn(piece, expected(name))


class PlateauOfTheFigure(unittest.TestCase):
    def test_where_the_plateau_ends(self):
        before, after = periods()
        top = max(after["fast"])
        self.assertEqual(f"{top:.2f}", "216.25")
        below = sum(x <= top for x in after["slow"])
        self.assertEqual((below, len(after["slow"]), sum(x <= top for x in after["fast"])), (12, 900, 100))
        # All the fast requests and 12 of 900 slow ones lie at or below it: (1 - s) + s 12/900 = 95 %.
        end = Fraction(5, 100) / (1 - Fraction(below, 900))
        self.assertEqual(end, Fraction(15, 296))
        self.assertEqual(f"{100 * float(end):.5f}", "5.06757")
        self.assertEqual(mix.at_share(after, 5067, 100_000), top)
        self.assertGreater(mix.at_share(after, 5068, 100_000), top)
        self.assertEqual(f"{mix.at_share(after, 60):.2f}", "318.33")
        # In the figure it runs from 4.096 % to the right edge.
        on_it = [k for k in range(3800, 4801) if mix.at_share(after, k, 100_000) == top]
        self.assertEqual((on_it[0], on_it[-1], len(on_it)), (4096, 4800, 705))

    def test_where_the_order_turns(self):
        before, after = periods()

        def higher(k):                                       # k in units of 0.00001 %
            return mix.at_share(after, k, 10_000_000) > mix.at_share(before, k, 10_000_000)
        self.assertEqual([higher(k) for k in (406_770, 406_790, 468_220, 468_240)], [False, True, True, False])

    def test_the_figure_is_drawn_on_a_grid(self):
        source = (ROOT / "figures" / "make_figures.py").read_text(encoding="utf-8")
        for piece in ("parts = range(3800, 4801)", "ys = [c23.at_share(rec, k, 100_000) for k in parts]"):
            self.assertIn(piece, source)                     # 1001 mixes, 0.001 percentage point apart


class OneRequestLess(unittest.TestCase):
    def test_the_table_of_the_review(self):
        before, after = periods()
        fast, slow = sorted(after["fast"]), after["slow"]
        share = {"fast": Fraction(959, 1000), "slow": Fraction(41, 1000)}
        early = exact_p95({"fast": (before["fast"], share["fast"]), "slow": (before["slow"], share["slow"])})
        whole = exact_p95({"fast": (fast, share["fast"]), "slow": (slow, share["slow"])})
        self.assertEqual((f"{early:.2f}", f"{whole:.2f}", f"{whole - early:+.2f}"), ("194.92", "216.25", "+21.33"))
        want = [exact_p95({"fast": (fast[:i] + fast[i + 1:], share["fast"]), "slow": (slow, share["slow"])}) for i in range(100)]
        self.assertEqual({f"{x:.2f}" for x in want[:-1]}, {"216.25"})                     # any one of the other 99
        self.assertEqual((f"{want[-1]:.2f}", f"{want[-1] - early:+.2f}"), ("159.62", "-35.30"))
        self.assertEqual(mix.less_one(after, 41), want)
        self.assertEqual(mix.at_share(after, 41), whole)
        self.assertEqual(len(after["fast"]), 100)                                         # the record is left as it was

    def test_one_request_decides_nowhere_else_in_the_table(self):
        # At the mixes that the periods had, no single fast request of the second period turns the comparison.
        before, after = periods()
        for slow in (100, 900):
            early = mix.at_share(before, slow)
            self.assertTrue(all(x < early for x in mix.less_one(after, slow)), slow)
            self.assertLess(mix.at_share(after, slow), early)

    def test_the_transcript(self):
        before, after = periods()
        out = printed(mix.one_mix, before, after)
        self.assertTrue(out.endswith("at 4.1 %, with one fast request\nof the 100 after left out:\n  the largest: 159.62 ms\n"
                                     "  any other:   216.25 ms\n"))
        self.assertIn(out, expected("ch23_mix"))
        self.assertEqual(len(expected("ch23_mix").splitlines()), 126)   # 103 until Version 57, 108 until 65, 122 until 69


class StatisticsTheRuleCovers(unittest.TestCase):
    KNOWN = [100.0] * 94 + [120.0] * 5
    MIX = {"known": range(99), "lost": range(1)}

    @staticmethod
    def error_p95(pairs):
        return mix.p95([(abs(x - 100.0), w) for x, w in pairs])

    def test_the_example_of_the_review(self):
        rec = {"known": self.KNOWN}
        ends = [self.error_p95(mix.weighted(rec, self.MIX, fill=x)) for x in (-math.inf, math.inf)]
        self.assertEqual(ends, [20.0, 20.0])                 # the two ends agree, and do not enclose
        at_target = self.error_p95(mix.weighted(dict(rec, lost=[100.0]), self.MIX))
        self.assertEqual(at_target, 0.0)
        # The remedy of the review: records of the errors, and the missing error anywhere from nothing upward.
        errors = {"known": [abs(x - 100.0) for x in self.KNOWN]}
        low, top = (mix.p95(mix.weighted(errors, self.MIX, fill=x)) for x in (0.0, math.inf))
        self.assertEqual((low, top), (0.0, 20.0))
        self.assertEqual(mix.p95(mix.weighted(errors, self.MIX, fill=-math.inf)), 0.0)   # below every error is as good
        out = printed(mix.components, errors, dict(errors, lost=[3.0]), mix.p95, "p95", "distributions")
        self.assertIn("  mix first: not determined,\n    no lost requests before\n", out)

    def test_the_condition_is_stated_where_the_statistic_is_given(self):
        for function in (mix.components, mix.weighted):
            text = " ".join(function.__doc__.split())
            self.assertIn("does not fall when a duration rises", text)

    def test_the_mean_and_the_percentile_meet_it(self):
        rng = random.Random(2123)
        for _ in range(300):
            pairs = [(float(rng.randint(1, 500)), Fraction(rng.randint(1, 9))) for _ in range(rng.randint(1, 30))]
            i = rng.randrange(len(pairs))
            raised = list(pairs)
            raised[i] = (pairs[i][0] + rng.randint(1, 400), pairs[i][1])
            for stat in (mix.mean, mix.p95):
                self.assertLessEqual(stat(pairs), stat(raised))
            self.assertGreaterEqual(self.error_p95(pairs), 0)
        # The distance from a target does fall when a duration below the target rises.
        self.assertLess(self.error_p95([(100.0, 1)]), self.error_p95([(60.0, 1)]))


class RequestsOfSeveralDurations(unittest.TestCase):
    WHOLE = [(0, 30)]
    CASES = (([(1, 30)], 60.0, "0.09986"), ([(0.5, 10), (0.5, 50)], 50.0, "0.11983"), ([(0.9, 1), (0.1, 291)], 33.9, "0.17674"))

    def test_the_table_of_the_review(self):
        for shares, covered, bound in self.CASES:
            self.assertAlmostEqual(sum(p * d for p, d in shares), 30.0, places=12)       # one mean duration
            want = 30 + sum(p * min(d, 30) for p, d in shares)                           # T + E[min(D, T)]
            self.assertAlmostEqual(want, covered, places=12)
            self.assertAlmostEqual(silence.mean_reach(self.WHOLE, shares), covered, places=12)
            self.assertEqual(f"{2 * -math.log(0.05) / covered:.5f}", bound)
            self.assertEqual(f"{2 * quiet_bound(silence.mean_reach(self.WHOLE, shares)):.5f}", bound)
        # At the bound for the mean duration the equal mixture is silent more often than one time in twenty.
        low = 2 * -math.log(0.05) / 60
        self.assertEqual(f"{100 * math.exp(-low / 2 * 50):.2f}", "8.24")
        self.assertAlmostEqual(silence.paired_bound(self.WHOLE, 30), low, places=15)

    def test_the_mean_of_the_exposures_and_the_exposure_of_the_mean(self):
        rng = random.Random(2116)
        for _ in range(300):
            window = rng.choice((5, 30, 60))
            ds = [rng.choice((0, 0.5, 2.346, 10, 30, 50, 90, 291)) for _ in range(rng.randint(1, 4))]
            ws = [rng.randint(1, 9) for _ in ds]
            shares = [(w / sum(ws), d) for w, d in zip(ws, ds)]
            got = silence.mean_reach([(0, window)], shares)
            self.assertAlmostEqual(got, window + sum(p * min(d, window) for p, d in shares), places=9)
            mean = sum(p * d for p, d in shares)
            self.assertLessEqual(got, silence.reach([(0, window)], mean) + 1e-9)          # never above, for one window
            self.assertAlmostEqual(silence.mean_reach([(0, window)], [(1, ds[0])]), silence.reach([(0, window)], ds[0]), places=12)
        # Stretches with a gap: the mean of reach(), whatever its shape.
        spans = [(0, 10), (20, 30)]
        shares = [(0.25, 2), (0.5, 10), (0.25, 20)]
        want = 0.25 * silence.reach(spans, 2) + 0.5 * silence.reach(spans, 10) + 0.25 * silence.reach(spans, 20)
        self.assertEqual((silence.reach(spans, 2), silence.reach(spans, 10), silence.reach(spans, 20)), (24, 40, 30))
        self.assertAlmostEqual(silence.mean_reach(spans, shares), want, places=12)
        self.assertAlmostEqual(silence.mean_reach(once(spans), once(shares)), want, places=12)

    def test_shares_are_shares(self):
        for shares in ([(0.5, 10), (0.4, 50)], [(0.5, 10), (0.6, 50)], [(1.5, 10), (-0.5, 50)], [(2, 30)]):
            with self.assertRaisesRegex(ValueError, "shares must sum to 1"):
                silence.mean_reach(self.WHOLE, shares)
        self.assertEqual(silence.mean_reach(self.WHOLE, [(Fraction(1, 3), 10), (Fraction(2, 3), 40)]), 30 + Fraction(10, 3) + 20)

    def test_a_simulated_source(self):
        rng = random.Random(2117)
        for shares, covered, _ in self.CASES[:2]:
            rate = 2 * -math.log(0.05) / covered
            self.assertAlmostEqual(silent(rate, 30, shares, rng), 0.05, delta=0.008, msg=shares)
        # The bound for the mean duration, applied to the equal mixture.
        low = 2 * -math.log(0.05) / 60
        self.assertAlmostEqual(silent(low, 30, self.CASES[1][0], rng), 0.0824, delta=0.008)
        # The third source, with its long requests: fewer trials, a wider band.
        shares, covered, _ = self.CASES[2]
        self.assertAlmostEqual(silent(2 * -math.log(0.05) / covered, 30, shares, rng, 8000), 0.05, delta=0.012)

    def test_the_transcript(self):
        text = expected("ch16_silence")
        out = printed(silence.durations)
        self.assertEqual(out, "silence of 30 s, requests with a\nmean duration of 30 s:\n  all of 30 s:\n    starts covered: 60 s\n"
                              "    at most 0.09986 msg/s at 95 %\n  half of 10 s, half of 50 s:\n    starts covered: 50 s\n"
                              "    at most 0.11983 msg/s at 95 %\n  0.9 of 1 s, 0.1 of 291 s:\n    starts covered: 33.9 s\n"
                              "    at most 0.17674 msg/s at 95 %\n  at the first bound the second\n  stays silent: 8.24 %\n")
        self.assertIn(out, text)
        self.assertEqual(len(text.splitlines()), 193)            # 101 in Version 22, 118, 140, 152, 162, 166 in Version 27

    def test_the_figures(self):
        source = (ROOT / "figures" / "make_figures.py").read_text(encoding="utf-8")
        for piece in ('save(fig, "fig-16-2-mean-duration.png")', 'save(fig, "fig-16-3-counted-where.png")',
                      "two = c16s.mean_reach(whole, [(0.5, 10), (0.5, 50)])", "[c16s.reach(whole, d) for d in ds]"):
            self.assertIn(piece, source)
        self.assertNotIn("fig-16-2-counted-where.png", source)
        self.assertEqual(source.count("\n    save(fig, "), 32)


if __name__ == "__main__":
    unittest.main()
