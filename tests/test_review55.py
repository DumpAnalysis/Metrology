"""Regression tests added after the review of Version 55.

The review found that the percentile helpers of Listings 10.2 and 23.1 still sorted a record with a missing
duration, as Listing 22.1 had until Version 55; that known() of Listing 22.1 took booleans and text for
durations; that bracket() lost the bounds of its buckets when they came as an iterator; that the verified time
of the five lines of Chapter 16 is a time of stamps, which are truncated, and that the time of the events is
shorter by up to a step for each stretch that begins on a line; that a bound printed as it rounds is not itself
a bound; that the status "unresolved" of the record checker also covers what the checker cannot evaluate, and
that a changed estimate settles an undecided result as a smaller uncertainty does, while at the limit itself a
smaller uncertainty settles nothing; and that the table on the sketch of Listing 9.1 did not name the error
that is bounded, and stated a tolerance of the tests as if it were a bound.

Each test computes what it expects independently of the code under test: percentiles by sorting complete lists
of numbers, the guaranteed time of the events from every pair of event times on a grid within their
milliseconds, bounds from the logarithm that defines them, guard bands from limits within which the coverage
factor is known to lie, and the report of the sketch from powers of an exact fraction.
"""
import copy
import json
import math
import random
import sys
import unittest
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from examples import ch09_histogram as h9
from examples import ch10_censoring as censoring
from examples import ch16_trace_measurements as trace
from examples import ch22_budget as budget
from examples import ch23_case as case
from examples import ch23_mix as mix
from swmetro import record as R
from swmetro import stats

ROOT = Path(__file__).resolve().parent.parent
NAN = math.nan
KNOWN = [100.0] * 18 + [250.0]


def rank19(xs):
    """The inverse-ECDF p95 of twenty numbers: the nineteenth of them in order."""
    assert len(xs) == 20
    return sorted(xs)[18]


def stored():
    return R.read((ROOT / "data" / "measurement_record.json").read_text(encoding="utf-8"))


class WhatCountsAsADuration(unittest.TestCase):
    ENTRIES = (budget.p95, budget.from_raw, budget.known, lambda xs: budget.bracket(xs, budget.FITTED))

    def test_a_column_of_booleans(self):
        # Version 55: p95 False, a budget of -0.9 ms with u = 2.53 ms, and "met" from the fitted buckets.
        falses = json.loads("[" + ", ".join(["false"] * 2000) + "]")
        self.assertTrue(all(x is False for x in falses))
        for entry in self.ENTRIES:
            with self.assertRaisesRegex(ValueError, "no number: False"):
                entry(falses)
        for entry in self.ENTRIES:
            with self.assertRaisesRegex(ValueError, "no number: True"):
                entry(KNOWN + [True])

    def test_text_and_none(self):
        texts = ["10"] * 18 + ["2", "30"]
        self.assertEqual(rank19(texts), "2")                  # what a sort of text makes of them; Version 55
        self.assertEqual(rank19([10] * 18 + [2, 30]), 10)     # the same entries as numbers
        self.assertEqual(budget.p95([10] * 18 + [2, 30]), 10)
        for bad in (texts, KNOWN + [None], KNOWN + ["250"], KNOWN + [[250.0]], KNOWN + [2 + 3j], KNOWN + [b"1"]):
            for entry in self.ENTRIES:
                with self.assertRaisesRegex(ValueError, "no number"):
                    entry(bad)

    def test_a_number_that_is_not_known(self):
        for gone in (NAN, math.inf, -math.inf, Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")):
            for at in (0, 10, 19):
                xs = KNOWN[:at] + [gone] + KNOWN[at:]
                for entry in self.ENTRIES:
                    with self.assertRaisesRegex(ValueError, "a duration is missing"):
                        entry(xs)

    def test_numbers_keep_their_type_and_their_sign(self):
        xs = [Decimal("199.99999999999999999"), Fraction(1, 3), -2, 0, 0.5, -0.25, 10 ** 30]
        got = budget.known(iter(xs))
        self.assertEqual(got, xs)
        self.assertTrue(all(a is b for a, b in zip(got, xs)))
        # A corrected indication can be below zero, and the first bucket has no lower end for it.
        self.assertEqual(budget.bracket([-5.0] * 20, [0, 50]), (-math.inf, 0))
        self.assertEqual(budget.p95([-5.0] * 19 + [3.0]), -5.0)
        self.assertEqual(rank19([-5.0] * 19 + [3.0]), -5.0)

    def test_every_place_of_a_missing_duration(self):
        base = [100.0] * 1899 + [250.0] * 100
        for at in list(range(20)):
            xs = KNOWN[:at] + [NAN] + KNOWN[at:]
            for function in (budget.p95, case.p95, censoring.p95_all, censoring.p95_completed):
                with self.assertRaises(ValueError):
                    function(xs)
        for at in list(range(0, 2000, 97)) + [1898, 1899, 1900, 1999]:
            xs = base[:at] + [NAN] + base[at:]
            for function in (budget.p95, case.p95, censoring.p95_all):
                with self.assertRaises(ValueError):
                    function(xs)


class TheOtherTwoHelpers(unittest.TestCase):
    def test_chapter_23(self):
        # Version 55: 100 with the NaN first, 250 with it last.
        self.assertEqual(rank19(KNOWN + [1.0]), 100.0)
        self.assertEqual(rank19(KNOWN + [9999.0]), 250.0)
        for xs in ([NAN] + KNOWN, KNOWN + [NAN], [100.0] * 18 + [NAN, 250.0]):
            with self.assertRaisesRegex(ValueError, "a duration is missing"):
                case.p95(xs)
        self.assertIs(case.known, budget.known)
        self.assertEqual(case.p95(KNOWN + [100.0]), 100.0)
        self.assertEqual(case.p95(x for x in KNOWN + [300.0]), 250.0)
        with self.assertRaisesRegex(ValueError, "no number"):
            case.p95(KNOWN + [False])

    def test_chapter_10_keeps_its_timeout(self):
        # None is a request that ran into the timeout: it is ranked after every completion.
        self.assertEqual(censoring.p95_all(KNOWN + [None]), (250.0, False))
        self.assertEqual(censoring.p95_all([100.0] * 18 + [None, None]), (censoring.TIMEOUT, True))
        self.assertEqual(censoring.p95_completed([100.0] * 18 + [250.0, None, None]), 250.0)
        self.assertEqual(sorted([100.0] * 18 + [250.0])[math.ceil(0.95 * 19) - 1], 250.0)
        # A NaN is no timeout. Version 55: (100.0, False) with it first and (250.0, False) with it last.
        for xs in ([NAN] + KNOWN, KNOWN + [NAN], [100.0] * 18 + [NAN, 250.0]):
            for function in (censoring.p95_all, censoring.p95_completed):
                with self.assertRaisesRegex(ValueError, "a duration is missing"):
                    function(xs)

    def test_chapter_10_states_what_a_completion_is(self):
        self.assertIs(censoring.known, budget.known)
        for bad, why in ((math.inf, "a duration is missing"), (True, "no number"), ("100", "no number"),
                         ([1], "no number")):
            for function in (censoring.p95_all, censoring.p95_completed):
                with self.assertRaisesRegex(ValueError, why):
                    function(KNOWN + [bad])
        # The week of the listing is read as before: its completions in order, its timeouts counted.
        rng = random.Random(10)
        week = censoring.week(2000, 0.08, 150, rng)
        done = sorted(x for x in week if x is not None)
        rank = math.ceil(Fraction(95, 100) * 2000)
        self.assertEqual(rank, 1900)
        want = (done[rank - 1], False) if rank <= len(done) else (censoring.TIMEOUT, True)
        self.assertEqual(censoring.p95_all(week), want)
        self.assertEqual(censoring.p95_completed(week), done[math.ceil(Fraction(95, 100) * len(done)) - 1])
        self.assertIn("A NaN is no timeout", " ".join(censoring.p95_all.__doc__.split()))

    def test_the_weighted_percentile_of_listing_23_2_already_refused(self):
        # weighted_quantile() of the package refuses a missing value since Version 49, and is unchanged.
        self.assertEqual(mix.p95([(x, 1) for x in KNOWN + [100.0]]), 100.0)
        for bad in (NAN, None, "100"):
            for at in (0, 20):
                xs = KNOWN + [100.0]
                xs.insert(at, bad)
                with self.assertRaisesRegex(ValueError, "a value is no number"):
                    mix.p95([(x, 1) for x in xs])


class BoundsThatAreReadOnce(unittest.TestCase):
    def test_the_brackets_of_the_review(self):
        xs = [100.0] * 20
        self.assertEqual(sum(x <= 100 for x in xs), 20)        # all in (50, 100]
        for bounds in (budget.FITTED, tuple(budget.FITTED), iter(budget.FITTED), (b for b in budget.FITTED)):
            got = budget.bracket(xs, bounds)
            self.assertEqual(got, (50, 100))                   # Version 55 with an iterator: (-inf, inf)
            self.assertEqual(budget.verdict(*got), "met")
        self.assertEqual(budget.bracket(iter(xs), iter(budget.COARSE)), (50, 100))

    def test_exact_edges_from_an_iterator(self):
        below = Decimal("199.99999999999999999")
        got = budget.bracket([200.0] * 20, iter([below, 250]))
        self.assertEqual(got, (below, 250))
        self.assertIs(got[0], below)
        self.assertEqual(budget.verdict(*got), "unresolved")
        for bad in ([100, 100], [250, 100], [100, NAN]):
            with self.assertRaisesRegex(ValueError, "bounds do not rise"):
                budget.bracket([150.0] * 20, iter(bad))

    def test_the_interpolation_helper(self):
        xs = [0.1] * 20
        want = h9.bucket_quantile(xs, 0.95, list(h9.BOUNDS))
        # Rank 19 of 20 in the bucket from 0.05 to 0.1: nineteen twentieths of the way.
        self.assertAlmostEqual(want, 0.05 + 0.05 * 19 / 20, places=12)
        for bounds in (tuple(h9.BOUNDS), iter(h9.BOUNDS), (b for b in h9.BOUNDS)):
            self.assertEqual(h9.bucket_quantile(xs, 0.95, bounds), want)        # Version 55: TypeError
        with self.assertRaisesRegex(ValueError, "no rising bounds >= 0"):
            h9.bucket_quantile(xs, 0.95, iter([100, 2]))


class TwoTimeScales(unittest.TestCase):
    LINES = ("000041 2204 09:15:00.100 svc  request 9 start",
             "000042 3120 09:15:00.250 db   query 3310 done",
             "000044 2204 09:15:00.400 svc  request 9 start",
             "000045 2204 09:15:00.700 svc  request 9 end",
             "000046 3120 09:15:00.900 db   query 3311 done")
    SPAN = (trace.stamp("09:15:00.000"), trace.stamp("09:15:01.000"))

    def stretches(self, counter):
        """Verified stretches by the stamps, in ms of the capture, each with whether it begins on a line."""
        rows = [trace.parse(x) for x in self.LINES]
        stamps = {seq: ms - self.SPAN[0] for seq, ms, _, _ in rows}
        out = []
        for x, y, lines in trace.runs(rows, 0, 1, self.SPAN, counter):
            out.append((x, y, stamps[lines[0]] == x))
        return out

    def test_by_the_stamps(self):
        self.assertEqual(self.stretches(None), [(100, 250, True), (400, 900, True)])
        self.assertEqual(self.stretches((41, 48)), [(0, 250, False), (400, 900, True)])
        rows = [trace.parse(x) for x in self.LINES]
        self.assertEqual(trace.verified(rows, 0, 1, self.SPAN), 0.65)
        self.assertEqual(trace.verified(rows, 0, 1, self.SPAN, (41, 48)), 0.75)

    def test_what_is_certain_of_the_events(self):
        # A line stamped s was written at some time in [s, s + 1) ms. Whatever the two times are, the
        # events of a run certainly span [s1 + 1, s2], and no more: on a grid of sixteenths the common
        # part of all the spans is [s1 + 15/16, s2].
        grid = [Fraction(k, 16) for k in range(16)]
        common = (max(400 + a for a in grid), min(900 + b for b in grid))
        self.assertEqual(common, (400 + Fraction(15, 16), 900))
        sure = {}
        for name, counter in (("lines", None), ("readings", (41, 48))):
            sure[name] = sum(y - x - (1 if on_a_line else 0) for x, y, on_a_line in self.stretches(counter))
        self.assertEqual(sure, {"lines": 648, "readings": 749})
        shown = {name: f"{-math.log(0.05) / (ms / 1000):.5f}" for name, ms in sure.items()}
        self.assertEqual(shown, {"lines": "4.62304", "readings": "3.99964"})
        self.assertAlmostEqual(stats.quiet_bound(0.648), -math.log(0.05) / 0.648, places=12)
        # To the nearest hundredth, as the text of the chapter had the four bounds until Version 70: three of the
        # four lie below what they bound. Since Version 71 the text has them upward at three places, as its table
        # has had them since Version 70 (tests/test_review70.py).
        self.assertEqual((f"{stats.quiet_bound(0.648):.2f}", f"{stats.quiet_bound(0.749):.2f}"), ("4.62", "4.00"))
        self.assertEqual((f"{stats.quiet_bound(0.65):.2f}", f"{stats.quiet_bound(0.75):.2f}"), ("4.61", "3.99"))
        self.assertEqual([stats.upward(stats.quiet_bound(t), 3) for t in (0.749, 0.648, 0.75, 0.65)],
                         ["4.000", "4.624", "3.995", "4.609"])
        self.assertIn("by the stamps", " ".join(trace.verified.__doc__.split()))

    def test_an_end_inside_a_step(self):
        # Two events of one millisecond lie on either side of 2.0075 s and bear one stamp.
        self.assertEqual({int(t) for t in (2007.25, 2007.75)}, {2007})
        self.assertTrue(2007.25 < 2007.5 < 2007.75)
        rows = [(1, 2007, "svc", "x"), (2, 2007, "svc", "y")]
        self.assertEqual(trace.count(rows, "svc", 0, 2.0075, (0, 3000)), 2)
        self.assertEqual(trace.count(rows, "svc", 2.0075, 3, (0, 3000)), 0)
        self.assertEqual(trace.window(0, 2.0075, (0, 3000)), (0, Fraction(4015, 2)))

    def test_a_bound_as_it_rounds_is_no_bound(self):
        working = {"3.00": -math.log(0.05), "4.61": -math.log(0.05) / 0.65, "3.99": -math.log(0.05) / 0.75,
                   "6.30": stats.rate_upper(2, 1.0)}
        self.assertEqual({k: f"{v:.5f}" for k, v in working.items()},
                         {"3.00": "2.99573", "4.61": "4.60882", "3.99": "3.99431", "6.30": "6.29579"})
        for shown, value in working.items():
            self.assertEqual(f"{value:.2f}", shown)
        # A limit of 3.992: the number as it prints is below it, and the bound is not.
        self.assertTrue(3.99 <= 3.992 < working["3.99"])
        self.assertEqual(math.ceil(working["3.99"] * 100) / 100, 4.0)
        one = stats.rate_upper(1, 20)
        self.assertEqual((f"{stats.quiet_bound(20):.5f}", f"{one:.5f}", f"{one:.3f}"), ("0.14979", "0.23719", "0.237"))
        self.assertTrue(0.237 < one)
        # The 0.237 stood in the text until Version 70, below the bound. Version 71 has it upward, as the caption
        # of Figure 16.3 had it.
        self.assertEqual(stats.upward(one, 3), "0.238")
        self.assertEqual(f"{stats.quiet_bound(30):.5f}", "0.09986")                # the 0.100 of the chapter


class TheCheckerAndTheRule(unittest.TestCase):
    def test_what_the_checker_cannot_evaluate(self):
        rec = stored()
        rec["decision"]["risk"] = Decimal("1e-301")
        self.assertEqual((R.missing(rec), R.invalid(rec)), ([], []))
        self.assertEqual(R.decide(rec), ("unresolved", ["risk not supported"]))
        rec = stored()
        rec["decision"]["procedure"] = "simple acceptance"
        self.assertEqual(R.decide(rec), ("unresolved", ["procedure not supported"]))
        rec = stored()
        rec["covariances"] = [{"of": ["a", "b"], "r": 0.5}]
        self.assertEqual(R.decide(rec), ("unresolved", ["correlations not supported"]))
        # Nothing is missing in any of the three: no further observation would resolve them.
        self.assertEqual(R.missing(rec), [])

    def test_the_estimate_settles_it_at_one_uncertainty(self):
        rec = stored()
        limit, u = rec["decision"]["limit"], float(rec["result"]["u"])
        self.assertEqual((limit, f"{u:.6f}"), (200, "3.061021"))
        # One-sided at 95 % with more than fifteen degrees of freedom the factor lies between the normal
        # 1.645 and the 1.76 of fourteen: the band is between 5.0 and 5.4 ms.
        self.assertGreater(float(rec["result"]["dof"]), 15)
        low, high = 1.645 * u, 1.76 * u
        self.assertTrue(5.0 < low < high < 5.4)
        want = {}
        for value in (190, 200, 210):
            if value + high <= limit:
                want[value] = "accept"
            elif value - high > limit:
                want[value] = "reject"
            else:
                self.assertTrue(value + low > limit and value - low <= limit)
                want[value] = "undecided"
        self.assertEqual(want, {190: "accept", 200: "undecided", 210: "reject"})
        for value, status in want.items():
            rec = stored()
            rec["result"]["value"] = value
            self.assertEqual(R.decide(rec), (status, []), value)
            self.assertEqual(float(R.combined(rec)[0]), float(R.combined(stored())[0]))

    def test_at_the_limit_a_smaller_uncertainty_settles_nothing(self):
        # With one degree of freedom and a risk of one quarter the factor is exactly 1.
        for u in (1, Decimal("0.1"), Decimal("0.000001")):
            rec = stored()
            rec["result"].update(value=200, u=u, dof=1)
            rec["decision"]["risk"] = Decimal("0.25")
            rec["components"] = [{"name": "one", "type": "B", "kind": "stipulated", "u": u}]
            self.assertTrue(200 + u > 200 and not 200 - u > 200)
            self.assertEqual(R.decide(rec), ("undecided", []), u)
        rec["result"]["u"] = rec["components"][0]["u"] = 0
        self.assertEqual(R.decide(rec), ("accept", []))


class TheErrorThatTheSketchBounds(unittest.TestCase):
    def single(self, x, alpha):
        sk = h9.LogSketch(alpha)
        sk.add(x)
        return sk.quantile(0.5)

    def test_one_value_of_a_thousand(self):
        # The bucket of 1000 at 1 %: the least power of g = 1.01 / 0.99 that reaches 1000, times 0.99.
        g, power, i = Fraction(101, 99), Fraction(1), 0
        while power < 1000:
            power, i = power * g, i + 1
        report = float(power * Fraction(99, 100))
        got = self.single(1000.0, 0.01)
        self.assertAlmostEqual(got / report, 1, places=12)
        self.assertEqual(f"{got:.2f}", "1002.43")
        self.assertEqual(f"{got - 1000:.2f}", "2.43")                      # the error, in the unit of the value
        self.assertEqual(f"{100 * (got - 1000) / 1000:.2f}", "0.24")       # and relative to it, in percent
        self.assertLess(abs(got - 1000) / 1000, 0.01)
        # A value at the upper end of its bucket is reported the whole accuracy below itself: 1 as 0.99.
        one = self.single(1.0, 0.01)
        self.assertEqual(f"{one:.2f}", "0.99")
        self.assertAlmostEqual(abs(one - 1.0) / 1.0, 0.01, places=12)

    def test_the_tolerance_of_the_tests(self):
        rng = random.Random(5506)
        lo, hi = math.log(sys.float_info.min), math.log(sys.float_info.max)
        extra, worst = Fraction(1, 10 ** 12), 0
        alphas = [0.9, 0.5, 0.25, 0.1, 0.05, 0.01, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-9, 1e-10, 1e-12, 1e-14, 1e-16]
        for alpha in alphas:
            for _ in range(150):
                x = math.exp(rng.uniform(lo, hi))
                excess = abs(Fraction(self.single(x, alpha)) / Fraction(x) - 1) - Fraction(repr(alpha))
                self.assertLessEqual(excess, extra, (alpha, x))
                worst = max(worst, excess)
        self.assertLess(worst, extra / 2)
        doc = " ".join(h9.LogSketch.__doc__.split())
        self.assertIn("|reported - x| / x", doc)
        self.assertIn("a tolerance observed", doc)


class WhatTheRepositoryHolds(unittest.TestCase):
    def test_the_two_optional_checks_have_their_place(self):
        for name in ("certify_t_quantile.py", "certify_coverage_factor.py"):
            text = (ROOT / "tools" / name).read_text(encoding="utf-8")
            self.assertIn(f"python tools/{name}", text)
        for name in ("README.md", "data/RECORD_FORMAT.md"):
            self.assertIn("tools/certify_coverage_factor.py", (ROOT / name).read_text(encoding="utf-8"))

    def test_the_readme_names_what_is_there(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for path in ("swmetro/", "examples/", "expected/", "data/", "figures/", "tests/", "tools/", "run_all.py",
                     "LICENSE", ".github/workflows/replay.yml"):
            self.assertIn(f"| `{path}` |", readme, path)
            self.assertTrue((ROOT / path).exists(), path)


if __name__ == "__main__":
    unittest.main()
