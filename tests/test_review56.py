"""Regression tests added after the review of Version 56.

The review found that p95_all() of Listing 10.2 ranked a completion at or beyond the client's deadline before a
timeout, and so gave one value for a percentile that the record leaves between two; that the histogram of
Listing 9.1, the threshold count of Listing 10.1 and the classes of Listing 23.2 still took boolean flags for
durations, and marked an objective as met on them; that from_raw() of Listing 22.1 failed on the decimals that
its entry admits; that the entry check itself failed on a signaling NaN and on a decimal beyond the default
context; that a sort does not leave a NaN where it met it; and that the bounds of the five lines of Chapter 16
called "before rounding" were themselves rounded to five places.

Each test computes what it expects independently of the code under test: percentiles by sorting complete lists
with the unknown duration set to each end of what is known of it, shares by counting, the budget of a record of
decimals from the same record as floats, and bounds from the logarithm that defines them.
"""
import math
import random
import sys
import unittest
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from examples import ch09_histogram as h9
from examples import ch10_censoring as censoring
from examples import ch10_coordinated_omission as omission
from examples import ch22_budget as budget
from examples import ch23_case as case
from examples import ch23_mix as mix
from swmetro import stats

ROOT = Path(__file__).resolve().parent.parent
NAN = math.nan
FLAGS = [False] * 20


def rank19(xs):
    """The inverse-ECDF p95 of twenty numbers: the nineteenth of them in order."""
    assert len(xs) == 20
    return sorted(xs)[18]


class TheDeadlineOfTheClient(unittest.TestCase):
    def test_the_two_records_of_the_review(self):
        # One timeout among twenty. With a completion of 250 ms the timeout cannot move the percentile.
        self.assertEqual({rank19([100.0] * 18 + [250.0, t]) for t in (1000.0, 1001.0, 5000.0)}, {250.0})
        self.assertEqual(censoring.p95_all([100.0] * 18 + [250.0, None]), (250.0, False))
        # With a completion of 1200 ms it can: every value from 1000 to 1200 ms fits the record.
        fits = {rank19([100.0] * 18 + [1200.0, t]) for t in (1000.0, 1001.0, 1100.0, 1200.0, 1300.0, 9999.0)}
        self.assertEqual(fits, {1000.0, 1001.0, 1100.0, 1200.0})
        with self.assertRaisesRegex(ValueError, "need 0 <= x < TIMEOUT"):
            censoring.p95_all([100.0] * 18 + [1200.0, None])                    # Version 56: (1200.0, False)

    def test_what_a_completion_of_the_client_is(self):
        self.assertEqual(censoring.TIMEOUT, 1000.0)
        for late in (1000.0, 1000.5, 1200.0, -0.5, -100):
            for at in (0, 10, 20):
                xs = [100.0] * 18 + [250.0, None]
                xs.insert(at, late)
                with self.assertRaisesRegex(ValueError, "need 0 <= x < TIMEOUT"):
                    censoring.p95_all(xs)
        edge = [0] + [100.0] * 17 + [999.0, 999.999]
        self.assertEqual(rank19(edge), 999.0)
        self.assertEqual(censoring.p95_all(edge), (999.0, False))
        self.assertEqual(censoring.p95_all([100.0] * 18 + [None, None]), (1000.0, True))
        for bad, why in ((NAN, "a duration is missing"), (math.inf, "a duration is missing"),
                         (True, "no number"), ("100", "no number")):
            with self.assertRaisesRegex(ValueError, why):
                censoring.p95_all([100.0] * 18 + [bad, None])
        self.assertIs(censoring.known, stats.known)

    def test_the_report_that_drops_timeouts_has_the_same_entry(self):
        self.assertEqual(censoring.p95_completed([100.0] * 18 + [250.0, None]), 250.0)
        for late in (1000.0, 1200.0, -0.5):
            with self.assertRaisesRegex(ValueError, "need 0 <= x < TIMEOUT"):  # Version 56: 1200.0 for the second
                censoring.p95_completed([100.0] * 18 + [late, None])
        self.assertEqual(censoring.completions(iter([None, 3, 1.5, None, 2])), [1.5, 2, 3])
        self.assertEqual(censoring.completions([None, None]), [])

    def test_the_weeks_of_the_listing(self):
        rng = random.Random(10)
        for stuck, median in ((0.0, 200), (0.08, 150)):
            week = censoring.week(3000, stuck, median, rng)
            done = sorted(x for x in week if x is not None)
            self.assertTrue(all(0 <= x < censoring.TIMEOUT for x in done))
            rank = math.ceil(Fraction(95, 100) * len(week))
            want = (done[rank - 1], False) if rank <= len(done) else (censoring.TIMEOUT, True)
            self.assertEqual(censoring.p95_all(week), want)


class FlagsAreNoDurations(unittest.TestCase):
    def test_the_histogram(self):
        # Version 56: 47.5 for twenty False, with bounds at 50, 100 and 200.
        self.assertEqual(0 + 50 * Fraction(19, 20), Fraction(95, 2))             # what a zero in every place gives
        self.assertEqual(h9.bucket_quantile([0.0] * 20, 0.95, [50, 100, 200]), 47.5)
        for bad in (FLAGS, [True] * 20, [0.0] * 19 + [False], ["1"] * 20, [None]):
            with self.assertRaisesRegex(ValueError, "no number"):
                h9.bucket_quantile(bad, 0.95, [50, 100, 200])
        for gone in (NAN, math.inf):
            with self.assertRaisesRegex(ValueError, "a duration is missing"):
                h9.bucket_quantile([1.0, gone], 0.5, [50, 100, 200])
        with self.assertRaisesRegex(ValueError, "need values >= 0"):
            h9.bucket_quantile([1.0, -1.0], 0.5, [50, 100, 200])
        self.assertTrue(math.isnan(h9.bucket_quantile([], 0.5)))
        self.assertEqual(h9.bucket_quantile(iter([25, Fraction(51, 2), Decimal("30")]), 0.5, [50, 100]), 25.0)

    def test_the_sketch(self):
        for bad in (True, False, "1.0", None):
            with self.assertRaisesRegex(ValueError, "no number"):
                h9.LogSketch(0.01).add(bad)                                      # Version 56 took True for 1.0
        sk = h9.LogSketch(0.01)
        for x in (1, 1.0, Fraction(3, 2), Decimal("2.5")):
            sk.add(x)
        self.assertEqual(sk.n, 4)

    def test_the_threshold_count(self):
        self.assertEqual(omission.over([100.0] * 19 + [600.0]), (1, 95.0))
        self.assertEqual(sum(x > 500 for x in [100.0] * 19 + [600.0]), 1)
        for bad in (FLAGS, [100.0, True], ["100"], [None]):
            with self.assertRaisesRegex(ValueError, "no number"):                # Version 56: (0, 100.0) for flags
                omission.over(bad)
        for bad_limit in (True, "500", None):
            with self.assertRaisesRegex(ValueError, "no number"):
                omission.over([100.0], bad_limit)
        with self.assertRaisesRegex(ValueError, "a limit is missing"):
            omission.over([100.0], NAN)
        for gone in (NAN, math.inf):
            with self.assertRaisesRegex(ValueError, "a duration is missing"):
                omission.over([100.0, gone])
        with self.assertRaisesRegex(ValueError, "need durations >= 0"):
            omission.over([100.0, -1.0])
        with self.assertRaisesRegex(ValueError, "need durations >= 0"):
            omission.over([100.0], -500)

    def test_the_classes_of_the_weighted_example(self):
        rec = mixed = {"fast": FLAGS}
        for call in (lambda: mix.p95(mix.weighted(rec, mixed)), lambda: mix.p95(mix.weighted(rec)),
                     lambda: mix.mean(mix.weighted(rec)), lambda: mix.within(rec, mixed, 200)):
            with self.assertRaisesRegex(ValueError, "no number: False"):         # Version 56: False, 0, [1, 1], "met"
                call()
        with self.assertRaisesRegex(ValueError, "no number"):
            mix.within({"fast": [100.0, "100"]}, {"fast": [1, 1]}, 200)
        with self.assertRaisesRegex(ValueError, "no number"):
            mix.within({"fast": [100.0, (">", True)]}, {"fast": [1, 1]}, 200)

    def test_what_the_example_gives_meaning_stays(self):
        # Five requests: one within 200 ms, one without a duration, one that is no number, one known to
        # exceed 300 ms, one known to have reached 100 ms. One is within for certain, and three more may be.
        rec = {"fast": [100.0, None, NAN, (">", 300), (">=", 100)]}
        self.assertEqual(mix.within(rec, {"fast": [1] * 5}, 200), (Fraction(1, 5), Fraction(4, 5)))
        # A duration stored as infinity is beyond every limit, as in the transcript of the listing.
        rec = {"fast": [100.0] * 9 + [math.inf]}
        self.assertEqual(mix.within(rec, rec, 250), (Fraction(9, 10), Fraction(9, 10)))
        # A class without observations stands at either end with its whole weight.
        seen, ref = {"fast": [100.0] * 20}, {"fast": [1] * 18, "slow": [1] * 2}
        self.assertEqual(mix.p95(mix.weighted(seen, ref, fill=math.inf)), math.inf)
        self.assertEqual(mix.p95(mix.weighted(seen, ref, fill=-math.inf)), 100.0)
        # The weighted quantile of the package keeps its contract.
        self.assertIs(stats.weighted_quantile([(True, 1)], 0.5), True)
        self.assertEqual(mix.observed([1, 2.5, None, NAN, math.inf, (">", 3)])[0], 1)
        text = (ROOT / "expected" / "ch23_mix.txt").read_text(encoding="utf-8")
        self.assertIn("and stored as infinity:", text)


class DecimalsThroughTheBudget(unittest.TestCase):
    def report(self, xs):
        bud, r = budget.from_raw(xs)
        return r.value, r.u

    def test_the_three_records_of_the_review(self):
        floats = self.report([100.0] * 20)
        self.assertEqual(floats[0], 99.1)                                        # 100 less the 0.9 of the probe
        self.assertEqual(f"{floats[1]:.5f}", "2.53443")
        for xs in ([100] * 20, [Decimal("100.0")] * 20, [Fraction(100)] * 20, [Decimal(100)] * 10 + [100.0] * 10):
            self.assertEqual(self.report(xs), floats, xs[0])                     # Version 56: TypeError for decimals
        mixed = [Decimal(100)] * 18 + [100.0, 105.0]
        self.assertEqual(self.report(mixed), self.report([100.0] * 19 + [105.0]))
        self.assertEqual(budget.p95(mixed), 100)

    def test_a_decimal_is_subtracted_as_a_decimal(self):
        # 0.1 + 0.2 + 0.9 as floats is not 1.2; the readings and the correction are taken exactly.
        self.assertNotEqual(0.1 + 0.2, 0.3)
        xs = [Decimal("1.2")] * 20
        self.assertEqual(self.report(xs)[0], float(Fraction(12, 10) - Fraction(9, 10)))
        self.assertEqual(self.report(xs)[0], 0.3)
        self.assertEqual(self.report([Decimal("182.1")] * 20)[0], 181.2)

    def test_the_record_of_the_book_is_as_it_was(self):
        rng = random.Random(22)
        ms = budget.record(rng)
        bud, r = budget.from_raw(ms)
        self.assertEqual(r.report(), "(182.1 ± 6.5) ms, k = 2.12, p = 0.95")
        self.assertEqual(r.value, sorted(ms)[math.ceil(Fraction(95, 100) * len(ms)) - 1] - 0.9)
        text = (ROOT / "expected" / "ch22_budget.txt").read_text(encoding="utf-8")
        self.assertIn("combined u_c                 3.0610\n", text)
        self.assertEqual(budget.PROBE, 0.9)

    def test_the_edges_of_what_is_admitted(self):
        for gone in (Decimal("sNaN"), Decimal("NaN"), Decimal("Infinity"), Decimal("1e1000000"), 10 ** 400,
                     Fraction(10 ** 400, 3), NAN, math.inf, -math.inf):
            for entry in (budget.known, budget.p95, budget.from_raw, case.p95):
                with self.assertRaisesRegex(ValueError, "a duration is missing or too large"):
                    entry([100.0] * 19 + [gone])                                 # InvalidOperation and Overflow
        self.assertTrue(Decimal("1e1000000").is_finite())
        self.assertFalse(Decimal("sNaN").is_finite())
        kept = [Decimal("1e-400"), Decimal("-0.0"), sys.float_info.max, -sys.float_info.max, 10 ** 300]
        self.assertEqual(budget.known(kept), kept)
        self.assertIs(budget.known, stats.known)
        self.assertIs(case.known, stats.known)
        with self.assertRaisesRegex(ValueError, "a reading is missing or too large"):
            stats.known([NAN], "reading")


class OneRecordOfThreeTypes(unittest.TestCase):
    """Six decimals, six fractions and six floats of 100.5 ms, a decimal of 250 ms and a fraction of 260 ms."""
    MIXED = [Decimal("100.5")] * 6 + [Fraction(201, 2)] * 6 + [100.5] * 6 + [Decimal("250"), Fraction(260)]

    def test_python_orders_the_three_types_exactly(self):
        self.assertTrue(Fraction(1, 2) < Decimal("0.6") and 0.5 < Decimal("0.6") < Fraction(7, 10))
        with self.assertRaises(TypeError):
            Decimal("0.6") * Fraction(1, 2)                                      # but it does not multiply them

    def test_through_the_listings(self):
        xs = self.MIXED
        self.assertEqual(sum(x > 250 for x in xs), 1)
        self.assertEqual(sorted(xs)[18], 250)                                    # rank 19 of 20
        self.assertEqual(budget.p95(xs), 250)
        self.assertEqual(case.p95(xs), 250)
        self.assertEqual(censoring.p95_completed(xs + [None]), 250)
        self.assertEqual(censoring.p95_all(xs + [None]), (260, False))           # rank 20 of 21
        self.assertEqual(omission.over(xs, Decimal("250")), (1, 95.0))
        # Nineteen of the twenty lie in the bucket from 100 to 250, and rank 19 is its upper end.
        self.assertEqual(h9.bucket_quantile(xs, 0.95, [50, 100, 250, 500]), 100 + 150 * 19 / 19)
        self.assertEqual(budget.bracket(xs, [50, 100, 250, 500]), (100, 250))
        bud, r = budget.from_raw(xs)
        self.assertEqual(r.value, 249.1)
        self.assertEqual((r.value, r.u), tuple(vars(budget.from_raw([float(x) for x in xs])[1])[k]
                                               for k in ("value", "u")))
        sk = h9.LogSketch(0.01)
        for x in xs:
            sk.add(x)
        self.assertLessEqual(abs(sk.quantile(0.5) / 100.5 - 1), 0.01 + 1e-12)

    def test_through_the_classes(self):
        rec = {"fast": self.MIXED}
        self.assertEqual(mix.p95(mix.weighted(rec, rec)), 250)
        self.assertEqual(mix.within(rec, rec, 200), (Fraction(9, 10), Fraction(9, 10)))
        # Version 57 as first drafted: TypeError, since Python does not multiply a Decimal by a Fraction.
        self.assertEqual(float(mix.mean(mix.weighted(rec, rec))), (18 * 100.5 + 250 + 260) / 20)
        self.assertEqual(mix.mean([(Decimal("0.1"), Fraction(1)), (Decimal("0.2"), Fraction(1))]), Fraction(3, 20))
        self.assertEqual(mix.mean(mix.weighted({"fast": [100.5] * 20})), 100.5)
        self.assertIsInstance(mix.mean(mix.weighted({"fast": [100.5] * 20})), float)

    def test_what_is_no_duration_and_no_gap(self):
        for bad in (10 ** 400, Fraction(10 ** 400, 3), Decimal("1e1000000")):
            with self.assertRaisesRegex(ValueError, "a duration is missing or too large"):
                mix.observed([100.0, bad])
        gaps = [100.0, None, NAN, Decimal("NaN"), Decimal("sNaN"), math.inf, -math.inf, Decimal("Infinity"),
                (">", 300), (">=", Decimal("100"))]
        self.assertIs(mix.observed(gaps), gaps)
        # A signaling NaN is not a number either. Version 56: decimal.InvalidOperation from the comparison.
        self.assertTrue(mix.unknown(Decimal("sNaN")) and mix.unknown(Decimal("NaN")) and mix.unknown(NAN))
        self.assertFalse(mix.unknown(Decimal("Infinity")) or mix.unknown(Decimal("2.5")) or mix.unknown(0))
        # Four of five within the limit, and the fifth on either side of it.
        rec = {"fast": [100.0] * 4 + [Decimal("sNaN")]}
        self.assertEqual(mix.within(rec, rec, 200), (Fraction(4, 5), Fraction(1)))
        rec = {"fast": [100.0] * 4 + [Decimal("Infinity")]}
        self.assertEqual(mix.within(rec, rec, 200), (Fraction(4, 5), Fraction(4, 5)))


class WhereASortPutsANaN(unittest.TestCase):
    def test_it_is_moved(self):
        # The review: [1, 2, 3, nan] under one version of Python and [1, nan, 2, 3] under another. Either
        # way the NaN is not where it stood, and nothing in the language says where it will be.
        got = sorted([3, 1, NAN, 2])
        self.assertEqual(sorted(x for x in got if x == x), [1, 2, 3])
        # What holds under every version: the place of the NaN in the result follows the order of the input.
        first, last = sorted([NAN, 1, 2, 3]), sorted([1, 2, 3, NAN])
        self.assertTrue(first[0] != first[0] and last[3] != last[3])
        self.assertEqual((first[1:], last[:3]), ([1, 2, 3], [1, 2, 3]))
        doc = " ".join(stats.known.__doc__.split())
        self.assertIn("A NaN has no rank among the others", doc)
        self.assertIn("can make a percentile depend on the order of its rows", doc)
        self.assertNotIn("where it met it", doc)
        for name in ("examples/ch22_budget.py", "swmetro/stats.py"):
            self.assertNotIn("where it met it", " ".join((ROOT / name).read_text(encoding="utf-8").split()))


class WorkingPrecision(unittest.TestCase):
    def test_the_six_bounds_to_nine_places(self):
        bound = -math.log(0.05)
        got = [f"{bound / t:.9f}" for t in (1.0, 0.65, 0.75, 0.648, 0.749)] + [f"{stats.rate_upper(2, 1.0):.9f}"]
        self.assertEqual(got, ["2.995732274", "4.608818882", "3.994309698", "4.623043632", "3.999642555",
                               "6.295793622"])
        # The five places of the book, and the two of its table.
        self.assertEqual([f"{float(x):.5f}" for x in got],
                         ["2.99573", "4.60882", "3.99431", "4.62304", "3.99964", "6.29579"])

    def test_a_limit_between_the_bound_and_its_five_places(self):
        computed = stats.quiet_bound(0.75)
        self.assertAlmostEqual(computed, -math.log(0.05) / 0.75, places=12)
        self.assertTrue(computed <= 3.9943098 < 3.99431)                         # met by the bound, not by its display
        self.assertTrue(3.99 <= 3.992 < computed)                                # met by the display, not by the bound
        # Rounded upward, a printed bound is still a bound.
        up2 = math.ceil(computed * 100) / 100
        up5 = math.ceil(computed * 10 ** 5) / 10 ** 5
        self.assertEqual((up2, up5), (4.0, 3.99431))
        self.assertTrue(computed <= up5 <= up2)

    def test_the_bounds_of_figure_16_3(self):
        none, one = stats.quiet_bound(20), stats.rate_upper(1, 20)
        self.assertEqual((f"{none:.9f}", f"{one:.9f}"), ("0.149786614", "0.237193226"))
        self.assertEqual((f"{none:.5f}", f"{one:.5f}"), ("0.14979", "0.23719"))
        self.assertEqual((math.ceil(none * 1000) / 1000, math.ceil(one * 1000) / 1000), (0.15, 0.238))


if __name__ == "__main__":
    unittest.main()
