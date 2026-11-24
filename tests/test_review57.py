"""Regression tests added after the review of Version 57.

The review found that observed() of Listing 23.2, new in Version 57, read a class given as an iterator to the end
while it checked it and handed the empty iterator on, so that two slow requests of twenty were lost and an objective
that the twenty miss was reported as met; that recorded() of Listing 23.1 added a flag to a duration when the
duration starts at accept, and read any start that it did not know as the handler's; that the budget of Listing
22.1 gave every record the storage step of its own probe; that decimals failed in the arithmetic of Listing 23.1
and as the edges of the histogram of Listing 9.1; that a boolean limit and a malformed bound passed the counting
of Listing 23.2; and that the table of Chapter 22 ranked a timeout after every completion, which holds for the
client of Chapter 10 alone.

Each test computes what it expects independently of the code under test: percentiles and shares by sorting and
counting complete lists, the interpolation of a bucket from its two ends, uncertainties from the components of
the budget, and bounds from the expressions that define them.
"""
import inspect
import math
import random
import statistics as st
import unittest
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from examples import ch09_histogram as h9
from examples import ch10_censoring as censoring
from examples import ch22_budget as budget
from examples import ch23_case as case
from examples import ch23_mix as mix
from swmetro import stats

ROOT = Path(__file__).resolve().parent.parent
NAN = math.nan
FAST, SLOW = [100] * 18, [500, 500]


def gen(xs):
    """A generator: a class that can be read once."""
    yield from xs


def p95_of(xs):
    """The inverse-ECDF p95 of a complete list, by sorting it."""
    xs = sorted(xs)
    return xs[math.ceil(Fraction(95, 100) * len(xs)) - 1]


class ARequestIsNotLostToTheCheck(unittest.TestCase):
    FORMS = (list, tuple, iter, gen)

    def test_the_record_of_the_review(self):
        self.assertEqual(p95_of(FAST + SLOW), 500)
        self.assertEqual(sum(x <= 200 for x in FAST + SLOW), 18)
        for form in self.FORMS:
            pairs = mix.weighted({"fast": form(FAST), "slow": form(SLOW)})
            self.assertEqual(len(pairs), 20, form)                               # Version 57: 18 for an iterator
            self.assertEqual({w for _, w in pairs}, {1}, form)
            self.assertEqual(mix.p95(pairs), 500, form)                          # Version 57: 100
            self.assertEqual(float(mix.mean(pairs)), (18 * 100 + 2 * 500) / 20, form)

    def test_count_and_objective_at_a_reference_mix(self):
        ref = {"fast": range(18), "slow": range(2)}
        for form in self.FORMS:
            rec = {"fast": form(FAST), "slow": form(SLOW)}
            low, high = mix.within(rec, ref, 200)                                # Version 57: TypeError for iter
            self.assertEqual((low, high, mix.verdict(low, high)), (Fraction(9, 10), Fraction(9, 10), "missed"), form)
            rec = {"fast": form(FAST), "slow": form(SLOW)}
            pairs = mix.weighted(rec, ref)
            self.assertEqual((len(pairs), mix.p95(pairs)), (20, 500), form)
        # Twice as many slow requests observed as the mix has: each counts for half.
        pairs = mix.weighted({"fast": iter(FAST), "slow": iter(SLOW * 2)}, ref)
        self.assertEqual(sorted({w for _, w in pairs}), [Fraction(1, 2), 1])
        self.assertEqual(sum(w for _, w in pairs), 20)

    def test_a_list_is_handed_back_as_it_is(self):
        xs = [100, None, NAN, math.inf, (">", 300), (">=", Decimal("100"))]
        self.assertIs(mix.observed(xs), xs)
        self.assertEqual(len(xs), 6)
        for form in (tuple, iter, gen):
            got = mix.observed(form(xs[:1] + xs[3:]))
            self.assertIsInstance(got, list)
            self.assertEqual(got, xs[:1] + xs[3:])
        rec = {"fast": iter(FAST), "slow": SLOW}
        held = mix.classes(rec)
        self.assertEqual({k: len(v) for k, v in held.items()}, {"fast": 18, "slow": 2})
        self.assertIs(held["slow"], SLOW)
        self.assertIsNot(held, rec)

    def test_a_class_read_to_its_end_is_empty_and_is_said_to_be(self):
        ref = {"fast": range(18), "slow": range(2)}
        with self.assertRaisesRegex(ValueError, "no observations for slow"):     # Version 57: TypeError
            mix.weighted({"fast": iter(FAST), "slow": iter([])}, ref)
        # In the count such a class is one request that may lie on either side.
        self.assertEqual(mix.within({"fast": iter(FAST), "slow": iter([])}, ref, 200), (Fraction(9, 10), Fraction(1)))

    def test_every_class_an_iterator(self):
        rng = random.Random(57)
        for _ in range(50):
            sizes = [rng.randint(1, 6) for _ in range(3)]
            lists = {f"c{i}": [rng.choice((50, 150, 250)) for _ in range(n)] for i, n in enumerate(sizes)}
            flat = [x for xs in lists.values() for x in xs]
            pairs = mix.weighted({name: iter(xs) for name, xs in lists.items()})
            self.assertEqual(sorted(x for x, _ in pairs), sorted(flat))
            low, high = mix.within({name: iter(xs) for name, xs in lists.items()}, lists, 200)
            self.assertEqual(low, high)
            self.assertEqual(low, Fraction(sum(x <= 200 for x in flat), len(flat)))


class MissingIsNotDiscarded(unittest.TestCase):
    def test_two_recorded_requests_without_a_duration(self):
        rec = {"fast": FAST, "slow": [None, None]}
        low, high = mix.within(rec, rec, 200)
        # Eighteen are within 200 ms, and each of the other two may be: 18/20 to 20/20.
        self.assertEqual((low, high), (Fraction(18, 20), Fraction(20, 20)))
        self.assertEqual(mix.verdict(low, high), "unresolved")
        # The same two rows dropped on the way: eighteen of eighteen, and an objective met.
        kept = {"fast": FAST}
        low, high = mix.within(kept, kept, 200)
        self.assertEqual((low, high, mix.verdict(low, high)), (Fraction(1), Fraction(1), "met"))
        for form in (iter, gen):
            low, high = mix.within({"fast": form(FAST), "slow": form([None, NAN])}, rec, 200)
            self.assertEqual((low, high), (Fraction(9, 10), Fraction(1)), form)

    def test_the_three_lines_of_the_transcript(self):
        text = (ROOT / "expected" / "ch23_mix.txt").read_text(encoding="utf-8")
        low, high = 100 * Fraction(18, 20), 100 * Fraction(20, 20)
        for name in ("in a list:", "in an iterator:"):
            self.assertIn(f"  {name:<16}{low} % to {high} %, unresolved\n", text)
        self.assertIn("  dropped:        100 % to 100 %, met\n", text)


class TheStartOfADuration(unittest.TestCase):
    EDGES = {"bounds": [50, 100, 200]}
    WIDE = {"bounds": [50, 100, 200, 400]}

    def test_a_flag_is_not_added_to_a_duration(self):
        # Twenty durations of 100 ms lie in the bucket from 50 to 100: rank 19 of its 20.
        self.assertEqual(50 + 50 * 19 / 20, 97.5)
        for rows in ([(100.0, False)] * 20, [(False, 100.0)] * 20, [(True, 100.0)] * 20, [("0", 100.0)] * 20,
                     [(None, 100.0)] * 20, [(NAN, 100.0)] * 20, [(100.0, math.inf)] * 20):
            with self.assertRaises(ValueError):                                  # Version 57: 97.5 for the first two
                case.reported(rows, dict(self.EDGES, start="accept"))
        with self.assertRaisesRegex(ValueError, "no number: False"):
            case.recorded([(100.0, False)], {"start": "accept"})
        with self.assertRaisesRegex(ValueError, "no number: False"):
            case.reported([(100.0, False)] * 20, dict(self.EDGES, start="handler"))

    def test_from_handler_entry_the_queue_is_not_read(self):
        for queue in (None, NAN, False, "lost"):
            rows = [(queue, 100.0)] * 20
            self.assertEqual(case.recorded(rows, {"start": "handler"}), [100.0] * 20)
            self.assertEqual(case.reported(rows, dict(self.EDGES, start="handler")), 97.5)

    def test_a_start_that_is_not_named(self):
        rows = [(150, 50)] * 20
        # From accept the duration is 200 ms, in the bucket from 100 to 200; from the handler 50 ms, in the first.
        self.assertEqual((100 + 100 * 19 / 20, 0 + 50 * 19 / 20), (195.0, 47.5))
        self.assertEqual(case.reported(rows, dict(self.WIDE, start="accept")), 195.0)
        self.assertEqual(case.reported(rows, dict(self.WIDE, start="handler")), 47.5)
        for start in ("accpet", "Accept", "ACCEPT", "", None, 0, True, "handler "):
            with self.assertRaisesRegex(ValueError, "no start"):                 # Version 57: 47.5 for each
                case.reported(rows, dict(self.WIDE, start=start))
        with self.assertRaises(KeyError):
            case.recorded(rows, {"bounds": [50]})
        self.assertEqual({case.OLD["start"], case.NEW["start"]}, {"handler", "accept"})

    def test_the_periods_of_the_book_are_recorded_as_they_were(self):
        rng = random.Random(23)
        before, after = case.period(rng), case.period(rng)
        for rows in (before, after):
            self.assertEqual(case.recorded(rows, case.NEW), [q + h for q, h in rows])
            self.assertEqual(case.recorded(iter(rows), case.NEW), [q + h for q, h in rows])
            self.assertEqual(case.recorded(rows, case.OLD), [h for _, h in rows])
        text = (ROOT / "expected" / "ch23_case.txt").read_text(encoding="utf-8")
        a, b = case.reported(before, case.OLD), case.reported(after, case.NEW)
        self.assertIn(f"reported p95: {a:.2f} -> {b:.2f} ms", text)


class TypesToTheLastOperation(unittest.TestCase):
    def test_the_sum_of_two_fields(self):
        self.assertEqual(case.recorded([(Decimal("2.5"), 100.0)], {"start": "accept"}), [102.5])   # TypeError
        self.assertEqual(case.recorded([(Decimal("0.1"), Decimal("0.2"))], {"start": "accept"}), [0.3])
        self.assertNotEqual(0.1 + 0.2, 0.3)
        self.assertEqual(case.recorded([(Fraction(1, 3), 1)], {"start": "accept"}), [float(Fraction(4, 3))])
        got = case.recorded([(2, 3)], {"start": "accept"})
        self.assertEqual((got, type(got[0])), ([5.0], float))
        kept = case.recorded([(None, Decimal("7.5"))], {"start": "handler"})
        self.assertEqual((kept, type(kept[0])), ([Decimal("7.5")], Decimal))

    def test_the_change_between_two_periods(self):
        before = [(0, Decimal("100"))] * 20
        for after in ([(0, 105.0)] * 20, [(0, Decimal("105"))] * 20, [(0, Fraction(105))] * 20, [(0, 105)] * 20):
            d = case.handler_change(before, after)                               # Version 57: TypeError for the first
            self.assertEqual((d.value, d.u, type(d.value)), (5.0, 0.0, float), after[0])
            self.assertEqual(case.upper_bound(d), 5.0)                           # Version 57: TypeError for the second
            self.assertFalse(case.within(case.upper_bound(d), 5.0))
            self.assertTrue(case.within(case.upper_bound(d), 5.5))

    def test_readings_of_two_types_in_one_period(self):
        rng = random.Random(57)
        floats = [(0, float(rng.randint(90, 130))) for _ in range(40)]
        mixed = [(q, Decimal(int(h)) if i % 2 else h) for i, (q, h) in enumerate(floats)]
        later = [(q, h + 5) for q, h in floats]
        want = case.handler_change(floats, later)
        got = case.handler_change(mixed, later)
        self.assertEqual((got.value, got.u), (want.value, want.u))
        self.assertGreater(want.u, 0)

    def test_the_change_of_the_book_is_the_number_it_was(self):
        rng = random.Random(23)
        before, after = case.period(rng), case.period(rng)
        hb, ha = [h for _, h in before], [h for _, h in after]
        # The arithmetic of Version 57, in floats.
        ub = st.stdev(stats.bootstrap(hb, case.p95, 300, 23))
        ua = st.stdev(stats.bootstrap(ha, case.p95, 300, 23))
        d = case.handler_change(before, after)
        self.assertEqual((d.value, d.u), (case.p95(ha) - case.p95(hb), math.hypot(ua, ub)))
        text = (ROOT / "expected" / "ch23_case.txt").read_text(encoding="utf-8")
        self.assertIn(f"difference {d}", text)

    def test_the_edges_of_the_histogram(self):
        # Twenty values of 100 lie in the first bucket, from 0 to 100, and twenty of 150 in the second.
        self.assertEqual((0 + 100 * 19 / 20, 100 + 100 * 19 / 20), (95.0, 195.0))
        kinds = (int, float, Decimal, Fraction)
        for kind in kinds:
            edges = [kind(100), kind(200), kind(250)]
            self.assertEqual(h9.bucket_quantile([100] * 20, 0.95, edges), 95.0, kind)   # Version 57: TypeError
            self.assertEqual(h9.bucket_quantile([150] * 20, 0.95, edges), 195.0, kind)  # for Decimal
            self.assertIsInstance(h9.bucket_quantile([150] * 20, 0.95, edges), float)
        mixed = [100, Decimal("200"), Fraction(250)]
        self.assertEqual(h9.bucket_quantile([Decimal("150")] * 20, 0.95, mixed), 195.0)
        # A rank in the open-ended bucket returns the last bound as it was given.
        self.assertEqual(h9.bucket_quantile([300] * 20, 0.95, [Decimal("100"), Decimal("250")]), Decimal("250"))
        self.assertEqual(budget.bracket([150] * 20, [Decimal(100), Decimal(200), Decimal(250)]),
                         (Decimal(100), Decimal(200)))

    def test_the_histogram_in_floats_is_as_it_was(self):
        rng = random.Random(9)
        for _ in range(300):
            xs = [rng.choice((0.004, 0.02, 0.07, 0.3, 0.8, 3.0, 20.0)) * rng.random() for _ in range(rng.randint(1, 40))]
            q = rng.choice((0, 0.5, 0.9, 0.95, 0.99, 1))
            got = h9.bucket_quantile(xs, q)
            # The rule of the docstring, with the fraction of the bucket taken exactly.
            rank, seen, lo = Fraction(str(q)) * len(xs), 0, Fraction(0)
            want = h9.BOUNDS[-1]
            for b in h9.BOUNDS:
                c = sum(lo < x <= b if lo else x <= b for x in xs)
                if c and seen + c >= rank:
                    want = float(lo + (Fraction(b) - lo) * (rank - seen) / c)
                    break
                seen, lo = seen + c, Fraction(b)
            self.assertAlmostEqual(got, want, places=12)


class TheStorageStep(unittest.TestCase):
    def test_the_step_is_an_argument(self):
        self.assertEqual(inspect.signature(budget.from_raw).parameters["step"].default, budget.STEP)
        self.assertEqual(budget.STEP, 1.0)
        xs = [100.0] * 20
        base = budget.from_raw(xs)[1]
        for step in (1, 1.0, Decimal("1"), Fraction(1)):
            r = budget.from_raw(xs, step)[1]
            self.assertEqual((r.value, r.u), (base.value, base.u), step)

    def test_the_two_steps_of_the_review(self):
        # Twenty equal readings have no sampling spread: what is left is 2.5 ms between runs, 0.30 ms for
        # the correction, and the step, rectangular over half a step on either side: step / sqrt(12).
        xs = [Decimal("182.1")] * 20
        for step, storage, combined in ((1, "0.288675", "2.534430"), (Decimal("0.1"), "0.028868", "2.518101")):
            want = math.sqrt(2.5 ** 2 + 0.30 ** 2 + float(step) ** 2 / 12)
            bud, r = budget.from_raw(xs, step)
            self.assertEqual(f"{float(step) / math.sqrt(12):.6f}", storage)
            self.assertEqual(f"{want:.6f}", combined)
            self.assertAlmostEqual(r.u, want, places=12)
            self.assertEqual(r.value, 181.2)
            name, kind, u, _ = bud.rows[-1]
            self.assertEqual((name, kind, f"{u:.6f}"), ("stored step", "B", storage))
        for step in (0.1, Fraction(1, 10)):
            self.assertAlmostEqual(budget.from_raw(xs, step)[1].u, math.sqrt(2.5 ** 2 + 0.09 + 0.01 / 12), places=12)
        self.assertAlmostEqual(budget.from_raw(xs, 0)[1].u, math.sqrt(2.5 ** 2 + 0.09), places=12)

    def test_two_readings_that_no_grid_of_one_millisecond_holds(self):
        a, b = Fraction("182.1"), Fraction("182.2")
        self.assertNotEqual((b - a) % 1, 0)
        self.assertEqual((b - a) % Fraction(1, 10), 0)

    def test_what_is_no_step(self):
        xs = [100.0] * 20
        for bad, why in ((True, "no number"), ("1", "no number"), (None, "no number"),
                         (NAN, "a step is missing or too large"), (math.inf, "a step is missing or too large"),
                         (-1, "need a width >= 0")):
            with self.assertRaisesRegex(ValueError, why):
                budget.from_raw(xs, bad)

    def test_the_record_of_the_book(self):
        ms = budget.record(random.Random(22))
        self.assertEqual(budget.from_raw(ms)[1].report(), "(182.1 ± 6.5) ms, k = 2.12, p = 0.95")
        self.assertTrue(all(float(x).is_integer() for x in ms))                 # the probe stores whole ms


class WhatATimeoutEstablishes(unittest.TestCase):
    REC = {"all": [100] * 18 + [1200, (">=", 1000)]}

    def by_hand(self, limit):
        """The share within the limit for every duration that the bound allows."""
        shares = {Fraction(18 + (1200 <= limit) + (t <= limit), 20)
                  for t in (1000, 1000.5, 1099, 1100, 1101, 1199, 1200, 1201, 5000)}
        return min(shares), max(shares)

    def test_the_three_objectives_of_the_review(self):
        for limit, low, high, verdict in ((999, "9/10", "9/10", "missed"), (1100, "9/10", "19/20", "unresolved"),
                                          (1200, "19/20", "1", "met")):
            self.assertEqual(self.by_hand(limit), (Fraction(low), Fraction(high)), limit)
            got = mix.within(self.REC, self.REC, limit)
            self.assertEqual(got, (Fraction(low), Fraction(high)), limit)
            self.assertEqual(mix.verdict(*got), verdict, limit)

    def test_the_client_with_one_deadline_has_no_such_record(self):
        with self.assertRaisesRegex(ValueError, "need 0 <= x < TIMEOUT"):
            censoring.p95_all([100] * 18 + [1200, None])
        # Where every completion came before the deadline, the timeout does rank after them all.
        done = [100.0] * 18 + [999.0]
        for t in (1000.0, 1001.0, 5000.0):
            self.assertEqual(sorted(done + [t])[-1], t)
        self.assertEqual(censoring.p95_all(done + [None]), (999.0, False))


class TheLimitAndTheBound(unittest.TestCase):
    REC = {"all": [1] * 20}

    def test_a_limit_is_a_duration_or_an_infinity(self):
        for flag in (True, False):
            with self.assertRaisesRegex(ValueError, "no number"):                # Version 57: [1, 1] met, [0, 0] missed
                mix.within(self.REC, self.REC, flag)
        with self.assertRaisesRegex(ValueError, "no number: '200'"):             # Version 57: TypeError
            mix.within(self.REC, self.REC, "200")
        for none in (None, NAN, Decimal("NaN")):
            with self.assertRaisesRegex(ValueError, "no limit"):
                mix.within(self.REC, self.REC, none)
        with self.assertRaisesRegex(ValueError, "a limit is missing or too large"):
            mix.within(self.REC, self.REC, 10 ** 400)
        for limit in (1, 1.0, Decimal("1"), Fraction(1), 200, math.inf):
            self.assertEqual(mix.within(self.REC, self.REC, limit), (1, 1), limit)
        for limit in (0, 0.5, Decimal("0.999"), -math.inf):
            self.assertEqual(mix.within(self.REC, self.REC, limit), (0, 0), limit)

    def test_a_bound_is_a_sign_and_a_duration(self):
        cases = (((">=", None), "no number: None"), ((">=", True), "no number: True"), ((">", "300"), "no number"),
                 ((">=", Decimal("sNaN")), "a bound is missing or too large"),     # Version 57: InvalidOperation
                 ((">=", NAN), "a bound is missing or too large"), ((">", math.inf), "a bound is missing or too large"),
                 ((">=",), "a bound is > or >="), ((">=", 1, 2), "a bound is > or >="), ((), "a bound is > or >="),
                 (("<", 100), "a bound is > or >="), (("=>", 100), "a bound is > or >="), ((100, ">"), "a bound is > or >="))
        for bad, why in cases:
            rec = {"all": [100] * 19 + [bad]}
            for call in (lambda: mix.within(rec, rec, 200), lambda: mix.weighted(rec), lambda: mix.observed(rec["all"])):
                with self.assertRaisesRegex(ValueError, why, msg=repr(bad)):     # Version 57: TypeError for the first
                    call()

    def test_what_a_good_bound_and_a_missing_duration_still_mean(self):
        for bound, low, high in (((">", 300), "19/20", "19/20"), ((">=", 200.5), "19/20", "19/20"),
                                 ((">", 200), "19/20", "19/20"), ((">=", 200), "19/20", "1"),
                                 ((">", 100), "19/20", "1"), ((">=", Decimal("150")), "19/20", "1"),
                                 (None, "19/20", "1"), (NAN, "19/20", "1"), (math.inf, "19/20", "19/20")):
            rec = {"all": [100] * 19 + [bound]}
            self.assertEqual(mix.within(rec, rec, 200), (Fraction(low), Fraction(high)), bound)


class TheBoundThatIsReported(unittest.TestCase):
    def test_at_two_places(self):
        bound = -math.log(0.05) / 0.75
        self.assertEqual(f"{bound:.9f}", "3.994309698")
        self.assertEqual(f"{bound:.2f}", "3.99")
        self.assertLess(3.99, bound)                                             # as it rounds, it is no bound
        self.assertEqual(math.ceil(bound * 100) / 100, 4.0)
        # The other two of the three cases round upward as they stand.
        whole, lost = -math.log(0.05) / 1.0, stats.rate_upper(2, 1.0)
        self.assertEqual((f"{whole:.9f}", f"{whole:.2f}"), ("2.995732274", "3.00"))
        self.assertEqual((f"{lost:.9f}", f"{lost:.2f}"), ("6.295793622", "6.30"))
        self.assertTrue(whole <= 3.00 and lost <= 6.30)

    def test_the_sixth_rate_is_no_logarithm(self):
        m = stats.rate_upper(2, 1.0)
        # The mean at which at most two events have a probability of 5 %.
        self.assertAlmostEqual(math.exp(-m) * (1 + m + m * m / 2), 0.05, places=12)
        self.assertGreater(m, 2 * -math.log(0.05))
        for seconds in (1.0, 0.65, 0.75, 0.648, 0.749):
            self.assertAlmostEqual(stats.quiet_bound(seconds), -math.log(0.05) / seconds, places=12)


class RankZero(unittest.TestCase):
    @staticmethod
    def classic(counts, uppers, q):
        """Cumulative bucket counts through the steps of BucketQuantile for classic histograms, as read in
        promql/quantile.go of Prometheus 3.16.0 on 24 November 2026: the rank, the first bucket whose count
        reaches it, and the interpolation, with no case of its own for a rank of zero."""
        observations = counts[-1]
        if observations == 0:
            return NAN
        rank = q * observations
        b = next((i for i in range(len(counts) - 1) if counts[i] >= rank), len(counts) - 1)
        if b == len(counts) - 1:
            return uppers[-2]
        if b == 0 and uppers[0] <= 0:
            return uppers[0]
        start, end, count = 0.0, uppers[b], counts[b]
        if b > 0:
            start = uppers[b - 1]
            count -= counts[b - 1]
            rank -= counts[b - 1]
        if count == 0:
            return NAN if rank == 0 else math.copysign(math.inf, rank)           # what 0/0 and x/0 give in floats
        return start + (end - start) * (rank / count)

    def test_one_observation_between_two_bounds(self):
        self.assertEqual(h9.bucket_quantile([150], 0, [100, 200]), 100)
        self.assertEqual(sum(x <= 100 for x in [150]), 0)                        # the first bucket is empty
        self.assertTrue(math.isnan(self.classic([0, 1, 1], [100, 200, math.inf], 0)))
        for q in (0.25, 0.5, 0.95, 1):
            self.assertEqual(h9.bucket_quantile([150], q, [100, 200]), 100 + 100 * q)
            self.assertEqual(self.classic([0, 1, 1], [100, 200, math.inf], q), 100 + 100 * q)

    def test_the_two_agree_above_rank_zero(self):
        rng = random.Random(916)
        for _ in range(300):
            xs = [rng.uniform(0, 12) for _ in range(rng.randint(1, 30))]
            q = rng.choice((0.1, 0.5, 0.9, 0.95, 0.99))
            counts = [sum(x <= b for x in xs) for b in h9.BOUNDS] + [len(xs)]
            want = self.classic(counts, h9.BOUNDS + [math.inf], q)
            self.assertAlmostEqual(h9.bucket_quantile(xs, q), want, places=9)


class TheArchiveThatTheChecksumNames(unittest.TestCase):
    def test_the_readme_names_the_file(self):
        import re
        from data import make_record
        number = re.fullmatch(r"book Version (\d+), \w+ \d{4}", make_record.RELEASE).group(1)
        readme = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())
        self.assertIn(f"MetrologyForSoftwareDiagnostics-Version{number}.zip", readme)
        self.assertIn("A different checksum identifies a different archive file", readme)


if __name__ == "__main__":
    unittest.main()
