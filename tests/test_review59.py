"""Regression tests added after the review of Version 59.

The review found that a reference mix which shares an iterator with the record, as a shallow copy of the record
does, lost the requests of that class, so that two slow requests in twenty went uncounted and the objective was
reported as met; that a duration without a value was left open against a limit below zero, which no duration can
meet; that upward() and downward() of the package placed the decimal point by an operation of the decimal module,
which rounds once more in the context of the caller and can turn a bound to its wrong side; that Transcript 16.4
gave a chance at a rate as printed which is the chance at the bound as computed; that Transcript 14.1 printed the
acceptance limit rounded to the nearest, above the limit that the listing applies; that the text gave 4.71 ms as
the computed bound; that the conclusion at a margin of 5 ms rests on a normal approximation, which a bound from
order statistics does without; that a histogram bound given as a fraction could not be shown under Python 3.11;
and that a decimal limit could not be put to a result with an exact value and a positive uncertainty.

Each test computes what it expects independently of the code under test: weights and shares by counting, a
printed bound by the decimal module in a context wide enough to hold it, chances from powers and binomial sums
in whole numbers, and the cutoffs of a decision from the normal distribution of the statistics module.
"""
import contextlib
import io
import math
import random
import tracemalloc
import unittest
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal, Inexact, Rounded, localcontext
from fractions import Fraction
from pathlib import Path
from statistics import NormalDist

from examples import ch09_dependent_tail as tail
from examples import ch11_leak_rate as leak
from examples import ch14_decisions as decisions
from examples import ch16_silence as silence
from examples import ch16_trace_measurements as trace
from examples import ch18_calibration as cal
from examples import ch23_case as case
from examples import ch23_mix as mix
from swmetro import Result, stats
from swmetro.core import fixed
from swmetro.dist import t_cdf

ROOT = Path(__file__).resolve().parent.parent
D = Decimal


def expected(name):
    return (ROOT / "expected" / f"{name}.txt").read_text(encoding="utf-8")


def printed(function, *args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        function(*args)
    return out.getvalue()


def wide(x, places, rounding):
    """x rounded toward one side at so many places, by the decimal module in a context of 400 digits. A fraction
    is divided there first; one of these tests that does not end lies far more than 400 digits from a place."""
    with localcontext() as ctx:
        ctx.prec = 400
        ctx.traps[Inexact] = ctx.traps[Rounded] = False
        if isinstance(x, Fraction):
            value = Decimal(x.numerator) / Decimal(x.denominator)
        else:
            value = Decimal(x)
        return format(value.quantize(Decimal(1).scaleb(-places), rounding=rounding), "f")


def workload():
    """Eighteen requests of 100 ms, and two of 500 ms that can be read once."""
    return {"fast": [100] * 18, "slow": iter([500, 500])}


class ACopiedWorkload(unittest.TestCase):
    def facts(self, reference):
        rec = workload()
        pairs = mix.weighted(rec, reference(rec))
        rec = workload()
        share = mix.within(rec, reference(rec), 200)
        return sum(w for _, w in pairs), mix.p95(pairs), share, mix.verdict(*share)

    def test_the_four_references_of_the_review(self):
        # By counting: the nineteenth of the twenty sorted durations is 500 ms, and 18 of 20 are within 200 ms.
        durations = sorted([100] * 18 + [500, 500])
        want = (20, durations[math.ceil(Fraction(95, 100) * 20) - 1], (Fraction(18, 20),) * 2, "missed")
        self.assertEqual(want[1], 500)

        def held(rec):
            rec.update(mix.classes(rec))
            return rec.copy()
        references = {"the record itself": lambda rec: rec,
                      "a shallow copy": lambda rec: rec.copy(),                # Version 59: 18, 100 ms, (1, 1), met
                      "independent counts": lambda rec: {"fast": [0] * 18, "slow": [0, 0]},
                      "materialized, then copied": held}
        for name, reference in references.items():
            self.assertEqual(self.facts(reference), want, name)

    def test_every_class_shared(self):
        rec = {"fast": iter([100] * 18), "slow": iter([500, 500])}
        self.assertEqual(mix.within(rec, rec.copy(), 200), (Fraction(9, 10),) * 2)   # Version 59: empty reference mix
        rec = {"fast": iter([100] * 18), "slow": iter([500, 500])}
        pairs = mix.weighted(rec, dict(rec))
        self.assertEqual((len(pairs), sum(w for _, w in pairs), mix.p95(pairs)), (20, 20, 500))

    def test_a_shared_class_under_another_name_and_another_weight(self):
        slow = iter([500, 500])
        rec = {"fast": [100] * 18, "slow": slow}
        ref = {"fast": [0] * 2, "slow": slow}                                  # one tenth fast at the reference
        pairs = mix.weighted(rec, ref)
        self.assertEqual(sum(w for _, w in pairs), 4)
        self.assertEqual(sum(w for x, w in pairs if x == 500), 2)
        self.assertEqual(sum(w for x, w in pairs if x == 100), 2)
        stream = iter([100] * 18)
        held, sizes = mix.both({"fast": stream}, {"quick": stream})
        self.assertEqual((len(held["fast"]), len(sizes["quick"])), (18, 18))

    def test_a_reference_is_counted_in_no_list(self):
        rec = {"fast": [100] * 18, "slow": [500, 500]}
        for form in (list, tuple, iter, lambda xs: (x for x in xs), lambda xs: range(len(xs))):
            ref = {"fast": form([0] * 18), "slow": form([0] * 2)}
            self.assertEqual(mix.within(rec, ref, 200), (Fraction(9, 10),) * 2)
        self.assertEqual(len(mix.both(rec, {"fast": range(10 ** 12)})[1]["fast"]), 10 ** 12)
        # A list of 300 000 entries would hold more than two megabytes of references.
        tracemalloc.start()
        try:
            sizes = mix.both(rec, {"fast": (0 for _ in range(300_000)), "slow": iter(range(7))})[1]
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        self.assertEqual((len(sizes["fast"]), len(sizes["slow"])), (300_000, 7))
        self.assertLess(peak, 200_000)

    def test_every_observed_class_is_still_held_to_the_check(self):
        for bad in (False, "100", -5):
            rec = {"fast": [100] * 20, "odd": iter([bad])}
            with self.assertRaises(ValueError):
                mix.within(rec, {"fast": [0] * 20}, 200)
            rec = {"fast": [100] * 20, "odd": iter([bad])}
            with self.assertRaises(ValueError):
                mix.weighted(rec, rec.copy())
        with self.assertRaises(ValueError):
            mix.within({"fast": iter([])}, {"fast": iter([])}, 200)            # an empty reference mix is one still
        # A class of the reference without observations is still reported, with its share.
        rec, ref = {"fast": [100] * 18}, {"fast": iter(range(18)), "slow": iter(range(2))}
        held, sizes = mix.both(rec, ref)
        self.assertEqual(mix.absent(held, sizes), {"slow": Fraction(1, 10)})


class ALimitBelowZero(unittest.TestCase):
    BELOW = (-1, -0.001, D("-0.001"), Fraction(-1, 10 ** 9), -math.inf, D("-Infinity"))

    def test_no_duration_meets_it(self):
        gone = {"all": [None] * 20}
        some = {"all": [100] * 18 + [None, math.nan]}
        for limit in self.BELOW:
            for rec in (gone, some):                                           # Version 59: (0, 1) and (0, 1/10)
                share = mix.within(rec, rec, limit)
                self.assertEqual((share, mix.verdict(*share)), ((0, 0), "missed"), limit)
            for entry in (None, math.nan, D("NaN"), 0, 100, (">", 0), (">=", 250), math.inf):
                self.assertIs(mix.side(entry, limit), False, (entry, limit))
        # A class of the reference that was not observed stands for one request of unknown duration.
        self.assertEqual(mix.within({"fast": [100]}, {"fast": [0], "slow": [0]}, -1), (0, 0))

    def test_zero_is_a_limit_that_a_duration_can_meet(self):
        gone = {"all": [None] * 20}
        some = {"all": [100] * 18 + [None, None]}
        self.assertEqual(mix.within(gone, gone, 0), (0, 1))
        self.assertEqual(mix.verdict(*mix.within(gone, gone, 0)), "unresolved")
        self.assertEqual(mix.within(some, some, 0), (0, Fraction(1, 10)))
        self.assertIs(mix.side(0, 0), True)
        self.assertIs(mix.side(None, 0), None)
        self.assertIs(mix.side((">=", 0), 0), None)
        self.assertIs(mix.side((">", 0), 0), False)
        # Above zero and at infinity nothing changed.
        self.assertIs(mix.side(None, 200), None)
        self.assertIs(mix.side(None, math.inf), True)
        self.assertEqual(mix.within(some, some, 200), (Fraction(9, 10), 1))


class ABoundKeepsItsSide(unittest.TestCase):
    def test_the_two_calls_of_the_review(self):
        with localcontext() as ctx:
            ctx.prec = 3
            self.assertEqual(stats.upward(D("0.23719322592"), 5), "0.23720")   # Version 59: 0.237
            self.assertEqual(stats.downward(D("9.8765"), 3), "9.876")          # Version 59: 9.88
        # In the default context of 28 digits, a whole number of 29.
        n = 10 ** 28 + 1
        self.assertEqual((stats.upward(n, 0), stats.downward(n, 0)), (str(n), str(n)))   # Version 59: one too low
        self.assertEqual(stats.upward(Fraction(n, 7), 2), wide(Fraction(n, 7), 2, ROUND_CEILING))
        with localcontext() as ctx:
            ctx.prec = 3
            ctx.traps[Inexact] = ctx.traps[Rounded] = True
            self.assertEqual(stats.upward(D("0.23719322592"), 5), "0.23720")   # Version 59: raised Inexact

    def test_both_sides_in_narrow_contexts(self):
        rng = random.Random(59)
        for _ in range(1500):
            places = rng.randint(0, 9)
            kind = rng.randrange(4)
            if kind == 0:
                x = rng.uniform(-50, 50) * 10 ** rng.randint(-7, 4)
            elif kind == 1:
                x = D(rng.randint(-10 ** 12, 10 ** 12)).scaleb(-rng.randint(0, 14))
            elif kind == 2:
                x = Fraction(rng.randint(-10 ** 9, 10 ** 9), rng.randint(1, 10 ** 6))
            else:
                x = rng.randint(-10 ** 35, 10 ** 35)
            most, least = wide(x, places, ROUND_CEILING), wide(x, places, ROUND_FLOOR)
            with localcontext() as ctx:
                ctx.prec = rng.randint(1, 6)
                ctx.traps[Inexact] = ctx.traps[Rounded] = bool(rng.randrange(2))
                got = stats.upward(x, places), stats.downward(x, places)
            self.assertEqual((Fraction(got[0]), Fraction(got[1])), (Fraction(most), Fraction(least)), (x, places))
            exact, unit = Fraction(x), Fraction(1, 10 ** places)
            self.assertTrue(Fraction(got[1]) <= exact <= Fraction(got[0]), (x, places))
            self.assertTrue(Fraction(got[0]) - exact < unit and exact - Fraction(got[1]) < unit, (x, places))
            for text in got:
                digits = text.partition(".")[2]
                self.assertEqual(len(digits), places, (x, places, text))
                self.assertNotIn("E", text.upper())

    def test_the_sign_and_the_places(self):
        self.assertEqual((stats.upward(-0.0001, 2), stats.downward(-0.0001, 2)), ("0.00", "-0.01"))
        self.assertEqual((stats.upward(-5.93, 1), stats.downward(-5.93, 1)), ("-5.9", "-6.0"))
        self.assertEqual((stats.upward(0, 3), stats.downward(7, 0)), ("0.000", "7"))
        for places in (-1, -2, 2.5, "2", None):                                # Version 59 took -2 through a float
            with self.assertRaises(ValueError):
                stats.upward(1234.5, places)
            with self.assertRaises(ValueError):
                stats.downward(1234.5, places)

    def test_the_transcripts_in_a_context_of_three_digits(self):
        # Every example that prints a bound to its safe side prints the same lines whatever the context.
        for module, name in ((silence, "ch16_silence"), (trace, "ch16_trace_measurements"), (case, "ch23_case"),
                             (decisions, "ch14_decisions"), (tail, "ch09_dependent_tail"),
                             (leak, "ch11_leak_rate"), (cal, "ch18_calibration")):
            with localcontext() as ctx:
                ctx.prec = 3
                self.assertEqual(printed(module.main), expected(name), name)


class TheRateOfAChance(unittest.TestCase):
    def test_which_rate_gives_40_13(self):
        def found(rate):                                                       # one of ten windows of 30 s is empty
            return 1 - (1 - math.exp(-30 * rate)) ** 10
        bound = -math.log(0.05) / 30
        self.assertAlmostEqual(stats.quiet_bound(30), bound, places=15)
        self.assertEqual(stats.upward(bound, 5), "0.09986")
        self.assertEqual((f"{100 * found(0.09986):.2f}", f"{100 * found(bound):.2f}"), ("40.12", "40.13"))
        exact = 1 - Fraction(95, 100) ** 10                                    # at the bound a window is empty 5 %
        self.assertEqual(f"{float(100 * exact):.5f}", "40.12631")
        text = expected("ch16_silence")
        self.assertIn("  at the first bound a search of ten\n  finds an empty window: 40.13 %\n", text)
        self.assertNotIn("at 0.09986 msg/s a search", text)                    # Version 59

    def test_no_chance_is_stated_at_printed_digits(self):
        lines = expected("ch16_silence").splitlines()
        for line in lines:
            if "msg/s" in line and not line.strip().startswith("at most"):
                # A rate that is no bound in name: the nominal ones, the estimate and its two intervals.
                self.assertTrue("nominal 95 %" in line or line.startswith("one message in 30 s:")
                                or line.startswith("  a rate of 0.10 msg/s") or line.startswith("  0.30 msg/s"), line)
        named = [x.strip() for x in lines if "bound" in x and ("silent" in x or x.strip().startswith("at the"))]
        self.assertEqual(named, ["silent at the first bound: 19.9 %", "at the first bound the second",
                                 "silent at the first bound: 6.42 %", "silent at the first bound: 13.57 %",
                                 "at the first bound a search of ten", "at the second bound such a gap",
                                 "at the single-message search bound"])
        self.assertIn("    silent at the nominal rate: 6.26 %\n", expected("ch16_silence"))
        self.assertEqual(len(lines), 193)                                      # as many lines as in Version 59

    def test_only_that_chance_turns_on_the_digits(self):
        # The other chances of the transcript, each at its bound as computed and at the five digits printed for it.
        whole = [(0, 30)]
        ten = [(30 * i, 30 * i + 30) for i in range(10)]
        one, low = stats.quiet_bound(30), silence.paired_bound(whole, 30)
        kept, searched = stats.quiet_bound(7.5), silence.searched_bound(30, 10)
        mixed = silence.mean_reach(whole, [(0.5, 10), (0.5, 50)])
        cases = ((one, lambda r: math.exp(-r / 2 * 32.346), 1, "19.9"),       # starts covered, pairs 2.346 s apart
                 (low, lambda r: math.exp(-r / 2 * mixed), 2, "8.24"),
                 (kept, lambda r: math.exp(-r * 6.875), 2, "6.42"), (kept, lambda r: math.exp(-r * 5), 2, "13.57"),
                 (searched, lambda r: silence.gap_somewhere(r, 300, 30), 2, "22.48"),
                 (searched, lambda r: silence.some_empty(r, silence.selections(ten, 10)), 2, "25.17"))
        for bound, chance, places, shown in cases:
            digits = float(stats.upward(bound, 5))
            self.assertGreater(digits, bound)
            self.assertEqual([f"{100 * chance(r):.{places}f}" for r in (bound, digits)], [shown, shown])
            self.assertIn(f": {shown} %\n", expected("ch16_silence"))

    def test_at_that_is_the_same_at_either_rate(self):
        # "the source is silent at that": the four chances print alike at the rate as computed and as printed.
        whole, source = [(0, 30)], [(0.5, 10), (0.5, 50)]
        true = silence.mean_reach(whole, source)
        shown = []
        for rule in ("start", "end", "both", "either"):
            seen = silence.selected(source, 30, rule)
            top = 2 * stats.quiet_bound(silence.mean_reach(whole, seen))
            as_printed = float(stats.upward(top, 5))
            pair = [f"{100 * math.exp(-rate / 2 * true):.2f}" for rate in (top, as_printed)]
            self.assertEqual(pair[0], pair[1], rule)
            shown.append(pair[0])
        self.assertEqual(shown, ["5.00", "5.00", "2.36", "5.61"])
        for value in shown:
            self.assertIn(f"    the source is silent at that: {value} %\n", expected("ch16_silence"))

    def test_the_two_bounds_of_figure_16_3(self):
        none, one = stats.quiet_bound(20), stats.rate_upper(1, 20)
        self.assertEqual((f"{none:.6f}", f"{one:.6f}"), ("0.149787", "0.237193"))
        self.assertEqual((stats.upward(none, 3), stats.upward(one, 3)), ("0.150", "0.238"))
        self.assertEqual((stats.upward(none, 5), stats.upward(one, 5)), ("0.14979", "0.23720"))
        self.assertIn("    at most 0.23720 msg/s at 95 %\n", expected("ch16_silence"))


class TheCutoffsOfChapterFourteen(unittest.TestCase):
    Z = NormalDist().inv_cdf(0.95)

    @staticmethod
    def decision(y):
        """The rule of Listing 14.1, with the normal distribution of the statistics module."""
        p = NormalDist().cdf((200 - y) / 3)
        return "accept" if p >= 0.95 else "reject" if p < 0.05 else "undecided"

    def test_the_working_cutoffs(self):
        accept, reject = 200 - 3 * self.Z, 200 + 3 * self.Z
        self.assertEqual((f"{accept:.6f}", f"{reject:.6f}"), ("195.065439", "204.934561"))
        self.assertEqual((f"{accept:.4f}", f"{reject:.4f}"), ("195.0654", "204.9346"))   # as the text gives them
        self.assertEqual((f"{accept:.1f}", f"{reject:.1f}"), ("195.1", "204.9"))         # the limits of Version 59
        # The two readings of the review, and the printed limits taken literally.
        rows = {195.08: ("undecided", "0.949497"), 204.92: ("undecided", "0.050503"),
                195.1: ("undecided", "0.948801"), 204.9: ("undecided", "0.051199")}
        for y, (verdict, p) in rows.items():
            self.assertEqual((self.decision(y), f"{NormalDist().cdf((200 - y) / 3):.6f}"), (verdict, p), y)

    def test_the_two_lines_hold_as_printed(self):
        text = expected("ch14_decisions")
        self.assertTrue(text.startswith("accept <= 195.06, reject >= 204.94 ms\n measured"))  # reworded in Version 61
        self.assertEqual(printed(decisions.conformity), "".join(text.splitlines(True)[:9]))
        self.assertEqual(len(text.splitlines()), 14)                           # as many lines as it had
        accept, reject = 200 - 3 * self.Z, 200 + 3 * self.Z
        low, high = Fraction("195.06"), Fraction("204.94")
        self.assertTrue(low <= accept < low + Fraction(1, 100))                # downward, and by less than a unit
        self.assertTrue(high - Fraction(1, 100) < reject <= high)              # upward
        # Every reading on a grid of 0.01 ms: at or below the first number accepted, at or above the second
        # rejected, and undecided between. The line of Version 60, "reject above 204.94 ms", left 204.94 ms
        # itself unnamed; the rule rejects it.
        for hundredths in range(18500, 21501):
            y = hundredths / 100
            verdict = self.decision(y)
            self.assertEqual(verdict, "accept" if y <= accept else "reject" if y > reject else "undecided", y)
            if Fraction(hundredths, 100) <= low:
                self.assertEqual(verdict, "accept", y)
            elif Fraction(hundredths, 100) >= high:
                self.assertEqual(verdict, "reject", y)
            else:
                self.assertEqual(verdict, "undecided", y)
        # On a grid of 0.1 ms the three ranges are those of the review.
        tenths = {t / 10: self.decision(t / 10) for t in range(1850, 2151)}
        self.assertEqual(max(y for y, v in tenths.items() if v == "accept"), 195.0)
        self.assertEqual(min(y for y, v in tenths.items() if v == "reject"), 205.0)
        self.assertEqual([y for y, v in tenths.items() if v == "undecided"][::98], [195.1, 204.9])

    def test_the_rows_are_what_they_were(self):
        rows = expected("ch14_decisions").splitlines()[1:9]
        self.assertEqual(rows, [" measured  P(conform)  decision", "   188 ms       1.000  accept",
                                "   192 ms       0.996  accept", "   195 ms       0.952  accept",
                                "   197 ms       0.841  undecided", "   200 ms       0.500  undecided",
                                "   203 ms       0.159  undecided", "   206 ms       0.023  reject"])
        for row in rows[1:]:
            y, _, p, verdict = row.split()
            self.assertEqual((self.decision(int(y)), f"{NormalDist().cdf((200 - int(y)) / 3):.3f}"), (verdict, p))
        # With the earlier result of the chapter, 205 ms with the same uncertainty: the limit of the text.
        prior = 2 * (200 - 3 / math.sqrt(2) * self.Z) - 205
        self.assertEqual((f"{prior:.3f}", stats.downward(prior, 1)), ("188.021", "188.0"))


def periods():
    rng = random.Random(23)
    return case.period(rng), case.period(rng)


def tail_at_most(n, m):
    """P(Y <= m) for Y binomial with n and 1/20, the number of n draws above a 95th percentile: exact."""
    return Fraction(sum(math.comb(n, j) * 19 ** (n - j) for j in range(m + 1)), 20 ** n)


class TheMarginAndTheBound(unittest.TestCase):
    def test_4_71_is_a_rounded_number(self):
        before, after = periods()
        upper = case.upper_bound(case.handler_change(before, after))
        # Since Version 62 the bound is a fraction, which Python 3.11 formats only through a float, and the
        # least margin above it is no neighboring float: any number above it will do.
        shown = float(upper)
        self.assertEqual((f"{shown:.5f}", f"{shown:.2f}", stats.upward(upper, 1)), ("4.70787", "4.71", "4.8"))
        self.assertLess(upper, 4.71)
        for margin, below in ((4.71, True), (D("4.71"), True), (5.0, True), (2.0, False), (upper, False),
                              (upper + Fraction(1, 10 ** 40), True)):
            self.assertIs(case.within(upper, margin), below, margin)


class ASecondBoundByRanks(unittest.TestCase):
    def test_the_ranks_and_their_chances(self):
        before, after = periods()
        n_old, n_new = len(before), len(after)
        self.assertEqual((n_old, n_new), (1905, 1955))
        # The lower end is the r-th of the sorted draws. It lies above the percentile when at most r - 1 draws lie
        # below the percentile, that is when at least n - r + 1 lie above it.
        def low_fails(n, r):
            return 1 - tail_at_most(n, n - r)
        # The upper end is the s-th. It lies below the percentile when at most n - s draws lie above it.
        def high_fails(n, s):
            return tail_at_most(n, n - s)
        limit = Fraction(25, 1000)
        r = max(r for r in range(1750, 1830) if low_fails(n_old, r) <= limit)
        s = min(s for s in range(1840, 1900) if high_fails(n_new, s) <= limit)
        self.assertEqual((r, s), (1791, 1877))
        a, b = low_fails(n_old, r), high_fails(n_new, s)
        self.assertEqual((f"{float(100 * a):.4f}", f"{float(100 * b):.4f}"), ("2.3892", "2.0215"))
        self.assertEqual(f"{float(100 * (1 - a - b)):.4f}", "95.5893")         # at least, by the union of the two
        self.assertEqual((stats.quantile_ranks(n_old, 0.95)[0], stats.quantile_ranks(n_new, 0.95)[1]), (r, s))

    def test_the_bound_of_the_two_periods(self):
        before, after = periods()
        old = sorted(h for _, h in before)
        new = sorted(h for _, h in after)
        low, high = old[1791 - 1], new[1877 - 1]
        self.assertEqual((f"{low:.6f}", f"{high:.6f}", f"{high - low:.6f}"), ("177.045659", "185.697641", "8.651982"))
        self.assertEqual(case.rank_bound(before, after), (low, high, high - low))
        shown = stats.downward(low, 2), stats.upward(high, 2), stats.upward(high - low, 2)
        self.assertEqual(shown, ("177.04", "185.70", "8.66"))
        self.assertEqual(Fraction(shown[1]) - Fraction(shown[0]), Fraction(shown[2]))
        ranks = case.rank_bound(before, after)[2]
        normal = case.upper_bound(case.handler_change(before, after))
        self.assertEqual([case.within(ranks, m) for m in (5.0, 2.0)], [False, False])
        self.assertEqual([case.within(normal, m) for m in (5.0, 2.0)], [True, False])
        text = expected("ch23_case")
        self.assertIn("margin 5 ms: slowdown below the margin\nmargin 2 ms: not established\n"
                      "the p95 by ranks:\n  before, at least 177.04 ms at 97.5 %\n"
                      "  after, at most 185.70 ms at 97.5 %\n  change at most 8.66 ms at 95 %\n"
                      "margin 5 ms: not established\nmargin 2 ms: not established\n"
                      "20 requests of 100 ms, twice:\n", text)
        self.assertEqual(printed(case.main), text)
        self.assertEqual(len(text.splitlines()), 41)                            # 35 in Version 60

    def test_too_few_for_an_upper_end(self):
        rows = [(0.0, 100.0)] * 20
        # Version 60 returned None for all three. The lower end of twenty requests exists, and is returned.
        self.assertEqual(case.rank_bound(rows, rows), (100.0, None, None))
        self.assertTrue(Fraction(19, 20) ** 72 <= Fraction(25, 1000) < Fraction(19, 20) ** 71)
        self.assertTrue(Fraction(19, 20) ** 59 <= Fraction(5, 100) < Fraction(19, 20) ** 58)
        self.assertEqual((stats.run_length(0.95, 0.025), stats.run_length(0.95, 0.05)), (72, 59))
        self.assertEqual((stats.quantile_ranks(71, 0.95)[1], stats.quantile_ranks(72, 0.95)[1]), (None, 72))
        rng = random.Random(72)
        short = [(0.0, rng.uniform(90, 110)) for _ in range(71)]
        long = [(0.0, rng.uniform(90, 110)) for _ in range(72)]
        self.assertEqual(case.rank_bound(long, short)[1:], (None, None))
        low, high, top = case.rank_bound(short, long)
        self.assertEqual(high, max(h for _, h in long))
        self.assertEqual(top, high - low)
        self.assertIn("  no upper end with fewer than 72 after\n", expected("ch23_case"))
        # Exact readings keep an exact bound, and a time below zero is refused here as elsewhere.
        exact = [(0, D(100 + i)) for i in range(80)]
        self.assertEqual(case.rank_bound(exact, exact)[2], Fraction(179) - Fraction(case.rank_bound(exact, exact)[0]))
        with self.assertRaises(ValueError):
            case.rank_bound([(0, -1.0)] * 80, exact)

    def test_the_bound_covers_in_simulation(self):
        # Two periods of 150 draws each from one population: the change of its p95 is 0, and the bound has to
        # lie at or above 0 in at least 95 % of pairs. The ends are independent here, and it does so more often.
        rng = random.Random(959)
        trials, covered = 3000, 0
        for _ in range(trials):
            one = [(0.0, rng.lognormvariate(0, 0.25)) for _ in range(150)]
            two = [(0.0, rng.lognormvariate(0, 0.25)) for _ in range(150)]
            covered += case.rank_bound(one, two)[2] >= 0
        sd = math.sqrt(0.05 * 0.95 / trials)
        self.assertGreater(covered / trials, 0.95 - 3 * sd)
        self.assertGreater(covered / trials, 0.95)


class OldFraction(Fraction):
    """A fraction as Python 3.11 has it: with no format of its own for a float-style specification."""

    def __format__(self, spec):
        if spec:
            raise TypeError("unsupported format string passed to Fraction.__format__")
        return str(self)


class ExactNumbersAtTheBoundary(unittest.TestCase):
    def test_a_fractional_edge_is_shown(self):
        edge = OldFraction(1, 3)
        with self.assertRaises(TypeError):
            format(edge, ".2f")
        got = case.reported([(0, 2)], {"start": "handler", "bounds": [edge]})
        self.assertIs(got, edge)                                               # the open-ended bucket: the edge itself
        self.assertEqual(case.shown(got), "0.33")                              # Version 59 under Python 3.11: TypeError
        self.assertEqual(case.shown(Fraction(1, 3)), "0.33")
        self.assertEqual(case.shown(OldFraction(2, 3)), "0.67")
        self.assertEqual(case.shown(OldFraction(1, 8)), "0.12")                # half to even, as a float is formatted
        self.assertEqual((case.shown(D("250")), case.shown(D("0.125")), case.shown(7)), ("250.00", "0.12", "7.00"))
        self.assertEqual(case.shown(case.reported([], case.OLD)), "no observations")
        self.assertEqual(case.shown(math.nan), "no observations")

    def test_floats_are_shown_as_they_were(self):
        rng = random.Random(311)
        for _ in range(2000):
            x = rng.uniform(-1000, 1000) * 10 ** rng.randint(-3, 3)
            # Since Version 63 a value that rounds to zero is shown without a sign.
            want = f"{x:.2f}".lstrip("-") if not float(f"{x:.2f}") else f"{x:.2f}"
            self.assertEqual(case.shown(x), want)
            self.assertEqual(fixed(x, 2), want)
        before, after = periods()
        self.assertEqual((case.shown(case.reported(before, case.OLD)), case.shown(case.reported(after, case.NEW))),
                         ("191.57", "241.35"))

    def test_a_decimal_limit_on_a_spread(self):
        before = [(0, D("100"))] * 20
        after = [(0, D("100"))] * 19 + [(0, D("200"))]
        d = case.handler_change(before, after)
        self.assertEqual((d.value, type(d.value)), (0, Fraction))
        self.assertGreater(d.u, 0)
        want = NormalDist().cdf(5 / d.u)
        for limit in (D("5"), Fraction(5), 5, 5.0):                            # Version 59: TypeError for the first
            self.assertAlmostEqual(d.prob_below(limit), want, places=12, msg=repr(limit))
        self.assertEqual(f"{d.prob_below(D('5')):.8f}", "0.54426046")          # the value of the review
        self.assertEqual(d.prob_below(D("5")), d.prob_below(Fraction(5)))
        self.assertEqual((d.prob_below(D("Infinity")), d.prob_below(math.inf)), (1.0, 1.0))
        self.assertLess(d.prob_below(D("-Infinity")), 1e-300)
        self.assertEqual(d.prob_below(D("-Infinity")), d.prob_below(-math.inf))

    def test_the_other_pairs_that_python_does_not_subtract(self):
        want = NormalDist().cdf(1.0)
        for value, limit in ((D("1.5"), 2.0), (1.5, D("2")), (D("1.5"), D("2")), (D("1.5"), Fraction(2)),
                             (Fraction(3, 2), D("2.0")), (D("1.5"), 2)):
            self.assertAlmostEqual(Result(value, 0.5).prob_below(limit), want, places=12, msg=repr((value, limit)))
        # A tenth as a decimal and as a float are two numbers, and the difference is that of the two.
        got = Result(D("0.1"), 2e-14).prob_below(0.1)                             # as fine as the package admits
        self.assertAlmostEqual(got, NormalDist().cdf(float(Fraction(0.1) - Fraction(1, 10)) / 2e-14), places=12)
        self.assertGreater(got, 0.5)

    def test_what_was_defined_stays(self):
        for bad in (math.nan, D("NaN")):
            with self.assertRaises(ValueError):
                Result(1.5, 0.5).prob_below(bad)
        for bad in (None, "2"):
            with self.assertRaises(TypeError):
                Result(Fraction(3, 2), 0.5).prob_below(bad)
        # With no uncertainty the result is a point mass, and a decimal limit is compared exactly.
        point = Result(Fraction(3, 10), 0)
        self.assertEqual([point.prob_below(x) for x in (D("0.3"), D("0.29999"), 0.3, math.inf, D("-Infinity"))],
                         [1.0, 0.0, 0.0, 1.0, 0.0])
        # For floats Version 60 divided the float difference by u: two roundings. Since Version 61 the distance
        # is the exact ratio, rounded once; the two differ in the last places at most.
        rng = random.Random(180)
        for _ in range(500):
            value, u, limit = rng.gauss(0, 5), rng.uniform(0.1, 3), rng.gauss(0, 5)
            dof = rng.choice((math.inf, 4, 30))
            z = float((Fraction(limit) - Fraction(value)) / Fraction(u))
            got = Result(value, u, dof=dof).prob_below(limit)
            self.assertEqual(got, t_cdf(z, dof))
            self.assertAlmostEqual(got, t_cdf((limit - value) / u, dof), places=13)


if __name__ == "__main__":
    unittest.main()
