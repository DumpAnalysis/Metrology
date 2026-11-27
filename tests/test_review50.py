"""Regression tests added after the review of Version 50.

The review found that verdict() of Listing 23.2 compared a share with an objective given as a float in binary,
where its default and Chapter 16 read the decimal that the float prints; that within() counted only answers
that are Python's own True, so that readings whose comparison answers with another kind of truth were all
counted as beyond the limit; that an objective for the percentile of rank zero was met by a share of nothing;
that weighted_quantile() returned a value of no weight for that rank; and that it took a complex number for a
value. It asked what a mean that is known only as it prints leaves of Figure 23.2, which Figure 23.3 now draws,
and for the exact mixes at which the comparison of Figure 23.1 reverses.

Each test computes what it expects independently of the code under test: verdicts from every completion of the
unknown requests, with the rank of a quantile taken from its definition; the most durations within a limit by
trying each count against the sums that it allows; the reversal from every mix at which a cumulative share of
either period reaches 95 %; quantiles from a plain cumulative sum.
"""
import contextlib
import io
import itertools
import math
import random
import unittest
from decimal import ROUND_HALF_EVEN, Decimal
from fractions import Fraction

from pathlib import Path

from examples import ch16_bounds as bounds
from examples import ch23_mix as mix
from swmetro.stats import ecdf_quantile, weighted_quantile

try:
    import numpy
except ImportError:                                                                 # the examples need only Python
    numpy = None

ROOT = Path(__file__).resolve().parent.parent


def expected(name):
    return (ROOT / "expected" / f"{name}.txt").read_text(encoding="utf-8")


def printed(function, *args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        function(*args)
    return out.getvalue()


def rank(q, n):
    """The rank, counted from 1, of the q-quantile of n values: the least k with k / n >= q, and 1 for q = 0."""
    return max(1, math.ceil(Fraction(q) * n))


def by_completions(n, q, within, unknown):
    """The objective that the q-quantile of n durations be at most a limit, with `within` of them known to be
    within it and `unknown` open: each number of the open ones is put within in turn, and the quantile is
    within the limit if its rank is among the durations that are."""
    answers = {rank(q, n) <= within + more for more in range(unknown + 1)}
    return "unresolved" if len(answers) == 2 else "met" if answers == {True} else "missed"


class AnObjectiveGivenAsAFloat(unittest.TestCase):
    def test_the_three_calls_of_the_review(self):
        share = Fraction(9, 10)
        self.assertEqual(bounds.quantile_verdict(10, 0.9, 9, 0), "met")
        self.assertEqual(by_completions(10, Fraction(9, 10), 9, 0), "met")
        self.assertEqual(mix.verdict(share, share, need=0.9), "met")
        self.assertEqual(mix.verdict(share, share, need=share), "met")
        self.assertGreater(Fraction(0.9), share)                                 # the binary 0.9 lies above nine tenths

    def test_every_whole_percentage(self):
        above = 0
        for k in range(1, 101):
            share = Fraction(k, 100)
            above += Fraction(k / 100) > share                                   # where Version 50 said "missed"
            for need in (k / 100, share, str(k / 100), Decimal(k) / 100):
                self.assertEqual(mix.verdict(share, share, need), "met", (k, need))
                self.assertEqual(mix.verdict(share - Fraction(1, 1000), share - Fraction(1, 1000), need), "missed")
                self.assertEqual(mix.verdict(share - Fraction(1, 1000), share, need), "unresolved")
            self.assertEqual(bounds.quantile_verdict(100, k / 100, k, 0), "met")
            self.assertEqual(by_completions(100, share, k, 0), "met")
        self.assertEqual(above, 52)

    def test_the_default_is_what_it_was(self):
        for low, high, want in ((Fraction(95, 100), Fraction(95, 100), "met"), (Fraction(94, 100), Fraction(95, 100), "unresolved"),
                                (Fraction(94, 100), Fraction(9499, 10000), "missed"), (Fraction(891, 1000), Fraction(991, 1000), "unresolved"),
                                (Fraction(98, 100), 1, "met")):
            self.assertEqual(mix.verdict(low, high), want)
            self.assertEqual(mix.verdict(low, high, 0.95), want)
            self.assertEqual(mix.verdict(low, high, Fraction(19, 20)), want)

    def test_an_objective_is_a_share(self):
        for need in (-0.1, 1.5, 95, math.nan, None, "many"):
            with self.assertRaises((ValueError, TypeError)):
                mix.verdict(Fraction(1, 2), Fraction(1, 2), need)

    def test_against_every_completion(self):
        rng = random.Random(5023)
        for _ in range(3000):
            n = rng.randint(1, 30)
            within = rng.randint(0, n)
            unknown = rng.randint(0, n - within)
            q = Fraction(rng.randint(0, 20), 20)
            want = by_completions(n, q, within, unknown)
            low, high = Fraction(within, n), Fraction(within + unknown, n)
            for need in (q, float(q), str(float(q))):
                self.assertEqual(mix.verdict(low, high, need), want, (n, within, unknown, need))
            self.assertEqual(bounds.quantile_verdict(n, float(q), within, unknown), want, (n, within, unknown, q))


class TheLeastDuration(unittest.TestCase):
    def test_ten_requests_of_300_ms(self):
        xs = [300.0] * 10
        self.assertEqual(ecdf_quantile(xs, 0), 300.0)                            # rank one: the least of them
        self.assertEqual(rank(0, 10), 1)
        low, high = mix.within({"all": xs}, {"all": range(10)}, 250)
        self.assertEqual((low, high), (0, 0))
        self.assertEqual(mix.verdict(low, high, 0), "missed")                    # "met" in Version 50
        self.assertEqual(bounds.quantile_verdict(10, 0, 0, 0), "missed")
        self.assertEqual(by_completions(10, 0, 0, 0), "missed")

    def test_the_rule_for_rank_zero(self):
        half, none, all_ = Fraction(1, 2), Fraction(0), Fraction(1)
        table = (((none, none), "missed"), ((none, half), "unresolved"), ((none, all_), "unresolved"),
                 ((Fraction(1, 1000), half), "met"), ((half, half), "met"), ((all_, all_), "met"))
        for (low, high), want in table:
            for need in (0, 0.0, Fraction(0), "0"):
                self.assertEqual(mix.verdict(low, high, need), want, (low, high, need))

    def test_weights_and_counts_agree_at_rank_zero(self):
        for n in range(1, 8):
            for within in range(n + 1):
                for unknown in range(n - within + 1):
                    want = by_completions(n, 0, within, unknown)
                    self.assertEqual(mix.verdict(Fraction(within, n), Fraction(within + unknown, n), 0), want)
                    self.assertEqual(bounds.quantile_verdict(n, 0, within, unknown), want)
                    xs = [100.0] * within + [None] * unknown + [300.0] * (n - within - unknown)
                    low, high = mix.within({"all": xs}, {"all": range(n)}, 200)
                    self.assertEqual(mix.verdict(low, high, 0), want)

    def test_a_share_above_nothing_is_as_before(self):
        self.assertEqual(mix.verdict(0, 0, 0.5), "missed")
        self.assertEqual(mix.verdict(0, Fraction(1, 2), 0.5), "unresolved")
        self.assertEqual(mix.verdict(0, 0, 1), "missed")
        self.assertEqual(mix.verdict(1, 1, 1), "met")


class Yes:
    """An answer to a comparison that is true or false and is not Python's True or False."""

    def __init__(self, value):
        self.value = value

    def __bool__(self):
        return self.value


class Reading(float):
    """A number whose comparisons answer with such an object, as readings of an array library do."""

    def __le__(self, other):
        return Yes(float(self) <= other)


class AnswersThatAreNotPythonsTrue(unittest.TestCase):
    REF = {"fast": range(90), "slow": range(10)}

    def records(self, make):
        fast = [make(100.0)] * 90
        slow = [make(100.0)] * 4 + [make(300.0)] * 5
        return {"fast": fast, "slow": slow + [make(210.0)]}, {"fast": fast, "slow": slow + [(">", 200)]}

    def check(self, make):
        done, bound = self.records(make)
        low, high = mix.within(done, self.REF, 250)
        self.assertEqual((low, high, mix.verdict(low, high)), (Fraction(95, 100), Fraction(95, 100), "met"))
        low, high = mix.within(bound, self.REF, 250)
        self.assertEqual((low, high, mix.verdict(low, high)), (Fraction(94, 100), Fraction(95, 100), "unresolved"))
        self.assertIs(mix.side(make(100.0), 250), True)
        self.assertIs(mix.side(make(300.0), 250), False)
        self.assertIs(mix.side(make(250.0), 250), True)

    def test_a_stand_in(self):
        self.assertIsNot(Reading(100.0) <= 250, True)                            # true, and not True: counted as
        self.assertTrue(Reading(100.0) <= 250)                                   # beyond the limit in Version 50
        self.check(Reading)
        self.check(float)

    @unittest.skipUnless(numpy, "needs NumPy, which the examples do not")
    def test_the_table_of_the_review(self):
        self.assertIsNot(numpy.float64(100.0) <= 250, True)
        for make in (numpy.float64, numpy.float32, numpy.int64):
            self.check(make)
        self.assertEqual(weighted_quantile([(numpy.float64(2.5), 1), (numpy.float32(1.5), 1), (numpy.int64(3), 1)], 0.5), 2.5)

    def test_other_numbers(self):
        self.check(lambda x: Fraction(int(x)))
        self.check(lambda x: Decimal(int(x)))
        self.check(int)

    def test_what_is_open_stays_open(self):
        for x in (None, math.nan, (">", 200), (">=", 250)):
            self.assertIsNone(mix.side(x, 250))
        self.assertIs(mix.side((">", 250), 250), False)
        self.assertIs(mix.side((">", 300), 250), False)


def allowed(k, n, total, limit, cap):
    """Whether k of n durations, none below zero and none above the cap, can be at or below the limit and
    the others above it, with this sum."""
    rest = n - k
    if rest == 0:
        return 0 <= total <= n * min(limit, cap)
    if cap <= limit or not total > rest * limit:
        return False
    return total <= k * limit + rest * cap


def most_with(known, mean, n=900, limit=200, cap=810):
    """The most of n durations within the limit when some of them are known: those of the known that are
    within it, and the largest count of the others that their sum allows."""
    total = n * Fraction(mean) - sum(known)
    rest = n - len(known)
    return sum(x <= limit for x in known) + next(k for k in range(rest, -1, -1) if allowed(k, rest, total, limit, cap))


class TheMeanAsItPrints(unittest.TestCase):
    FAST = Fraction(891, 1000)

    def share(self, k):
        return self.FAST + Fraction(1, 10) * Fraction(k, 900)

    def test_what_a_printed_mean_allows(self):
        for digits, half in ((0, "0.5"), (1, "0.05"), (2, "0.005")):
            unit = Decimal(1).scaleb(-digits)
            least, tiny = Decimal(450) - Decimal(half), Decimal("0.0000001")
            self.assertEqual(least.quantize(unit, ROUND_HALF_EVEN), Decimal(450))           # the tie goes to the even 450
            self.assertLess((least - tiny).quantize(unit, ROUND_HALF_EVEN), Decimal(450))
            self.assertEqual(f"{float(least + tiny):.{digits}f}", f"{450:.{digits}f}")

    def test_the_table_of_the_review(self):
        mean = Fraction(100 + 530 * 200 + 369 * 810, 900)
        self.assertEqual(f"{float(mean):.6f}", "449.988889")
        self.assertEqual(f"{float((450 - mean) * 1000):.3f}", "11.111")          # that much below 450 ms, in microseconds
        self.assertEqual([f"{float(mean):.{d}f}" for d in (0, 1, 2)], ["450", "450.0", "449.99"])
        self.assertEqual(most_with([100], mean), 531)
        self.assertEqual(most_with([100], 450), 530)
        self.assertEqual(100 + 529 * 200 + 370 * Fraction(29910, 37), 900 * 450)
        self.assertEqual((mix.verdict(self.FAST, self.share(531)), mix.verdict(self.FAST, self.share(530))), ("unresolved", "missed"))
        self.assertEqual(mix.verdict(self.share(531), self.share(531)), "met")   # with all 900 durations as in that row
        self.assertGreater(Fraction("449.995"), mean)                            # 450.00 rules that row out

    def test_with_one_is_the_two_lines_of_version_50(self):
        for x in (0, 50, 100, 109, 110, 150, 200, 201, 400, 700, 719, 720, 800, 810):
            for mean in (450, Fraction("449.995"), Fraction("449.95"), Fraction("449.5"), "449.95", 449.5):
                want = most_with([x], Fraction(str(mean)))
                self.assertEqual(mix.with_one(x, mean), want, (x, mean))
                self.assertEqual(mix.with_one(x, mean, 900), want)
        self.assertEqual(mix.with_one(100, 450), int(100 <= 200) + mix.most_within(899, Fraction(405000 - 100, 899), 200, 810))

    def test_where_a_miss_is_certain(self):
        found = {}
        for name, half in (("exact", 0), ("450.00", Fraction(5, 1000)), ("450.0", Fraction(5, 100)), ("450", Fraction(1, 2))):
            runs, before = [], None
            for k in range(810 * 4 + 1):                                         # 0 to 810 ms in quarters
                x = Fraction(k, 4)
                certain = most_with([x], 450 - half) <= 530
                self.assertEqual(mix.with_one(x, 450 - half) <= 530, certain)
                # A mean anywhere in what the print allows gives no more than its least value does.
                self.assertTrue(all(most_with([x], 450 - half + part * 2 * half) <= most_with([x], 450 - half)
                                    for part in (Fraction(1, 3), Fraction(1, 2), 1)))
                if certain != before:
                    runs.append((x, certain))
                    before = certain
            found[name] = runs
            first, second = 110 - 900 * half, 720 - 900 * half                   # where 531 first fit the least mean
            for end in (first, second):
                if end >= 0:
                    self.assertGreater(most_with([end], 450 - half), 530)        # the end itself is not in the bar
            self.assertLessEqual(most_with([Fraction(200)], 450 - half) - 531, 0)
            self.assertGreater(most_with([Fraction(200)], 450 - half), 530)      # 200 ms leaves it open in every row
        quarter = Fraction(1, 4)
        self.assertEqual(found["exact"], [(0, True), (110, False), (200 + quarter, True), (720, False)])
        self.assertEqual(found["450.00"], [(0, True), (Fraction(211, 2), False), (200 + quarter, True), (Fraction(1431, 2), False)])
        self.assertEqual(found["450.0"], [(0, True), (65, False), (200 + quarter, True), (675, False)])
        self.assertEqual(found["450"], [(0, False), (200 + quarter, True), (270, False)])

    def test_two_recovered_durations(self):
        self.assertEqual(531 * 200 + 369 * 810 - 900 * 450, 90)                  # what can be given up
        self.assertEqual(most_with([150, 150], 450), 530)                        # they give up 100
        self.assertEqual(most_with([150, 160], 450), 531)                        # exactly 90
        self.assertEqual(most_with([150, 159], 450), 530)
        self.assertEqual(150 + 160 + 529 * 200 + 369 * 810, 900 * 450)

    def test_the_transcript(self):
        lines = expected("ch23_mix").splitlines()
        self.assertEqual(len(lines), 122)                                        # 100 in Version 51, 103 until 57, 108 until 65
        at = lines.index("  at 100 ms, the mean as it prints")
        rows = []
        for digits, half in ((2, Fraction(5, 1000)), (1, Fraction(5, 100)), (0, Fraction(1, 2))):
            top = self.share(most_with([100], 450 - half))
            verdict = "missed" if top < Fraction(95, 100) else "unresolved"
            rows.append(f"    {450:.{digits}f} ms: to {100 * float(top):.2f} %, {verdict}")
        self.assertEqual(lines[at + 1:at + 4], rows)
        self.assertEqual(rows, ["    450.00 ms: to 94.99 %, missed", "    450.0 ms: to 95.00 %, unresolved", "    450 ms: to 95.00 %, unresolved"])
        self.assertEqual(lines[at - 4:at], ["  none above 810 ms, one recovered", "    at 100 ms: to 94.99 %, missed",
                                            "    at 110 ms: to 95.00 %, unresolved", "    at 700 ms: to 94.99 %, missed"])
        self.assertEqual(lines[at + 7], "  98 % known within, 2 % missing: met")    # after three lines of Version 52
        self.assertEqual(printed(mix.main), expected("ch23_mix"))


class TheGeneratedRecord(unittest.TestCase):
    def test_what_the_printed_seed_gives(self):
        rng = random.Random(23)
        before = mix.record(mix.BEFORE, rng)
        after = mix.record(mix.AFTER, rng)
        slow, fast = after["slow"], after["fast"]
        self.assertEqual((len(slow), len(fast)), (900, 100))
        self.assertEqual(sum(x > 810 for x in slow), 19)
        self.assertEqual(f"{max(slow):.2f}", "1340.45")
        self.assertEqual((sum(x <= 200 for x in slow), sum(x <= 200 for x in fast)), (9, 99))
        share = Fraction(9, 10) * Fraction(99, 100) + Fraction(1, 10) * Fraction(9, 900)
        self.assertEqual(f"{100 * float(share):.2f}", "89.20")
        self.assertEqual(mix.within(after, before, 200), (share, share))
        line = next(x for x in expected("ch23_mix").splitlines() if x.startswith("at old mix"))
        self.assertEqual(line.split(), ["at", "old", "mix", "126.00", "432.40", "89.20", "%"])
        # The cap of 810 ms and the request of 100 ms are suppositions: the record has neither.
        self.assertGreater(max(slow), 810)
        self.assertGreater(min(slow), 100)


def exact_p95(rec, w):
    """The 95th percentile at a fraction w of slow requests: the smallest duration with a cumulative share of
    at least 95 %, the shares added as fractions."""
    points = sorted([(x, (1 - w) / len(rec["fast"])) for x in rec["fast"]] + [(x, w / len(rec["slow"])) for x in rec["slow"]])
    acc = Fraction(0)
    for x, weight in points:
        acc += weight
        if acc >= Fraction(95, 100):
            return x


def reaching(rec):
    """Every fraction of slow requests at which the cumulative share at some duration of the record is
    exactly 95 %: the share is linear in the fraction."""
    out = set()
    for x in rec["fast"] + rec["slow"]:
        fast = Fraction(sum(y <= x for y in rec["fast"]), len(rec["fast"]))
        slow = Fraction(sum(y <= x for y in rec["slow"]), len(rec["slow"]))
        if fast != slow:
            w = (Fraction(95, 100) - fast) / (slow - fast)
            if 0 <= w <= 1:
                out.add(w)
    return out


class TheReversalOfFigure231(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = random.Random(23)
        cls.before = mix.record(mix.BEFORE, rng)
        cls.after = mix.record(mix.AFTER, rng)

    def test_the_two_fractions(self):
        marks = sorted(reaching(self.before) | reaching(self.after) | {Fraction(0), Fraction(1)})
        higher = []                                                              # where the second period is above
        for a, b in zip(marks, marks[1:]):
            for w in (a, (a + b) / 2):
                if exact_p95(self.after, w) > exact_p95(self.before, w):
                    higher.append(w)
        if exact_p95(self.after, marks[-1]) > exact_p95(self.before, marks[-1]):
            higher.append(marks[-1])
        low, high = Fraction(12, 295), Fraction(14, 299)
        self.assertTrue(all(low < w <= high for w in higher))
        self.assertIn(high, higher)                                              # the upper end belongs to it
        self.assertNotIn(low, higher)                                            # and the lower does not
        between = [w for w in marks if low < w <= high]
        self.assertTrue(all(w in higher for w in between))
        self.assertEqual((f"{100 * float(low):.6f}", f"{100 * float(high):.6f}"), ("4.067797", "4.682274"))
        self.assertEqual((f"{100 * float(low):.2f}", f"{100 * float(high):.2f}"), ("4.07", "4.68"))      # the caption

    def test_the_table_of_the_review(self):
        tiny = Fraction(1, 10 ** 12)
        rows = ((Fraction(12, 295), "194.917", "193.332"), (Fraction(12, 295) + tiny, "194.917", "195.388"),
                (Fraction(14, 299), "209.249", "216.250"), (Fraction(14, 299) + tiny, "220.154", "216.250"))
        for w, first, second in rows:
            self.assertEqual((f"{exact_p95(self.before, w):.3f}", f"{exact_p95(self.after, w):.3f}"), (first, second), w)

    def test_the_counts_behind_the_two_equalities(self):
        x = exact_p95(self.after, Fraction(12, 295))
        counts = (sum(y <= x for y in self.after["fast"]), sum(y <= x for y in self.after["slow"]))
        self.assertEqual((f"{x:.6f}", counts), ("193.332170", (99, 6)))
        w = Fraction(12, 295)
        self.assertEqual((1 - w) * Fraction(99, 100) + w * Fraction(6, 900), Fraction(95, 100))
        x = exact_p95(self.before, Fraction(14, 299))
        counts = (sum(y <= x for y in self.before["fast"]), sum(y <= x for y in self.before["slow"]))
        self.assertEqual((f"{x:.6f}", counts), ("209.249487", (897, 0)))
        w = Fraction(14, 299)
        self.assertEqual((1 - w) * Fraction(897, 900), Fraction(95, 100))

    def test_the_listing_at_those_mixes(self):
        for slow, of, second_higher in ((12, 295, False), (14, 299, True), (41, 1000, True), (40, 1000, False), (47, 1000, False)):
            first, second = mix.at_share(self.before, slow, of), mix.at_share(self.after, slow, of)
            self.assertEqual((first, second), (exact_p95(self.before, Fraction(slow, of)), exact_p95(self.after, Fraction(slow, of))))
            self.assertEqual(second > first, second_higher, (slow, of))


def plain_quantile(pairs, q):
    """The smallest value of some weight with at least a share q of the weight at or below it."""
    pairs = sorted((x, w) for x, w in pairs if w)
    total = sum(w for _, w in pairs)
    for i, (x, _) in enumerate(pairs):
        if sum(w for _, w in pairs[:i + 1]) * q.denominator >= q.numerator * total:
            return x


class TheLeastValueThatHasWeight(unittest.TestCase):
    def test_the_call_of_the_review(self):
        self.assertEqual(weighted_quantile([(1, 0), (10, 1)], 0), 10)            # 1 in Version 50
        self.assertEqual(weighted_quantile([(3, 0), (5, 2), (9, 0)], 0), 5)
        self.assertEqual(weighted_quantile([(3, 0), (5, 2), (9, 0)], 1), 5)
        self.assertEqual(weighted_quantile([(10, 1), (1, 0.0), (0, Fraction(0))], 0), 10)

    def test_whole_weights_are_repetitions(self):
        rng = random.Random(5024)
        for _ in range(400):
            pairs = [(rng.randint(0, 9) + rng.choice([0, 0.5]), rng.randint(0, 3)) for _ in range(rng.randint(1, 8))]
            if not any(w for _, w in pairs):
                pairs.append((4.5, 2))
            repeated = sorted(x for x, w in pairs for _ in range(w))
            for q in (0, Fraction(1, 20), 0.5, 0.95, 1):
                self.assertEqual(weighted_quantile(pairs, q), ecdf_quantile(repeated, q), (pairs, q))
                self.assertEqual(weighted_quantile(pairs, q), plain_quantile(pairs, Fraction(str(q))), (pairs, q))

    def test_only_rank_zero_changed(self):
        rng = random.Random(5025)
        for _ in range(400):
            pairs = [(rng.uniform(-5, 5), rng.choice([0, 0, 1, 2, Fraction(1, 3)])) for _ in range(rng.randint(1, 10))]
            if not any(w for _, w in pairs):
                pairs.append((0.25, 1))
            q = Fraction(rng.randint(1, 20), 20)
            ordered = sorted(pairs)                                              # with the values of no weight in place
            total = sum(w for _, w in ordered)
            want = next(x for i, (x, _) in enumerate(ordered)
                        if sum(w for _, w in ordered[:i + 1]) * q.denominator >= q.numerator * total)
            self.assertEqual(weighted_quantile(pairs, q), want, (pairs, q))

    def test_weights_are_checked_first(self):
        for pairs in ([], [(1.0, 0)], [(1.0, 2), (2.0, -1)], [(1.0, 0), (2.0, 0)]):
            with self.assertRaises(ValueError) as caught:
                weighted_quantile(pairs, 0)
            self.assertEqual(str(caught.exception), "need weights >= 0, not all zero")


class RealNumbersInOrder(unittest.TestCase):
    def test_a_complex_number_is_no_measurement_here(self):
        for pairs in ([(2 + 3j, 1)], [(2 + 3j, 1), (1.0, 1)], [(1.0, 1), (2 + 0j, 1)], [(complex(1, 0), 2)]):
            with self.assertRaises(ValueError) as caught:
                weighted_quantile(pairs, 0.5)
            self.assertEqual(str(caught.exception), "a value is no number")

    def test_decimals_and_fractions_stay(self):
        pairs = [(Decimal("1.5"), 1), (Fraction(7, 3), 1), (2, 1), (0.5, 1), (True, 1)]
        self.assertEqual([weighted_quantile(pairs, q) for q in (0, 0.5, 1)], [0.5, Decimal("1.5"), Fraction(7, 3)])
        self.assertEqual(weighted_quantile([(math.inf, 1), (-math.inf, 1)], 0.5), -math.inf)

    def test_what_was_refused_is_refused(self):
        for value in (None, math.nan, Decimal("NaN"), "2", (">", 200), [1], {1}):
            for pairs in ([(value, 1)], [(1.0, 1), (value, 1)], [(value, 0), (1.0, 1)]):
                with self.assertRaises(ValueError) as caught:
                    weighted_quantile(pairs, 0.5)
                self.assertEqual(str(caught.exception), "a value is no number")


class TheThirtyFirstFigure(unittest.TestCase):
    def test_the_script(self):
        source = (ROOT / "figures" / "make_figures.py").read_text(encoding="utf-8")
        self.assertEqual(source.count("\n    save(fig, "), 32)
        for piece in ("def mean_as_printed():", 'save(fig, "fig-23-3-mean-as-printed.png")', "c23.with_one(x, 450 - half) <= 530",
                      "[110, 720, 105.5, 715.5, 65, 675, 270]", "    one_mix()\n    one_recovered()\n    mean_as_printed()\n"):
            self.assertIn(piece, source)

    def test_what_it_draws(self):
        ends = []
        for half in (0, Fraction(5, 1000), Fraction(5, 100), Fraction(1, 2)):
            first, second = 110 - 900 * half, 720 - 900 * half
            for a, b in ([(0, first)] if first > 0 else []) + [(200, second)]:
                inside = {most_with([Fraction(k, 4)], 450 - half) <= 530 for k in range(int(4 * a) + 1, int(4 * b))}
                self.assertEqual(inside, {True}, (half, a, b))
                self.assertGreater(most_with([b], 450 - half), 530)              # an open circle at the right end
                self.assertEqual(most_with([Fraction(a)], 450 - half) <= 530, a == 0)   # filled at 0, open at 200
                ends.append(float(b))
        self.assertEqual(ends, [110, 720, 105.5, 715.5, 65, 675, 270])


if __name__ == "__main__":
    unittest.main()
