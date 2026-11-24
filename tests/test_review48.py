"""Regression tests added after the review of Version 48.

The review found that partly() of Listing 17.1 read its batches more than once, so that batches given as an
iterator were gone after the first reading and one probability came back that no table of the batches allows;
and that within() of Listing 23.2 counted a request whose duration is not a number as one beyond the limit, and
gave a verdict that the record does not support. It asked what one doubtful batch label does to the fifty
pairs of Chapter 17, which Figure 17.3 now draws, and what a mean and an upper limit settle of a class whose
durations are lost. The caption of Figure 6.6 is a matter of the book; the test here keeps the figure itself.

Each test computes what it expects independently of the code under test: the joint outcomes of a batch from
every pair of sets of right answers, the batches behind doubtful labels by moving a pair of answers at a time, the
bounds of a record with unknown durations from every way of completing it, and the bound from a mean and an
upper limit by trying each count and building the durations that attain it.
"""
import contextlib
import io
import itertools
import math
import random
import unittest
from fractions import Fraction
from pathlib import Path

from examples import ch17_accuracy as accuracy
from examples import ch23_mix as mix
from swmetro.stats import weighted_quantile

ROOT = Path(__file__).resolve().parent.parent
KNOWN = (1560, 132, 163, 95)
SCORES = (1722, 1758, 2000)
LEVEL = Fraction(1, 20)


def expected(name):
    return (ROOT / "expected" / f"{name}.txt").read_text(encoding="utf-8")


def printed(function, *args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        function(*args)
    return out.getvalue()


def two_sided(only_a, only_b):
    """The exact two-sided probability of a division at least as uneven, from binomial coefficients."""
    n, low = only_a + only_b, min(only_a, only_b)
    return min(Fraction(1), 2 * Fraction(sum(math.comb(n, k) for k in range(low + 1)), 2 ** n))


def paired(size, right_a, right_b):
    """Each number of items of a batch that both models can have right: every set of right answers of A is laid
    over every set of right answers of B."""
    items = range(size)
    return sorted({len(set(a) & set(b)) for a in itertools.combinations(items, right_a)
                   for b in itertools.combinations(items, right_b)})


def cells(size, right_a, right_b):
    """The same for batches too large to lay out: the four cells of the table are tried."""
    return [both for both in range(size + 1)
            if min(right_a - both, right_b - both, size - right_a - right_b + both) >= 0]


def probabilities(known, batches, joint=cells):
    """The totals of items right for both that the batches allow, from the most, and their probabilities."""
    both, only_a, only_b, _ = known
    right_a = sum(a for _, a, _ in batches)
    right_b = sum(b for _, _, b in batches)
    totals = sorted({sum(choice) for choice in itertools.product(*(joint(*batch) for batch in batches))}, reverse=True)
    return totals, [two_sided(only_a + right_a - j, only_b + right_b - j) for j in totals]


def relabeled(first, second, most):
    """The right answers of B that two batches can have had when up to `most` answers of B in each carry the
    label of the other: put back one pair at a time, two answers that change places move a right answer one way
    and a wrong one the other, or change nothing."""
    (m1, _, b1), (m2, _, b2) = first, second
    seen = {(b1, b2)}
    for _ in range(most):
        for x, y in list(seen):
            if x > 0 and y < m2:                                                 # a right one leaves the first batch
                seen.add((x - 1, y + 1))
            if x < m1 and y > 0:                                                 # a right one comes into it
                seen.add((x + 1, y - 1))
    return sorted(seen)


def allowed(first, second, most):
    """Totals right for both, of 2000, and probabilities that the two batches allow with such labels."""
    totals, ps = set(), set()
    for b1, b2 in relabeled(first, second, most):
        ts, qs = probabilities(KNOWN, [(first[0], first[1], b1), (second[0], second[1], b2)])
        totals |= {KNOWN[0] + t for t in ts}
        ps |= set(qs)
    return sorted(totals), sorted(ps)


FIRST = ((25, 23, 23), (25, 7, 12))
SECOND = ((25, 23, 13), (25, 7, 22))


class BatchesReadOnce(unittest.TestCase):
    ROWS = [(25, 23, 23), (25, 7, 12)]

    def test_the_call_of_the_review(self):
        want = probabilities(KNOWN, self.ROWS)[1]
        self.assertEqual(len(want), 10)
        self.assertEqual((f"{float(want[0]):.6f}", f"{float(want[-1]):.6f}"), ("0.043128", "0.049510"))
        givens = (self.ROWS, tuple(self.ROWS), iter(self.ROWS), (row for row in self.ROWS),
                  map(list, self.ROWS), iter(map(iter, self.ROWS)), reversed(self.ROWS[::-1]))
        for parts in givens:
            self.assertEqual(accuracy.partly(KNOWN, *SCORES, parts), want, type(parts).__name__)

    def test_what_version_48_returned(self):
        # With the batches gone after one reading the sums over them were sums over nothing: no item of the fifty
        # right for both, which the first batch alone excludes, since 23 and 23 of 25 share 21 at least.
        lost = two_sided(132 + 30, 163 + 35)
        self.assertEqual(f"{float(lost):.6f}", "0.064937")
        self.assertEqual(cells(25, 23, 23)[0], 21)
        self.assertNotIn(lost, accuracy.partly(KNOWN, *SCORES, iter(self.ROWS)))
        self.assertNotIn(lost, accuracy.partly(KNOWN, *SCORES))
        self.assertGreater(lost, max(accuracy.partly(KNOWN, *SCORES)))

    def test_the_verified_pairs_are_read_once_too(self):
        want = accuracy.partly(KNOWN, *SCORES, self.ROWS)
        for known in (list(KNOWN), iter(KNOWN), (k for k in KNOWN), map(int, KNOWN)):
            self.assertEqual(accuracy.partly(known, *SCORES, iter(self.ROWS)), want)
        for known in (KNOWN[:3], KNOWN + (0,), iter(KNOWN[:2])):
            with self.assertRaises(ValueError):
                accuracy.partly(known, *SCORES)

    def test_the_arguments_are_left_as_they_were(self):
        rows, known = [list(row) for row in self.ROWS], list(KNOWN)
        accuracy.partly(known, *SCORES, rows)
        self.assertEqual((rows, known), ([list(row) for row in self.ROWS], list(KNOWN)))

    def test_every_pair_of_small_batches(self):
        known, cases = (2, 1, 3, 1), 0
        shapes = [(m, a, b) for m in range(4) for a in range(m + 1) for b in range(m + 1)]
        self.assertEqual(len(shapes), 1 + 4 + 9 + 16)
        for one, two in itertools.product(shapes, repeat=2):
            rows = [one, two]
            right_a = known[0] + known[1] + one[1] + two[1]
            right_b = known[0] + known[2] + one[2] + two[2]
            n = sum(known) + one[0] + two[0]
            totals, want = probabilities(known, rows, paired)
            self.assertEqual(totals, probabilities(known, rows, cells)[0])       # the two ways of counting agree
            for parts in (rows, iter(rows), (row for row in rows), iter(map(iter, rows))):
                self.assertEqual(accuracy.partly(known, right_a, right_b, n, parts), want, rows)
            cases += 1
        self.assertEqual(cases, 30 ** 2)


class RowsThatAreBatches(unittest.TestCase):
    def refused(self, parts, message, known=KNOWN, scores=SCORES):
        for given in (parts, iter(parts)):
            with self.assertRaises(ValueError) as caught:
                accuracy.partly(known, *scores, given)
            self.assertEqual(str(caught.exception), message, parts)

    def test_three_whole_counts_in_each(self):
        good = [(25, 23, 23), (25, 7, 12)]
        self.assertEqual(len(accuracy.partly(KNOWN, *SCORES, good)), 10)
        for row, place in itertools.product(range(2), range(3)):
            for bad in (-1, 1.0, 2.5, True, "3", None, Fraction(3), math.nan):
                parts = [list(r) for r in good]
                parts[row][place] = bad
                self.refused(parts, "need whole counts >= 0")
        self.refused([(60, 40, 35), (-10, -10, 0)], "need whole counts >= 0")   # sums that fit, and no batch

    def test_neither_fewer_nor_more(self):
        for parts in ([(25, 23), (25, 7)], [(25, 23, 23, 0), (25, 7, 12, 0)], [(25, 23, 23), (25, 7, 12, 0)],
                      [(25, 23, 23, 0), (25, 7, 12)], [(25, 23, 23), (25, 7)], [(), ()], [(50,), (30,), (35,)]):
            self.refused(parts, "counts do not fit")

    def test_scores_within_the_batch_and_sums_that_fit(self):
        for parts in ([(25, 26, 23), (25, 4, 12)], [(25, 23, 26), (25, 7, 9)], [(25, 23, 23), (25, 7, 11)],
                      [(25, 23, 23), (24, 7, 12)], [(25, 23, 23), (25, 7, 12), (0, 0, 1)], [(50, 31, 35)],
                      [(25, 23, 23)], [(25, 23, 23), (25, 7, 12), (25, 0, 0)]):
            self.refused(parts, "counts do not fit")

    def test_no_batches_are_not_one_batch(self):
        # An empty list was read as no list in Version 48, and the answer for all fifty pairs came back.
        self.assertEqual(len(accuracy.partly(KNOWN, *SCORES, None)), 16)
        self.refused([], "counts do not fit")
        self.refused((), "counts do not fit")
        all_known, scores = (3, 1, 2, 4), (4, 5, 10)
        want = [two_sided(1, 2)]
        for parts in (None, [], (), iter([]), [(0, 0, 0)], [(0, 0, 0), (0, 0, 0)]):
            self.assertEqual(accuracy.partly(all_known, *scores, parts), want)

    def test_a_batch_of_no_items_changes_nothing(self):
        want = accuracy.partly(KNOWN, *SCORES, [(25, 23, 23), (25, 7, 12)])
        self.assertEqual(accuracy.partly(KNOWN, *SCORES, [(25, 23, 23), (0, 0, 0), (25, 7, 12)]), want)

    def test_random_batches_in_any_container(self):
        rng = random.Random(1748)
        for _ in range(300):
            rows = []
            for _ in range(rng.randint(1, 4)):
                size = rng.randint(0, 6)
                rows.append((size, rng.randint(0, size), rng.randint(0, size)))
            known = tuple(rng.randint(0, 4) for _ in range(4))
            right_a = known[0] + known[1] + sum(a for _, a, _ in rows)
            right_b = known[0] + known[2] + sum(b for _, _, b in rows)
            n = sum(known) + sum(size for size, _, _ in rows)
            want = probabilities(known, rows)[1]
            for parts in (rows, iter(rows), (list(row) for row in rows)):
                self.assertEqual(accuracy.partly(iter(known), right_a, right_b, n, parts), want, rows)


class OneLabelInDoubt(unittest.TestCase):
    def test_what_one_label_can_hide(self):
        self.assertEqual(relabeled(*FIRST, 0), [(23, 12)])
        self.assertEqual(relabeled(*FIRST, 1), [(22, 13), (23, 12), (24, 11)])
        self.assertEqual(relabeled(*SECOND, 1), [(12, 23), (13, 22), (14, 21)])
        self.assertEqual(relabeled(*FIRST, 3), [(23 + j, 12 - j) for j in (-3, -2, -1, 0, 1, 2)])   # 25 at most
        self.assertEqual(len(relabeled(*SECOND, 30)), 26 - 10)                   # B is right 35 times in all

    def test_the_table_of_the_review(self):
        table = ((FIRST, 0, (1581, 1590), ("0.043128", "0.049510")),
                 (FIRST, 1, (1580, 1590), ("0.043128", "0.050229")),
                 (SECOND, 0, (1575, 1580), ("0.050229", "0.053854")),
                 (SECOND, 1, (1575, 1581), ("0.049510", "0.053854")))
        for batches, most, both, shown in table:
            totals, ps = allowed(*batches, most)
            self.assertEqual(totals, list(range(both[0], both[1] + 1)), (batches, most))
            self.assertEqual((accuracy.shown(ps[0]), accuracy.shown(ps[-1])), shown)
            self.assertEqual((f"{float(ps[0]):.6f}", f"{float(ps[-1]):.6f}"), shown)

    def test_either_decision_after_one(self):
        for batches in (FIRST, SECOND):
            before, after = allowed(*batches, 0)[1], allowed(*batches, 1)[1]
            self.assertTrue(all(p < LEVEL for p in before) or all(p > LEVEL for p in before))
            self.assertTrue(after[0] < LEVEL < after[-1])
            self.assertEqual(len(after), len(before) + 1)                        # one more table, across the level
        # The table that one doubtful label adds to each export is the nearest one of the other.
        self.assertEqual(allowed(*FIRST, 1)[1][-1], allowed(*SECOND, 0)[1][0])
        self.assertEqual(allowed(*SECOND, 1)[1][0], allowed(*FIRST, 0)[1][-1])
        # Of the counts of B that one such label can hide, one pair reaches across in each export.
        self.assertTrue(max(probabilities(KNOWN, [(25, 23, 24), (25, 7, 11)])[1]) < LEVEL)
        self.assertTrue(max(probabilities(KNOWN, [(25, 23, 22), (25, 7, 13)])[1]) > LEVEL)
        self.assertTrue(min(probabilities(KNOWN, [(25, 23, 12), (25, 7, 23)])[1]) > LEVEL)
        self.assertTrue(min(probabilities(KNOWN, [(25, 23, 14), (25, 7, 21)])[1]) < LEVEL)

    def test_when_the_batches_say_nothing_more(self):
        whole = accuracy.partly(KNOWN, *SCORES)
        self.assertEqual(len(whole), 16)
        reach = {name: [allowed(*batches, most)[1] for most in range(12)]
                 for name, batches in (("first", FIRST), ("second", SECOND))}
        self.assertEqual([ps[-1] == whole[-1] for ps in reach["first"]].index(True), 6)
        self.assertEqual([ps[0] == whole[0] for ps in reach["second"]].index(True), 10)
        self.assertEqual([ps == whole for ps in reach["first"]].index(True), 6)
        self.assertEqual([ps == whole for ps in reach["second"]].index(True), 10)
        self.assertEqual(reach["first"][11], whole)
        percent = lambda p: f"{100 * float(p):.4f}"
        self.assertEqual([percent(reach["first"][s][-1]) for s in (0, 1, 6)], ["4.9510", "5.0229", "5.3854"])
        self.assertEqual([percent(reach["second"][s][0]) for s in (0, 1, 10)], ["5.0229", "4.9510", "4.3128"])
        for name in reach:                                                       # more doubt never allows less
            for fewer, more in zip(reach[name], reach[name][1:]):
                self.assertTrue(set(fewer) <= set(more))

    def test_the_transcript(self):
        text = expected("ch17_accuracy")
        lines = text.splitlines()
        self.assertEqual(len(lines), 43)
        rows = []
        for b1, b2 in ((23, 12), (13, 22), (22, 13), (14, 21)):
            ps = probabilities(KNOWN, [(25, 23, b1), (25, 7, b2)])[1]
            rows.append(f"  A 23, 7; B {b1}, {b2}: {float(min(ps)):.6f} to {float(max(ps)):.6f}")
        self.assertEqual(lines[38:], ["the 50 in two batches of 25:"] + rows)
        self.assertEqual(rows[2:], ["  A 23, 7; B 22, 13: 0.043828 to 0.050229", "  A 23, 7; B 14, 21: 0.049510 to 0.053854"])
        self.assertEqual(printed(accuracy.main), text)


class TheTwentyNinthFigure(unittest.TestCase):
    def test_the_script(self):
        source = (ROOT / "figures" / "make_figures.py").read_text(encoding="utf-8")
        self.assertEqual(source.count("\n    save(fig, "), 32)
        for piece in ("def batch_labels():", 'save(fig, "fig-17-3-batch-labels.png")',
                      "c17.partly(known, 1722, 1758, 2000, rows)", "first.index(max(whole)) == 6",
                      "second.index(min(whole)) == 10", "    unlinked_pairs()\n    batch_labels()\n    reliability()\n",
                      '"the four\\nexchanges\\nagree"'):
            self.assertIn(piece, source)

    def test_what_it_draws(self):
        # The figure tries the counts of B from b1 - s to b1 + s that stay within a batch of 25.
        for batches in (FIRST, SECOND):
            (_, _, b1), (_, _, b2) = batches
            for most in range(11):
                tried = [(b1 + j, b2 - j) for j in range(-most, most + 1) if 0 <= b1 + j <= 25 and 0 <= b2 - j <= 25]
                self.assertEqual(sorted(tried), relabeled(*batches, most))
        first = [allowed(*FIRST, most)[1][-1] for most in range(11)]
        second = [allowed(*SECOND, most)[1][0] for most in range(11)]
        self.assertEqual(first[6:], [first[6]] * 5)
        self.assertEqual(len(set(first[:7])), 7)
        self.assertEqual(len(set(second)), 11)
        self.assertEqual((first[0], first[1]), (second[1], second[0]))


def missing(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def every_share(rec, reference, limit):
    """Each share within the limit that a record allows at the class counts of the reference: a request without
    a duration is put on either side, and a class without requests anywhere from none within to all."""
    total = sum(len(xs) for xs in reference.values())
    choices = []
    for name, ref in reference.items():
        xs = rec.get(name) or []
        weight = Fraction(len(ref), total)
        if not xs:
            choices.append([Fraction(0), weight])
            continue
        sure = sum(1 for x in xs if not missing(x) and x <= limit)
        open_ = sum(1 for x in xs if missing(x))
        choices.append([weight * Fraction(sure + more, len(xs)) for more in range(open_ + 1)])
    return [sum(choice) for choice in itertools.product(*choices)]


class ADurationThatIsNotKnown(unittest.TestCase):
    def record(self, last):
        return {"fast": [100.0] * 90, "slow": [100.0] * 4 + [300.0] * 5 + [last]}

    def test_the_example_of_the_review(self):
        for nothing in (math.nan, None, float("nan")):
            rec = self.record(nothing)
            low, high = mix.within(rec, rec, 200)
            self.assertEqual((low, high), (Fraction(94, 100), Fraction(95, 100)))
            self.assertEqual(mix.verdict(low, high), "unresolved")
        met = mix.within(self.record(100.0), self.record(0.0), 200)
        missed = mix.within(self.record(300.0), self.record(0.0), 200)
        self.assertEqual((met, mix.verdict(*met)), ((Fraction(95, 100),) * 2, "met"))
        self.assertEqual((missed, mix.verdict(*missed)), ((Fraction(94, 100),) * 2, "missed"))
        # Version 48 compared: not a number is not at or below any limit, and the request counted as beyond it.
        self.assertFalse(math.nan <= 200)
        self.assertEqual(sum(x <= 200 for x in self.record(math.nan)["slow"]), 4)

    def test_where_it_stands_in_the_record(self):
        slow = [100.0] * 4 + [300.0] * 5 + [math.nan]
        for shift in range(10):
            rec = {"fast": [100.0] * 90, "slow": slow[shift:] + slow[:shift]}
            self.assertEqual(mix.within(rec, rec, 200), (Fraction(94, 100), Fraction(95, 100)))

    def test_the_rule_by_class(self):
        # Weight w, n requests of the class, k of them known to be within, u unknown: w k / n to w (k + u) / n.
        rec = {"a": [1, 2, None, 9, math.nan, 3], "b": [None], "c": [9, 9, 1, None]}
        ref = {"a": range(5), "b": range(3), "c": range(2)}
        low = Fraction(5, 10) * Fraction(3, 6) + Fraction(3, 10) * 0 + Fraction(2, 10) * Fraction(1, 4)
        high = Fraction(5, 10) * Fraction(5, 6) + Fraction(3, 10) * 1 + Fraction(2, 10) * Fraction(2, 4)
        self.assertEqual(mix.within(rec, ref, 5), (low, high))

    def test_a_class_without_records_is_all_unknown(self):
        ref = {"fast": range(900), "slow": range(100)}
        fast = [100.0] * 99 + [250.0]
        want = (Fraction(891, 1000), Fraction(991, 1000))
        for rec in ({"fast": fast}, {"fast": fast, "slow": []}, {"fast": fast, "slow": [None]},
                    {"fast": fast, "slow": [math.nan] * 7}, {"fast": fast, "slow": [None, math.nan]}):
            self.assertEqual(mix.within(rec, ref, 200), want)

    def test_a_known_exceedance_is_classified(self):
        ref = {"all": range(10)}
        for beyond in (math.inf, 201, 1e300):
            self.assertEqual(mix.within({"all": [100.0] * 9 + [beyond]}, ref, 200), (Fraction(9, 10),) * 2)
        self.assertEqual(mix.within({"all": [100.0] * 9 + [-math.inf]}, ref, 200), (Fraction(1),) * 2)
        self.assertEqual(mix.within({"all": [100.0] * 9 + [200]}, ref, 200), (Fraction(1),) * 2)

    def test_unknown(self):
        from decimal import Decimal
        for x in (None, math.nan, float("nan"), -math.nan, Decimal("NaN")):
            self.assertTrue(mix.unknown(x), x)
        for x in (0, 0.0, -1, 1e300, math.inf, -math.inf, Fraction(1, 3), Decimal("2.5"), True):
            self.assertFalse(mix.unknown(x), x)

    def test_random_records_against_every_completion(self):
        rng = random.Random(2348)
        for _ in range(400):
            names = ["a", "b", "c"][:rng.randint(1, 3)]
            ref = {name: range(rng.randint(0, 5)) for name in names}
            if not any(len(r) for r in ref.values()):
                ref[names[0]] = range(1)
            rec = {}
            for name in names:
                if rng.random() < 0.8:
                    rec[name] = [rng.choice([50.0, 150.0, 250.0, 200, None, math.nan]) for _ in range(rng.randint(0, 5))]
            shares = every_share(rec, ref, 200)
            self.assertEqual(mix.within(rec, ref, 200), (min(shares), max(shares)), (rec, ref))

    def test_records_of_numbers_are_as_before(self):
        rng = random.Random(2349)
        for _ in range(200):
            rec = {name: [rng.uniform(0, 400) for _ in range(rng.randint(1, 30))] for name in "ab"}
            ref = {name: range(rng.randint(1, 9)) for name in "ab"}
            total = sum(len(r) for r in ref.values())
            share = sum(Fraction(sum(x <= 200 for x in rec[name]), len(rec[name])) * len(ref[name]) / total for name in "ab")
            self.assertEqual(mix.within(rec, ref, 200), (share, share))

    def test_the_transcript(self):
        lines = expected("ch23_mix").splitlines()
        at = lines.index("90 fast requests and 10 slow, one")
        self.assertEqual(lines[at:at + 4], ["90 fast requests and 10 slow, one", "of the slow without a duration:",
                                            "  within 200 ms: 94 % to 95 %", "  p95 at most 200 ms: unresolved"])
        # Version 50 goes on with a bound for that request, eight lines, before the new class.
        self.assertEqual(lines[at + 12], "18 fast requests and 2 slow without")   # five lines of Version 58
        self.assertEqual(lines[at + 17], "after, with a new class, bulk:")
        self.assertEqual(printed(mix.one_unknown).splitlines()[:4], lines[at:at + 4])


def naive_quantile(pairs, q):
    """The smallest value with a share q of the weight at or below it, in whole numbers. A value of no weight
    is left out, which matters for q = 0 alone, where Version 51 no longer returns such a value."""
    pairs = sorted(p for p in pairs if p[1])
    total = sum(w for _, w in pairs)
    for i, (x, _) in enumerate(pairs):
        if sum(w for _, w in pairs[:i + 1]) * q.denominator >= q.numerator * total:
            return x


class NoPlaceInAnOrder(unittest.TestCase):
    def test_the_order_of_the_input_decided(self):
        # Sorting leaves a list with a value that is not a number in an order that depends on where it stood,
        # and the percentile of Version 48 was read from that list: 300 or 100 ms for one record.
        slow = [100.0] * 4 + [300.0] * 5 + [math.nan]
        turned = slow[-1:] + slow[:-1]
        self.assertNotEqual([x == x and x for x in sorted(slow)], [x == x and x for x in sorted(turned)])
        for xs in (slow, turned):
            rec = {"fast": [100.0] * 90, "slow": xs}
            with self.assertRaises(ValueError) as caught:
                mix.p95(mix.weighted(rec))
            self.assertEqual(str(caught.exception), "a value is no number")
            self.assertTrue(math.isnan(mix.mean(mix.weighted(rec))))             # the mean says so by itself

    def test_wherever_it_stands(self):
        values = [5.0, 1.0, 4.0, 2.0, 3.0]
        for place in range(6):
            xs = values[:place] + [math.nan] + values[place:]
            for q in (0, 0.5, 0.95, 1):
                with self.assertRaises(ValueError) as caught:
                    weighted_quantile([(x, 1) for x in xs], q)
                self.assertEqual(str(caught.exception), "a value is no number")

    def test_numbers_are_ordered_as_before(self):
        rng = random.Random(4823)
        for _ in range(300):
            pairs = [(rng.choice([rng.uniform(-5, 5), rng.randint(-3, 3), math.inf, -math.inf]), rng.randint(0, 4))
                     for _ in range(rng.randint(1, 12))]
            if not any(w for _, w in pairs):
                pairs.append((0.5, 1))
            q = Fraction(rng.randint(0, 20), 20)
            self.assertEqual(weighted_quantile(pairs, q), naive_quantile(pairs, q), (pairs, q))
            self.assertEqual(weighted_quantile(iter(pairs), q), naive_quantile(pairs, q))

    def test_weights_are_checked_as_before(self):
        for pairs in ([], [(1.0, 0)], [(1.0, 2), (2.0, -1)]):
            with self.assertRaises(ValueError) as caught:
                weighted_quantile(pairs, 0.5)
            self.assertEqual(str(caught.exception), "need weights >= 0, not all zero")


def most_by_trial(n, mean, limit, cap):
    """The most of n durations at or below the limit, each count tried from n down: k durations no greater
    than the limit and n - k greater than it and no greater than the cap have to add up to n times the mean.
    Returns the count and durations that show it. Durations are not negative."""
    total = n * mean
    for k in range(n, -1, -1):
        rest = n - k
        if rest == 0:
            if 0 <= total <= k * limit:
                return k, [mean] * n
            continue
        top = k * limit + rest * (cap if cap is not None else total + limit + 1)
        if not rest * limit < total <= top:
            continue
        below = min(Fraction(limit), Fraction(total - rest * limit, 2 * k)) if k else None   # leave room above
        above = Fraction(total - k * below, rest) if k else Fraction(total, rest)
        if cap is not None and above > cap:                                      # then the k have to rise
            above = Fraction(cap)
            below = Fraction(total - rest * above, k)
        return k, [below] * k + [above] * rest
    raise ValueError("no such durations")


class AMeanAndAnUpperLimit(unittest.TestCase):
    FAST = Fraction(891, 1000)                                                   # 99 of 100 fast, nine tenths

    def share(self, k):
        return self.FAST + Fraction(1, 10) * Fraction(k, 900)

    def test_what_the_slow_class_has_to_supply(self):
        need = next(k for k in range(901) if self.share(k) >= Fraction(95, 100))
        self.assertEqual(need, 531)
        self.assertEqual(self.share(531), Fraction(95, 100))

    def test_the_table_of_the_review(self):
        for cap, most, percent, verdict in ((800, 525, "94.9333", "missed"), (810, 531, "95.0000", "unresolved")):
            self.assertEqual(mix.most_within(900, 450, 200, cap), most)
            self.assertEqual(most_by_trial(900, 450, 200, cap)[0], most)
            top = self.share(most)
            self.assertEqual(f"{100 * float(top):.4f}", percent)
            self.assertEqual(mix.verdict(self.FAST, top), verdict)

    def test_both_are_attained(self):
        self.assertEqual(525 * 200 + 375 * 800, 900 * 450)
        last = Fraction(900 * 450 - 531 * 200, 369)
        self.assertEqual(last, Fraction(33200, 41))
        self.assertEqual(f"{float(last):.4f}", "809.7561")
        self.assertTrue(200 < last <= 810)
        for cap in (800, 810, last):
            k, xs = most_by_trial(900, 450, 200, cap)
            self.assertEqual((len(xs), sum(xs), sum(x <= 200 for x in xs)), (900, 900 * 450, k))
            self.assertTrue(all(0 <= x <= cap for x in xs))
        self.assertEqual(sum([450] * 900), 900 * 450)                            # and none within fits either cap

    def test_the_least_cap_that_leaves_it_open(self):
        least = Fraction(33200, 41)
        self.assertEqual(mix.most_within(900, 450, 200, least), 531)
        self.assertEqual(mix.most_within(900, 450, 200, least - Fraction(1, 10 ** 9)), 530)
        self.assertEqual(mix.most_within(900, 450, 200, 809.7560975609756), 530)  # the float lies just below it
        self.assertLess(Fraction("809.7560975609756"), least)
        self.assertEqual(mix.most_within(900, 450, 200, 809.76), 531)

    def test_the_mean_alone(self):
        self.assertEqual(mix.most_within(900, 450, 200), 899)
        self.assertEqual(most_by_trial(900, 450, 200, None)[0], 899)
        top = self.share(899)
        self.assertEqual((f"{100 * float(top):.4f}", f"{100 * float(top):.2f}", f"{100 * float(top):.1f}"),
                         ("99.0889", "99.09", "99.1"))
        self.assertEqual(f"{100 * float(self.share(900)):.1f}", "99.1")
        self.assertEqual(mix.verdict(self.FAST, top), "unresolved")

    def test_small_cases_by_trial(self):
        cases = 0
        for n in range(1, 9):
            for mean, limit, cap in itertools.product((1, 2, 3, 5, 8), (2, 4), (None, 3, 5, 6, 9)):
                if cap is not None and cap < mean:
                    with self.assertRaises(ValueError) as caught:
                        mix.most_within(n, mean, limit, cap)
                    self.assertEqual(str(caught.exception), "a mean above its cap")
                    continue
                k, xs = most_by_trial(n, mean, limit, cap)
                self.assertEqual(mix.most_within(n, mean, limit, cap), k, (n, mean, limit, cap))
                self.assertEqual((len(xs), sum(xs), sum(x <= limit for x in xs)), (n, n * mean, k))
                self.assertTrue(all(x >= 0 and (cap is None or x <= cap) for x in xs))
                cases += 1
        self.assertEqual(cases, 8 * (5 * 2 * 5 - sum(1 for m in (1, 2, 3, 5, 8) for c in (3, 5, 6, 9) if c < m) * 2))

    def test_the_numbers_are_read_as_they_print(self):
        self.assertEqual(mix.most_within(2, 0.2, 0.1, 0.3), 1)                   # one at 0.1 and one at 0.3
        self.assertEqual(most_by_trial(2, Fraction(2, 10), Fraction(1, 10), Fraction(3, 10))[0], 1)
        self.assertEqual(math.floor(2 * (0.3 - 0.2) / (0.3 - 0.1)), 0)           # in floats it is none

    def test_the_transcript(self):
        lines = expected("ch23_mix").splitlines()
        self.assertEqual(len(lines), 108)                                        # 84 in Version 49, 96 in 50, 100 in 51, 103 until 57
        at = lines.index("  within 200 ms: 89.1 % to 99.1 %")
        rows = ["  p95 at most 200 ms: unresolved", "  with the slow mean of 450 ms and"]
        for cap, name in ((None, "no other limit"), (800, "none above 800 ms"), (810, "none above 810 ms")):
            k = most_by_trial(900, 450, 200, cap)[0]
            top = self.share(k)
            rows += [f"  {name}:", f"    to {100 * float(top):.2f} %, {mix.verdict(self.FAST, top)}"]
        rows.append("  98 % known within, 2 % missing: met")
        # Version 50 has four lines on one recovered duration before the last of these.
        self.assertEqual(lines[at + 1:at + 9] + lines[at + 20:at + 21], rows)      # four more lines in 51 and three in 52
        self.assertEqual(rows[3], "    to 99.09 %, unresolved")
        self.assertEqual(rows[5], "    to 94.93 %, missed")
        self.assertEqual(rows[7], "    to 95.00 %, unresolved")


if __name__ == "__main__":
    unittest.main()
