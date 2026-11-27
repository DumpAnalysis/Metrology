"""Regression tests added after the review of Version 31.

The review found that the two-sided test of Chapter 17 doubled the tail for B ahead whichever model was ahead, so
that it changed when the two models exchanged their names and exceeded 1 at a tie; that two models with the same
answers divided zero by zero; that the transcript gave a difference of two shares in percent; and that the matched
requests of Chapter 16 were printed for numbers of requests in the window both times that cannot occur. It
proposed three exact tests for the questions of Chapter 20 that share a passage, which differ in what is exchanged.

Each test computes what it expects independently of the code under test: binomial sums written out, every outcome
at least as far from the middle, pairs of sets of requests enumerated, signs of differences enumerated, a count
formed from products of binomial coefficients, and a seeded simulation.
"""
import contextlib
import io
import itertools
import math
import random
import unittest
from collections import Counter
from fractions import Fraction
from pathlib import Path

from examples import ch16_silence as silence
from examples import ch17_accuracy as chapter17
from examples import ch20_error_bars as bars
from swmetro import stats

ROOT = Path(__file__).resolve().parent.parent


def expected(name):
    return (ROOT / "expected" / f"{name}.txt").read_text(encoding="utf-8")


def printed(function, *args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        function(*args)
    return out.getvalue()


def flat(text):
    return " ".join(text.split())


def at_most(k, n):
    """The chance of k or fewer of n, each with one half."""
    return Fraction(sum(math.comb(n, i) for i in range(k + 1)), 2 ** n)


def as_uneven(only_a, only_b):
    """The chance of a division of the pairs at least as uneven as this one, in either direction: every outcome
    at least as far from the middle."""
    n = only_a + only_b
    return Fraction(sum(math.comb(n, i) for i in range(n + 1) if abs(2 * i - n) >= abs(only_b - only_a)), 2 ** n)


def answers(only_a, only_b, both=0, neither=0):
    a = [True] * only_a + [False] * only_b + [True] * both + [False] * neither
    b = [False] * only_a + [True] * only_b + [True] * both + [False] * neither
    return a, b


def the_two(out):
    """The one-sided and the two-sided p of a printed comparison."""
    lines = out.splitlines()
    assert lines[-3] == "exact test for the pairs that differ:" and lines[-2].startswith("  B ahead, one-sided  p = ")
    assert lines[-1].startswith("  two-sided           p = ")
    return lines[-2].split("= ")[1], lines[-1].split("= ")[1]


class TheNamesOfTheTwoModels(unittest.TestCase):
    def test_the_table_of_the_review(self):
        table = ((112, 148, "0.014882", "0.029765", "0.029765"),
                 (148, 112, "0.989213", "1.978427", "0.029765"),
                 (2, 2, "0.687500", "1.375000", "1.000000"))
        for only_a, only_b, one, version_31, two in table:
            tail = at_most(only_a, only_a + only_b)
            self.assertEqual((f"{float(tail):.6f}", f"{float(2 * tail):.6f}"), (one, version_31))
            self.assertEqual(f"{float(as_uneven(only_a, only_b)):.6f}", two)
            self.assertEqual(the_two(printed(chapter17.compare, *answers(only_a, only_b, 1000, 740))), (one, two))

    def test_the_same_answers_in_opposite_columns(self):
        a, b = chapter17.two_models(2000, random.Random(17))
        first, second = printed(chapter17.compare, a, b), printed(chapter17.compare, b, a)
        self.assertEqual(first, printed(chapter17.compare))
        self.assertEqual((the_two(first), the_two(second)), (("0.014882", "0.029765"), ("0.989213", "0.029765")))
        # The two one-sided answers share the outcome that was observed and nothing else.
        self.assertEqual(at_most(112, 260) + at_most(148, 260) - 1, Fraction(math.comb(260, 112), 2 ** 260))
        self.assertIn("accuracy A 87.90 %, B 86.10 %\nonly A right 148, only B right 112\n"
                      "correlation of right answers +0.427\n"         # this line since Version 42
                      "difference B - A  -1.80 percentage points\n"
                      "u, unpaired   1.06 points, difference / u = -1.69\n"
                      "u, paired     0.81 points, difference / u = -2.24\n", second)

    def test_a_two_sided_p_for_every_division(self):
        # Never above 1, the same for either order, 1 at a tie and where no pair differs, and equal to the sum
        # over the outcomes at least as far from the middle.
        for only_a in range(0, 13):
            for only_b in range(0, 13):
                one, two = the_two(printed(chapter17.compare, *answers(only_a, only_b, 3, 2)))
                want = as_uneven(only_a, only_b)
                self.assertEqual((one, two), (f"{float(at_most(only_a, only_a + only_b)):.6f}", f"{float(want):.6f}"))
                self.assertLessEqual(want, 1)
                self.assertEqual(want, as_uneven(only_b, only_a))
                if only_a == only_b:
                    self.assertEqual(want, 1)

    def test_models_with_the_same_answers(self):
        a, _ = chapter17.two_models(2000, random.Random(17))
        out = printed(chapter17.compare, a, a)                   # Version 31: ZeroDivisionError
        self.assertEqual(out, "accuracy A 86.10 %, B 86.10 %\nonly A right 0, only B right 0\n"
                              "correlation of right answers +1.000\n"  # this line since Version 42
                              "difference B - A  0.00 percentage points\n"
                              "u, unpaired   1.09 points, difference / u = 0.00\n"
                              "u, paired     0.00 points, difference / u undefined\n"
                              "exact test for the pairs that differ:\n"
                              "  B ahead, one-sided  p = 1.000000\n  two-sided           p = 1.000000\n")
        # Both uncertainties are zero where both models are right on everything, and the paired one where
        # every item goes to the same model.
        out = printed(chapter17.compare, [True] * 8, [True] * 8)
        self.assertEqual(out.count("difference / u undefined"), 2)
        out = printed(chapter17.compare, [False] * 10, [True] * 10)
        self.assertIn("difference B - A  100.00 percentage points\n", out)
        self.assertIn("u, unpaired   0.00 points, difference / u undefined\nu, paired     0.00 points, difference / u undefined\n", out)
        self.assertEqual(the_two(out), (f"{1 / 1024:.6f}", f"{2 / 1024:.6f}"))                  # all ten for B
        self.assertEqual(the_two(printed(chapter17.compare, [True] * 10, [False] * 10)), (f"{1:.6f}", f"{2 / 1024:.6f}"))
        for a, b in (([], []), ([True, False], [True])):
            with self.assertRaisesRegex(ValueError, "one answer an item"):
                chapter17.compare(a, b)

    def test_percentage_points(self):
        lines = expected("ch17_accuracy").splitlines()
        self.assertEqual(lines[4:10], ["accuracy A 86.10 %, B 87.90 %", "only A right 112, only B right 148",
                                      "correlation of right answers +0.427",                # since Version 42
                                      "difference B - A  1.80 percentage points",
                                      "u, unpaired   1.06 points, difference / u = 1.69",
                                      "u, paired     0.81 points, difference / u = 2.24"])
        # 12 lines until Version 41, which ended here; a second pair of lists follows since Version 42.
        self.assertEqual((len(lines), lines[11:13]), (43, ["  B ahead, one-sided  p = 0.014882", "  two-sided           p = 0.029765"]))
        self.assertIn(printed(chapter17.compare), expected("ch17_accuracy"))
        # From 86.10 % to 87.90 % is 1.80 points, 2.09 % of the first, and 12.9 % of the errors of A.
        a, b = chapter17.two_models(2000, random.Random(17))
        self.assertEqual((sum(a), sum(b), f"{100 * (sum(b) - sum(a)) / sum(a):.2f}"), (1722, 1758, "2.09"))
        self.assertEqual(f"{100 * (sum(b) - sum(a)) / (2000 - sum(a)):.1f}", "12.9")
        self.assertEqual(f"{float(at_most(148, 260)):.3f}", "0.989")                # B ahead, with the names exchanged
        self.assertTrue(all(len(line) <= 50 for line in lines))

    def test_the_helper_says_which_tail(self):
        text = flat(stats.paired_exact.__doc__)
        self.assertNotIn("Twice that, where fewer <= more, is the two-sided one", text)          # Version 31
        self.assertIn("The two-sided test is twice the smaller of this and of the chance with the two counts exchanged, "
                      "and at most 1", text)
        self.assertIn("the exact test of McNemar", text)
        # The function is one tail, whichever count is the larger.
        for fewer, more in ((148, 112), (2, 2), (0, 0), (5, 0)):
            self.assertEqual(stats.paired_exact(fewer, more), at_most(fewer, fewer + more))


def rows_of(out):
    """The rows of a printed table of matched requests, and its last line."""
    lines = out.splitlines()
    assert lines[2] == "  those in it both times:" and lines[-1].startswith("  two independent records: ")
    return {int(line.split(":")[0]): line.split(": ")[1] for line in lines[3:-1]}, lines[-1].split(": ")[1]


class OverlapsThatCanOccur(unittest.TestCase):
    def test_the_two_tables_of_the_review(self):
        rows, last = rows_of(printed(silence.matched, 9, 6, 10))                  # Version 31: seven rows, from 0
        self.assertEqual(rows, {5: "18.75 %", 6: "12.50 %"})
        self.assertEqual([10 - 9 - 6 + both for both in rows], [0, 1])            # outside both times
        self.assertEqual([at_most(6 - both, 15 - 2 * both) for both in rows], [Fraction(3, 16), Fraction(1, 8)])
        rows, last = rows_of(printed(silence.matched, 3, 9, 60))                  # Version 31: ValueError at the fifth
        self.assertEqual(rows, {0: "98.07 %", 1: "98.93 %", 2: "99.61 %", 3: "100.00 %"})
        self.assertEqual([f"{100 * float(at_most(9 - both, 12 - 2 * both)):.2f} %" for both in range(4)], list(rows.values()))
        # A rise read by the test for a fall: the two tails share the outcome that was observed.
        fall, _ = rows_of(printed(silence.matched, 9, 3, 60))
        self.assertEqual(at_most(9, 12) + at_most(3, 12) - 1, Fraction(math.comb(12, 3), 2 ** 12))
        self.assertEqual(fall, {0: "7.30 %", 1: "5.47 %", 2: "3.52 %", 3: "1.56 %"})

    def test_every_pair_of_sets_of_requests(self):
        # For up to six requests: the numbers in the window both times are those that two sets of the two
        # sizes can share, and no others.
        for n in range(0, 7):
            for before in range(n + 1):
                for now in range(n + 1):
                    shared = {len(set(s) & set(t)) for s in itertools.combinations(range(n), before)
                              for t in itertools.combinations(range(n), now)}
                    rows, last = rows_of(printed(silence.matched, before, now, n))
                    self.assertEqual(set(rows), shared, (before, now, n))
                    self.assertEqual((min(shared), max(shared)), (max(0, before + now - n), min(before, now)))
                    for both, share in rows.items():
                        self.assertGreaterEqual(n - before - now + both, 0)
                        self.assertEqual(share, f"{100 * float(at_most(now - both, before + now - 2 * both)):.2f} %")
                    self.assertEqual(last, f"{100 * float(silence.depleted(before + now, n, n, now)):.2f} %")

    def test_what_is_refused(self):
        for args in ((9, 3, 8), (3, 9, 8), (61, 3, 60), (3, 61, 60)):             # Version 31 printed the first
            with self.assertRaisesRegex(ValueError, "counts are at most n"):
                silence.matched(*args)
        for args in ((9.5, 3, 60), (9, 2.5, 60), (9, 3, 60.5), (-1, 3, 60), (9, -3, 60), (9, 3, -60), (None, 3, 60)):
            out = io.StringIO()
            with contextlib.redirect_stdout(out), self.assertRaises(ValueError):
                silence.matched(*args)
            self.assertEqual(out.getvalue(), "", args)                            # refused before anything is printed
        self.assertEqual(printed(silence.matched, 9.0, 3.0, 60.0), printed(silence.matched))
        self.assertIn("They are at least before + now - n", flat(silence.matched.__doc__))

    def test_the_transcript_is_what_it_was(self):
        text = expected("ch16_silence")
        self.assertIn(printed(silence.matched), text)
        self.assertEqual(printed(silence.matched).count("\n"), 8)
        self.assertEqual(len(text.splitlines()), 193)
        self.assertTrue(text.endswith(printed(silence.search)))


def gain_or_more(counts, gain):
    """Of all assignments of signs to differences of the sizes in `counts`, a mapping from a size to the number
    of pairs with it, those whose sum is `gain` or more: (number, of how many). By the number of positive signs
    at each size."""
    sizes = sorted(counts)
    number = 0
    for positive in itertools.product(*(range(counts[size] + 1) for size in sizes)):
        total = sum(size * (2 * k - counts[size]) for size, k in zip(sizes, positive))
        if total >= gain:
            number += math.prod(math.comb(counts[size], k) for size, k in zip(sizes, positive))
    return number, 2 ** sum(counts.values())


def signs_written_out(diffs):
    """The same share, with every assignment of signs written out."""
    sizes = [abs(d) for d in diffs if d]
    ways = list(itertools.product((-1, 1), repeat=len(sizes)))
    return Fraction(sum(sum(s * m for s, m in zip(way, sizes)) >= sum(diffs) for way in ways), len(ways))


class LabelsExchangedByQuestionOrByPassage(unittest.TestCase):
    def setUp(self):
        self.rows = bars.evaluate(random.Random(20))
        net = Counter()
        for passage, a, b in self.rows:
            net[passage] += b - a
        self.net = [net[passage] for passage in range(bars.PASSAGES)]

    def test_the_counts_of_the_review(self):
        rows = self.rows
        right_a, right_b = sum(r[1] for r in rows), sum(r[2] for r in rows)
        self.assertEqual((len(rows), right_a, right_b, f"{100 * (right_b - right_a) / len(rows):.2f}"), (300, 226, 236, "3.33"))
        self.assertEqual((sum(a and not b for _, a, b in rows), sum(b and not a for _, a, b in rows)), (9, 19))
        self.assertEqual(sorted(Counter(self.net).items()), [(-2, 1), (-1, 6), (0, 40), (1, 8), (2, 5)])
        self.assertEqual((sum(self.net), sum(d > 0 for d in self.net), sum(d < 0 for d in self.net)), (10, 13, 7))
        self.assertEqual(sorted(bars.by_passage(rows)), sorted(self.net))

    def test_the_three_rows_of_the_review(self):
        # By question: 9 answers for A alone and 19 for B alone, each free to change sides.
        question = at_most(9, 28)
        # By passage: 14 passages differ by one answer and 6 by two; a gain of ten answers or more.
        number, of = gain_or_more({1: 14, 2: 6}, 10)
        self.assertEqual((number, of), (75_967, 1_048_576))
        passage = Fraction(number, of)
        # Passages won: 13 for B and 7 for A.
        won = at_most(7, 20)
        text = [f"{float(p):.5f}" for p in (question, 2 * question, passage, 2 * passage, won, 2 * won)]
        self.assertEqual(text, ["0.04358", "0.08716", "0.07245", "0.14490", "0.13159", "0.26318"])
        by_question = [b - a for _, a, b in self.rows]
        self.assertEqual(stats.exchange_exact(by_question), question)
        self.assertEqual(stats.exchange_exact(self.net), passage)
        self.assertEqual(stats.exchange_exact([(d > 0) - (d < 0) for d in self.net]), won)
        # The other tail of each, and with it the two-sided share of the listing.
        self.assertEqual(stats.exchange_exact([-d for d in self.net]), Fraction(gain_or_more({1: 14, 2: 6}, -10)[0], of))
        for diffs, one in ((by_question, question), (self.net, passage)):
            split, up, both = bars.tails(diffs)
            self.assertEqual((up, both), (float(one), float(2 * one)))
        self.assertEqual(bars.tails(self.net)[0], [7, 13, 40])
        self.assertEqual(bars.tails(by_question)[0], [9, 19, 272])

    def test_every_assignment_of_signs(self):
        rng = random.Random(3101)
        for _ in range(300):
            diffs = [rng.choice((-3, -2, -1, -1, 0, 0, 1, 1, 2, 3)) for _ in range(rng.randint(0, 11))]
            want = signs_written_out(diffs)
            self.assertEqual(stats.exchange_exact(diffs), want, diffs)
            self.assertEqual(stats.exchange_exact(diffs + [0, 0.0]), want)                # no difference, no sign
            self.assertEqual(stats.exchange_exact([float(d) for d in diffs]), want)
            rng.shuffle(diffs)
            self.assertEqual(stats.exchange_exact(iter(diffs)), want)
            # The two tails share the assignments that give the observed sum, and nothing else.
            counts = Counter(abs(d) for d in diffs if d)
            total = sum(diffs)
            same = gain_or_more(counts, total)[0] - gain_or_more(counts, total + 1)[0]
            other = stats.exchange_exact([-d for d in diffs])
            self.assertEqual(want + other - 1, Fraction(same, 2 ** sum(counts.values())))
            self.assertEqual(bars.tails(diffs)[2], float(min(1, 2 * min(want, other))))
        # The last: of the eight sums 8, 4, 4, 0, 0, -4, -4, -8, five are the observed 0 or more.
        self.assertEqual([stats.exchange_exact(d) for d in ([], [0], [1, -1], [3], [-3], [2, 2, -4])],
                         [1, 1, Fraction(3, 4), Fraction(1, 2), 1, Fraction(5, 8)])

    def test_sizes_of_one_are_the_paired_test(self):
        for against in range(0, 15):
            for for_it in range(0, 15):
                diffs = [-1] * against + [1] * for_it
                self.assertEqual(stats.exchange_exact(diffs), stats.paired_exact(against, for_it))
                self.assertEqual(stats.exchange_exact(diffs), at_most(against, against + for_it))
        self.assertEqual(stats.exchange_exact([-1] * 112 + [1] * 148), at_most(112, 260))
        self.assertIn("Sizes of 1 alone give paired_exact()", flat(stats.exchange_exact.__doc__))

    def test_a_passage_that_favors_one_model(self):
        # Benchmarks in which the labels mean nothing, but each passage favors one of the two models, chosen
        # by a fair coin, in all five of its questions. The labels can be exchanged for a passage as a whole,
        # and not for one question. At the 5 % level, one-sided, the test by passage holds its level, and the
        # test by question claims a gain more than twice as often as it may.
        rng = random.Random(3102)
        runs, by_question, by_passage = 1500, 0, 0
        for _ in range(runs):
            questions, passages = [], []
            for _ in range(60):
                favored = rng.random() < 0.5
                first, second = (1.6, 0.4) if favored else (0.4, 1.6)
                shift, net = rng.gauss(0, 1.0), 0
                for _ in range(5):
                    item = shift + rng.gauss(0, 0.7)
                    d = (item + rng.gauss(0, 0.5) < first) - (item + rng.gauss(0, 0.5) < second)
                    questions.append(d)
                    net += d
                passages.append(net)
            by_question += stats.exchange_exact(questions) <= Fraction(1, 20)
            by_passage += stats.exchange_exact(passages) <= Fraction(1, 20)
        margin = 4.5 * math.sqrt(0.05 * 0.95 / runs)
        self.assertLessEqual(by_passage / runs, 0.05 + margin)
        self.assertGreater(by_question / runs, 2 * 0.05 + margin)

    def test_what_is_refused(self):
        for diffs in ([0.5], [1, -2.5], [Fraction(1, 3)], [math.inf], [math.nan]):
            with self.assertRaises(ValueError):
                stats.exchange_exact(diffs)
        with self.assertRaises(TypeError):
            stats.exchange_exact([1, None])
        self.assertEqual(stats.exchange_exact([1, 1, 0]), Fraction(1, 4))
        # Until Version 67 the three differences above were True, True and False, which Python counts as
        # 1, 1 and 0. Since the review of Version 67 a truth value is no number in the package: native().
        with self.assertRaises(TypeError):
            stats.exchange_exact([True, True, False])

    def test_the_transcript(self):
        text = expected("ch20_error_bars")
        lines = text.splitlines()
        self.assertEqual(len(lines), 18)                          # 6 in Version 31
        self.assertEqual(lines[5:], ["B - A, unpaired u  0.059",
                                     "B - A in answers, by passage:", "  -2: 1  -1: 6  0: 40  +1: 8  +2: 5",
                                     "labels exchanged, exact tests:",
                                     "  by question", "    for A 9, for B 19, none 272", "    B ahead 0.04358, two-sided 0.08716",
                                     "  by passage, gain in answers", "    for A 7, for B 13, none 40", "    B ahead 0.07245, two-sided 0.14490",
                                     "  by passage, won or lost", "    for A 7, for B 13, none 40", "    B ahead 0.13159, two-sided 0.26318"])
        self.assertEqual(printed(bars.main), text)
        self.assertTrue(text.endswith(printed(bars.exchanged, self.rows)))
        self.assertTrue(all(len(line) <= 50 for line in lines))


if __name__ == "__main__":
    unittest.main()
