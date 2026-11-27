"""Regression tests added after the review of Version 51.

The review found that with_one() of Listing 23.2 subtracted the recovered duration as it was given, so that a
float turned the mean of the others into a float and one request was lost from the count, and that it took a
duration beyond the cap. It asked that the rule of rounding behind Figure 23.3 be named, that the two recovered
durations of the text be tied to the exact mean and to distinct requests, what a duration of 720 ms settles as
that of one request and as the longest of all, which Figure 23.4 now draws, and that verdict() say what its
bounds are. Its examples for Chapter 21, a time to the first chunk and tokens counted on nested spans, are
arithmetic of the text and are kept here as such.

Each test computes what it expects independently of the code under test: counts of durations within a limit by
trying each count against the sums that it allows, with the durations that attain it, and the rounding of a
mean from the decimal module.
"""
import contextlib
import io
import itertools
import unittest
from decimal import ROUND_HALF_DOWN, ROUND_HALF_EVEN, Decimal
from fractions import Fraction
from pathlib import Path

from examples import ch23_mix as mix

ROOT = Path(__file__).resolve().parent.parent
FAST = Fraction(891, 1000)


def expected(name):
    return (ROOT / "expected" / f"{name}.txt").read_text(encoding="utf-8")


def printed(function, *args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        function(*args)
    return out.getvalue()


def share(k):
    return FAST + Fraction(1, 10) * Fraction(k, 900)


def allowed(k, rest, total, limit, cap):
    """Whether k of `rest` durations, none below zero and none above the cap, can be at or below the limit and
    the others above it, with this sum."""
    above = rest - k
    if above == 0:
        return 0 <= total <= rest * min(limit, cap)
    if cap <= limit or not total > above * limit:
        return False
    return total <= k * limit + above * cap


def counts(known, mean, n=900, limit=200, cap=810):
    """Every number of the n durations that can be within the limit when some are known and none is above the
    cap: those of the known that are within it, and each count of the others that their sum allows."""
    total = n * Fraction(mean) - sum(known)
    rest = n - len(known)
    if any(not 0 <= x <= cap for x in known):
        return []
    own = sum(x <= limit for x in known)
    return [own + k for k in range(rest + 1) if allowed(k, rest, total, limit, cap)]


def most(known, mean, n=900, limit=200, cap=810):
    found = counts(known, mean, n, limit, cap)
    return max(found) if found else None


class TheRecoveredReadingStaysExact(unittest.TestCase):
    def test_the_table_of_the_review(self):
        self.assertEqual(100 + 18 * 200 + 810, 20 * Fraction("225.5"))           # nineteen within, and attained
        self.assertGreater(Fraction("225.5"), 200)                               # all twenty cannot be
        self.assertEqual(most([100], Fraction("225.5"), n=20), 19)
        for x in (100, 100.0, Decimal("100.0"), "100", Fraction(100), Decimal(100)):
            for mean in (225.5, "225.5", Fraction(451, 2), Decimal("225.5")):
                self.assertEqual(mix.with_one(x, mean, 20), 19, (x, mean))

    def test_what_the_float_did(self):
        exact = (20 * Fraction("225.5") - 100) / 19
        mixed = (20 * Fraction("225.5") - 100.0) / 19                            # a float, by the rules of the language
        self.assertIsInstance(mixed, float)
        self.assertEqual(exact, Fraction(4410, 19))
        self.assertNotEqual(Fraction(str(mixed)), exact)                         # and no longer 4410/19 when read
        self.assertEqual(19 * (810 - exact) // 610, 18)
        self.assertEqual(19 * (810 - Fraction(str(mixed))) // 610, 17)           # one request fewer

    def test_a_duration_within_the_cap(self):
        self.assertIsNone(most([811], 450))
        for x in (811, 810.5, "1000", -1, Decimal("-0.001")):
            with self.assertRaises(ValueError) as caught:
                mix.with_one(x, 450)
            self.assertEqual(str(caught.exception), "x beyond the cap", x)
        self.assertEqual(mix.with_one(810, 450), most([810], 450))
        self.assertEqual(mix.with_one(0, 450), most([0], 450))

    def test_one_duration_is_its_own_mean(self):
        self.assertEqual((mix.with_one(150, 150, 1), mix.with_one(300.0, "300", 1), mix.with_one(200, 200.0, 1)), (1, 0, 1))
        for x, mean in ((150, 151), (300, 450), (0, 0.5)):
            with self.assertRaises(ValueError) as caught:
                mix.with_one(x, mean, 1)
            self.assertEqual(str(caught.exception), "x is not the mean")
        for n in (0, -3, 2.5, None):
            with self.assertRaises(ValueError):
                mix.with_one(100, 450, n)

    def test_small_cases_by_trial(self):
        fits = fails = 0
        halves = [Fraction(k, 2) for k in range(0, 1800, 150)] + [Fraction(1621, 2), 811, 900]
        for n, x, mean in itertools.product(range(1, 9), halves, (0, 100, 150.5, 200, 225.5, 450, 700, 810, 815)):
            want = most([x], Fraction(str(mean)), n=n)
            for given in ((x, mean), (float(x), mean), (str(float(x)), str(mean)), (Decimal(str(float(x))), mean)):
                if want is None:
                    with self.assertRaises(ValueError):
                        mix.with_one(*given, n)
                else:
                    self.assertEqual(mix.with_one(*given, n), want, (given, n))
            fits += want is not None
            fails += want is None
        self.assertEqual(fits + fails, 8 * 15 * 9)
        self.assertTrue(fits and fails)                                          # both kinds are among them

    def test_the_numbers_of_the_book_are_what_they_were(self):
        for x, want in ((0, 530), (100, 530), (110, 531), (200, 531), (201, 530), (700, 530), (720, 531), (810, 531)):
            self.assertEqual(mix.with_one(x, 450), want)
            self.assertEqual(mix.with_one(float(x), 450.0, 900), want)
            self.assertEqual(most([x], 450), want)
        for half, want in ((Fraction(5, 1000), 530), (Fraction(5, 100), 531), (Fraction(1, 2), 531)):
            self.assertEqual(mix.with_one(100, 450 - half), want)


class BoundsThatAreShares(unittest.TestCase):
    def test_what_the_docstring_says(self):
        doc = " ".join(mix.verdict.__doc__.split())
        self.assertIn("`low` and `high` are exact shares, the fractions that within() returns", doc)
        self.assertIn("a float among them is the binary number that it is", doc)

    def test_fractions_and_floats(self):
        exact = Fraction(19, 20)
        self.assertEqual(mix.verdict(exact, exact), "met")
        self.assertEqual(mix.verdict(Fraction(94, 100), exact), "unresolved")
        self.assertLess(Fraction(0.95), exact)                                   # the float lies below 19/20,
        self.assertEqual(mix.verdict(0.95, 0.95), "missed")                      # and is compared as that
        low, high = mix.within({"all": [100.0] * 19 + [300.0]}, {"all": range(20)}, 200)
        self.assertEqual((low, high, mix.verdict(low, high)), (exact, exact, "met"))

    def test_a_share_known_as_it_prints_is_an_interval(self):
        # 95 % to two digits is anything from 94.5 % to 95.5 %, and the verdict is open.
        low, high = Fraction("0.945"), Fraction("0.955")
        self.assertEqual(mix.verdict(low, high), "unresolved")
        self.assertEqual(mix.verdict(Fraction("0.9495"), Fraction("0.9505")), "unresolved")
        self.assertEqual(mix.verdict(Fraction("0.9505"), Fraction("0.9515")), "met")


class TwoRecoveredDurations(unittest.TestCase):
    def test_the_table_of_the_review(self):
        self.assertEqual(most([150, 150], 450), 530)                             # exact mean, two requests
        self.assertEqual(most([150, 150], Fraction("449.95")), 531)              # the mean printed as 450.0
        self.assertEqual(most([150, 160], 450), 531)
        self.assertEqual(most([150], 450), 531)                                  # two entries of one request
        self.assertEqual((mix.verdict(FAST, share(530)), mix.verdict(FAST, share(531))), ("missed", "unresolved"))

    def test_the_sums_behind_it(self):
        self.assertEqual(531 * 200 + 369 * 810 - 900 * 450, 90)
        total = 2 * 150 + 529 * 200 + 369 * 810
        self.assertEqual((total, 900 * 450 - total), (404990, 10))
        mean = Decimal(total) / 900
        self.assertEqual(mean.quantize(Decimal("0.1"), ROUND_HALF_EVEN), Decimal("450.0"))
        self.assertEqual(f"{float(mean):.6f}", "449.988889")
        self.assertEqual(2 * 150 + 528 * 200 + 370 * Fraction(29910, 37), 900 * 450)    # the same two, and a miss
        self.assertEqual(150 + 530 * 200 + 369 * Fraction(298850, 369), 900 * 450)      # one request, and 531
        self.assertLessEqual(Fraction(298850, 369), 810)
        self.assertEqual(150 + 160 + 529 * 200 + 369 * 810, 900 * 450)

    def test_with_one_serves_the_single_request(self):
        self.assertEqual(mix.with_one(150, 450), 531)
        self.assertEqual(mix.with_one(150, "449.95"), most([150], Fraction("449.95")))


class TheRuleOfRounding(unittest.TestCase):
    def test_the_ends_at_65_and_675_ms(self):
        for total in (65 + 530 * 200 + 369 * 810, 675 + 531 * 200 + 368 * 810):
            self.assertEqual(total, 404955)
            mean = Decimal(total) / 900
            self.assertEqual(mean, Decimal("449.95"))                            # the midpoint below 450.0
            self.assertEqual(mean.quantize(Decimal("0.1"), ROUND_HALF_EVEN), Decimal("450.0"))
            self.assertEqual(mean.quantize(Decimal("0.1"), ROUND_HALF_DOWN), Decimal("449.9"))
        self.assertEqual(most([65], Fraction("449.95")), 531)                    # so that 65 ms leaves it open
        self.assertEqual(most([675], Fraction("449.95")), 531)
        self.assertEqual(most([Fraction(129, 2)], Fraction("449.95")), 530)
        # With ties toward zero the mean is above 449.95 ms, however little, and both ends are settled.
        self.assertEqual(most([65], Fraction("449.95") + Fraction(1, 10 ** 9)), 530)
        self.assertEqual(most([675], Fraction("449.95") + Fraction(1, 10 ** 9)), 530)

    def test_the_other_midpoints(self):
        for digits, half, ends in ((2, "0.005", ("105.5", "715.5")), (0, "0.5", (None, "270"))):
            least = Decimal(450) - Decimal(half)
            self.assertEqual(least.quantize(Decimal(1).scaleb(-digits), ROUND_HALF_EVEN), Decimal(450))
            for end in ends:
                if end:
                    self.assertEqual(most([Fraction(end)], Fraction(str(least))), 531)
                    self.assertEqual(most([Fraction(end) - Fraction(1, 4)], Fraction(str(least))), 530)


class OneReadingOrTheLongest(unittest.TestCase):
    def test_one_request_of_720_ms(self):
        found = counts([720], 450)
        self.assertEqual(found, list(range(532)))                                # every number from none to 531
        self.assertEqual(mix.with_one(720, 450), 531)
        self.assertEqual(531 * 200 + 720 + 368 * 810, 900 * 450)
        self.assertGreater(Fraction(900 * 450 - 720, 899), 200)                  # the others equal: none within
        self.assertEqual(mix.verdict(FAST, share(531)), "unresolved")

    def test_no_request_above_720_ms(self):
        found = counts([720], 450, cap=720)                                      # one of them is the longest
        self.assertEqual(found, list(range(468)))
        self.assertEqual(counts([], 450, cap=720)[-1], 467)                      # and so it is without naming it
        self.assertEqual(mix.most_within(900, 450, 200, 720), 467)
        self.assertEqual(467 * 200 + 560 + 432 * 720, 900 * 450)
        self.assertEqual(900 * (720 - 450) // (720 - 200), 467)
        self.assertEqual(f"{100 * float(share(467)):.4f}", "94.2889")
        self.assertEqual(mix.verdict(FAST, share(467)), "missed")

    def test_the_transcript(self):
        lines = expected("ch23_mix").splitlines()
        self.assertEqual(len(lines), 126)                                        # 103 until Version 57, 108 until 65, 122 until 69
        at = lines.index("  720 ms, of one request or of all")
        rows = [f"    one: to {100 * float(share(counts([720], 450)[-1])):.2f} %, unresolved",
                f"    the longest: to {100 * float(share(counts([720], 450, cap=720)[-1])):.2f} %, missed"]
        self.assertEqual(lines[at + 1:at + 3], rows)
        self.assertEqual(rows, ["    one: to 95.00 %, unresolved", "    the longest: to 94.29 %, missed"])
        self.assertEqual(lines[at - 1], "    450 ms: to 95.00 %, unresolved")
        self.assertEqual(lines[at + 3], "  98 % known within, 2 % missing: met")
        self.assertEqual(printed(mix.main), expected("ch23_mix"))


class TheThirtySecondFigure(unittest.TestCase):
    def test_the_script(self):
        source = (ROOT / "figures" / "make_figures.py").read_text(encoding="utf-8")
        self.assertEqual(source.count("\n    save(fig, "), 32)
        for piece in ("def one_or_the_longest():", 'save(fig, "fig-23-4-one-or-the-longest.png")', "c23.with_one(720, 450)",
                      "c23.most_within(900, 450, 200, 720)", "== [531, 467]",
                      "    one_recovered()\n    mean_as_printed()\n    one_or_the_longest()\n"):
            self.assertIn(piece, source)

    def test_what_it_draws(self):
        self.assertEqual((counts([720], 450)[0], counts([720], 450)[-1]), (0, 531))
        self.assertEqual((counts([720], 450, cap=720)[0], counts([720], 450, cap=720)[-1]), (0, 467))
        need = next(k for k in range(901) if share(k) >= Fraction(95, 100))
        self.assertEqual(need, 531)


class TheExamplesOfChapter21(unittest.TestCase):
    def test_time_to_the_first_chunk(self):
        begins, retry, chunk, handled = 0, 120, 180, 195                         # ms on one clock of the client
        self.assertEqual(chunk - begins, 180)                                    # the quantity of the convention
        self.assertEqual(chunk - retry, 60)                                      # a timer started at the retry
        self.assertEqual(handled - begins, 195)                                  # one stopped in the application

    def test_tokens_on_nested_spans(self):
        calls, agent = (600, 400), 1000
        self.assertEqual(sum(calls), agent)
        self.assertEqual(sum(calls) + agent, 2000)                               # every span summed: twice the use


if __name__ == "__main__":
    unittest.main()
