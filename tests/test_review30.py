"""Regression tests added after the review of Version 30.

The review found that a line fitted to readings with one float among them
came back with its intercept rounded and with the rounding added to the
uncertainty, so that the line returned was not the line of least squares
and its residual scatter was not the scatter returned; that the chance of
a silent window against a second trace needs the two traces to be
independent, and that its function used one of its counts unvalidated;
and that a float zero added to a time stamp of nineteen digits moved it.
It proposed the exact test for matched requests, and the same test for
the two models of Chapter 17.

Each test computes what it expects independently of the code under test:
a line fitted about the means in fractions and the residuals of the line
that comes back, sums formed as fractions, tables counted cell by cell,
changes of matched pairs written out one by one, and seeded simulations.
"""
import contextlib
import decimal
import io
import itertools
import math
import random
import statistics
import unittest
from fractions import Fraction
from pathlib import Path

from examples import ch16_silence as silence
from examples import ch17_accuracy as chapter17
from swmetro import stats
from swmetro.units import Q

ROOT = Path(__file__).resolve().parent.parent
EPOCH = 1_700_000_000_000_000_000                # ns from 1970 to 14 November 2023, 22:13:20 UTC
RUNNING = 5_000_000_000_000_000                  # the N of the review, ns


def expected(name):
    return (ROOT / "expected" / f"{name}.txt").read_text(encoding="utf-8")


def printed(function, *args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        function(*args)
    return out.getvalue()


def flat(text):
    return " ".join(text.split())


def exact_line(xs, ys):
    """Least squares about the two means, in fractions: a, b, the squares of u(a) and u(b), and the sum of the
    squared residuals."""
    fx, fy = [Fraction(x) for x in xs], [Fraction(y) for y in ys]
    n = len(fx)
    mx, my = sum(fx) / n, sum(fy) / n
    sxx = sum((x - mx) ** 2 for x in fx)
    b = sum((x - mx) * (y - my) for x, y in zip(fx, fy)) / sxx
    a = my - b * mx
    rss = sum((y - a - b * x) ** 2 for x, y in zip(fx, fy))
    var = rss / (n - 2)
    return a, b, var * (Fraction(1, n) + mx ** 2 / sxx), var / sxx, rss


def residuals_of(line, xs, ys):
    """The sum of the squared residuals of the readings from the line that was returned."""
    a, b = Fraction(line[0]), Fraction(line[1])
    return sum((Fraction(y) - a - b * Fraction(x)) ** 2 for x, y in zip(xs, ys))


def rooted(square):
    """The float nearest to the root of a fraction, by the decimal module with fifty digits."""
    with decimal.localcontext() as ctx:
        ctx.prec = 50
        return float((decimal.Decimal(square.numerator) / decimal.Decimal(square.denominator)).sqrt())


def pairs_of_the_review():
    return [(x, RUNNING + x + j) for x in (-1, 1) for j in (0, 1) for _ in range(50)]


class TheRoundingStaysOutOfTheUncertainty(unittest.TestCase):
    def test_the_table_of_the_review(self):
        xs, ys = map(list, zip(*pairs_of_the_review()))
        first = stats.linfit(xs, ys)
        xs[0] = -1.0                                         # the same number, stored as a float
        second = stats.linfit(xs, ys)
        self.assertEqual(first, second)
        self.assertEqual([type(v) for v in second], [Fraction, float, float, float, float])
        for line in (first, second):
            a, b, ua, ub, s = line
            self.assertEqual((a - RUNNING, b, f"{ua:.5f}", f"{s:.5f}"), (Fraction(1, 2), 1.0, "0.03553", "0.50252"))
            self.assertEqual(residuals_of(line, xs, ys), 50)
            self.assertEqual(s, rooted(Fraction(50, 198)))
            # For an interface that needs the rounded float: the float, and the signed correction that it leaves.
            self.assertEqual((float(a), a - Fraction(float(a))), (float(RUNNING), Fraction(1, 2)))
        # Version 30, for the second call: the nearest float, and its rounding in the uncertainty.
        old = (float(RUNNING), 1.0, rooted(Fraction(1, 4 * 198) + Fraction(1, 4)))
        self.assertEqual((f"{old[2]:.5f}", residuals_of(old, xs, ys)), ("0.50126", 100))
        self.assertEqual(f"{math.sqrt(100 / 198):.5f}", "0.71067")                   # and not the 0.50252 returned

    def test_however_the_readings_are_stored(self):
        xs, ys = zip(*pairs_of_the_review())
        want = stats.linfit(xs, ys)
        rng = random.Random(3001)
        for _ in range(60):
            fx = [float(x) if rng.random() < 0.3 else x for x in xs]
            fy = [float(y) if rng.random() < 0.3 else Fraction(y) if rng.random() < 0.2 else y for y in ys]
            self.assertEqual(stats.linfit(fx, fy), want)
        # Other readings that floats hold exactly, as whole numbers, as fractions and as floats.
        for _ in range(200):
            origin = rng.choice((0, 2 ** 44, 10 ** 15, RUNNING, -RUNNING))
            n = rng.randint(4, 30)
            xs = [rng.randint(-3, 3) for _ in range(n)]
            ys = [origin + 2 * x + rng.choice((0, 0, 1)) for x in xs]
            if len(set(xs)) < 2:
                continue
            want = stats.linfit(xs, ys)
            stored = stats.linfit([float(x) for x in xs], [float(y) if rng.random() < 0.5 else y for y in ys])
            self.assertEqual(stored, want, (xs, ys))

    def test_the_line_returned_is_the_line_of_least_squares(self):
        rng = random.Random(3002)
        kinds = {"stamps": 0, "sixteen digits": 0, "decimal floats": 0, "scatter": 0}
        for _ in range(400):
            kind = rng.choice(list(kinds))
            n = rng.randint(4, 40)
            if kind == "stamps":
                xs = [EPOCH + t for t in sorted(rng.sample(range(0, 4000), n))]
                ys = [x - EPOCH + rng.choice((0, 0, 1, -1)) for x in xs]
            elif kind == "sixteen digits":
                xs = [rng.choice((-1, 1, -1.0, 1.0, 2)) for _ in range(n)]
                ys = [RUNNING + int(x) + rng.choice((0, 1)) for x in xs]
            elif kind == "decimal floats":
                xs = [0.01 * k for k in range(n)]
                ys = [0.3 + 0.1 * x for x in xs]
            else:
                xs = [rng.uniform(0, 60) for _ in range(n)]
                ys = [2.5 + 0.75 * x + rng.gauss(0, 0.3) for x in xs]
            if len(set(xs)) < 2:
                continue
            a, b, ua2, ub2, rss = exact_line(xs, ys)
            got = stats.linfit(xs, ys)
            kinds[kind] += 1
            # The uncertainties and the scatter are those of the fit, each a root rounded once.
            self.assertEqual(got[2:], (rooted(ua2), rooted(ub2), rooted(rss / (n - 2))), (kind, xs, ys))
            # A reader who takes the two coefficients and forms the residuals finds the scatter returned.
            again = residuals_of(got, xs, ys)
            self.assertGreaterEqual(again, rss)
            self.assertLessEqual(again - rss, rss / 10 ** 6, (kind, xs, ys))
            for value, u, exact in ((got[0], got[2], a), (got[1], got[3], b)):
                self.assertLessEqual(1000 * abs(Fraction(value) - exact), Fraction(u))
                if type(value) is not float:
                    self.assertEqual(value, exact)
            # And nowhere is the line returned further from the fitted line than a thousandth of the standard
            # uncertainty of the fitted value there: at the origin, at the readings, between them, far away.
            mean_x = sum(Fraction(x) for x in xs) / n
            for x in (0, xs[0], xs[-1], mean_x, mean_x + 10 ** 9, -5 * mean_x, 10 ** 20):
                moved = (Fraction(got[0]) - a) + (Fraction(got[1]) - b) * Fraction(x)
                square = rss / (n - 2) / n + (Fraction(x) - mean_x) ** 2 * ub2
                self.assertLessEqual(10 ** 6 * moved ** 2, square, (kind, x))
        self.assertTrue(all(count > 60 for count in kinds.values()), kinds)

    def test_two_clocks_read_at_stamps(self):
        # Found with the check of the review, the residuals of the line that is returned. Twenty stamps of
        # one clock against another, a millisecond apart, with scatter: for each coefficient alone a float
        # shows a thousandth of its uncertainty, which was the rule of Version 27 to Version 30 for these
        # readings. The slope as a float times a stamp of nineteen digits is tens of nanoseconds off.
        rng = random.Random(3007)
        xs = [EPOCH + 10 ** 6 * i + rng.randint(-50, 50) for i in range(20)]
        ys = [x + 800 + rng.randint(-25, 25) for x in xs]
        a, b, ua2, ub2, rss = exact_line(xs, ys)
        for exact, square in ((a, ua2), (b, ub2)):
            self.assertGreater(rooted(square), 1000 * math.ulp(float(exact)))
        both_rounded = residuals_of((float(a), float(b)), xs, ys)
        self.assertGreater(both_rounded, 4 * rss)
        got = stats.linfit(xs, ys)
        self.assertLessEqual(residuals_of(got, xs, ys) - rss, rss / 10 ** 6)
        self.assertEqual((got[1], got[2:]), (b, (rooted(ua2), rooted(ub2), rooted(rss / 18))))
        self.assertIn(got[0], (a, float(a)))
        # With an origin subtracted from both, in whole numbers, the five are floats: the same slope, the
        # same uncertainty of it and the same scatter, and an offset near 800 ns that is known to 6 ns or so.
        near = stats.linfit([x - EPOCH for x in xs], [y - EPOCH for y in ys])
        self.assertEqual(([type(v) for v in near], near[1], near[3:]), ([float] * 5, float(b), got[3:]))
        self.assertTrue(780 < near[0] < 820 and 3 < near[2] < 9, near)

    def test_decimal_readings_and_an_origin_carried_beside(self):
        # The consequence that the last response avoided: ten decimal points, computed in floats, come back
        # as the fractions of the binary numbers that they are. float() gives the 1.0 and 2.0 of before.
        xs = [0.1 * k for k in range(10)]
        ys = [1 + 2 * x for x in xs]
        a, b, ua2, ub2, rss = exact_line(xs, ys)
        got = stats.linfit(xs, ys)
        self.assertEqual((got[0], got[1], type(got[0]), type(got[1])), (a, b, Fraction, Fraction))
        self.assertEqual((float(got[0]), float(got[1]), got[2], got[3]), (1.0, 2.0, rooted(ua2), rooted(ub2)))
        self.assertEqual(residuals_of(got, xs, ys), rss)
        # The origin, the offset and its uncertainty carried separately, as the review puts it: with N
        # subtracted in whole numbers the fit returns floats, and N stays a whole number beside them.
        xs, ys = zip(*pairs_of_the_review())
        near = stats.linfit(xs, [y - RUNNING for y in ys])
        self.assertEqual((near[:2], [type(v) for v in near], f"{near[2]:.5f}"), ((0.5, 1.0), [float] * 5, "0.03553"))
        self.assertEqual(RUNNING + Fraction(near[0]), stats.linfit(xs, ys)[0])


class AFloatZeroAddedToAStamp(unittest.TestCase):
    STAMP = EPOCH + 100

    def test_the_cases_of_the_review(self):
        reading = Q(self.STAMP, "ns")
        self.assertEqual([(self.STAMP + 0.0) - EPOCH, (self.STAMP + 1.0) - EPOCH], [0.0, 0.0])   # Version 30
        got = [(reading + Q(step, "ns")).value for step in (0, 0.0, 1.0, 1)]
        self.assertEqual((got, [type(v) for v in got]), ([self.STAMP, self.STAMP, self.STAMP + 1, self.STAMP + 1], [int] * 4))
        # The other way round as well: the stamp added to the float.
        self.assertEqual([(Q(step, "ns") + reading).value for step in (0.0, 1.0)], [self.STAMP, self.STAMP + 1])

    def test_fractional_increments(self):
        reading = Q(self.STAMP, "ns")
        self.assertEqual((reading + Q(0.5, "ns")).value, Fraction(2 * self.STAMP + 1, 2))
        self.assertEqual((reading + Q(1.0, "us")).value, self.STAMP + 1000)
        self.assertEqual((self.STAMP + 1.0 * 1000) - EPOCH, 1024.0)                  # Version 30: 76 ns too few
        self.assertEqual((reading + Q(0.25, "s")).value, self.STAMP + 250_000_000)
        self.assertEqual((Q(1, "s") + Q(0.5, "s")).value, Fraction(3, 2))
        tenth = (Q(1, "s") + Q(0.1, "s")).value                                      # the float 0.1 as it is stored
        self.assertEqual((tenth, type(tenth), float(tenth)), (1 + Fraction(0.1), Fraction, 1.1))

    def test_a_sum_with_an_exact_term_is_exact(self):
        rng = random.Random(3003)
        scale = {"s": 1, "ms": Fraction(1, 10 ** 3), "us": Fraction(1, 10 ** 6), "ns": Fraction(1, 10 ** 9)}
        for _ in range(1000):
            exact = rng.choice((rng.randint(-10 ** 6, 10 ** 6), EPOCH + rng.randint(0, 10 ** 9), Fraction(rng.randint(-99, 99), 7)))
            rounded = rng.choice((0.0, 1.0, -2.0, 0.5, 1e-3, rng.uniform(-5, 5), float(rng.randint(-10 ** 6, 10 ** 6))))
            first, second = rng.choice(list(scale)), rng.choice(list(scale))
            want = Fraction(exact) + Fraction(rounded) * scale[second] / scale[first]
            got = (Q(exact, first) + Q(rounded, second)).value
            self.assertEqual((got, type(got)), (want, int if want.denominator == 1 else Fraction), (exact, rounded, first, second))
            back = (Q(rounded, second) + Q(exact, first)).value
            self.assertEqual(back * scale[second], want * scale[first])

    def test_two_floats_and_what_is_refused(self):
        rng = random.Random(3004)
        for _ in range(500):
            x, y = rng.uniform(-9, 9), rng.uniform(-9, 9)
            got = (Q(x, "s") + Q(y, "ms")).value
            self.assertEqual((got, type(got)), (x + float(Fraction(y) / 1000), float))
        self.assertEqual((Q(1.0, "s") + Q(500.0, "ms")).value, 1.5)
        self.assertEqual((Q(self.STAMP, "ns") + Q(math.inf, "ns")).value, math.inf)
        self.assertTrue(math.isnan((Q(1, "s") + Q(math.nan, "s")).value))
        with self.assertRaisesRegex(TypeError, "information is not time"):
            Q(5, "ms") + Q(4, "KiB")
        self.assertEqual(expected("ch03_scales_units").splitlines()[-1], "5 ms + 4 KiB: information is not time")


def table_tail(k, n, m, here):
    """Of the divisions of n + m messages between two records that keep n and m, with k of the messages in a
    window, the share that puts `here` or fewer of the k into the first record. Cell by cell."""
    cells = [math.comb(k, i) * math.comb(n + m - k, n - i) if 0 <= n - i <= n + m - k else 0 for i in range(k + 1)]
    return Fraction(sum(cells[:here + 1]), sum(cells))


class TwoRecordsThatAreIndependent(unittest.TestCase):
    def test_validated_counts(self):
        want = Fraction(math.comb(60, 9), math.comb(120, 9))
        for args in ((9, 60, 60), (9, 60.0, 60), (9, 60, 60.0), (9.0, 60, 60), (9, 60, 60, 0.0)):
            self.assertEqual(silence.depleted(*args), want, args)                    # Version 30: TypeError for the third
        for k in (121, 500):
            with self.assertRaisesRegex(ValueError, "k is at most n \\+ m"):
                silence.depleted(k, 60, 60)                                          # Version 30: Fraction(0, 0)
        self.assertEqual((silence.depleted(120, 60, 60), silence.depleted(120, 60, 60, 60)), (0, 1))
        for args in ((-1, 60, 60), (2.5, 60, 60), (3, 60, -1), (3, 60.5, 60), (3, 60, 60, -1), (3, 60, 60, 0.5)):
            with self.assertRaises(ValueError):
                silence.depleted(*args)

    def test_the_pooled_count_and_the_tail(self):
        # k is the count of the window in both records together; `here` or fewer of them in the first.
        for k, n, m in ((6, 60, 60), (12, 60, 60), (3, 60, 90), (5, 4, 3), (7, 4, 3)):
            tails = [silence.depleted(k, n, m, here) for here in range(k + 1)]
            self.assertEqual(tails, [table_tail(k, n, m, here) for here in range(k + 1)])
            self.assertEqual((tails[-1], tails == sorted(tails)), (1, True))
        self.assertEqual(f"{100 * float(silence.depleted(6)):.4f}", "1.3706")
        self.assertEqual(f"{100 * float(silence.depleted(12, 60, 60, 3)):.4f}", "6.2724")
        # Seven messages, every division written out.
        for n, m, k, here in ((3, 4, 3, 1), (4, 3, 5, 2), (2, 5, 4, 0)):
            divisions = list(itertools.combinations(range(n + m), n))
            few = sum(sum(i < k for i in division) <= here for division in divisions)
            self.assertEqual(silence.depleted(k, n, m, here), Fraction(few, len(divisions)))

    def test_two_copies_of_one_stream(self):
        # The table of the review. Two collectors that keep every event of one stream: each record is the
        # stream, with its rate profile, and of six messages in the window between them each has three.
        rng = random.Random(3005)
        seen = set()
        for _ in range(20_000):
            stream = [rng.uniform(0, 300) for _ in range(60)]
            first, second = sorted(stream), sorted(stream)
            pooled = sum(210 <= t < 240 for t in first + second)
            if pooled == 6:
                seen.add(sum(210 <= t < 240 for t in first))
        self.assertEqual(seen, {3})
        # Two independent records with the same totals and the same six: none to six, and none in 1.37 %.
        steps = [silence.depleted(6, 60, 60, here) for here in range(7)]
        self.assertTrue(all(a < b for a, b in zip(steps, steps[1:])))
        self.assertIn("The two records are independent.", flat(silence.depleted.__doc__))
        self.assertIn("Two copies of one stream of events are not independent", flat(silence.depleted.__doc__))


def changes_at_most(fewer, more):
    """Of all the ways in which fewer + more pairs can each have changed one way or the other, the share with
    `fewer` or fewer the first way. Written out."""
    n = fewer + more
    ways = list(itertools.product((0, 1), repeat=n))
    return Fraction(sum(sum(way) <= fewer for way in ways), len(ways))


class TheSameRequestsInBothRuns(unittest.TestCase):
    def test_the_table_of_the_review(self):
        rows = []
        for both in range(4):
            reference_only, incident_only = 9 - both, 3 - both
            self.assertEqual(60 - both - reference_only - incident_only, 48 + both)  # outside in both runs
            rows.append(stats.paired_exact(incident_only, reference_only))
        self.assertEqual(rows, [Fraction(299, 4096), Fraction(7, 128), Fraction(9, 256), Fraction(1, 64)])
        self.assertEqual([f"{100 * float(p):.4f}" for p in rows], ["7.2998", "5.4688", "3.5156", "1.5625"])
        self.assertEqual(rows, [changes_at_most(3 - both, 9 - both) for both in range(4)])
        # The same counts from two independent records.
        self.assertEqual(f"{100 * float(table_tail(12, 60, 60, 3)):.4f}", "6.2724")

    def test_simulated_pairs(self):
        # Sixty requests, each in the window in either run with one chance, independently; kept are the runs
        # in which twelve requests changed their place.
        rng = random.Random(3006)
        kept = few = 0
        for _ in range(40_000):
            into = out = 0
            for _ in range(60):
                before, now = rng.random() < 0.1, rng.random() < 0.1
                into += now and not before
                out += before and not now
            if into + out == 12:
                kept += 1
                few += into <= 3
        want = float(changes_at_most(3, 9))
        self.assertGreater(kept, 3000)
        self.assertAlmostEqual(few / kept, want, delta=4.5 * math.sqrt(want * (1 - want) / kept))

    def test_the_two_models_of_chapter_17(self):
        one = stats.paired_exact(112, 148)
        self.assertEqual(one, Fraction(sum(math.comb(260, i) for i in range(113)), 2 ** 260))
        self.assertEqual((f"{float(one):.6f}", f"{float(2 * one):.6f}"), ("0.014882", "0.029765"))
        # Near the normal tail with a continuity correction, as a check of the size.
        near = statistics.NormalDist(130, math.sqrt(65)).cdf(112.5)
        self.assertAlmostEqual(float(one) / near, 1.0, delta=0.02)
        out = printed(chapter17.compare)
        # Since Version 32 the difference and its uncertainties are in percentage points, not in percent.
        self.assertTrue(out.endswith("u, paired     0.81 points, difference / u = 2.24\n"
                                     "exact test for the pairs that differ:\n"
                                     "  B ahead, one-sided  p = 0.014882\n  two-sided           p = 0.029765\n"))
        self.assertIn("only A right 112, only B right 148\n", out)
        text = expected("ch17_accuracy")
        self.assertIn(out, text)                             # its end until Version 41; a second pair of lists follows
        self.assertEqual(len(text.splitlines()), 43)         # 9 in Version 30, 12 until 41, 23 in 42, 27 in 43, 28 in 44

    def test_what_is_refused_and_small_cases(self):
        self.assertEqual([stats.paired_exact(0, 0), stats.paired_exact(0, 1), stats.paired_exact(1, 0), stats.paired_exact(2, 2)],
                         [1, Fraction(1, 2), 1, Fraction(11, 16)])
        for fewer, more in ((0, 5), (2, 3), (4, 1), (3, 3)):
            self.assertEqual(stats.paired_exact(fewer, more), changes_at_most(fewer, more))
        for args in ((-1, 3), (1.5, 3), (2, -1), (2, 0.5)):
            with self.assertRaises(ValueError):
                stats.paired_exact(*args)
        self.assertEqual(stats.paired_exact(3.0, 9), Fraction(299, 4096))

    def test_the_transcript(self):
        out = printed(silence.search)
        piece = ("    k = 9: 0.1414 %\n"
                 "  the same 60 requests twice, 9\n  in the window and then 3; by\n  those in it both times:\n"
                 "    0: 7.30 %\n    1: 5.47 %\n    2: 3.52 %\n    3: 1.56 %\n"
                 "  two independent records: 6.27 %\n"
                 "the same for requests of 10 s that\n")
        self.assertIn(piece, out)
        self.assertEqual(printed(silence.matched).count("\n"), 8)
        text = expected("ch16_silence")
        self.assertTrue(text.endswith(out))
        self.assertEqual(len(text.splitlines()), 193)        # 185 in Version 30


if __name__ == "__main__":
    unittest.main()
