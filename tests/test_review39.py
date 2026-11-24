"""Regression tests added after the review of Version 39.

The review found that lpu() of swmetro.propagate tested only the variance of the output that was asked for, so
that correlations which cannot hold together passed for one output and failed for another; that wait() of
Listing 15.2 formed the chance of a signal as 1 less a probability and lost it at wide limits; and that
reversed_of() of Listing 6.2 gave two outcomes to a selection of no pairs. It proposed an addition to Chapter 6,
the offsets that matched exchanges between two hosts allow, which is Listing 6.3 and Figure 6.2.

Each test computes what it expects independently of the code under test: a correlation matrix is judged by all
of its principal minors, in fractions; the tail of the normal distribution comes from a continued fraction in
50 digits; the offsets of an exchange are found by trying each one, and from exchanges built with a known
offset and known delays.
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

from examples import ch05_propagation as chapter5
from examples import ch06_exchange as exchange
from examples import ch06_ordering as ordering
from examples import ch15_run_length as run_length
from swmetro import propagate

ROOT = Path(__file__).resolve().parent.parent


def expected(name):
    return (ROOT / "expected" / f"{name}.txt").read_text(encoding="utf-8")


def printed(function, *args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        function(*args)
    return out.getvalue()


def refusal(function, *args, **named):
    try:
        function(*args, **named)
    except ValueError as error:
        return str(error)
    return None


def version_39_lpu(f, xs, us, corr=None):
    """lpu() as Version 39 printed it, with the root taken in floating point."""
    xs, us = list(xs), list(us)
    exact = [Fraction(x) for x in xs]
    cu = []
    for i, u in enumerate(us):
        hi, lo = list(exact), list(exact)
        hi[i] += Fraction(u)
        lo[i] -= Fraction(u)
        cu.append(Fraction(f(hi) - f(lo)) / 2)
    var = sum(c * c for c in cu)
    for (i, j), r in (corr or {}).items():
        var += 2 * Fraction(r) * cu[i] * cu[j]
    if var < 0:
        raise ValueError("negative variance")
    return math.sqrt(var)


def determinant(m):
    """By the sum over all permutations, in the arithmetic of the entries."""
    n, total = len(m), 0
    for perm in itertools.permutations(range(n)):
        sign = 1
        for i in range(n):
            for j in range(i + 1, n):
                if perm[i] > perm[j]:
                    sign = -sign
        term = Fraction(sign)
        for i in range(n):
            term *= m[i][perm[i]]
        total += term
    return total


def holds_together(m):
    """A symmetric matrix is a covariance matrix exactly when none of its principal minors is negative."""
    n = len(m)
    for size in range(1, n + 1):
        for keep in itertools.combinations(range(n), size):
            if determinant([[m[i][j] for j in keep] for i in keep]) < 0:
                return False
    return True


def matrix(corr, n):
    m = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    for (i, j), r in corr.items():
        m[i][j] = m[j][i] = Fraction(str(r))
    return m


TOTAL = sum


def contrast(x):
    return x[0] - x[1] + x[2]


class CorrelationsThatHoldTogether(unittest.TestCase):
    XS, US = [10, 10, 10], [1, 1, 1]
    ENTERED = {(0, 1): 0.9, (1, 2): 0.9}

    def test_the_example_of_the_review(self):
        self.assertEqual(f"{version_39_lpu(TOTAL, self.XS, self.US, self.ENTERED):.15f}", "2.569046515733026")
        self.assertEqual(refusal(version_39_lpu, contrast, self.XS, self.US, self.ENTERED), "negative variance")
        # The variance that the same entries give the contrast: 3 - 1.8 - 1.8, and nothing for the pair left out.
        r = Fraction(9, 10)
        self.assertEqual(3 - 2 * r - 2 * r + 2 * 0, Fraction(-3, 5))
        self.assertFalse(holds_together(matrix(self.ENTERED, 3)))
        for f in (TOTAL, contrast, lambda x: x[0], lambda x: 0 * x[0]):
            self.assertEqual(refusal(propagate.lpu, f, self.XS, self.US, self.ENTERED),
                             "the correlations cannot hold together")

    def test_the_third_correlation(self):
        # det = (1 - r)(r - 0.62): the correlations hold together for r from 0.62 to 1, the ends included.
        # The ends are the product of the two entries less and plus 0.19, the root of (1 - 0.81) squared.
        product, spread = Fraction(9, 10) ** 2, 1 - Fraction(9, 10) ** 2
        self.assertEqual((product - spread, product + spread), (Fraction(62, 100), 1))
        self.assertEqual((product, spread), (Fraction(81, 100), Fraction(19, 100)))
        for k in range(-100, 101):
            r = k / 100
            corr = {**self.ENTERED, (0, 2): r}
            m = matrix(corr, 3)
            self.assertEqual(determinant(m), (1 - Fraction(k, 100)) * (Fraction(k, 100) - Fraction(62, 100)))
            self.assertEqual(holds_together(m), 62 <= k <= 100, k)
            if holds_together(m):
                y, uc, cu = propagate.lpu(TOTAL, self.XS, self.US, corr)
                self.assertAlmostEqual(uc * uc, 6.6 + 2 * r, places=12)
            else:
                self.assertIsNotNone(refusal(propagate.lpu, TOTAL, self.XS, self.US, corr), k)

    def test_the_table_of_the_review(self):
        got = [propagate.lpu(TOTAL, self.XS, self.US, {**self.ENTERED, (0, 2): r})[1] for r in (0.62, 0.81, 1.0)]
        self.assertEqual([f"{u:.3f}" for u in got], ["2.800", "2.867", "2.933"])
        self.assertEqual(got[0], 2.8)                                            # 7.84 has a root
        want = [math.sqrt(6.6 + 2 * r) for r in (0.62, 0.81, 1.0)]
        for a, b in zip(got, want):
            self.assertAlmostEqual(a, b, places=12)
        self.assertEqual(f"{math.sqrt(6.6):.3f}", "2.569")                       # what the entries gave before

    def test_the_decimal_that_was_written(self):
        # As the binary numbers that the floats are, 0.9, 0.9 and 0.62 do not hold together: the determinant
        # is below zero by some 3e-17. A float is read as the decimal that Python prints for it.
        a, b = Fraction(0.9), Fraction(0.62)
        self.assertLess(1 + 2 * a * a * b - 2 * a * a - b * b, 0)
        self.assertGreater(float(1 + 2 * a * a * b - 2 * a * a - b * b), -1e-16)
        corr = {**self.ENTERED, (0, 2): 0.62}
        self.assertEqual(propagate.lpu(TOTAL, self.XS, self.US, corr)[1], 2.8)
        as_written = {pair: Fraction(str(r)) for pair, r in corr.items()}
        self.assertEqual(propagate.lpu(TOTAL, self.XS, self.US, as_written)[1], 2.8)
        as_decimals = {pair: decimal.Decimal(str(r)) for pair, r in corr.items()}
        self.assertEqual(propagate.lpu(TOTAL, self.XS, self.US, as_decimals)[1], 2.8)
        # A Fraction is taken as it stands, and the binary numbers are refused.
        as_binary = {pair: Fraction(r) for pair, r in corr.items()}
        self.assertEqual(refusal(propagate.lpu, TOTAL, self.XS, self.US, as_binary),
                         "the correlations cannot hold together")

    def test_against_the_principal_minors(self):
        rng = random.Random(39)
        kept = refused = 0
        for _ in range(1500):
            n = rng.choice((2, 3, 3, 4))
            pairs = list(itertools.combinations(range(n), 2))
            corr = {}
            for pair in pairs:
                if rng.random() < 0.8:
                    corr[pair] = rng.randrange(-20, 21) / 20                     # steps of 0.05, some pairs left out
            m = matrix(corr, n)
            good = holds_together(m)
            xs = [rng.randrange(1, 50) for _ in range(n)]
            us = [rng.randrange(1, 9) / 4 for _ in range(n)]
            weights = [rng.randrange(-3, 4) for _ in range(n)]

            def f(x):
                return sum(w * v for w, v in zip(weights, x))
            if not good:
                refused += 1
                self.assertEqual(refusal(propagate.lpu, f, xs, us, corr), "the correlations cannot hold together")
                continue
            kept += 1
            y, uc, cu = propagate.lpu(f, xs, us, corr)
            c = [Fraction(w) * Fraction(u) for w, u in zip(weights, us)]
            var = sum(m[i][j] * c[i] * c[j] for i in range(n) for j in range(n))
            self.assertGreaterEqual(var, 0)
            self.assertAlmostEqual(uc, math.sqrt(var), places=9)
            self.assertEqual(propagate.correlations(corr, n), m)
        self.assertGreater(min(kept, refused), 300)

    def test_on_the_edge(self):
        # A variance of zero is allowed. Two readings can share all of their error, and three as well.
        self.assertEqual(propagate.lpu(lambda x: x[1] - x[0], [5, 9], [2, 2], {(0, 1): 1.0})[1], 0.0)
        self.assertEqual(propagate.lpu(TOTAL, [0, 0], [1, 1], {(0, 1): -1.0})[1], 0.0)
        ones = {(0, 1): 1, (0, 2): 1, (1, 2): 1}
        self.assertEqual(propagate.lpu(TOTAL, self.XS, self.US, ones)[1], 3.0)
        self.assertEqual(propagate.lpu(contrast, self.XS, self.US, ones)[1], 1.0)
        # Two that go with a third in opposite ways go against each other: +1, -1 needs -1.
        for r, good in ((-1, True), (-0.99, False), (1, False), (0, False)):
            corr = {(0, 1): 1, (1, 2): -1, (0, 2): r}
            self.assertEqual(holds_together(matrix(corr, 3)), good)
            self.assertEqual(refusal(propagate.lpu, TOTAL, self.XS, self.US, corr) is None, good, r)
        for r in (1.01, -1.01, 1.5, -1.5):
            self.assertEqual(refusal(propagate.lpu, TOTAL, [0, 0], [1, 1], {(0, 1): r}),
                             "the correlations cannot hold together")
        self.assertIsNotNone(refusal(propagate.lpu, TOTAL, [0, 0], [1, 1], {(0, 1): float("nan")}))

    def test_how_a_pair_is_entered(self):
        once = propagate.lpu(TOTAL, [0, 0], [1, 1], {(0, 1): 0.5})[1]
        self.assertEqual(once, math.sqrt(3))
        both_ways = {(0, 1): 0.5, (1, 0): 0.5}
        self.assertEqual(version_39_lpu(TOTAL, [0, 0], [1, 1], both_ways), 2.0)  # the pair was counted twice
        self.assertEqual(propagate.lpu(TOTAL, [0, 0], [1, 1], both_ways)[1], once)
        self.assertEqual(refusal(propagate.lpu, TOTAL, [0, 0], [1, 1], {(0, 1): 0.5, (1, 0): 0.4}),
                         "two values for one pair")
        self.assertEqual(propagate.lpu(TOTAL, [0, 0], [1, 1], {(0, 0): 1, (0, 1): 0.5, (1, 1): 1.0})[1], once)
        self.assertEqual(refusal(propagate.lpu, TOTAL, [0, 0], [1, 1], {(0, 0): 0.5}), "an input with itself has 1")
        # An index that no input has. Version 39 let -1 stand for the last input and raised IndexError for 2.
        self.assertEqual(version_39_lpu(TOTAL, [0, 0], [1, 1], {(0, -1): 0.5}), math.sqrt(3))
        with self.assertRaises(IndexError):
            version_39_lpu(TOTAL, [0, 0], [1, 1], {(0, 2): 0.5})
        for pair in ((0, -1), (0, 2), (2, 0), (-1, -2)):
            self.assertEqual(refusal(propagate.lpu, TOTAL, [0, 0], [1, 1], {pair: 0.5}), "no such input", pair)

    def test_a_pair_left_out_is_zero(self):
        base = propagate.lpu(chapter5.speedup, [842.0, 655.0], [12.0, 15.0])
        for corr in ({}, None, {(0, 1): 0}, {(0, 1): 0.0}, {(1, 0): Fraction(0)}):
            self.assertEqual(propagate.lpu(chapter5.speedup, [842.0, 655.0], [12.0, 15.0], corr), base)
        m = propagate.correlations({(0, 2): 0.5}, 3)
        self.assertEqual(m, [[1, 0, Fraction(1, 2)], [0, 1, 0], [Fraction(1, 2), 0, 1]])
        self.assertEqual(propagate.correlations(None, 2), [[1, 0], [0, 1]])
        self.assertEqual(propagate.correlations({}, 0), [])

    def test_the_check_comes_first(self):
        calls = []

        def f(x):
            calls.append(x)
            return sum(x)
        self.assertIsNotNone(refusal(propagate.lpu, f, self.XS, self.US, self.ENTERED))
        self.assertEqual(calls, [])                                              # nothing was propagated

    def test_the_transcript(self):
        text = expected("ch05_propagation")
        lines = text.splitlines()
        self.assertEqual(len(lines), 40)                                         # 33 in Version 39
        self.assertEqual(lines[21:28], [
            "three stages of 10(1) ms, summed;",
            "r = 0.9 for the first two and the",
            "last two; first and third left out:",
            "  the correlations cannot hold together",
            f"first and third r = 0.62:  u_c = {math.sqrt(7.84):.3f} ms",
            f"first and third r = 0.81:  u_c = {math.sqrt(8.22):.3f} ms",
            f"first and third r = 1.00:  u_c = {math.sqrt(8.6):.3f} ms"])
        # The line for r = 0.6, with the decimal 6/10: the two contributions and their product, in fractions.
        a = (Fraction(842 + 12, 655) - Fraction(842 - 12, 655)) / 2
        b = (Fraction(842, 655 + 15) - Fraction(842, 655 - 15)) / 2
        u = math.sqrt(a * a + b * b + 2 * Fraction(6, 10) * a * b)
        self.assertEqual(lines[1], f"r = 0.6  S = {842 / 655:.4f}  u_c = {u:.4f}")
        self.assertEqual(lines[20], "the second:  u_c = 0.5774 ns")
        self.assertEqual(lines[28], "MC, independent normal, 200000 draws:")
        self.assertEqual(printed(chapter5.main), text)


def version_39_wait(lo, hi, mean, sigma=1.5):
    """wait() as Version 39 printed it."""
    n = statistics.NormalDist(mean, sigma)
    return 1 / (n.cdf(lo) + 1 - n.cdf(hi))


def upper_tail(x, digits=50):
    """P(Z > x) for a standard normal Z and x > 0, in a context of its own: the density times Mills' ratio,
    1 / (x + 1 / (x + 2 / (x + 3 / ...))), taken to 2000 terms from the bottom up."""
    ctx = decimal.Context(prec=digits)
    x = ctx.create_decimal(repr(float(x)))
    tail = x
    for k in range(2000, 0, -1):
        tail = ctx.add(x, ctx.divide(decimal.Decimal(k), tail))
    pi = ctx.create_decimal("3.14159265358979323846264338327950288419716939937510582097494")
    density = ctx.divide(ctx.exp(ctx.divide(ctx.multiply(x, x), decimal.Decimal(-2))),
                         ctx.sqrt(ctx.multiply(decimal.Decimal(2), pi)))
    return ctx.divide(density, tail)


def exact_wait(k):
    """The run length for limits at -k and +k standard deviations and no shift, as a float."""
    return float(1 / (2 * upper_tail(k)))


class TheTwoTails(unittest.TestCase):
    def test_the_table_of_the_review(self):
        self.assertEqual(f"{version_39_wait(-3, 3, 0, 1):.5f}", "370.39835")
        self.assertEqual(f"{version_39_wait(-8, 8, 0, 1):.4e}", "7.5060e+14")
        with self.assertRaises(ZeroDivisionError):
            version_39_wait(-9, 9, 0, 1)
        self.assertEqual(f"{100 * (1 - version_39_wait(-8, 8, 0, 1) / exact_wait(8)):.1f}", "6.6")
        got = [run_length.wait(-k, k, 0, 1) for k in (3, 8, 9)]
        self.assertEqual((f"{got[0]:.5f}", f"{got[1]:.4e}", f"{got[2]:.4e}"),
                         ("370.39835", "8.0373e+14", "4.4303e+18"))

    def test_the_continued_fraction(self):
        # The reference itself, where a closed check exists: P(Z > x) from erfc at moderate x.
        for x in (0.5, 1.0, 2.0, 3.0, 5.0):
            self.assertAlmostEqual(float(upper_tail(x)) / (0.5 * math.erfc(x / math.sqrt(2))), 1.0, places=12)

    def test_wide_limits(self):
        for k in (1, 2, 3, 4.5, 6, 8, 9, 12, 20, 30, 37):
            self.assertAlmostEqual(run_length.wait(-k, k, 0, 1) / exact_wait(k), 1.0, places=11, msg=k)
            self.assertAlmostEqual(run_length.wait(100 - 1.5 * k, 100 + 1.5 * k, 100) / exact_wait(k), 1.0,
                                   places=11, msg=k)
        # Beyond some 38 standard deviations the chance is below the smallest float, and the wait above the
        # largest: infinity is that wait as a float, and no division by zero.
        self.assertLess(upper_tail(40), decimal.Decimal("1e-340"))
        self.assertEqual(run_length.wait(-40, 40, 0, 1), math.inf)

    def test_limits_that_are_not_symmetric(self):
        for lo, hi, mean, sigma in ((94.41053, 105.20792, 101.5, 1.5), (95.5, 104.5, 103.0, 1.5), (-9.0, 2.0, 0.0, 1.0),
                                    (-2.0, 30.0, 0.0, 1.0), (-7.5, 8.25, 0.5, 1.0)):
            below, above = (mean - lo) / sigma, (hi - mean) / sigma
            p = upper_tail(below) + upper_tail(above)
            self.assertAlmostEqual(run_length.wait(lo, hi, mean, sigma) * float(p), 1.0, places=11)
        self.assertEqual(run_length.wait(-math.inf, 0.0, 0.0), 2.0)              # one limit: a coin
        self.assertEqual(run_length.wait(0.0, math.inf, 0.0), 2.0)

    def test_the_transcript_is_what_it_was(self):
        self.assertEqual(expected("ch15_run_length").splitlines()[2:4],
                         ["[95.50, 104.50]    370.4   43.9    6.3", "[94.41, 105.21]   2813.0  148.8   14.2"])
        self.assertEqual(printed(run_length.main), expected("ch15_run_length"))
        for d in (0, 1, 2):
            new = run_length.wait(95.5, 104.5, 100 + 1.5 * d)
            self.assertAlmostEqual(new / version_39_wait(95.5, 104.5, 100 + 1.5 * d), 1.0, places=9)


def version_39_reversed_of(n, p, shared):
    """reversed_of() as Version 39 printed it."""
    if shared:
        return [1 - p] + [0.0] * (n - 1) + [p]
    return [math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(n + 1)]


class NoPairs(unittest.TestCase):
    def test_the_example_of_the_review(self):
        self.assertEqual(version_39_reversed_of(0, 0.36, True), [0.64, 0.36])    # a chance for one of no pairs
        self.assertEqual(ordering.reversed_of(0, 0.36, True), [1.0])
        self.assertEqual(ordering.reversed_of(0, 0.36, False), [1.0])

    def test_by_counting_outcomes(self):
        # One offset: all the pairs are reversed, with the chance p, or none is. The number reversed is n or 0.
        for n in range(0, 7):
            for p in (0.0, 0.36, 0.5, 1.0):
                want = [0.0] * (n + 1)
                want[n] += p
                want[0] += 1 - p
                self.assertEqual(ordering.reversed_of(n, p, True), want, (n, p))
                anew = [0.0] * (n + 1)
                for outcome in itertools.product((0, 1), repeat=n):
                    chance = 1.0
                    for bit in outcome:
                        chance *= p if bit else 1 - p
                    anew[sum(outcome)] += chance
                got = ordering.reversed_of(n, p, False)
                self.assertEqual(len(got), n + 1)
                for a, b in zip(got, anew):
                    self.assertAlmostEqual(a, b, places=12)
        for n in (1, 2, 10):
            self.assertEqual(ordering.reversed_of(n, 0.36, True), version_39_reversed_of(n, 0.36, True))

    def test_the_transcript_is_what_it_was(self):
        self.assertEqual(len(expected("ch06_ordering").splitlines()), 12)
        self.assertEqual(printed(ordering.main), expected("ch06_ordering"))


FIRST, SECOND = (100, 98, 99, 105), (200, 196.5, 197.5, 204.5)


def fits(theta, a1, b2, b3, a4):
    """Whether an offset leaves both messages of an exchange a time on the way that is not below zero."""
    return b2 - a1 - theta >= 0 and a4 - b3 + theta >= 0


class MatchedExchanges(unittest.TestCase):
    GRID = [Fraction(k, 8) for k in range(-96, 65)]                              # offsets from -12 to 8 ms

    def test_the_table_of_the_review(self):
        self.assertEqual(exchange.EXCHANGES, (FIRST, SECOND))
        self.assertEqual(exchange.offset(*FIRST), (-6, -2))
        self.assertEqual(exchange.offset(*SECOND), (-7.0, -3.5))
        spans = [exchange.offset(*FIRST), exchange.offset(*SECOND)]
        self.assertEqual(exchange.both(spans), (-6, -3.5))
        self.assertEqual(exchange.gap(0, -3, spans[0]), (-1, 3))
        self.assertEqual(exchange.gap(0, -3, spans[1]), (0.5, 4.0))
        self.assertEqual(exchange.gap(0, -3, exchange.both(spans)), (0.5, 3))
        self.assertEqual(-3 - sum(spans[0]) / 2, 1.0)                            # with the middle of the first: +1 ms

    def test_every_offset_tried(self):
        for stamps in (FIRST, SECOND, (0, 5, 6, 7), (10, 3, 3, 10), (0, 0, 0, 0)):
            good = [t for t in self.GRID if fits(t, *stamps)]
            lo, hi = exchange.offset(*stamps)
            self.assertEqual((min(good), max(good)), (lo, hi), stamps)
            self.assertEqual(good, [t for t in self.GRID if lo <= t <= hi])      # an interval, with its ends
        good = [t for t in self.GRID if fits(t, *FIRST) and fits(t, *SECOND)]
        self.assertEqual((min(good), max(good)), (-6, Fraction(-7, 2)))
        # The true gap of two events stamped 0 on A and -3 on B, for every offset that both exchanges allow.
        gaps = [-3 - t for t in good]
        self.assertEqual((min(gaps), max(gaps)), (Fraction(1, 2), 3))
        self.assertTrue(all(g > 0 for g in gaps))                                # the event on B came second
        first_only = [-3 - t for t in self.GRID if fits(t, *FIRST)]
        self.assertTrue(min(first_only) < 0 < max(first_only))                   # either order

    def test_exchanges_with_a_known_offset(self):
        rng = random.Random(6)
        for _ in range(2000):
            theta = Fraction(rng.randrange(-4000, 4000), 16)
            out, back = (Fraction(rng.randrange(0, 800), 16) for _ in range(2))
            a1 = Fraction(rng.randrange(0, 10 ** 6), 16)
            b2 = a1 + out + theta                                                # B stamps the arrival on its clock
            b3 = b2 + Fraction(rng.randrange(0, 400), 16)
            a4 = b3 - theta + back
            lo, hi = exchange.offset(a1, b2, b3, a4)
            self.assertTrue(lo <= theta <= hi)
            self.assertEqual(hi - lo, out + back)                                # as wide as the two delays together
            self.assertEqual((lo + hi) / 2 - theta, (out - back) / 2)            # the middle is right for equal delays
            # The estimate and the delay in the form in which the time protocol states them.
            self.assertEqual((lo + hi) / 2, ((b2 - a1) + (b3 - a4)) / 2)
            self.assertEqual(hi - lo, (a4 - a1) - (b3 - b2))
            at_a, at_b = (Fraction(rng.randrange(0, 10 ** 6), 16) for _ in range(2))
            early, late = exchange.gap(at_a, at_b + theta, (lo, hi))
            self.assertTrue(early <= at_b - at_a <= late)
            self.assertEqual(late - early, hi - lo)

    def test_stamps_with_an_error(self):
        # Every stamp within e of the instant that it records: the exchange of a known offset again, with each
        # stamp moved by up to e, and the two ends reached when the stamps are moved against each other.
        rng = random.Random(60)
        for _ in range(2000):
            e = Fraction(rng.randrange(0, 40), 16)
            theta = Fraction(rng.randrange(-4000, 4000), 16)
            out, back = (Fraction(rng.randrange(0, 800), 16) for _ in range(2))
            a1 = Fraction(rng.randrange(0, 10 ** 6), 16)
            b2 = a1 + out + theta
            b3 = b2 + Fraction(rng.randrange(0, 400), 16)
            a4 = b3 - theta + back
            moved = [x + Fraction(rng.randrange(-16, 17), 16) * e for x in (a1, b2, b3, a4)]
            lo, hi = exchange.offset(*moved, err=e)
            self.assertTrue(lo <= theta <= hi)
            self.assertEqual(exchange.offset(a1, b2, b3, a4, err=e), (b3 - a4 - 2 * e, b2 - a1 + 2 * e))
            # From Version 41 the reply is not sent before the request has arrived: the lower end comes from
            # the later of the two stamps of B, where Version 40 took the reply alone (theta - back - 4 * e).
            self.assertEqual(exchange.offset(a1 - e, b2 + e, b3 - e, a4 + e, err=e)[0],
                             max(b2 + e, b3 - e) - a4 - 3 * e)
            self.assertEqual(exchange.offset(a1 + e, b2 - e, b3 + e, a4 - e, err=e), (theta - back, theta + out))
            at_a, at_b = (Fraction(rng.randrange(0, 10 ** 6), 16) for _ in range(2))
            stamps = [at_a + Fraction(rng.randrange(-16, 17), 16) * e, at_b + theta + Fraction(rng.randrange(-16, 17), 16) * e]
            early, late = exchange.gap(*stamps, (lo, hi), err=e)
            self.assertTrue(early <= at_b - at_a <= late)
        # The lines of the transcript: each stamp within half a millisecond.
        wide = [exchange.offset(*stamps, err=0.5) for stamps in (FIRST, SECOND)]
        self.assertEqual(wide, [(-7, -1), (-8, -2.5)])
        self.assertEqual(exchange.both(wide), (-7, -2.5))
        self.assertEqual(exchange.gap(0, -3, exchange.both(wide), err=0.5), (-1.5, 5))
        # By trying: stamps of the two events anywhere within 0.5 ms, offsets anywhere in -7 to -2.5 ms.
        tried = [(-3 + da) - t for da in (-1, 0, 1) for t in MatchedExchanges.GRID if -7 <= t <= Fraction(-5, 2)]
        self.assertEqual((min(tried), max(tried)), (Fraction(-3, 2), 5))

    def test_exchanges_that_do_not_agree(self):
        self.assertIsNone(exchange.both([(-6, -2), (-1.5, 3)]))
        self.assertEqual(exchange.both([(-6, -2), (-2, 3)]), (-2, -2))           # one offset is left
        self.assertEqual(exchange.both([(-6, -2)]), (-6, -2))
        # A reply stamped before the request can have arrived: the round trip is shorter than the stay.
        self.assertEqual(refusal(exchange.offset, 100, 98, 104, 105), "no offset fits the four events")
        self.assertEqual([t for t in self.GRID if fits(t, 100, 98, 104, 105)], [])

    def test_the_transcript(self):
        text = expected("ch06_exchange")
        lines = text.splitlines()
        self.assertEqual(len(lines), 39)                                         # 13 in Version 40, 18 in 41, 21 until 43, 29 in 44
        self.assertEqual(lines[10:13], ["stamps within 0.5 ms of their events:",
                                      f"{'offset, both':<12}{-7:>7.2f}{-2.5:>7.2f}",
                                      f"{'B after A by':<12}{-1.5:>+7.2f}{5:>+7.2f}"])
        rows = (("exchange 1", -6, -2), ("exchange 2", -7, -3.5), ("both", -6, -3.5))
        self.assertEqual(lines[2:5], [f"{name:<12}{lo:>7.2f}{hi:>7.2f}{(lo + hi) / 2:>8.2f}" for name, lo, hi in rows])
        self.assertEqual(lines[7:10], [f"{name:<12}{-3 - hi:>+7.2f}{-3 - lo:>+7.2f}{-3 - (lo + hi) / 2:>+8.2f}"
                                       for name, lo, hi in rows])
        self.assertEqual(printed(exchange.main), text)


class TheTwentyFirstFigure(unittest.TestCase):
    def test_the_script(self):
        source = (ROOT / "figures" / "make_figures.py").read_text(encoding="utf-8")
        for piece in ("def exchange_gap():", "c6x.offset(*stamps)", "c6x.gap(0, -3, spans[0])", "c6x.both(spans)",
                      'save(fig, "fig-06-2-exchange-gap.png")', "    reversed_pairs()\n    exchange_gap()\n"):
            self.assertIn(piece, source)
        self.assertEqual(source.count("\n    save(fig, "), 32)                   # twenty in Version 39

    def test_what_it_shows(self):
        ends = [(min(g), max(g)) for g in (
            [-3 - t for t in MatchedExchanges.GRID if fits(t, *FIRST)],
            [-3 - t for t in MatchedExchanges.GRID if fits(t, *SECOND)],
            [-3 - t for t in MatchedExchanges.GRID if fits(t, *FIRST) and fits(t, *SECOND)])]
        self.assertEqual(ends, [(-1, 3), (Fraction(1, 2), 4), (Fraction(1, 2), 3)])
        self.assertTrue(all(-1.4 < lo and hi < 4.4 for lo, hi in ends))          # within the axis of the figure


if __name__ == "__main__":
    unittest.main()
