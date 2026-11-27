"""Regression tests added after the review of Version 64.

The review found that checked() of Listing 18.1 made floats of a logit and of a temperature before it divided
them, so that two decimals below the least float had a quotient of 1 and a temperature of 1e-400 was refused as
not above zero; that fit_temperature() added its derivative from rounded products, lost the sign where a product
rounded to zero, and returned the lower end of its search for a minimum that lies inside it; that ece() added
the residuals of a bin in the order of the rows, so that one order gave 0 and another 1.4e-18; that components()
of Listing 23.2 made floats of its statistics before it subtracted and printed them, so that a change above half
a millisecond printed as +0 ms; and that prob_below() failed for the least number of degrees of freedom that
Result admits and was not monotone next to it.

Each test computes what it expects independently of the code under test: quotients and sums from fractions, a
fitted temperature by bisection in decimals of sixty digits or from its closed form, changes of a mean from
fractions with a rounding of their own, and Student's tail for few degrees of freedom from the series of the
incomplete beta function in decimals of sixty digits.
"""
import contextlib
import io
import itertools
import math
import random
import unittest
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from fractions import Fraction
from pathlib import Path

from examples import ch18_calibration as cal
from examples import ch23_mix as mix
from swmetro import Result, dist

ROOT = Path(__file__).resolve().parent.parent
D = Decimal
PI = D("3.14159265358979323846264338327950288419716939937510582097494")


def expected(name):
    return (ROOT / "expected" / f"{name}.txt").read_text(encoding="utf-8")


def printed(function, *args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        function(*args)
    return out.getvalue()


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def decimal_fit(data, lo, hi):
    """The temperature at which the derivative of the summed loss by 1/T is zero, by bisection on 1/T in decimals
    of sixty digits. Every logit is taken as the number that it is."""
    with localcontext() as ctx:
        ctx.prec = 60
        ms = [D(z) if y else -D(z) for z, y in data]

        def slope(beta):
            return sum(-m / (1 + (beta * m).exp()) for m in ms)
        a, b = 1 / D(hi), 1 / D(lo)
        assert slope(a) < 0 < slope(b), (slope(a), slope(b))
        for _ in range(220):
            mid = (a + b) / 2
            if slope(mid) > 0:
                b = mid
            else:
                a = mid
        return float(2 / (a + b))


def whole(x, plus=False):
    """A fraction in whole units, rounded half to even by the decimal module; with `plus`, signed, and a value
    that rounds to zero keeps its sign."""
    with localcontext() as ctx:
        ctx.prec = 80
        n = int((D(x.numerator) / D(x.denominator)).quantize(D(1), rounding=ROUND_HALF_EVEN))
    if not plus:
        return str(n)
    if n == 0:
        return "-0" if x < 0 else "+0"
    return f"{n:+d}"


def few_tail(t, dof):
    """Upper tail of Student's distribution for very few degrees of freedom, in decimals of sixty digits, from the
    series of the incomplete beta function. With a = dof / 2 and x = dof / (dof + t * t),
        I_x(a, 1/2) = x ** a * G * (1 + a * sum over k >= 1 of (1/2)_k * x ** k / (k! * (a + k))),
    where G = Gamma(a + 1/2) / (Gamma(a + 1) * Gamma(1/2)) = exp(-2 a ln 2 + pi ** 2 * a ** 2 / 6) up to terms in
    a ** 3. The tail is half of it. For x <= 1/2, that is, t * t >= dof; t is a float, a whole number or a
    fraction of any size."""
    with localcontext() as ctx:
        ctx.prec = 60
        ctx.Emin, ctx.Emax = -10 ** 9, 10 ** 9
        t = Fraction(t)
        t, v = D(t.numerator) / D(t.denominator), D(dof)
        a = v / 2
        x = v / (v + t * t)
        assert x <= D("0.5"), x
        series, term, k = D(0), D(1), 0
        while True:
            k += 1
            term *= (D(k) - D("0.5")) / k * x           # (1/2)_k * x ** k / k!
            series += term / (a + k)
            if term < D("1e-70"):
                break
        g = (-2 * a * D(2).ln() + PI * PI * a * a / 6).exp()
        return float((a * x.ln()).exp() * g * (1 + a * series) / 2)


class ALogitIsDividedAsTheNumberThatItIs(unittest.TestCase):
    def test_the_case_of_the_review(self):
        one = [(D("3e-324"), 1)]
        self.assertEqual(float(D("3e-324")), float(D("5e-324")))               # as floats they are one number
        self.assertEqual(cal.checked(one, D("5e-324")), [(0.6, 1)])            # Version 64: [(1.0, 1)]
        self.assertEqual(float(Fraction(D("3e-324")) / Fraction(D("5e-324"))), 0.6)
        # The three scores at a scaled logit of 0.6 with the label 1, from their definitions.
        p = sigmoid(0.6)
        self.assertAlmostEqual(cal.nll(one, D("5e-324")), -math.log(p), places=14)
        self.assertAlmostEqual(cal.brier(one, D("5e-324")), (1 - p) ** 2, places=14)
        self.assertAlmostEqual(cal.ece(one, D("5e-324")), 1 - p, places=14)
        for got, table in ((cal.nll(one, D("5e-324")), "0.437488"), (cal.brier(one, D("5e-324")), "0.125559"),
                           (cal.ece(one, D("5e-324")), "0.354344")):
            self.assertEqual(f"{got:.6f}", table)                              # the table of the review

    def test_a_temperature_below_the_least_float_is_above_zero(self):
        tiny = D("1e-400")
        self.assertEqual(float(tiny), 0.0)
        self.assertEqual(cal.checked([(tiny, 1)], tiny), [(1.0, 1)])           # Version 64: "need T > 0"
        self.assertEqual(cal.checked([(3 * tiny, 0), (-tiny, 1)], 2 * tiny), [(1.5, 0), (-0.5, 1)])
        self.assertEqual(cal.checked([(Fraction(1, 10 ** 400), 1)], Fraction(3, 10 ** 400)), [(1 / 3, 1)])
        for zero in (D("0E-500"), D("-1e-400"), Fraction(-1, 10 ** 400), 0, 0.0, -0.0, -5e-324):
            with self.assertRaisesRegex(ValueError, "need T > 0"):
                cal.checked([(1.0, 1)], zero)

    def test_a_quotient_is_rounded_once(self):
        rng = random.Random(6401)
        for _ in range(3000):
            z = rng.choice((1, -1)) * rng.uniform(0, 1) * 10.0 ** rng.randint(-320, 300)
            temp = rng.uniform(0.01, 1) * 10.0 ** rng.randint(-20, 20)
            want = Fraction(z) / Fraction(temp)
            if abs(want) >= Fraction(2) ** 1024:
                continue
            got = cal.checked([(z, 1)], temp)[0][0]
            self.assertEqual(got, float(want), (z, temp))                      # two floats: the machine's quotient
            for kind in (Fraction, D):                                         # and any other pair, the same number
                self.assertEqual(cal.checked([(kind(z), 1)], temp)[0][0], float(want))
                self.assertEqual(cal.checked([(z, 1)], kind(temp))[0][0], float(want))
        self.assertEqual(cal.checked([(1, 1)], 3), [(1 / 3, 1)])
        self.assertEqual(cal.checked([(10 ** 300, 0)], D("1e299")), [(10.0, 0)])

    def test_a_quotient_beyond_a_float_is_refused(self):
        for z, temp in ((1e300, 1e-300), (D("1e300"), D("1e-300")), (10 ** 300, Fraction(1, 10 ** 300)),
                        (-1e308, 0.5), (D("-1e308"), D("0.5")), (D("5e-324"), D("1e-700"))):
            with self.assertRaisesRegex(ValueError, "logit / T too large"):
                cal.checked([(z, 1)], temp)
            for score in (cal.nll, cal.brier, cal.ece, cal.bins_of):
                with self.assertRaisesRegex(ValueError, "logit / T too large"):
                    score([(z, 1)], temp)
        self.assertEqual(cal.checked([(D("1e-300"), 1)], D("1e300")), [(0.0, 1)])   # a quotient below a float is 0
        self.assertEqual(cal.checked([(1e308, 1)], 1e308), [(1.0, 1)])
        with self.assertRaisesRegex(ValueError, "logit is missing or too large"):
            cal.checked([(D("1e400"), 1)], 1)                                  # known() admits what a float can hold
        with self.assertRaisesRegex(ValueError, "T is missing or too large"):
            cal.checked([(D("1e400"), 1)], D("1e400"))

    def test_the_transcript_is_unchanged(self):
        self.assertEqual(printed(cal.main), expected("ch18_calibration"))


class TheSignOfTheSlope(unittest.TestCase):
    REVIEW = [(1000.0, 1), (5e-324, 0)]

    def test_the_case_of_the_review(self):
        # The slope is zero where 1000 * exp(-1000 b) is 5e-324 / 2, b being 1 / T, and 5e-324 is 2 ** -1074:
        # 1000 b = ln 1000 + 1075 ln 2.
        want = 1000 / (math.log(1000) + 1075 * math.log(2))
        self.assertEqual(f"{want:.14f}", "1.32971478159577")                   # the figure of the review
        got = cal.fit_temperature(self.REVIEW)
        self.assertAlmostEqual(got / want, 1, places=12)                       # Version 64: 0.2, the lower end
        self.assertAlmostEqual(got / decimal_fit(self.REVIEW, 0.2, 10), 1, places=12)
        ms = cal.margins(cal.checked(self.REVIEW))
        self.assertEqual(ms, [1000.0, -5e-324])
        self.assertEqual(-1000.0 * cal.sigmoid(-5000.0), 0.0)                  # the two products at the lower end,
        self.assertEqual(5e-324 * cal.sigmoid(5 * 5e-324), 0.0)                # both rounded to zero
        self.assertEqual(cal.lean(ms, 5.0), 1)                                 # the slope is above zero there,
        self.assertEqual(cal.lean(ms, 0.1), -1)                                # and below zero at the upper end
        self.assertEqual(cal.lean(ms, 1 / want * (1 + 1e-9)), 1)
        self.assertEqual(cal.lean(ms, 1 / want * (1 - 1e-9)), -1)

    def test_a_minimum_outside_the_interval_is_reported(self):
        with self.assertRaisesRegex(ValueError, "the minimum is below lo"):
            cal.fit_temperature(self.REVIEW, 2, 10)
        with self.assertRaisesRegex(ValueError, "the minimum is above hi"):
            cal.fit_temperature(self.REVIEW, 0.2, 1.0)
        self.assertAlmostEqual(cal.fit_temperature(self.REVIEW, 1.0, 2.0), cal.fit_temperature(self.REVIEW), places=12)

    def test_interior_minima_with_terms_that_underflow(self):
        rng = random.Random(6402)
        sizes = (5e-324, 1e-320, 1e-300, 1e-200, 1e-30, 1e-6, 0.5, 3.0, 40.0, 300.0, 1000.0, 20000.0)
        seen = {"inside": 0, "logs": 0}
        for _ in range(400):
            right = [rng.choice(sizes) * rng.uniform(0.5, 1) for _ in range(rng.randint(1, 3))]
            wrong = [rng.choice(sizes[:8]) * rng.uniform(0.5, 1) for _ in range(rng.randint(1, 2))]
            data = [(m, 1) for m in right] + [(m, 0) for m in wrong]
            if sum(map(Fraction, right)) <= sum(map(Fraction, wrong)):
                continue
            ms = cal.margins(cal.checked(data))
            try:
                want = decimal_fit(data, 0.2, 10)
            except AssertionError:                                              # the minimum is not inside
                with self.assertRaisesRegex(ValueError, "the minimum is"):
                    cal.fit_temperature(data)
                continue
            seen["inside"] += 1
            got = cal.fit_temperature(data)
            self.assertAlmostEqual(got / want, 1, places=9, msg=str(data))
            terms = [(m, cal.sigmoid(-m / got)) for m in ms]
            seen["logs"] += any(min(s, abs(m * s)) < 2.0 ** -1022 for m, s in terms)
        self.assertGreater(seen["inside"], 50, seen)
        self.assertGreater(seen["logs"], 20, seen)                              # terms below the least normal float

    def test_a_minimum_at_an_end_is_returned(self):
        # Two positive labels and one negative at a logit of ln 2: the loss is least where the stated probability
        # is 2/3, at T = 1. The float nearest ln 2 is the logit, and exp() of its negative is the float 0.5.
        x = math.log(2)
        if math.exp(-x) != 0.5:
            self.skipTest("exp(-ln 2) is not the float 0.5 here")
        data = [(x, 1), (x, 1), (x, 0)]
        self.assertEqual(cal.lean(cal.margins(data), 1.0), 0)
        self.assertEqual(cal.fit_temperature(data, 1.0, 10.0), 1.0)
        self.assertEqual(cal.fit_temperature(data, 0.2, 1.0), 1.0)
        self.assertAlmostEqual(cal.fit_temperature(data), 1.0, places=12)
        # Where every term is a normal float the sign is that of their exact sum, as until Version 64.
        rng = random.Random(6403)
        for _ in range(300):
            ms = [rng.uniform(-6, 6) for _ in range(rng.randint(1, 8))]
            beta = rng.uniform(0.05, 5)
            exact = sum(Fraction(-m * cal.sigmoid(-beta * m)) for m in ms)
            self.assertEqual(cal.lean(ms, beta), (exact > 0) - (exact < 0))
        self.assertEqual(cal.lean([0.0, 0.0], 1.0), 0)
        self.assertEqual(cal.lean([], 1.0), 0)

    def test_the_shape_of_the_loss_is_still_settled_first(self):
        for data, message in (([(0, 1), (0, 0)], "the loss is the same at every T"),
                              ([(200.0, 1), (-200.0, 0)], "the loss falls as T falls"),
                              ([(5e-324, 1), (5e-324, 0)], "the loss falls as T rises"),
                              ([(1000.0, 0), (5e-324, 1)], "the loss falls as T rises")):
            with self.assertRaisesRegex(ValueError, message):
                cal.fit_temperature(data)


class TheOrderOfTheCases(unittest.TestCase):
    def test_the_case_of_the_review(self):
        rows = [(0, 0), (40, 1), (0, 1)]
        small = math.exp(-40) / (1 + math.exp(-40))                             # what the case at 40 lacks to 1
        want = float(Fraction(small) / 3)                                       # the two halves cancel exactly
        self.assertEqual(want, 1.4161180850971963e-18)                          # the figure of the review
        self.assertEqual(0.5 - small - 0.5, 0.0)                                # added in this order it is lost
        for order in itertools.permutations(rows):
            self.assertEqual(cal.ece(order, bins=1), want, order)               # Version 64: 0.0 in two orders of six

    def test_no_order_changes_a_bin_or_a_score(self):
        rng = random.Random(6404)
        for _ in range(40):
            n = rng.randint(2, 60)
            data = [(rng.gauss(0, 4) * rng.choice((1, 1, 10)), rng.randint(0, 1)) for _ in range(n)]
            bins = rng.choice((1, 2, 5, 10))
            temp = rng.choice((1.0, 0.7, 3.0))
            first = cal.bins_of(data, temp, bins)
            score = cal.ece(data, temp, bins)
            # the sums of a bin, from fractions, by a count of their own
            want = [[0, 0, 0, 0] for _ in range(bins)]
            for z, y in data:
                p = cal.sigmoid(z / temp)
                b = min(int(Fraction(p) * bins), bins - 1)
                r = Fraction(-cal.sigmoid(-z / temp) if y else cal.sigmoid(z / temp))
                want[b] = [want[b][0] + 1, want[b][1] + Fraction(p), want[b][2] + y, want[b][3] + r]
            self.assertEqual(first, want)
            self.assertEqual(score, float(sum(abs(b[3]) for b in want) / n))
            for _ in range(5):
                rng.shuffle(data)
                self.assertEqual(cal.bins_of(data, temp, bins), first)
                self.assertEqual(cal.ece(data, temp, bins), score)

    def test_the_table_of_the_transcript(self):
        test = cal.make(5000, 2.0, random.Random(6405))
        for n, s, k, r in cal.bins_of(test, 1.0, 5):
            self.assertIsInstance(s, Fraction)
            self.assertIsInstance(r, Fraction)
            self.assertLess(abs(s - k - r), n * Fraction(2) ** -52)             # sum of p - y, within its rounding
        text = expected("ch18_calibration")
        self.assertIn(" 0.064     0.171", text)
        self.assertIn(" 0.932     0.819", text)


class AChangeIsRoundedOnce(unittest.TestCase):
    def lines(self, before, after, *stat):
        return printed(mix.components, before, after, *stat).splitlines()

    def test_the_case_of_the_review(self):
        after = D("100.500000000000001")
        for stat in ((), (mix.p95, "p95", "distributions")):
            name, what = (stat[1], stat[2]) if stat else ("mean", "means")
            got = self.lines({"a": [100]}, {"a": [after]}, *stat)
            self.assertEqual(got, [f"{name}, 100 -> 101 ms: +1 ms",              # Version 64: 100 -> 100 ms: +0 ms
                                   f"  class {what} first, via 101 ms:", "    within +1, mix +0",
                                   "  mix first, via 100 ms:", "    within +1, mix +0",
                                   "  average of the two orders:", "    within +1, mix +0"])
        got = self.lines({"a": [10 ** 18]}, {"a": [10 ** 18 + 1]})
        self.assertEqual(got[0], "mean, 1000000000000000000 -> 1000000000000000001 ms: +1 ms")
        self.assertEqual(float(10 ** 18), float(10 ** 18 + 1))                  # as floats the two are one number

    def test_a_tie_and_its_two_sides(self):
        for after, end, change in ((D("100.5"), "100", "+0"),                    # a true tie goes to the even number
                                   (D("100.500000000000000000001"), "101", "+1"),
                                   (D("100.499999999999999999999"), "100", "+0"),
                                   (D("101.5"), "102", "+2"), (D("102.5"), "102", "+2"),
                                   (D("99.5"), "100", "-0"), (D("98.5"), "98", "-2"),
                                   (D("99.499999999999999999"), "99", "-1"),
                                   (Fraction(201, 2), "100", "+0"), (100.5, "100", "+0"),
                                   (100.50000000000001, "101", "+1")):
            got = self.lines({"a": [100]}, {"a": [after]})
            self.assertEqual(got[0], f"mean, 100 -> {end} ms: {change} ms", after)
            self.assertEqual(got[2], f"    within {change}, mix +0", after)
            self.assertEqual(end, whole(Fraction(after)))
            self.assertEqual(change, whole(Fraction(after) - 100, True))
        # Each number is rounded from what it is: 101 -> 101.5 prints its end as 102 and its change as +0.
        self.assertEqual(self.lines({"a": [101]}, {"a": [D("101.5")]})[0], "mean, 101 -> 102 ms: +0 ms")

    def test_both_orders_and_their_average_from_fractions(self):
        rng = random.Random(6406)
        for _ in range(60):
            xs = {k: [D(rng.randint(1, 4000)) / rng.choice((2, 4, 8, 10, 1000)) for _ in range(4)] for k in "abcd"}
            na, nb = (rng.sample((1, 2, 3, 5), 2) for _ in range(2))
            before = {"a": xs["a"][:na[0]], "b": xs["b"][:na[1]]}
            after = {"a": xs["c"][:nb[0]], "b": xs["d"][:nb[1]]}

            def mean_at(rec, ref):
                total = sum(len(ref[k]) for k in ref)
                return sum(Fraction(len(ref[k]), total) * sum(map(Fraction, rec[k])) / len(rec[k]) for k in ref)
            start, end = mean_at(before, before), mean_at(after, after)
            first = mean_at(after, before)                                      # the durations after at the mix before
            second = mean_at(before, after)                                     # the durations before at the mix after
            one = (first - start, end - first)
            two = (end - second, second - start)
            mean = ((one[0] + two[0]) / 2, (one[1] + two[1]) / 2)
            want = [f"mean, {whole(start)} -> {whole(end)} ms: {whole(end - start, True)} ms",
                    f"  class means first, via {whole(first)} ms:",
                    f"    within {whole(one[0], True)}, mix {whole(one[1], True)}",
                    f"  mix first, via {whole(second)} ms:",
                    f"    within {whole(two[0], True)}, mix {whole(two[1], True)}",
                    "  average of the two orders:",
                    f"    within {whole(mean[0], True)}, mix {whole(mean[1], True)}"]
            self.assertEqual(self.lines(before, after), want)
            self.assertEqual(one[0] + one[1], end - start)                      # each order adds up to the change

    def test_exact_durations(self):
        pairs = [(1, 1), (0.1, 2), (D("0.1"), 3), (Fraction(1, 3), 4), (None, 5), ((">", 5), 6),
                 (math.inf, 7), (-math.inf, 8), (D("Infinity"), 9)]
        got = mix.exact(pairs)
        self.assertEqual([w for _, w in got], [w for _, w in pairs])
        self.assertEqual(got[:4], [(Fraction(1), 1), (Fraction(0.1), 2), (Fraction(1, 10), 3), (Fraction(1, 3), 4)])
        self.assertTrue(all(isinstance(x, Fraction) for x, _ in got[:4]))
        self.assertEqual([x for x, _ in got[4:]], [x for x, _ in pairs[4:]])
        nan = mix.exact([(math.nan, 1), (D("NaN"), 1)])
        self.assertTrue(all(x != x for x, _ in nan))
        self.assertEqual(mix.shown(Fraction(5, 2)), "2")
        self.assertEqual(mix.shown(Fraction(7, 2)), "4")
        self.assertEqual(mix.shown(Fraction(-1, 3), "+"), "-0")
        self.assertEqual(mix.shown(Fraction(0), "+"), "+0")
        self.assertEqual(mix.shown(math.inf), "inf")
        self.assertEqual(mix.shown(-math.inf, "+"), "-inf")

    def test_the_transcript_is_unchanged(self):
        self.assertEqual(printed(mix.main), expected("ch23_mix"))


class FewDegreesOfFreedom(unittest.TestCase):
    DOFS = (5e-324, 1e-320, 1e-300, 1e-200, 1e-100, 1e-30, 1e-15, 1e-12, 1e-10, 9.9e-10)

    def test_the_case_of_the_review(self):
        least = Result(0, 1, dof=5e-324)
        self.assertEqual(5e-324 / 2, 0.0)                                       # half the least float is no float
        self.assertEqual(least.prob_below(-1), 0.5)                             # Version 64: math domain error
        self.assertEqual(least.prob_below(1), 0.5)
        r = Result(0, 1, dof=1e-300)
        self.assertEqual(r.prob_below(-1), 0.5)                                 # Version 64: 0.4999999999999825
        self.assertEqual(r.prob_below(-1e200), 0.5)
        self.assertLessEqual(r.prob_below(-1e200), r.prob_below(-1))
        for dof in self.DOFS:                                                   # what both round to, from the series
            self.assertEqual(few_tail(1, dof), dist.t_sf(1.0, dof))
        self.assertEqual(few_tail(1, 1e-300), 0.5)

    def test_the_tail_from_the_series(self):
        worst = 0
        for dof in self.DOFS:
            for t in (1e-150, 1e-100, 1e-20, 1e-8, 1e-4, 0.5, 1, 30, 1e6, 1e20, 1e100, 1e150, 1e200, 1e300,
                      10 ** 400, Fraction(10 ** 500, 3)):
                if Fraction(t) ** 2 < Fraction(dof):
                    continue
                want, got = few_tail(t, dof), dist.t_sf(t, dof)
                worst = max(worst, abs(got - want) / want)
                self.assertLessEqual(got, 0.5)
        self.assertLess(worst, 2.3e-16, worst)                                  # within one unit of the last place
        # Next to 1e-9 degrees of freedom the limiting form hands over to the continued fraction.
        for dof in (9.999999e-10, 1.0000001e-9, 3e-9, 1e-8):
            for t in (1e-3, 1.0, 1e3, 1e9, 1e100, 1e150, 1e300, 10 ** 400):
                want, got = few_tail(t, dof), dist.t_sf(t, dof)
                self.assertLess(abs(got - want) / want, 2e-15, (dof, t))
        self.assertAlmostEqual(dist.t_sf(1.0, 1e-12), 0.5 * (1 - 1e-12 * (math.log(2) + math.log(1e-12) / -2)), delta=1e-15)

    def test_the_probability_does_not_fall_as_the_limit_rises(self):
        limits = sorted({s * m * 10.0 ** e for s in (-1, 1) for m in (1, 2.5, 7) for e in range(-320, 308, 2)} | {0.0})
        self.assertTrue(all(map(math.isfinite, limits)))
        limits = [-math.inf, -10 ** 400] + limits + [10 ** 400, math.inf]
        for dof in self.DOFS + (1e-9, 1e-6, 0.001, 0.5, 1, 4):
            r = Result(0, 1, dof=dof)
            ps = [r.prob_below(x) for x in limits]
            self.assertTrue(all(p <= q for p, q in zip(ps, ps[1:])), dof)
            self.assertEqual((ps[0], ps[-1]), (0.0, 1.0))
            self.assertEqual(r.prob_below(0), 0.5)
            self.assertTrue(all(p <= 0.5 for p, x in zip(ps, limits) if x < 0), dof)
            self.assertTrue(all(p >= 0.5 for p, x in zip(ps, limits) if x > 0), dof)
            edge = 1e153 * math.sqrt(dof)                                       # where the tail turns to logarithms
            near = [Fraction(s * edge * f) for s in (-1, 1) for f in (0.5, 1 - 1e-12, 1.0, 1 + 1e-12, 2)]
            ps = [r.prob_below(x) for x in sorted(near)]
            self.assertTrue(all(p <= q for p, q in zip(ps, ps[1:])), dof)

    def test_the_factor_of_the_incomplete_beta_function(self):
        # I_x(a, 1/2) for an a next to 0, where the logarithm of Gamma(a) is large: against the series.
        for a, x in ((5e-301, 1e-300), (1e-200, 0.25), (1e-20, 1e-3), (5e-10, 0.3), (5e-10, 1e-200)):
            dof = 2 * a
            t = math.sqrt(dof * (1 - x) / x)
            self.assertAlmostEqual(dist.betainc(a, 0.5, x) / (2 * few_tail(t, dof)), 1, delta=5e-15)
        self.assertAlmostEqual(dist.betainc(0.0, 0.5, 0.25), 1.0, delta=1e-15)  # no logarithm of Gamma(0) is taken
        self.assertEqual(dist.betainc(0.5, 0.0, 0.25), 0.0)
        # Elsewhere the function is what it was: the closed forms for one and two degrees of freedom.
        for t in (0.0, 1e-9, 0.3, 1.0, 7.5, 1e6, 1e12):
            self.assertAlmostEqual(dist.t_sf(t, 1), 0.5 - math.atan(t) / math.pi, delta=1e-15)
            self.assertAlmostEqual(dist.t_sf(t, 2), 0.5 - t / (2 * math.sqrt(2 + t * t)), delta=1e-15)

    def test_a_coverage_factor_is_still_refused(self):
        for dof in (5e-324, 1e-300):
            with self.assertRaisesRegex(ValueError, "quantile too large"):
                Result(0, 1, dof=dof).k(0.95)
            with self.assertRaisesRegex(ValueError, "quantile too large"):
                Result(0, 1, dof=dof).interval(0.95)
            self.assertEqual(Result(3, 0, dof=dof).prob_below(3), 1.0)
            self.assertEqual(Result(3, 0, dof=dof).prob_below(2), 0.0)


if __name__ == "__main__":
    unittest.main()
