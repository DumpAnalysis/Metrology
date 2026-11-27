"""Regression tests added after the review of Version 66.

The review found that components() of Listing 23.2 rounded the limit of a span to the nearest millisecond,
so that a mean of 199.95 ms or more printed as "200 ms or more" and shut out what the record allows; that
"no limit" stood both for a change that can be anything and for the difference of two infinite means, which
is not defined; that the change of the mix of a mean was reported as not determined where the class with the
unknown duration keeps its share and the contribution is exact; that quick() of Listing 18.1 raised
OverflowError while it added its allowances for logits near the end of the floats, before the decimal
calculation could be asked; that a finite decimal beyond the range of a float was taken for infinitely many
degrees of freedom, and bytes for a number; that the search for a quantile gave up at 2**498, below the
ceiling of 1e150 that it states; and that the normal tail above 1e25 degrees of freedom was said to be the
same to the last place. It also showed that the response to the review before it had bounded the error of
the tail at 1 000 001 degrees of freedom by the errors of its even neighbors, which bound nothing.

Each test computes what it expects independently of the code under test: limits from fractions, rounded to
their safe side by the decimal module; the parts of a change of a mean from the definition, for durations put
in the place of the open ones; a fitted temperature from its closed form in decimals; the Cauchy quantile from
its closed form; the normal tail from its series and its continued fraction in decimals; and Student's tail
for an odd number of degrees of freedom from its finite sum, or the rest of that series, in decimals.
"""
import contextlib
import io
import math
import random
import re
import unittest
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal, localcontext
from fractions import Fraction
from pathlib import Path

from examples import ch18_calibration as cal
from examples import ch23_mix as mix
from swmetro import Budget, Result, dist, welch_satterthwaite

ROOT = Path(__file__).resolve().parent.parent
D = Decimal
PI = D("3.14159265358979323846264338327950288419716939937510582097494")


def printed(function, *args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        function(*args)
    return out.getvalue()


def refusal(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except (ValueError, TypeError) as error:
        return f"{type(error).__name__}: {error}"
    return None


def dec(x):
    """A float, a whole number, a fraction or a decimal as a decimal, in the precision of the context."""
    x = Fraction(x)
    return D(x.numerator) / D(x.denominator)


def safe(x, up, plus=False):
    """A fraction as text to the hundredth, rounded upward or downward by the decimal module, without zeros at
    its end, and with its sign where `plus` asks for one."""
    with localcontext() as ctx:
        ctx.prec = 80
        q = dec(x).quantize(D("0.01"), rounding=ROUND_CEILING if up else ROUND_FLOOR)
    text = format(q, "f")
    text = text.rstrip("0").rstrip(".")
    if text in ("-0", ""):
        text = "0"
    return "+" + text if plus and not text.startswith("-") else text


def off(got, want):
    """The relative error of a float against a decimal."""
    with localcontext() as ctx:
        ctx.prec = 60
        return float(abs((D(got) - want) / want))


def normal_tail(t, digits=60, series=None):
    """P(Z > t) of the standard normal distribution for t >= 0, in decimals. Below t = 3 it is one half less
    phi(t) * (t + t**3 / 3 + t**5 / (3 * 5) + ...), a series of positive terms. From there on it is phi(t)
    divided by the continued fraction t + 1 / (t + 2 / (t + 3 / ...)), taken from its 4000th quotient back."""
    with localcontext() as ctx:
        ctx.prec = digits
        t = dec(t)
        density = (-t * t / 2).exp() / (2 * PI).sqrt()
        if series if series is not None else t < 3:
            term, total, k = t, D(0), 0
            while term > D(10) ** -(digits + 5):
                total += term
                k += 1
                term *= t * t / (2 * k + 1)
            return D("0.5") - density * total
        v = t
        for k in range(4000, 0, -1):
            v = t + k / v
        return density / v


def odd_tail(t, n, digits=60):
    """P(T > t) of Student's distribution for an odd number n of degrees of freedom, in decimals. With theta
    the angle whose tangent is t / sqrt(n), and w the square of its cosine, the central probability
    P(-t < T < t) is 2 / pi times theta + sin(theta) * cos(theta) * (sum over j < (n - 1) / 2 of d_j * w ** j),
    where d_0 = 1 and d_j = d_(j-1) * 2j / (2j + 1). The d_j are the coefficients of the series of
    arcsin(sqrt(w)) / sqrt(w * (1 - w)), so that the terms from (n - 1) / 2 on add up to what is left of pi / 2
    less theta. The tail is taken from those terms where they fall fast enough, and otherwise as one half less
    the finite sum, with theta from the series of the arctangent."""
    with localcontext() as ctx:
        ctx.prec = digits
        q = dec(t) ** 2
        w = n / (n + q)
        both = (q * n).sqrt() / (n + q)                                         # sin(theta) * cos(theta)
        term, low = D(1), D(0)
        for j in range(1, (n - 1) // 2 + 1):
            low += term
            term *= w * (2 * j) / (2 * j + 1)
        if (1 - w) * 20000 < 1:
            u = (q / n).sqrt()                                                  # the tangent, below 0.01 here
            theta, power, k = D(0), u, 0
            while abs(power) > D(10) ** -(digits + 5):
                theta += power / (2 * k + 1)
                power *= -u * u
                k += 1
            return D("0.5") - (theta + both * low) / PI
        high, j = D(0), (n - 1) // 2
        while term > high * (1 - w) * D(10) ** (5 - digits):
            high += term
            j += 1
            term *= w * (2 * j) / (2 * j + 1)
        return both * high / PI


def filled(rec, values):
    """The record with a duration from `values` in the place of every open one, in the order of the record."""
    values = iter(values)
    return {name: [Fraction(next(values)) if mix.is_open(x) else Fraction(x) for x in xs]
            for name, xs in rec.items()}


def parts(before, after):
    """The change of the mean of all requests as a change within the classes and a change of the mix, from the
    definition, for complete records: (via, within, mix) with the class means changed first and with the mix
    changed first, and (within, mix) of their average."""
    def total(rec):
        return sum(len(xs) for xs in rec.values())

    def at(durations, weights):
        """The mean of the class means of `durations` at the class counts of `weights`."""
        return sum(Fraction(len(weights[c]), total(weights)) * Fraction(sum(durations[c]), len(durations[c]))
                   for c in weights if weights[c])
    start, end = at(before, before), at(after, after)
    first = at(after, before)
    second = at(before, after)
    one = (first, first - start, end - first)
    two = (second, end - second, second - start)
    return one, two, ((one[1] + two[1]) / 2, (one[2] + two[2]) / 2)


class ALimitThatIsPrinted(unittest.TestCase):
    A = {"web": [200] * 20}
    B = {"web": [200] * 18 + [399, None]}

    def test_the_record_of_the_review(self):
        # With the unknown duration x, the mean after is (3999 + x) / 20: at x = 0 it is 199.95 ms.
        least = Fraction(3999 + 0, 20)
        self.assertEqual(least, Fraction("199.95"))
        self.assertEqual(least - 200, Fraction("-0.05"))
        self.assertEqual(round(least), 200)                                     # what Version 66 printed as a limit
        self.assertEqual(printed(mix.components, self.A, self.B).splitlines(),
                         ["mean, 200 -> 199.95 ms or more: -0.05 ms or more",
                          "  open after: 1 of 20 durations",
                          "  the mix is the same in both periods:",
                          "    within -0.05 or more, mix +0"])
        self.assertEqual(printed(mix.components, self.B, self.A).splitlines(),
                         ["mean, 199.95 or more -> 200 ms: +0.05 ms or less",
                          "  open before: 1 of 20 durations",
                          "  the mix is the same in both periods:",
                          "    within +0.05 or less, mix +0"])

    def test_no_printed_limit_shuts_out_what_the_record_allows(self):
        rng = random.Random(6601)
        number = r"([-+]?\d+(?:\.\d+)?)"
        for _ in range(300):
            n = rng.randint(2, 40)
            first = [rng.randint(0, 400) for _ in range(n)]
            second = [rng.randint(0, 400) for _ in range(n - 1)]
            bound = rng.choice((None, (">=", rng.randint(0, 300)), (">", rng.randint(0, 300))))
            lowest = 0 if bound is None else bound[1]
            start = Fraction(sum(first), n)
            least = Fraction(sum(second) + lowest, n)
            head = printed(mix.components, {"web": first}, {"web": second + [bound]}).splitlines()[0]
            found = re.fullmatch(rf"mean, \d+ -> {number} ms or more: {number} ms or more", head)
            self.assertIsNotNone(found, head)
            self.assertEqual(found.group(1), safe(least, False), head)
            self.assertEqual(found.group(2), safe(least - start, False, True), head)
            self.assertLessEqual(Fraction(found.group(1)), least)
            self.assertLessEqual(Fraction(found.group(2)), least - start)
            # The other order of the periods: the change has an upper limit, which is rounded upward.
            head = printed(mix.components, {"web": second + [bound]}, {"web": first}).splitlines()[0]
            found = re.fullmatch(rf"mean, {number} or more -> \d+ ms: {number} ms or less", head)
            self.assertIsNotNone(found, head)
            self.assertEqual(found.group(2), safe(start - least, True, True), head)
            self.assertGreaterEqual(Fraction(found.group(2)), start - least)

    def test_edges(self):
        for x, down, up in (("199.95", "199.95", "199.95"), ("199.9951", "199.99", "200"), ("95", "95", "95"),
                            ("1900", "1900", "1900"), ("-0.001", "-0.01", "0"), ("0.001", "0", "0.01"),
                            ("-4.7619", "-4.77", "-4.76"), ("100.5", "100.5", "100.5")):
            self.assertEqual(mix.edge(Fraction(x), 0), down, x)
            self.assertEqual(mix.edge(Fraction(x), 1), up, x)
            self.assertEqual(safe(Fraction(x), False), down)
            self.assertEqual(safe(Fraction(x), True), up)
        self.assertEqual(mix.edge(Fraction("-0.001"), 1, "+"), "+0")
        self.assertEqual(mix.edge(Fraction("0.05"), 1, "+"), "+0.05")
        self.assertEqual(mix.edge(Fraction("-0.05"), 0, "+"), "-0.05")
        rng = random.Random(6602)
        for _ in range(500):
            x = Fraction(rng.randint(-10 ** 6, 10 ** 6), rng.randint(1, 10 ** 4))
            low, top = Fraction(mix.edge(x, 0)), Fraction(mix.edge(x, 1))
            self.assertLessEqual(low, x)
            self.assertLessEqual(x, top)
            self.assertLess(top - low, Fraction(2, 100))
            self.assertEqual(low == top, (100 * x).denominator == 1)            # exact where two places hold it

    def test_one_number_is_rounded_as_before(self):
        # A statistic that the record settles is no limit: it is rounded half to even, once.
        self.assertEqual(mix.shown(Fraction(3999, 20)), "200")
        self.assertEqual(mix.spanned((Fraction(3999, 20), Fraction(3999, 20))), "200")
        self.assertEqual(mix.spanned((Fraction(3999, 20), math.inf)), "199.95 or more")
        self.assertEqual(mix.spanned((-math.inf, Fraction(3999, 20)), unit=" ms"), "199.95 ms or less")
        self.assertEqual(mix.spanned((Fraction("18.005"), Fraction("19.005")), unit=" ms"), "18 to 19.01 ms")

    def test_a_percentile_that_one_request_can_move(self):
        # Nineteen requests of 1 ms to 19 ms and one without a duration: the 19th of the twenty in order is 18
        # for a duration up to 18, the duration itself up to 19, and 19 beyond.
        known = list(range(1, 20))
        values = set()
        for x in [Fraction(k, 4) for k in range(0, 121)]:
            twenty = sorted(known + [x])
            values.add(twenty[18])
        self.assertEqual((min(values), max(values)), (18, 19))
        text = printed(mix.components, {"web": known + [10]}, {"web": known + [None]}, mix.p95, "p95",
                       "distributions").splitlines()
        self.assertEqual(text[0], "p95, 18 -> 18 to 19 ms: +0 to +1 ms")
        self.assertEqual(text[-1], "    within +0 to +1, mix +0")
        # And with nineteen requests of 100 ms the same request moves nothing, as the listing says.
        self.assertIn("the other nineteen requests all took", " ".join(mix.one_lost.__doc__.split()))
        self.assertNotIn("does not reach", mix.one_lost.__doc__)

    def test_a_strict_bound_is_not_attained(self):
        # Known to be above 200 ms: the mean (1900 + x) / 20 is above 105 ms for every such x, and 105 ms is
        # its greatest lower bound.
        for x in (Fraction(2001, 10), Fraction(200001, 1000), 201):
            self.assertGreater(Fraction(1900 + x, 20), 105)
        self.assertEqual(Fraction(1900 + 200, 20), 105)
        head = printed(mix.components, {"web": [100] * 20}, {"web": [100] * 19 + [(">", 200)]}).splitlines()[0]
        self.assertEqual(head, "mean, 100 -> 105 ms or more: +5 ms or more")
        self.assertIn("a strict bound does not attain", " ".join(mix.ends.__doc__.split()))

    def test_a_difference_that_is_not_defined(self):
        lost = {"web": [math.inf]}
        self.assertEqual(printed(mix.components, lost, lost).splitlines(),
                         ["mean, inf -> inf ms: not defined",
                          "  the mix is the same in both periods:",
                          "    within not defined, mix +0"])
        # A change that can be anything is another matter: an open duration in each period.
        text = printed(mix.components, {"web": [100, None]}, {"web": [100, None]}).splitlines()
        self.assertEqual(text[0], "mean, 50 or more -> 50 ms or more: no limit")
        self.assertEqual(mix.apart((math.inf, math.inf), (math.inf, math.inf))[0] !=
                         mix.apart((math.inf, math.inf), (math.inf, math.inf))[0], True)   # no number
        self.assertEqual(mix.apart((50, math.inf), (50, math.inf)), (-math.inf, math.inf))
        self.assertEqual(mix.apart((math.inf, math.inf), (50, 60)), (math.inf, math.inf))


class TheMixContributionOfAMean(unittest.TestCase):
    BEFORE = {"x": [100], "fast": [100, 100], "slow": [200]}
    AFTER = {"x": [None], "fast": [100], "slow": [200, 200]}

    def test_the_table_of_the_review(self):
        # One quarter of both periods is the class x. With its unknown duration x the means are 125 and
        # 125 + x / 4, and in both orders the change within the classes is x / 4 - 25 and that of the mix +25.
        for x in (0, 1, 100, 250, 10 ** 6):
            one, two, average = parts(self.BEFORE, filled(self.AFTER, [x]))
            self.assertEqual(one, (100 + Fraction(x, 4), Fraction(x, 4) - 25, 25))
            self.assertEqual(two, (150, Fraction(x, 4) - 25, 25))
            self.assertEqual(average, (Fraction(x, 4) - 25, 25))
        self.assertEqual(printed(mix.components, self.BEFORE, self.AFTER).splitlines(),
                         ["mean, 125 -> 125 ms or more: +0 ms or more",
                          "  open after: 1 of 4 durations",
                          "  class means first, via 100 ms or more:",
                          "    within -25 or more, mix +25 if defined",
                          "  mix first, via 150 ms:",
                          "    within -25 or more, mix +25",
                          "  average of the two orders:",
                          "    within -25 or more, mix +25 if defined"])
        # Until Version 68 the first order and the average read "mix +25", and this test expected that. It
        # is +25 for every x that is a number, as the loop above finds. With the request never completed,
        # x is infinite, the comparison (125 + x / 4) - (100 + x / 4) takes an infinity from an infinity, and
        # the review of Version 68 asked for the mark "if defined" (tests/test_review68.py).

    def test_a_percentile_is_another_matter(self):
        # It depends on ranks. Until Version 67 an order whose way is a span was reported as not determined,
        # "a duration after is open", and this test expected that line. The review of Version 67 showed that
        # one open duration stands in both percentiles of such an order and can leave their difference
        # settled; the order is now bounded, see test_review67. Here the p95 is the third of four durations:
        # 200 ms and more at the mix of either period, and for an x up to 200 ms it is 200 ms at both.
        text = printed(mix.components, self.BEFORE, self.AFTER, mix.p95, "p95", "distributions")
        self.assertNotIn("not determined", text)
        self.assertIn("  class distributions first, via 200 ms or more:\n", text)

    def test_every_completion_lies_in_the_spans(self):
        rng = random.Random(6603)
        exact = 0
        for _ in range(250):
            names = ["a", "b", "c"][:rng.randint(1, 3)]

            def period():
                rec = {}
                for name in names:
                    xs = [rng.choice((rng.randint(0, 300),) * 5 + (None, (">=", rng.randint(0, 300))))
                          for _ in range(rng.randint(1, 6))]
                    rec[name] = xs
                return rec
            before, after = period(), period()
            if rng.random() < 0.3:                                              # complete records
                before, after = filled(before, [50] * 99), filled(after, [70] * 99)
            spans = [mix.of_mean(before, after, f) for f in (0, 1, Fraction(1, 2))]
            floors = [[0 if x is None else x[1] for xs in rec.values() for x in xs if mix.is_open(x)]
                      for rec in (before, after)]
            opened = sum(map(len, floors))
            for trial in range(6):
                values = [[low + (rng.randint(0, 900) if trial else 0) for low in lows] for lows in floors]
                one, two, average = parts(filled(before, values[0]), filled(after, values[1]))
                for span, value in zip(spans[0] + spans[1] + spans[2][1:], one + two + average):
                    self.assertLessEqual(span[0], value)
                    self.assertLessEqual(value, span[1])
                    if not opened:
                        exact += 1
                        self.assertEqual(span, (value, value))
            # The way of the first order at its lower bound is attained with every open duration at its own.
            one, two, _ = parts(filled(before, floors[0]), filled(after, floors[1]))
            self.assertEqual((spans[0][0][0], spans[1][0][0]), (one[0], two[0]))
        self.assertGreater(exact, 1000)

    def test_a_class_that_keeps_its_share_adds_nothing(self):
        rng = random.Random(6604)
        for _ in range(100):
            k = rng.randint(1, 4)
            before = {"x": [rng.randint(1, 300) for _ in range(k)],
                      "fast": [100] * rng.randint(1, 5), "slow": [200] * rng.randint(1, 5)}
            n = len(before["fast"]) + len(before["slow"])
            fast = rng.randint(1, n - 1)
            after = {"x": [None] * k, "fast": [100] * fast, "slow": [200] * (n - fast)}
            for f in (0, 1, Fraction(1, 2)):
                _, inner, mixed = mix.of_mean(before, after, f)
                self.assertTrue(mix.one(mixed), (before, after))
                self.assertEqual(inner[1], math.inf)
            shift = Fraction(fast, n + k) - Fraction(len(before["fast"]), n + k)
            self.assertEqual(mix.of_mean(before, after, 0)[2][0], shift * 100 - shift * 200)

    def test_the_transcript_is_what_it_was(self):
        text = (ROOT / "expected" / "ch23_mix.txt").read_text(encoding="utf-8")
        self.assertEqual(printed(mix.main), text)
        self.assertEqual(len(text.splitlines()), 122)
        self.assertIn("  mix first: not determined,\n    no bulk requests before\n", text)


class ASumBeyondAFloat(unittest.TestCase):
    def nearest(self, z):
        """Two labels of 1 and one of 0 at one logit z: the loss is least where the stated probability is 2/3,
        at T = z / ln 2."""
        with localcontext() as ctx:
            ctx.prec = 60
            return float(dec(z) / D(2).ln())

    def test_the_case_of_the_review(self):
        z = 2e307
        data = [(z, 1), (z, 1), (z, 0)]
        self.assertEqual(self.nearest(z), 2.8853900817779266e307)
        # At the upper end of the search z / T is 0.5. The three terms are z / (1 + exp(0.5)), twice, and
        # z / (1 + exp(-0.5)), and the allowance of each is 2 * 0.5 + 8 times its size: each a float, and
        # their sum beyond one.
        allowances = [z / (1 + math.exp(0.5)) * 9] * 2 + [z / (1 + math.exp(-0.5)) * 9]
        self.assertLess(max(allowances), 1.2e308)
        with self.assertRaises(OverflowError):
            math.fsum(allowances)                                               # Version 66 added them so
        ms = [Fraction(z), Fraction(z), Fraction(-z)]
        self.assertIsNone(cal.quick(ms, 1 / Fraction(4e307)))
        self.assertEqual(cal.fit_temperature(data, 1e307, 4e307), self.nearest(z))

    def test_the_same_fit_at_other_scales(self):
        for z, lo, hi in ((1, 0.5, 2), (2e307, 2e307, 1.7e308), (2e-307, 1e-307, 4e-307), (5e-324, 5e-324, 1e-320),
                          (Fraction(10 ** 30, 3), 1e29, 1e31)):
            data = [(z, 1), (z, 1), (z, 0)]
            self.assertEqual(cal.fit_temperature(data, lo, hi), self.nearest(z), z)
        self.assertEqual(f"{cal.fit_temperature(cal.make(2000, 2.0, random.Random(18))):.2f}", "2.04")


class DegreesOfFreedomAsTheyAreGiven(unittest.TestCase):
    def test_a_finite_number_beyond_a_float(self):
        for big in (Decimal("1e400"), 10 ** 400, Fraction(10 ** 400), Decimal("1.8e308")):
            self.assertEqual(refusal(Result, 0, 1, "", big), "ValueError: dof beyond a float", big)
            self.assertEqual(refusal(Budget().add, "x", "A", 1, big), "ValueError: dof beyond a float", big)
            self.assertEqual(refusal(welch_satterthwaite, [1], [big]), "ValueError: dof beyond a float", big)
        self.assertEqual(float(Decimal("1e400")), math.inf)                     # what Version 66 took it for

    def test_an_infinity_that_is_meant(self):
        for meant in (math.inf, Decimal("Infinity")):
            self.assertEqual(Result(0, 1, dof=meant).dof, math.inf)
            self.assertEqual(Budget().add("x", "B", 1, meant).dof(), math.inf)
            self.assertEqual(welch_satterthwaite([1, 2], [meant, meant]), math.inf)
        for wrong in (-math.inf, Decimal("-Infinity"), Decimal("NaN"), math.nan):
            self.assertEqual(refusal(Result, 0, 1, "", wrong), "ValueError: dof must be > 0", wrong)
        self.assertEqual(Result(0, 1, dof=Decimal("1.7e308")).dof, 1.7e308)

    def test_what_is_no_number(self):
        for bad in (b"4", bytearray(b"4"), "4", None, True, [4], 4j):
            self.assertEqual(refusal(Result, 0, 1, "", bad), "TypeError: dof is no number", bad)
            self.assertEqual(refusal(Budget().add, "x", "A", 1, bad), "TypeError: dof is no number", bad)
            self.assertEqual(refusal(welch_satterthwaite, [1], [bad]), "TypeError: dof is no number", bad)
        self.assertEqual(float(b"4"), 4.0)                                      # what Version 66 took it for
        for good, stored in ((4, 4.0), (4.5, 4.5), (Fraction(9, 2), 4.5), (Decimal("4.5"), 4.5)):
            self.assertEqual(Result(0, 1, dof=good).dof, stored)


class TheCeilingOfAQuantile(unittest.TestCase):
    def test_the_two_calls_of_the_review(self):
        # One degree of freedom is the Cauchy distribution: the quantile at an upper tail p is cot(pi p),
        # which for so small a p is 1 / (pi p) less pi p / 3.
        p = 3.5e-151
        with localcontext() as ctx:
            ctx.prec = 60
            want = float(1 / (PI * dec(p)) - PI * dec(p) / 3)
        self.assertAlmostEqual(want / 9.09456817668e149, 1, delta=1e-11)        # the figure of the review
        self.assertLess(2.0 ** 498, want)
        self.assertLess(want, min(2.0 ** 499, 1e150))                           # Version 66 gave up between the two
        self.assertAlmostEqual(dist.t_isf(p, 1) / want, 1, delta=1e-12)
        k = Result(0, 1, dof=0.0086).k(0.95)
        self.assertAlmostEqual(k / 8.91873391176e149, 1, delta=1e-11)           # the figure of the review
        self.assertGreater(dist.t_sf(k * (1 - 1e-9), 0.0086), 0.025)
        self.assertLess(dist.t_sf(k * (1 + 1e-9), 0.0086), 0.025)

    def test_up_to_the_ceiling_and_not_beyond(self):
        with localcontext() as ctx:
            ctx.prec = 60
            below, above = (float(1 / (PI * D(t))) for t in ("0.99e150", "1.01e150"))
        self.assertAlmostEqual(dist.t_isf(below, 1) / 0.99e150, 1, delta=1e-12)
        self.assertEqual(refusal(dist.t_isf, above, 1), "ValueError: quantile too large")
        self.assertEqual(refusal(dist.t_isf, 1e-300, 0.5), "ValueError: quantile too large")
        self.assertEqual(refusal(Result(0, 1, dof=0.004).k, 0.95), "ValueError: quantile too large")
        self.assertIn("hi is doubled, up to 1e150", " ".join(dist._bisect.__doc__.split()))


class TheNormalTailAndAnOddNumber(unittest.TestCase):
    def test_the_references(self):
        with localcontext() as ctx:
            ctx.prec = 50
            self.assertEqual(normal_tail(0), D("0.5"))
            for t in (2.5, 3.0, 4.0):                                           # the two forms against each other
                self.assertLess(abs(normal_tail(t, series=True) / normal_tail(t, series=False) - 1), D("1e-45"))
        self.assertAlmostEqual(float(normal_tail(1.959963984540054)), 0.025, delta=1e-16)
        # One degree of freedom: 1/2 - atan(t) / pi. Three: 1/2 - (atan(u) + u / (1 + u * u)) / pi, u = t / sqrt(3).
        # As floats the closed forms lose digits where the tail is small, so they are asked up to t = 2.
        for t in (0.3, 1.0, 2.0):
            self.assertAlmostEqual(float(odd_tail(t, 1)) / (0.5 - math.atan(t) / math.pi), 1, delta=1e-13)
            u = t / math.sqrt(3)
            self.assertAlmostEqual(float(odd_tail(t, 3)) / (0.5 - (math.atan(u) + u / (1 + u * u)) / math.pi),
                                   1, delta=1e-13)
        # Far out the tail with one degree of freedom is atan(1 / t) / pi, with no digits lost.
        for t in (7.5, 40.0, 1e6):
            self.assertAlmostEqual(float(odd_tail(t, 1)) / (math.atan(1 / t) / math.pi), 1, delta=1e-14)
        self.assertLess(abs(odd_tail(1, 1) - D("0.25")), D("1e-50"))

    def test_an_odd_million(self):
        # The review measured 1.21e-15 and 3.09e-13 here, against an integral of the density. The response
        # before it had quoted the errors at 1 000 000 and at 1 000 002, which do not bound the error between.
        for t, review, limit in ((1.96, "0.02499803379249625", 1e-13), (35.0, "1.637491202023676e-268", 1e-12)):
            want = odd_tail(t, 1000001)
            self.assertLess(abs(want / D(review) - 1), D("1e-15"), t)             # the reference of the review
            self.assertLess(off(dist.t_sf(t, 1000001.0), want), limit, t)
        for dof in (3, 5, 31, 63, 65, 1001, 99999):
            for t in (0.3, 1.0, 1.96, 6.0, 20.0):
                want = odd_tail(t, dof)
                self.assertLess(off(dist.t_sf(t, float(dof)), want), 1e-12 if want < 1e-12 else 1e-13, (dof, t))

    def test_the_normal_tail_beyond_1e25(self):
        rng = random.Random(6605)
        ts = [1e-9, 0.01, 0.3, 1.0, 1.645, 1.96, 3.0, 6.0, 12.0, 20.0, 30.0, 35.0, 35.00000000002145, 37.0]
        ts += [math.exp(rng.uniform(-3, 3.6)) for _ in range(60)]
        for t in ts:
            want = normal_tail(t)
            for dof in (1.0000000000000003e25, 1e30, 1e300, math.inf):
                self.assertLess(off(dist.t_sf(t, dof), want), 1e-12 if want < 1e-12 else 1e-13, (t, dof))

    def test_not_the_same_to_the_last_place(self):
        # The point of the review. Student's tail exceeds the normal one by phi(t) * (t ** 3 + t) / (4 v), to
        # the first order in 1 / v: 4 parts in 10 ** 20 of the tail here, and enough to round to the next float.
        t, v = 35.00000000002145, 1.0000000000000003e25
        with localcontext() as ctx:
            ctx.prec = 60
            normal = normal_tail(t)
            density = (-dec(t) ** 2 / 2).exp() / (2 * PI).sqrt()
            student = normal + density * (dec(t) ** 3 + dec(t)) / (4 * dec(v))
            self.assertLess(student / normal - 1, D("1e-19"))
        self.assertEqual((float(normal), float(student)), (1.1249107056271405e-268, 1.1249107056271407e-268))
        self.assertEqual(math.nextafter(float(normal), math.inf), float(student))
        self.assertLess(off(dist.t_sf(t, v), student), 1e-12)
        text = " ".join(dist.t_sf.__doc__.split())
        self.assertNotIn("to the last place", text)
        self.assertIn("an approximation within the accuracy measured here", text)


if __name__ == "__main__":
    unittest.main()
