"""Regression tests added after the review of Version 65.

The review found that components() of Listing 23.2 failed for a request that is in the record without a
duration, or with a bound for one, although within() of the same listing reads such requests, and that a
duration that is no number printed "+nan" and a line that spoke of requests that were not there; that
at_one_logit() of Listing 18.1 returned 2 where 1 is right for the float below one half, because 1 - p rounds
half of the distance away, and divided by zero for a decimal next to one half; that fit_temperature() added
rounded terms, so that on a nearly flat loss it returned an end of its search, or a third above the minimum,
that it made a float of a margin of 1e-400 and reported no finite minimum, and that a lower end of 1e-400
passed its check and then divided by zero; that the upper tail of Student's distribution rose by 8.76e-11
between two adjacent arguments at 990 000 degrees of freedom, where the continued fraction changed sides, and
was wrong by up to 2e-7 beyond a million; that a budget of one component with 1e-310 degrees of freedom had none, that a
Result took a decimal number of them and then failed, and that betainc() overflowed for two shapes next to
zero; and that the figure generator called its drawing width the text width of the book. The review also asked the
text of Chapter 18 to say that a decision made from a rounded probability can change with the temperature, and
which probability a calibration error is of; the three numerical examples of that text are tested here.

Each test computes what it expects independently of the code under test: means from fractions and a
percentile by counting, a logit and a fitted temperature from decimals of 60 to 120 digits, rounded once to
the nearest float, the sign of a slope from decimals, and Student's tail for an even number of degrees of
freedom from its finite sum, or the rest of that series, in decimals.
"""
import contextlib
import io
import math
import random
import sys
import unittest
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from unittest import mock

from examples import ch18_calibration as cal
from examples import ch23_mix as mix
from swmetro import Budget, Result, dist, welch_satterthwaite

ROOT = Path(__file__).resolve().parent.parent
D = Decimal


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


def even_tail(t, n, digits=60):
    """P(T > t) of Student's distribution for an even number n of degrees of freedom, in decimals. With
    w = n / (n + t * t) the central probability P(-t < T < t) is sqrt(1 - w) times the sum over j < n / 2 of
    c_j * w ** j, where c_0 = 1 and c_j = c_(j-1) * (2j - 1) / (2j); the c_j are the coefficients of the
    binomial series of (1 - w) ** (-1/2), so that the terms from n / 2 on add up to what is left of 1. The
    tail is half of that rest: it is taken from the terms from n / 2 on where they fall fast enough, and
    otherwise as one less the finite sum."""
    with localcontext() as ctx:
        ctx.prec = digits
        q = dec(t) ** 2
        w = n / (n + q)
        root = (q / (n + q)).sqrt()
        term, low = D(1), D(0)
        for j in range(1, n // 2 + 1):
            low += term
            term *= w * (2 * j - 1) / (2 * j)
        if (1 - w) * 20000 < 1:
            return (1 - root * low) / 2
        high, j = D(0), n // 2
        while term > high * (1 - w) * D(10) ** (5 - digits):
            high += term
            j += 1
            term *= w * (2 * j - 1) / (2 * j)
        return root * high / 2


def off(got, want):
    """The relative error of a float against a decimal."""
    with localcontext() as ctx:
        ctx.prec = 60
        return float(abs((D(got) - want) / want))


def decimal_slope(ms, beta, digits=120):
    """The derivative of the summed loss by beta = 1 / T, in decimals: the sum of -m / (1 + exp(beta * m))."""
    with localcontext() as ctx:
        ctx.prec = digits
        b = dec(beta)
        return sum(-dec(m) / (1 + (b * dec(m)).exp()) for m in ms)


def decimal_minimum(data, digits=120):
    """The temperature at which that derivative is zero, as a decimal, by bisection on beta. The derivative
    rises with beta; it is below zero next to beta = 0 where the margins sum to more than zero, and above
    zero far out where one of them is below zero."""
    ms = [Fraction(z) if y else -Fraction(z) for z, y in data]
    with localcontext() as ctx:
        ctx.prec = digits
        a, b = D(0), D(1)
        while decimal_slope(ms, b, digits) < 0:
            a, b = b, 2 * b
        for _ in range(3 * digits):
            mid = (a + b) / 2
            if decimal_slope(ms, mid, digits) < 0:
                a = mid
            else:
                b = mid
        return 2 / (a + b)


def decimal_logit(p, digits=60):
    with localcontext() as ctx:
        ctx.prec = digits
        p = dec(p)
        return (p / (1 - p)).ln()


def nineteenth(durations):
    """The inverse empirical 95th percentile of twenty durations, by counting: the least of them with at
    least nineteen at or below it."""
    assert len(durations) == 20
    return min(x for x in durations if sum(1 for y in durations if y <= x) >= 19)


class ADurationThatWasLost(unittest.TestCase):
    BEFORE = {"web": [100] * 20}

    def after(self, last):
        return {"web": [100] * 19 + [last]}

    def test_the_table_of_the_review(self):
        # Twenty requests of one class in each period. With the last duration x, the mean after is
        # (1900 + x) / 20, and the 19th of the twenty in order is 100 whatever x is.
        least = Fraction(1900 + 0, 20)
        self.assertEqual(least, 95)
        for x in (0, 50, 100, 10 ** 9):
            self.assertEqual(nineteenth([100] * 19 + [x]), 100)
        for last in (None, float("nan"), Decimal("NaN")):
            text = printed(mix.components, self.BEFORE, self.after(last))
            self.assertEqual(text.splitlines(), ["mean, 100 -> 95 ms or more: -5 ms or more",
                                                 "  open after: 1 of 20 durations",
                                                 "  the mix is the same in both periods:",
                                                 "    within -5 or more, mix +0"])
            text = printed(mix.components, self.BEFORE, self.after(last), mix.p95, "p95", "distributions")
            self.assertEqual(text.splitlines()[0], "p95, 100 -> 100 ms: +0 ms")
            self.assertEqual(text.count("within +0, mix +0"), 3)
        # Known to be 200 ms or more: the mean after is (1900 + 200) / 20 or more.
        self.assertEqual(Fraction(1900 + 200, 20) - 100, 5)
        for last in ((">=", 200), (">", 200)):
            text = printed(mix.components, self.BEFORE, self.after(last))
            self.assertEqual(text.splitlines()[0], "mean, 100 -> 105 ms or more: +5 ms or more")
            self.assertIn("    within +5 or more, mix +0", text)
            text = printed(mix.components, self.BEFORE, self.after(last), mix.p95, "p95", "distributions")
            self.assertEqual(text.splitlines()[0], "p95, 100 -> 100 ms: +0 ms")

    def test_no_line_speaks_of_requests_that_are_missing(self):
        # Version 65 printed "+nan ms" and "no  requests after" for a NaN: twenty requests are in each record.
        for last in (None, float("nan"), (">=", 200), math.inf):
            for args in ((), (mix.p95, "p95", "distributions")):
                text = printed(mix.components, self.BEFORE, self.after(last), *args)
                self.assertNotIn("nan", text)
                self.assertNotIn("requests after", text)
                self.assertNotIn("requests before", text)

    def test_a_request_that_never_ended(self):
        # Stored as infinity it is a duration, and no open one: the mean is infinite, and the mix is the same.
        text = printed(mix.components, self.BEFORE, self.after(math.inf))
        self.assertEqual(text.splitlines(), ["mean, 100 -> inf ms: +inf ms",
                                             "  the mix is the same in both periods:",
                                             "    within +inf, mix +0"])
        text = printed(mix.components, self.BEFORE, self.after(math.inf), mix.p95, "p95", "distributions")
        self.assertEqual(text.splitlines()[0], "p95, 100 -> 100 ms: +0 ms")

    def test_the_ends_of_a_statistic(self):
        # reach() against a mean from fractions and a percentile by counting, with every open duration at
        # its least and then far above every other.
        rng = random.Random(6501)
        for _ in range(200):
            xs = [rng.choice((rng.randint(0, 300), None, (">", rng.randint(0, 300)), (">=", rng.randint(0, 300))))
                  for _ in range(20)]
            if rng.random() < 0.3:
                xs = [x if isinstance(x, int) else rng.randint(0, 300) for x in xs]
            pairs = [(x, 1) for x in xs]
            least = [0 if x is None else x[1] if isinstance(x, tuple) else x for x in xs]
            most = [x if isinstance(x, int) else 10 ** 9 for x in xs]
            opened = sum(1 for x in xs if not isinstance(x, int))
            low, top = mix.reach(mix.mean, pairs)
            self.assertEqual(low, Fraction(sum(least), 20))
            self.assertEqual(top, math.inf if opened else Fraction(sum(most), 20))
            low, top = mix.reach(mix.p95, pairs)
            self.assertEqual(low, nineteenth(least))
            far = nineteenth(most)
            self.assertEqual(top, math.inf if far == 10 ** 9 else far)

    def test_two_mixes_and_one_open_duration(self):
        # Before: 10 web requests of 100 ms and 5 api requests of 50 ms. After: 19 web requests of 100 ms,
        # one more without a duration, and 2 api requests of 60 ms.
        before = {"web": [100.0] * 10, "api": [50.0] * 5}
        after = {"web": [100.0] * 19 + [None], "api": [60.0] * 2}
        start = Fraction(10 * 100 + 5 * 50, 15)
        least = Fraction(19 * 100 + 0 + 2 * 60, 22)
        via = Fraction(20 * 100 + 2 * 50, 22)                                   # the durations of before at the mix of after
        self.assertEqual([round(x) for x in (start, least, via)], [83, 92, 95])
        self.assertEqual((round(least - start), round(via - start), round(least - via)), (8, 12, -4))
        text = printed(mix.components, before, after).splitlines()
        # Version 66 printed "92 ms or more" for a least mean of 91.82 ms, and this test expected it: the limit
        # was rounded to the nearest whole millisecond, which shut out the means from 91.82 ms to 92 ms. It also
        # gave up the first order. From Version 67 a limit is rounded to its safe side, to the hundredth, and
        # each part of the change of a mean is a sum over the classes (tests/test_review66.py).
        self.assertLess(least, 92)
        self.assertEqual(text, ["mean, 83 -> 91.81 ms or more: +8.48 ms or more",
                                "  open after: 1 of 22 durations",
                                "  class means first, via 83.33 ms or more:",
                                "    within +0 or more, mix +8.48 or more",
                                "  mix first, via 95 ms:",
                                "    within -3.64 or more, mix +12",
                                "  average of the two orders:",
                                "    within -1.82 or more, mix +10.3 or more"])
        # A request stored as infinity before: the statistic of that period is infinite at any mix.
        text = printed(mix.components, {"web": [100.0] * 10 + [math.inf], "api": [50.0] * 5},
                       {"web": [100.0] * 19, "api": [60.0] * 2})
        self.assertIn("  mix first: not determined,\n    a duration before is infinite\n", text)
        self.assertNotIn("nan", text)

    def test_a_class_without_requests_keeps_its_line(self):
        text = printed(mix.one_bulk)
        self.assertIn("  mix first: not determined,\n    no bulk requests before\n", text)

    def test_the_transcript(self):
        text = (ROOT / "expected" / "ch23_mix.txt").read_text(encoding="utf-8")
        self.assertEqual(printed(mix.main), text)
        last = text.splitlines()[-14:]
        self.assertEqual(last[:6], ["20 web requests of 100 ms in each",
                                    "period, one duration lost after:",
                                    "mean, 100 -> 95 ms or more: -5 ms or more",
                                    "  open after: 1 of 20 durations",
                                    "  the mix is the same in both periods:",
                                    "    within -5 or more, mix +0"])
        self.assertEqual(last[6], "p95, 100 -> 100 ms: +0 ms")
        self.assertLessEqual(max(map(len, text.splitlines())), 50)


class TheLogitOfAProbabilityNextToOneHalf(unittest.TestCase):
    def reference(self, logit, p):
        with localcontext() as ctx:
            ctx.prec = 60
            return float(dec(logit) / decimal_logit(p))

    def test_the_two_cases_of_the_review(self):
        p, z = math.nextafter(0.5, 0.0), -math.ulp(1.0)
        self.assertEqual(1 - p, 0.5)                                            # half of the distance is rounded away
        self.assertEqual(z / math.log(p / (1 - p)), 2.0)                        # what Version 65 returned
        self.assertEqual(self.reference(z, p), 1.0)
        self.assertEqual(cal.at_one_logit(z, p), 1.0)
        z, p = Decimal("4e-18"), Decimal("0.500000000000000001")
        with self.assertRaises(ZeroDivisionError):
            float(z) / math.log(p / (1 - p))                                    # Version 65
        self.assertEqual(self.reference(z, p), 1.0)
        self.assertEqual(cal.at_one_logit(z, p), 1.0)

    def test_every_part_of_the_range(self):
        rng = random.Random(6502)
        ps = [0.5 + s * 2.0 ** -k for k in (2, 3, 10, 29, 30, 31, 40, 52, 53) for s in (1, -1)]
        ps += [math.nextafter(0.5, 1.0), 0.25, 0.75, math.nextafter(0.25, 0), math.nextafter(0.75, 1), 0.98, 0.02]
        ps += [5e-324, 1e-320, 2.0 ** -1022, 1e-300, 1e-17, 1 - 2.0 ** -53, 1 - 1e-9]
        ps += [rng.random() for _ in range(60)]
        ps += [Fraction(1, 3), Decimal("0.7"), Decimal("1e-400"), Decimal("0.5") + Decimal("1e-400")]
        for p in ps:
            if p == 0.5:
                continue
            for size in (1e-300, 1e-9, 1.0, 40, 1e9):
                logit = size if p > 0.5 else -size
                want = self.reference(logit, p)
                if not 0 < want < math.inf:
                    self.assertEqual(refusal(cal.at_one_logit, logit, p), "ValueError: T is beyond a float")
                    continue
                self.assertAlmostEqual(cal.at_one_logit(logit, p) / want, 1, delta=1e-15, msg=(logit, p))

    def test_what_is_refused(self):
        for logit, p in ((40, 0.5), (40, 0.4), (-40, 0.6), (0, 0.6), (40, 1.0), (40, 0.0), (40, 1.2)):
            self.assertEqual(refusal(cal.at_one_logit, logit, p), "ValueError: no T above zero states this p")
        # A logit of 1e308 at the float above one half: the temperature is beyond a float.
        self.assertEqual(refusal(cal.at_one_logit, 1e308, math.nextafter(0.5, 1.0)),
                         "ValueError: T is beyond a float")
        self.assertEqual(refusal(cal.at_one_logit, 5e-324, 1 - 2.0 ** -53), "ValueError: T is beyond a float")

    def test_the_printed_interval_is_unchanged(self):
        self.assertEqual(printed(cal.main), (ROOT / "expected" / "ch18_calibration.txt").read_text(encoding="utf-8"))
        self.assertEqual(f"{cal.at_one_logit(40, 0.98):.2f}", "10.28")


class WhichProbabilityWasScored(unittest.TestCase):
    def test_a_tie_that_a_temperature_makes(self):
        # The sign of a logit is that of the logit divided by a temperature above zero. The probability that
        # the listing stores for a logit of -1e-16 is below one half at T = 1 and one half at T = 10.
        z = -1e-16
        self.assertLess(cal.scaled(z, 10), 0)
        self.assertEqual(cal.sigmoid(cal.scaled(z, 1)), 0.49999999999999994)
        self.assertEqual(cal.sigmoid(cal.scaled(z, 10)), 0.5)
        self.assertEqual([cal.sigmoid(cal.scaled(z, t)) >= 0.5 for t in (1, 10)], [False, True])

    def test_twenty_cases_from_a_log(self):
        # Twenty cases at a model probability of 0.54, eleven of them positive: one bin, with a gap of
        # |20 * 0.54 - 11| / 20 on the model probabilities and of |20 * 0.5 - 11| / 20 on values logged as 0.5.
        self.assertEqual(abs(20 * Fraction(54, 100) - 11) / 20, Fraction(1, 100))
        self.assertEqual(abs(20 * Fraction(1, 2) - 11) / 20, Fraction(5, 100))
        z = math.log(0.54 / 0.46)
        self.assertAlmostEqual(cal.ece([(z, 1)] * 11 + [(z, 0)] * 9), 0.01, delta=1e-15)
        self.assertEqual(cal.ece([(0.0, 1)] * 11 + [(0.0, 0)] * 9), 0.05)

    def test_a_probability_stored_as_one(self):
        # One positive case at a logit of 40: the bin holds the stored probability, 1, and the residual is
        # taken from the logit, -1 / (1 + exp(40)).
        count, stated, positive, residual = cal.bins_of([(40.0, 1)])[-1]
        self.assertEqual((count, stated, positive), (1, 1, 1))
        with localcontext() as ctx:
            ctx.prec = 40
            want = -1 / (1 + D(40).exp())
        self.assertLess(off(float(residual), want), 1e-15)
        self.assertEqual(f"{float(residual):.2e}", "-4.25e-18")
        self.assertEqual(stated - positive, 0)                                  # the gap of the stored probability


class AFitOnANearlyFlatLoss(unittest.TestCase):
    def nearest(self, data):
        return float(decimal_minimum(data))

    def flat(self, e):
        b = 2.0 ** -e
        return [(math.nextafter(b, math.inf), 1), (b, 0)]

    def test_the_table_of_the_review(self):
        # Two cases, a logit one step above the other, the upper with the label 1. To the second order the
        # derivative by beta is -(a - b) / 2 + beta * (a * a + b * b) / 4, zero near T = b * b / (a - b).
        for e, about in ((50, 4.0), (52, 1.0), (55, 0.125)):
            a, b = self.flat(e)[0][0], self.flat(e)[1][0]
            self.assertAlmostEqual(b * b / (a - b), about, delta=1e-9)
            self.assertAlmostEqual(self.nearest(self.flat(e)) / about, 1, delta=1e-12)
        self.assertEqual(self.nearest(self.flat(50)), 4.000000000000001)
        self.assertEqual(self.nearest(self.flat(52)), 1.0000000000000002)
        self.assertEqual(self.nearest(self.flat(55)), 0.12500000000000003)
        self.assertEqual(cal.fit_temperature(self.flat(50)), 4.000000000000001)          # Version 65: 10.0
        self.assertEqual(cal.fit_temperature(self.flat(52)), 1.0000000000000002)         # Version 65: 1.33...
        self.assertEqual(refusal(cal.fit_temperature, self.flat(55)),
                         "ValueError: the minimum is below lo")                          # Version 65: 0.2
        self.assertEqual(cal.fit_temperature(self.flat(55), 0.1, 10), 0.12500000000000003)

    def test_the_rounded_terms_add_up_to_nothing(self):
        # Why Version 65 stopped at an end: at T = 10 the two rounded terms are equal and opposite, and
        # their sum as fractions is zero, while the terms as they are leave a slope below zero there.
        (a, _), (b, _) = self.flat(50)
        beta = 1 / 10.0
        rounded = Fraction(-a * cal.sigmoid(-beta * a)) + Fraction(b * cal.sigmoid(beta * b))
        self.assertEqual(rounded, 0)
        self.assertLess(decimal_slope([a, -b], Fraction(1, 10)), 0)
        self.assertEqual(cal.lean([a, -b], Fraction(1, 10)), -1)

    def test_the_sign_of_the_slope(self):
        rng = random.Random(6503)
        for _ in range(150):
            ms = [rng.choice((1, -1)) * math.exp(rng.uniform(-40, 6)) for _ in range(rng.randint(1, 6))]
            beta = Fraction(math.exp(rng.uniform(-5, 5)))
            want = decimal_slope(ms, beta)
            self.assertEqual(cal.lean(ms, beta), (want > 0) - (want < 0), (ms, beta))
        # On either side of the minimum of the first row of the table, one float apart.
        data = self.flat(50)
        ms = [data[0][0], -data[1][0]]
        best = self.nearest(data)
        down, up = math.nextafter(best, 0.0), math.nextafter(best, math.inf)
        self.assertEqual(cal.lean(ms, 1 / Fraction(down)), 1)                   # the minimum is above
        self.assertEqual(cal.lean(ms, 1 / Fraction(up)), -1)                    # the minimum is below
        self.assertIsNone(cal.quick([Fraction(m) for m in ms], 1 / Fraction(down)))
        self.assertEqual(cal.lean([], 1.0), 0)
        self.assertEqual(cal.lean([0.0, 0], Fraction(1, 3)), 0)

    def test_floats_and_decimals_agree_where_floats_decide(self):
        rng = random.Random(6504)
        decided = 0
        for _ in range(300):
            ms = [Fraction(rng.uniform(-6, 6)) for _ in range(rng.randint(1, 8))]
            beta = Fraction(rng.uniform(0.05, 5))
            fast = cal.quick(ms, beta)
            if fast is not None:
                decided += 1
                self.assertEqual(fast, cal.sure(ms, beta, 40))
        self.assertGreater(decided, 250)

    def test_a_slope_without_a_sign_is_refused(self):
        data = self.flat(50)
        ms = [data[0][0], -data[1][0]]
        with mock.patch.object(cal, "sure", return_value=None) as asked:
            self.assertEqual(refusal(cal.lean, ms, Fraction(1, 4)),
                             "ValueError: the slope has no sign in 640 digits")
        self.assertEqual([call.args[2] for call in asked.call_args_list], [40, 80, 160, 320, 640])

    def test_the_float_nearest_the_minimum(self):
        rng = random.Random(6505)
        inside = 0
        for _ in range(60):
            n = rng.randint(2, 9)
            data = [(rng.choice((rng.uniform(-8, 8), rng.randint(-5, 5), Fraction(rng.randint(-40, 40), 7))),
                     rng.randint(0, 1)) for _ in range(n)]
            ms = [Fraction(z) if y else -Fraction(z) for z, y in data]
            if min(ms) >= 0 or sum(ms) <= 0:
                continue
            want = self.nearest(data)
            if want < 0.2:
                self.assertEqual(refusal(cal.fit_temperature, data), "ValueError: the minimum is below lo")
            elif want > 10:
                self.assertEqual(refusal(cal.fit_temperature, data), "ValueError: the minimum is above hi")
            else:
                inside += 1
                self.assertEqual(cal.fit_temperature(data), want, data)
        self.assertGreater(inside, 15)

    def test_a_minimum_next_to_an_end(self):
        # Two labels of 1 and one of 0 at one logit z: the loss is least where the stated probability is
        # 2/3, at T = z / ln 2. With z the float nearest ln 2 the minimum is below 1 by less than half a
        # step of the floats there, and 1 is the float nearest to it, in the search and at either end.
        z = math.log(2)
        data = [(z, 1), (z, 1), (z, 0)]
        with localcontext() as ctx:
            ctx.prec = 60
            exact = D(z) / D(2).ln()
            self.assertLess(exact, 1)
            self.assertLess(1 - exact, D(2) ** -54)
        self.assertEqual(self.nearest(data), 1.0)
        for lo, hi in ((0.2, 10), (1.0, 10), (0.2, 1.0), (1, 10)):
            self.assertEqual(cal.fit_temperature(data, lo, hi), 1.0)
        # A logit lower by one part in 10**10: the minimum is below 1 by as much, and beyond an end at 1.
        z = math.log(2) * (1 - 1e-10)
        data = [(z, 1), (z, 1), (z, 0)]
        want = self.nearest(data)
        self.assertAlmostEqual(1 - want, 1e-10, delta=1e-15)
        self.assertEqual(cal.fit_temperature(data, 0.2, 1.0), want)
        self.assertEqual(refusal(cal.fit_temperature, data, 1.0, 10), "ValueError: the minimum is below lo")
        self.assertEqual(refusal(cal.fit_temperature, data, 0.2, math.nextafter(want, 0.0)),
                         "ValueError: the minimum is above hi")

    def test_a_margin_below_the_least_float(self):
        # The derivative is -1000 / (1 + exp(1000 beta)) + m / (1 + exp(-m beta)) with m = 1e-400: zero
        # near beta = ln(2000 / m) / 1000, a temperature of 1.0768.
        data = [(1000, 1), (Decimal("1e-400"), 0)]
        self.assertEqual(float(Decimal("1e-400")), 0.0)                         # Version 65: "no finite T"
        about = 1000 / (math.log(2000) + 400 * math.log(10))
        want = self.nearest(data)
        self.assertAlmostEqual(want / about, 1, delta=1e-9)
        self.assertEqual(f"{want:.15f}", "1.076849424133435")
        self.assertEqual(cal.fit_temperature(data), want)

    def test_the_ends_of_the_search_are_floats(self):
        data = [(2, 1), (1, 0), (0, 1)]
        for lo in (Decimal("1e-400"), 0, -1, True, "0.2", None, math.nan, math.inf):
            self.assertEqual(refusal(cal.fit_temperature, data, lo, 10), "ValueError: need 0 < lo < hi < inf", lo)
        for hi in (math.inf, 0.2, 0.1, 10 ** 400, None):
            self.assertEqual(refusal(cal.fit_temperature, data, 0.2, hi), "ValueError: need 0 < lo < hi < inf", hi)
        want = cal.fit_temperature(data)
        self.assertEqual(want, self.nearest(data))
        self.assertEqual(cal.fit_temperature(data, Decimal("0.2"), Fraction(10)), want)
        self.assertEqual(cal.fit_temperature(data, 5e-324, 1.7e308), want)


class StudentsTailAtManyDegreesOfFreedom(unittest.TestCase):
    PAIR = (1.7320490580631764, 1.7320490580631767)                            # two adjacent floats

    def test_the_two_arguments_of_the_review(self):
        self.assertEqual(math.nextafter(self.PAIR[0], math.inf), self.PAIR[1])
        low, high = (dist.t_sf(t, 990000.0) for t in self.PAIR)
        for got, t in ((low, self.PAIR[0]), (high, self.PAIR[1])):
            self.assertLess(off(got, even_tail(t, 990000)), 1e-13)
        # Version 65 gave 0.04163256972115614 and 0.041632569808741894: a rise of 8.76e-11, and the first
        # of them wrong by 2e-9 of itself. What is left between the two is rounding.
        self.assertGreater(abs(0.04163256972115614 / low - 1), 1e-9)
        self.assertLess(abs(high / low - 1), 1e-13)
        r = Result(0, 1, dof=990000)
        self.assertEqual(r.prob_below(-self.PAIR[0]), low)

    def test_the_errors_of_the_review(self):
        # Relative errors of Version 65: 1.63e-9, 2.37e-10 and, at 1 000 001 degrees of freedom and t = 35,
        # 2.05e-7. The reference here needs an even number.
        for dof, t in ((990000, 1.645), (990000, 1.96), (1000000, 35.0), (1000002, 35.0), (1000002, 1.96)):
            self.assertLess(off(dist.t_sf(t, float(dof)), even_tail(t, dof)), 1e-12, (dof, t))
        # The odd number between two even ones: the tail falls as the degrees of freedom rise, and its
        # logarithm is close to a straight line over two of them.
        below, at, above = (dist.t_sf(35.0, dof) for dof in (1000000.0, 1000001.0, 1000002.0))
        self.assertGreater(below, at)
        self.assertGreater(at, above)
        self.assertAlmostEqual(at / (math.sqrt(below) * math.sqrt(above)), 1, delta=1e-11)
        self.assertGreater(below / above - 1, 5e-7)                             # the two neighbors differ by more

    def test_one_method_for_every_number(self):
        rng = random.Random(6506)
        ts = [0.3, 1.0, 1.2, 1.645, 1.96, 3.0, 12.0, 35.0] + [math.exp(rng.uniform(-2, 3.6)) for _ in range(6)]
        for dof in (2, 4, 30, 32, 34, 64, 1000, 10000, 100000):
            for t in ts:
                want = even_tail(t, dof)
                err = off(dist.t_sf(t, float(dof)), want)
                self.assertLess(err, 1e-12, (dof, t))
                if want > 1e-12:
                    self.assertLess(err, 1e-13, (dof, t))

    def test_the_tail_falls_as_t_rises(self):
        dofs = (1, 3, 31.9, 32, 64, 1e3, 989999.0, 990000.0, 990001.0, 999999.0, 1e6, 1000001.0, 1e7, 1e12, 1e18,
                9.9e24, 1e25, 1.1e25, math.inf)
        for dof in dofs:
            for k in range(-45, 36):
                t = 1.7320490580631764 * 1.1 ** k
                here = dist.t_sf(t, dof)
                if here < 1e-300:                                               # next to the least float, or 0
                    continue
                self.assertGreater(here, dist.t_sf(t * (1 + 1e-9), dof), (dof, t))
                # One float further the tail can rise by rounding, and by no more.
                self.assertLess(dist.t_sf(math.nextafter(t, math.inf), dof) / here - 1, 1e-12, (dof, t))

    def test_where_the_fraction_changes_sides(self):
        # The tail is found directly from t = 1 on, about, and as one less the central part below it. Three
        # adjacent floats there: each is right, and they differ by no more than rounding.
        ts = (math.nextafter(1.0, 0.0), 1.0, math.nextafter(1.0, math.inf))
        for dof in (4, 100, 990000, 1000000):
            got = [dist.t_sf(t, float(dof)) for t in ts]
            for t, value in zip(ts, got):
                self.assertLess(off(value, even_tail(t, dof)), 1e-13, (dof, t))
            self.assertLess(abs(got[1] / got[0] - 1), 1e-13, dof)
            self.assertLess(abs(got[2] / got[1] - 1), 1e-13, dof)
        got = [dist.t_sf(t, 1e12) for t in ts]
        self.assertLess(abs(got[2] / got[0] - 1), 1e-13)

    def test_the_normal_tail_takes_over_without_a_step(self):
        for t in (1e-3, 0.5, 1.0, 1.7, 3.0, 8.0, 30.0, 37.0):
            below, above = dist.t_sf(t, 1e25), dist.t_sf(t, math.nextafter(1e25, math.inf))
            self.assertEqual(above, 0.5 * math.erfc(t / math.sqrt(2)))
            self.assertAlmostEqual(below / above, 1, delta=1e-12, msg=t)
        # A tail next to the least normal float keeps its digits: its factors are multiplied in an order in
        # which no product of two of them falls below that float.
        self.assertAlmostEqual(dist.t_sf(37.5, 9.9e24) / (0.5 * math.erfc(37.5 / math.sqrt(2))), 1, delta=1e-12)

    def test_quantiles_above_a_million(self):
        for dof in (1000001.0, 1e7, 1e12, 1e18, 9.9e24):
            for tail in (0.25, 0.2, 0.05, 0.025, 1e-3, 1e-9, 1e-30, 1e-100, 1e-300):
                t = dist.t_isf(tail, dof)
                self.assertAlmostEqual(dist.t_sf(t, dof) / tail, 1, delta=1e-11, msg=(dof, tail))
                self.assertGreater(t, -dist.z_ppf(tail) * (1 - 1e-14), (dof, tail))    # heavier than the normal
            for gap in (1e-100, 1e-12, 1e-3, 0.0011, 0.1, 0.2499):
                t = dist.t_mid(gap, dof)
                self.assertAlmostEqual(dist._centre(t, dof) / gap, 1, delta=1e-12, msg=(dof, gap))
            # Next to t = 0 the probability is in proportion to t, also where t * t is below every float.
            self.assertAlmostEqual(dist.t_mid(1e-300, dof) / dist.t_mid(1e-100, dof), 1e-200, delta=1e-212)
        self.assertFalse(hasattr(dist, "_far"))                                 # no expansion about the normal quantile
        for tail in (0.2, 0.05, 1e-9):
            self.assertEqual(dist.t_isf(tail, math.inf), -dist.z_ppf(tail))
            self.assertEqual(dist.t_isf(tail, 1.1e25), -dist.z_ppf(tail))

    def test_what_the_docstrings_claim(self):
        text = " ".join(dist.t_sf.__doc__.split())
        self.assertIn("below 1e-12 down to tails of 1e-300", text)
        self.assertIn("it can rise by rounding", text)
        text = " ".join(dist.betainc.__doc__.split())
        self.assertIn("written for Student's t", text)


class DegreesOfFreedomAtTheEdge(unittest.TestCase):
    def test_a_budget_of_one_component(self):
        # (u ** 2) ** 2 / (u ** 4 / dof) is dof, whatever u is.
        for u, dof in ((1, 1e-310), (1, 5e-324), (0.8, 4), (1e-200, 3.7), (1e200, 12), (3, 1e300), (2.5, math.inf)):
            self.assertEqual(welch_satterthwaite([u], [dof]), dof, (u, dof))
            self.assertEqual(Budget().add("a", "A", u, dof).dof(), dof, (u, dof))
        self.assertEqual(1.0 ** 4 / 1e-310, math.inf)                           # Version 65 returned 1 / inf

    def test_from_the_budget_to_the_probability(self):
        # The measurement, step by step: a budget of one component, its degrees of freedom, a result, and
        # the probability at a limit.
        budget = Budget("ms").add("repeatability", "A", 1, 1e-310)
        result = Result(0, budget.u_c(), "ms", budget.dof())
        self.assertEqual(result.dof, 1e-310)
        # With next to no degrees of freedom the tail beyond any finite limit is one half, less a part far
        # below the rounding of a float.
        self.assertEqual(result.prob_below(-1), 0.5)
        self.assertEqual(result.prob_below(1), 0.5)
        budget = Budget("ms").add("repeatability", "A", 0.8, 4)
        result = Result(10, budget.u_c(), "ms", budget.dof())
        self.assertEqual(result.dof, 4.0)
        for t in (0.5, 1.0, 2.776, 8.0):
            want = even_tail(t, 4)
            got = result.prob_below(Fraction(10) - Fraction(0.8) * Fraction(t))
            self.assertLess(off(got, want), 1e-13, t)

    def test_several_components(self):
        rng = random.Random(6507)
        for _ in range(100):
            n = rng.randint(2, 5)
            us = [math.exp(rng.uniform(-300, 300)) if rng.random() < 0.2 else rng.uniform(0, 5) for _ in range(n)]
            dofs = [rng.choice((rng.uniform(0.5, 60), rng.randint(1, 30), math.inf, 1e-300)) for _ in range(n)]
            den = sum(Fraction(u) ** 4 / Fraction(d) for u, d in zip(us, dofs) if d != math.inf)
            got = welch_satterthwaite(us, dofs)
            if not den:
                self.assertEqual(got, math.inf)
                continue
            want = sum(Fraction(u) ** 2 for u in us) ** 2 / den
            if want > Fraction(sys.float_info.max):                             # as many as none can count
                self.assertEqual(got, math.inf)
                continue
            self.assertEqual(got, float(want))
            self.assertGreaterEqual(got, min(dofs))                              # never fewer than the fewest
        self.assertEqual(welch_satterthwaite([1, 1], [1e308, 1e308]), math.inf)  # 2e308 is beyond a float

    def test_the_types_of_a_number_of_degrees_of_freedom(self):
        result = Result(0, 1, dof=Decimal("1"))
        self.assertIs(type(result.dof), float)
        self.assertAlmostEqual(result.prob_below(-1), 0.25, delta=1e-15)        # Cauchy: 1/2 - atan(1) / pi
        self.assertEqual(Result(0, 1, dof=Fraction(7, 2)).dof, 3.5)
        self.assertEqual(Result(0, 1, dof=3).dof, 3.0)
        self.assertEqual(Result(0, 1, dof=Decimal("Infinity")).dof, math.inf)
        self.assertEqual(Budget().add("a", "B", 1, Decimal("2.5")).rows[0][3], 2.5)
        self.assertEqual(welch_satterthwaite([1], [Decimal("2.5")]), 2.5)
        for bad in (True, False, "3"):
            self.assertEqual(refusal(Result, 0, 1, "", bad), "TypeError: dof is no number")
            self.assertEqual(refusal(Budget().add, "a", "A", 1, bad), "TypeError: dof is no number")
            self.assertEqual(refusal(welch_satterthwaite, [1], [bad]), "TypeError: dof is no number")
        for bad in (0, -1, math.nan, Decimal("0"), Decimal("1e-400"), Fraction(1, 10 ** 400)):
            self.assertEqual(refusal(Result, 0, 1, "", bad), "ValueError: dof must be > 0")
            self.assertEqual(refusal(Budget().add, "a", "A", 1, bad), "ValueError: dof must be > 0")
            self.assertEqual(refusal(welch_satterthwaite, [1], [bad]), "ValueError: dof must be > 0")
        self.assertEqual(refusal(Result, 0, 1, "", 10 ** 400), "ValueError: dof beyond a float")

    def test_two_shapes_next_to_zero(self):
        # I_x(a, a) at x = 1/2 is one half, by the symmetry of the beta distribution.
        for a in (5e-324, 1e-310, 1e-300, 1e-9, 0.5, 3, 30, 100):
            self.assertAlmostEqual(dist.betainc(a, a, 0.5), 0.5, delta=1e-12, msg=a)
        self.assertEqual(dist.betainc(0.5, 0.5, 0.0), 0.0)
        self.assertEqual(dist.betainc(0.5, 0.5, 1.0), 1.0)


class TheWidthOfADrawing(unittest.TestCase):
    def test_the_generator_names_its_width(self):
        text = (ROOT / "figures" / "make_figures.py").read_text(encoding="utf-8")
        self.assertNotIn("the text width of", text)
        self.assertIn("224 pt is the width of the drawing", text)
        self.assertIn("342 pt", text)
        self.assertIn("WIDTH = 224 / 72", text)                                 # the drawings are as they were


if __name__ == "__main__":
    unittest.main()
