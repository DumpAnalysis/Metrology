"""Regression tests after the review of Version 70.

The review found confidence limits in the text that were still rounded toward the estimate, and three sentences
that gave a chance at a rate as it is printed where the chance belongs to the rate before it is rounded. In the
programs it found upper bounds that were too small before any printing: three bounds of Listing 16.4 that came out
of floats, the limits of the package at a level whose tail no float holds, and a bound for no event one float above
the least float that holds it, or infinite where a float holds it. It found that the sort of a list with a NaN was
described as leaving the NaN in place; that an interpolated quantile, a mean and a fitted line stopped at a float
that could not hold an exact result; that the residual scatter of a fitted line was described as that of the
rounded line; that a negative clock error, or an interval of offsets the wrong way round, gave two events an
order; and that the display of a guarded value could show an end on the limit, or two ends as one value.

The references of these tests are their own: logarithms, exponentials and sums of Poisson terms in decimals of 80
digits or more, from the numbers that the inputs are; binomial sums in fractions; and arithmetic in fractions.
"""
import contextlib
import copy
import io
import math
import random
import sys
import unittest
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

from examples import ch06_exchange as exchange
from examples import ch11_leak_rate as leak
from examples import ch16_silence as silence
from examples import ch20_judge as judge
from examples import ch22_record_check as check
from swmetro import stats
from swmetro.record import decide, guard, invalid, missing

ROOT = Path(__file__).resolve().parent.parent
D = Decimal
F = Fraction
INF = math.inf
NAN = math.nan
TOP = sys.float_info.max
TEN = [(30 * i, 30 * i + 30) for i in range(10)]


def expected(name):
    return (ROOT / "expected" / f"{name}.txt").read_text(encoding="utf-8")


def printed(function, *args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        function(*args)
    return out.getvalue()


def refusal(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except Exception as error:                                            # the kind and the words of a refusal
        return f"{type(error).__name__}: {error}"
    return None


def dec(x):
    """A number as the decimal that it is, in the precision of the context."""
    x = Fraction(x)
    return D(x.numerator) / D(x.denominator)


def before(x):
    return math.nextafter(x, -INF)


def poisson(n, mean):
    """P(X <= n) for a Poisson count, in the decimals of the context."""
    term = total = (-mean).exp()
    for k in range(1, n + 1):
        term = term * mean / k
        total += term
    return total


def halved(holds, lo, hi):
    """The number in (lo, hi) from which on holds() is true, between two decimals 1e-70 of it apart."""
    lo, hi = D(lo), D(hi)
    while hi - lo > hi * D("1e-70"):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if holds(mid) else (mid, hi)
    return lo, hi


def binomial(k, n, p):
    """P(X <= k) for k of n trials with the probability p, as the fraction that it is."""
    p = Fraction(p)
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


class Precision(unittest.TestCase):
    def setUp(self):
        context = localcontext()
        context.__enter__().prec = 80
        self.addCleanup(context.__exit__, None, None, None)

    def least(self, rate, chance, tail, note=None):
        """`rate` is the least float at which chance(), which falls as its argument grows, is `tail` or less."""
        self.assertLessEqual(chance(dec(rate)), tail, note)
        self.assertGreater(chance(dec(before(rate))), tail, note)


class LimitsOfTheText(Precision):
    """The first table of the review: seven limits that the text rounded toward the estimate."""

    def test_the_growth_after_twenty_minutes(self):
        ts, ms = leak.readings(240, random.Random(11))
        fit = stats.linfit(ts[:21], ms[:21])
        upper = dec(fit[1]) + D("1.645") * dec(fit[3])
        self.assertEqual(f"{upper:.10f}", "0.2938976506")                       # the review's working limit
        self.assertEqual((stats.upward(upper, 4), stats.upward(upper, 2)), ("0.2939", "0.30"))
        self.assertLess(D("0.29"), upper)                                       # the 0.29 of Version 70 lay below it
        self.assertIn(" 20 min  0.1039  0.1155  0.2939  0.3800  no\n", expected("ch11_leak_rate"))

    def test_the_bounds_of_chapter_16(self):
        log20 = -(1 - dec(0.95)).ln()
        pairs = 2 * log20 / D("32.346")
        self.assertEqual(f"{pairs:.10f}", "0.1852304627")                       # the review's
        one_in_30 = halved(lambda m: poisson(1, m) <= 1 - dec(0.95), 0, 60)[1] / 30
        one_in_20 = one_in_30 * 30 / 20
        self.assertEqual(f"{one_in_30:.10f}", "0.1581288173")                   # the review's
        for bound, three, five in ((pairs, "0.186", "0.18524"), (one_in_30, "0.159", "0.15813"),
                                   (one_in_20, "0.238", "0.23720")):
            self.assertEqual((stats.upward(bound, 3), stats.upward(bound, 5)), (three, five))
            self.assertLess(D(three) - D("0.001"), bound)                       # what Version 70 had lay below
            self.assertIn(f"at most {five} msg/s at 95 %\n", expected("ch16_silence"))
        got = (silence.paired_bound([(0, 30)], F("2.346")), stats.rate_upper(1, 30), stats.rate_upper(1, 20))
        for rate, bound in zip(got, (pairs, one_in_30, one_in_20)):
            self.assertTrue(dec(before(rate)) < bound <= dec(rate))

    def test_the_verified_event_times(self):
        log20 = -(1 - dec(0.95)).ln()
        bounds = [log20 / D(t) for t in ("0.749", "0.648", "0.75", "0.65")]
        self.assertEqual([stats.upward(b, 3) for b in bounds], ["4.000", "4.624", "3.995", "4.609"])
        self.assertEqual([stats.upward(b, 2) for b in bounds], ["4.00", "4.63", "4.00", "4.61"])   # the review's
        # Version 70: "4.00 and 4.62 where the stamps gave 3.99 and 4.61". Three of the four lay below.
        self.assertEqual([D(old) < b for old, b in zip(("4.00", "4.62", "3.99", "4.61"), bounds)],
                         [False, True, True, False])
        self.assertEqual(str(bounds[2])[:14], "3.994309698071")                 # the review's digits, cut off
        self.assertEqual(stats.upward(bounds[2], 9), "3.994309699")
        self.assertLess(D("3.994309698"), bounds[2])

    def test_three_hundred_of_three_hundred(self):
        # p ** 300 = 0.05: the lower bound is the 300th root, and a root is no float, so that an end is searched.
        low, high = halved(lambda p: p ** 300 >= D("0.05"), "0.9", 1)
        self.assertEqual(f"{100 * low:.10f}", "99.0063918056")                  # the review's
        self.assertEqual((stats.downward(100 * low, 3), stats.downward(100 * low, 2)), ("99.006", "99.00"))
        self.assertLess(low, D("0.9901"))                                       # the 99.01 % of Version 70 lay above
        got = judge.exact_interval(300, 300, F(9, 10))[0]                       # a tail of 5 % below, exactly
        self.assertTrue(F(got) ** 300 <= F(1, 20) < F(math.nextafter(got, 1.0)) ** 300)
        self.assertEqual(stats.downward(100 * got, 3), "99.006")
        # None in 1000: the 0.30 % of the text is an upper bound, and the chance of none at it is not 5 %.
        top = halved(lambda p: (1 - p) ** 1000 <= D("0.05"), 0, "0.01")[1]
        self.assertEqual(stats.upward(100 * top, 2), "0.30")
        self.assertEqual(f"{100 * (1 - D('0.0030')) ** 1000:.4f}", "4.9563")    # the review's


class ChancesAtRatesAsPrinted(Precision):
    """Section "Printed rates and probabilities use different precision", and the sentences beside it."""

    def test_the_wrap_of_the_counter(self):
        wrap = F(2 ** 32, 60_000_000)
        self.assertEqual(f"{dec(wrap):.8f}", "71.58278827")                     # the review's
        self.assertTrue(F("71.58") < F("71.581") < wrap)
        self.assertEqual(F("71.581") * 60_000_000, 4_294_860_000)               # below 2 ** 32 = 4 294 967 296
        self.assertLess(4_294_860_000, 2 ** 32)
        self.assertEqual(stats.downward(wrap, 4), "71.5827")                    # "at least", on the next page

    def test_a_chance_belongs_to_the_rate_before_it_is_rounded(self):
        single = -(1 - dec(0.95)).ln() / 30
        ninety = 2 * -(1 - dec(0.95)).ln() / 90
        rows = ((single, D("0.100"), D("32.346"), "19.9", "19.8434"),
                (ninety, D("0.067"), D(60), "13.6", "13.3989"))
        for rate, shown, covered, at_the_bound, as_printed in rows:
            self.assertEqual(stats.upward(rate, 3), f"{shown}")
            self.assertEqual(f"{100 * (-rate / 2 * covered).exp():.1f}", at_the_bound)
            self.assertEqual(f"{100 * (-shown / 2 * covered).exp():.4f}", as_printed)       # the review's
        # One message or none from pairs 10 s apart in a window of 30 s: the text gave the chance at 0.150.
        twice = -(1 - dec(0.95)).ln() / 20

        def one_or_none(rate):
            mean = rate / 2 * 20
            return (-2 * mean).exp() * (1 + mean)
        self.assertEqual((f"{100 * one_or_none(twice):.1f}", f"{100 * one_or_none(D('0.14979')):.1f}",
                          f"{100 * one_or_none(D('0.150')):.1f}"), ("12.5", "12.5", "12.4"))
        self.assertEqual(stats.upward(twice, 5), "0.14979")

    def test_the_divided_risk(self):
        top = judge.exact_interval(2, 20)[1]
        covered = 60 - 20 * dec(top)
        self.assertEqual(f"{covered:.8f}", "53.66034572")                       # the review's
        bound = 2 * -(dec(0.025)).ln() / covered
        self.assertEqual(stats.upward(bound, 5), "0.13749")
        self.assertEqual(f"{2 * D('3.69') / D('53.66'):.7f}", "0.1375326")      # the review's: not the bound
        self.assertIn("    at most 0.13749 msg/s at 95 %\n", expected("ch16_silence"))
        half_minute = -(1 - dec(0.95)).ln() / 30
        self.assertEqual(f"{half_minute:.11f}", "0.09985774245")                # the review's
        self.assertEqual(stats.upward(half_minute, 5), "0.09986")


class BoundsOfTheListing(Precision):
    """Section "Ordinary 95 % confidence": three bounds of Listing 16.4 that were floats too low."""

    def test_the_three_calls_of_the_review(self):
        t = 2.99573227355399
        got = silence.searched_bound(t, 1)
        self.assertEqual((got, stats.upward(got, 5)), (math.nextafter(1.0, 2.0), "1.00001"))   # Version 70: 1.0
        self.assertEqual(f"{-(1 - dec(0.95)).ln() / dec(t):.18f}", "1.000000000000000046")      # the review's
        t = 4.743864518390577
        got = silence.paired_upper(1, [(0, t)], t)
        self.assertEqual((got, stats.upward(got, 5)), (math.nextafter(1.0, 2.0), "1.00001"))   # Version 70: 1.0
        # A request brings one message at most: exp(-m) (1 + m) for the requests, twice their rate for messages.
        mean = halved(lambda m: poisson(1, m) <= 1 - dec(0.95), 0, 60)[1]
        self.assertEqual(f"{mean / dec(t) / 2 * 2 / 2:.18f}"[:3], "0.5")
        self.assertEqual(f"{mean / (2 * dec(t)) * 2:.18f}", "1.000000000000000065")             # the review's
        t = F("0.0299573227355399")
        got = silence.paired_bound([(0, t)], t)
        self.assertEqual(stats.upward(got, 5), "100.00001")                                     # Version 70: 100.00000
        self.assertEqual(f"{-(1 - dec(0.95)).ln() / dec(t):.16f}", "100.0000000000000035")      # the review's
        self.assertEqual(got, math.nextafter(100.0, 200.0))

    def test_a_level_next_to_one_as_a_fraction(self):
        level = 1 - F(1, 10 ** 20)
        want = "4.605170185988092"
        self.assertEqual(repr(math.log(1e20) / 10), want)
        for got in (silence.searched_bound(10, 1, level), stats.quiet_bound(10, level),      # Version 70: a domain error
                    silence.searched_bound(10, 1, D(1) - D("1e-20"))):
            self.assertEqual(repr(got), want)

    def test_one_window_is_the_bound_of_the_package(self):
        rng = random.Random(7001)
        for _ in range(60):
            t, level = rng.uniform(0.01, 500), rng.choice((0.5, 0.9, 0.95, 0.999, 1e-6))
            self.assertEqual(silence.searched_bound(t, 1, level), stats.quiet_bound(t, level))

    def test_searched_windows(self):
        rng = random.Random(7002)
        cases = [(30, 10, 0.95), (30, 1000, 1e-300), (30, 2, 1 - 2.0 ** -53)]
        cases += [(rng.uniform(0.5, 90), rng.randint(2, 40), rng.choice((0.5, 0.9, 0.95, 0.999))) for _ in range(25)]
        for t, windows, level in cases:
            got = silence.searched_bound(t, windows, level)

            def all_hold(rate):                                                 # every window holds a message
                return (1 - (-rate * dec(t)).exp()) ** windows
            self.assertGreaterEqual(all_hold(dec(got)), dec(level), (t, windows, level))
            self.assertLess(all_hold(dec(before(got))), dec(level), (t, windows, level))
        self.assertEqual(stats.upward(silence.searched_bound(30, 10), 5), "0.17585")

    def test_messages_in_pairs(self):
        rng = random.Random(7003)
        cases = [(1, [(0, 30)], 2.346, 0.95), (1, [(0, 30)], 60, 0.95), (1, [(0, 20), (10, 30)], 10, 0.95),
                 (3, [(0, 30)], 10, 1e-6), (40, [(0, 30)], 10, 0.95)]
        cases += [(rng.randint(1, 9), [(0, rng.uniform(5, 60))], rng.uniform(0.1, 80),
                   rng.choice((0.5, 0.9, 0.95, 0.999))) for _ in range(20)]
        for n, spans, delta, level in cases:
            got = silence.paired_upper(n, spans, delta, level)
            two, one = (dec(x) for x in silence.by_messages(spans, delta))

            def at_most_n(rate):
                # Requests that bring two messages and requests that bring one, counted apart: the chance of
                # 2 j + i <= n as a sum of products of their Poisson terms.
                a, b = rate / 2 * two, rate / 2 * one
                total, first = D(0), (-a).exp()
                for j in range(n // 2 + 1):
                    total += first * poisson(n - 2 * j, b)
                    first = first * a / (j + 1)
                return total
            self.least(got, at_most_n, 1 - dec(level), (n, spans, delta, level))
        text = expected("ch16_silence")
        for line in ("    at most 0.21000 msg/s at 95 %\n", "    at most 0.20566 msg/s at 95 %\n"):
            self.assertIn(line, text)

    def test_exposures_are_the_numbers_that_they_are(self):
        # A length rounded upward to a float gives a bound that is too low, and the three lengths are fractions.
        t = F("0.0299573227355399")
        self.assertEqual(silence.reach([(0, t)], t), 2 * t)
        self.assertNotEqual(float(2 * t), 2 * t)
        self.assertEqual(silence.by_messages([(0, 30)], F("2.346")), (F("27.654"), F("4.692")))
        self.assertEqual(silence.mean_reach([(0, 30)], [(0.5, 10), (0.5, 50)]), 50)
        self.assertEqual(silence.mean_reach([(0, 30)], [(0.9, 1), (0.1, 291)]), F(0.9) * 31 + F(0.1) * 60)
        self.assertEqual(silence.kept_starts([(0, 30)], 10, 0.25, "message"), F(55, 4))
        self.assertEqual(silence.kept_starts([(0, 30)], 10, 0.25, "request"), 10)
        # A message rate is twice a rate of requests: one search on half the exposure, and no doubled float.
        for spans, delta in (([(0, 30)], 10), ([(0, 30)], 2.346), ([(0, 20), (40, 47.5)], 0.3)):
            got = silence.paired_bound(spans, delta)
            covered = dec(silence.reach(spans, delta))
            self.least(got, lambda rate: (-rate / 2 * covered).exp(), 1 - dec(0.95))

    def test_the_two_searched_bounds(self):
        # A gap somewhere, and at least one empty window of ten for pairs: sums with signs that alternate.
        def gap(rate, total, length):
            out = D(0)
            for k in range(1, int(total // length) + 1):
                m = rate * (total - k * length)
                term = m ** (k - 1) * (m / k + 1) / math.factorial(k - 1) * (-k * rate * length).exp()
                out += term if k % 2 else -term
            return out
        for total, length, level in ((300, 30, 0.95), (300, 30, 1e-12), (300, 30, 1 - 1e-12), (100, 7.5, 0.9)):
            got = silence.gap_bound(total, length, level)
            self.least(got, lambda rate: gap(rate, D(total), dec(length)), 1 - dec(level), (total, length, level))
        terms = silence.selections(TEN, 10)
        self.assertEqual(len(terms), 1023)

        def some_empty(rate):
            return sum(sign * (-rate / 2 * dec(a)).exp() for sign, a in terms)
        for level in (0.95, 0.5, 1e-6):
            got = silence.searched_pairs(TEN, 10, level)
            self.least(got, some_empty, 1 - dec(level), level)
        text = expected("ch16_silence")
        self.assertIn("    at most 0.23854 msg/s at 95 %\n", text)
        self.assertIn("    at most 0.26314 msg/s at 95 %\n", text)
        self.assertFalse(hasattr(silence, "rate_at"))                           # the search in floats is gone
        # The chance that is printed beside a bound is still a sum of floats, from the same terms.
        self.assertAlmostEqual(silence.gap_somewhere(0.2, 300, 30), float(gap(D("0.2"), D(300), D(30))), places=12)
        for bad in ((0.1, 300, 0), (0.1, 30, 300), (0.1, 300, 10), (-0.1, 300, 30), (NAN, 300, 30), (0.1, INF, 30)):
            self.assertIn("ValueError", refusal(silence.gap_somewhere, *bad), bad)


class LevelsThatNoFloatHolds(Precision):
    """Section "Extreme confidence levels accepted by the package"."""

    LEVEL = 1 - F(1, 10 ** 400)

    def setUp(self):
        super().setUp()
        self.tail = D(1).scaleb(-400)

    def test_the_three_rows_of_the_review(self):
        alpha = self.tail
        one_sided = halved(lambda r: (1 + r).ln() - r <= alpha.ln(), 1, 2000)[1]
        two_sided = halved(lambda r: (1 + r).ln() - r <= (alpha / 2).ln(), 1, 2000)[1]
        none = -(alpha / 2).ln()
        self.assertEqual([f"{x:.6f}" for x in (one_sided, two_sided, none)],
                         ["927.868004", "928.561898", "921.727184"])           # the review's model endpoints
        got = (stats.rate_upper(1, 1, self.LEVEL), stats.rate_limits(1, 1, self.LEVEL)[1],
               stats.rate_limits(0, 1, self.LEVEL)[1])
        # Version 70: 751.751664, 751.751664 and 745.124878, where a tail in floats becomes zero.
        for rate, model in zip(got, (one_sided, two_sided, none)):
            self.assertTrue(dec(before(rate)) < model <= dec(rate), (rate, model))
        # The lower limit for one event, 5e-401, is below every float above zero.
        # It is -ln(1 - 5e-401), which is above 5e-401 and below 1e-400.
        self.assertEqual(stats.rate_limits(1, 1, self.LEVEL)[0], 0.0)
        self.assertLess(F(1, 10 ** 400), F(5e-324))

    def test_levels_at_both_ends(self):
        for n, exposure, level in ((1, 30, 1e-300), (3, 30, 5e-324), (2, 7, 1 - F(1, 10 ** 60)),
                                   (0, 30, 1 - F(1, 10 ** 2000)), (5, F(1, 3), F(1, 10 ** 50))):
            with localcontext() as context:
                context.prec = 2100
                tail = 1 - dec(level)
                got = stats.rate_upper(n, exposure, level)
                self.assertLessEqual(poisson(n, dec(got) * dec(exposure)), tail, (n, level))
                self.assertGreater(poisson(n, dec(before(got)) * dec(exposure)), tail, (n, level))

    def test_what_is_not_decided_is_refused(self):
        asked = []

        def never(x, digits):
            asked.append(digits)
        self.assertEqual(refusal(stats.least, never), "ValueError: not decided in 5120 digits")
        self.assertEqual(asked, [40 * 2 ** k for k in range(8)])
        self.assertEqual(stats.DEEPEST, 5120)
        # A test that is false at every float has no float for an answer.
        self.assertEqual(refusal(stats.least, lambda x, digits: False), "ValueError: no rate in a float")
        self.assertEqual(stats.least(lambda x, digits: True), 5e-324)
        self.assertEqual(stats.least(lambda x, digits: x >= 3), 3.0)
        self.assertEqual(stats.least(lambda x, digits: x > 3), math.nextafter(3.0, 4.0))
        self.assertEqual(stats.least(lambda x, digits: x >= TOP), TOP)


class TheLeastFloat(Precision):
    """Section "The least floating-point upper bound"."""

    NEAR = D("0.693147180559945309417232121458176568075500134361")              # a little above ln 2

    def test_a_bound_just_below_a_float(self):
        self.assertLess(D(2).ln(), self.NEAR)
        self.assertLess(self.NEAR - D(2).ln(), D("1e-47"))
        bound = D(2).ln() / self.NEAR
        self.assertTrue(dec(before(1.0)) < bound < 1)
        self.assertEqual(stats.quiet_bound(self.NEAR, 0.5), 1.0)                # Version 70: 1.0000000000000002
        self.assertEqual(stats.upward(stats.quiet_bound(self.NEAR, 0.5), 5), "1.00000")
        # The float next to ln 2 is the other case: the bound is above 1, by 3.3e-17.
        self.assertEqual(stats.quiet_bound(0.6931471805599453, 0.5), math.nextafter(1.0, 2.0))

    def test_the_largest_float(self):
        t = F(self.NEAR) / F(TOP)
        bound = D(2).ln() / dec(t)
        self.assertTrue(dec(before(TOP)) < bound < dec(TOP))
        self.assertEqual(stats.quiet_bound(t, 0.5), TOP)                        # Version 70: inf
        self.assertEqual(refusal(stats.quiet_bound, t / 2, 0.5), "ValueError: no rate in a float")
        self.assertEqual(stats.quiet_bound(1e308, 1e-16), 5e-324)

    def test_every_bound_is_the_least_float(self):
        rng = random.Random(7004)
        for _ in range(150):
            exposure = rng.choice((rng.uniform(0.001, 1000), F(rng.randint(1, 10 ** 6), rng.randint(1, 999)),
                                   D(rng.randint(1, 10 ** 9)).scaleb(-rng.randint(0, 9))))
            level = rng.choice((0.5, 0.9, 0.95, 0.975, 1e-9, 1 - 1e-12, F(2, 3), D("0.999")))
            got = stats.quiet_bound(exposure, level)
            self.least(got, lambda rate: (-rate * dec(exposure)).exp(), 1 - dec(level), (exposure, level))
        for n in (1, 2, 5, 17, 60):
            for exposure in (1, 30, 0.75, F(1, 3)):
                for level in (0.5, 0.95, 0.999, F(1, 4)):
                    tail = 1 - dec(level)
                    self.least(stats.rate_upper(n, exposure, level), lambda r: poisson(n, r * dec(exposure)), tail)
                    low, high = stats.rate_limits(n, exposure, level)
                    self.least(high, lambda r: poisson(n, r * dec(exposure)), tail / 2)
                    # The lower limit is the greatest float at which so high a count has half the tail or less.
                    self.assertLessEqual(1 - poisson(n - 1, dec(low) * dec(exposure)), tail / 2)
                    above = math.nextafter(low, INF)
                    self.assertGreater(1 - poisson(n - 1, dec(above) * dec(exposure)), tail / 2)

    def test_the_enclosures(self):
        rng = random.Random(7005)
        for _ in range(200):
            n, mean = rng.randint(0, 40), rng.choice((rng.uniform(0, 80), F(rng.randint(0, 999), 7), 1e-300, 3000))
            for digits in (40, 80):
                with localcontext() as context:
                    context.prec = 2000
                    lo, hi = stats.poisson_within(n, mean, digits)
                    true = poisson(n, dec(mean)) if mean else D(1)
                    self.assertTrue(dec(lo) <= true <= dec(hi), (n, mean, digits))
                    # Close by some digits less than were carried, or zero and the least power that is kept.
                    self.assertLessEqual(dec(hi) - dec(lo), max(true * D(10) ** (5 - digits), D(10) ** (-10 * digits)))
        self.assertEqual(stats.poisson_within(3, 0), (1, 1))
        self.assertEqual(stats.poisson_within(0, 10 ** 30), (0, F(1, 10 ** 400)))  # too small to keep: 0 to 1e-400
        lo, hi = stats.sum_within([(F(1, 3), 2), (-2, F(1, 2)), (5, 0)])
        true = D(1) / 3 * D(-2).exp() - 2 * D("-0.5").exp() + 5
        self.assertTrue(dec(lo) <= true <= dec(hi) and dec(hi) - dec(lo) < D("1e-36"))
        self.assertEqual([stats.not_above(ends, F(1, 2)) for ends in ((F(1, 4), F(1, 2)), (F(51, 100), 1), (0, 1))],
                         [True, False, None])
        for bad in ((-1, 2.0), (2.5, 2.0), (True, 2.0), (2, -1.0), (2, INF), (2, NAN), (2, True)):
            self.assertIsNotNone(refusal(stats.poisson_within, *bad), bad)


class BinomialLimits(Precision):
    """Beyond the review: the exact limits of Listing 20.2 were floats from a search in floats as well."""

    def test_each_limit_is_the_float_next_to_the_limit_of_the_model(self):
        cases = [(2, 20, 0.95), (0, 25, 0.95), (40, 40, 0.95), (33, 60, 0.95), (1, 2, 0.5), (7, 9, 0.999),
                 (99, 100, 0.95), (0, 1, 0.9), (3, 50, F(9, 10)), (17, 40, D("0.99"))]
        for k, n, level in cases:
            a = (1 - F(level)) / 2
            low, high = judge.exact_interval(k, n, level)
            if k < n:                                                           # a count this low: a or less
                self.assertTrue(binomial(k, n, high) <= a < binomial(k, n, before(high)), (k, n, level))
            else:
                self.assertEqual(high, 1.0)
            if k:                                                               # a count this high: a or less
                above = math.nextafter(low, 1.0)
                self.assertTrue(1 - binomial(k - 1, n, low) <= a < 1 - binomial(k - 1, n, above), (k, n, level))
            else:
                self.assertEqual(low, 0.0)

    def test_a_chance_that_is_equal_to_the_level(self):
        # One trial at a level of one half: the chance of no success is 1 - p, and it is 1/4 at the float 0.75.
        self.assertEqual(judge.exact_interval(0, 1, 0.5), (0.0, 0.75))
        self.assertEqual(judge.exact_interval(1, 1, 0.5), (0.25, 1.0))          # Version 70: 0.24999999999999994
        self.assertEqual(stats.binomial_within(0, 1, 0.75, 80), (F(1, 4), F(1, 4)))
        lo, hi = stats.binomial_within(0, 1, 0.75, 40)
        self.assertTrue(lo <= F(1, 4) <= hi)
        self.assertEqual(stats.binomial_within(7, 9, F(2, 3), 41), (binomial(7, 9, F(2, 3)),) * 2)

    def test_the_enclosure_and_its_refusals(self):
        rng = random.Random(7006)
        for _ in range(150):
            n = rng.randint(1, 60)
            k, p = rng.randint(0, n), rng.choice((rng.random(), F(rng.randint(0, 50), 50), 5e-324, 1.0))
            lo, hi = stats.binomial_within(k, n, p)
            true = binomial(k, n, p)
            self.assertTrue(lo <= true <= hi, (k, n, p))
            self.assertLessEqual(hi - lo, max(true, F(1, 10 ** 400)) * F(1, 10 ** 30), (k, n, p))
        for bad in ((1, 2, 1.5), (1, 2, -0.1), (1, 2, NAN), (-1, 2, 0.5), (1, 2.5, 0.5), (True, 2, 0.5), (1, 2, True)):
            self.assertIsNotNone(refusal(stats.binomial_within, *bad), bad)

    def test_the_lines_of_the_book(self):
        self.assertEqual(stats.enclosed(*judge.exact_interval(2, 20), 3), ("0.012", "0.317"))
        self.assertEqual(stats.upward(100 * judge.exact_interval(2, 20)[1], 3), "31.699")
        self.assertIn("  at most 31.699 % of 10 s, at 97.5 %:\n", expected("ch16_silence"))
        for line in ("0.000 to 0.138", "0.911 to 1.000"):
            self.assertIn(line, expected("ch20_judge"))
        self.assertIn("binomial_within()", " ".join(judge.exact_interval.__doc__.split()))


class ValuesWithoutARank(Precision):
    """Section "Sorting, interpolation and the stated input domain"."""

    def test_where_a_sort_leaves_a_nan(self):
        # The review: [0, 1, 2, nan] under CPython 3.12 and [0, nan, 1, 2] under 3.13. Neither is where it stood.
        got = [("nan" if x != x else x) for x in sorted([2, 0, NAN, 1])]
        self.assertIn(got, ([0, 1, 2, "nan"], [0, "nan", 1, 2]))
        self.assertNotEqual(got.index("nan"), 2)                                # it stood third
        text = " ".join(stats.ordered.__doc__.split())
        self.assertIn("depends on the order of the list and on the version of Python", text)
        self.assertNotIn("stays where it stood", text)                          # the words of Version 70
        for xs in ([2, 0, NAN, 1], [NAN, 1], [1, D("NaN")]):
            self.assertEqual(refusal(stats.ordered, xs), "ValueError: a NaN has no rank")

    def test_an_infinity_between_two_ranks(self):
        self.assertEqual(stats.quantile([100, INF], 0.5), INF)                  # Version 70: an OverflowError
        self.assertEqual(stats.quantile([100, INF], 0), 100)
        self.assertEqual(stats.quantile([100, INF], F(1, 10 ** 9)), INF)
        self.assertEqual(stats.quantile([-INF, 100], 0.25), -INF)
        self.assertEqual(stats.quantile([-INF, 100], 1), 100)
        self.assertEqual(stats.quantile([1, 2, INF, INF], 0.9), INF)            # between two that are equal
        self.assertEqual(stats.quantile([100, D("Infinity")], 0.5, exact=True), D("Infinity"))
        self.assertEqual(stats.quantile([-INF, 3, INF], 0.5), 3)
        for xs, q in (([-INF, INF], 0.5), ([D("-Infinity"), INF], 0.01)):
            self.assertEqual(refusal(stats.quantile, xs, q), "ValueError: no value between the infinities")
        # Selection by rank needs none of this.
        self.assertEqual(stats.ecdf_quantile([100, INF], 0.5), 100)
        self.assertEqual(stats.ecdf_quantile([100, INF], 0.51), INF)
        self.assertEqual(stats.order_stat([-INF, 100, INF], 0.5), 100)
        self.assertIn("Between a finite value and an infinity it is that infinity", " ".join(stats.quantile.__doc__.split()))


class ResultsThatNoFloatHolds(Precision):
    """Section "Exact finite results can overflow an intermediate float"."""

    H = 10 ** 310

    def test_the_three_calls_of_the_review(self):
        h = self.H
        self.assertEqual(refusal(float, h), "OverflowError: int too large to convert to float")
        median = stats.quantile([h, h + 2], 0.5)                                # Version 70: an OverflowError
        self.assertEqual((median, type(median)), (h + 1, int))
        mean = stats.mean([h, h + 1])
        self.assertEqual((mean, type(mean)), (h + F(1, 2), F))
        fit = stats.linfit([0, 1, 2], [h, h + 1, h + 2])
        self.assertEqual(fit, (h, 1, 0.0, 0.0, 0.0))
        self.assertEqual([type(v) for v in fit[:2]], [int, float])

    def test_beside_them(self):
        h = self.H
        self.assertEqual(stats.quantile([h, h + 1], 0.5), h + F(1, 2))
        self.assertEqual(stats.quantile([1.5, h], 0.5), (h + F(3, 2)) / 2)
        self.assertEqual(stats.mean([h, h]), h)
        self.assertEqual(stats.mean([h, INF]), INF)                             # Version 70: an OverflowError
        self.assertEqual(stats.mean([h, -INF, 2.5]), -INF)
        self.assertTrue(math.isnan(stats.mean([h, INF, -INF])))
        self.assertTrue(math.isnan(stats.mean([1, NAN])))
        self.assertEqual(stats.mean([1, D("Infinity")]), INF)
        self.assertEqual(stats.mean([1, 2, 4]), 7 / 3)
        # The intercept alone is beyond the floats.
        fit = stats.linfit([0, 1, 2], [h, h + F(1, 2), h + 1])
        self.assertEqual(fit[:2], (h, 0.5))
        # What a float holds comes back as before.
        self.assertEqual(stats.quantile([100, 101], 0.95), 100.95)
        self.assertEqual(stats.mean([2 ** 60, 2 ** 60 + 2]), 2 ** 60 + 1)


class TheScatterOfAFittedLine(Precision):
    """Section "Which line supplies the reported residual scatter"."""

    def test_s_is_that_of_the_line_of_least_squares(self):
        xs, ys = [0, 1, 2], [10 ** 12, 10 ** 12 + 1, 10 ** 12 + 3]
        a, b, ua, ub, s = stats.linfit(xs, ys)
        # In fractions: slope 3/2, intercept 1e12 - 1/6, residuals 1/6, -1/3 and 1/6, one degree of freedom.
        slope, intercept = F(3, 2), 10 ** 12 - F(1, 6)
        rss = sum((y - intercept - slope * x) ** 2 for x, y in zip(xs, ys))
        self.assertEqual(rss, F(1, 6))
        self.assertEqual(f"{s:.11f}", f"{dec(rss).sqrt():.11f}")
        self.assertEqual(f"{s:.11f}", "0.40824829046")                          # the review's
        self.assertEqual((type(a), b), (float, 1.5))
        moved = sum((y - F(a) - F(b) * x) ** 2 for x, y in zip(xs, ys))
        self.assertEqual(f"{dec(moved).sqrt():.11f}", "0.40824829655")          # the review's: the rounded line
        self.assertLessEqual(3 * (F(a) - intercept) ** 2 * 10 ** 6, rss)        # within a thousandth of u
        text = " ".join(stats.linfit.__doc__.split())
        self.assertIn("those of the line of least squares itself", text)

    def test_decimals_and_floats_written_as_decimals(self):
        fit = stats.linfit([0, 1, 2], [D("0.1"), D("0.2"), D("0.3")])
        self.assertEqual(fit, (F(1, 10), F(1, 10), 0.0, 0.0, 0.0))
        self.assertEqual([type(v) for v in fit[:2]], [F, F])
        a, b, _, _, s = stats.linfit([0, 1, 2], [0.1, 0.2, 0.3])
        # As floats the three are binary numbers off their line: 0.1 + 0.3 is not twice 0.2.
        self.assertNotEqual(F(0.1) + F(0.3), 2 * F(0.2))
        self.assertGreater(s, 0)
        self.assertEqual(type(a), F)
        self.assertNotEqual(a, F(1, 10))
        self.assertEqual(a, (5 * F(0.1) + 2 * F(0.2) - F(0.3)) / 6)


class AnOrderFromABoundThatIsNone(Precision):
    """Section "An invalid clock-error bound establishes a false order"."""

    def test_the_two_cases_of_the_review(self):
        # Version 70: gap(0, 0, (-1, 1), err=-1) was (1, -1), and holds() gave the order 50 seconds.
        self.assertEqual(refusal(exchange.gap, 0, 0, (-1, 1), err=-1), "ValueError: err must be >= 0")
        self.assertEqual(refusal(exchange.holds, 0, 0, (-1, 1), err=-1), "ValueError: err must be >= 0")
        self.assertEqual(refusal(exchange.gap, 0, 0, (1, -1)), "ValueError: no offset in the span")
        self.assertEqual(refusal(exchange.holds, 0, 0, (1, -1)), "ValueError: no offset in the span")
        self.assertEqual(exchange.gap(0, 0, (-1, 1)), (-1, 1))
        self.assertEqual(exchange.holds(0, 0, (-1, 1)), 0)                      # either event can come first
        self.assertEqual(exchange.gap(0, 0, (1, 1)), (-1, -1))                  # one offset is an interval
        self.assertIn("err must be >= 0", refusal(exchange.offset, 0, 1, 2, 3, -0.5))

    def test_the_transcript_is_what_it_was(self):
        self.assertEqual(printed(exchange.main), expected("ch06_exchange"))
        low, high = exchange.gap(3, 0, (-6, F("-3.5")))
        self.assertEqual((low, high), (F(1, 2), 3))
        self.assertEqual(exchange.holds(3, 0, (-6, F("-3.5"))), 25)


class TheSideOfTheLimit(Precision):
    """Section "The display loses which side of the limit a bound occupies"."""

    def setUp(self):
        super().setUp()
        self.rec = check.load("measurement_record.json")

    def one(self, value, u, dof, risk, digits="all", step=None):
        """The published record with one component, whose uncertainty is that of the result."""
        out = copy.deepcopy(self.rec)
        part = copy.deepcopy(out["components"][1])
        part["u"], part["dof"] = u, dof
        out["components"] = [part]
        result = out["result"]
        result["value"], result["u"], result["dof"] = value, u, dof
        result["significant_digits"] = digits
        if step is not None:
            result["value_step"] = step
        out["decision"]["risk"] = risk
        return out

    def shown(self, rec):
        return printed(check.near, "a title", rec, rec["result"]["value"]).splitlines()[-1]

    def test_a_range_without_an_upper_end(self):
        rec = self.one(D("199.981"), D(1), D("0.0001"), D("0.49"))
        self.assertEqual((missing(rec), invalid(rec)), ([], []))
        self.assertEqual(decide(rec), ("undecided", []))
        self.assertEqual(guard(rec)[0], (D("200.001"), D("Infinity")))
        self.assertEqual(self.shown(rec), "  estimate + band: 200.001 or more")  # Version 70: "200.00 or more"
        # Away from the limit two places serve.
        rec = self.one(D("210.981"), D(1), D("0.0001"), D("0.49"))
        self.assertEqual(self.shown(rec), "  estimate + band: 211.00 or more")

    def test_two_ends_that_forty_digits_took_for_one(self):
        rec = self.one(D(200), D(0), D(1), D("0.25"), 2, D("1e-45"))
        self.assertEqual((missing(rec), invalid(rec)), ([], []))
        self.assertEqual(decide(rec), ("unresolved", ["rounding decides: wants working values"]))
        lo, hi = guard(rec)[0]
        self.assertEqual((F(lo), F(hi)), (200 - F(5, 10 ** 46), 200 + F(5, 10 ** 46)))   # Version 70: 200 and 200
        line = self.shown(rec)
        self.assertEqual(line, "  estimate + band: 199." + "9" * 45 + "5 to 200." + "0" * 45 + "5")
        # A decimal of forty digits or fewer is what it was.
        self.assertEqual(str(guard(check.load("measurement_record.json"))[0][0]), str(guard(self.rec)[0][0]))

    def test_one_value_next_to_the_limit(self):
        # A working value: one number, to the nearest, and with more places where two would show the limit.
        rec = self.one(D("200.004"), D(0), D(1), D("0.25"))
        self.assertEqual(guard(rec)[0], (D("200.004"),) * 2)
        self.assertEqual(self.shown(rec), "  estimate + band: 200.004")          # to two places: 200.00, the limit
        rec = self.one(D("199.9996"), D(0), D(1), D("0.25"))
        self.assertEqual(self.shown(rec), "  estimate + band: 199.9996")
        rec = self.one(D("200"), D(0), D(1), D("0.25"))
        self.assertEqual(self.shown(rec), "  estimate + band: 200.00")           # at the limit
        rec = self.one(D("194.6"), D(0), D(1), D("0.25"))
        self.assertEqual(self.shown(rec), "  estimate + band: 194.60")

    def test_the_transcript_is_what_it_was(self):
        text = expected("ch22_record_check")
        for line in ("  estimate + band: 199.86 to 200.12\n", "  estimate + band: 200.06 to 200.32\n",
                     "  estimate + band: 199.95\n"):
            self.assertIn(line, text)


class WhatTheBookPrints(Precision):
    """Beyond the review: two ends of the tables of Chapter 17, and the chances that Chapter 16 gives at a bound."""

    def test_two_ends_of_the_tables_of_chapter_17(self):
        text = expected("ch17_accuracy")
        for line in ("  A 23, 7; B 23, 12: 0.043128 to 0.049510\n", "  A 23, 7; B 22, 13: 0.043828 to 0.050230\n",
                     "  A 23, 7; B 13, 22: 0.050229 to 0.053854\n", "  A 23, 7; B 14, 21: 0.049509 to 0.053854\n"):
            self.assertIn(line, text)
        # An upper end of 0.049510 or of 0.050230, at four places: 0.0496 and 0.0503. Version 70 had 0.0495 and 0.0502.
        self.assertEqual((stats.upward(D("0.049510") - D("1e-6"), 4), stats.upward(D("0.050230") - D("1e-6"), 4)),
                         ("0.0496", "0.0503"))
        self.assertEqual((stats.downward(D("0.043128"), 4), stats.downward(D("0.049509"), 4)), ("0.0431", "0.0495"))

    def test_chances_that_do_not_depend_on_the_last_digits(self):
        single = -(1 - dec(0.95)).ln() / 30
        self.assertEqual({f"{100 * (-r / 2 * 50).exp():.2f}" for r in (single, D("0.09986"))}, {"8.24"})
        searched = dec(silence.searched_bound(30, 10))
        terms = silence.selections(TEN, 10)
        for rate in (searched, D("0.17585")):
            some = sum(sign * (-rate / 2 * dec(a)).exp() for sign, a in terms)
            self.assertEqual(f"{100 * some:.2f}", "25.17")
            self.assertEqual(f"{100 * silence.gap_somewhere(float(rate), 300, 30):.2f}", "22.48")


if __name__ == "__main__":
    unittest.main()
