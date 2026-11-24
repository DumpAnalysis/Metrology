"""Regression tests added after the review of Version 38.

The review found that temperatures() of Listing 18.1 lost its interval when the two class labels were exchanged,
returned negative temperatures for a logit below zero and "T up to 0.00" for a logit of zero; that over() of
Listing 10.1 checked the durations and not the limit, so that a limit that is not a number put every request
within it; that the bound on the increase of a counter has to be taken for the longest gap that a rounded record
of the times allows; that Transcript 6.2 printed 0.000 for seven reversals in 50 000; and it proposed two
additions: ten pairs of events under a clock offset that persists, with a figure, and the exact run lengths of
the Shewhart chart of Listing 15.1, for known limits and for the limits of its baseline (the new Listing 15.2).

Each test computes what it expects independently of the code under test: exact binomial limits from binomial
coefficients, the temperatures by trying each one against those limits, the verdict of an objective by trying
both outcomes of every unknown request, the increases of a counter one wrap at a time, the number of reversed
pairs by enumeration of all outcomes and by simulation of the two models, and run lengths from the
complementary error function and from simulated waits.
"""
import contextlib
import io
import math
import random
import unittest
from fractions import Fraction
from pathlib import Path

from examples import ch06_ordering as ordering
from examples import ch08_counter as counter
from examples import ch10_coordinated_omission as omission
from examples import ch15_control_chart as chart
from examples import ch15_run_length as run_length
from examples import ch16_bounds as bounds
from examples import ch18_calibration as cal

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


def tail(k, n, p):
    """P(X >= k) for X binomial(n, p)."""
    return math.fsum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def exact_limits(k, n, level=0.95):
    """Clopper-Pearson limits by bisection on binomial tails, independently of Listing 20.2."""
    alpha = (1 - level) / 2

    def solve(below):
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if below(mid):
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    low = 0.0 if k == 0 else solve(lambda p: tail(k, n, p) < alpha)
    high = 1.0 if k == n else solve(lambda p: 1 - tail(k + 1, n, p) > alpha)
    return low, high


def stated(logit, temp):
    """The probability of the positive label that a logit states at a temperature."""
    z = logit / temp
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    return math.exp(z) / (1 + math.exp(z))


def version_38_temperatures(logit, positive, n):
    """temperatures() as Version 38 printed it, with the limits computed here."""
    p_low, p_high = exact_limits(positive, n)
    if p_high <= 0.5:
        return None
    low, high = 0.0, math.inf
    if p_high < 1:
        low = logit / math.log(p_high / (1 - p_high))
    if p_low > 0.5:
        high = logit / math.log(p_low / (1 - p_low))
    return low, high


GRID = [math.exp(i / 40) for i in range(-80, 281)]            # temperatures from 0.14 to 1097


class TheLabelsExchanged(unittest.TestCase):
    def test_what_version_38_returned(self):
        low, high = version_38_temperatures(40, 99, 100)
        self.assertEqual((f"{low:.2f}", f"{high:.2f}"), ("4.83", "14.01"))
        self.assertIsNone(version_38_temperatures(-40, 1, 100))                  # the same cases, the labels exchanged
        low, high = version_38_temperatures(-40, 99, 100)
        self.assertEqual((f"{low:.2f}", f"{high:.2f}"), ("-4.83", "-14.01"))
        self.assertEqual(version_38_temperatures(0, 99, 100), (0.0, 0.0))

    def test_the_table_of_the_review(self):
        for logit, positive in ((40, 99), (-40, 1)):
            low, high = cal.temperatures(logit, positive, 100)
            self.assertEqual((f"{low:.2f}", f"{high:.2f}"), ("4.83", "14.01"))
        self.assertIsNone(cal.temperatures(-40, 99, 100))
        self.assertIsNone(cal.temperatures(0, 99, 100))
        self.assertEqual(cal.temperatures(0, 50, 100), (0.0, math.inf))          # a half is within the interval
        self.assertEqual(cal.temperatures(40, 1, 1), (0.0, math.inf))            # one trial: 0.025 to 1
        self.assertEqual(cal.worded(cal.temperatures(40, 1, 1)), "any T above zero")
        self.assertEqual(cal.worded(cal.temperatures(-40, 1, 100)), "4.83 to 14.01")
        self.assertEqual(cal.worded(cal.temperatures(-40, 99, 100)), "no T above zero")
        self.assertEqual(cal.worded(cal.temperatures(0, 99, 100)), "no T above zero")

    def test_every_temperature_tried(self):
        # The definition, without the closed form: T belongs when the probability that it states lies within the
        # exact limits of the frequency. Away from the two ends, the interval has to say the same.
        for n in (1, 2, 5, 20, 100):
            for positive in range(n + 1):
                p_low, p_high = exact_limits(positive, n)
                for logit in (40, -40, 3.5, -0.25, 0):
                    ends = cal.temperatures(logit, positive, n)
                    for temp in GRID:
                        belongs = p_low <= stated(logit, temp) <= p_high
                        if ends is None:
                            self.assertFalse(belongs, (logit, positive, n, temp))
                            continue
                        low, high = ends
                        if min(abs(temp - low), abs(temp - high)) < 1e-6 * temp:
                            continue                                             # at an end, rounding decides
                        self.assertEqual(belongs, low < temp < high, (logit, positive, n, temp, ends))

    def test_the_labels_are_names(self):
        for n in (1, 7, 100):
            for positive in range(n + 1):
                for logit in (40, 2.5, 0.125, 0):
                    self.assertEqual(cal.temperatures(logit, positive, n), cal.temperatures(-logit, n - positive, n))

    def test_a_logit_of_zero(self):
        # Every T states a half. All of them belong, or none.
        for n in (1, 4, 30, 100):
            for positive in range(n + 1):
                p_low, p_high = exact_limits(positive, n)
                want = (0.0, math.inf) if p_low <= 0.5 <= p_high else None
                self.assertEqual(cal.temperatures(0, positive, n), want, (positive, n))
                self.assertEqual(cal.temperatures(0.0, positive, n), cal.temperatures(-0.0, positive, n))
        self.assertIsNone(cal.temperatures(0, 61, 100))
        self.assertEqual(cal.temperatures(0, 60, 100), (0.0, math.inf))

    def test_the_fit_lies_within(self):
        data = [(40.0, 1)] * 99 + [(40.0, 0)]
        turned = [(-z, 1 - y) for z, y in data]
        self.assertEqual(cal.fit_temperature(data), cal.fit_temperature(turned))
        for cases, logit, positive in ((data, 40, 99), (turned, -40, 1)):
            low, high = cal.temperatures(logit, positive, 100)
            self.assertTrue(low < cal.fit_temperature(cases) < high)
            self.assertAlmostEqual(cal.fit_temperature(cases), 40 / math.log(99), places=6)

    def test_one_logit(self):
        self.assertAlmostEqual(cal.at_one_logit(-40, 0.01), 40 / math.log(99), places=12)
        self.assertAlmostEqual(cal.at_one_logit(40, 0.99), cal.at_one_logit(-40, 0.01), places=12)
        for logit, p in ((40, 0.3), (-40, 0.7), (0, 0.6), (0, 0.5), (40, 0.5), (-40, 0.5), (40, 1.0), (-40, 0.0),
                         (40, 0.0), (-40, 1.0), (40, 1.2), (-40, -0.1), (40, float("nan"))):
            self.assertEqual(refusal(cal.at_one_logit, logit, p), "no T above zero states this p", (logit, p))
        # Back from the probability that a temperature states. The quotient logit / T stays at 10 or below:
        # beyond 37 a float holds the stated probability as 1, which no temperature states.
        for logit, temps in ((25.0, (2.5, 8.7, 150.0)), (-25.0, (2.5, 8.7, 150.0)), (0.5, (0.3, 1.0, 150.0)),
                             (-3.0, (0.3, 1.0, 8.7))):
            for temp in temps:
                self.assertAlmostEqual(cal.at_one_logit(logit, stated(logit, temp)), temp, places=6)
        self.assertEqual(stated(25.0, 0.3), 1.0)
        self.assertIsNotNone(refusal(cal.at_one_logit, 25.0, stated(25.0, 0.3)))

    def test_a_logit_that_is_no_number(self):
        for logit in (math.inf, -math.inf, float("nan")):
            self.assertEqual(refusal(cal.temperatures, logit, 99, 100), "need a finite logit")

    def test_the_transcript(self):
        text = expected("ch18_calibration")
        lines = text.splitlines()
        self.assertEqual(len(lines), 24)                                         # 23 in Version 38
        low, high = exact_limits(99, 100)
        ends = (40 / math.log(high / (1 - high)), 40 / math.log(low / (1 - low)))
        self.assertEqual(lines[23], f"  1 positive at -40: {ends[0]:.2f} to {ends[1]:.2f}")
        self.assertEqual(lines[16], f"  95 % interval for T: {ends[0]:.2f} to {ends[1]:.2f}")
        self.assertEqual(printed(cal.main), text)


def version_38_over(xs, limit=500.0):
    """over() as Version 38 printed it."""
    xs = list(xs)
    if not xs:
        raise ValueError("no entries")
    if not all(0 <= x < math.inf for x in xs):
        raise ValueError("need durations: finite and not below zero")
    n = sum(x > limit for x in xs)
    return n, 100 * (len(xs) - n) / len(xs)


def decided(total, within, above, need):
    """The verdict on "at least `need` of `total` within the limit" by trying every number of the unknown
    requests that could be within it."""
    unknown = total - within - above
    outcomes = {within + k >= need for k in range(unknown + 1)}
    if outcomes == {True}:
        return "met"
    if outcomes == {False}:
        return "missed"
    return "unresolved"


class TheLimitIsADurationToo(unittest.TestCase):
    def test_the_example_of_the_review(self):
        self.assertEqual(omission.over([100, 600], 500), (1, 50.0))
        self.assertEqual(version_38_over([100, 600], float("nan")), (0, 100.0))  # every request "within" no limit
        self.assertEqual(refusal(omission.over, [100, 600], float("nan")), "need durations: finite and not below zero")

    def test_what_is_a_limit(self):
        for limit in (float("nan"), math.inf, -math.inf, -1, -0.001):
            self.assertIsNotNone(refusal(omission.over, [100, 600], limit), limit)
        self.assertEqual(omission.over([0, 1], 0), (1, 50.0))                    # a limit of zero is a duration
        self.assertEqual(omission.over([100, 600], 600), (0, 100.0))
        self.assertEqual(omission.over(iter([100, 600]), 99.5), (2, 0.0))
        self.assertEqual(refusal(omission.over, [], float("nan")), "no entries")
        self.assertEqual(refusal(omission.over, [float("nan")], 500), "need durations: finite and not below zero")

    def test_decided_before_every_duration_is_known(self):
        # 99.97 % of 5800 within the limit is 5799 or more of them.
        need = math.ceil(Fraction(9997, 10000) * 5800)
        self.assertEqual(need, 5799)
        rows = ((5799, 0, "met"),             # one unknown, and it no longer matters
                (5798, 2, "missed"),          # two known above: no unknown is left
                (5797, 2, "missed"),          # two known above and one unknown
                (5798, 1, "unresolved"),      # the case that the book prints
                (5798, 0, "unresolved"),
                (5800, 0, "met"))
        for within, above, verdict in rows:
            unknown = 5800 - within - above
            self.assertEqual(decided(5800, within, above, need), verdict, (within, above))
            self.assertEqual(bounds.quantile_verdict(5800, 0.9997, within, unknown), verdict, (within, above))
        # The same through over(), on lists in which each unknown request is given either outcome.
        for within, above in ((5799, 0), (5797, 2), (5798, 1)):
            unknown = 5800 - within - above
            seen = set()
            for k in range(unknown + 1):
                xs = [100.0] * (within + k) + [600.0] * (above + unknown - k)
                count, share = omission.over(xs, 500)
                self.assertEqual(count, 5800 - within - k)
                seen.add(Fraction(5800 - count, 5800) >= Fraction(9997, 10000))
            self.assertEqual(len(seen) == 1, decided(5800, within, above, need) != "unresolved")

    def test_small_totals_exhaustively(self):
        for total in range(1, 13):
            for q in (0.5, 0.9, 0.99):
                need = math.ceil(Fraction(str(q)) * total)
                for within in range(total + 1):
                    for above in range(total - within + 1):
                        unknown = total - within - above
                        self.assertEqual(bounds.quantile_verdict(total, q, within, unknown),
                                         decided(total, within, above, need), (total, q, within, above))


def allowed(first, last, most, modulus=2 ** 32):
    """Every increase up to `most` that leaves the two readings, found one wrap at a time."""
    found, total = [], (last - first) % modulus
    while total <= most:
        found.append(total)
        total += modulus
    return found


class AGapThatWasRounded(unittest.TestCase):
    RATE = 60_000_000                                                            # bytes/s, a hard bound

    def test_the_table_of_the_review(self):
        shortest = Fraction(2 ** 32, self.RATE)                                  # the least time for one wrap
        self.assertEqual(f"{float(shortest):.8f}", "71.58278827")
        rows = (("71.580", "71.580", 4_294_800_000, [0]),                        # the gap is exact
                ("71.58", "71.585", 4_295_100_000, [0, 2 ** 32]),                # rounded to 0.01 s
                ("71.57", "71.575", 4_294_500_000, [0]))
        for shown, longest, most, want in rows:
            self.assertEqual(math.floor(self.RATE * Fraction(longest)), most)
            ms = int(Fraction(longest) * 1000)
            self.assertEqual(self.RATE * ms // 1000, most)
            self.assertEqual(Fraction(longest) > shortest, len(want) == 2, shown)
            self.assertEqual(allowed(0, 0, most), want)
            self.assertEqual(counter.increments(0, 0, most), want)
        self.assertLess(Fraction("71.58"), shortest)                             # the number shown is below a wrap,
        self.assertGreater(Fraction("71.585"), shortest)                         # the gap behind it need not be

    def test_two_stamps_rounded_apart(self):
        # Each stamp within 0.005 s of what is shown: the gap can exceed the difference shown by 0.01 s.
        for shown, want in (("71.57", [0]), ("71.58", [0, 2 ** 32])):
            longest = Fraction(shown) + 2 * Fraction("0.005")
            most = self.RATE * int(longest * 1000) // 1000
            self.assertEqual(counter.increments(0, 0, most), want, shown)
        self.assertEqual(self.RATE * 71_580 // 1000, 4_294_800_000)

    def test_a_whole_bound(self):
        # A product of floats can fall short of a bound that is a whole number, and lose the largest increase.
        rate, ms = 176_242_650, 531_800
        exact = Fraction(rate * ms, 1000)
        self.assertEqual(exact, 93_725_841_270)
        self.assertEqual(math.floor(rate * (ms / 1000)), 93_725_841_269)
        last = int(exact) % 2 ** 32
        self.assertEqual(counter.increments(0, last, rate * ms // 1000)[-1], exact)
        self.assertNotIn(exact, allowed(0, last, math.floor(rate * (ms / 1000))))
        for most in (4.2951e9, 4_295_100_000.0, Fraction(4_295_100_000), "4295100000", None):
            self.assertEqual(refusal(counter.increments, 0, 0, most), "need a whole bound", most)
        self.assertEqual(counter.increments(0, 0, -1), [])
        self.assertEqual(counter.increments(0, 0, 0), [0])

    def test_against_a_search(self):
        rng = random.Random(38)
        for _ in range(300):
            modulus = rng.choice((10, 256, 2 ** 16, 2 ** 32))
            first, last = rng.randrange(modulus), rng.randrange(modulus)
            most = rng.randrange(0, 5 * modulus)
            self.assertEqual(counter.increments(first, last, most, modulus), allowed(first, last, most, modulus))

    def test_the_transcript(self):
        text = expected("ch08_counter")
        lines = text.splitlines()
        self.assertEqual(len(lines), 28)                                         # 25 in Version 38
        self.assertEqual(lines[25:], ["readings 0 and 0, the gap rounded to",
                                      f"  71.58 s: 0 or {2 ** 32} bytes",
                                      "  71.57 s: 0 bytes"])
        self.assertEqual(printed(counter.main), text)


def enumerated(n, p):
    """Chance of each number of reversed pairs among n independent ones, from all 2 ** n outcomes."""
    chances = [0.0] * (n + 1)
    for outcome in range(2 ** n):                                                # which of the n are reversed
        k = bin(outcome).count("1")
        chances[k] += p ** k * (1 - p) ** (n - k)
    return chances


def reversal(gap, sigma=2.0):
    """Chance that two stamps with independent normal errors of standard deviation sigma reverse a true gap:
    the upper tail of a normal difference, from the complementary error function."""
    return 0.5 * math.erfc(gap / (sigma * math.sqrt(2)) / math.sqrt(2))


class TenPairs(unittest.TestCase):
    def test_one_pair(self):
        p = reversal(1.0)
        self.assertEqual(f"{p:.7f}", "0.3618368")
        self.assertEqual(f"{100 * reversal(10.0):.5f}", "0.02035")
        self.assertEqual(f"{10 * p:.2f}", "3.62")
        self.assertEqual(f"{100 * p ** 10:.5f}", "0.00385")                      # all ten, errors drawn anew
        self.assertEqual((f"{100 * (1 - p):.2f}", f"{100 * p:.2f}"), ("63.82", "36.18"))

    def test_every_outcome_of_ten(self):
        p = reversal(1.0)
        want = enumerated(10, p)
        got = ordering.reversed_of(10, p, False)
        self.assertEqual(len(got), 11)
        for a, b in zip(got, want):
            self.assertAlmostEqual(a, b, places=12)
        self.assertAlmostEqual(sum(got), 1.0, places=12)
        kept = ordering.reversed_of(10, p, True)
        self.assertEqual(kept, [1 - p] + [0.0] * 9 + [p])
        for ps in (got, kept):
            self.assertAlmostEqual(sum(k * q for k, q in enumerate(ps)), 10 * p, places=12)
        self.assertEqual(max(range(11), key=got.__getitem__), 3)                 # the tallest bar of Figure 6.1

    def test_the_two_models_simulated(self):
        # Stamps as Listing 6.2 draws them, for ten pairs 1 ms apart: errors drawn anew for every pair, and
        # one error for each clock that all ten pairs of a capture share.
        p, captures = reversal(1.0), 20_000
        for shared in (False, True):
            rng = random.Random(6 + shared)
            counts = [0] * 11
            for _ in range(captures):
                a, b = rng.gauss(0, 2.0), rng.gauss(0, 2.0)
                k = 0
                for _ in range(10):
                    if not shared:
                        a, b = rng.gauss(0, 2.0), rng.gauss(0, 2.0)
                    k += 1.0 + b < a
                counts[k] += 1
            model = ordering.reversed_of(10, p, shared)
            for k in range(11):
                se = math.sqrt(model[k] * (1 - model[k]) / captures)
                self.assertLessEqual(abs(counts[k] / captures - model[k]), 4 * se + 1e-12, (shared, k))
            if shared:
                self.assertEqual(sum(counts[1:10]), 0)                           # none, or all ten

    def test_the_transcript(self):
        text = expected("ch06_ordering")
        lines = text.splitlines()
        self.assertEqual(len(lines), 12)                                         # 8 in Version 38
        rng = random.Random(6)
        for line, gap in zip(lines[2:7], (0.5, 1.0, 2.0, 5.0, 10.0)):
            bad = sum(gap + rng.gauss(0, 2.0) < first for first in (rng.gauss(0, 2.0) for _ in range(50_000)))
            self.assertEqual(line, f"{gap:>6.1f} ms{100 * bad / 50_000:>9.3f} %{100 * reversal(gap):>9.3f} %")
            se = math.sqrt(reversal(gap) * (1 - reversal(gap)) / 50_000)
            self.assertLess(abs(bad / 50_000 - reversal(gap)), 4 * se)
        self.assertEqual(bad, 7)                                                 # at 10 ms: printed as 0.000 before
        self.assertEqual(lines[6], "  10.0 ms    0.014 %    0.020 %")
        p = reversal(1.0)
        self.assertEqual(lines[8:10], ["ten pairs 1 ms apart, reversed:", f"{'':<19}{'none':>9}{'all ten':>12}{'mean':>6}"])
        self.assertEqual(lines[10:], [
            f"  errors drawn anew{100 * (1 - p) ** 10:>7.2f} %{100 * p ** 10:>10.5f} %{10 * p:>6.2f}",
            f"  one offset kept  {100 * (1 - p):>7.2f} %{100 * p:>10.5f} %{10 * p:>6.2f}"])
        self.assertEqual(printed(ordering.main), text)

    def test_the_gap_for_95_percent(self):
        lo, hi = 0.0, 20.0
        for _ in range(100):                                                     # reversal() falls as the gap grows
            mid = (lo + hi) / 2
            if reversal(mid) > 0.05:
                lo = mid
            else:
                hi = mid
        self.assertEqual(f"{lo:.1f}", "4.7")
        self.assertIn("gap for 95 % order: 4.7 ms", expected("ch06_ordering"))


def outside(lo, hi, mean, sd):
    """Chance that a normal result falls outside two limits, from the complementary error function."""
    root = sd * math.sqrt(2)
    return 0.5 * math.erfc((hi - mean) / root) + 0.5 * math.erfc((mean - lo) / root)


class WhatTheChartPredicts(unittest.TestCase):
    def limits(self):
        xs = chart.series(random.Random(15))
        base = xs[:30]
        ranges = [abs(b - a) for a, b in zip(base, base[1:])]
        center = math.fsum(base) / 30
        sigma = math.fsum(ranges) / 29 / 1.128
        return center - 3 * sigma, center + 3 * sigma

    def test_the_table_of_the_review(self):
        lo, hi = self.limits()
        self.assertEqual((f"{lo:.5f}", f"{hi:.5f}"), ("94.41053", "105.20792"))
        known = [f"{1 / outside(95.5, 104.5, 100 + 1.5 * d, 1.5):.1f}" for d in (0, 1, 2)]
        drawn = [f"{1 / outside(lo, hi, 100 + 1.5 * d, 1.5):.1f}" for d in (0, 1, 2)]
        self.assertEqual(known, ["370.4", "43.9", "6.3"])
        self.assertEqual(drawn, ["2813.0", "148.8", "14.2"])
        for d in (0, 1, 2):
            mean = 100 + 1.5 * d
            self.assertAlmostEqual(run_length.wait(95.5, 104.5, mean) * outside(95.5, 104.5, mean, 1.5), 1.0, places=9)
            self.assertAlmostEqual(run_length.wait(lo, hi, mean) * outside(lo, hi, mean, 1.5), 1.0, places=9)
        self.assertAlmostEqual(run_length.wait(-3, 3, 0, 1), 1 / math.erfc(3 / math.sqrt(2)), places=6)
        self.assertEqual(run_length.wait(-math.inf, 0.0, 0.0), 2.0)              # one limit: a coin

    def test_waits_simulated(self):
        # Nights drawn until one is outside the limits, independently of the formula.
        lo, hi = self.limits()
        for limits, shift, runs in (((95.5, 104.5), 1, 20_000), ((lo, hi), 2, 20_000), ((lo, hi), 1, 6_000)):
            rng = random.Random(15 + shift)
            mean = 100 + 1.5 * shift
            waits = []
            for _ in range(runs):
                nights = 1
                while limits[0] <= rng.gauss(mean, 1.5) <= limits[1]:
                    nights += 1
                waits.append(nights)
            p = outside(*limits, mean, 1.5)
            se = math.sqrt(1 - p) / p / math.sqrt(runs)                          # a geometric wait
            self.assertLess(abs(math.fsum(waits) / runs - 1 / p), 4 * se, (limits, shift))

    def test_the_simulated_row_agrees(self):
        lines = expected("ch15_control_chart").splitlines()
        row = [float(cell) for cell in lines[7].split()[1:]]
        self.assertEqual(row, [363.3, 42.5, 6.1])
        for got, d in zip(row, (0, 1, 2)):
            p = outside(95.5, 104.5, 100 + 1.5 * d, 1.5)
            se = math.sqrt(1 - p) / p / math.sqrt(1000)                          # arl() takes 1000 runs
            self.assertLess(abs(got - 1 / p), 3 * se, d)

    def test_forty_nights_without_a_signal(self):
        lo, hi = self.limits()
        quiet = [f"{100 * (1 - outside(a, b, 101.5, 1.5)) ** 40:.1f}" for a, b in ((95.5, 104.5), (lo, hi))]
        self.assertEqual(quiet, ["39.8", "76.4"])
        # The same by simulation: forty nights after a shift of one standard deviation, none outside the limits.
        rng = random.Random(40)
        for (a, b), want in (((95.5, 104.5), 0.398), ((lo, hi), 0.764)):
            runs = 20_000
            silent = sum(all(a <= rng.gauss(101.5, 1.5) <= b for _ in range(40)) for _ in range(runs))
            p = (1 - outside(a, b, 101.5, 1.5)) ** 40
            self.assertLess(abs(silent / runs - p), 4 * math.sqrt(p * (1 - p) / runs))
            self.assertAlmostEqual(p, want, places=3)

    def test_the_transcripts(self):
        # Listing 15.1 and its transcript are those of Version 38; the exact run lengths are Listing 15.2.
        self.assertEqual(len(expected("ch15_control_chart").splitlines()), 10)
        text = expected("ch15_run_length")
        lines = text.splitlines()
        self.assertEqual(len(lines), 7)
        lo, hi = self.limits()
        pairs = ((95.5, 104.5), (lo, hi))
        names = [f"[{a:.2f}, {b:.2f}]".ljust(17) for a, b in pairs]
        self.assertEqual(names, ["[95.50, 104.50]  ", "[94.41, 105.21]  "])
        cells = ["".join(f"{1 / outside(a, b, 100 + 1.5 * d, 1.5):>7.1f}" for d in (0, 1, 2)) for a, b in pairs]
        quiet = [f"{100 * (1 - outside(a, b, 101.5, 1.5)) ** 40:>7.1f} %" for a, b in pairs]
        self.assertEqual(lines, ["Shewhart, exact ARL at a shift of",
                                 f"{'limits, ms':<17}{'0':>7}{'1 s':>7}{'2 s':>7}",
                                 names[0] + cells[0], names[1] + cells[1],
                                 "no signal in 40 nights at 1 s",
                                 names[0] + quiet[0], names[1] + quiet[1]])
        self.assertEqual(printed(run_length.main), text)
        # The limits of Listing 15.2 are those that Listing 15.1 prints.
        self.assertIn(f"limits [{lo:.2f}, {hi:.2f}] ms", expected("ch15_control_chart"))


class TheTwentiethFigure(unittest.TestCase):
    def test_the_script(self):
        source = (ROOT / "figures" / "make_figures.py").read_text(encoding="utf-8")
        for piece in ("def reversed_pairs():", "c6.reversed_of(10, p, False)", "c6.reversed_of(10, p, True)",
                      'save(fig, "fig-06-1-reversed-pairs.png")', "    reversed_pairs()\n    exchange_gap()\n"):     # Figure 6.2 follows
            self.assertIn(piece, source)
        self.assertEqual(source.count("\n    save(fig, "), 32)                   # nineteen in Version 38

    def test_what_it_shows(self):
        p = reversal(1.0)
        kept = ordering.reversed_of(10, p, True)
        self.assertEqual((f"{100 * kept[0]:.2f} %", f"{100 * kept[10]:.2f} %"), ("63.82 %", "36.18 %"))
        anew = ordering.reversed_of(10, p, False)
        want = enumerated(10, p)
        self.assertEqual([f"{100 * q:.1f}" for q in anew], [f"{100 * q:.1f}" for q in want])
        self.assertEqual(sorted(range(11), key=want.__getitem__)[-2:], [4, 3])   # the two tallest bars: three and four
        self.assertLess(100 * max(max(want), 1 - p), 78)                         # the vertical axis ends at 78 %


if __name__ == "__main__":
    unittest.main()
