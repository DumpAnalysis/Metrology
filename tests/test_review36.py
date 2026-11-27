"""Regression tests added after the review of Version 36.

The review found that fit_temperature() of Listing 18.1 could report a temperature inside its interval where the
loss has no minimum there, because rounded losses are equal and the search then stops anywhere; that estimated()
of Listing 8.3 refused a kept trace without a failure, and left the uncertainty unknown for an empty capture even
when everything had been kept; and that the caption of Figure 10.2 gave the two ends of its ramp to three
decimals as if they were exact. It asked for the interval that a hundred labels leave for a fitted temperature,
and for the requests of Listing 10.1 counted against an objective.

Each test computes what it expects independently of the code under test: the search of Version 36 rebuilt beside
the new one, a closed form for the temperature of cases at one logit, the loss on a grid of temperatures, exact
binomial tails from binomial coefficients, every outcome of a small sampling design, and counts of a list.
"""
import contextlib
import io
import itertools
import math
import random
import unittest
from fractions import Fraction
from pathlib import Path

from examples import ch08_sampling as sampling
from examples import ch10_coordinated_omission as omission
from examples import ch18_calibration as cal
from swmetro import stats

ROOT = Path(__file__).resolve().parent.parent


def expected(name):
    return (ROOT / "expected" / f"{name}.txt").read_text(encoding="utf-8")


def printed(function, *args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        function(*args)
    return out.getvalue()


def version_36(data, lo=0.2, hi=10.0):
    """fit_temperature() as Version 36 printed it: a golden-section search among computed losses, refused when
    it ends at an end of its interval."""
    g = (math.sqrt(5) - 1) / 2
    a, b = ends = math.log(lo), math.log(hi)
    for _ in range(40):
        c, d = b - g * (b - a), a + g * (b - a)
        if cal.nll(data, math.exp(c)) < cal.nll(data, math.exp(d)):
            b = d
        else:
            a = c
    if a == ends[0] or b == ends[1]:
        raise ValueError("no minimum between lo and hi")
    return math.exp((a + b) / 2)


def slope_v64(ms, beta):
    """slope() of Listing 18.1 until Version 64: the derivative of the summed loss by 1/T as an exact sum of
    rounded products. Version 65 has lean() in its place, which returns the sign and keeps it where a product
    rounds to zero. (Note added with Version 65.)"""
    return sum(Fraction(-m * cal.sigmoid(-beta * m)) for m in ms)


def refusal(function, *args, **named):
    try:
        function(*args, **named)
    except ValueError as error:
        return str(error)
    return None


def at_one_logit(logit, positive, n):
    """The temperature at which n cases that all state one logit, `positive` of them with the label 1, have
    their least loss: the stated probability is then the observed frequency."""
    return logit / math.log(Fraction(positive, n - positive))


def total_loss(data, temp):
    """The summed loss, each term from its margin: ln(1 + e ** t) for t = -margin / T, with the exponent kept
    from overflowing."""
    terms = (-(z if y else -z) / temp for z, y in data)
    return math.fsum(max(t, 0.0) + math.log1p(math.exp(-abs(t))) for t in terms)


class TheShapeOfTheLossBeforeTheSearch(unittest.TestCase):
    def test_the_table_of_the_review(self):
        right, half, flat = [(-200, 0), (200, 1)], [(0, 0), (8, 1)], [(0, 0), (0, 1)]
        self.assertEqual((f"{version_36(right):.6f}", f"{version_36(half):.6f}"), ("0.268408", "0.213733"))
        self.assertEqual(refusal(version_36, flat), "no minimum between lo and hi")
        self.assertEqual(refusal(cal.fit_temperature, right), "no finite T: the loss falls as T falls")
        self.assertEqual(refusal(cal.fit_temperature, half), "no finite T: the loss falls as T falls")
        self.assertEqual(refusal(cal.fit_temperature, flat), "the loss is the same at every T")
        # Why Version 36 stopped inside the interval: the loss at its lower end is not above the loss at the
        # temperature that it reported. In the first case both have rounded to zero.
        self.assertEqual((cal.nll(right, 0.2), cal.nll(right, 0.268408)), (0.0, 0.0))
        self.assertLess(cal.nll(half, 0.2), cal.nll(half, 0.213733))
        self.assertLess(cal.nll(half, 0.213733) - cal.nll(half, 0.2), 1e-13)

    def test_cases_at_one_logit(self):
        # n cases state one logit, and some of them have the label 0: the loss is least where the stated
        # probability is the observed frequency.
        rng = random.Random(3601)
        for _ in range(200):
            n = rng.randint(3, 400)
            negative = rng.randint(1, (n - 1) // 2)
            logit = rng.uniform(0.2, 60)
            data = [(logit, 1)] * (n - negative) + [(logit, 0)] * negative
            rng.shuffle(data)
            want = at_one_logit(logit, n - negative, n)
            self.assertAlmostEqual(cal.fit_temperature(data, 1e-4, 1e4) / want, 1, places=9)
            message = refusal(cal.fit_temperature, data)
            if want < 0.2:
                self.assertEqual(message, "the minimum is below lo")
            elif want > 10:
                self.assertEqual(message, "the minimum is above hi")
            else:
                self.assertIsNone(message)
        # The cases of the review: 99 and 98 of 100 at a logit of 40.
        self.assertEqual((f"{at_one_logit(40, 99, 100):.2f}", f"{at_one_logit(40, 98, 100):.2f}"), ("8.70", "10.28"))
        self.assertAlmostEqual(cal.fit_temperature([(40, 1)] * 99 + [(40, 0)]), 40 / math.log(99), places=9)
        self.assertEqual(refusal(cal.fit_temperature, [(40, 1)] * 98 + [(40, 0)] * 2), "the minimum is above hi")
        self.assertAlmostEqual(cal.fit_temperature([(40, 1)] * 98 + [(40, 0)] * 2, hi=20), 40 / math.log(49), places=9)

    def test_the_shape_on_a_grid(self):
        # Losses of moderate margins on 400 temperatures from 0.001 to 1000. At the smallest of them the loss of
        # a case on the side of its logit rounds to zero, so the order of the losses is tested and not their strict
        # fall: equal neighbors are what stopped the search of Version 36.
        rng = random.Random(3602)
        grid = [10 ** (-3 + 6 * k / 399) for k in range(400)]
        seen = set()
        # One set made for the lower end, which random margins of this size seldom reach: eight small margins
        # against one. Its minimum is at T = 0.026.
        made = [[(0.06, 1)] * 8 + [(0.05, 0)]]
        for number in range(600):
            kind = rng.choice(("right", "wrong", "mixed", "mixed", "mixed", "zeros"))
            data = []
            for _ in range(rng.randint(1, 12)):
                size = rng.choice((0.0, rng.uniform(0.05, 3)))
                y = rng.randint(0, 1)
                sign = {"right": 1, "wrong": -1, "zeros": 0}.get(kind, rng.choice((1, -1)))
                data.append((sign * size if y else -sign * size, y))
            if number < len(made):
                data = made[number]
            margins = [z if y else -z for z, y in data]
            losses = [total_loss(data, t) for t in grid]
            message = refusal(cal.fit_temperature, data, 0.05, 50)               # an interval well inside the grid
            if not any(margins):
                self.assertEqual(max(losses), min(losses))
                self.assertEqual(message, "the loss is the same at every T")
            elif min(margins) >= 0:
                self.assertEqual(losses, sorted(losses))                         # no worse at any smaller T
                self.assertLess(losses[0], losses[-1])
                self.assertEqual(message, "no finite T: the loss falls as T falls")
            elif sum(Fraction(m) for m in margins) <= 0:
                self.assertEqual(losses, sorted(losses, reverse=True))           # no worse at any larger T
                self.assertLess(losses[-1], losses[0])
                self.assertEqual(message, "no finite T: the loss falls as T rises")
            else:
                best = min(range(400), key=losses.__getitem__)
                below, above = grid[max(best - 1, 0)], grid[min(best + 1, 399)]  # the minimum lies between these
                if above < 0.05 or best == 0:
                    self.assertEqual(message, "the minimum is below lo")
                elif below > 50 or best == 399:
                    self.assertEqual(message, "the minimum is above hi")
                elif below > 0.05 and above < 50:
                    self.assertIsNone(message)
                    self.assertTrue(below <= cal.fit_temperature(data, 0.05, 50) <= above, data)
                else:
                    continue                                                     # too near an end for the grid to say
            seen.add(message)
        self.assertEqual(len(seen), 6, seen)                                     # every outcome occurred

    def test_the_slope(self):
        rng = random.Random(3603)
        for _ in range(200):
            data = [(rng.uniform(-4, 4), rng.randint(0, 1)) for _ in range(rng.randint(1, 10))]
            margins = cal.margins(data)
            self.assertEqual(margins, [z if y else -z for z, y in data])
            beta, step = rng.uniform(0.05, 5), 1e-6
            numeric = (total_loss(data, 1 / (beta + step)) - total_loss(data, 1 / (beta - step))) / (2 * step)
            self.assertAlmostEqual(slope_v64(margins, beta), numeric, places=5)
            self.assertLessEqual(slope_v64(margins, beta), slope_v64(margins, beta * 1.5) + 1e-12)   # it rises with beta
            if abs(numeric) > 1e-4:                                              # Version 65: the sign, from lean()
                self.assertEqual(cal.lean(margins, beta), (numeric > 0) - (numeric < 0))
        # The slope rounds as the losses do: for the first case of the review it is below zero at T = 10 and
        # exactly zero at T = 0.25, where e ** -800 is no float. The function therefore decides such a case
        # from the signs of the margins, and uses the slope only where a margin is below zero.
        self.assertLess(slope_v64([200.0, 200.0], 1 / 10), 0)
        self.assertEqual(slope_v64([200.0, 200.0], 1 / 0.25), 0.0)
        # Version 65: lean() compares logarithms and keeps the sign that the product had lost.
        self.assertEqual(cal.lean([200.0, 200.0], 1 / 10), -1)
        self.assertEqual(cal.lean([200.0, 200.0], 1 / 0.25), -1)
        self.assertEqual(refusal(cal.fit_temperature, [(200.0, 1), (-200.0, 0)]), "no finite T: the loss falls as T falls")
        # With one margin below zero, however small, the slope at the lower end is above zero and is computed so.
        self.assertGreater(slope_v64([200.0, 200.0, -1e-300], 1 / 0.2), 0)
        self.assertEqual(cal.lean([200.0, 200.0, -1e-300], 1 / 0.2), 1)

    def test_an_iterator_is_read_once(self):
        data = [(2.0, 1), (-2.0, 0)] * 3 + [(2.0, 0), (-2.0, 1)]
        self.assertEqual(cal.fit_temperature(iter(data)), cal.fit_temperature(data))
        self.assertAlmostEqual(cal.fit_temperature(data), 2 / math.log(3), places=9)
        self.assertEqual(refusal(cal.fit_temperature, []), "no cases")
        self.assertEqual(refusal(cal.fit_temperature, data, 0, 10), "need 0 < lo < hi < inf")   # finite since Version 63

    def test_the_fit_of_the_listing(self):
        # The two searches agree where the loss has a minimum that floats can show: the data of the listing.
        rng = random.Random(18)
        data = cal.make(2000, 2.0, rng)
        self.assertAlmostEqual(cal.fit_temperature(data), version_36(data), places=6)
        self.assertEqual(f"{cal.fit_temperature(data):.2f}", "2.04")
        self.assertAlmostEqual(slope_v64(cal.margins(data), 1 / cal.fit_temperature(data)), 0, delta=1e-9)


def tail(n, k, p, upper):
    """P(X >= k) or P(X <= k) for X binomial(n, p), summed from binomial coefficients."""
    ks = range(k, n + 1) if upper else range(0, k + 1)
    return math.fsum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in ks)


def exact_limits(k, n, level=0.95):
    """Clopper and Pearson: the proportions at which k or more, and k or fewer, successes have the chance
    (1 - level) / 2, found by halving."""
    a = (1 - level) / 2
    out = []
    for upper in (True, False):
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if (tail(n, k, mid, upper) < a) == upper:
                lo = mid
            else:
                hi = mid
        out.append((lo + hi) / 2)
    return out


class TheIntervalOfAFittedTemperature(unittest.TestCase):
    def test_the_table_of_the_review(self):
        want = {1: ("8.70", "4.83", "14.01"), 2: ("10.28", "6.65", "15.50"), 3: ("11.51", "7.89", "16.85"), 5: ("13.58", "9.78", "19.40")}
        for negative, row in want.items():
            low, high = exact_limits(100 - negative, 100)
            estimate = at_one_logit(40, 100 - negative, 100)
            ends = (40 / math.log(high / (1 - high)), 40 / math.log(low / (1 - low)))        # the limits change places
            self.assertEqual((f"{estimate:.2f}", f"{ends[0]:.2f}", f"{ends[1]:.2f}"), row)
            self.assertAlmostEqual(cal.at_one_logit(40, (100 - negative) / 100), estimate, places=9)
            self.assertAlmostEqual(cal.at_one_logit(40, high), ends[0], places=9)
        low, high = exact_limits(99, 100)
        self.assertEqual((f"{low:.6f}", f"{high:.6f}"), ("0.945541", "0.999747"))
        self.assertGreater(40 / math.log(low / (1 - low)), 10)                   # beyond the upper end of the search

    def test_what_a_resample_holds(self):
        none = Fraction(99, 100) ** 100
        one = 100 * Fraction(1, 100) * Fraction(99, 100) ** 99
        self.assertEqual([f"{100 * float(v):.2f}" for v in (none, one, 1 - none - one)], ["36.60", "36.97", "26.42"])
        # Two thousand resamples of the hundred cases: each is fitted, or refused for its own reason.
        rng = random.Random(3604)
        data = [(40.0, 1)] * 99 + [(40.0, 0)]
        outcomes = {}
        for _ in range(2000):
            resample = rng.choices(data, k=100)
            negative = sum(1 for _, y in resample if not y)
            message = refusal(cal.fit_temperature, resample)
            want = {0: "no finite T: the loss falls as T falls", 1: None}.get(negative, "the minimum is above hi")
            self.assertEqual(message, want)
            if message is None:
                self.assertEqual(f"{cal.fit_temperature(resample):.2f}", "8.70")
            outcomes[min(negative, 2)] = outcomes.get(min(negative, 2), 0) + 1
        for k, share in ((0, none), (1, one), (2, 1 - none - one)):
            self.assertAlmostEqual(outcomes[k] / 2000, float(share), delta=4.5 * math.sqrt(float(share) * float(1 - share) / 2000))

    def test_the_transcript(self):
        text = expected("ch18_calibration")
        lines = text.splitlines()
        self.assertEqual(len(lines), 24)                                         # 16 in Version 36, 20 in 37, 23 in 38
        self.assertEqual(lines[:16], expected("ch18_calibration").splitlines()[:16])
        self.assertEqual((lines[0], lines[15]), ("fitted temperature T = 2.04", "  T = 8.70, NLL 0.400 -> 0.056"))
        low, high = exact_limits(99, 100)
        none = Fraction(99, 100) ** 100
        one = 100 * Fraction(1, 100) * Fraction(99, 100) ** 99
        self.assertEqual(lines[16:20], [
            # Since Version 63 the two ends are rounded outward: 4.8302 downward and 14.0139 upward.
            f"  95 % interval for T: {math.floor(4000 / math.log(high / (1 - high))) / 100:.2f} to "
            f"{math.ceil(4000 / math.log(low / (1 - low))) / 100:.2f}",
            f"  with two negative: T = {at_one_logit(40, 98, 100):.2f}",
            "  resampled, they hold no negative",
            f"  case in {100 * float(none):.1f} %, one in {100 * float(one):.1f} %"])
        self.assertEqual(lines[16], "  95 % interval for T: 4.83 to 14.02")       # 14.01 to the nearest, until Version 62
        self.assertEqual(printed(cal.main), text)

    def test_the_script_of_the_figure(self):
        source = (ROOT / "figures" / "make_figures.py").read_text(encoding="utf-8")
        for piece in ("def temperature_interval():", "c20.exact_interval(n - negative, n)", "c18.at_one_logit(logit, high)",
                      'save(fig, "fig-18-2-temperature-interval.png")', "    reliability()\n    temperature_interval()\n    judge_model()\n"):
            self.assertIn(piece, source)
        self.assertEqual(source.count("\n    save(fig, "), 32)                   # seventeen in Version 36


def every_outcome(sizes, rate):
    rate = Fraction(rate)
    for outcome in itertools.product((0, 1), repeat=len(sizes)):
        chance = math.prod(rate if kept else 1 - rate for kept in outcome)
        kept = [m for m, k in zip(sizes, outcome) if k]
        yield chance, kept, sum(kept) / rate, (1 - rate) * sum(m * m for m in kept) / rate ** 2


class KeptTracesWithoutAFailure(unittest.TestCase):
    def test_the_example_of_the_review(self):
        total, u = sampling.estimated([0, 1, 0, 3], 0.1)                         # Version 36: ValueError
        self.assertAlmostEqual(total, 40, places=9)
        self.assertAlmostEqual(u, math.sqrt(Fraction(9, 10) * (1 + 9) * 100), places=9)
        self.assertEqual(f"{u:.4f}", "30.0000")

    def test_a_trace_without_a_failure_changes_nothing(self):
        rng = random.Random(3605)
        for _ in range(200):
            sizes = [rng.randint(1, 20) for _ in range(rng.randint(1, 10))]
            rate = rng.choice((0.01, 0.1, 0.5, 1))
            padded = sizes + [0] * rng.randint(1, 5)
            rng.shuffle(padded)
            self.assertEqual(sampling.estimated(padded, rate), sampling.estimated(sizes, rate))

    def test_a_capture_without_a_failure(self):
        for kept in ([], [0], [0, 0, 0]):
            for rate in (0.1, 0.5, Fraction(1, 3)):
                self.assertEqual(sampling.estimated(kept, rate), (0.0, None), (kept, rate))       # unknown, not zero
            for rate in (1, 1.0, Fraction(1)):
                self.assertEqual(sampling.estimated(kept, rate), (0.0, 0.0), (kept, rate))        # all kept: none occurred
        self.assertEqual(sampling.estimated([], 1), (0.0, 0.0))                  # Version 36: (0.0, None)

    def test_both_estimates_stay_unbiased(self):
        # Designs in which some units hold no event: the estimates over every outcome, the empty captures among
        # them with the sum that the function leaves unknown, which is zero.
        rng = random.Random(3606)
        for _ in range(30):
            sizes = [rng.choice((0, 0, rng.randint(1, 9))) for _ in range(rng.randint(1, 8))]
            rate = Fraction(rng.randint(1, 9), 10)
            mean = square = variance = Fraction(0)
            for chance, kept, estimate, estimated_variance in every_outcome(sizes, rate):
                mean += chance * estimate
                square += chance * estimate ** 2
                variance += chance * estimated_variance
                got, u = sampling.estimated(kept, float(rate))
                self.assertAlmostEqual(got, float(estimate), places=9)
                if any(kept):
                    self.assertAlmostEqual(u * u / float(estimated_variance), 1, places=9)
                else:
                    self.assertIsNone(u)
                    self.assertEqual(estimated_variance, 0)
            self.assertEqual((mean, variance), (sum(sizes), square - mean ** 2))

    def test_what_is_still_refused(self):
        for kept in ([2, -1], [1.5], ["x"], [None]):
            with self.assertRaises(ValueError, msg=repr(kept)):
                sampling.estimated(kept, 0.5)
        for rate in (0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                sampling.estimated([0, 1], rate)

    def test_the_transcript_is_that_of_version_36(self):
        lines = expected("ch08_sampling").splitlines()
        self.assertEqual(len(lines), 25)
        self.assertEqual(printed(sampling.main).splitlines(), lines)


class CountsAgainstAnObjective(unittest.TestCase):
    def setUp(self):
        measured = omission.closed_loop()
        self.rows = (("as measured", measured), ("back-filled", omission.back_fill(measured)), ("intended", omission.open_loop()))

    def test_the_table_of_the_review(self):
        want = ((5800, 2, "99.9655"), (6000, 102, "98.3000"), (6000, 112, "98.1333"))
        for (name, xs), (n, above, within) in zip(self.rows, want):
            self.assertEqual((len(xs), sum(1 for x in xs if x > 500)), (n, above), name)
            self.assertEqual(f"{float(100 * Fraction(n - above, n)):.4f}", within)
            count, share = omission.over(xs)
            self.assertEqual((count, f"{share:.4f}"), (above, within))
        # The back-filled row: each of the two requests of 1001 ms brings the entries 991, 981, ... down to 1 ms,
        # and fifty of those hundred are above 500 ms.
        added = [1001 - 10 * k for k in range(1, 101)]
        self.assertEqual((added[-1], sum(1 for x in added if x > 500)), (1, 50))
        self.assertEqual(2 + 2 * 50, 102)

    def test_what_the_objective_allows(self):
        # At least 99.97 % of 5800 within the limit: 5799 or more, so one request may be above it.
        needed = math.ceil(Fraction(9997, 10000) * 5800)
        self.assertEqual((needed, 5800 - needed), (5799, 1))
        self.assertGreater(omission.over(self.rows[0][1])[0], 5800 - needed)     # the record has two
        self.assertEqual(omission.over([1, 500, 500.5, 2000]), (2, 50.0))        # the limit itself is within
        self.assertEqual(omission.over([1, 2, 3], limit=2), (1, 100 * 2 / 3))

    def test_the_transcript(self):
        text = expected("ch10_coordinated_omission")
        lines = text.splitlines()
        self.assertEqual(len(lines), 12)                                         # 6 in Version 36, 10 until 45
        want = ["over 500 ms     n  over     within"]
        for name, xs in self.rows:
            above = sum(1 for x in xs if x > 500)
            want.append(f"{name:<12}{len(xs):>5}{above:>6}{float(100 * Fraction(len(xs) - above, len(xs))):>9.4f} %")
        self.assertEqual(lines[6:10], want)
        self.assertEqual(lines[7], "as measured  5800     2  99.9655 %")
        self.assertEqual(printed(omission.main), text)


class TheEndsOfTheRamp(unittest.TestCase):
    def test_they_are_not_round_numbers(self):
        xs = omission.closed_loop()
        start, end = Fraction(5797, 5799), Fraction(5798, 5799)
        self.assertEqual((f"{100 * float(start):.5f}", f"{100 * float(end):.5f}"), ("99.96551", "99.98276"))
        # At 99.966 exactly the interpolated percentile is already 29.34 ms, and not the 1 ms of the caption of
        # Version 36.
        self.assertEqual(f"{stats.quantile(xs, 0.99966):.2f}", "29.34")
        position = Fraction(99966, 100000) * 5799 - 5797
        self.assertEqual(f"{float(1 + 1000 * position):.2f}", "29.34")
        self.assertEqual(f"{stats.quantile(xs, 0.9997):.1f}", "261.3")


if __name__ == "__main__":
    unittest.main()
