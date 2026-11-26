"""Regression tests added after the review of Version 35.

The review found that the negative log-likelihood of Listing 18.1 was taken from a probability that rounds to 1
at a scaled logit of 40, so that one confident error raised an exception where the loss is 40, and a fit with a
finite optimum failed; that the bound on the units behind an empty record needs independent retention, and is
another number under a fixed quota; and that the correction of Chapter 12 was described as an observed error. It
asked for the estimate of a count and of its sampling variance from one capture.

Each test computes what it expects independently of the code under test: logarithms and exponentials in decimal
arithmetic of 400 digits, a closed form for the fitted temperature, a search over a grid, every outcome of a small
sampling design written out in fractions, and binomial coefficients formed as products.
"""
import contextlib
import decimal
import io
import itertools
import math
import random
import unittest
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from examples import ch08_sampling as sampling
from examples import ch10_coordinated_omission as omission
from examples import ch12_calibration as probe
from examples import ch17_seeds as seeds
from examples import ch18_calibration as cal
from swmetro import stats

ROOT = Path(__file__).resolve().parent.parent
EXACT = decimal.Context(prec=400)       # 1 + e ** -700 needs more than 304 digits; the context is not the global one


def expected(name):
    return (ROOT / "expected" / f"{name}.txt").read_text(encoding="utf-8")


def printed(function, *args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        function(*args)
    return out.getvalue()


def loss(z, y):
    """-ln of the probability that the logit z gives to the label y, in decimal arithmetic of 400 digits."""
    z = Decimal(repr(float(z)))
    return float(EXACT.ln(EXACT.add(1, EXACT.exp(-z if y else z))))


def version_35(data, temp=1.0):
    """nll() as Version 35 printed it: from the probability."""
    acc = 0.0
    for logit, y in data:
        p = 1 / (1 + math.exp(-logit / temp))
        acc -= math.log(p if y else 1 - p)
    return acc / len(data)


def raised(function, *args):
    try:
        function(*args)
    except Exception as error:      # the kind of failure is what is compared
        return type(error).__name__
    return None


class TheLossFromTheLogit(unittest.TestCase):
    def test_the_table_of_the_review(self):
        for z, y, before, now in ((40, 0, "ValueError", "40.0"), (40, 1, None, "4.24835e-18"), (-1000, 1, "OverflowError", "1000.0")):
            self.assertEqual(raised(version_35, [(z, y)]), before)
            value = cal.nll([(z, y)])
            self.assertEqual(f"{value:.5e}" if value < 1 else repr(value), now)
            self.assertAlmostEqual(value / loss(z, y), 1, places=12)
        self.assertEqual(version_35([(40, 1)]), 0.0)                             # the small loss was lost as well
        self.assertEqual(1 / (1 + math.exp(-40)), 1.0)                           # why: the probability rounds to 1
        # A raw logit of 8 at the lower end of the search, a temperature of 0.2, is the scaled logit of 40.
        self.assertEqual(raised(version_35, [(8, 0)], 0.2), "ValueError")
        self.assertEqual(cal.nll([(8, 0)], 0.2), 40.0)

    def test_against_four_hundred_digits(self):
        rng = random.Random(3501)
        for _ in range(400):
            z = rng.choice((rng.uniform(-60, 60), rng.uniform(-3, 3), rng.uniform(-700, 700)))
            y = rng.randint(0, 1)
            temp = rng.choice((1.0, 0.2, 2.5, 10.0))
            want = loss(z / temp, y)
            if want < 1e-300:                                                    # below what a float holds: e ** -700 and less
                self.assertLess(cal.nll([(z, y)], temp), 1e-300)
            else:
                self.assertAlmostEqual(cal.nll([(z, y)], temp) / want, 1, places=11, msg=(z, y, temp))
        data = [(rng.gauss(0, 4), rng.randint(0, 1)) for _ in range(300)]
        want = math.fsum(loss(z / 1.7, y) for z, y in data) / 300
        self.assertAlmostEqual(cal.nll(data, 1.7) / want, 1, places=12)
        # Where the probability does not round, the two ways of computing the loss agree.
        mild = [(z, y) for z, y in data if abs(z) < 20]
        self.assertAlmostEqual(cal.nll(mild) / version_35(mild), 1, places=12)

    def test_no_logit_is_too_large(self):
        # Version 62 and before took an infinite logit as well. Since Version 63 a case has a finite logit:
        # the review of Version 62 showed a NaN among the logits that was fitted as if it were a number.
        for z in (math.inf, -math.inf, math.nan):
            with self.assertRaises(ValueError):
                cal.nll([(z, 0)])
        for z in (709.0, 710.0, 1e6, 1e308):
            self.assertEqual(cal.nll([(z, 0)]), z)
            self.assertEqual(cal.nll([(-z, 1)]), z)
            # The loss of the right label is e ** -z, which a float still holds at 709 (1.2e-308) and at 710.
            if z < 1000:
                self.assertAlmostEqual(cal.nll([(z, 1)]) / loss(z, 1), 1, places=6)
            else:
                self.assertEqual(cal.nll([(z, 1)]), 0.0)
            self.assertLess(cal.nll([(z, 1)]), 1e-307)
            self.assertEqual(cal.sigmoid(z), 1.0)
            self.assertLess(cal.sigmoid(-z), 1e-307)
        self.assertEqual(raised(lambda: 1 / (1 + math.exp(710.0))), "OverflowError")        # the sigmoid of Version 35 at -710

    def test_the_sigmoid(self):
        rng = random.Random(3502)
        self.assertEqual(cal.sigmoid(0), 0.5)
        for _ in range(400):
            z = rng.uniform(-40, 40)
            want = float(EXACT.divide(1, EXACT.add(1, EXACT.exp(-Decimal(repr(z))))))
            self.assertAlmostEqual(cal.sigmoid(z) / want, 1, places=13)
            self.assertAlmostEqual(cal.sigmoid(z) + cal.sigmoid(-z), 1, places=14)
        zs = sorted(rng.uniform(-50, 50) for _ in range(200))
        ps = [cal.sigmoid(z) for z in zs]
        self.assertEqual(ps, sorted(ps))

    def test_one_confident_error_among_a_hundred(self):
        data = [(40.0, 1)] * 99 + [(40.0, 0)]
        self.assertEqual(raised(version_35, data), "ValueError")                 # the fit of Version 35 stopped here
        temp = cal.fit_temperature(data)
        # All cases state one probability; the loss is least where it is the observed frequency, 0.99.
        self.assertEqual(f"{40 / math.log(99):.6f}", "8.704887")
        self.assertAlmostEqual(temp, 40 / math.log(99), places=5)
        entropy = -(0.99 * math.log(0.99) + 0.01 * math.log(0.01))
        self.assertEqual((f"{cal.nll(data):.5f}", f"{cal.nll(data, temp):.5f}", f"{entropy:.5f}"), ("0.40000", "0.05600", "0.05600"))
        self.assertTrue(0.2 < temp < 10)

    def test_the_fit_is_a_minimum(self):
        # Against a search over 2000 temperatures, equally spaced in their logarithms.
        rng = random.Random(3503)
        grid = [0.2 * 50 ** (k / 2000) for k in range(2001)]
        for scale in (0.6, 1.0, 2.0, 3.5):
            data = cal.make(400, scale, rng)
            temp = cal.fit_temperature(data)
            best = min(grid, key=lambda t: cal.nll(data, t))
            self.assertAlmostEqual(math.log(temp), math.log(best), delta=math.log(50) / 2000)
            self.assertLessEqual(cal.nll(data, temp), cal.nll(data, best) + 1e-12)

    def test_a_search_that_ends_at_a_boundary(self):
        separable = [(2.0, 1)] * 50 + [(-2.0, 0)] * 50                           # ever better as T falls
        flat = [(0.0, 1), (0.0, 0)]                                              # the same loss at every T
        against = [(2.0, 0)] * 30 + [(-2.0, 1)] * 30 + [(2.0, 1)] * 20 + [(-2.0, 0)] * 20   # ever better as T rises
        # Version 36 said "no minimum between lo and hi" for all three; each has its own message since.
        for data, message in ((separable, "falls as T falls"), (flat, "the same at every T"), (against, "falls as T rises")):
            with self.assertRaisesRegex(ValueError, message):
                cal.fit_temperature(data)
        # The loss falls toward the end of the interval in each case that has a slope.
        self.assertLess(cal.nll(separable, 0.2), cal.nll(separable, 0.21))
        self.assertLess(cal.nll(against, 10.0), cal.nll(against, 9.5))
        # A minimum outside the interval is found when the interval is made to contain it.
        pairs = [(0.5, 1), (-0.3, 0), (1.2, 1), (0.1, 0)]
        with self.assertRaisesRegex(ValueError, "the minimum is below lo"):
            cal.fit_temperature(pairs)
        temp = cal.fit_temperature(pairs, lo=0.01)
        self.assertTrue(0.01 < temp < 0.2)
        self.assertLess(cal.nll(pairs, temp), min(cal.nll(pairs, temp * 1.02), cal.nll(pairs, temp / 1.02)))
        for lo, hi in ((0, 10), (-1, 10), (2, 2), (5, 1)):
            with self.assertRaisesRegex(ValueError, "need 0 < lo < hi"):
                cal.fit_temperature(pairs, lo, hi)

    def test_the_transcript(self):
        text = expected("ch18_calibration")
        lines = text.splitlines()
        self.assertEqual(len(lines), 24)                                         # 14 in Version 35, 16 in 36, 20 in 37, 23 in 38
        self.assertEqual((lines[0], lines[2], lines[3]), ("fitted temperature T = 2.04", "as reported    0.102   0.187   0.589",
                                                          "corrected      0.014   0.176   0.525"))
        self.assertEqual(lines[14:16], ["100 cases at logit 40, one negative:",
                                      f"  T = {40 / math.log(99):.2f}, NLL {40 / 100:.3f} -> "
                                      f"{-(0.99 * math.log(0.99) + 0.01 * math.log(0.01)):.3f}"])
        self.assertEqual(lines[15], "  T = 8.70, NLL 0.400 -> 0.056")
        self.assertEqual(printed(cal.main), text)
        # The printed test loss, from the loss in decimal arithmetic and the data of the listing drawn again.
        rng = random.Random(18)
        cal.make(2000, 2.0, rng)
        test = cal.make(5000, 2.0, rng)
        self.assertEqual(f"{math.fsum(loss(z, y) for z, y in test) / 5000:.3f}", "0.589")


def every_outcome(sizes, rate):
    """For every outcome of one independent decision for each unit: its chance, the estimate of the count, and
    the estimate of the variance, each kept unit of m events adding (1 - rate) * (m / rate) ** 2."""
    rate = Fraction(rate)
    for outcome in itertools.product((0, 1), repeat=len(sizes)):
        chance = math.prod(rate if kept else 1 - rate for kept in outcome)
        kept = [m for m, k in zip(sizes, outcome) if k]
        yield chance, kept, sum(kept) / rate, (1 - rate) * sum(m * m for m in kept) / rate ** 2


class OneCapture(unittest.TestCase):
    def test_the_table_of_the_review(self):
        for sizes, want in (([1] * 20, "42.4"), ([10] * 2, "134.2"), ([20], "189.7")):
            total, u = sampling.estimated(sizes, 0.1)
            self.assertAlmostEqual(total, 200, places=9)
            self.assertEqual(f"{math.sqrt(Fraction(9, 10) * sum(m * m for m in sizes) * 100):.1f}", want)
            self.assertEqual(f"{u:.1f}", want)

    def test_both_estimates_are_unbiased(self):
        rng = random.Random(3504)
        designs = [([6, 1, 1], Fraction(1, 3)), ([180, 5, 5, 5], Fraction(1, 100))]
        designs += [([rng.randint(1, 12) for _ in range(rng.randint(1, 8))], Fraction(rng.randint(1, 9), 10)) for _ in range(30)]
        for sizes, rate in designs:
            mean = variance = square = Fraction(0)
            for chance, kept, estimate, estimated_variance in every_outcome(sizes, rate):
                mean += chance * estimate
                square += chance * estimate ** 2
                variance += chance * estimated_variance
                got, u = sampling.estimated(kept, float(rate))
                self.assertAlmostEqual(got, float(estimate), places=9)
                if kept:
                    self.assertAlmostEqual(u * u / float(estimated_variance), 1, places=9) if rate < 1 else self.assertEqual(u, 0)
                else:
                    self.assertIsNone(u)
            self.assertEqual(mean, sum(sizes))                                   # the estimate of the count
            self.assertEqual(variance, square - mean ** 2)                       # the estimate of its variance
            self.assertEqual(variance, (1 - rate) / rate * sum(m * m for m in sizes))
            self.assertAlmostEqual(sampling.spread(sum(sizes), float(rate), sizes) ** 2 / float(variance), 1, places=9)

    def test_an_empty_capture(self):
        # The sum for the variance is zero, and it is not reported as an uncertainty of zero.
        self.assertEqual(sampling.estimated([], 0.1), (0.0, None))
        self.assertEqual(sampling.estimated(iter([]), 0.1), (0.0, None))
        self.assertEqual(sampling.most_units(0.1), 28)

    def test_what_is_refused(self):
        for rate in (0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                sampling.estimated([1, 2], rate)
        for kept in ([2, -1], [1.5], ["x"]):                                    # [0] was refused too, until Version 36
            with self.assertRaises(ValueError, msg=repr(kept)):
                sampling.estimated(kept, 0.5)
        self.assertEqual(sampling.estimated([3, 4], 1), (7.0, 0.0))              # with everything kept, nothing is uncertain

    def test_the_transcript(self):
        lines = expected("ch08_sampling").splitlines()
        self.assertEqual(len(lines), 25)                                         # 20 in Version 35
        want = ["20 errors kept at rate 0.10, on"]
        for sizes in ([1] * 20, [10] * 2, [20]):
            u = math.sqrt(Fraction(9, 10) * sum(m * m for m in sizes) * 100)
            want.append(f"  traces of {sizes[0]:>2}: estimate {sum(sizes) * 10}, u = {u:.1f}")
        self.assertEqual(lines[17:21], want)
        self.assertEqual(printed(sampling.main).splitlines(), lines)


def all_missed(total, kept, g):
    """The chance that `kept` units drawn from `total` without replacement avoid g marked ones, as a product."""
    chance = Fraction(1)
    for i in range(kept):
        chance *= Fraction(max(total - g - i, 0), total - i)
    return chance


class AFixedQuota(unittest.TestCase):
    def test_the_table_of_the_review(self):
        self.assertEqual((sampling.most_units(0.1), sampling.most_of_quota(100, 10)), (28, 24))
        self.assertEqual([f"{100 * float(Fraction(9, 10) ** g):.2f}" for g in (28, 29)], ["5.23", "4.71"])
        self.assertEqual([f"{100 * float(all_missed(100, 10, g)):.2f}" for g in (24, 25)], ["5.51", "4.79"])
        self.assertEqual(all_missed(100, 10, 24), Fraction(math.comb(76, 10), math.comb(100, 10)))

    def test_the_definition(self):
        rng = random.Random(3505)
        cases = [(100, 10, Fraction(1, 20)), (10, 10, Fraction(1, 20)), (100, 1, Fraction(1, 20)), (20, 5, Fraction(1, 2)),
                 (4, 2, Fraction(1, 6)), (4, 2, Fraction(1, 2))]
        cases += [(rng.randint(1, 60), 0, Fraction(rng.randint(1, 99), 100)) for _ in range(300)]
        for total, kept, miss in cases:
            kept = kept or rng.randint(1, total)
            g = 0
            while g < total and all_missed(total, kept, g + 1) >= miss:
                g += 1
            self.assertEqual(sampling.most_of_quota(total, kept, miss), g, (total, kept, miss))
            self.assertLessEqual(g, total - kept)
        self.assertEqual(sampling.most_of_quota(10, 10), 0)                      # every unit is kept
        self.assertEqual(sampling.most_of_quota(100, 1), 95)                     # one unit in a hundred is looked at
        self.assertEqual(sampling.most_of_quota(4, 2, Fraction(1, 6)), 2)        # the chance is reached exactly

    def test_what_is_refused(self):
        for total, kept, miss in ((10, 11, 0.05), (10, 0, 0.05), (0, 0, 0.05), (10, 2.5, 0.05), (10, 2, 0), (10, 2, 1), (-5, 2, 0.05)):
            with self.assertRaises(ValueError, msg=(total, kept, miss)):
                sampling.most_of_quota(total, kept, miss)

    def test_the_transcript(self):
        lines = expected("ch08_sampling").splitlines()
        self.assertEqual(lines[21:], ["an empty record: at 95 %, at most", "  rate 0.01: 298 units", "  rate 0.10:  28 units",
                                      "  10 of 100 kept: 24 units"])


class TheSeventeenthFigure(unittest.TestCase):
    def test_what_it_shows(self):
        xs = omission.closed_loop()
        n, fast = len(xs), xs.count(1.0)
        self.assertEqual((n, fast), (5800, 5798))
        # The inverse of the empirical distribution steps at the share 5798/5800 and has the lower value there.
        step = Fraction(fast, n)
        self.assertEqual(f"{100 * float(step):.4f}", "99.9655")
        self.assertEqual((stats.ecdf_quantile(xs, step), stats.ecdf_quantile(xs, step + Fraction(1, 10 ** 9))), (1.0, 1001.0))
        # The interpolated percentile rises between the ranks 5797 and 5798 of 5799, counted from 0.
        start, end = Fraction(fast - 1, n - 1), Fraction(fast, n - 1)
        self.assertEqual((f"{100 * float(start):.3f}", f"{100 * float(end):.3f}"), ("99.966", "99.983"))
        self.assertEqual(stats.quantile(xs, float(start) - 1e-9), 1.0)
        self.assertAlmostEqual(stats.quantile(xs, float(start)), 1.0, places=6)      # the float is not the fraction
        self.assertAlmostEqual(stats.quantile(xs, float(end)), 1001.0, places=6)
        self.assertEqual(stats.quantile(xs, float(end) + 1e-9), 1001.0)
        middle = stats.quantile(xs, float((start + end) / 2))
        self.assertAlmostEqual(middle, 501.0, places=6)
        # At the 99.97th percentile the two values lie on the two sides of 500 ms.
        self.assertEqual((f"{stats.quantile(xs, 0.9997):.1f}", stats.ecdf_quantile(xs, 0.9997)), ("261.3", 1001.0))

    def test_the_script(self):
        source = (ROOT / "figures" / "make_figures.py").read_text(encoding="utf-8")
        for piece in ("def two_conventions():", "measured = c10.closed_loop()", "quantile(measured, 0.9997), ecdf_quantile(measured, 0.9997)",
                      'save(fig, "fig-10-2-two-conventions.png")', "    omission()\n    two_conventions()\n    calibration_readings()\n    control_chart()\n"):
            self.assertIn(piece, source)
        self.assertEqual(source.count("\n    save(fig, "), 32)                   # sixteen in Version 35


class TheCorrectionOfChapter12(unittest.TestCase):
    def test_the_distance_in_standard_uncertainties(self):
        # The indication lies above the corrected value by the correction. With the two numbers of the transcript
        # the ratio is 8.4, and with the unrounded ones 8.6: more than eight in both.
        rng = random.Random(12)
        xs, fit = probe.calibrate(rng)
        result = probe.measure(37.90, xs, fit)
        text = expected("ch12_calibration")
        self.assertIn("correction        -1.35 ms\n", text)
        self.assertIn("measured value    36.55(16) ms\n", text)
        distance = 37.90 - result.value
        self.assertEqual((f"{distance:.2f}", f"{result.u:.2f}"), ("1.35", "0.16"))
        self.assertEqual((f"{1.35 / 0.16:.1f}", f"{distance / result.u:.1f}"), ("8.4", "8.6"))


class TheScoresOfListing172(unittest.TestCase):
    def test_they_stay_far_from_an_overflow(self):
        # The logistic function of ch17_seeds is written as that of Version 35 was. Its exponential overflows
        # for a score below -709.78; in the runs of the listing no score is beyond 10 in size.
        largest = [0.0]
        dot = seeds.dot

        def watched(w, x):
            value = dot(w, x)
            largest[0] = max(largest[0], abs(value))
            return value
        seeds.dot = watched
        try:
            printed(seeds.main)
        finally:
            seeds.dot = dot
        self.assertLess(largest[0], 10)
        self.assertGreater(largest[0], 5)
        self.assertEqual(raised(math.exp, 709.79), "OverflowError")
        self.assertIsNone(raised(math.exp, 709.78))


if __name__ == "__main__":
    unittest.main()
