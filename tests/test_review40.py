"""Regression tests added after the review of Version 40.

The review found that offset() of Listing 6.3 let host B reply before the request had arrived, and so allowed
offsets that the order of the four events of an exchange excludes; that lpu() of swmetro.propagate took a
negative standard uncertainty and returned zero for a sum; and that lognormal_ratio() of Listing 5.1 lost a
relative uncertainty of 1e-9 in 1 + 1e-18. It showed that correlations computed from complete requests and
rounded one by one can be refused, and it proposed an addition to Chapter 6, the age up to which a calibration
still settles an order, which is holds() of Listing 6.3 and Figure 6.3.

Each test computes what it expects independently of the code under test: the offsets of an exchange by placing
the four events one after another, as early as their stamps allow; the sample statistics of three requests in
fractions; the moments of a lognormal ratio in 65 decimal digits.
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


def version_40_offset(a1, b2, b3, a4, err=0):
    """offset() as Version 40 printed it."""
    lo = b3 - a4 - 2 * err
    hi = b2 - a1 + 2 * err
    if lo > hi:
        raise ValueError("no offset fits the four stamps")
    return lo, hi


def in_order(theta, a1, b2, b3, a4, e):
    """Whether four events can follow one another on the time scale of A, each within e of its stamp, for an
    offset theta of clock B: every event is placed as early as its stamp and the event before it allow."""
    at = -math.inf
    for stamp in (a1, b2 - theta, b3 - theta, a4):
        at = max(at, stamp - e)
        if at > stamp + e:
            return False
    return True


def tried(a1, b2, b3, a4, e, step=Fraction(1, 8)):
    """The least and the greatest offset on a grid for which the events can be in order, or None."""
    grid = [k * step for k in range(int(-40 / step), int(40 / step) + 1)]
    good = [t for t in grid if in_order(t, a1, b2, b3, a4, e)]
    if not good:
        return None
    assert good == [t for t in grid if good[0] <= t <= good[-1]]                 # an interval
    return good[0], good[-1]


REVIEW = (100, 104, 103, 106)


class TheOrderOfFourEvents(unittest.TestCase):
    def test_the_table_of_the_review(self):
        self.assertEqual(version_40_offset(*REVIEW), (-3, 4))                    # B replies before it receives
        self.assertEqual(version_40_offset(*REVIEW, err=1), (-5, 6))
        self.assertIsNone(tried(*REVIEW, 0))
        self.assertEqual(tried(*REVIEW, 1), (-4, 5))
        self.assertEqual(refusal(exchange.offset, *REVIEW), "no offset fits the four events")
        self.assertEqual(exchange.offset(*REVIEW, err=1), (-4, 5))

    def test_what_follows_for_two_other_events(self):
        # Stamped 7.5 ms later on B than on A, every stamp within 1 ms.
        self.assertEqual(exchange.gap(0, 7.5, version_40_offset(*REVIEW, err=1), err=1), (-0.5, 14.5))
        self.assertEqual(exchange.gap(0, 7.5, exchange.offset(*REVIEW, err=1), err=1), (0.5, 13.5))
        # By trying: the two stamps moved within 1 ms, and every offset for which the four events are in order.
        gaps = [Fraction(15, 2) + move - t for move in (-2, 0, 2)
                for t in (Fraction(k, 8) for k in range(-80, 81)) if in_order(t, *REVIEW, 1)]
        self.assertEqual((min(gaps), max(gaps)), (Fraction(1, 2), Fraction(27, 2)))
        self.assertGreater(min(gaps), 0)                                         # the event on B came second

    def test_placing_the_events_is_right(self):
        # The reference itself, against every choice of four event times on a grid of quarters.
        rng = random.Random(4)
        for _ in range(40):
            stamps = [rng.randrange(0, 12) for _ in range(4)]
            e = rng.choice((0, 1, 2))
            theta = rng.randrange(-8, 9)
            a1, b2, b3, a4 = stamps
            steps = [Fraction(k, 2) for k in range(-2 * e, 2 * e + 1)]
            found = any(a1 + d1 <= b2 + d2 - theta <= b3 + d3 - theta <= a4 + d4
                        for d1, d2, d3, d4 in itertools.product(steps, repeat=4))
            self.assertEqual(in_order(theta, a1, b2, b3, a4, e), found, (stamps, e, theta))

    def test_against_the_events_placed(self):
        rng = random.Random(40)
        fits = refused = 0
        for _ in range(400):
            a1 = rng.randrange(0, 80)
            b2 = a1 + rng.randrange(-40, 60)
            b3 = b2 + rng.randrange(-16, 60)                                     # also before the request arrived
            a4 = a1 + rng.randrange(-8, 120)
            stamps = [Fraction(x, 8) for x in (a1, b2, b3, a4)]
            e = Fraction(rng.choice((0, 0, 1, 2, 5, 16)), 8)
            want = tried(*stamps, e)
            if want is None:
                refused += 1
                self.assertEqual(refusal(exchange.offset, *stamps, err=e), "no offset fits the four events", stamps)
            else:
                fits += 1
                self.assertEqual(exchange.offset(*stamps, err=e), want, (stamps, e))
        self.assertGreater(min(fits, refused), 80)

    def test_the_exchanges_of_the_book(self):
        self.assertEqual([exchange.offset(*stamps) for stamps in exchange.EXCHANGES], [(-6, -2), (-7.0, -3.5)])
        for stamps in exchange.EXCHANGES:
            for e in (0, Fraction(1, 2)):
                self.assertEqual(exchange.offset(*stamps, err=e), version_40_offset(*stamps, err=e))
                self.assertEqual(tuple(map(Fraction, exchange.offset(*stamps, err=e))), tried(*map(Fraction, stamps), e))

    def test_what_is_refused(self):
        self.assertIsNotNone(refusal(exchange.offset, 100, 90, 91, 95))          # A receives before it sends
        self.assertEqual(exchange.offset(100, 90, 91, 95, err=3), (-10, -4))
        self.assertIsNone(tried(100, 90, 91, 95, 0))
        self.assertEqual(tried(100, 90, 91, 95, 3), (-10, -4))
        for bad in (float("nan"), math.inf, -math.inf):
            for place in range(4):
                stamps = [100, 98, 99, 105]
                stamps[place] = bad
                self.assertEqual(refusal(exchange.offset, *stamps), "need finite numbers", stamps)
            self.assertEqual(refusal(exchange.offset, 100, 98, 99, 105, err=bad), "need finite numbers")
        self.assertEqual(refusal(exchange.offset, 100, 98, 99, 105, err=-0.5), "err must be >= 0")
        self.assertEqual(version_40_offset(100, float("nan"), 99, 105)[0], -6)   # half an answer, before


def aged_gap(e, age, rate=Fraction(20, 1000)):
    """The time from the event on A to the one stamped 3 ms earlier on B, age seconds after the two exchanges
    of the book, with every stamp within e ms of its event and an offset that moves by rate at most."""
    spans = [(b3 - a4 - 2 * e, b2 - a1 + 2 * e) for a1, b2, b3, a4 in
             ((100, 98, 99, 105), (200, Fraction(393, 2), Fraction(395, 2), Fraction(409, 2)))]
    lo = max(s[0] for s in spans) - rate * age                                   # the offsets, moved either way
    hi = min(s[1] for s in spans) + rate * age
    return -3 - hi - 2 * e, -3 - lo + 2 * e


class TheAgeOfACalibration(unittest.TestCase):
    def test_the_interval_of_the_review(self):
        for e in (Fraction(k, 40) for k in range(0, 7)):
            for age in range(0, 31):
                self.assertEqual(aged_gap(e, age), (Fraction(1, 2) - 4 * e - Fraction(age, 50),
                                                    3 + 4 * e + Fraction(age, 50)))

    def test_the_numbers_of_the_review(self):
        self.assertEqual(aged_gap(0, 25)[0], 0)                                  # at 25 s the two can coincide
        self.assertGreater(aged_gap(0, Fraction(2499, 100))[0], 0)
        self.assertEqual(aged_gap(Fraction(1, 10), 5)[0], 0)
        self.assertEqual(aged_gap(Fraction(1, 8), 0)[0], 0)                      # 125 us use up the margin at once
        self.assertEqual(exchange.RATE, 0.020)
        got = []
        for e in (0, 0.1, 0.125):
            span = exchange.both([exchange.offset(*stamps, err=e) for stamps in exchange.EXCHANGES])
            got.append(exchange.holds(0, -3, span, e))
        self.assertEqual(got[0], 25.0)
        self.assertAlmostEqual(got[1], 5.0, places=9)
        self.assertEqual(got[2], 0.0)

    def test_holds_against_the_interval(self):
        for k in range(0, 8):
            e = Fraction(k, 50)
            span = exchange.both([exchange.offset(*stamps, err=e) for stamps in ((100, 98, 99, 105),
                                 (200, Fraction(393, 2), Fraction(395, 2), Fraction(409, 2)))])
            seconds = exchange.holds(0, -3, span, e, Fraction(20, 1000))
            self.assertEqual(seconds, max(0, (Fraction(1, 2) - 4 * e) * 50))
            for age in (Fraction(q, 4) for q in range(0, 130)):
                self.assertEqual(aged_gap(e, age)[0] > 0, age < seconds, (e, age))
        # An order the other way, and an order that is open at the exchanges.
        self.assertEqual(exchange.holds(0, -12, (-6, Fraction(-7, 2)), 0, Fraction(1, 50)), 300)   # A later by 6 ms
        self.assertEqual(exchange.holds(0, -5, (-6, Fraction(-7, 2)), 0, Fraction(1, 50)), 0)

    def test_bounds_that_differ(self):
        # Stamps of the exchanges within c, stamps of the two events within t: twice the one and twice the other.
        for c, t in ((Fraction(1, 20), Fraction(1, 10)), (Fraction(1, 8), 0), (0, Fraction(1, 5))):
            span = exchange.both([exchange.offset(*stamps, err=c) for stamps in ((100, 98, 99, 105),
                                 (200, Fraction(393, 2), Fraction(395, 2), Fraction(409, 2)))])
            self.assertEqual(exchange.gap(0, -3, span, err=t), (Fraction(1, 2) - 2 * c - 2 * t, 3 + 2 * c + 2 * t))

    def test_an_offset_that_moved(self):
        # Exchanges at a known offset, then a rate within the bound for some seconds: the events are where
        # the aged interval says, and at its ends when everything moves one way.
        rng = random.Random(41)
        rate = Fraction(1, 50)
        for _ in range(500):
            theta = Fraction(rng.randrange(-400, 400), 16)
            out, back = (Fraction(rng.randrange(0, 80), 16) for _ in range(2))
            a1 = Fraction(rng.randrange(0, 1000), 16)
            b2 = a1 + out + theta
            b3 = b2 + Fraction(rng.randrange(0, 40), 16)
            a4 = b3 - theta + back
            lo, hi = exchange.offset(a1, b2, b3, a4)
            age = Fraction(rng.randrange(0, 400), 8)
            moved = theta + Fraction(rng.randrange(-16, 17), 16) * rate * age    # the offset at the two events
            at_a, at_b = (Fraction(rng.randrange(0, 1000), 16) for _ in range(2))
            early, late = exchange.gap(at_a, at_b + moved, (lo - rate * age, hi + rate * age))
            self.assertTrue(early <= at_b - at_a <= late)
            seconds = exchange.holds(at_a, at_b + moved, (lo, hi), 0, rate)
            if age < seconds:
                first, last = exchange.gap(at_a, at_b + moved, (lo, hi))
                self.assertEqual(first > 0, at_b > at_a)                         # the order that was established

    def test_the_transcript(self):
        text = expected("ch06_exchange")
        lines = text.splitlines()
        self.assertEqual(len(lines), 39)                                         # 13 in Version 40, 18 in 41, 21 until 43, 29 in 44
        # Version 41 printed "0.0 s with stamps within 0.125 ms" in the last line; reworded in Version 42.
        seconds = [float(max(0, (Fraction(1, 2) - 4 * e) * 50)) for e in (0, Fraction(1, 10), Fraction(1, 8))]
        self.assertEqual(seconds, [25.0, 5.0, 0.0])
        self.assertEqual(lines[13:18], ["an offset that moves by 0.020 ms/s;", "stamps within   B is the later",
                                      "  0.000 ms      for less than 25.0 s", "  0.100 ms      for less than 5.0 s",
                                      "  0.125 ms      order open at once"])
        self.assertEqual(printed(exchange.main), text)


class AStandardUncertaintyBelowZero(unittest.TestCase):
    def test_the_example_of_the_review(self):
        # What Version 40 summed: contributions of +1 and -1 with a correlation of 1.
        self.assertEqual(1 * 1 + (-1) * (-1) + 2 * 1 * 1 * (-1), 0)
        self.assertEqual(refusal(propagate.lpu, sum, [10, 10], [1, -1], {(0, 1): 1}), "u must be >= 0")
        self.assertEqual(propagate.lpu(sum, [10, 10], [1, 1], {(0, 1): 1}), (20, 2.0, [1.0, 1.0]))
        # The sign of a contribution belongs to the sensitivity: a difference, with both u above zero.
        self.assertEqual(propagate.lpu(lambda x: x[0] - x[1], [10, 10], [1, 1], {(0, 1): 1}), (0, 0.0, [1.0, -1.0]))

    def test_what_a_standard_uncertainty_is(self):
        for bad in (-1, -0.001, Fraction(-1, 10 ** 9), float("nan"), math.inf, -math.inf):
            for us in ([bad, 1], [1, bad]):
                self.assertEqual(refusal(propagate.lpu, sum, [10, 10], us), "u must be >= 0", us)
        for good in (0, 0.0, -0.0, Fraction(0), 1e-300, Fraction(1, 10 ** 9), 5):
            self.assertEqual(propagate.lpu(sum, [10, 10], [good, 0])[1], float(good))
        self.assertEqual(refusal(propagate.lpu, sum, [10, 10], [1]), "one u for each x")

    def test_the_check_comes_first(self):
        calls = []

        def f(x):
            calls.append(x)
            return sum(x)
        self.assertIsNotNone(refusal(propagate.lpu, f, [10, 10], [1, -1]))
        self.assertEqual(calls, [])


class CompleteRequestsRounded(unittest.TestCase):
    A, B = [10, 10, 11], [9, 10, 10]

    def moments(self):
        a, b = self.A, self.B
        t = [x + y for x, y in zip(a, b)]

        def cov(xs, ys):
            mx, my = Fraction(sum(xs), 3), Fraction(sum(ys), 3)
            return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / 2
        return t, cov

    def test_the_sample(self):
        t, cov = self.moments()
        self.assertEqual(t, [19, 20, 21])
        self.assertEqual((cov(self.A, self.A), cov(self.B, self.B), cov(t, t), cov(self.A, self.B)),
                         (Fraction(1, 3), Fraction(1, 3), 1, Fraction(1, 6)))
        self.assertEqual(cov(self.A, self.B) / cov(self.A, self.A), Fraction(1, 2))          # r(A, B)
        for stage in (self.A, self.B):                                           # r squared with the total: 3/4
            self.assertEqual(cov(stage, t) ** 2 / (cov(stage, stage) * cov(t, t)), Fraction(3, 4))
        self.assertTrue(all(z - x - y == 0 for x, y, z in zip(self.A, self.B, t)))

    def test_the_table_of_the_review(self):
        # The determinant with 0.5 between the stages and r between each stage and the total: 3/4 - r squared.
        for r, good in (("0.8660254037844387", False), ("0.87", False), ("0.8660254037844386", True), ("0.866", True)):
            f = Fraction(r)
            m = [[1, Fraction(1, 2), f], [Fraction(1, 2), 1, f], [f, f, 1]]
            det = (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                   + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
            self.assertEqual(det, Fraction(3, 4) - f * f)
            self.assertEqual(det >= 0, good, r)
            corr = {(0, 1): 0.5, (0, 2): float(r), (1, 2): float(r)}
            self.assertEqual(refusal(propagate.correlations, corr, 3) is None, good, r)
        self.assertEqual(f"{float(Fraction(3, 4) - Fraction('0.8660254037844387') ** 2):.2e}", "-9.22e-17")
        self.assertEqual(float(Fraction(3, 4) - Fraction("0.87") ** 2), -0.0069)

    def test_what_floating_point_returns(self):
        # One of the two floats next to the root of 3/4, which is not a float; which one is left to the library.
        t, _ = self.moments()
        r = statistics.correlation(self.A, t)
        self.assertIn(repr(r), ("0.8660254037844386", "0.8660254037844387"))
        self.assertLess(Fraction("0.8660254037844386") ** 2, Fraction(3, 4))
        self.assertGreater(Fraction("0.8660254037844387") ** 2, Fraction(3, 4))
        corr = {(0, 1): 0.5, (0, 2): r, (1, 2): r}
        self.assertEqual(refusal(propagate.correlations, corr, 3) is None, Fraction(repr(r)) ** 2 <= Fraction(3, 4))

    def test_from_the_two_stages(self):
        # The total from its two stages, with their correlation of one half: the standard deviation of the totals.
        u = math.sqrt(1 / 3)
        y, uc, cu = propagate.lpu(sum, [10, 9], [u, u], {(0, 1): 0.5})
        self.assertAlmostEqual(uc, 1.0, places=12)
        t, cov = self.moments()
        self.assertEqual(cov(t, t), 1)
        self.assertEqual(statistics.pvariance([z - x - y for x, y, z in zip(self.A, self.B, t)]), 0)


def version_40_lognormal_ratio(m1, s1, m2, s2):
    """lognormal_ratio() as Version 40 printed it, with the normal quantile of the standard library."""
    v1 = math.log(1 + (s1 / m1) ** 2)
    v2 = math.log(1 + (s2 / m2) ** 2)
    mu = math.log(m1 / m2) - (v1 - v2) / 2
    sd = math.sqrt(v1 + v2)
    mean = math.exp(mu + sd * sd / 2)
    u = mean * math.sqrt(math.expm1(sd * sd))
    z = statistics.NormalDist().inv_cdf(0.975)
    return mean, u, math.exp(mu - z * sd), math.exp(mu + z * sd)


def in_65_digits(m1, s1, m2, s2):
    """The same four numbers in decimal arithmetic of 65 digits, in a context of its own."""
    ctx = decimal.Context(prec=65)
    m1, s1, m2, s2 = (ctx.create_decimal(repr(float(x))) for x in (m1, s1, m2, s2))
    one = decimal.Decimal(1)

    def log_variance(m, s):
        ratio = ctx.divide(s, m)
        return ctx.ln(ctx.add(one, ctx.multiply(ratio, ratio)))
    v1, v2 = log_variance(m1, s1), log_variance(m2, s2)
    mu = ctx.subtract(ctx.ln(ctx.divide(m1, m2)), ctx.divide(ctx.subtract(v1, v2), decimal.Decimal(2)))
    var = ctx.add(v1, v2)
    sd = ctx.sqrt(var)
    mean = ctx.exp(ctx.add(mu, ctx.divide(var, decimal.Decimal(2))))
    u = ctx.multiply(mean, ctx.sqrt(ctx.subtract(ctx.exp(var), one)))
    z = ctx.create_decimal(repr(statistics.NormalDist().inv_cdf(0.975)))
    return (mean, u, ctx.exp(ctx.subtract(mu, ctx.multiply(z, sd))), ctx.exp(ctx.add(mu, ctx.multiply(z, sd))))


class ASmallUncertaintyOfARatio(unittest.TestCase):
    def test_the_table_of_the_review(self):
        self.assertEqual(1 + 1e-18, 1.0)                                         # where the uncertainty went
        self.assertEqual(version_40_lognormal_ratio(1, 1e-9, 1, 1e-9), (1.0, 0.0, 1.0, 1.0))
        mean, u, lo, hi = chapter5.lognormal_ratio(1, 1e-9, 1, 1e-9)
        self.assertEqual((f"{u:.9e}", f"{lo:.12f}", f"{hi:.12f}"), ("1.414213562e-09", "0.999999997228", "1.000000002772"))
        want = in_65_digits(1, 1e-9, 1, 1e-9)
        self.assertEqual(f"{want[1]:.9e}".lower(), "1.414213562e-9")
        self.assertAlmostEqual(u / float(want[1]), 1.0, places=12)
        self.assertAlmostEqual((hi - 1) / float(want[3] - 1), 1.0, places=7)     # hi - 1 is taken from a float
        self.assertAlmostEqual((1 - lo) / float(1 - want[2]), 1.0, places=7)

    def test_small_and_large_uncertainties(self):
        rng = random.Random(5)
        for _ in range(300):
            m1, m2 = rng.uniform(0.5, 2000), rng.uniform(0.5, 2000)
            r1, r2 = (10 ** rng.uniform(-12, -0.5) for _ in range(2))            # relative uncertainties
            got = chapter5.lognormal_ratio(m1, m1 * r1, m2, m2 * r2)
            want = in_65_digits(m1, m1 * r1, m2, m2 * r2)
            self.assertAlmostEqual(got[0] / float(want[0]), 1.0, places=12)
            self.assertAlmostEqual(got[1] / float(want[1]), 1.0, places=9)
            width = float(want[3] - want[2])
            self.assertAlmostEqual((got[3] - got[2]) / width, 1.0, places=3)

    def test_the_inputs_of_the_book(self):
        got = chapter5.lognormal_ratio(842, 12, 655, 15)
        before = version_40_lognormal_ratio(842, 12, 655, 15)
        for a, b in zip(got, before):
            self.assertAlmostEqual(a / b, 1.0, places=12)
        text = expected("ch05_propagation")
        # Until Version 69 the ends of an interval were printed to the nearest digit. Since Version 70 the
        # lower end is rounded downward and the upper one upward, so that the printed interval holds the
        # computed one (tests/test_review69.py). The line read [1.220, 1.355].
        low, top = math.floor(Fraction(got[2]) * 1000) / 1000, math.ceil(Fraction(got[3]) * 1000) / 1000
        self.assertEqual((low, top), (1.219, 1.356))
        self.assertIn(f"  mean {got[0]:.4f}  sd {got[1]:.4f}\n  central 95 %  [{low:.3f}, {top:.3f}]\n", text)
        self.assertEqual(len(text.splitlines()), 40)                             # as in Version 40
        self.assertEqual(printed(chapter5.main), text)


class TheTwentySecondFigure(unittest.TestCase):
    def test_the_script(self):
        source = (ROOT / "figures" / "make_figures.py").read_text(encoding="utf-8")
        for piece in ("def calibration_age():", "c6x.holds(0, -3, span, bound_ms)", 'save(fig, "fig-06-3-calibration-age.png")',
                      "c6x.gap(0, -3, wide, err=0.5)", "    exchange_gap()\n    calibration_age()\n    third_host()\n    three_host_bound()\n    stamps_at_fault()\n    aliasing()\n"):
            self.assertIn(piece, source)
        self.assertEqual(source.count("\n    save(fig, "), 32)                   # twenty-one in Version 40

    def test_what_they_show(self):
        # Figure 6.3: the line from 125 us at the exchanges to 25 s with exact stamps, through 5 s at 100 us.
        line = [(e, (Fraction(1, 2) - 4 * e) * 50) for e in (0, Fraction(1, 10), Fraction(1, 8))]
        self.assertEqual(line, [(0, 25), (Fraction(1, 10), 5), (Fraction(1, 8), 0)])
        self.assertTrue(all(aged_gap(e, s)[0] == 0 for e, s in line))            # on the line the two can coincide
        self.assertTrue(all(e < 0.158 and s < 32 for e, s in line))              # within the axes
        # Figure 6.2, the fourth row: every stamp within half a millisecond.
        self.assertEqual(aged_gap(Fraction(1, 2), 0), (Fraction(-3, 2), 5))
        self.assertTrue(-1.9 < -1.5 and 5 < 5.4)


if __name__ == "__main__":
    unittest.main()
