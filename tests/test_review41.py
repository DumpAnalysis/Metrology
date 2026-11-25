"""Regression tests added after the review of Version 41.

The review found that offset() of Listing 6.3 added its bound to a stamp before comparing, so that stamps in
nanoseconds since 1970 were judged by floats 256 ns apart, and that holds() divided by a rate of zero. It asked
that Chapter 17 not attribute the gain of a paired comparison to pairing as such, with a second set of joint
outcomes at the same two scores, which Listing 17.1 now prints and Figure 17.1 draws, and it gave two examples
that the book states in its text: stamps that are only ever late, and a conformance probability with and
without earlier results.

Each test computes what it expects independently of the code under test: offsets in fractions by placing the
four events one after another; the paired statistics from the lists item by item, and the exact test from
binomial coefficients; the posterior of a measurand by summing prior times likelihood over a grid.
"""
import contextlib
import io
import math
import random
import statistics
import unittest
from fractions import Fraction
from pathlib import Path

from examples import ch06_exchange as exchange
from examples import ch14_decisions as decisions
from examples import ch17_accuracy as accuracy

ROOT = Path(__file__).resolve().parent.parent
ORIGIN = 1_700_000_000_000_000_000                                              # ns since 1970, late in 2023


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


def version_41_offset(a1, b2, b3, a4, err=0):
    """offset() as Version 41 printed it."""
    w = 2 * err
    lo = max(b2, b3) - a4 - w
    hi = min(b2, b3) - a1 + w
    if a1 > a4 + w or b2 > b3 + w or lo > hi:
        raise ValueError("no offset fits the four events")
    return lo, hi


def placed(stamps, before, after, thetas):
    """The offsets among `thetas` for which the four events of an exchange can follow one another, when each
    event lies between `before` earlier and `after` later than its stamp. In fractions."""
    a1, b2, b3, a4 = stamps
    good = []
    for theta in thetas:
        at = -math.inf
        for stamp in (a1, b2 - theta, b3 - theta, a4):
            at = max(at, stamp - before)
            if at > stamp + after:
                break
        else:
            good.append(theta)
    return good


ROWS = ((0, 2080, 100, 5000), (0, 2020, 0, 5000))


class AnOriginInNanoseconds(unittest.TestCase):
    GRID = [Fraction(k * 20) for k in range(-400, 401)]                          # offsets from -8000 to 8000 ns

    def test_the_table_of_the_review(self):
        first, second = ROWS
        self.assertEqual(version_41_offset(*first, err=1000.0), (-4920.0, 2100.0))
        self.assertIsNotNone(refusal(version_41_offset, *second, err=1000.0))
        moved = [[ORIGIN + s for s in row] for row in ROWS]
        self.assertIsNotNone(refusal(version_41_offset, *moved[0], err=1000.0))  # refused at the other origin
        self.assertEqual(version_41_offset(*moved[1], err=1000.0), (-4980.0, 2000.0))    # and accepted there
        for row in (first, moved[0]):
            self.assertEqual(exchange.offset(*row, err=1000.0), (-4920.0, 2100.0))
        for row in (second, moved[1]):
            self.assertEqual(refusal(exchange.offset, *row, err=1000.0), "no offset fits the four events")

    def test_why(self):
        self.assertEqual(math.ulp(float(ORIGIN)), 256.0)
        self.assertEqual(float(ORIGIN + 2080), float(ORIGIN + 2020))             # one float for both readings
        self.assertEqual(ORIGIN % 256, 0)
        self.assertNotEqual(float(ORIGIN + 100) + 2000.0, ORIGIN + 2100)         # the sum of Version 41

    def test_the_events_placed(self):
        want = placed(ROWS[0], 1000, 1000, self.GRID)
        self.assertEqual((min(want), max(want)), (-4920, 2100))
        self.assertEqual(placed(ROWS[1], 1000, 1000, self.GRID), [])
        self.assertEqual(placed((0, 2000, 0, 5000), 1000, 1000, self.GRID)[-1], 2000)    # 2000 ns apart still fit

    def test_any_origin(self):
        rng = random.Random(1700)
        fits = refused = 0
        for _ in range(600):
            a1 = rng.randrange(0, 4000)
            b2 = a1 + rng.randrange(-3000, 6000)
            b3 = b2 + rng.randrange(-2600, 3000)
            a4 = a1 + rng.randrange(-2600, 9000)
            err = rng.choice((0, 250.0, 1000.0, 1250.5))
            stamps = (a1, b2, b3, a4)
            good = placed(stamps, Fraction(err), Fraction(err), [Fraction(k * 10) for k in range(-1600, 1601)])
            for origin in (0, 10 ** 15, ORIGIN, 2 ** 62):
                moved = [origin + s for s in stamps]
                if good:
                    lo, hi = exchange.offset(*moved, err=err)
                    self.assertTrue(lo <= good[0] and good[-1] <= hi, (stamps, err, origin))
                    self.assertTrue(good[0] - lo < 10 and hi - good[-1] < 10)    # to the step of the grid
                    self.assertEqual((lo, hi), exchange.offset(*stamps, err=err))
                else:
                    self.assertIsNotNone(refusal(exchange.offset, *moved, err=err), (stamps, err, origin))
            fits += bool(good)
            refused += not good
        self.assertGreater(min(fits, refused), 100)

    def test_the_exchanges_of_the_book(self):
        # From Version 43 offset() returns exact fractions, -32/5 where Version 41 had the float next to -6.4.
        for stamps in exchange.EXCHANGES:
            for err in (0, 0.5):                                                 # floats hold these sums exactly
                self.assertEqual(exchange.offset(*stamps, err=err), version_41_offset(*stamps, err=err))
            for now, then in zip(exchange.offset(*stamps, err=0.2), version_41_offset(*stamps, err=0.2)):
                self.assertAlmostEqual(float(now), then, places=12)
        self.assertEqual(exchange.offset(*exchange.EXCHANGES[0], err=0.2), (Fraction(-32, 5), Fraction(-8, 5)))


class AnOffsetThatCannotMove(unittest.TestCase):
    SPAN = (-6, Fraction(-7, 2))

    def test_a_rate_of_zero(self):
        self.assertEqual(exchange.holds(0, -3, self.SPAN, 0, 0), math.inf)       # B later by 0.5 ms at least
        self.assertEqual(exchange.holds(0, -12, self.SPAN, 0, 0), math.inf)      # A later
        self.assertEqual(exchange.holds(0, -5, self.SPAN, 0, 0), 0.0)            # open
        self.assertEqual(exchange.holds(0, Fraction(-7, 2), self.SPAN, 0, 0), 0.0)       # zero is still possible
        # Every stamp within 1/8 ms, those of the exchanges as well: the offsets are -6.25 to -3.25 ms, and
        # the two events can coincide.
        wide = (Fraction(-25, 4), Fraction(-13, 4))
        self.assertEqual(exchange.holds(0, -3, wide, Fraction(1, 8), 0), 0.0)
        self.assertEqual(exchange.holds(0, -3, wide, Fraction(1, 8), 0.0), 0.0)
        self.assertEqual(exchange.holds(0, -3, self.SPAN, Fraction(1, 8), 0), math.inf)  # exact exchanges: 0.25 ms

    def test_a_rate_that_falls_to_zero(self):
        for rate in (Fraction(1, 50), Fraction(1, 5000), Fraction(1, 10 ** 12)):
            self.assertEqual(exchange.holds(0, -3, self.SPAN, 0, rate), Fraction(1, 2) / rate)
            self.assertEqual(exchange.holds(0, -5, self.SPAN, 0, rate), 0)

    def test_what_is_a_rate(self):
        for bad in (-1, -1e-9, float("nan"), math.inf, -math.inf):
            self.assertEqual(refusal(exchange.holds, 0, -3, self.SPAN, 0, bad), "rate must be >= 0", bad)
        self.assertEqual(exchange.holds(0, -3, self.SPAN), 25.0)

    def test_the_transcript(self):
        text = expected("ch06_exchange")
        lines = text.splitlines()
        self.assertEqual(len(lines), 39)                                         # 18 in Version 41, 21 until 43, 29 in 44
        rows = []
        for e in (Fraction(0), Fraction(1, 10), Fraction(1, 8)):
            seconds = max(0, (Fraction(1, 2) - 4 * e) * 50)
            said = f"for less than {float(seconds):.1f} s" if seconds else "order open at once"
            rows.append(f"  {float(e):.3f} ms      {said}")
        self.assertEqual(lines[13:18], ["an offset that moves by 0.020 ms/s;", "stamps within   B is the later"] + rows)
        self.assertEqual(lines[17], "  0.125 ms      order open at once")       # "0.0 s" in Version 41
        self.assertEqual(printed(exchange.main), text)


FIRST = (100, 98, 99, 105)
SECOND = (200, Fraction(393, 2), Fraction(395, 2), Fraction(409, 2))


class StampsWrittenLate(unittest.TestCase):
    D = Fraction(1, 5)                                                           # the bound of the review, ms
    GRID = [Fraction(k, 20) for k in range(-200, 1)]                             # offsets from -10 to 0 ms

    def gaps(self, before, after):
        """The time from the event on A to the one stamped 3 ms earlier on B, for every offset that both
        exchanges allow and every place of the two events between `before` and `after` of their stamps."""
        good = [t for t in placed(FIRST, before, after, self.GRID) if t in placed(SECOND, before, after, self.GRID)]
        found = [(-3 + move_b) - t - move_a for t in good
                 for move_a in (-before, after) for move_b in (-before, after)]
        return min(found), max(found)

    def test_the_table_of_the_review(self):
        self.assertEqual(self.gaps(self.D, self.D), (Fraction(-3, 10), Fraction(19, 5)))     # early or late
        self.assertEqual(self.gaps(self.D, 0), (Fraction(1, 10), Fraction(17, 5)))           # written late only
        self.assertEqual(self.gaps(0, 0), (Fraction(1, 2), 3))                               # exact
        # What the listing gives: the bound either way, and half the bound after half of it is taken off.
        self.assertEqual(exchange.late(self.D), self.D / 2)
        for err, want in ((self.D, (Fraction(-3, 10), Fraction(19, 5))), (exchange.late(self.D), (Fraction(1, 10), Fraction(17, 5)))):
            span = exchange.both([exchange.offset(*stamps, err=err) for stamps in (FIRST, SECOND)])
            self.assertEqual(exchange.gap(0, -3, span, err=err), want)

    def test_half_of_the_delay_taken_off(self):
        # The stamps as they are and with half of the delay taken off every one: the same offsets and the
        # same gap, so that the listing need not take it off.
        half = exchange.late(self.D)
        moved = [[s - half for s in stamps] for stamps in (FIRST, SECOND)]
        spans = [exchange.both([exchange.offset(*stamps, err=half) for stamps in rows]) for rows in ((FIRST, SECOND), moved)]
        self.assertEqual(spans[0], spans[1])
        self.assertEqual(exchange.gap(0 - half, -3 - half, spans[1], half), exchange.gap(0, -3, spans[0], half))
        # After it every event is within half of the delay of its stamp, either way.
        for stamp in (0, -3):
            for event in (stamp - self.D, stamp):                                # the earliest and the latest event
                self.assertLessEqual(abs(event - (stamp - half)), half)

    def test_what_is_a_delay(self):
        for bad in (-1, -1e-9, float("nan"), math.inf):
            self.assertEqual(refusal(exchange.late, bad), "delay must be >= 0", bad)
        self.assertEqual((exchange.late(0), exchange.late(0.2)), (0, 0.1))

    def test_the_transcript(self):
        lines = expected("ch06_exchange").splitlines()
        want = []
        for name, before, after in (("either way", self.D, self.D), ("late only", self.D, 0)):
            lo, hi = self.gaps(before, after)
            want.append(f"{name:<12}{float(lo):>+7.2f}{float(hi):>+7.2f}")
        self.assertEqual(lines[18:21], ["B after A by, stamps off by 0.2 ms"] + want)
        self.assertEqual(want, ["either way    -0.30  +3.80", "late only     +0.10  +3.40"])

    def test_one_delay_for_each_host(self):
        # Every stamp of a host is late by the same time. The offset that the exchanges give has the two
        # delays in it, and so have the stamps of the two events: the gap comes out as it is.
        rng = random.Random(20)
        for _ in range(500):
            theta = Fraction(rng.randrange(-400, 400), 16)
            late_a, late_b = (Fraction(rng.randrange(0, 80), 16) for _ in range(2))
            out, back = (Fraction(rng.randrange(0, 80), 16) for _ in range(2))
            t1 = Fraction(rng.randrange(0, 1000), 16)
            t2 = t1 + out
            t3 = t2 + Fraction(rng.randrange(0, 40), 16)
            t4 = t3 + back
            span = exchange.offset(t1 + late_a, t2 + theta + late_b, t3 + theta + late_b, t4 + late_a)
            self.assertTrue(span[0] <= theta + late_b - late_a <= span[1])
            at_a, at_b = (Fraction(rng.randrange(0, 1000), 16) for _ in range(2))
            early, late = exchange.gap(at_a + late_a, at_b + theta + late_b, span)
            self.assertTrue(early <= at_b - at_a <= late)
            self.assertEqual(late - early, out + back)                           # no wider than without the delays


def counts(a, b):
    both = sum(1 for x, y in zip(a, b) if x and y)
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if y and not x)
    return both, only_a, only_b, len(a) - both - only_a - only_b


def uneven(only_a, only_b):
    """Two-sided exact probability of a division at least as uneven, from binomial coefficients."""
    n, low = only_a + only_b, min(only_a, only_b)
    tail = Fraction(sum(math.comb(n, k) for k in range(low + 1)), 2 ** n)
    return float(min(1, 2 * tail))


class TwoModelsWithOneScore(unittest.TestCase):
    def lists(self):
        return accuracy.two_models(2000, random.Random(17)), accuracy.same_scores()

    def test_the_table_of_the_review(self):
        first, second = self.lists()
        self.assertEqual(counts(*first), (1610, 112, 148, 130))
        self.assertEqual(counts(*second), (1500, 222, 258, 20))
        want = (("+0.427", "0.805", "0.029765"), ("-0.060", "1.095", "0.110053"))
        for (a, b), (r, u, p) in zip((first, second), want):
            self.assertEqual((sum(a), sum(b)), (1722, 1758))
            a, b = [int(x) for x in a], [int(x) for x in b]
            self.assertEqual(f"{statistics.correlation(a, b):+.3f}", r)
            gains = [y - x for x, y in zip(a, b)]                                # item by item
            self.assertEqual(f"{100 * math.sqrt(statistics.pvariance(gains) / 2000):.3f}", u)
            both, only_a, only_b, neither = counts(a, b)
            self.assertEqual(f"{uneven(only_a, only_b):.6f}", p)
            self.assertEqual(f"{float(accuracy.two_sided(only_a, only_b)[1]):.6f}", p)   # a fraction since Version 45
            unpaired = math.sqrt((statistics.pvariance(a) + statistics.pvariance(b)) / 2000)
            self.assertEqual(f"{100 * unpaired:.3f}", "1.063")
            cov = statistics.fmean(x * y for x, y in zip(a, b)) - statistics.fmean(a) * statistics.fmean(b)
            self.assertAlmostEqual(statistics.pvariance(gains),
                                   statistics.pvariance(a) + statistics.pvariance(b) - 2 * cov, places=12)
            self.assertEqual(cov > 0, r.startswith("+"))

    def test_every_joint_count(self):
        ps = {both: uneven(1722 - both, 1758 - both) for both in range(1480, 1723)}
        self.assertEqual(len(ps), 243)
        self.assertEqual(min(both for both, p in ps.items() if p < 0.05), 1581)
        ordered = [ps[both] for both in sorted(ps)]
        self.assertTrue(all(x >= y for x, y in zip(ordered, ordered[1:])))       # more agreement, less probability
        self.assertEqual((f"{ps[1500]:.3f}", f"{ps[1610]:.3f}"), ("0.110", "0.030"))
        for both in (1480, 1500, 1581, 1610, 1700, 1722):
            self.assertAlmostEqual(accuracy.two_sided(1722 - both, 1758 - both)[1], ps[both], places=12)
        self.assertEqual(1722 + 1758 - 2000, 1480)                               # no fewer can be right in both

    def test_what_the_text_says_of_them(self):
        n, right_a, right_b = 2000, 1722, 1758
        independent = Fraction(right_a * right_b, n)                             # both right, if the answers do not depend
        self.assertEqual((float(independent), f"{float(independent) / n:.4f}"), (1513.638, "0.7568"))
        self.assertEqual((f"{1610 / n:.4f}", f"{1500 / n:.4f}"), ("0.8050", "0.7500"))
        for both in range(1480, 1723):
            b, c = right_a - both, right_b - both
            paired = Fraction(b + c - Fraction((c - b) ** 2, n), n ** 2)         # the square of the formula in the text
            a = [1] * right_a + [0] * (n - right_a)
            other = [1] * both + [0] * b + [1] * c + [0] * (n - both - b - c)
            if both in (1480, 1500, 1513, 1514, 1610, 1722):                     # item by item, for some of them
                gains = [y - x for x, y in zip(a, other)]
                self.assertAlmostEqual(statistics.pvariance(gains) / n, float(paired), places=15)
            unpaired = Fraction(right_a * (n - right_a) + right_b * (n - right_b), n ** 3)
            self.assertEqual(paired < unpaired, both >= 1514, both)              # more than 1513.6, and no tie
            self.assertEqual(unpaired - paired, 2 * (Fraction(both, n) - Fraction(right_a * right_b, n * n)) / n)
        self.assertEqual([f"{100 * math.sqrt((b + c - (c - b) ** 2 / n) / n ** 2):.2f}" for b, c in ((112, 148), (222, 258))],
                         ["0.81", "1.09"])
        # B on the items that A answers correctly, and on the others, in the second pair of lists.
        self.assertEqual((f"{100 * 1500 / 1722:.1f}", f"{100 * 258 / 278:.1f}"), ("87.1", "92.8"))
        # The two ends of Figure 17.1: no item wrong for both, and B right wherever A is.
        self.assertEqual((n - right_a - right_b + 1480, right_a - 1722), (0, 0))
        self.assertEqual(f"{uneven(242, 278):.3f}", "0.125")
        self.assertEqual(uneven(0, 36), 2 / 2 ** 36)
        self.assertEqual(f"{uneven(0, 36) * 1e11:.1f}", "2.9")                   # about 3 in 100 billion

    def test_scores_that_do_not_vary(self):
        out = printed(accuracy.compare, [1, 1, 1, 1], [1, 0, 1, 1])
        self.assertIn("correlation of right answers undefined\n", out)
        self.assertIn("correlation of right answers undefined\n", printed(accuracy.compare, [0, 0], [0, 0]))

    def test_the_transcript(self):
        text = expected("ch17_accuracy")
        lines = text.splitlines()
        self.assertEqual(len(lines), 43)                                         # 12 in Version 41, 23 in 42, 27 in 43, 28 in 44
        self.assertEqual(lines[4:6], ["accuracy A 86.10 %, B 87.90 %", "only A right 112, only B right 148"])
        self.assertEqual(lines[6], "correlation of right answers +0.427")
        self.assertEqual(lines[7:13], [
            "difference B - A  1.80 percentage points",
            "u, unpaired   1.06 points, difference / u = 1.69",
            "u, paired     0.81 points, difference / u = 2.24",
            "exact test for the pairs that differ:",
            "  B ahead, one-sided  p = 0.014882",
            "  two-sided           p = 0.029765"])
        self.assertEqual(lines[13:17], ["the same scores, other pairs:", "accuracy A 86.10 %, B 87.90 %",
                                        "only A right 222, only B right 258", "correlation of right answers -0.060"])
        a, b = accuracy.same_scores()
        u = 100 * math.sqrt(statistics.pvariance([y - x for x, y in zip(a, b)]) / 2000)
        self.assertEqual(lines[19], f"u, paired     {u:.2f} points, difference / u = {1.8 / u:.2f}")
        self.assertEqual(lines[19], "u, paired     1.09 points, difference / u = 1.64")
        self.assertEqual(lines[22], f"  two-sided           p = {uneven(222, 258):.6f}")
        self.assertEqual(printed(accuracy.main), text)


def below_limit(y, prior=None, limit=200.0, u=3.0):
    """P(X <= limit) after a reading y that is normal about X with standard deviation u, for a flat prior or a
    normal one (mean, sd): prior times likelihood summed over a grid of X, 0.002 ms apart."""
    xs = [150 + k * 0.002 for k in range(50_001)]                                # 150 to 250 ms
    weights = []
    for x in xs:
        w = math.exp(-0.5 * ((y - x) / u) ** 2)
        if prior:
            w *= math.exp(-0.5 * ((x - prior[0]) / prior[1]) ** 2)
        weights.append(w)
    return math.fsum(w for x, w in zip(xs, weights) if x <= limit) / math.fsum(weights)


def acceptance(prior=None):
    """The reading at which P(X <= 200) is 0.95, by bisection on the sum above."""
    lo, hi = 170.0, 200.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if below_limit(mid, prior) > 0.95:
            lo = mid
        else:
            hi = mid
    return lo


class AReadingAndALimit(unittest.TestCase):
    HISTORY = (205.0, 3.0)

    def test_the_table_of_the_review(self):
        want = {195: ("95.22", "50.00"), 197: ("84.13", "31.87"), 200: ("50.00", "11.93")}
        for y, (flat, informed) in want.items():
            self.assertAlmostEqual(100 * below_limit(y), float(flat), places=1)
            self.assertAlmostEqual(100 * below_limit(y, self.HISTORY), float(informed), places=1)
        # The closed forms: the reading alone, and the mean of reading and history with 3 over the root of 2.
        normal = statistics.NormalDist()
        for y in (195, 197, 200):
            self.assertAlmostEqual(below_limit(y), normal.cdf((200 - y) / 3), places=3)
            closed = normal.cdf((200 - (205 + y) / 2) / (3 / math.sqrt(2)))
            self.assertAlmostEqual(below_limit(y, self.HISTORY), closed, places=3)
        self.assertEqual(f"{100 * normal.cdf(5 / 3):.2f}", "95.22")
        self.assertEqual(f"{100 * normal.cdf((200 - 201) / (3 / math.sqrt(2))):.2f}", "31.87")

    def test_the_acceptance_limits(self):
        self.assertAlmostEqual(acceptance(), 195.07, places=1)
        self.assertAlmostEqual(acceptance(self.HISTORY), 188.02, places=1)
        z = statistics.NormalDist().inv_cdf(0.95)
        self.assertEqual((f"{200 - 3 * z:.4f}", f"{2 * (200 - z * 3 / math.sqrt(2)) - 205:.2f}"), ("195.0654", "188.02"))

    def test_repeated_readings_at_201(self):
        # The measurand held at 201 ms: how often a reading falls at or below the acceptance limit.
        limit = 200 - 3 * statistics.NormalDist().inv_cdf(0.95)
        rate = statistics.NormalDist(201, 3).cdf(limit)
        self.assertEqual(f"{100 * rate:.2f}", "2.40")
        rng = random.Random(14)
        runs = 200_000
        seen = sum(rng.gauss(201, 3) <= limit for _ in range(runs)) / runs
        self.assertLess(abs(seen - rate), 4 * math.sqrt(rate * (1 - rate) / runs))

    def test_the_listing_is_what_it_was(self):
        text = expected("ch14_decisions")
        # Until Version 59 the transcript began "acceptance limit 195.1 ms": the limit rounded to the nearest,
        # above the working limit of 195.0654 ms. The decisions of the rows are what they were.
        self.assertEqual(len(text.splitlines()), 14)
        self.assertIn("   195 ms       0.952  accept\n", text)
        self.assertTrue(text.startswith("accept up to 195.06, reject above 204.94 ms\n"))
        self.assertEqual((decisions.LIMIT, decisions.U, decisions.RISK), (200.0, 3.0, 0.05))


class TheTwentyThirdFigure(unittest.TestCase):
    def test_the_script(self):
        source = (ROOT / "figures" / "make_figures.py").read_text(encoding="utf-8")
        for piece in ("def joint_outcomes():", "c17.two_sided(right_a - both, right_b - both)[1]", "c17.same_scores()",
                      'save(fig, "fig-17-1-joint-outcomes.png")', "    shared_starts()\n    joint_outcomes()\n    unlinked_pairs()\n    batch_labels()\n    reliability()\n",
                      '"simultaneous events possible"', "marks = [(seconds(e), e) for e in (0, 0.1, top)]"):
            self.assertIn(piece, source)
        self.assertEqual(source.count("\n    save(fig, "), 32)                   # twenty-two in Version 41

    def test_what_it_shows(self):
        self.assertTrue(all(0 <= uneven(1722 - both, 1758 - both) < 0.135 for both in range(1480, 1723)))
        # Figure 6.3: its three marked points lie on the line, where the two events can coincide.
        for e, seconds in ((0, 25), (Fraction(1, 10), 5), (Fraction(1, 8), 0)):
            self.assertEqual(Fraction(1, 2) - 4 * e - Fraction(seconds, 50), 0)


if __name__ == "__main__":
    unittest.main()
