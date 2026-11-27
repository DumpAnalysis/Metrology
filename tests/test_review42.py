"""Regression tests added after the review of Version 42.

The review found that Listing 6.3 still put two events in an order that their stamps do not establish, where
one host counts from its start and the other from 1970 and the bound of a stamp is a float: the offset is then
as large as the stamps of the second host, and its sum with the bound was rounded to a float. The listing now
computes in exact fractions. The review asked that Chapter 17 say what two scores on the same items do allow,
limits for the paired uncertainty and for the exact probability; that it pair evaluation records by the
identifier of the item; and what no differing item establishes. For Chapter 12 it asked what further readings
of the unknown and further calibration readings would each buy, which Listing 12.1 now prints and Figure 12.1
draws.

Each test computes what it expects independently of the code under test: offsets and gaps by placing the events
of an exchange one after another in fractions; the exact test from binomial coefficients; the calibration by a
least-squares fit written out here, and by simulating the whole procedure.
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
from examples import ch12_calibration as calibration
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


def version_42_offset(a1, b2, b3, a4, err=0):
    """offset() as Version 42 printed it."""
    w = 2 * err
    lo = max(b2, b3) - a4 - w
    hi = min(b2, b3) - a1 + w
    if a1 - a4 > w or b2 - b3 > w or lo > hi:
        raise ValueError("no offset fits the four events")
    return lo, hi


def version_42_gap(stamp_a, stamp_b, span, err=0):
    d = stamp_b - stamp_a
    return d - span[1] - 2 * err, d - span[0] + 2 * err


def version_42_holds(stamp_a, stamp_b, span, err=0, rate=exchange.RATE):
    lo, hi = version_42_gap(stamp_a, stamp_b, span, err)
    room = max(lo, -hi, 0)
    if not rate:
        return math.inf if room else 0.0
    return room / rate


def placed(stamps, err, thetas):
    """The offsets among `thetas` for which the four events of an exchange can follow one another, when each
    event lies within `err` of its stamp. In fractions."""
    a1, b2, b3, a4 = stamps
    good = []
    for theta in thetas:
        at = -math.inf
        for stamp in (a1, b2 - theta, b3 - theta, a4):
            at = max(at, stamp - err)
            if at > stamp + err:
                break
        else:
            good.append(theta)
    return good


EXCHANGE = (0, 100, 600, 3000)                                                  # ns: A sends, B receives, B replies, A receives
EVENTS = (4000, 8050)                                                           # ns, on A and on B


class AnOriginForEachHost(unittest.TestCase):
    def shifted(self, on_a, on_b):
        a1, b2, b3, a4 = EXCHANGE
        return (a1 + on_a, b2 + on_b, b3 + on_b, a4 + on_a), (EVENTS[0] + on_a, EVENTS[1] + on_b)

    def test_the_table_of_the_review(self):
        rows = ((0, 1000.0, (-50, 10450), 0.0),                                  # small origins on both hosts
                (ORIGIN, 1000.0, (48, 10448), math.inf),                         # 1970 on B, a float for the bound
                (ORIGIN, 1000, (-50, 10450), 0.0))                               # 1970 on B, an integer
        for on_b, err, gap_then, holds_then in rows:
            stamps, (at_a, at_b) = self.shifted(0, on_b)
            span = version_42_offset(*stamps, err)
            self.assertEqual(version_42_gap(at_a, at_b, span, err), gap_then)
            self.assertEqual(version_42_holds(at_a, at_b, span, err, 0), holds_then)
            span = exchange.offset(*stamps, err)
            self.assertEqual((span[0] - on_b, span[1] - on_b), (-4400, 2100))
            self.assertEqual(exchange.gap(at_a, at_b, span, err), (-50, 10450))
            self.assertEqual(exchange.holds(at_a, at_b, span, err, 0), 0.0)
            self.assertEqual(exchange.holds(at_a, at_b, span, err), 0)

    def test_why(self):
        # The ends of the offset are near 1.7 * 10^18 ns, and a float there is a multiple of 256 ns.
        self.assertEqual(int((ORIGIN + 600) - 3000 - 2000.0) - ORIGIN, -4352)    # -4400, in integers
        self.assertEqual(int((ORIGIN + 100) - 0 + 2000.0) - ORIGIN, 2048)        # +2100
        self.assertEqual((8050 + ORIGIN - 4000) - (ORIGIN + 2100) - 2000, -50)
        # The difference of the two stamps of the events, an integer, is rounded as well where it meets the
        # float: to 4096 ns after the origin. Hence the +48 ns of Version 42 for the lower end.
        self.assertEqual((8050 + ORIGIN - 4000) - float(ORIGIN + 2048) - 2000.0, 48.0)
        self.assertEqual(int(float(ORIGIN + 4050)) - ORIGIN, 4096)
        self.assertEqual(math.ulp(float(ORIGIN)), 256.0)

    def test_the_events_placed(self):
        # Offsets that the exchange allows with every stamp within 1000 ns, on a grid of 50 ns; and the time
        # from the event on A to the event on B for each of them, with either event at an end of its interval.
        good = placed(EXCHANGE, 1000, [Fraction(50 * k) for k in range(-200, 201)])
        self.assertEqual((good[0], good[-1]), (-4400, 2100))
        times = [(EVENTS[1] + db - theta) - (EVENTS[0] + da) for theta in good for da in (-1000, 1000) for db in (-1000, 1000)]
        self.assertEqual((min(times), max(times)), (-50, 10450))
        self.assertTrue(min(times) < 0 < max(times))                             # either event can be the first

    def test_two_origins_at_will(self):
        rng = random.Random(42)
        origins = (0, 1, 10 ** 15, ORIGIN, -ORIGIN, 2 ** 62, 3 * 10 ** 20)
        bounds = (0, 250.0, 1000.0, 1250.5, Fraction(1000, 3), 0.1)
        open_orders = settled = 0
        for _ in range(400):
            a1 = rng.randrange(0, 4000)
            theta = rng.randrange(-3000, 3000)
            b2 = a1 + theta + rng.randrange(0, 3000)
            b3 = b2 + rng.randrange(0, 2000)
            a4 = b3 - theta + rng.randrange(0, 3000)
            at_a, at_b = rng.randrange(0, 9000), rng.randrange(0, 9000) + theta
            err = rng.choice(bounds)
            base = exchange.offset(a1, b2, b3, a4, err)
            want_gap = exchange.gap(at_a, at_b, base, err)
            # Independently: the offsets at which the four events can be placed, and the gaps they leave.
            e = Fraction(str(err))
            good = placed((a1, b2, b3, a4), e, [Fraction(theta + 25 * k) for k in range(-400, 401)])
            self.assertTrue(base[0] <= good[0] < base[0] + 25 and base[1] - 25 < good[-1] <= base[1])
            self.assertEqual(want_gap, (at_b - at_a - base[1] - 2 * e, at_b - at_a - base[0] + 2 * e))
            for on_a in rng.sample(origins, 3):
                for on_b in rng.sample(origins, 3):
                    span = exchange.offset(a1 + on_a, b2 + on_b, b3 + on_b, a4 + on_a, err)
                    self.assertEqual((span[0] - base[0], span[1] - base[1]), (on_b - on_a, on_b - on_a))
                    self.assertEqual(exchange.gap(at_a + on_a, at_b + on_b, span, err), want_gap)
                    for rate in (0, exchange.RATE, Fraction(1, 7)):
                        self.assertEqual(exchange.holds(at_a + on_a, at_b + on_b, span, err, rate),
                                         exchange.holds(at_a, at_b, base, err, rate))
            open_orders += want_gap[0] <= 0 <= want_gap[1]
            settled += not want_gap[0] <= 0 <= want_gap[1]
        self.assertGreater(min(open_orders, settled), 60)

    def test_a_number_as_it_is_printed(self):
        self.assertEqual(exchange.exact(0.1), Fraction(1, 10))
        self.assertEqual(exchange.exact(196.5), Fraction(393, 2))
        self.assertEqual(exchange.exact(1000.0), 1000)
        self.assertEqual(exchange.exact(ORIGIN + 1), ORIGIN + 1)
        self.assertEqual(exchange.exact(Fraction(1, 3)), Fraction(1, 3))
        self.assertEqual(exchange.exact(1e-07), Fraction(1, 10 ** 7))
        self.assertNotEqual(Fraction(0.1), Fraction(1, 10))                      # the binary number next to it
        for bad in (math.nan, math.inf, -math.inf):
            for call in (lambda: exchange.offset(100, 98, 99, bad), lambda: exchange.gap(0, bad, (-6, -2)),
                         lambda: exchange.gap(0, -3, (-6, bad)), lambda: exchange.holds(0, -3, (-6, -2), bad)):
                with self.assertRaises(ValueError):
                    call()

    def test_the_exchanges_of_the_book(self):
        # In fractions the numbers of the transcript are exact, where Version 42 had them to a rounding.
        spans = {e: exchange.both([exchange.offset(*x, err=e) for x in exchange.EXCHANGES]) for e in (0, 0.1, 0.125, 0.5)}
        self.assertEqual(spans[0], (-6, Fraction(-7, 2)))
        self.assertEqual(spans[0.5], (-7, Fraction(-5, 2)))
        self.assertEqual(exchange.gap(0, -3, spans[0.5], 0.5), (Fraction(-3, 2), 5))
        self.assertEqual([exchange.holds(0, -3, spans[e], e) for e in (0, 0.1, 0.125)], [25, 5, 0])
        was = version_42_holds(0, -3, version_42_offset(-0.0, -3.3, -3.3, 0.0, 0), 0.1)   # the same room, in floats
        self.assertNotEqual(was, 5)
        self.assertAlmostEqual(was, 5, places=9)
        for stamps in exchange.EXCHANGES:
            for err in (0, 0.5, 0.2, 0.1):
                for now, then in zip(exchange.offset(*stamps, err=err), version_42_offset(*stamps, err=err)):
                    self.assertAlmostEqual(float(now), then, places=12)
        self.assertEqual(printed(exchange.main), expected("ch06_exchange"))
        self.assertEqual(len(expected("ch06_exchange").splitlines()), 39)        # 21 until Version 43, 29 in 44


def uneven(only_a, only_b):
    """Two-sided exact probability of a division at least as uneven, from binomial coefficients."""
    n, low = only_a + only_b, min(only_a, only_b)
    tail = Fraction(sum(math.comb(n, k) for k in range(low + 1)), 2 ** n)
    return float(min(1, 2 * tail))


def every_table(right_a, right_b, n):
    """Each number of items that both models can have right, with the items that fall to one model alone."""
    tables = []
    for both in range(0, n + 1):
        only_a, only_b = right_a - both, right_b - both
        neither = n - both - only_a - only_b
        if min(only_a, only_b, neither) >= 0:
            tables.append((both, only_a, only_b))
    return tables


class WhatTwoScoresAllow(unittest.TestCase):
    N, A = 2000, 1722

    def test_the_tables_that_two_scores_allow(self):
        for right_b in (1727, 1758, 1768, 300, 2000):
            tables = every_table(self.A, right_b, self.N)
            self.assertEqual([t[0] for t in tables],
                             list(range(max(0, self.A + right_b - self.N), min(self.A, right_b) + 1)))
        self.assertEqual(len(every_table(self.A, 1758, self.N)), 243)

    def test_the_table_of_the_review(self):
        want = {1727: ("0.06250", "0.86471", "never"), 1758: ("2.91e-11", "0.12474", "depends"),
                1768: ("2.84e-14", "0.04620", "always")}
        for right_b, (least, most, decision) in want.items():
            ps = [uneven(only_a, only_b) for _, only_a, only_b in every_table(self.A, right_b, self.N)]
            self.assertEqual(f"{max(ps):.5f}", most)
            self.assertEqual(f"{min(ps):.5f}" if min(ps) > 1e-4 else f"{min(ps):.2e}", least)
            self.assertEqual("never" if min(ps) >= 0.05 else "always" if max(ps) < 0.05 else "depends", decision)
            self.assertTrue(all(x >= y for x, y in zip(ps, ps[1:])))             # the more in common, the smaller
            lo, hi = accuracy.limits(self.A, right_b, self.N)
            self.assertAlmostEqual(lo, min(ps), delta=1e-15)
            self.assertAlmostEqual(hi, max(ps), delta=1e-15)

    def test_every_lead(self):
        never, depends, always = [], [], []
        for lead in range(0, self.N - self.A + 1):
            ps = [uneven(a, b) for _, a, b in every_table(self.A, self.A + lead, self.N)]
            (never if min(ps) >= 0.05 else always if max(ps) < 0.05 else depends).append(lead)
        self.assertEqual((never, depends), (list(range(0, 6)), list(range(6, 46))))
        self.assertEqual(always, list(range(46, 279)))
        self.assertIn(1758 - self.A, depends)

    def test_the_paired_uncertainty(self):
        us = []
        for both, only_a, only_b in every_table(self.A, 1758, self.N):
            gains = [-1] * only_a + [1] * only_b + [0] * (self.N - only_a - only_b)
            us.append(100 * math.sqrt(statistics.pvariance(gains) / self.N))
        self.assertEqual((f"{min(us):.4f}", f"{max(us):.4f}"), ("0.2973", "1.1395"))
        self.assertEqual((us.index(max(us)), us.index(min(us))), (0, len(us) - 1))
        self.assertEqual((f"{min(us):.2f}", f"{max(us):.2f}"), ("0.30", "1.14"))  # as the text has them

    def test_the_crossing_of_figure_17_1(self):
        self.assertEqual(f"{uneven(1722 - 1580, 1758 - 1580):.6f}", "0.050229")
        self.assertEqual(f"{uneven(1722 - 1581, 1758 - 1581):.6f}", "0.049510")
        self.assertEqual((f"{uneven(142, 178):.4f}", f"{uneven(141, 177):.4f}"), ("0.0502", "0.0495"))

    def test_small_cases(self):
        self.assertEqual(accuracy.limits(5, 5, 5), (1.0, 1.0))                   # all right for both: one table
        self.assertEqual(accuracy.limits(0, 3, 3), (uneven(0, 3), uneven(0, 3)))
        lo, hi = accuracy.limits(3, 4, 10)
        self.assertEqual((lo, hi), (min(uneven(3 - h, 4 - h) for h in range(4)), max(uneven(3 - h, 4 - h) for h in range(4))))

    def test_the_transcript(self):
        text = expected("ch17_accuracy")
        lines = text.splitlines()
        self.assertEqual(len(lines), 43)                                         # 23 in Version 42, 27 in 43, 28 in 44
        rows = []
        for right_b in (1727, 1758, 1767, 1768):                                 # 1767 since Version 44
            ps = [uneven(a, b) for _, a, b in every_table(self.A, right_b, self.N)]
            low = f"{min(ps):.6f}" if min(ps) >= 1e-4 else f"{min(ps):.2e}"      # ".3g" and ".4f" until Version 44
            # Until Version 69 both ends of a range were printed to the nearest digit. Since Version 70 the least
            # probability is rounded downward and the greatest upward, so that the printed range holds them all
            # (tests/test_review69.py). The upper ends of two rows read 0.124738 and 0.046195.
            top = math.ceil(Fraction(max(ps)) * 10 ** 6) / 10 ** 6
            rows.append(f"  b = {right_b}: {low} to {top:.6f}")
        self.assertEqual(lines[23:28], ["two-sided p that 1722 and b allow:"] + rows)
        self.assertEqual(rows, ["  b = 1727: 0.062500 to 0.864711", "  b = 1758: 2.91e-11 to 0.124739",
                                "  b = 1767: 5.68e-14 to 0.051495", "  b = 1768: 2.84e-14 to 0.046196"])
        self.assertEqual(printed(accuracy.main), text)


class PairsByPosition(unittest.TestCase):
    def test_the_table_of_the_review(self):
        a, b = accuracy.two_models(2000, random.Random(17))
        moved = b[1:] + b[:1]                                                    # the first answer of B at the end
        self.assertEqual(sorted(moved), sorted(b))                               # the same answers, and the same score
        both = sum(1 for x, y in zip(a, moved) if x and y)
        only_a = sum(1 for x, y in zip(a, moved) if x and not y)
        only_b = sum(1 for x, y in zip(a, moved) if y and not x)
        self.assertEqual((both, only_a, only_b), (1512, 210, 246))
        self.assertEqual(f"{uneven(only_a, only_b):.6f}", "0.101103")
        out = printed(accuracy.compare, a, moved)
        self.assertIn("accuracy A 86.10 %, B 87.90 %\nonly A right 210, only B right 246\n", out)
        self.assertTrue(out.endswith("  two-sided           p = 0.101103\n"))

    def test_joined_on_the_identifier(self):
        a, b = accuracy.two_models(2000, random.Random(17))
        records_a = [(f"item-{k:04d}", x) for k, x in enumerate(a)]
        records_b = [(f"item-{k:04d}", y) for k, y in enumerate(b)]
        records_b = records_b[1:] + records_b[:1]                                # another order of export
        random.Random(1).shuffle(records_a)
        by_a, by_b = dict(records_a), dict(records_b)
        self.assertEqual((len(by_a), len(by_b)), (2000, 2000))                   # one result for each model and item
        self.assertEqual(by_a.keys(), by_b.keys())                               # and the same items
        ids = sorted(by_a)
        out = printed(accuracy.compare, [by_a[i] for i in ids], [by_b[i] for i in ids])
        self.assertEqual(out, printed(accuracy.compare))
        self.assertIn("only A right 112, only B right 148\n", out)


class NoItemOnWhichTheyDiffer(unittest.TestCase):
    def test_what_the_listing_prints(self):
        same = [1] * 1740 + [0] * 260
        out = printed(accuracy.compare, same, list(same))
        self.assertIn("only A right 0, only B right 0\n", out)
        self.assertIn("u, paired     0.00 points, difference / u undefined\n", out)
        self.assertTrue(out.endswith("  two-sided           p = 1.000000\n"))

    def test_the_limit_for_the_share_of_such_items(self):
        # The largest share q of differing items for which none among n still has a probability of 5 %.
        q = 1 - 0.05 ** (1 / 2000)
        self.assertAlmostEqual((1 - q) ** 2000, 0.05, places=12)
        self.assertEqual((f"{100 * q:.4f}", f"{100 * q:.2f}"), ("0.1497", "0.15"))
        self.assertAlmostEqual(q, 1 - math.exp(math.log(0.05) / 2000), places=15)
        self.assertGreater(Fraction(999, 1000) ** 2994, Fraction(5, 100))        # 2994 items do not suffice
        self.assertLessEqual(Fraction(999, 1000) ** 2995, Fraction(5, 100))      # for 0.10 points; 2995 do

    def test_the_difference_is_no_larger(self):
        # An item counts toward the difference of two accuracies only if one model alone is right on it.
        rng = random.Random(5)
        for _ in range(300):
            a = [rng.random() < 0.8 for _ in range(50)]
            b = [x if rng.random() < 0.9 else not x for x in a]
            differ = sum(x != y for x, y in zip(a, b))
            self.assertLessEqual(abs(sum(b) - sum(a)), differ)


def fitted(xs, ys):
    """Intercept, slope and residual standard deviation of a least-squares line, written out."""
    n = len(xs)
    mx, my = math.fsum(xs) / n, math.fsum(ys) / n
    sxx = math.fsum((x - mx) ** 2 for x in xs)
    slope = math.fsum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    rest = math.fsum((y - intercept - slope * x) ** 2 for x, y in zip(xs, ys))
    return intercept, slope, math.sqrt(rest / (n - 2)), mx, sxx


class ReadingsOfTheUnknown(unittest.TestCase):
    IND = 37.90

    def data(self):
        rng = random.Random(12)
        xs = [r for r in calibration.REFS for _ in range(calibration.REPEATS)]
        ys = [calibration.probe(x, rng) for x in xs]
        return xs, ys

    def parts(self):
        xs, ys = self.data()
        a, b, s, mx, sxx = fitted(xs, ys)
        x0 = (self.IND - a) / b
        return x0, s / b, s / b * math.sqrt(1 / len(xs) + (x0 - mx) ** 2 / sxx)

    def test_the_three_sources(self):
        x0, one, line = self.parts()
        self.assertEqual((f"{one:.5f}", f"{line:.5f}", f"{calibration.U_REF:.5f}"), ("0.15249", "0.02821", "0.02000"))
        self.assertEqual(f"{math.sqrt(one ** 2 + line ** 2 + calibration.U_REF ** 2):.5f}", "0.15636")
        xs, fit = calibration.calibrate(random.Random(12))
        got = calibration.sources(self.IND, xs, fit)
        for now, want in zip(got, (x0, one, line)):
            self.assertAlmostEqual(now, want, places=10)

    def test_the_formula_of_the_review(self):
        _, one, line = self.parts()

        def u(m, times):
            return math.sqrt(one ** 2 / m + line ** 2 / times + calibration.U_REF ** 2)
        self.assertEqual([f"{u(5, 1):.5f}", f"{u(1, 5):.5f}"], ["0.07646", "0.15431"])
        self.assertEqual([f"{u(1, 1):.4f}", f"{u(1, 5):.4f}", f"{u(30, 1):.4f}", f"{u(30, 5):.4f}"],
                         ["0.1564", "0.1543", "0.0444", "0.0365"])
        xs, fit = calibration.calibrate(random.Random(12))
        for m in (1, 2, 5, 30, 100):
            for times in (1, 5, 40):
                r = calibration.measure(self.IND, xs, fit, m, times)
                self.assertAlmostEqual(r.u, u(m, times), places=12)
                self.assertEqual(r.dof, times * 30 - 2)
        # After thirty indications their mean and the fitted line contribute about equally.
        self.assertLess(abs(one / math.sqrt(30) - line) / line, 0.02)
        self.assertGreater(one / math.sqrt(29), line)
        # No repeat of either kind reduces what the references share.
        self.assertGreater(calibration.measure(self.IND, xs, fit, 10 ** 9, 10 ** 9).u, calibration.U_REF)
        self.assertAlmostEqual(calibration.measure(self.IND, xs, fit, 10 ** 9, 10 ** 9).u, calibration.U_REF, places=6)

    def test_one_indication_is_what_it_was(self):
        xs, fit = calibration.calibrate(random.Random(12))
        a, b, _, _, s = fit                                                      # measure() of Version 42
        x = (self.IND - a) / b
        mx = statistics.fmean(xs)
        sxx = sum((v - mx) ** 2 for v in xs)
        then = math.hypot(s / b * math.sqrt(1 + 1 / len(xs) + (x - mx) ** 2 / sxx), calibration.U_REF)
        r = calibration.measure(self.IND, xs, fit)
        self.assertAlmostEqual(r.u, then, places=15)
        self.assertEqual((r.value, r.dof, str(r)), (x, 28, "36.55(16) ms"))

    def test_the_whole_procedure_simulated(self):
        # A delay of 36.5 ms measured 6000 times with a calibration of its own each time, for three plans. The
        # spread of the corrected values is what the formula gives for the scatter and the gain of the probe.
        true, sigma, gain = 36.5, 0.15, 1.012
        design = [r for r in calibration.REFS for _ in range(calibration.REPEATS)]
        for m, times, seed in ((1, 1, 1), (5, 1, 2), (30, 5, 3)):
            rng = random.Random(seed)
            xs = design * times
            mx = math.fsum(xs) / len(xs)
            sxx = math.fsum((x - mx) ** 2 for x in xs)
            errors = []
            for _ in range(6000):
                shift = rng.gauss(0, calibration.U_REF)                          # the references are all off by this
                ys = [0.85 + gain * (x + shift) + rng.gauss(0, sigma) for x in xs]
                my = math.fsum(ys) / len(ys)
                slope = math.fsum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
                ind = math.fsum(0.85 + gain * true + rng.gauss(0, sigma) for _ in range(m)) / m
                errors.append((ind - (my - slope * mx)) / slope - true)
            line = sigma / gain * math.sqrt(1 / len(xs) + (true - mx) ** 2 / sxx)
            want = math.sqrt((sigma / gain) ** 2 / m + line ** 2 + calibration.U_REF ** 2)
            seen = statistics.pstdev(errors)
            self.assertLess(abs(seen - want) / want, 0.04, (m, times, seen, want))
            self.assertLess(abs(statistics.fmean(errors)), 4 * want / math.sqrt(6000))

    def test_the_transcript(self):
        text = expected("ch12_calibration")
        lines = text.splitlines()
        self.assertEqual(len(lines), 13)                                         # 7 in Version 42, 12 in 43
        _, one, line = self.parts()
        self.assertEqual(lines[7], f"one indication {one:.3f}, line {line:.3f}, ref. {calibration.U_REF:.3f} ms")
        self.assertEqual(lines[8], "u, ms, with calibration readings:   30    150")
        rows = []
        for m in (1, 5, 30):
            us = [math.sqrt(one ** 2 / m + line ** 2 / times + calibration.U_REF ** 2) for times in (1, 5)]
            rows.append(f"  indications in the mean: {m:>2}    {us[0]:.3f}  {us[1]:.3f}")
        self.assertEqual(lines[8].index("30") + 2, rows[0].index("0.156") + 5)   # the columns end together
        self.assertEqual(len(lines[8]), len(rows[0]))
        self.assertEqual(lines[9:12], rows)
        self.assertEqual([row[-12:] for row in rows], ["0.156  0.154", "0.076  0.072", "0.044  0.037"])
        self.assertEqual(lines[:7], ["ind = 0.882 + 1.0127 * delay", "u(a) = 0.038 ms, u(b) = 0.00081", "residual s = 0.154 ms",
                                     "indication        37.90 ms", "correction        -1.35 ms", "measured value    36.55(16) ms",
                                     "result  (36.55 ± 0.32) ms, k = 2.05, p = 0.95"])
        self.assertEqual(printed(calibration.main), text)


class TheTwentyFourthFigure(unittest.TestCase):
    def test_the_script(self):
        source = (ROOT / "figures" / "make_figures.py").read_text(encoding="utf-8")
        for piece in ("def calibration_readings():", "c12.measure(37.90, xs, fit, m, times).u", "c12.calibrate(random.Random(12))",
                      'save(fig, "fig-12-1-calibration-readings.png")', "    two_conventions()\n    calibration_readings()\n    control_chart()\n",
                      "float(c6x.holds(0, -3, span, bound_ms))"):
            self.assertIn(piece, source)
        self.assertEqual(source.count("\n    save(fig, "), 32)                   # twenty-three in Version 42

    def test_what_it_shows(self):
        xs, fit = calibration.calibrate(random.Random(12))
        for times in (1, 5):
            us = [calibration.measure(37.90, xs, fit, m, times).u for m in range(1, 101)]
            self.assertTrue(all(x > y for x, y in zip(us, us[1:])))              # every further indication helps
            self.assertTrue(calibration.U_REF < us[-1] < us[0] < 0.17)           # within the axes, above the floor
        one, five = (calibration.measure(37.90, xs, fit, 100, times).u for times in (1, 5))
        self.assertEqual((f"{one:.3f}", f"{five:.3f}"), ("0.038", "0.028"))


if __name__ == "__main__":
    unittest.main()
