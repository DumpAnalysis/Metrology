"""Regression tests added after the review of Version 37.

The review found that over() of Listing 10.1 counted a duration that is not a number as within the limit, so that
a missed objective became a pass; that the text let two readings of a counter stand for its whole increase,
where a counter of fixed width fixes the increase only up to whole wraps; that fit_temperature() of Listing 18.1
called a minimum at an end of its interval "below" or "above" it; and that at_one_logit() returned a negative
temperature, or divided by zero, for a probability that no temperature can state.

Each test computes what it expects independently of the code under test: counts of a list, the verdict of
Chapter 16 from its definition, every total that a small counter allows, searched one by one, a closed form for
the temperature of cases at one logit, and exact binomial tails from binomial coefficients.
"""
import contextlib
import io
import math
import random
import unittest
from fractions import Fraction
from pathlib import Path

from examples import ch08_counter as counter
from examples import ch10_coordinated_omission as omission
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


def version_37_over(xs, limit=500.0):
    """over() as Version 37 printed it."""
    n = sum(x > limit for x in xs)
    return n, 100 * (len(xs) - n) / len(xs)


class ADurationThatIsNotThere(unittest.TestCase):
    def test_the_example_of_the_review(self):
        xs = omission.closed_loop()
        self.assertEqual(omission.over(xs)[0], 2)
        xs[-1] = float("nan")                                                    # one of the two slow durations is lost
        count, share = version_37_over(xs)
        self.assertEqual((count, f"{share:.4f}"), (1, "99.9828"))                # a missed objective would pass
        # Since Version 57 the refusal is that of known() of the package, with its message.
        self.assertEqual(refusal(omission.over, xs), "a duration is missing or too large")

    def test_what_is_a_duration(self):
        for bad in ([], [1.0, float("nan")], [float("inf")], [-0.5, 3.0], [1.0, -math.inf]):
            self.assertIsNotNone(refusal(omission.over, bad), bad)
        self.assertEqual(refusal(omission.over, []), "no entries")
        self.assertEqual(omission.over([0, 0.0, 500, 500.0, 500.5, 2000]), (2, 100 * 4 / 6))
        self.assertEqual(omission.over(iter([1.0, 600.0])), (1, 50.0))           # read once

    def test_the_table_of_the_review(self):
        # At least 99.97 % of 5800 within the limit is 5799 or more. What is known decides, or does not.
        need = math.ceil(Fraction(9997, 10000) * 5800)
        self.assertEqual(need, 5799)
        rows = ((5798, 0, "missed"), (5798, 1, "unresolved"), (5799, 0, "met"))
        for within, unknown, verdict in rows:
            low, high = within, within + unknown                                 # the least and the most that can be within
            want = "met" if low >= need else "missed" if high < need else "unresolved"
            self.assertEqual(want, verdict)
            self.assertEqual(bounds.quantile_verdict(5800, 0.9997, within, unknown), verdict)
        self.assertEqual([f"{100 * k / 5800:.4f}" for k in (5798, 5799)], ["99.9655", "99.9828"])

    def test_the_counts_behind_a_hundred_requests(self):
        measured = omission.closed_loop()
        filled = omission.back_fill(measured)
        intended = omission.open_loop()
        above = [sum(1 for x in xs if x > 500) for xs in (measured, filled, intended)]
        self.assertEqual(above, [2, 102, 112])
        self.assertEqual(len(intended) - len(measured), 200)                     # sends that the generator skipped
        self.assertEqual(len(filled) - len(measured), 200)                       # entries that the repair adds
        self.assertEqual((above[1] - above[0], above[2] - above[1], above[2] - above[0]), (100, 10, 110))

    def test_the_transcript_is_that_of_version_37(self):
        text = expected("ch10_coordinated_omission")
        self.assertEqual(len(text.splitlines()), 12)
        self.assertEqual(printed(omission.main), text)


def totals(first, last, most, modulus):
    """Every increase up to `most` that takes a counter kept modulo `modulus` from one reading to the other."""
    return [t for t in range(most + 1) if (first + t) % modulus == last]


class WhatTwoReadingsAllow(unittest.TestCase):
    def test_the_example_of_the_review(self):
        first, last = counter.wrapping(100, 2)
        middle = counter.wrapping(50, 2)[1]
        self.assertEqual((first, last, middle), (0, 705032704, 2500000000))
        self.assertEqual(5_000_000_000 - 2 ** 32, 705032704)
        allowed = counter.increments(first, last, 60_000_000 * 100)
        self.assertEqual(allowed, [705032704, 5_000_000_000])
        self.assertEqual([f"{t / 100 / 1e6:.4f}" for t in allowed], ["7.0503", "50.0000"])
        # A half can hold 3 000 000 000 bytes at most, which is below 2 ** 32: one total for each.
        self.assertLess(60_000_000 * 50, 2 ** 32)
        halves = [counter.increments(a, b, 60_000_000 * 50) for a, b in ((first, middle), (middle, last))]
        self.assertEqual(halves, [[2500000000], [2500000000]])
        self.assertEqual((f"{2 ** 32 / 6e7:.2f}", f"{2 ** 32 / 5e7:.2f}"), ("71.58", "85.90"))

    def test_every_total_of_a_small_counter(self):
        rng = random.Random(3701)
        for _ in range(400):
            modulus = rng.randint(1, 40)
            first, last = rng.randrange(modulus), rng.randrange(modulus)
            most = rng.randint(0, 200)
            want = totals(first, last, most, modulus)
            self.assertEqual(counter.increments(first, last, most, modulus), want)
            if most < modulus:
                self.assertLessEqual(len(want), 1)                               # a bound below the width: one total or none
            self.assertTrue(all((t - (last - first)) % modulus == 0 for t in want))

    def test_readings_close_enough_recover_a_varying_rate(self):
        # A rate that varies below its bound, read at gaps shorter than the width divided by the bound: every
        # interval has one total, and their sum is the true increase, wraps included.
        rng = random.Random(3702)
        bound, modulus = 60_000_000, 2 ** 32
        for _ in range(50):
            reading, true, recovered = rng.randrange(modulus), 0, 0
            for _ in range(rng.randint(5, 40)):
                gap = rng.randint(1, 71)                                         # seconds; 71 * bound is below 2 ** 32
                step = rng.randint(0, bound * gap)
                after = (reading + step) % modulus
                (only,) = counter.increments(reading, after, bound * gap)
                reading, true, recovered = after, true + step, recovered + only
            self.assertEqual(recovered, true)
        self.assertGreater(len(counter.increments(0, 0, bound * 72)), 1)         # at 72 s the bound no longer suffices

    def test_what_is_refused(self):
        for first, last in ((-1, 5), (0, 2 ** 32), (2 ** 32, 0), (5, -3)):
            self.assertEqual(refusal(counter.increments, first, last, 10), "a reading outside the counter")
        self.assertEqual(counter.increments(7, 3, 2, 10), [])                    # no total within the bound
        self.assertEqual(counter.increments(7, 3, 6, 10), [6])

    def test_the_transcript(self):
        text = expected("ch08_counter")
        lines = text.splitlines()
        self.assertEqual(len(lines), 28)                                         # 17 in Version 37, 25 in Version 38
        self.assertEqual(lines[16], "  modulo          7.1 to 7.1")
        want = ["readings 0 and 705032704, at most", "60 MB/s for 100 s: an increase of"]
        allowed = [t for t in (705032704 + wraps * 2 ** 32 for wraps in range(4)) if t <= 6_000_000_000]
        self.assertEqual(len(allowed), 2)
        want += [f"  {t:>10} bytes, {t / 100 / 1e6:.2f} MB/s" for t in allowed]
        want += ["with the reading 2500000000 at 50 s"] + [f"  {2500000000:>10} bytes, {2500000000 / 50 / 1e6:.2f} MB/s"] * 2
        want += [f"one answer for every gap below {2 ** 32 / 6e7:.2f} s"]
        self.assertEqual(lines[17:25], want)                                     # three lines follow in Version 39
        self.assertEqual(printed(counter.main), text)


def at_one(logit, positive, n):
    return logit / math.log(Fraction(positive, n - positive))


class AMinimumAtALimit(unittest.TestCase):
    def test_the_table_of_the_review(self):
        # Two positive labels and one negative at a logit of ln 2: the stated probability at T = 1 is 2/3, the
        # observed frequency, and the loss is least there.
        data = [(math.log(2), 1), (math.log(2), 1), (math.log(2), 0)]
        self.assertAlmostEqual(at_one(math.log(2), 2, 3), 1.0, places=12)
        self.assertEqual(cal.slope(cal.margins(data), 1.0), 0.0)
        self.assertEqual(f"{cal.fit_temperature(data, 0.2, 10):.4f}", "1.0000")
        self.assertEqual(cal.fit_temperature(data, 0.2, 1), 1)                   # Version 37: "the minimum is above hi"
        self.assertEqual(cal.fit_temperature(data, 1, 10), 1)                    # Version 37: "the minimum is below lo"
        self.assertEqual(cal.fit_temperature(data, 1.0, 10.0), 1.0)
        # Outside the interval it is still outside.
        self.assertEqual(refusal(cal.fit_temperature, data, 0.2, 0.99), "the minimum is above hi")
        self.assertEqual(refusal(cal.fit_temperature, data, 1.01, 10), "the minimum is below lo")
        # At T = 1 the loss is not above the loss beside it.
        self.assertLessEqual(cal.nll(data, 1.0), min(cal.nll(data, 0.99), cal.nll(data, 1.01)))

    def test_the_order_of_the_shapes(self):
        # Margins that are all zero are none below zero as well; the constant loss is reported first.
        self.assertEqual(refusal(cal.fit_temperature, [(0.0, 1), (0.0, 0)]), "the loss is the same at every T")
        self.assertEqual(refusal(cal.fit_temperature, [(0.0, 1), (2.0, 1)]), "no finite T: the loss falls as T falls")
        self.assertEqual(refusal(cal.fit_temperature, [(1.0, 1), (1.0, 0)]), "no finite T: the loss falls as T rises")

    def test_interior_fits_are_unchanged(self):
        rng = random.Random(3703)
        for _ in range(100):
            n = rng.randint(3, 300)
            negative = rng.randint(1, (n - 1) // 2)
            logit = rng.uniform(0.5, 30)
            want = at_one(logit, n - negative, n)
            data = [(logit, 1)] * (n - negative) + [(logit, 0)] * negative
            if 0.2 < want < 10:
                self.assertAlmostEqual(cal.fit_temperature(data) / want, 1, places=9)


def tail(n, k, p, upper):
    ks = range(k, n + 1) if upper else range(0, k + 1)
    return math.fsum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in ks)


def exact_limits(k, n, level=0.95):
    """Clopper and Pearson, by halving: 0 below when no success was seen, and 1 above when every trial was one."""
    a = (1 - level) / 2
    out = []
    for upper in (True, False):
        if (upper and k == 0) or (not upper and k == n):
            out.append(0.0 if upper else 1.0)
            continue
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if (tail(n, k, mid, upper) < a) == upper:
                lo = mid
            else:
                hi = mid
        out.append((lo + hi) / 2)
    return out


class TemperaturesThatTheLabelsAllow(unittest.TestCase):
    def test_the_table_of_the_review(self):
        want = {99: ("0.94554", "0.99975"), 55: ("0.44728", "0.64968"), 100: ("0.96378", "1.00000"), 35: ("0.25729", "0.45185")}
        for positive, limits in want.items():
            low, high = exact_limits(positive, 100)
            self.assertEqual((f"{low:.5f}", f"{high:.5f}"), limits)
        low, high = cal.temperatures(40, 99, 100)
        self.assertEqual((f"{low:.2f}", f"{high:.2f}"), ("4.83", "14.01"))
        low, high = cal.temperatures(40, 55, 100)
        self.assertEqual((f"{low:.2f}", high), ("64.76", math.inf))              # no finite upper limit
        low, high = cal.temperatures(40, 100, 100)
        self.assertEqual((low, f"{high:.2f}"), (0.0, "12.19"))                   # every T up to 12.20
        self.assertIsNone(cal.temperatures(40, 35, 100))                         # no positive T
        # The four numbers again, from the limits computed here.
        p = exact_limits(99, 100)
        self.assertAlmostEqual(cal.temperatures(40, 99, 100)[1], 40 / math.log(p[0] / (1 - p[0])), places=6)
        p = exact_limits(55, 100)
        self.assertAlmostEqual(cal.temperatures(40, 55, 100)[0], 40 / math.log(p[1] / (1 - p[1])), places=6)
        p = exact_limits(100, 100)
        self.assertAlmostEqual(cal.temperatures(40, 100, 100)[1], 40 / math.log(p[0] / (1 - p[0])), places=6)

    def test_what_version_37_returned(self):
        # The unguarded quotient: a negative temperature for a probability below a half, and a division by zero
        # at a probability of 1.
        low, high = exact_limits(55, 100)
        self.assertEqual(f"{40 / math.log(low / (1 - low)):.2f}", "-188.98")
        with self.assertRaises(ZeroDivisionError):
            40 / math.log(1.0 / (1 - 1.0))
        for p in (0.44728, 0.5, 1.0, 0.0, 1.2, -0.1):
            self.assertEqual(refusal(cal.at_one_logit, 40, p), "no T above zero states this p", p)   # reworded in 39
        self.assertAlmostEqual(cal.at_one_logit(40, 0.99), 40 / math.log(99), places=12)

    def test_every_count_of_a_hundred(self):
        # The set is what the definition says: the temperatures whose stated probability is within the interval.
        for positive in range(0, 101):
            p_low, p_high = exact_limits(positive, 100)
            got = cal.temperatures(40, positive, 100)
            if p_high <= 0.5:
                self.assertIsNone(got, positive)
                continue
            low, high = got
            self.assertEqual(low == 0.0, positive == 100)
            self.assertEqual(high == math.inf, p_low <= 0.5)
            for temp in (0.5, 3.0, 5.0, 8.0, 12.0, 13.0, 20.0, 50.0, 70.0, 200.0, 1e4):
                stated = cal.sigmoid(40 / temp)
                if abs(stated - p_low) < 1e-9 or abs(stated - p_high) < 1e-9:
                    continue
                inside = low < temp <= high if low == 0.0 else low <= temp <= high
                self.assertEqual(inside, p_low <= stated <= p_high, (positive, temp))
        self.assertEqual(cal.temperatures(40, 100, 100)[0], 0.0)

    def test_the_transcript(self):
        text = expected("ch18_calibration")
        lines = text.splitlines()
        self.assertEqual(len(lines), 24)                                         # 20 in Version 37, 23 in Version 38
        self.assertEqual(lines[16], "  95 % interval for T: 4.83 to 14.02")       # 14.01 to the nearest, until Version 62
        p55, p100 = exact_limits(55, 100), exact_limits(100, 100)
        self.assertEqual(lines[20:23], [
            f"  55 positive: T from {40 / math.log(p55[1] / (1 - p55[1])):.2f}, no upper end",
            # An upper end is printed rounded upward since Version 59: 12.1901 is at most 12.20.
            f"  100 positive: T up to {math.ceil(40 / math.log(p100[0] / (1 - p100[0])) * 100) / 100:.2f}",
            "  35 positive: no T above zero"])
        self.assertEqual(printed(cal.main), text)


class TheNineteenthFigure(unittest.TestCase):
    def test_the_script(self):
        source = (ROOT / "figures" / "make_figures.py").read_text(encoding="utf-8")
        for piece in ("def counter_readings():", "c8c.wrapping(100, 2)", "c8c.increments(first, last, 60_000_000 * 100)",
                      'save(fig, "fig-08-2-counter-readings.png")', 'save(fig, "fig-08-3-empty-record.png")',
                      "    aliasing()\n    counter_readings()\n    empty_record()\n"):
            self.assertIn(piece, source)
        self.assertNotIn("fig-08-2-empty-record.png", source)                    # the empty record is Figure 8.3 now
        self.assertEqual(source.count("\n    save(fig, "), 32)                   # eighteen in Version 37

    def test_what_it_shows(self):
        first, last = counter.wrapping(100, 2)
        slow, fast = counter.increments(first, last, 60_000_000 * 100)
        self.assertEqual((f"{slow / 1e8:.2f}", f"{fast / 1e8:.0f}"), ("7.05", "50"))        # MB/s over 100 s
        self.assertEqual(f"{100 * 2 ** 32 / fast:.1f}", "85.9")                  # the faster stream wraps here, s
        self.assertEqual(f"{last / 1e9:.3f}", "0.705")
        # At 50 s the faster stream reads 2.5 GB and the slower one 0.35 GB.
        self.assertEqual((fast // 2 % 2 ** 32, f"{slow / 2 / 1e9:.2f}"), (2500000000, "0.35"))


if __name__ == "__main__":
    unittest.main()
