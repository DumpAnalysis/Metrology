"""Regression tests added after the review of Version 29.

The review found that a line fitted to whole readings of sixteen digits,
each of which a float holds, lost half a nanosecond of its intercept,
fourteen times the uncertainty reported beside it; that one second
converted to nanoseconds came out below a billion, and that a conversion
of a value to its own unit changed it; and that the probability printed
beside a coverage interval was rounded to six digits, so that 0.9999999
read as 1. It proposed an example in which the reference for a silence is
a second trace and not a rate.

Each test computes what it expects independently of the code under test:
group means and residuals worked out by hand, a line fitted about the
means in fractions, the factors of the prefixes written out as powers,
the normal quantile of the statistics module, divisions of messages
between two records counted one by one, and a seeded simulation.
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

from examples import ch16_silence as silence
from swmetro import record, stats, units
from swmetro.core import Result
from swmetro.units import Q

ROOT = Path(__file__).resolve().parent.parent
EPOCH = 1_700_000_000_000_000_000                # ns from 1970 to 14 November 2023, 22:13:20 UTC
RUNNING = 5_000_000_000_000_000                  # the N of the review, ns: about 58 days on a running clock


def expected(name):
    return (ROOT / "expected" / f"{name}.txt").read_text(encoding="utf-8")


def printed(function, *args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        function(*args)
    return out.getvalue()


def exact_line(xs, ys):
    """Least squares about the two means, in fractions: a, b, and the squares of u(a) and u(b)."""
    fx, fy = [Fraction(x) for x in xs], [Fraction(y) for y in ys]
    n = len(fx)
    mx, my = sum(fx) / n, sum(fy) / n
    sxx = sum((x - mx) ** 2 for x in fx)
    b = sum((x - mx) * (y - my) for x, y in zip(fx, fy)) / sxx
    a = my - b * mx
    var = sum((y - a - b * x) ** 2 for x, y in zip(fx, fy)) / (n - 2)
    return a, b, var * (Fraction(1, n) + mx ** 2 / sxx), var / sxx


def rooted(square):
    """The float nearest to the root of a fraction, by the decimal module with fifty digits."""
    with decimal.localcontext() as ctx:
        ctx.prec = 50
        return float((decimal.Decimal(square.numerator) / decimal.Decimal(square.denominator)).sqrt())


def pairs_of_the_review(origin=RUNNING):
    """At the settings -1 ns and +1 ns, the readings N - 1 and N, and N + 1 and N + 2, fifty of each."""
    return [(x, origin + x + j) for x in (-1, 1) for j in (0, 1) for _ in range(50)]


class AnInterceptBetweenTwoFloats(unittest.TestCase):
    def test_the_case_of_the_review(self):
        xs, ys = zip(*pairs_of_the_review())
        # By hand: the two group means are N - 1/2 and N + 3/2, the slope is 1, the intercept N + 1/2, and
        # each of the 200 residuals is half a nanosecond.
        self.assertEqual(exact_line(xs, ys), (Fraction(2 * RUNNING + 1, 2), 1, Fraction(1, 4 * 198), Fraction(1, 4 * 198)))
        self.assertTrue(all(float(v) == v for v in xs + ys) and RUNNING < 2 ** 53 and math.ulp(float(RUNNING)) == 1.0)
        self.assertEqual(f"{RUNNING / 86_400e9:.1f}", "57.9")                         # days
        # Version 29: readings that floats hold, with residuals, gave the nearest float.
        self.assertEqual(float(Fraction(2 * RUNNING + 1, 2)), RUNNING)
        self.assertEqual(f"{0.5 / (0.5 / math.sqrt(198)):.2f}", "14.07")             # half a nanosecond, in u(a)
        a, b, ua, ub, s = stats.linfit(xs, ys)
        self.assertEqual((a, type(a), b), (Fraction(2 * RUNNING + 1, 2), Fraction, 1.0))
        self.assertAlmostEqual(ua / (0.5 / math.sqrt(198)), 1.0, places=15)
        self.assertEqual((f"{ua:.5f}", ua, f"{s:.4f}"), ("0.03553", ub, f"{math.sqrt(50 / 198):.4f}"))
        # The test that the review asks for: with N subtracted first the intercept is 0.5, and the two differ by N.
        near = stats.linfit(xs, [y - RUNNING for y in ys])
        self.assertEqual((near[0], type(near[0]), near[1:]), (0.5, float, (b, ua, ub, s)))
        self.assertEqual(Fraction(a) - Fraction(near[0]), RUNNING)

    def test_whole_readings_and_fractions_with_residuals(self):
        rng = random.Random(2901)
        kept = 0
        for _ in range(300):
            origin = rng.choice((2 ** 44, 10 ** 15, RUNNING, 2 ** 53 - 5000, -RUNNING))
            n = rng.randint(4, 40)
            xs = [rng.randint(-3, 3) for _ in range(n)]
            # Halves and quarters where floats of the size of the origin still hold them, whole steps above.
            steps = (0, 0, 1) if abs(origin) >= 2 ** 51 else (0, 0, 1, Fraction(1, 2), Fraction(1, 4))
            small = [2 * x + rng.choice(steps) for x in xs]
            if len(set(xs)) < 2 or exact_line(xs, small)[3] == 0:
                continue
            ys = [origin + y for y in small]
            self.assertTrue(all(abs(y) <= 2 ** 53 and float(y) == y for y in ys))    # floats hold every reading
            a, b, ua2, ub2 = exact_line(xs, ys)
            got, near = stats.linfit(xs, ys), stats.linfit(xs, small)
            for value, u, want, u2 in ((got[0], got[2], a, ua2), (got[1], got[3], b, ub2)):
                self.assertAlmostEqual(u / math.sqrt(u2), 1.0, places=12)
                self.assertLessEqual(1000 * abs(Fraction(value) - want), Fraction(u), (xs, ys))
            # The origin comes back: exactly where the intercept of the small readings is a float, as in the
            # case of the review, and otherwise but for the rounding of that float, which its uncertainty
            # cannot show.
            back = Fraction(got[0]) - Fraction(near[0]) - origin
            self.assertLessEqual(1000 * abs(back), Fraction(got[2]), (xs, ys))
            if float(a - origin) == a - origin:
                self.assertEqual(back, 0)
            self.assertEqual(got[1:], near[1:])
            kept += type(got[0]) is not float
        self.assertGreater(kept, 150)

    def test_a_float_among_the_readings(self):
        # Version 30 returned the nearest float for these and added half a nanosecond of rounding to its
        # uncertainty, 0.5013 ns. Since Version 31 the coefficient is not rounded, however the readings are
        # stored, and the uncertainty is that of the fit: the review of Version 30, and test_review30.
        xs, ys = zip(*pairs_of_the_review())
        square = Fraction(1, 4 * 198)
        self.assertEqual(f"{rooted(square + Fraction(1, 4)):.4f}", "0.5013")
        for fx, fy in (([float(x) for x in xs], ys), (xs, [float(y) for y in ys]), ([float(x) for x in xs], [float(y) for y in ys])):
            a, b, ua, ub, s = stats.linfit(fx, fy)
            self.assertEqual((a, type(a), b, ua, ub), (Fraction(2 * RUNNING + 1, 2), Fraction, 1.0, rooted(square), rooted(square)))
        # Decimal readings stored as floats, on their line to the last bit or two: each coefficient is exact,
        # or a float within a thousandth of its uncertainty, and the uncertainty is that of the fit.
        rng = random.Random(2902)
        kept = 0
        for _ in range(300):
            n = rng.randint(3, 120)
            step, a0, b0 = rng.choice((0.1, 0.01, 0.37)), rng.choice((3.0, 0.3, 1.0)), rng.choice((2.0, 0.1, 0.75))
            xs = [step * k for k in range(n)]
            ys = [a0 + b0 * x for x in xs]
            a, b, ua2, ub2 = exact_line(xs, ys)
            got = stats.linfit(xs, ys)
            for value, u, want, u2 in ((got[0], got[2], a, ua2), (got[1], got[3], b, ub2)):
                self.assertEqual(u, rooted(u2))
                self.assertLessEqual(1000 * abs(Fraction(value) - want), Fraction(u))
                kept += type(value) is not float
        self.assertGreater(kept, 200)

    def test_what_did_not_change(self):
        # Floats with a scatter that floats show: the coefficients and the uncertainties of every version.
        rng = random.Random(2903)
        for _ in range(200):
            n = rng.randint(3, 12)
            xs = [rng.choice((rng.uniform(0, 60), rng.randint(0, 60))) for _ in range(n)]
            ys = [2.5 + 0.75 * x + rng.gauss(0, 0.3) for x in xs]
            if len(set(xs)) < 2:
                continue
            a, b, ua2, ub2 = exact_line(xs, ys)
            got = stats.linfit(xs, ys)
            self.assertEqual(got[:2], (float(a), float(b)))
            self.assertAlmostEqual(got[2] / math.sqrt(ua2), 1.0, places=13)
            self.assertAlmostEqual(got[3] / math.sqrt(ub2), 1.0, places=13)
        # A fit without residuals rounds nothing, with floats among the readings or without.
        self.assertEqual(stats.linfit([0.0, 3.0, 6.0], [0.0, 1.0, 2.0]), (0.0, Fraction(1, 3), 0.0, 0.0, 0.0))
        self.assertEqual(stats.linfit([0, 3, 6], [0, 1, 2]), (0.0, Fraction(1, 3), 0.0, 0.0, 0.0))
        stamps = [EPOCH + 100 + i for i in range(3)]
        self.assertEqual(stats.linfit(stamps, [0, 1, 2]), (-EPOCH - 100, 1.0, 0.0, 0.0, 0.0))
        # Stamps that no float holds, with a float among the other readings: not rounded, as in Version 29.
        a = stats.linfit([0.0, 1.0, 2.0, 3.0], [EPOCH + 100, EPOCH + 101, EPOCH + 101, EPOCH + 102])[0]
        self.assertEqual((a - EPOCH, type(a)), (Fraction(1001, 10), Fraction))
        # Whole numbers of ordinary size with a scatter of their own: floats.
        self.assertEqual([type(v) for v in stats.linfit([1, 2, 3, 4], [2, 4, 5, 8])], [float] * 5)


POWERS = {"k": 10 ** 3, "M": 10 ** 6, "G": 10 ** 9, "T": 10 ** 12, "m": Fraction(1, 10 ** 3), "u": Fraction(1, 10 ** 6),
          "n": Fraction(1, 10 ** 9), "Ki": 2 ** 10, "Mi": 2 ** 20, "Gi": 2 ** 30, "Ti": 2 ** 40, "": 1}


def factor(unit):
    """The factor of a unit to seconds or to bits, from the powers of its prefix: SI Brochure, table 7, and
    IEC 80000-13 for the binary prefixes. A byte is eight bits."""
    for base, size in (("bit", 1), ("B", 8), ("s", 1)):
        if unit.endswith(base) and unit[:len(unit) - len(base)] in POWERS:
            return POWERS[unit[:len(unit) - len(base)]] * size
    raise AssertionError(unit)


TIMES = ("s", "ms", "us", "ns", "ks")
SIZES = ("bit", "B", "kB", "KiB", "MB", "MiB", "GB", "GiB", "TB", "TiB", "kbit", "Mibit")


class OneSecondInNanoseconds(unittest.TestCase):
    def test_the_table_of_the_review(self):
        # The arithmetic of Version 29, value * factor / factor in floats.
        self.assertEqual([1 * 1.0 / 1e-9, 1_000_000_000 * 1e-9 / 1e-9, 9007199254740993 * 8.0 / 8.0],
                         [999999999.9999999, 999999999.9999999, 9007199254740992.0])
        got = [Q(1, "s").to("ns").value, Q(1_000_000_000, "ns").to("ns").value, Q(9007199254740993, "B").to("B").value]
        self.assertEqual((got, [type(v) for v in got]), ([10 ** 9, 10 ** 9, 2 ** 53 + 1], [int] * 3))
        # Two stamps one nanosecond apart stay one nanosecond apart.
        self.assertEqual([Q(EPOCH + d, "ns").to("ns").value - EPOCH for d in (100, 101)], [100, 101])
        self.assertEqual(float(EPOCH + 100), float(EPOCH + 101))

    def test_a_strict_deadline(self):
        second = Q(1, "s").to("ns").value
        self.assertFalse(second < 1_000_000_000)                                     # one second is not below one second
        self.assertEqual((second, int(second), type(second)), (1_000_000_000, 1_000_000_000, int))
        self.assertTrue(Q(999_999_999, "ns").to("s").value < 1)
        self.assertEqual(Q(999_999_999, "ns").to("s").value, Fraction(999_999_999, 10 ** 9))
        back = Q(1_000_000_000, "ns").to("s").value
        self.assertEqual((back, type(back)), (1, int))
        for unit in TIMES:
            for value in (1, 30, 86_400, Fraction(1, 3)):
                self.assertEqual(Q(value, "s").to(unit).value * factor(unit), value)

    def test_the_same_unit_returns_the_value(self):
        for value in (1, 10 ** 9, EPOCH + 100, 2 ** 53 + 1, 0.1, 999999999.9999999, Fraction(1, 3), decimal.Decimal("0.10"), True):
            for unit in TIMES + SIZES:
                self.assertIs(Q(value, unit).to(unit).value, value)
        self.assertTrue(math.isnan(Q(math.nan, "s").to("s").value))

    def test_a_round_trip_through_seconds(self):
        rng = random.Random(2904)
        for _ in range(500):
            value = rng.choice((rng.randint(-10 ** 6, 10 ** 6), EPOCH + rng.randint(0, 10 ** 9), rng.randint(0, 2 ** 70),
                                Fraction(rng.randint(-999, 999), rng.randint(1, 999))))
            first, second = rng.choice(TIMES), rng.choice(TIMES)
            there = Q(value, first).to(second)
            self.assertEqual(there.value * factor(second), value * factor(first), (value, first, second))
            if first == second:
                self.assertIs(there.value, value)                                    # as it was given
            else:
                self.assertIs(type(there.value), int if there.value == there.value // 1 else Fraction)
            back = there.to("s").to(first).value
            self.assertEqual(back, value)
            if isinstance(value, int):
                self.assertIs(type(back), int)
        for _ in range(300):
            value, first, second = rng.randint(1, 2 ** 60), rng.choice(SIZES), rng.choice(SIZES)
            self.assertEqual(Q(value, first).to(second).to(first).value, value)
        # Sums are exact as well, in the unit of the first term.
        self.assertEqual((Q(1, "s") + Q(500, "ms")).value, Fraction(3, 2))
        self.assertEqual((Q(EPOCH, "ns") + Q(1, "s")).value, EPOCH + 10 ** 9)
        self.assertEqual((Q(1, "s") + Q(1, "ns")).value, Fraction(10 ** 9 + 1, 10 ** 9))

    def test_exact_factors(self):
        for unit in TIMES + SIZES:
            kind, got = units.parse(unit)
            self.assertEqual((got, kind), (factor(unit), "time" if unit in TIMES else "information"), unit)
            self.assertIn(type(got), (int, Fraction))
        self.assertEqual(units.parse("ns"), ("time", Fraction(1, 10 ** 9)))
        self.assertNotEqual(Fraction(1e-9), Fraction(1, 10 ** 9))                    # the float is another number
        self.assertEqual([Q(1, big).to("B").value for big in ("kB", "KiB", "TB", "TiB")], [1000, 1024, 10 ** 12, 2 ** 40])
        self.assertEqual([Q(1, "s").to(small).value for small in ("ms", "us", "ns")], [10 ** 3, 10 ** 6, 10 ** 9])
        self.assertEqual(Q(64, "GiB").to("GB").value, Fraction(64 * 2 ** 30, 10 ** 9))
        with self.assertRaisesRegex(TypeError, "information is not time"):
            Q(5, "ms") + Q(4, "KiB")
        with self.assertRaisesRegex(ValueError, "unknown unit"):
            Q(1, "parsec")

    def test_floats_are_rounded_once(self):
        rng = random.Random(2905)
        for _ in range(2000):
            value = rng.choice((1e-9, 1.0, 842.0, 1e12)) * rng.uniform(-1.5, 1.5)
            first, second = rng.sample(TIMES, 2) if rng.random() < 0.5 else rng.sample(SIZES, 2)
            got = Q(value, first).to(second).value
            want = float(Fraction(value) * factor(first) / factor(second))
            self.assertEqual((got, type(got)), (want, float), (value, first, second))
        self.assertEqual([Q(0.2414, "s").to("ms").value, Q(1.5, "ms").to("s").value, Q(1.0, "s").to("ns").value], [241.4, 0.0015, 1e9])
        self.assertEqual(Q(math.inf, "s").to("ms").value, math.inf)
        self.assertTrue(math.isnan(Q(math.nan, "s").to("ms").value))

    def test_the_text_of_a_quantity(self):
        self.assertEqual(repr(Q(1, "s").to("ns")), "1000000000 ns")
        self.assertEqual(f"{10 ** 9:g} ns", "1e+09 ns")                              # Version 29, for 999999999.9999999 too
        self.assertEqual(repr(Q(EPOCH + 100, "ns")), "1700000000000000100 ns")
        self.assertEqual([repr(Q(1.5, "ms")), repr(Q(3, "s")), repr(Q(1500, "ms").to("s")), repr(Q(64, "GiB").to("GB"))],
                         ["1.5 ms", "3 s", "1.5 s", "68.7195 GB"])
        self.assertEqual(repr(Q(Fraction(1, 3), "s")), "0.333333 s")

    def test_the_transcript_and_the_record(self):
        text = expected("ch03_scales_units")
        for line in ("1 KiB = 1.024 kB  (+2.4 %)\n", "1 MiB = 1.04858 MB  (+4.9 %)\n", "1 GiB = 1.07374 GB  (+7.4 %)\n",
                     "1 TiB = 1.09951 TB  (+10.0 %)\n", "64 GiB = 68.7195 GB\n", "5 ms + 4 KiB: information is not time\n"):
            self.assertIn(line, text)
        # in_unit() of the record module, which the review found correct, takes its factor from the same two
        # numbers now.
        rec = {"measurand": {"unit": "s"}, "result": {"value": 1, "u": decimal.Decimal("0.2414")}}
        self.assertEqual(record.in_unit(rec, "ns"), (decimal.Decimal(10 ** 9), decimal.Decimal(241_400_000)))
        self.assertEqual(record.in_unit(rec, "ms")[1], decimal.Decimal("241.4"))
        self.assertEqual(record.in_unit(rec, "s"), (decimal.Decimal(1), decimal.Decimal("0.2414")))


def normal_k(p):
    """The coverage factor of a normal distribution for the probability p, from the tail (1 - p) / 2."""
    return -statistics.NormalDist().inv_cdf(float((1 - Fraction(p)) / 2))


class TheProbabilityBesideAnInterval(unittest.TestCase):
    def test_the_table_of_the_review(self):
        result = Result(0, 1, "ms")
        tail = 1 - Fraction(1, 10 ** 17)
        rows = ((0.95, "(0.0 ± 2.0) ms, k = 1.96, p = 0.95"), (0.9999999, "(0.0 ± 5.3) ms, k = 5.33, p = 0.9999999"),
                (0.99999999, "(0.0 ± 5.7) ms, k = 5.73, p = 0.99999999"), (tail, "(0.0 ± 8.6) ms, k = 8.57, p = 0.99999999999999999"))
        for p, want in rows:
            self.assertEqual(result.report(p), want)
            self.assertEqual(f"{normal_k(p):.2f}", want.split("k = ")[1].split(",")[0])
        # Version 29 and before: six significant digits, and for the fraction a float first.
        self.assertEqual([f"{float(p):g}" for p, _ in rows], ["0.95", "1", "1", "1"])
        self.assertEqual(float(tail), 1.0)

    def test_the_probability_reads_back(self):
        result = Result(0, 1, "ms")
        rng = random.Random(2906)
        floats = [0.5, 0.6827, 0.9, 0.95, 0.99, 0.999, 1 - 2 ** -53, 1e-5, 0.1 + 0.2] + [rng.random() for _ in range(300)]
        for p in floats:
            text = result.report(p).split("p = ")[1]
            self.assertEqual((float(text), type(float(text))), (p, float), text)
        exact = [Fraction(k, 1000) for k in range(0, 1000, 37)] + [1 - Fraction(1, 10 ** j) for j in range(1, 40, 3)]
        exact += [Fraction(2, 3), Fraction(1, 7), Fraction(19, 20)]
        for p in exact:
            text = result.report(p).split("p = ")[1]
            self.assertEqual(Fraction(text), p, text)
        self.assertEqual(result.report(decimal.Decimal("0.950")).split("p = ")[1], "0.950")
        self.assertEqual((result.report(0).split("p = ")[1], result.report(0.0).split("p = ")[1]), ("0", "0.0"))
        # Two probabilities that six digits both showed as 1 have two coverage factors, and now two texts.
        self.assertEqual(len({result.report(p) for p in (0.9999999, 0.99999999, 1 - 2 ** -53)}), 3)
        with self.assertRaisesRegex(ValueError, "need 0 <= p < 1"):
            result.report(1)

    def test_ordinary_reports_are_what_they_were(self):
        self.assertEqual(Result(1.0, 0.1, "s").report(0.99), "(1.00 ± 0.26) s, k = 2.58, p = 0.99")
        self.assertEqual(Result(1.0, 0.1, "s").report(Fraction(19, 20)), Result(1.0, 0.1, "s").report())
        self.assertEqual(Result(12.311, 0.0097, "ms").report(), "(12.311 ± 0.019) ms, k = 1.96, p = 0.95")
        for name in ("ch02_first_result", "ch04_type_a_b", "ch22_budget"):
            path = ROOT / "expected" / f"{name}.txt"
            if path.exists():
                self.assertNotIn("p = 1\n", path.read_text(encoding="utf-8"))


def by_products(k, n, m):
    """All k messages of the window from the m of the second record: m/(n + m) for the first, and so on."""
    out = Fraction(1)
    for i in range(k):
        out *= Fraction(m - i, n + m - i)
    return out


class ASecondRecordAsTheReference(unittest.TestCase):
    def test_the_table_of_the_review(self):
        got = [silence.depleted(k) for k in (3, 6, 9)]
        self.assertEqual(got, [by_products(k, 60, 60) for k in (3, 6, 9)])
        self.assertEqual([f"{100 * float(g):.4f}" for g in got], ["12.1849", "1.3706", "0.1414"])
        self.assertEqual(got[0], Fraction(math.comb(60, 3), math.comb(120, 3)))
        # With the nine expected messages taken as known the same silence had 0.005823 %.
        self.assertEqual(f"{100 * 0.85 ** 60:.4g}", "0.005823")
        self.assertGreater(got[2], 24 * 0.85 ** 60)

    def test_divisions_counted_one_by_one(self):
        # Seven messages, the first k of them in the window; the first record takes n of the seven.
        for n, m, k in ((3, 4, 2), (4, 3, 2), (3, 4, 4), (2, 5, 1), (4, 4, 0), (3, 4, 5)):
            total = n + m
            if k > total:
                continue
            divisions = list(itertools.combinations(range(total), n))
            none = sum(all(i >= k for i in division) for division in divisions)
            self.assertEqual(silence.depleted(k, n, m), Fraction(none, len(divisions)), (n, m, k))
        self.assertEqual((silence.depleted(0), silence.depleted(61), silence.depleted(60)), (1, 0, Fraction(1, math.comb(120, 60))))
        for call in (lambda: silence.depleted(-1), lambda: silence.depleted(2.5), lambda: silence.depleted(3, -1), lambda: silence.depleted(3, 60, 1.5)):
            with self.assertRaises(ValueError):
                call()

    def test_two_simulated_records(self):
        # Each message of either record falls into the window with one chance, whatever the rates of the two
        # records are; kept are the pairs of records with three messages in the window between them.
        rng = random.Random(2907)
        for n, m, runs in ((60, 60, 60_000), (60, 90, 90_000)):
            kept = none = 0
            for _ in range(runs):
                here = sum(rng.random() < 0.05 for _ in range(n))
                there = sum(rng.random() < 0.05 for _ in range(m))
                if here + there == 3:
                    kept += 1
                    none += here == 0
            three = math.comb(n + m, 3) * 0.05 ** 3 * 0.95 ** (n + m - 3)             # the binomial chance of three
            self.assertAlmostEqual(kept / runs, three, delta=4.5 * math.sqrt(three * (1 - three) / runs), msg=(n, m))
            want = float(by_products(3, n, m))
            self.assertAlmostEqual(none / kept, want, delta=4.5 * math.sqrt(want * (1 - want) / kept), msg=(n, m))
            self.assertEqual(silence.depleted(3, n, m), by_products(3, n, m))

    def test_the_transcript(self):
        out = printed(silence.search)
        piece = ("      a gap of 9 or more: 0.3551 %\n"
                 "  a second record of 60 messages as\n  the reference, k in the window,\n  and none in it here:\n"
                 "    k = 3: 12.18 %\n    k = 6: 1.371 %\n    k = 9: 0.1414 %\n"
                 "  the same 60 requests twice, 9\n")           # the line that follows since Version 31
        self.assertIn(piece, out)
        self.assertEqual(printed(silence.second).count("\n"), 6)
        text = expected("ch16_silence")
        self.assertTrue(text.endswith(out))
        self.assertEqual(len(text.splitlines()), 193)        # 179 in Version 29, 185 in Version 30


if __name__ == "__main__":
    unittest.main()
