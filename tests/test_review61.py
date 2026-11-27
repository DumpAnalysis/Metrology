"""Regression tests added after the review of Version 61.

The review found that Result.interval() returned its limits as floats where a float holds the value, so that the
upper limit of a reading of 10 ** 18 ns lay 39 ns inside its place, on a number at which prob_below() finds less
than the probability of the interval, and a decision by that limit depended on the origin of the clock; that the
step of a float at a time stamp was tied to its number of digits; that a margin of 1.2 ms was named without its
type; that one object whose __iter__() returns one cursor passed as two observed classes; that the lines of the
bound by ranks raised an exception for periods given as generators; that the paired example added a float offset
in floats, which rounds it into two offsets or away; that the half-width of the upper bound was a product of
floats, which can fall below a margin that the exact product exceeds; that report() at a coverage of zero printed
a rounded value beside a width of zero, and rounded a decimal value by the decimal context; that an interval of
a result without uncertainty looked for a coverage factor first; and that sums, half-widths and limits beyond
the range of a float, a coverage below the least float and a trapped decimal context raised errors that named
none of this. It proposed an exercise on request pairs of which six are missing, and a qualifier for the slower
rate of Figure 8.2.

Each test computes what it expects independently of the code under test: the normal distribution by the
statistics module, products and sums by the decimal module at a precision that holds them exactly, Student's
density at zero by the gamma function, percentiles by sorting and counting, and resamples by its own draws.
"""
import contextlib
import datetime
import decimal
import io
import itertools
import math
import random
import unittest
from decimal import ROUND_CEILING, Decimal, FloatOperation, Inexact, localcontext
from fractions import Fraction
from pathlib import Path
from statistics import NormalDist

from examples import ch08_counter as counter
from examples import ch23_case as case
from examples import ch23_mix as mix
from swmetro import Result, core, stats

ROOT = Path(__file__).resolve().parent.parent
D = Decimal
KINDS = (int, float, Decimal, Fraction)
TOP = 1.7976931348623157e308                      # the largest float
Z95 = NormalDist().inv_cdf(0.95)


def expected(name):
    return (ROOT / "expected" / f"{name}.txt").read_text(encoding="utf-8")


def printed(function, *args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        function(*args)
    return out.getvalue()


def product(*floats):
    """The exact product of floats, by the decimal module: no digit may be dropped on the way."""
    with localcontext() as context:
        context.prec = 4000
        context.traps[Inexact] = True
        out = D(1)
        for x in floats:
            out *= D(x)
        return Fraction(out)


def p95(xs):
    """Inverse empirical CDF: the least value with at least 95 % of the values at or below it."""
    xs = sorted(xs)
    for x in xs:
        if 100 * sum(y <= x for y in xs) >= 95 * len(xs):
            return x


class AnIntervalDoesNotMoveWithItsOrigin(unittest.TestCase):
    ORIGIN = 10 ** 18

    def test_the_case_of_the_review(self):
        r = Result(self.ORIGIN, 149_980, "ns")
        low, high = r.interval(0.90)
        self.assertEqual((type(low), type(high)), (Fraction, Fraction))
        # The half-width is 1.6449 standard uncertainties, 246 695.147 ns, and not the 246 656 ns of Version 61,
        # which is the nearest float to the limit, 39.147 ns inside it.
        self.assertAlmostEqual(float(high - self.ORIGIN), 149_980 * Z95, places=6)
        self.assertAlmostEqual(float(high - self.ORIGIN), 246_695.146970, places=5)
        self.assertEqual(self.ORIGIN - low, high - self.ORIGIN)
        self.assertNotEqual(high, self.ORIGIN + 246_656)
        self.assertEqual(float(high), self.ORIGIN + 246_656)                   # what a float makes of the limit
        self.assertAlmostEqual(246_695.146970 - 246_656, 39.147, places=3)
        # prob_below() of the upper limit is the probability of the factor, and that of the float is below it.
        self.assertAlmostEqual(r.prob_below(high), 0.95, places=12)
        self.assertAlmostEqual(r.prob_below(self.ORIGIN + 246_656), NormalDist().cdf(246_656 / 149_980), places=12)
        self.assertLess(r.prob_below(self.ORIGIN + 246_656), 0.95)
        self.assertAlmostEqual(NormalDist().cdf(246_656 / 149_980), 0.9499730743, places=9)
        # The interval covers what it is asked to cover: Version 61 covered 0.89994615.
        cover = NormalDist(0, 149_980)
        self.assertAlmostEqual(cover.cdf(float(high - self.ORIGIN)) - cover.cdf(float(low - self.ORIGIN)), 0.90, places=12)
        self.assertAlmostEqual(2 * NormalDist().cdf(246_656 / 149_980) - 1, 0.89994615, places=8)

    def test_a_decision_by_the_upper_limit(self):
        # A reading is accepted if its upper limit is at or below the tolerance. Whether the origin is
        # subtracted first or kept in the stamp, the answer is one.
        for tolerance in (246_656, 246_695, 246_696, 246_700, 246_784, 300_000):
            there = Result(self.ORIGIN, 149_980, "ns").interval(0.90)[1] <= self.ORIGIN + tolerance
            here = Result(0, 149_980, "ns").interval(0.90)[1] <= tolerance
            self.assertIs(there, here, tolerance)
            self.assertIs(here, tolerance >= 246_696, tolerance)               # the limit is 246 695.147 ns
        # One nanosecond more made Version 61 change the type and the limit; now the limit moves by 1 ns.
        one = Result(self.ORIGIN, 149_980, "ns").interval(0.90)
        two = Result(self.ORIGIN + 1, 149_980, "ns").interval(0.90)
        self.assertEqual((two[0] - one[0], two[1] - one[1]), (1, 1))

    def test_origins_offsets_and_uncertainties(self):
        origins = (0, 10 ** 18, 10 ** 18 + 1, 1_700_000_000_000_000_000 + 128, 2 ** 63 + 1024, -(10 ** 18) - 64)
        checked = 0
        for origin, offset, u in itertools.product(origins, (0, 1, 64, 12_345), (2_048_001, 2_999_999.5)):
            for p in (0.5, 0.9, 0.95):
                near = Result(offset, u, "ns").interval(p)
                far = Result(origin + offset, u, "ns").interval(p)
                self.assertEqual((far[0] - origin, far[1] - origin), near, (origin, offset, u, p))
                half = product(Result(0, 1).k(p), u)
                self.assertEqual(near, (offset - half, offset + half))
                checked += 1
        self.assertEqual(checked, 6 * 4 * 2 * 3)

    def test_a_value_of_any_type(self):
        want = Result(Fraction(1951, 10), 3, "ms").interval()
        self.assertEqual(Result(D("195.1"), 3, "ms").interval(), want)
        # A decimal level is the fraction that it is, which is not the float 0.95.
        self.assertEqual(Result(D("195.1"), D("3"), "ms").interval(D("0.95")),
                         Result(Fraction(1951, 10), 3, "ms").interval(Fraction(19, 20)))
        for value, u in itertools.product(KINDS, repeat=2):
            got = Result(value(195), u(3), "ms").interval()
            self.assertEqual(got, (195 - product(Result(0, 1).k(), 3.0), 195 + product(Result(0, 1).k(), 3.0)))
            self.assertEqual([type(x) for x in got], [Fraction, Fraction])
        # A float rounds a limit once, and fixed() prints it from the fraction, under Python 3.11 as well.
        low, high = Result(195.0, 3.0, "ms").interval()
        self.assertEqual((core.fixed(low, 2), core.fixed(high, 2)), ("189.12", "200.88"))
        self.assertAlmostEqual(float(high), 195 + 3 * NormalDist().inv_cdf(0.975), places=11)

    def test_no_uncertainty_needs_no_factor(self):
        # Version 61 looked for the factor first, and the search for 0.99 at 0.01 degrees of freedom stops.
        self.assertEqual(Result(195, 0, dof=0.01).interval(0.99), (195, 195))
        with self.assertRaises(ValueError):
            Result(195, 3, dof=0.01).k(0.99)
        for value in (195, 195.5, D("195.0004"), Fraction(1, 3)):
            for dof, p in itertools.product((0.01, 1, 40, math.inf), (0, 0.5, 0.99, 1 - 1e-12, D("0.99"))):
                got = Result(value, 0, dof=dof).interval(p)
                self.assertEqual((got, type(got[0]), type(got[1])), ((value, value), type(value), type(value)))
            self.assertEqual(Result(value, 3).interval(0), (value, value))
            self.assertIs(type(Result(value, 3).interval(D("0"))[0]), type(value))
        # The level is still held to its range first.
        for p in (1, 1.5, -0.1, D("1"), Fraction(3, 2)):
            for u in (0, 3):
                with self.assertRaises(ValueError):
                    Result(195, u).interval(p)


class TheStepOfAFloatAtATimeStamp(unittest.TestCase):
    def test_nineteen_digits_do_not_fix_the_step(self):
        rows = ((10 ** 18 + 64, 128, -64), (1_700_000_000_000_000_000 + 128, 256, -128), (2 ** 63 + 1024, 2048, -1024))
        for stamp, step, moved in rows:
            self.assertEqual(len(str(stamp)), 19)
            self.assertEqual(math.ulp(float(stamp)), step)
            self.assertEqual(int(float(stamp)) - stamp, moved)                 # halfway, and the tie goes down
            for near in (stamp - step // 2, stamp + step // 2):
                self.assertEqual(int(float(near)), near)                       # both neighbors are floats
        self.assertGreater(2 ** 63 + 1024, 2 ** 63 - 1)                        # beyond a signed 64-bit count

    def test_the_stamp_of_chapter_22(self):
        stamp = 999_999_999_999_753_280
        self.assertEqual((len(str(stamp)), 10 ** 18 - stamp), (18, 246_720))
        self.assertEqual(math.ulp(float(stamp)), 128)
        self.assertEqual((int(float(stamp - 64)), int(float(stamp + 64))), (stamp - 64, stamp + 64))
        self.assertEqual(int(float(stamp)), stamp - 64)                        # the lower of the two
        low = float(stamp - 64).hex()
        self.assertEqual(int(low.split("p")[0][-1], 16) % 2, 0)                # its last bit is even
        self.assertEqual(10 ** 18 - int(float(stamp)), 246_784)
        r = Result(stamp, 150_000, "ns")
        self.assertAlmostEqual(r.prob_below(10 ** 18), NormalDist().cdf(246_720 / 150_000), places=12)
        self.assertEqual((round(NormalDist().cdf(246_720 / 150_000), 6), round(NormalDist().cdf(246_784 / 150_000), 6)),
                         (0.949994, 0.950038))

    def test_when_the_step_is_256_ns(self):
        # Floats from 2 ** 60 up to 2 ** 61 are 256 apart: as nanoseconds from 1970, from 2006 into 2043.
        self.assertEqual((math.ulp(float(2 ** 60)), math.ulp(math.nextafter(2.0 ** 61, 0)), math.ulp(2.0 ** 61)),
                         (256, 256, 512))
        self.assertEqual(math.ulp(math.nextafter(2.0 ** 60, 0)), 128)
        epoch = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
        years = [(epoch + datetime.timedelta(microseconds=2 ** e // 1000)).year for e in (59, 60, 61)]
        self.assertEqual(years, [1988, 2006, 2043])
        self.assertEqual((math.ulp(1.7e18), math.ulp(1e18)), (256, 128))
        self.assertEqual(datetime.datetime.fromtimestamp(1.7e9, datetime.timezone.utc).year, 2023)
        text = " ".join(core.resolved.__doc__.split())
        for words in ("of 2023", "1.7e18", "256 ns", "1e18", "128 ns"):
            self.assertIn(words, text)

    def test_the_type_of_a_margin(self):
        self.assertTrue(1.3 - 0.1 == 1.2)
        self.assertFalse(1.3 - 0.1 < 1.2)
        self.assertTrue(1.3 - 0.1 < D("1.20"))
        self.assertTrue(Fraction(1.3) - Fraction(0.1) > D("1.20"))
        change = case.exact(1.3, 0.1, -1)
        self.assertIs(case.within(change, D("1.20")), False)
        self.assertIs(case.within(1.3 - 0.1, D("1.20")), True)                 # the float difference would pass
        text = " ".join(case.exact.__doc__.split())
        self.assertIn("1.3 - 0.1 is 1.2, below the decimal 1.20", text)

    def test_the_resolution_check_is_exact(self):
        # 1000 steps of a float at 1e18 are 128 000 ns: an uncertainty of exactly that passes, in any type.
        for kind in KINDS:
            self.assertEqual(Result(10 ** 18, kind(128_000), "ns").u, 128_000)
        for u in (127_999, 127_999.9, D("127999.999999999999999999"), Fraction(127_999_999, 1000), 1, D("1")):
            with self.assertRaisesRegex(ValueError, "too fine"):
                Result(10 ** 18, u, "ns")


class EachStreamIsOpenedOnce(unittest.TestCase):
    class Capture:
        """One object, one cursor: __iter__() returns the cursor that it has."""
        def __init__(self, values):
            self.cursor = iter(values)

        def __iter__(self):
            return self.cursor

    class Once:
        """A stream that opens its cursor once and refuses a second request."""
        def __init__(self, values):
            self.values, self.opened = values, 0

        def __iter__(self):
            self.opened += 1
            if self.opened > 1:
                raise RuntimeError("opened twice")
            return iter(self.values)

    def test_one_cursor_under_two_names(self):
        for call in (lambda r: mix.within(r, r, 200), lambda r: mix.weighted(r, r), mix.classes,
                     lambda r: mix.both(r, r)):
            slow = self.Capture([500, 500])
            rec = {"fast": [100] * 18, "slowA": slow, "slowB": slow}
            with self.assertRaisesRegex(ValueError, "stream in two classes"):
                call(rec)
            self.assertEqual(list(slow.cursor), [500, 500])                    # refused before a request is taken
        # Two objects that hand out one cursor are caught by it as well.
        cursor = iter([500, 500])
        one, two = self.Capture([]), self.Capture([])
        one.cursor = two.cursor = cursor
        with self.assertRaisesRegex(ValueError, "stream in two classes"):
            mix.within({"fast": [100] * 18, "a": one, "b": two}, {"fast": [0] * 18, "a": [0], "b": [0]}, 200)
        self.assertEqual(list(cursor), [500, 500])
        # A generator under two names was refused in Version 61 already.
        stream = (x for x in [500, 500])
        with self.assertRaisesRegex(ValueError, "stream in two classes"):
            mix.classes({"a": stream, "b": stream})

    def test_what_the_two_requests_are(self):
        # Under one name the two slow requests are two of twenty: 90 % within 200 ms, and the p95 is 500 ms.
        slow = self.Capture([500, 500])
        rec = {"fast": [100] * 18, "slow": slow}
        self.assertEqual(mix.within(rec, rec, 200), (Fraction(18, 20), Fraction(18, 20)))
        rec = {"fast": [100] * 18, "slow": self.Capture([500, 500])}
        pairs = mix.weighted(rec, rec)
        self.assertEqual((sum(w for _, w in pairs), mix.p95(pairs)), (20, 500))
        self.assertEqual(p95([100] * 18 + [500] * 2), 500)

    def test_a_stream_that_opens_once(self):
        only = self.Once([100, 200, 300])
        rec = {"only": only}
        pairs = mix.weighted(rec, rec)
        self.assertEqual((sum(w for _, w in pairs), mix.p95(pairs), only.opened), (3, 300, 1))
        rec = {"fast": self.Once([100] * 18), "slow": self.Once([500] * 2)}
        self.assertEqual(mix.within(rec, rec, 200), (Fraction(9, 10), Fraction(9, 10)))
        self.assertEqual([x.opened for x in rec.values()], [1, 1])
        rec = {"fast": self.Once([100] * 18), "slow": self.Once([500] * 2)}
        held, sizes = mix.both(rec, {"fast": range(3), "slow": range(1)})
        self.assertEqual((held, {k: len(v) for k, v in sizes.items()}),
                         ({"fast": [100] * 18, "slow": [500] * 2}, {"fast": 3, "slow": 1}))

    def test_what_can_be_read_again_stays(self):
        again = [500, 500]
        for source in (again, tuple(again), range(500, 502)):
            rec = {"a": source, "b": source}
            held = mix.classes(rec)
            self.assertEqual((held["a"], held["b"]), (list(source), list(source)))
        held = mix.classes({"a": again, "b": again})
        self.assertIs(held["a"], again)                                        # a list is taken as it is
        pairs = mix.weighted({"a": again, "b": again})
        self.assertEqual(sum(w for _, w in pairs), 4)
        self.assertEqual(len(expected("ch23_mix").splitlines()), 126)          # Transcript 23.2; 108 lines until Version 65, 122 until 69


class PeriodsThatCanBeReadOnce(unittest.TestCase):
    @staticmethod
    def rows(n, base):
        return [(0, base + i) for i in range(n)]

    def test_twenty_and_twenty(self):
        lines = printed(case.ranked, iter(self.rows(20, 100)), (r for r in self.rows(20, 110))).splitlines()
        self.assertEqual(lines, ["the p95 by ranks:", "  requests: 20 before, 20 after",
                                 "  no upper end with fewer than 72 after"])
        self.assertEqual(case.rank_bound(iter(self.rows(20, 100)), iter(self.rows(20, 110)))[1:], (None, None))
        self.assertLess(1 - 0.95 ** 71, 0.975)                                 # 71 values do not reach 97.5 %
        self.assertGreaterEqual(1 - 0.95 ** 72, 0.975)

    def test_streams_and_lists_say_the_same(self):
        for n, m in ((20, 20), (2, 72), (1, 72), (1, 20), (72, 72), (100, 71), (300, 250)):
            lists = printed(case.ranked, self.rows(n, 100), self.rows(m, 110))
            streams = printed(case.ranked, iter(self.rows(n, 100)), (r for r in self.rows(m, 110)))
            self.assertEqual(streams, lists, (n, m))
            self.assertEqual("requests:" in lists, n < 2 or m < 72, (n, m))
            if n < 2 or m < 72:
                self.assertIn(f"  requests: {n} before, {m} after\n", lists)
        self.assertEqual(expected("ch23_case").count("the p95 by ranks:"), 2)


class AnOffsetIsAddedAsTheNumberThatItIs(unittest.TestCase):
    def test_the_three_calls_of_the_review(self):
        # h + 0.2 in floats: two offsets, neither of them the float 0.2.
        formed = {Fraction(100 + i + 0.2) - (100 + i) for i in range(30)}
        self.assertEqual(len(formed), 2)
        self.assertNotIn(Fraction(0.2), formed)
        self.assertEqual({100 + i + 1e-15 for i in range(100)}, {float(100 + i) for i in range(100)})
        for n, offset, shown in ((30, 0.2, format(D(0.2), "f")), (100, 1e-15, format(D(1e-15), "f")),
                                 (30, D("0.2"), "0.2"), (100, 1, "1"), (60, Fraction(5, 2), "2.5")):
            lines = printed(case.side_by_side, n, offset).splitlines()
            self.assertEqual(lines[0], f"{n} requests reported twice, {offset} ms apart:")
            # In pairs every replicate is the offset, whatever is drawn: the spread is zero and the value exact.
            self.assertEqual(lines[3], f"  as pairs, one draw for both: {shown}(0) ms", (n, offset))
        self.assertEqual(format(D(0.2), "f"), "0.200000000000000011102230246251565404236316680908203125")
        self.assertTrue(expected("ch23_case").endswith(printed(case.side_by_side)))

    def test_every_draw_gives_the_offset(self):
        rng = random.Random(61)
        for offset in (0.2, 1e-15, D("0.2"), 1, Fraction(1, 3)):
            one = [100 + i for i in range(30)]
            two = [Fraction(h) + Fraction(offset) for h in one]
            for _ in range(300):
                ids = [rng.randrange(30) for _ in range(30)]
                self.assertEqual(p95(two[i] for i in ids) - p95(one[i] for i in ids), Fraction(offset))


class TheHalfWidthIsAnExactProduct(unittest.TestCase):
    def test_the_factor(self):
        self.assertAlmostEqual(case.Z, Z95, places=13)

    def test_the_bound_of_the_review(self):
        u = 0.1884666178926485
        d = Result(0, u, "ms")
        exact = product(case.Z, u)
        self.assertGreater(exact, D("0.31"))                                   # 0.31000000000000000684...
        self.assertLess(case.Z * u, D("0.31"))                                 # 0.30999999999999999778...
        self.assertEqual(str(D(case.Z * u))[:22], "0.30999999999999999777")
        with localcontext() as context:
            context.prec = 60
            self.assertEqual(str(D(case.Z) * D(u))[:23], "0.310000000000000006841")
        upper = case.upper_bound(d)
        self.assertEqual(upper, exact)
        self.assertIs(case.within(upper, D("0.31")), False)                    # Version 61: True
        self.assertEqual((stats.upward(upper, 2), stats.upward(case.Z * u, 2)), ("0.32", "0.31"))

    def test_an_uncertainty_of_any_type(self):
        for kind in KINDS:                                                     # Version 61: TypeError for a decimal
            upper = case.upper_bound(Result(1, kind(3), "ms"))
            self.assertEqual(upper, 1 + product(case.Z, 3.0))
        self.assertEqual(case.upper_bound(Result(D("0.3"), 0, "ms")), Fraction(3, 10))
        rng = random.Random(61)
        for _ in range(2000):
            value, u = rng.gauss(0, 50), rng.uniform(0.001, 20)
            upper = case.upper_bound(Result(value, u, "ms"))
            self.assertEqual(upper, Fraction(value) + product(case.Z, u))
            self.assertIs(type(upper), Fraction)

    def test_the_transcript_keeps_its_bounds(self):
        rng = random.Random(23)
        before, after = case.period(rng), case.period(rng)
        d = case.handler_change(before, after)
        upper = case.upper_bound(d)
        self.assertEqual(upper, Fraction(d.value) + product(case.Z, d.u))
        self.assertAlmostEqual(float(upper), 4.7078728237, places=10)
        self.assertEqual((stats.upward(upper, 1), case.within(upper, 5.0), case.within(upper, 2.0)), ("4.8", True, False))
        self.assertIn("upper bound, 95 %: 4.8 ms\n", expected("ch23_case"))
        self.assertIn("    at most 6.49 ms at 95 %\n", expected("ch23_case"))


class AReportAtItsEdges(unittest.TestCase):
    def test_no_coverage(self):
        for value, shown in ((D("195.0004"), "195.0004"), (Fraction(1, 3), "1/3"), (195, "195"), (0.1, "0.1"),
                             (Fraction(1, 8), "0.125"), (10 ** 18 + 1, "1000000000000000001")):
            for u in (300_000, 300_000.0, D("300000"), 0):
                r = Result(value, u, "ms")
                for p in (0, 0.0, D("0"), Fraction(0)):
                    # Since Version 63 a result without uncertainty has "u = 0" where its factor stood.
                    tail = "k = 0.00" if u else "u = 0"
                    self.assertEqual(r.report(p), f"({shown} ± 0) ms, {tail}, p = {p}")
                    self.assertEqual(r.interval(p), (value, value))
        self.assertEqual(Result(D("195.0004"), 3, "ms").report(0), "(195.0004 ± 0) ms, k = 0.00, p = 0")
        # With a coverage above zero the value is rounded to the place of the width, as before.
        self.assertEqual(Result(D("195.0004"), 3, "ms").report(), "(195.0 ± 5.9) ms, k = 1.96, p = 0.95")

    def test_a_decimal_tie_and_the_context(self):
        want = {125: "(120 ± 200)", 135: "(140 ± 200)", 145: "(140 ± 200)", 155: "(160 ± 200)", -125: "(-120 ± 200)"}
        modes = (decimal.ROUND_HALF_EVEN, decimal.ROUND_HALF_UP, decimal.ROUND_UP, decimal.ROUND_DOWN,
                 decimal.ROUND_CEILING, decimal.ROUND_FLOOR, decimal.ROUND_HALF_DOWN, decimal.ROUND_05UP)
        for value, body in want.items():
            self.assertEqual(Fraction(body[1:].split(" ")[0]), round(Fraction(value, 10)) * 10)       # half to even
            for kind, mode, trap in itertools.product(KINDS, modes, (False, True)):
                with localcontext() as context:
                    context.rounding = mode
                    context.traps[Inexact] = trap
                    r = Result(kind(value), 100, "ms")
                    self.assertEqual(r.report(), f"{body} ms, k = 1.96, p = 0.95", (value, kind, mode, trap))
                    self.assertEqual(str(r), f"{body[1:].split(' ')[0]}(100) ms")
        # A width in the tens as well: 1250 with u = 100 at 68 % is (1250 ± 100) and has no tie.
        self.assertEqual(Result(D("1250"), 100, "ms").report(0.6827), "(1250 ± 100) ms, k = 1.00, p = 0.6827")
        # A negative float that rounds to zero at the tens has no sign there: Version 61 printed (-0 ± 57000).
        self.assertEqual(Result(-0.0003886662419115812, 29189.727626220632, "ms").report(),
                         "(0 ± 57000) ms, k = 1.96, p = 0.95")

    def test_a_sum_beyond_the_range_of_a_float(self):
        got = case.exact(1e308, 1e308)
        self.assertEqual((got, type(got)), (2 * Fraction(1e308), Fraction))    # Version 61: OverflowError
        got = case.exact(-1e308, 1e308, -1)
        self.assertEqual((got, type(got)), (-2 * Fraction(1e308), Fraction))
        got = case.exact(TOP, math.ulp(TOP))
        self.assertEqual((got, type(got)), (Fraction(TOP) + Fraction(math.ulp(TOP)), Fraction))
        got = case.exact(1e308, 1e308, -1)
        self.assertEqual((got, type(got)), (0.0, float))
        got = case.exact(1e308, 7e307)
        self.assertEqual((got, type(got)), (1e308 + 7e307, float) if Fraction(1e308 + 7e307) ==
                         Fraction(1e308) + Fraction(7e307) else (Fraction(1e308) + Fraction(7e307), Fraction))

    def test_a_half_width_beyond_the_range_of_a_float(self):
        r = Result(0, 1e308, "ms")
        half = product(r.k(), 1e308)
        self.assertGreater(half, Fraction(TOP))
        with self.assertRaisesRegex(OverflowError, "U is beyond a float"):
            r.expanded()
        self.assertEqual(r.interval(), (-half, half))
        # Two significant digits of 1.96e308 are 2.0e308, written out: 2 and 308 zeros.
        self.assertEqual(r.report(), "(0 ± 2" + "0" * 308 + ") ms, k = 1.96, p = 0.95")
        self.assertEqual(str(r), "0(" + str(round(Fraction(1e308) / 10 ** 307) * 10 ** 307) + ") ms")
        # The half-width fits, and the upper limit does not.
        r = Result(1.7e308, 1e307, "ms")
        half = product(r.k(), 1e307)
        self.assertLess(half, Fraction(TOP))
        self.assertEqual(r.expanded(), float(half))
        low, high = r.interval()
        self.assertEqual((low, high), (Fraction(1.7e308) - half, Fraction(1.7e308) + half))
        self.assertGreater(high, Fraction(TOP))
        with self.assertRaises(OverflowError):
            float(high)
        centre, width = round(Fraction(1.7e308) / 10 ** 306) * 10 ** 306, round(half / 10 ** 306) * 10 ** 306
        self.assertEqual(r.report(), f"({centre} ± {width}) ms, k = 1.96, p = 0.95")
        self.assertEqual((len(str(centre)), str(width)[:2], len(str(width))), (309, "20", 308))
        # A value or an uncertainty beyond the range of a float is refused, as the class says.
        for value, u in ((10 ** 400, 1), (Fraction(10) ** 400, 1), (D("1e400"), 1), (0, 10 ** 400), (0, D("1e400"))):
            with self.assertRaises((OverflowError, ValueError)):
                Result(value, u)

    def test_a_coverage_below_the_least_float(self):
        root = math.sqrt(2 * math.pi) / 2                                     # k / p of the normal distribution at 0
        r = Result(0, 1e300, "ms")
        self.assertAlmostEqual(r.expanded(D("1e-330")) / (root * 1e-30), 1, places=12)   # Version 61: ValueError
        self.assertEqual(r.k(D("1e-330")), 0.0)                                # no float is so small
        low, high = r.interval(D("1e-330"))
        self.assertEqual((-low, float(high)), (high, r.expanded(D("1e-330"))))
        self.assertGreater(high, 0)
        # Student's distribution: the density at zero is gamma((v + 1) / 2) / (sqrt(v pi) gamma(v / 2)).
        for dof in (0.5, 1, 3, 40):
            at_zero = math.gamma((dof + 1) / 2) / (math.sqrt(dof * math.pi) * math.gamma(dof / 2))
            for p, size in ((D("1e-330"), 1e-330), (D("1e-400"), 1e-400), (Fraction(1, 10 ** 500), 1e-500)):
                got = Result(0, 1e300, dof=dof).interval(p)[1]
                want = Fraction(p) * Fraction(1e300) / (2 * Fraction(at_zero))
                self.assertAlmostEqual(float(got / want), 1, places=10, msg=(dof, p))
        # Floats below 1e-300 keep the digits that they have: the width is in proportion to the level.
        for p in (1e-301, 1e-305, 1e-310, 4e-320, 5e-324):
            got = Fraction(r.interval(p)[1]) / Fraction(p)
            self.assertAlmostEqual(float(got) / (root * 1e300), 1, places=12, msg=p)
        # At 1e-300 and above nothing changed.
        for p in (1e-300, 1e-250, 1e-18, 0.1, 0.49):
            self.assertEqual(r.interval(p)[1], product(r.k(p), 1e300))
            self.assertAlmostEqual(2 * NormalDist().cdf(Result(0, 1).k(p)) - 1, p, delta=1e-15 + 1e-12 * p)
        # A report at such a level has its width in full: 1.3e-30 after 29 zeros.
        self.assertEqual(Result(0, 1, "ms").report(D("1e-30")),
                         "(" + "0." + "0" * 31 + " ± 0." + "0" * 29 + "13) ms, k = 0." + "0" * 29 + "13, p = 1E-30")
        # Version 62 printed "k = 0.00" beside that width: a factor keeps two significant digits since Version 63.

    def test_a_trapped_decimal_context(self):
        bound = Fraction(3, 2) + product(case.Z, 3.0)
        with localcontext() as context:
            context.traps[FloatOperation] = True
            r = Result(1.5, D("3"), "ms")                                      # Version 61: FloatOperation
            self.assertEqual(str(r), "1.5(30) ms")
            self.assertEqual(r.report(), "(1.5 ± 5.9) ms, k = 1.96, p = 0.95")
            self.assertEqual(r.interval(), Result(1.5, 3, "ms").interval())
            self.assertEqual(r.prob_below(D("1.5")), 0.5)
            self.assertAlmostEqual(r.prob_below(2.5), NormalDist().cdf(1 / 3), places=14)
            for value, limit, want in ((1.5, D("2"), 1.0), (1.5, D("1.5"), 1.0), (1.5, D("1.4999"), 0.0),
                                       (D("1.5"), 1.5, 1.0), (D("0.3"), 0.3, 0.0), (0.3, D("0.3"), 1.0),
                                       (1.5, D("Infinity"), 1.0), (1.5, D("-Infinity"), 0.0),
                                       (D("1.5"), math.inf, 1.0), (D("1.5"), -math.inf, 0.0)):
                self.assertEqual(Result(value, 0, "ms").prob_below(limit), want, (value, limit))
            with self.assertRaisesRegex(ValueError, "too fine"):
                Result(1e18, D("1"), "ns")
            self.assertEqual(case.upper_bound(Result(1.5, D("3"), "ms")), bound)
        # A point mass is compared exactly: a limit 1e-400 below the value is below it.
        tiny = Fraction(1, 10 ** 400)
        self.assertEqual([Result(Fraction(1), 0).prob_below(1 + step) for step in (-tiny, 0, tiny)], [0.0, 1.0, 1.0])
        self.assertEqual([Result(1.0, 0).prob_below(x) for x in (math.inf, -math.inf)], [1.0, 0.0])
        with self.assertRaises(ValueError):
            Result(1.0, 0).prob_below(math.nan)

    def test_the_decimal_exponent(self):
        rng = random.Random(61)
        xs = [10.0 ** k for k in range(-300, 301, 7)] + [5e-324, TOP, 2.2250738585072014e-308, 9.999999999999999e22]
        xs += [math.nextafter(x, 0) for x in xs[:80]] + [rng.uniform(0, 1) * 10 ** rng.randint(-300, 300) for _ in range(3000)]
        for x in xs:
            self.assertEqual(core._exp10(x), D(x).adjusted(), x)               # the exponent of the first digit
        self.assertEqual((core._exp10(Fraction(10) ** 400), core._exp10(Fraction(1, 10 ** 400)),
                          core._exp10(Fraction(10) ** 400 - 1), core._exp10(D("9.99e-5000"))), (400, -400, 399, -5000))
        self.assertEqual((core.decimals(Fraction(10) ** 400), core.decimals(0.0996), core.decimals(D("0.0996")),
                          core.decimals(0), core.decimals(math.inf), core.decimals(math.nan)), (-399, 2, 2, 3, 3, 3))


class SixPairsAreMissing(unittest.TestCase):
    A = [100] * 100
    B = [101] * 94

    def test_the_complete_pairs(self):
        self.assertEqual(p95(self.B) - p95(self.A[:94]), 1)
        rng = random.Random(61)
        for _ in range(300):
            ids = [rng.randrange(94) for _ in range(94)]
            self.assertEqual(p95(self.B[i] for i in ids) - p95(self.A[i] for i in ids), 1)

    def test_all_hundred_with_six_unknown(self):
        rec = {"requests": self.B + [None] * 6}
        low, high = mix.within(rec, rec, 105)
        self.assertEqual((low, high, mix.verdict(low, high)), (Fraction(94, 100), 1, "unresolved"))
        self.assertEqual((sum(x <= 105 for x in self.B), len(self.B) + 6), (94, 100))
        # One reading of 101 ms recovered: 95 of 100 are known to be within 105 ms.
        rec = {"requests": self.B + [101] + [None] * 5}
        low, high = mix.within(rec, rec, 105)
        self.assertEqual((low, high, mix.verdict(low, high)), (Fraction(95, 100), 1, "met"))

    def test_what_the_six_can_be(self):
        self.assertEqual(p95(self.B + [101] * 6) - p95(self.A), 1)
        self.assertEqual(p95(self.B + [201] * 6) - p95(self.A), 101)
        # Every completion: +1 ms if one of the six is at or below 101 ms, and else the least of them less 100.
        for six in itertools.combinations_with_replacement((0, 50, 101, 102, 201, 10 ** 6), 6):
            want = 1 if min(six) <= 101 else min(six) - 100
            self.assertEqual(p95(self.B + list(six)) - p95(self.A), want, six)

    def test_one_reading_recovered(self):
        for five in itertools.product((0, 101, 201, 10 ** 9), repeat=5):
            self.assertEqual(p95(self.B + [101] + list(five)) - p95(self.A), 1, five)
        # The 95th of a hundred in order: with 95 readings of 101 ms it is 101 ms wherever the other five lie.
        self.assertEqual(sorted(self.B + [101] + [10 ** 9] * 5)[94], 101)
        self.assertEqual(sorted(self.B + [10 ** 9] * 6)[94], 10 ** 9)


class TheSlowerRateOfFigure82(unittest.TestCase):
    def test_the_rate_and_its_label(self):
        first, last = counter.wrapping(100, 2)
        slow, fast = counter.increments(first, last, 60_000_000 * 100)
        self.assertEqual((first, last, slow, fast), (0, 705_032_704, 705_032_704, 5_000_000_000))
        self.assertEqual(5_000_000_000 - 2 ** 32, 705_032_704)
        self.assertEqual(Fraction(slow, 100) / 10 ** 6, Fraction("7.05032704"))       # MB/s, exactly
        self.assertEqual(Fraction("7.05") * 10 ** 6 * 100, 705_000_000)
        self.assertEqual(slow - 705_000_000, 32_704)
        self.assertEqual(Fraction(2 ** 32, 50 * 10 ** 6), Fraction("85.89934592"))    # s to the wrap
        self.assertEqual(Fraction(last, 10 ** 9), Fraction("0.705032704"))            # GB
        self.assertEqual((f"{slow / 1e8:.2f}", f"{2 ** 32 / 5e7:.1f}", f"{last / 1e9:.3f}"), ("7.05", "85.9", "0.705"))
        source = (ROOT / "figures" / "make_figures.py").read_text(encoding="utf-8")
        self.assertIn('f"\\u2248 {slow / 1e8:.2f} MB/s"', source)
        self.assertIn("   705032704 bytes, 7.05 MB/s\n", expected("ch08_counter"))


if __name__ == "__main__":
    unittest.main()
