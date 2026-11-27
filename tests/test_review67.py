"""Regression tests added after the review of Version 67.

The review found that whole numbers of a fixed width, as the arrays of NumPy hold them, wrapped around in the
sums of mean() and in the interpolation of quantile(), so that two readings of 100 in eight bits had the mean
-28, and that Result failed for them under one interpreter and for floats of four bytes under all; that
quantile() said it did not round between two readings that are no floats, and rounded; that a truth value was
taken for a value, an uncertainty or a count, so that Result(200, False, "ms") reported "(200 ± 0) ms" and
quantile_verdict() read False as no unknown request; that components() of Listing 23.2 printed "no limit" for
the change between an open duration and one known to be infinite, which is one infinity wherever it is
defined, and reported an order of a percentile as not determined although one open duration leaves the change
of the mix at exactly zero; that t_isf() refused a quantile just below its ceiling of 1e150, failed for a
decimal tail, lost the distance of a fraction from one half, and returned quantiles for tails below the least
float, where it has no accuracy; and that fit_temperature() refused a minimum that rounds to the largest
float.

Each test computes what it expects independently of the code under test: means and interpolated quantiles
from fractions; a percentile at a mix by writing every request as often as its weight says and counting; the
parts of a change of a percentile by putting durations in the place of the open one, at every duration that
is known, below them all and above them all; Poisson bounds from the sum of the two terms; Student's tail for
one, two and four degrees of freedom from its closed form in decimals; and a fitted temperature from its
closed form in decimals.
"""
import contextlib
import io
import math
import numbers
import random
import sys
import unittest
from decimal import Decimal, localcontext
from fractions import Fraction

from examples import ch16_bounds as bounds
from examples import ch18_calibration as cal
from examples import ch23_mix as mix
from swmetro import Budget, Result, combine, dist, fmt, rect_interval, stats, type_a, u_rect, u_resolution
from swmetro import u_tri, welch_satterthwaite
from swmetro import record
from swmetro.propagate import lpu, summarize
from swmetro.units import Q

try:
    import numpy as np
except ImportError:                                                       # the package itself needs no NumPy
    np = None

D = Decimal
INF = math.inf
PI = D("3.14159265358979323846264338327950288419716939937510582097494")
FLAGS = (False, True)
P95 = Fraction(95, 100)
CEILING = 1e150
LEAST = 2.0 ** -1022


def printed(function, *args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        function(*args)
    return out.getvalue()


def refusal(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except (ValueError, TypeError) as error:
        return f"{type(error).__name__}: {error}"
    return None


def kind(function, *args, **kwargs):
    """The name of the exception that a call raises, or None."""
    text = refusal(function, *args, **kwargs)
    return text and text.split(":")[0]


class Wrap8:
    """A whole number of eight bits that wraps around, as numpy.int8 does, for the interpreters without NumPy.
    It is registered as an Integral, as the integers of NumPy are, and its numerator is itself."""

    def __init__(self, n):
        self.n = (int(n) + 128) % 256 - 128

    numerator = property(lambda self: self)
    denominator = 1

    def __int__(self):
        return self.n

    __index__ = __int__

    def __float__(self):
        return float(self.n)

    def __add__(self, other):
        return Wrap8(self.n + int(other))

    __radd__ = __add__

    def __sub__(self, other):
        return Wrap8(self.n - int(other))

    def __rsub__(self, other):
        return Wrap8(int(other) - self.n)

    def __mul__(self, other):
        return Wrap8(self.n * int(other))

    __rmul__ = __mul__

    def __neg__(self):
        return Wrap8(-self.n)

    def __abs__(self):
        return Wrap8(abs(self.n))

    def __eq__(self, other):
        return self.n == other

    def __lt__(self, other):
        return self.n < other

    def __le__(self, other):
        return self.n <= other

    def __gt__(self, other):
        return self.n > other

    def __ge__(self, other):
        return self.n >= other

    def __hash__(self):
        return hash(self.n)

    def __repr__(self):
        return f"Wrap8({self.n})"


numbers.Integral.register(Wrap8)


def often(pairs):
    """(value, count) with whole counts in the ratio of the weights."""
    pairs = [(x, Fraction(w)) for x, w in pairs]
    scale = math.lcm(*(w.denominator for _, w in pairs))
    return sorted((x, int(w * scale)) for x, w in pairs)


def counted(pairs, q=P95):
    """The inverse empirical quantile of weighted values by counting: every value is written as often as its
    weight says, and the value of rank ceil(q n), counted from 1, is read off."""
    rows = often(pairs)
    total = sum(n for _, n in rows)
    rank = max(1, -((-q.numerator * total) // q.denominator))
    for x, n in rows:
        rank -= n
        if rank <= 0:
            return x
    raise AssertionError("no value of that rank")


def at_mix(rec, ref, x):
    """The pairs of a record at the class counts of `ref`, with x in the place of every open duration."""
    return [(x if d is None else d, Fraction(len(ref[c]), len(ds))) for c, ds in rec.items() for d in ds]


def less(a, b):
    """a - b, or None where both are the same infinity."""
    return None if a == b == INF else a - b


def extent(values):
    """(least, greatest, is any of them not defined) of differences that less() formed."""
    some = [v for v in values if v is not None]
    return min(some), max(some), len(some) < len(values)


def places(rec):
    """Where an open duration of a record can stand so that every least and greatest value of a statistic by
    rank is met: at zero, at each duration that is known, above them all, and at infinity."""
    seen = sorted({d for ds in rec.values() for d in ds if d is not None})
    return [0] + seen + [seen[-1] + 1, INF]


class FixedWidthReadings(unittest.TestCase):
    """Section 1 of the review: two readings of 100 in eight bits have the mean 100."""

    def test_native_keeps_what_is_python_s_own(self):
        for x in (7, -3, 0, 2 ** 80, 0.1, -0.0, 5e-324, INF, Fraction(1, 3), D("0.1"), D("1e-400")):
            self.assertIs(stats.native(x), x)
        self.assertTrue(math.isnan(stats.native(math.nan)))

    def test_native_refuses_what_is_no_number(self):
        for bad in (True, False, None, "7", b"7", 2 + 3j, [7], (7,), object()):
            self.assertEqual(kind(stats.native, bad), "TypeError", repr(bad))

    def test_a_whole_number_of_eight_bits_becomes_an_int(self):
        x = Wrap8(100)
        self.assertEqual(int(x + x), -56)                                     # the stand-in wraps as int8 does
        got = stats.native(x)
        self.assertIs(type(got), int)
        self.assertEqual(got, 100)

    def test_the_mean_of_two_readings_of_100(self):
        xs = [Wrap8(100), Wrap8(100)]
        got = stats.mean(xs)
        self.assertEqual((got, type(got)), (100, int))                        # Version 67: -28
        self.assertEqual(stats.mean([Wrap8(100), Wrap8(101)]), 100.5)

    def test_the_interpolated_percentile_of_100_and_101(self):
        xs = [Wrap8(100), Wrap8(101)]
        want = 100 + P95 * (101 - 100)                                        # rank 0.95 between two readings
        self.assertEqual(want, Fraction(2019, 20))
        self.assertEqual(stats.quantile(xs, 0.95, exact=True), want)          # Version 67: -1.45
        self.assertEqual(stats.quantile(xs, 0.95), float(want))
        self.assertEqual(stats.quantile(xs, 1), 101)
        self.assertIs(type(stats.quantile(xs, 1)), int)

    def test_the_decomposition_of_listing_23_2(self):
        text = printed(mix.components, {"web": [100]}, {"web": [Wrap8(100), Wrap8(100)]})
        self.assertEqual(text.splitlines()[0], "mean, 100 -> 100 ms: +0 ms")  # Version 67: 100 -> -28 ms: -128 ms
        got = mix.mean([(Wrap8(100), 1), (Wrap8(100), 1)])
        self.assertEqual(got, 100)
        self.assertEqual(mix.p95([(Wrap8(100), 1), (Wrap8(101), Wrap8(1))]), 101)

    def test_every_route_of_a_reading(self):
        x, y = Wrap8(100), Wrap8(120)
        self.assertEqual(stats.known([x, y]), [100, 120])
        self.assertTrue(all(type(v) is int for v in stats.known([x, y])))
        self.assertEqual(stats.weighted_quantile([(x, Wrap8(3)), (y, Wrap8(1))], 0.5), 100)
        self.assertEqual(stats.upward(x, 1), "100.0")
        self.assertEqual(stats.downward(y, 0), "120")
        a, b, _, _, s = stats.linfit([Wrap8(100), Wrap8(110), Wrap8(120)], [Wrap8(50), Wrap8(70), Wrap8(90)])
        self.assertEqual((a, b, s), (-150, 2, 0.0))                           # y = 2 x - 150, which no sum wraps
        r = Result(Wrap8(100), Wrap8(2), "ms")
        self.assertEqual((r.value, type(r.value), r.u, type(r.u)), (100, int, 2, int))
        self.assertEqual(r.report(), Result(100, 2, "ms").report())
        self.assertEqual(type_a([x, y, y]).value, float(Fraction(340, 3)))    # rounded once: floats hold them
        self.assertEqual(type_a([x, y, y]).u, type_a([100, 120, 120]).u)
        self.assertEqual(combine(Wrap8(3), Wrap8(4)), 5.0)
        self.assertEqual(Budget("ms").add("a", "B", Wrap8(3)).add("b", "A", Wrap8(4), Wrap8(7)).rows,
                         [("a", "B", 3, INF), ("b", "A", 4, 7)])
        self.assertEqual(welch_satterthwaite([Wrap8(3)], [5]), 5.0)
        self.assertEqual(u_rect(Wrap8(3)), 3 / math.sqrt(3))
        self.assertEqual(rect_interval(Wrap8(100), Wrap8(10), Fraction(1, 2)), (95, 105))
        self.assertEqual(fmt(Wrap8(100), Wrap8(2), "ms"), fmt(100, 2, "ms"))
        self.assertEqual(Q(Wrap8(100), "ms").to("us").value, 100000)
        y0, u, parts = lpu(lambda v: v[0] + v[1], [Wrap8(100), Wrap8(100)], [Wrap8(3), Wrap8(4)])
        self.assertEqual((y0, u, parts), (200, 5.0, [3.0, 4.0]))              # f is given ints: 200, not -56
        self.assertEqual(summarize([Wrap8(100)] * 30, 0.9), (100, 0.0, 100, 100))
        self.assertEqual(bounds.quantile_verdict(Wrap8(100), 0.95, Wrap8(94), Wrap8(1)), "unresolved")

    def test_time_stamps_keep_their_full_width(self):
        stamps = [2 ** 62, 2 ** 62 + 2]
        self.assertEqual(stats.mean(stamps), 2 ** 62 + 1)                     # their sum is beyond 64 bits
        self.assertEqual(stats.quantile(stamps, 0.5), 2 ** 62 + 1)            # a float does not hold it
        self.assertEqual(float(2 ** 62 + 1), float(2 ** 62))


@unittest.skipUnless(np, "NumPy is not installed")
class TheScalarsOfNumPy(unittest.TestCase):
    """The same with the types of NumPy themselves, where NumPy is installed."""

    def test_the_two_examples_of_the_review(self):
        xs = [np.int8(100), np.int8(100)]
        text = printed(mix.components, {"web": [100]}, {"web": xs})
        self.assertEqual(text.splitlines()[0], "mean, 100 -> 100 ms: +0 ms")  # Version 67: -28 ms and -128 ms
        got = stats.quantile([np.int8(100), np.int8(101)], 0.95)
        self.assertEqual((got, type(got)), (100.95, float))                   # Version 67: -1.45
        self.assertEqual(stats.quantile([np.int8(100), np.int8(101)], 0.95, exact=True), Fraction(2019, 20))

    def test_the_mean_of_two_readings(self):
        got = stats.mean([np.int8(100), np.int8(100)])
        self.assertEqual((got, type(got)), (100, int))
        got = stats.mean([np.int32(1_500_000_000)] * 2)
        self.assertEqual((got, type(got)), (1_500_000_000, int))              # Version 67: -647483648
        stamps = [np.int64(2 ** 62), np.int64(2 ** 62 + 2)]
        self.assertEqual(stats.mean(stamps), 2 ** 62 + 1)
        self.assertEqual(stats.native(np.uint64(2 ** 64 - 1)), 2 ** 64 - 1)

    def test_a_result_from_scalars(self):
        want = Result(200, 2, "ms").report()
        self.assertEqual(want, "(200.0 ± 3.9) ms, k = 1.96, p = 0.95")
        self.assertEqual(Result(np.int64(200), np.int64(2), "ms").report(), want)      # 3.12: AttributeError
        self.assertEqual(Result(np.float32(200), np.float32(2), "ms").report(), want)  # Version 67: TypeError
        self.assertEqual(Result(np.float64(200), np.float64(2), "ms").report(), want)
        r = Result(np.int64(200), np.float32(2), "ms", np.int64(7))
        self.assertEqual([type(v) for v in (r.value, r.u, r.dof)], [int, float, float])

    def test_floats_of_other_widths(self):
        x = np.float32(0.1)
        got = stats.native(x)
        self.assertIs(type(got), float)
        self.assertEqual(Fraction(got), Fraction(*x.as_integer_ratio()))      # the number that it is, not 0.1
        self.assertNotEqual(got, 0.1)
        self.assertIs(type(stats.native(np.float64(0.1))), float)
        self.assertEqual(stats.native(np.float64(0.1)), 0.1)
        self.assertEqual(stats.native(np.float16(0.5)), 0.5)
        self.assertTrue(math.isnan(stats.native(np.float32("nan"))))
        self.assertEqual(stats.native(np.float32("inf")), INF)
        wide = np.longdouble(1) / np.longdouble(3)
        got = stats.native(wide)
        self.assertEqual(Fraction(got), Fraction(*map(int, wide.as_integer_ratio())))
        if np.finfo(np.longdouble).nmant > 52:                                # more digits than a float has
            self.assertIs(type(got), Fraction)

    def test_a_flag_of_numpy_is_no_number(self):
        for flag in (np.True_, np.False_):
            self.assertEqual(kind(stats.native, flag), "TypeError")
            self.assertEqual(kind(stats.as_count, flag), "ValueError")
            self.assertEqual(kind(Result, 200, flag, "ms"), "TypeError")
        self.assertEqual(stats.as_count(np.int8(5)), 5)
        self.assertIs(type(stats.as_count(np.int8(5))), int)


class InterpolationAsItIsDescribed(unittest.TestCase):
    """Section 1, second part: the docstring of quantile() and what it returns."""

    def test_between_100_and_101(self):
        want = Fraction(2019, 20)
        for xs in ([100, 101], [D(100), D(101)], [Fraction(100), Fraction(101)], [100.0, 101.0]):
            got = stats.quantile(xs, 0.95)
            self.assertIs(type(got), float)
            self.assertEqual(got, 100.95)
            self.assertGreater(Fraction(got), want)                           # the float lies above 2019/20
            self.assertEqual(Fraction(got) - want, Fraction(1, 351843720888320))
            exact = stats.quantile(xs, 0.95, exact=True)
            self.assertEqual((exact, type(exact)), (want, Fraction))
            self.assertLessEqual(exact, want)                                 # a limit of 100.95 is met
        self.assertEqual(stats.ecdf_quantile([100, 101], 0.95), 101)          # the inverse empirical p95

    def test_exact_against_the_formula(self):
        rng = random.Random(67)
        for _ in range(300):
            n = rng.randint(1, 12)
            xs = sorted(rng.choice((rng.randint(-50, 50), rng.uniform(-5, 5), Fraction(rng.randint(-99, 99), 7),
                                    D(rng.randint(-999, 999)) / 10)) for _ in range(n))
            q = Fraction(rng.randint(0, 100), 100)
            pos = q * (n - 1)
            i = pos.numerator // pos.denominator
            want = Fraction(xs[i])
            if pos != i:
                want += (pos - i) * (Fraction(xs[i + 1]) - Fraction(xs[i]))
            got = stats.quantile(xs, q, exact=True)
            self.assertEqual(Fraction(got), want, (xs, q))
            rounded = stats.quantile(xs, q)
            self.assertLessEqual(abs(Fraction(rounded) - want), abs(want) * Fraction(1, 2 ** 52), (xs, q))
            self.assertTrue(xs[i] <= rounded <= xs[min(i + 1, n - 1)], (xs, q))

    def test_what_a_float_does_not_hold_was_never_rounded(self):
        stamps = [10 ** 18 + 1, 10 ** 18 + 4]
        self.assertEqual(stats.quantile(stamps, 0.5), Fraction(2 * 10 ** 18 + 5, 2))
        self.assertEqual(stats.quantile([D("0.1"), D("0.2")], 0.5), Fraction(3, 20))
        self.assertEqual(stats.quantile(stamps, 0.5, exact=True), stats.quantile(stamps, 0.5))

    def test_the_docstring_says_what_is_done(self):
        text = " ".join(stats.quantile.__doc__.split())
        self.assertIn("rounded once, to a float between them, if floats hold both", text)
        self.assertIn("With `exact` it is never rounded", text)
        self.assertNotIn("between two that are no floats it is not rounded at all", text)


class AFlagIsNoNumber(unittest.TestCase):
    """Section 2 of the review: each route with False, True, 0 and 1."""

    def test_a_value_and_an_uncertainty(self):
        for flag in FLAGS:
            self.assertEqual(refusal(Result, 200, flag, "ms"), f"TypeError: no number: {flag}")
            self.assertEqual(refusal(Result, flag, 0, "ms"), f"TypeError: no number: {flag}")
            self.assertEqual(kind(Result, 200, 2, "ms", flag), "TypeError")
        zero = Result(200, 0, "ms")
        self.assertEqual(zero.report(), "(200 ± 0) ms, u = 0, p = 0.95")      # a measured zero stays
        self.assertEqual(zero.interval(), (200, 200))
        self.assertEqual(zero.prob_below(200), 1.0)
        self.assertEqual(Result(1, 0, "ms").report(), "(1 ± 0) ms, u = 0, p = 0.95")
        self.assertEqual(Result(0, 1, "ms").interval(Fraction(1, 2)), (-Result(0, 1)._k(0.5), Result(0, 1)._k(0.5)))

    def test_a_level_and_a_limit(self):
        r = Result(200, 2, "ms")
        for flag in FLAGS:
            for method in (r.k, r.interval, r.expanded, r.report, r.prob_below):
                self.assertEqual(kind(method, flag), "TypeError", (method.__name__, flag))
            self.assertEqual(kind(stats.exact_q, flag), "TypeError")
            self.assertEqual(kind(rect_interval, 5, 1, flag), "TypeError")
            self.assertEqual(kind(rect_interval, flag, 1), "TypeError")
        self.assertEqual(r.interval(0), (200, 200))
        self.assertEqual(stats.exact_q(0), 0)
        self.assertEqual(stats.exact_q(1), 1)
        self.assertEqual(r.prob_below(0), 0.0)

    def test_a_component_of_a_budget(self):
        for flag in FLAGS:
            self.assertEqual(kind(Budget().add, "probe", "B", flag), "TypeError")
            self.assertEqual(kind(Budget().add, "probe", "B", 0.3, flag), "TypeError")
            self.assertEqual(kind(combine, flag, 3), "TypeError")
            self.assertEqual(kind(welch_satterthwaite, [flag], [5]), "TypeError")
            for width in (u_rect, u_tri, u_resolution):
                self.assertEqual(kind(width, flag), "TypeError")
            self.assertEqual(kind(type_a, [flag, 1, 2]), "TypeError")
            self.assertEqual(kind(fmt, flag, 0), "TypeError")
            self.assertEqual(kind(fmt, 5, flag), "TypeError")
            self.assertEqual(kind(Q, flag, "s"), "TypeError")
            self.assertEqual(kind(lpu, sum, [flag], [0.1]), "TypeError")
            self.assertEqual(kind(lpu, sum, [1.0], [flag]), "TypeError")
            self.assertEqual(kind(summarize, [flag] * 30, 0.9), "TypeError")
        self.assertEqual(Budget().add("probe", "B", 0).u_c(), 0.0)
        self.assertEqual(Budget().add("probe", "B", 1).u_c(), 1.0)
        self.assertEqual(combine(0, 3), 3.0)
        self.assertEqual(u_rect(0), 0.0)
        self.assertEqual(type_a([1, 0, 1]).value, float(Fraction(2, 3)))

    def test_a_count(self):
        quiet = -math.log(0.05) / 30                                          # no event in 30 s, at 95 %
        self.assertTrue(math.isclose(stats.rate_upper(0, 30), quiet, rel_tol=1e-14))
        self.assertEqual(f"{quiet:.8f}", "0.09985774")
        lo, hi = 4.0, 5.0                                                     # one event: exp(-m) (1 + m) = 0.05
        for _ in range(60):
            m = (lo + hi) / 2
            lo, hi = (m, hi) if math.exp(-m) * (1 + m) > 0.05 else (lo, m)
        self.assertAlmostEqual(stats.rate_upper(1, 30) * 30, lo, places=12)
        for flag in FLAGS:
            self.assertEqual(refusal(stats.rate_upper, flag, 30), "ValueError: need a whole n >= 0")
            self.assertEqual(kind(stats.rate_limits, flag, 30), "ValueError")
            self.assertEqual(kind(stats.rate_upper, 3, flag), "TypeError")    # nor is a flag an exposure
            self.assertEqual(kind(stats.quiet_bound, flag), "TypeError")
            self.assertEqual(kind(stats.as_count, flag), "ValueError")
            self.assertEqual(kind(stats.wilson, flag, 10), "ValueError")
            self.assertEqual(kind(stats.wilson, 3, flag), "ValueError")
            self.assertEqual(kind(stats.quantile_ranks, flag, 0.5), "ValueError")
            self.assertEqual(kind(stats.paired_exact, flag, 3), "ValueError")
            self.assertEqual(kind(stats.exchange_exact, [flag, 2]), "TypeError")
            self.assertEqual(kind(stats.upward, 1.5, flag), "ValueError")
            self.assertEqual(kind(stats.upward, flag, 2), "TypeError")
        self.assertEqual((stats.as_count(0), stats.as_count(1)), (0, 1))
        self.assertEqual(stats.as_count(3.0), 3)
        self.assertEqual(stats.paired_exact(0, 3), Fraction(1, 8))

    def test_a_service_verdict(self):
        # 94 of 100 within the limit, and the p95 needs 95: with no unknown request the objective is
        # missed, with one it is open. A flag is neither statement.
        self.assertEqual(bounds.quantile_verdict(100, 0.95, 94, 0), "missed")
        self.assertEqual(bounds.quantile_verdict(100, 0.95, 94, 1), "unresolved")
        self.assertEqual(bounds.quantile_verdict(100, 0.95, 95, 0), "met")
        for flag in FLAGS:                                                    # Version 67: missed, unresolved
            self.assertEqual(refusal(bounds.quantile_verdict, 100, 0.95, 94, flag),
                             "ValueError: need a whole unknown >= 0")
            self.assertEqual(kind(bounds.quantile_verdict, 100, 0.95, flag, 1), "ValueError")
            self.assertEqual(kind(bounds.quantile_verdict, flag, 0.95, 0, 0), "ValueError")
            self.assertEqual(kind(bounds.share_bounds, 100, 96, flag), "ValueError")
            self.assertEqual(kind(bounds.rank_bounds, 100, flag, 0.95), "ValueError")
            self.assertEqual(kind(bounds.verdict, Fraction(1, 50), Fraction(1, 20), flag), "TypeError")

    def test_a_reading(self):
        for flag in FLAGS:
            self.assertEqual(kind(stats.mean, [flag, 1]), "TypeError")
            self.assertEqual(kind(stats.quantile, [flag, 2], 0.5), "TypeError")
            self.assertEqual(kind(stats.weighted_quantile, [(flag, 1), (2, 1)], 0.5), "ValueError")
            self.assertEqual(kind(stats.weighted_quantile, [(1, flag), (2, 1)], 0.5), "TypeError")
            self.assertEqual(kind(stats.known, [flag]), "ValueError")
            self.assertEqual(kind(stats.linfit, [flag, 2, 3], [1, 2, 3]), "TypeError")
            self.assertEqual(kind(mix.observed, [flag]), "ValueError")
            self.assertEqual(kind(mix.observed, [(">", flag)]), "ValueError")
        self.assertEqual(stats.mean([0, 1]), 0.5)
        self.assertEqual(stats.quantile([0, 1], 0.5), 0.5)
        self.assertEqual(stats.weighted_quantile([(0, 1), (1, 1)], 0.5), 0)

    def test_in_memory_as_in_the_record(self):
        # The validator of the record already refused a flag in these fields; the constructors now agree.
        with open(record.__file__.replace("swmetro/record.py", "data/measurement_record.json"),
                  encoding="utf-8") as f:
            rec = record.read(f.read())
        self.assertEqual(record.invalid(rec), [])
        for flag in FLAGS:
            for section, key, text in (("result", "u", "a finite number >= 0"),
                                       ("result", "value", "a finite number"),
                                       ("result", "dof", "a finite number > 0")):
                rec2 = {k: dict(v) if isinstance(v, dict) else v for k, v in rec.items()}
                rec2[section][key] = flag
                self.assertEqual(record.invalid(rec2), [f"{section}.{key} is not {text}"])
            rec2 = {k: dict(v) if isinstance(v, dict) else v for k, v in rec.items()}
            rec2["components"] = [dict(c) for c in rec["components"]]
            rec2["components"][0]["u"] = flag
            self.assertEqual(record.invalid(rec2), ["components[0].u is not a finite number >= 0"])
            rec2 = {k: dict(v) if isinstance(v, dict) else v for k, v in rec.items()}
            rec2["capture"]["records_lost"] = flag
            self.assertEqual(record.invalid(rec2), ["capture.records_lost is not a count"])


class AKnownInfinityAndAnOpenDuration(unittest.TestCase):
    """Section 3 of the review, first part: the table of four cases."""

    CASES = ((None, INF, "+inf ms if defined"), (INF, None, "-inf ms if defined"),
             (INF, INF, "not defined"), (None, None, "no limit"))

    @staticmethod
    def changes(before, after, steps=(0, 1, 50, 10 ** 6, INF)):
        """after - before of one request for every pair of durations that the record allows, among a few: a
        number, an infinity, or None where an infinity is taken from an infinity."""
        xs = steps if before is None else (before,)
        ys = steps if after is None else (after,)
        return [less(y, x) for x in xs for y in ys]

    def test_what_the_record_permits(self):
        got = {(b, a): extent(self.changes(b, a)) for b, a, _ in self.CASES if (b, a) != (INF, INF)}
        self.assertEqual(got[None, INF], (INF, INF, True))                    # +inf for every finite completion
        self.assertEqual(got[INF, None], (-INF, -INF, True))
        self.assertEqual(got[None, None], (-INF, INF, True))                  # anything, or nothing
        self.assertEqual(self.changes(INF, INF), [None])                      # nothing is defined

    def test_what_is_printed(self):
        for before, after, want in self.CASES:
            for args, name in (((), "mean"), ((mix.p95, "p95", "distributions"), "p95")):
                lines = printed(mix.components, {"web": [before]}, {"web": [after]}, *args).splitlines()
                self.assertTrue(lines[0].startswith(f"{name}, "), lines[0])
                self.assertEqual(lines[0].split(": ")[1], want, (before, after, name))  # Version 67: no limit
                self.assertEqual(lines[-1], "    within " + want.replace(" ms", "") + ", mix +0")
                self.assertTrue(all(len(line) <= 50 for line in lines), lines)

    def test_the_spans(self):
        void = mix.apart((INF, INF), (0, INF))
        self.assertEqual((tuple(void), type(void)), ((INF, INF), mix.Void))
        void = mix.apart((0, INF), (INF, INF))
        self.assertEqual((tuple(void), type(void)), ((-INF, -INF), mix.Void))
        both = mix.apart((INF, INF), (INF, INF))
        self.assertTrue(both[0] != both[0] and both[1] != both[1])
        self.assertIs(type(both), tuple)
        self.assertEqual(mix.apart((0, INF), (0, INF)), (-INF, INF))
        self.assertIs(type(mix.apart((0, INF), (0, INF))), mix.Void)          # two infinities can meet
        self.assertIs(type(mix.apart((INF, INF), (50, 60))), tuple)           # +inf, and always defined
        self.assertEqual(mix.apart((INF, INF), (50, 60)), (INF, INF))
        self.assertEqual(mix.spanned(mix.apart((INF, INF), (50, 60)), "+", " ms"), "+inf ms")
        self.assertEqual(mix.spanned(mix.Void((0, 0)), "+"), "+0 if defined")
        self.assertEqual(mix.spanned(mix.Void((Fraction(-90), INF)), "+", " ms"), "-90 ms or more if defined")
        self.assertEqual(mix.spanned((-INF, INF)), "no limit")

    def test_a_sum_of_spans(self):
        # +inf for certain, and a term that can reach -inf: +inf wherever the sum is defined.
        got = mix.blend([(1, (INF, INF)), (-1, (5, INF))])
        self.assertEqual((tuple(got), type(got)), ((INF, INF), mix.Void))
        got = mix.blend([(1, (INF, INF)), (-1, (INF, INF))])
        self.assertTrue(got[0] != got[0] and got[1] != got[1])
        got = mix.blend([(Fraction(1, 2), mix.Void((0, 0))), (Fraction(-1, 2), (3, 3))])
        self.assertEqual((tuple(got), type(got)), ((Fraction(-3, 2), Fraction(-3, 2)), mix.Void))
        self.assertIs(type(mix.blend([(2, (1, 2)), (-1, (0, INF))])), tuple)
        self.assertEqual(mix.blend([(2, (1, 2)), (-1, (0, INF))]), (-INF, 4))

    def test_the_change_that_was_right_stays(self):
        lines = printed(mix.components, {"web": [100] * 20}, {"web": [100] * 19 + [None]}).splitlines()
        self.assertEqual(lines[0], "mean, 100 -> 95 ms or more: -5 ms or more")
        self.assertEqual(lines[-1], "    within -5 or more, mix +0")
        lines = printed(mix.components, {"web": [INF]}, {"web": [100]}).splitlines()
        self.assertEqual(lines[0], "mean, inf -> 100 ms: -inf ms")            # always defined: no mark


class TheMixContributionOfAPercentile(unittest.TestCase):
    """Section 3 of the review, second part: one open duration and the two mixes."""

    BEFORE = {"web": [100], "cache": [10, 10]}
    AFTER = {"web": [None], "cache": [10]}

    def parts(self, before, after, x, u=0):
        """The six parts of the change of the p95, by counting, with x for the open duration of the second
        period and u for one of the first: within and mix in the two orders, and their averages."""
        start = counted(at_mix(before, before, u))
        end = counted(at_mix(after, after, x))
        a = counted(at_mix(after, before, x))                                 # the second period at the first mix
        b = counted(at_mix(before, after, u))
        first = less(a, start), less(end, a)
        second = less(end, b), less(b, start)
        # Half the sum of the two orders, where both parts are defined: an infinity added to the same
        # infinity is that infinity.
        average = tuple(None if None in (p, q) else Fraction(1, 2) * (p + q) for p, q in zip(first, second))
        return first + second + average

    def test_the_example_of_the_review_by_hand(self):
        # x for the open duration. Each class is above 5 % in both mixes, so the p95 is the larger class
        # value: 100 before at either mix, max(10, x) after at either mix.
        for x in (0, 5, 10, 11, 99, 100, 101, 10 ** 9):
            self.assertEqual(counted(at_mix(self.AFTER, self.AFTER, x)), max(10, x))
            self.assertEqual(counted(at_mix(self.AFTER, self.BEFORE, x)), max(10, x))
            got = self.parts(self.BEFORE, self.AFTER, x)
            self.assertEqual(got, (max(10, x) - 100, 0) * 3)
        self.assertEqual(self.parts(self.BEFORE, self.AFTER, INF), (INF, None, INF, 0, INF, None))

    def test_the_example_of_the_review_as_printed(self):
        text = printed(mix.components, self.BEFORE, self.AFTER, mix.p95, "p95", "distributions")
        self.assertEqual(text, "p95, 100 -> 10 ms or more: -90 ms or more\n"
                               "  open after: 1 of 2 durations\n"
                               "  class distributions first, via 10 ms or more:\n"  # Version 67: not determined
                               "    within -90 or more, mix +0 if defined\n"
                               "  mix first, via 100 ms:\n"
                               "    within -90 or more, mix +0\n"
                               "  average of the two orders:\n"                     # Version 67: no average
                               "    within -90 or more, mix +0 if defined\n")
        self.assertTrue(all(len(line) <= 50 for line in text.splitlines()))

    def compare(self, before, after, grid_u):
        rows = mix.of_rank(mix.p95, *mix.periods(before, after))
        got = [span for _, inner, mixed in rows for span in (inner, mixed)]
        seen = [self.parts(before, after, x, u) for x in places(after) for u in grid_u]
        for i, span in enumerate(got):
            low, top, void = extent([row[i] for row in seen])
            self.assertEqual((span[0], span[1]), (low, top), (before, after, i))
            self.assertEqual(isinstance(span, mix.Void), void, (before, after, i))

    def test_one_open_duration_against_counting(self):
        rng = random.Random(6723)
        for _ in range(400):
            names = "abc"[:rng.randint(2, 3)]
            before = {c: [rng.randint(1, 9) * 10 for _ in range(rng.randint(1, 4))] for c in names}
            after = {c: [rng.randint(1, 9) * 10 for _ in range(rng.randint(1, 4))] for c in names}
            lost = rng.choice(names)
            after[lost][rng.randrange(len(after[lost]))] = None
            self.compare(before, after, [0])

    def test_one_open_duration_in_each_period(self):
        rng = random.Random(6724)
        for _ in range(150):
            names = "ab"
            before = {c: [rng.randint(1, 9) * 10 for _ in range(rng.randint(1, 3))] for c in names}
            after = {c: [rng.randint(1, 9) * 10 for _ in range(rng.randint(1, 3))] for c in names}
            for rec in (before, after):
                lost = rng.choice(names)
                rec[lost][rng.randrange(len(rec[lost]))] = None
            self.compare(before, after, places(before))

    def test_a_bound_in_the_place_of_the_open_duration(self):
        # Known to have taken 50 ms or more: max(10, x) for x >= 50 is x, and the change of the mix is 0.
        after = {"web": [(">=", 50)], "cache": [10]}
        lines = printed(mix.components, self.BEFORE, after, mix.p95, "p95", "distributions").splitlines()
        self.assertEqual(lines[0], "p95, 100 -> 50 ms or more: -50 ms or more")
        self.assertEqual(lines[2:4], ["  class distributions first, via 50 ms or more:",
                                      "    within -50 or more, mix +0 if defined"])

    def test_two_open_durations_are_not_bounded(self):
        after = {"web": [None, None], "cache": [10, 10]}
        before = {"web": [100], "cache": [10, 10]}
        text = printed(mix.components, before, after, mix.p95, "p95", "distributions")
        self.assertIn("  class distributions first: not bounded,\n    open after: more than one\n", text)
        self.assertIn("  mix first, via 100 ms:\n", text)
        self.assertNotIn("average", text)
        rows = mix.of_rank(mix.p95, *mix.periods(before, after))
        self.assertIsNone(rows[0][2])
        self.assertIsNone(rows[2][2])

    def test_a_missing_class_and_an_infinite_way_are_named_as_before(self):
        before = {"common": [80.0] * 92 + [100.0] * 6 + [180.0]}
        after = {"common": [70.0] * 92 + [90.0] * 6 + [160.0], "bulk": [500.0] * 30}
        text = printed(mix.components, before, after, mix.p95, "p95", "distributions")
        self.assertIn("  mix first: not determined,\n    no bulk requests before\n", text)
        text = printed(mix.components, {"a": [10], "b": [20, 20]}, {"a": [INF, 10], "b": [20]},
                       mix.p95, "p95", "distributions")
        # Until Version 68 this order read "not determined, a duration after is infinite", and this test
        # expected it. The review of Version 68 asked for the two parts where one of them survives: the way
        # is infinite, the change within the classes is +inf, and the change of the mix takes an infinity
        # from an infinity.
        self.assertIn("  class distributions first, via inf ms:\n    within +inf, mix not defined\n", text)

    def test_the_docstrings_say_what_is_assumed(self):
        text = " ".join(mix.components.__doc__.split())
        self.assertIn("taken to be a percentile by the inverse empirical CDF", text)
        self.assertIn("reports an order as not bounded with more", text)
        self.assertIn("is not bounded here: None", " ".join(mix.held.__doc__.split()))


def tail_of_four(t, digits=700):
    """Student's upper tail for four degrees of freedom from its closed form, in decimals: with
    s = t / sqrt(t^2 + 4) it is (1 - s (1 + (1 - s^2) / 2)) / 2, and 1 - s is formed without cancellation
    as 4 / ((t^2 + 4) (1 + s))."""
    with localcontext() as ctx:
        ctx.prec = digits
        t = D(Fraction(t).numerator) / D(Fraction(t).denominator)
        r = t * t + 4
        s = t / r.sqrt()
        one_less_s = 4 / (r * (1 + s))                                        # 1 - s
        # 1 - s (1 + (1 - s^2) / 2) = (1 - s) - s (1 - s)(1 + s) / 2
        return (one_less_s - s * one_less_s * (1 + s) / 2) / 2


def quantile_of_two(p, digits=400):
    """Student's upper quantile for two degrees of freedom: (1 - 2 p) / sqrt(2 p (1 - p))."""
    with localcontext() as ctx:
        ctx.prec = digits
        p = D(Fraction(p).numerator) / D(Fraction(p).denominator)
        return (1 - 2 * p) / (2 * p * (1 - p)).sqrt()


def quantile_of_one(p, digits=400):
    """The Cauchy upper quantile cot(pi p) for a small p, from the series 1/x - x/3 - x^3/45."""
    with localcontext() as ctx:
        ctx.prec = digits
        x = PI * D(Fraction(p).numerator) / D(Fraction(p).denominator)
        return 1 / x - x / 3 - x ** 3 / 45


def off(got, want):
    """The relative error of a float against a decimal."""
    with localcontext() as ctx:
        ctx.prec = 60
        return float(abs((D(got) - want) / want))


class TheRangeOfTheStudentQuantile(unittest.TestCase):
    """Section 4 of the review, items 1 and 2."""

    def test_the_quantile_of_the_review_is_no_longer_refused(self):
        p = 3.1830988618379074e-151
        want = quantile_of_one(p)
        self.assertLess(want, D(CEILING))                                     # 9.99999999999999786e149
        self.assertGreater(want, D("9.999999999999997e149"))
        got = dist.t_isf(p, 1)                                                # Version 67: quantile too large
        self.assertLessEqual(got, CEILING)
        self.assertLess(off(got, want), 1e-12)

    def last_accepted(self, dof, low, high):
        """The least float tail between low and high whose quantile is returned, by bisection on the
        floats."""
        self.assertIsNotNone(refusal(dist.t_isf, low, dof))
        self.assertIsNone(refusal(dist.t_isf, high, dof))
        while math.nextafter(low, high) < high:
            mid = low + (high - low) / 2
            if refusal(dist.t_isf, mid, dof):
                low = mid
            else:
                high = mid
        return low, high

    def test_the_last_quantile_that_is_accepted(self):
        for dof, exact, low, high in ((1, quantile_of_one, 3.1e-151, 3.3e-151),
                                      (2, quantile_of_two, 4.9e-301, 5.1e-301)):
            refused, accepted = self.last_accepted(dof, low, high)
            self.assertEqual(refusal(dist.t_isf, refused, dof), "ValueError: quantile too large")
            # What is refused lies beyond the ceiling, and what is accepted within 1e-12 of its quantile,
            # which is at most 1e150 as it is returned.
            self.assertGreater(exact(refused), D(CEILING))
            self.assertLess(exact(refused), D(CEILING) * (1 + D("3e-12")))
            p = accepted
            for _ in range(200):
                got = dist.t_isf(p, dof)
                self.assertLessEqual(got, CEILING)
                self.assertLess(off(got, exact(p)), 2e-12, (dof, p))
                p = math.nextafter(p, 1.0)
            # No quantile at or below the ceiling is refused: the tail of the ceiling itself is accepted.
            with localcontext() as ctx:
                ctx.prec = 400
                c = D(CEILING)
                at = {1: 1 / (PI * c) - 1 / (3 * PI * c ** 3),                # atan(1 / c) / pi
                      2: (1 - c / (c * c + 2).sqrt()) / 2}[dof]
                at = float(at)
            for p in (at, math.nextafter(at, 1.0), math.nextafter(at, 0.0)):
                self.assertIsNone(refusal(dist.t_isf, p, dof), (dof, p))

    def test_what_is_far_beyond_the_ceiling_is_refused(self):
        for p, dof in ((1e-152, 1), (1e-300, 0.5), (3e-151, 1), (1e-302, 2)):
            self.assertEqual(refusal(dist.t_isf, p, dof), "ValueError: quantile too large")
        self.assertEqual(refusal(Result(0, 1, dof=0.004).k, 0.95), "ValueError: quantile too large")

    def test_the_least_tail(self):
        want = "ValueError: need 2**-1022 <= tail <= 0.5"
        for tail in (5e-324, 1e-320, math.nextafter(LEAST, 0.0), Fraction(1, 10 ** 400), D("1e-400"), 0, 0.0,
                     -0.1, 0.5000001, 1):
            self.assertEqual(refusal(dist.t_isf, tail, 4), want, repr(tail))  # Version 67: 9.49e80 for 1e-400
        # Until Version 68 True and False stood in the list above and were refused by their range, as 1 and
        # 0. Since Version 69 the module takes no truth value for a number at all.
        for flag in (True, False):
            self.assertEqual(refusal(dist.t_isf, flag, 4), f"TypeError: no number: {flag}")
        for tail in (LEAST, 1e-307, 1e-300, 1e-200):
            got = dist.t_isf(tail, 4)
            with localcontext() as ctx:
                ctx.prec = 60
                self.assertLess(abs(tail_of_four(got) / D(tail) - 1), D("4e-12"), tail)  # a quarter of it in t
        self.assertEqual(refusal(dist.t_ppf, 5e-324, 4), want)
        self.assertEqual(dist.t_ppf(LEAST, 4), -dist.t_isf(LEAST, 4))
        self.assertEqual(refusal(Result(0, 1, dof=4).k, 1 - 2.0 ** -53), None)
        self.assertEqual(refusal(Result(0, 1, dof=4).k, Fraction(10 ** 400 - 1, 10 ** 400)),
                         "ValueError: p is too near 1")

    def test_a_tail_of_another_type(self):
        # tan(pi / 10) is sqrt(1 - 2 / sqrt(5)): the quantile of one degree of freedom at a tail of 0.4.
        with localcontext() as ctx:
            ctx.prec = 50
            want = (1 - 2 / D(5).sqrt()).sqrt()
        for tail in (0.4, D("0.4"), Fraction(2, 5)):
            self.assertLess(off(dist.t_isf(tail, 1), want), 1e-13, repr(tail))  # Version 67: TypeError
        # Next to one half the quantile is pi times the distance, which is kept.
        gap = Fraction(1, 10 ** 20)
        for tail in (Fraction(1, 2) - gap, D("0.49999999999999999999")):
            got = dist.t_isf(tail, 1)                                         # Version 67: ValueError
            self.assertLess(off(got, PI * D(10) ** -20), 1e-13)
        for tail in (D("0.01"), Fraction(1, 100)):
            self.assertLess(off(dist.t_isf(tail, 2), quantile_of_two(tail)), 1e-12)
        self.assertEqual(kind(dist.t_isf, "0.4", 1), "TypeError")
        # A distance from one half that no float holds is refused, and not read as a quantile of zero.
        self.assertEqual(refusal(dist.t_isf, Fraction(1, 2) - Fraction(1, 10 ** 400), 1),
                         "ValueError: need 0 < gap < 0.5")

    def test_the_quantile_function_keeps_a_decimal_too(self):
        p = D("0.5000000000000000000000000000001")                            # 31 digits: 1 - p rounds to 0.5
        got = dist.t_ppf(p, 1)
        self.assertLess(off(got, PI * D(10) ** -31), 1e-13)
        self.assertEqual(dist.t_ppf(0.95, 7), dist.t_isf(1 - 0.95, 7))        # a float as before
        self.assertEqual(dist.t_ppf(0.5, 7), 0.0)
        got = dist.t_ppf(D("0.999999999999999999999999999999"), 1)            # a tail of 1e-30
        self.assertLess(off(got, quantile_of_one(D("1e-30"))), 1e-12)

    def test_the_docstring_states_the_range(self):
        text = " ".join(dist.t_isf.__doc__.split())
        self.assertIn("a tail from the least normal float, 2.2e-308, to 0.5", text)
        self.assertIn("A smaller tail is refused", text)
        self.assertIn("taken as 1e150, see _capped()", text)                  # until Version 68 said here in full
        self.assertIn("exceeds `tail` by 1e-12 of it or less", " ".join(dist._capped.__doc__.split()))


class TheLargestFloatAsATemperature(unittest.TestCase):
    """Section 4 of the review, item 3."""

    TOP = sys.float_info.max

    @staticmethod
    def optimum(z):
        """z / ln(4 / 3): with four cases of logit z that are positive and three that are not, the slope of
        the loss is zero where exp(z / T) is 4 / 3."""
        with localcontext() as ctx:
            ctx.prec = 80
            return D(z) / (D(4) / D(3)).ln()

    @staticmethod
    def data(z):
        return [(z, 1)] * 4 + [(z, 0)] * 3

    def test_the_example_of_the_review(self):
        z = 5.171640866695299e307
        step = D(math.ulp(self.TOP))
        with localcontext() as ctx:
            ctx.prec = 80
            beyond = (self.optimum(z) - D(self.TOP)) / step
        self.assertTrue(0 < beyond < D("0.5"), beyond)                        # 0.18 of a step: it rounds to TOP
        self.assertEqual(f"{beyond:.5f}", "0.18371")
        got = cal.fit_temperature(self.data(z), self.TOP / 2, self.TOP)       # Version 67: above hi
        self.assertEqual(got, self.TOP)

    def test_one_step_of_the_logit_to_either_side(self):
        z = 5.171640866695299e307
        up, down = math.nextafter(z, INF), math.nextafter(z, 0.0)
        step = D(math.ulp(self.TOP))
        with localcontext() as ctx:
            ctx.prec = 80
            beyond = (self.optimum(up) - D(self.TOP)) / step
            within = (self.optimum(down) - D(self.TOP)) / step
        self.assertGreater(beyond, D("0.5"))                                  # past the middle to 2 ** 1024
        self.assertEqual(refusal(cal.fit_temperature, self.data(up), self.TOP / 2, self.TOP),
                         "ValueError: the minimum is above hi")
        self.assertLess(within, 0)
        got = cal.fit_temperature(self.data(down), self.TOP / 2, self.TOP)
        self.assertEqual(got, float(self.optimum(down)))                      # a float below the largest
        self.assertLess(got, self.TOP)

    def test_an_upper_end_below_the_largest_float(self):
        # The same rule at any hi: the minimum rounds to hi or it is beyond it.
        for z in (1.0, 2e307, 3.3):
            with localcontext() as ctx:
                ctx.prec = 80
                t = self.optimum(z)
            near = float(t)
            self.assertEqual(cal.fit_temperature(self.data(z), near / 2, near), near)
            below = math.nextafter(near, 0.0)
            self.assertEqual(refusal(cal.fit_temperature, self.data(z), below / 2, below),
                             "ValueError: the minimum is above hi")

    def test_the_docstring_states_the_rule(self):
        text = " ".join(cal.fit_temperature.__doc__.split())
        self.assertIn("the largest takes all below the middle between it and 2 ** 1024, and lean() refuses a tie", text)


if __name__ == "__main__":
    unittest.main()
