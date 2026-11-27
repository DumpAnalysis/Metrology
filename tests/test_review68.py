"""Regression tests added after the review of Version 68.

The review found that the two samplers of the propagation module checked their arguments as floats and then
drew with the arguments as they came, so that a thousand normal draws about 1e9 with a standard deviation of 1
were one number in the 32 bits of an array, and a thousand rectangular draws between -100 and 100 in eight
bits all lay below -100; that a truth value passed for a standard deviation and for a correlation coefficient,
and a negative number for a standard deviation; that type_a() stopped at two decimal indications, and mean()
at a float beside a decimal; that a Fraction can hold whole numbers of a fixed width as its two parts, which
native() and exact_q() let pass, so that such readings again had the mean -28; that the rank helpers looked
at two of their values and let a flag take a place among the others; that the three Poisson bounds checked a
converted exposure and divided by the one that came, so that a bound was rounded in sixteen bits and printed
below its value, and stopped at a decimal exposure; that held() of Listing 23.2 raised ValueError where a
duration known to be infinite leaves no difference defined; that the change of the mix of a mean was printed
as exact where an open duration can be infinite; and that the search of the Student quantile from the middle
of the distribution refused a quantile at its ceiling without the allowance that the search from the tail has.

Each test computes what it expects independently of the code under test: the draws from Python's generator
itself; means and variances from fractions, with a root taken in decimals; ranks by counting; a Poisson bound
from a logarithm in decimals; the parts of a change by putting durations in the place of the open one and
subtracting, with None where an infinity would be taken from an infinity; and Student's tail far out for very
few degrees of freedom from the leading term of the incomplete beta function, with the gamma function from
Stirling's series in decimals.
"""
import contextlib
import io
import math
import numbers
import random
import struct
import unittest
from decimal import ROUND_CEILING, Decimal, localcontext
from fractions import Fraction

from examples import ch16_bounds as bounds
from examples import ch23_mix as mix
from swmetro import Result, dist, stats, type_a
from swmetro.propagate import correlations, lpu, monte_carlo, normal, rect, summarize

try:
    import numpy as np
except ImportError:                                                       # the package itself needs no NumPy
    np = None

D = Decimal
F = Fraction
INF = math.inf
P95 = Fraction(95, 100)
CEILING = 1e150
PI = D("3.14159265358979323846264338327950288419716939937510582097494")
FLAGS = (False, True)


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


def dec(x):
    """A float, a whole number, a fraction or a decimal as a decimal, in the precision of the context."""
    x = Fraction(x)
    return D(x.numerator) / D(x.denominator)


def narrow(code):
    """A float of fewer bits whose arithmetic rounds to those bits, as numpy.float32 and numpy.float16 do,
    for the interpreters without NumPy: `code` is "f" for 32 bits and "e" for 16. It is registered as a Real,
    as the floats of NumPy are."""

    def cut(x):
        return struct.unpack(code, struct.pack(code, float(x)))[0]

    class Narrow:
        def __init__(self, x):
            self.x = cut(x)

        def __float__(self):
            return self.x

        def as_integer_ratio(self):
            return self.x.as_integer_ratio()

        def __add__(self, other):
            return Narrow(self.x + float(other))

        __radd__ = __add__

        def __sub__(self, other):
            return Narrow(self.x - float(other))

        def __rsub__(self, other):
            return Narrow(float(other) - self.x)

        def __mul__(self, other):
            return Narrow(self.x * float(other))

        __rmul__ = __mul__

        def __truediv__(self, other):
            return Narrow(self.x / float(other))

        def __rtruediv__(self, other):
            return Narrow(float(other) / self.x)

        def __neg__(self):
            return Narrow(-self.x)

        def __abs__(self):
            return Narrow(abs(self.x))

        def __eq__(self, other):
            return self.x == other

        def __lt__(self, other):
            return self.x < other

        def __le__(self, other):
            return self.x <= other

        def __gt__(self, other):
            return self.x > other

        def __ge__(self, other):
            return self.x >= other

        def __hash__(self):
            return hash(self.x)

        def __repr__(self):
            return f"narrow({code!r})({self.x!r})"

    numbers.Real.register(Narrow)
    return Narrow


F32, F16 = narrow("f"), narrow("e")


class Wrap8:
    """A whole number of eight bits that wraps around, as numpy.int8 does, for the interpreters without NumPy.
    With a float it gives a float, as NumPy's does. It is registered as an Integral, and its numerator is
    itself."""

    def __init__(self, n):
        self.n = (int(n) + 128) % 256 - 128

    numerator = property(lambda self: self)
    denominator = 1

    def __int__(self):
        return self.n

    __index__ = __int__

    def __float__(self):
        return float(self.n)

    def _with(self, other, op):
        if isinstance(other, float):
            return op(float(self.n), other)
        return Wrap8(op(self.n, int(other)))

    def __add__(self, other):
        return self._with(other, lambda a, b: a + b)

    __radd__ = __add__

    def __sub__(self, other):
        return self._with(other, lambda a, b: a - b)

    def __rsub__(self, other):
        return self._with(other, lambda a, b: b - a)

    def __mul__(self, other):
        return self._with(other, lambda a, b: a * b)

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


def part(n, d=1):
    """A Fraction that holds whole numbers of eight bits as its two parts, as Fraction(numpy.int8(n)) does.
    The parts are set directly, so that the test does not depend on how an interpreter builds a Fraction."""
    f = Fraction(n, d)
    f._numerator, f._denominator = Wrap8(f.numerator), Wrap8(f.denominator)
    return f


def root(f, digits=60):
    """The square root of a fraction as a float: taken in decimals and rounded once."""
    with localcontext() as ctx:
        ctx.prec = digits
        return float(dec(f).sqrt())


def spread(xs):
    """The experimental variance of GUM 4.2.2 from fractions: the squared distances from the mean, summed,
    over n - 1."""
    fs = [Fraction(x) for x in xs]
    m = sum(fs) / len(fs)
    return sum((f - m) ** 2 for f in fs) / (len(fs) - 1)


def draws(n, seed, method, *args):
    """n draws of Python's own generator, in the order in which monte_carlo() asks for them, sorted."""
    rng = random.Random(seed)
    return sorted(getattr(rng, method)(*args) for _ in range(n))


class DrawsInSixtyFourBits(unittest.TestCase):
    """Section 1 of the review: the two samplers."""

    def test_a_normal_sample_varies_about_a_large_center(self):
        want = draws(1000, 1, "gauss", 1e9, 1.0)
        self.assertEqual(len(set(want)), 1000)
        sd = root(spread(want))
        self.assertEqual(f"{sd:.10f}", "1.0036847395")                        # the figure of the review
        self.assertEqual(float(F32(1e9) + 1.0), 1e9)                          # in 32 bits the step is 64
        for mu, sigma in ((1e9, 1), (F32(1e9), F32(1)), (10 ** 9, F(1)), (D("1e9"), D("1.0"))):
            ys = monte_carlo(lambda q: q[0], [normal(mu, sigma)], n=1000, seed=1)
            self.assertEqual(ys, want, repr(mu))                              # Version 68: 1000 times 1e9
            self.assertTrue(all(type(y) is float for y in ys))
            self.assertEqual(summarize(ys)[1], sd)

    def test_a_rectangular_sample_stays_between_its_limits(self):
        want = draws(1000, 1, "uniform", -100.0, 100.0)
        self.assertEqual((f"{want[0]:.3f}", f"{want[-1]:.3f}"), ("-99.831", "99.632"))
        self.assertEqual(int(Wrap8(100) - Wrap8(-100)), -56)                  # the width in eight bits
        for lo, hi in ((-100, 100), (Wrap8(-100), Wrap8(100)), (D("-100"), F(100)), (F32(-100), 100.0)):
            ys = monte_carlo(lambda q: q[0], [rect(lo, hi)], n=1000, seed=1)
            self.assertEqual(ys, want, repr(lo))                              # Version 68: -155.897 to -100.047
            self.assertTrue(all(-100 <= y <= 100 for y in ys))

    def test_no_draw_outside_limits_that_no_float_holds(self):
        # 0.3 as a float lies below three tenths and 0.8 above eight tenths: the limits are taken inward.
        self.assertLess(Fraction(0.3), Fraction(3, 10))
        self.assertGreater(Fraction(0.8), Fraction(8, 10))
        rng = random.Random(68)
        for lo, hi in ((D("0.3"), D("0.8")), (F(3, 10), F(8, 10)), (F(1, 3), F(2, 3)), (D("-0.8"), D("-0.3")),
                       (F(-1, 3), D("1e-30")), (10 ** 30 + 1, 10 ** 30 + 10 ** 20)):
            draw = rect(lo, hi)
            ys = [draw(rng) for _ in range(3000)]
            self.assertTrue(all(Fraction(lo) <= Fraction(y) <= Fraction(hi) for y in ys), (lo, hi))
            mid = (Fraction(lo) + Fraction(hi)) / 2
            self.assertTrue(any(y < mid for y in ys) and any(y > mid for y in ys))

    def test_no_spread_is_the_center_itself(self):
        rng = random.Random(1)
        for center in (200, 10 ** 30 + 1, F(1, 3), D("0.1"), 0.1):
            self.assertIs(normal(center, 0)(rng), center)
            self.assertIs(rect(center, center)(rng), center)
        got = normal(Wrap8(5), Wrap8(0))(rng)
        self.assertEqual((got, type(got)), (5, int))
        got = rect(F32(0.5), F32(0.5))(rng)
        self.assertEqual((got, type(got)), (0.5, float))

    def test_what_a_sampler_refuses(self):
        for flag in FLAGS:
            self.assertEqual(kind(normal, 200, flag), "TypeError")             # Version 68: 200 at every draw
            self.assertEqual(kind(normal, flag, 1), "TypeError")
            self.assertEqual(kind(rect, flag, 5), "TypeError")
            self.assertEqual(kind(rect, -5, flag), "TypeError")
            self.assertEqual(kind(monte_carlo, sum, [normal(0, 1)], flag), "ValueError")
        for sigma in (-2, -1e-300, INF, math.nan, D("-0.1")):
            self.assertEqual(refusal(normal, 10, sigma), "ValueError: need a finite center and a spread >= 0")
        for mu in (INF, -INF, math.nan):
            self.assertEqual(kind(normal, mu, 1), "ValueError")
            self.assertEqual(kind(normal, mu, 0), "ValueError")
        for lo, hi in ((5, 1), (1, math.nan), (math.nan, 1), (-INF, 1), (1, INF), (D("0.2"), 0.1)):
            self.assertEqual(refusal(rect, lo, hi), "ValueError: need finite lo <= hi", (lo, hi))
        self.assertEqual(kind(normal, "10", 1), "TypeError")
        self.assertEqual(kind(rect, None, 1), "TypeError")
        # A decimal that is no number cannot be compared at all: the comparison itself stops, with an error
        # of the decimal module. The refusal is the same as for a float that is no number.
        nan = D("NaN")
        for mu, sigma in ((nan, 1), (1, nan), (D("Infinity"), 1), (1, D("Infinity"))):
            self.assertEqual(refusal(normal, mu, sigma), "ValueError: need a finite center and a spread >= 0")
        for lo, hi in ((nan, 1), (1, nan), (D("-Infinity"), 1)):
            self.assertEqual(refusal(rect, lo, hi), "ValueError: need finite lo <= hi")
        # A spread that a float of the size of the center cannot show is refused as before.
        self.assertEqual(kind(normal, 1e18, 1), "ValueError")
        self.assertEqual(kind(rect, 1e18, 1e18 + 1024), "ValueError")

    def test_the_docstrings_state_the_precision(self):
        self.assertIn("The draws are floats of 64 bits", " ".join(normal.__doc__.split()))
        self.assertIn("No draw lies outside the two limits", " ".join(rect.__doc__.split()))


class ACoefficientIsANumber(unittest.TestCase):
    """Section 1, second part: a flag as a correlation."""

    def test_the_difference_of_two_inputs(self):
        def u(r):
            return lpu(lambda x: x[0] - x[1], [100, 99], [1, 1], {(0, 1): r})[1]

        # u^2 = 1 + 1 - 2 r for the difference of two inputs with u = 1.
        for r, want in ((0, 2), (1, 0), (-1, 4), (0.5, 1), (F(1, 2), 1), (D("0.25"), F(3, 2))):
            self.assertEqual(u(r), root(F(want)), r)
        for flag in FLAGS:                                                    # Version 68: 0.0 for True
            self.assertEqual(refusal(u, flag), f"TypeError: no number: {flag}")
            self.assertEqual(kind(correlations, {(0, 1): flag}, 2), "TypeError")
            self.assertEqual(kind(correlations, {}, flag), "ValueError")
        self.assertEqual(kind(u, "0.5"), "TypeError")
        self.assertEqual(kind(u, None), "TypeError")
        self.assertEqual(u(part(1, 2)), 1.0)                                  # 1/2 with parts of eight bits
        self.assertEqual(correlations({(0, 1): 1}, 2), [[1, 1], [1, 1]])
        self.assertEqual(correlations({(0, 1): 0}, 2), [[1, 0], [0, 1]])


class IndicationsOfAnyType(unittest.TestCase):
    """Section 2 of the review: decimals, mixed types, and fractions with parts of a fixed width."""

    def test_two_decimal_indications(self):
        # GUM 4.2: the mean is 101, the variance (1 + 1) / (2 - 1) = 2, that of the mean 2 / 2 = 1.
        want = type_a([100, 102], "ms")
        self.assertEqual((want.value, want.u, want.dof), (101, 1.0, 1.0))
        for xs in ([D("100"), D("102")], [F(100), F(102)], [100.0, 102.0], [100.0, D("102")], [F(100), 102],
                   [D("100"), 102.0]):
            got = type_a(xs, "ms")                                            # Version 68: TypeError for decimals
            self.assertEqual((got.value, got.u, got.dof, got.unit), (101, 1.0, 1.0, "ms"), xs)
            # With one degree of freedom the factor is tan(0.475 pi) = 12.71, and U keeps two digits.
            self.assertEqual(f"{math.tan(0.475 * math.pi):.2f}", "12.71")
            self.assertEqual(got.report(), "(101 ± 13) ms, k = 12.71, p = 0.95")

    def test_a_mean_of_several_types(self):
        self.assertEqual(stats.mean([100.0, D("102")]), 101)                  # Version 68: TypeError
        self.assertEqual(stats.mean([D("0.1"), 0.5, F(1, 3), 2]), (F(1, 10) + F(1, 2) + F(1, 3) + 2) / 4)
        ys = sorted([100.0, D("102")] * 20, key=Fraction)
        m, s, lo, hi = summarize(ys, 0.9)
        self.assertEqual((m, lo, hi), (101, 100, 102))
        self.assertEqual(s, root(F(40, 39)))                                  # 40 squares of 1 over 39

    def test_against_fractions(self):
        rng = random.Random(6801)
        makers = (lambda: rng.randint(-10 ** 6, 10 ** 6), lambda: rng.uniform(-50, 50),
                  lambda: F(rng.randint(-999, 999), rng.randint(1, 97)), lambda: D(rng.randint(-99999, 99999)) / 1000,
                  lambda: 10 ** 18 + rng.randint(0, 9))
        for _ in range(300):
            kinds = rng.sample(makers, rng.randint(1, 3))
            xs = [rng.choice(kinds)() for _ in range(rng.randint(2, 9))]
            exact = sum(map(Fraction, xs)) / len(xs)
            got = stats.mean(xs)
            if isinstance(got, float):
                self.assertEqual(got, float(exact), xs)                       # rounded once
            else:
                self.assertEqual(got, exact, xs)
            self.assertEqual(stats.variance(xs), spread(xs), xs)
            if len(set(map(Fraction, xs))) > 1:
                try:
                    result = type_a(xs)
                except ValueError as error:                                   # too fine for a float of that size
                    self.assertIn("too fine", str(error))
                    continue
                self.assertEqual(result.u, root(spread(xs) / len(xs)), xs)
                self.assertEqual(result.dof, len(xs) - 1)

    def test_identical_observations_have_their_own_mean(self):
        cases = [(x, Fraction(x)) for x in (100, 100.5, 0.1, F(1, 3), D("0.1"), 10 ** 18 + 1)]
        cases += [(part(100), F(100)), (part(1, 3), F(1, 3)), (Wrap8(100), F(100)),
                  (F32(0.1), Fraction(float(F32(0.1))))]
        for x, value in cases:                                                # a reading and the number it is
            for k in (1, 2, 3, 7):
                self.assertEqual(Fraction(stats.mean([x] * k)), value, (x, k))
                self.assertEqual(Fraction(mix.mean([(x, 1)] * k)), value, (x, k))
                self.assertEqual(Fraction(mix.mean([(x, F(1, 3))] * k)), value, (x, k))
        with self.assertRaises(ValueError):                                   # one indication has no variance
            type_a([5.0])

    def test_indications_that_are_no_finite_numbers(self):
        for bad in (INF, -INF, math.nan, D("NaN"), D("Infinity")):
            self.assertEqual(refusal(type_a, [1.0, bad, 2.0], "ms"), "ValueError: no finite readings", repr(bad))
            self.assertEqual(refusal(stats.variance, [1.0, bad]), "ValueError: no finite readings", repr(bad))

    def test_an_uncertainty_that_rounds_to_zero(self):
        # Two indications one step of the floats apart, at the least float there is. Their standard deviation
        # of the mean is half of that step and rounds to 0.0. Version 68 refused the two, by the rule for a
        # spread that a float cannot show; as this version was first written, it reported them with an
        # uncertainty of zero, which no two different indications have. Found on rereading the listing.
        least = 5e-324
        self.assertEqual(stats.variance([least, 0.0]) / 2, (Fraction(least) / 2) ** 2)   # u is half a step
        self.assertEqual(refusal(type_a, [least, 0.0], "ms"), "ValueError: too fine for a float of this size")
        self.assertEqual(type_a([7.0, 7.0, 7.0], "ms").u, 0.0)                # no spread at all is a result

    def test_a_fraction_with_parts_of_eight_bits(self):
        f = part(100)
        self.assertIs(type(f.numerator), Wrap8)
        got = stats.native(f)
        self.assertEqual((got, type(got.numerator), type(got.denominator)), (100, int, int))
        clean = F(1, 3)
        self.assertIs(stats.native(clean), clean)
        # The table of the review.
        self.assertEqual(stats.mean([part(100)] * 2), 100)                    # Version 68: -28
        self.assertEqual(stats.quantile([part(100), part(101)], 0.95, exact=True), F(2019, 20))
        self.assertEqual(-((-19 * 101) // 20), 96)                            # the least k with k / 101 >= 19 / 20
        self.assertEqual(stats.ecdf_rank(part(19, 20), 101), 96)
        self.assertEqual(stats.weighted_quantile([(100, part(100)), (200, part(100))], 0.95), 200)
        self.assertEqual(stats.exact_q(part(19, 20)), F(19, 20))
        self.assertIs(type(stats.exact_q(part(19, 20)).numerator), int)
        self.assertEqual(stats.known([part(100), part(7, 2)]), [100, F(7, 2)])
        self.assertEqual(stats.upward(part(1, 3), 2), "0.34")
        self.assertEqual(Result(part(100), part(2), "ms").report(), Result(100, 2, "ms").report())

    @unittest.skipUnless(np, "NumPy is not installed")
    def test_the_same_with_the_types_of_numpy(self):
        self.assertEqual(stats.mean([F(np.int8(100))] * 2), 100)
        self.assertEqual(stats.quantile([F(np.int8(100)), F(np.int8(101))], 0.95, exact=True), F(2019, 20))
        self.assertEqual(stats.ecdf_rank(F(np.int8(19), np.int8(20)), 101), 96)
        pairs = [(100, F(np.int8(100))), (200, F(np.int8(100)))]
        self.assertEqual(stats.weighted_quantile(pairs, 0.95), 200)
        s = normal(np.float32(1e9), np.float32(1))
        ys = monte_carlo(lambda q: q[0], [s], n=1000, seed=1)
        self.assertEqual(ys, draws(1000, 1, "gauss", 1e9, 1.0))
        ys = monte_carlo(lambda q: q[0], [rect(np.int8(-100), np.int8(100))], n=1000, seed=1)
        self.assertEqual(ys, draws(1000, 1, "uniform", -100.0, 100.0))
        self.assertEqual(stats.upward(stats.quiet_bound(np.float16(10)), 5), "0.29958")
        self.assertEqual(dist.t_sf(2.0, np.float32(4)), dist.t_sf(2.0, 4.0))


class EveryValueBeforeARank(unittest.TestCase):
    """Section 3 of the review, first part: a flag among the values."""

    XS = [200] * 18 + [400]

    def test_a_flag_takes_no_place(self):
        for flag in FLAGS:
            xs = [flag] + self.XS
            for helper in (stats.ecdf_quantile, stats.order_stat, stats.quantile, stats.quantile_interval):
                self.assertEqual(refusal(helper, xs, 0.95), f"TypeError: no number: {flag}", helper.__name__)
            self.assertEqual(kind(stats.quantile, [flag, 100, 200], 0.95), "TypeError")   # Version 68: 190.0
            self.assertEqual(kind(stats.natives, xs), "TypeError")
        self.assertEqual(kind(stats.ecdf_quantile, [None] + self.XS, 0.95), "TypeError")

    def test_what_the_twenty_rows_establish(self):
        # A measured zero is a duration: the nineteenth of twenty is 200 ms.
        self.assertEqual(stats.ecdf_quantile([0] + self.XS, 0.95), 200)
        self.assertEqual(stats.quantile([0, 100, 200], 0.95), 190.0)
        # Without the row there are nineteen, and the p95 is the nineteenth: 400 ms.
        self.assertEqual(-((-95 * 19) // 100), 19)
        self.assertEqual(stats.ecdf_quantile(self.XS, 0.95), 400)
        # With the row as a request whose duration is missing, 18 of 20 are within 200 ms, one may be, and
        # the objective needs 19: unresolved.
        self.assertEqual(bounds.quantile_verdict(20, 0.95, 18, 1), "unresolved")
        rec = {"web": [None] + self.XS}
        self.assertEqual(mix.within(rec, rec, 200), (F(18, 20), F(19, 20)))
        self.assertEqual(mix.verdict(*mix.within(rec, rec, 200)), "unresolved")

    def test_natives(self):
        clean = [1, 2.5, D("0.1")]
        self.assertIs(stats.natives(clean), clean)                            # nothing to convert
        self.assertEqual(stats.natives((1, 2.5)), [1, 2.5])
        self.assertEqual(stats.natives(range(3)), [0, 1, 2])
        # A stream can be read once: a rank helper that read it would leave it empty for the next.
        self.assertEqual(refusal(stats.natives, iter([1, 2])), "TypeError: need a list of values")
        self.assertEqual(kind(stats.quantile, iter([1.0, 2.0]), 0.5), "TypeError")
        # A set has no order, and a table of names and values has no first value. As this version was first
        # written, it read both: the set in the order that it happened to have, and the table by its names
        # where Version 68 had read it by its places. Found on rereading the listing, and not by the review.
        for loose in ({1, 2, 3}, {0: 5, 1: 7}, frozenset([4.0])):
            self.assertEqual(refusal(stats.natives, loose), "TypeError: need a list of values")
            self.assertEqual(kind(stats.quantile, loose, 0.5), "TypeError")
            self.assertEqual(kind(stats.ecdf_quantile, loose, 0.5), "TypeError")
            self.assertEqual(kind(stats.order_stat, loose, 0.5), "TypeError")
        self.assertEqual(stats.mean(iter([1.0, 2.0])), 1.5)                   # mean() reads it into a list
        got = stats.natives([Wrap8(100), 1.5, part(7, 2), F(1, 3)])
        self.assertEqual(got, [100, 1.5, F(7, 2), F(1, 3)])
        self.assertEqual([type(v) for v in got], [int, float, Fraction, Fraction])
        self.assertIs(type(got[2].numerator), int)

    def test_a_count_of_values(self):
        for flag in FLAGS:
            self.assertEqual(kind(stats.ecdf_rank, 0.95, flag), "ValueError")
            self.assertEqual(kind(stats.lower_rank, 0.95, flag), "ValueError")
        self.assertEqual(kind(stats.ecdf_rank, 0.95, 0), "ValueError")
        self.assertEqual(stats.ecdf_rank(0.95, Wrap8(100)), 95)
        self.assertEqual(stats.lower_rank(0.95, Wrap8(101)), 95)
        self.assertEqual(kind(stats.bootstrap, [1.0, 2.0], max, True), "ValueError")
        self.assertEqual(kind(stats.block_bootstrap, [1.0, 2.0, 3.0], max, True), "ValueError")
        self.assertEqual(kind(stats.block_bootstrap, [1.0, 2.0, 3.0], max, 0), "ValueError")
        self.assertEqual(len(stats.block_bootstrap([1.0, 2.0, 3.0], max, Wrap8(2), Wrap8(5))), 5)

    def test_the_docstrings_say_what_is_checked(self):
        self.assertIn("Every value is looked at, once", " ".join(stats.natives.__doc__.split()))
        # Since Version 70 the rank helpers name ordered(), which is natives() and the refusal of a NaN.
        self.assertIn("ordered()", stats.ecdf_quantile.__doc__)
        self.assertIn("ordered()", stats.order_stat.__doc__)


class ABoundThatIsPrinted(unittest.TestCase):
    """Section 3 of the review, second part: the exposure in the Poisson bounds."""

    @staticmethod
    def silence(exposure, level, digits=40):
        """-ln(1 - level) / exposure in decimals, from the numbers that the two are."""
        with localcontext() as ctx:
            ctx.prec = digits
            return -(1 - dec(level)).ln() / dec(exposure)

    def test_ten_seconds_in_sixteen_bits(self):
        exact = self.silence(10, 0.95)
        self.assertEqual(f"{exact:.13f}", "0.2995732273554")                  # the figure of the review
        with localcontext() as ctx:
            ctx.prec = 40
            up = exact.quantize(D("0.00001"), rounding=ROUND_CEILING)
        self.assertEqual(str(up), "0.29958")
        self.assertEqual(float(1.0 / F16(10) * 2.995732273554), 0.299560546875)  # what 16 bits leave of it
        for exposure in (10, 10.0, F16(10), F32(10), D("10"), F(10), Wrap8(10)):
            got = stats.quiet_bound(exposure)
            self.assertIs(type(got), float)
            text = stats.upward(got, 5)
            self.assertEqual(text, "0.29958", repr(exposure))                 # Version 68: 0.29957 for 16 bits
            self.assertGreaterEqual(D(text), exact)                           # the printed bound is one
            self.assertLess(abs(dec(got) / exact - 1), D("1e-15"))

    def test_an_exposure_of_any_type(self):
        want = (stats.quiet_bound(30), stats.rate_upper(1, 30), stats.rate_limits(1, 30))
        self.assertEqual(f"{want[0]:.8f}", "0.09985774")
        for exposure in (D("30"), F(30), 30.0, F16(30), Wrap8(30), part(30)):   # Version 68: TypeError for D
            got = (stats.quiet_bound(exposure), stats.rate_upper(1, exposure), stats.rate_limits(1, exposure))
            self.assertEqual(got, want, repr(exposure))
        third = stats.quiet_bound(F(1, 3))
        self.assertLess(abs(dec(third) / self.silence(F(1, 3), 0.95) - 1), D("1e-15"))
        self.assertLess(abs(dec(stats.quiet_bound(D("0.1"))) / self.silence(D("0.1"), 0.95) - 1), D("1e-15"))
        for bad in (0, -30, math.nan, INF, D("0"), F(-1, 3)):
            for call in (stats.quiet_bound, lambda e: stats.rate_upper(2, e), lambda e: stats.rate_limits(2, e)):
                self.assertEqual(refusal(call, bad), "ValueError: no exposure", repr(bad))
        for flag in FLAGS:
            self.assertEqual(kind(stats.quiet_bound, 30, flag), "TypeError")
            self.assertEqual(kind(stats.rate_limits, 2, 30, flag), "TypeError")

    def test_a_rate_beyond_the_floats(self):
        # Three events in the least time that a float holds: the quotient of the two numbers is beyond the
        # floats. Version 68 gave inf for both limits, and inf is no lower limit of a rate; as first written,
        # this version stopped with an OverflowError from the division. Found on rereading the listing.
        least = 5e-324
        for call in (stats.quiet_bound, lambda e: stats.rate_upper(3, e), lambda e: stats.rate_limits(3, e)):
            self.assertEqual(refusal(call, least), "ValueError: no rate in a float")
        # Beside it, an exposure that leaves the rate inside the floats.
        got = stats.quiet_bound(1e-300)
        self.assertLess(abs(dec(got) / self.silence(1e-300, 0.95) - 1), D("1e-15"))

    def test_a_level_and_a_mean_of_any_type(self):
        self.assertEqual(stats.rate_upper(1, 30, F32(0.5)), stats.rate_upper(1, 30, 0.5))
        self.assertEqual(stats.rate_limits(3, 30, F32(0.5)), stats.rate_limits(3, 30, 0.5))
        self.assertEqual(stats.wilson(3, 40, F32(0.5)), stats.wilson(3, 40, 0.5))
        for mean in (F32(2.5), F16(2.5), D("2.5"), F(5, 2)):
            self.assertEqual(stats.poisson_tails(3, mean), stats.poisson_tails(3, 2.5), repr(mean))
        self.assertEqual(kind(stats.poisson_tails, 3, True), "TypeError")
        # P(X <= 3) at a mean of 2.5, from its four terms.
        terms = [math.exp(-2.5) * 2.5 ** k / math.factorial(k) for k in range(4)]
        self.assertAlmostEqual(stats.poisson_at_most(3, D("2.5")), sum(terms), places=14)


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


def averaged(pairs):
    """The weighted mean from fractions, or an infinity where a duration is one."""
    if any(x == INF for x, _ in pairs):
        return INF
    return sum(Fraction(x) * w for x, w in pairs) / sum(w for _, w in pairs)


def less(a, b):
    """a - b, or None where both are the same infinity."""
    return None if a == b == INF else a - b


def extent(values):
    """(least, greatest, is any of them not defined) of differences that less() formed, or None for the
    first two where none of them is defined."""
    some = [v for v in values if v is not None]
    if not some:
        return None, None, True
    return min(some), max(some), len(some) < len(values)


def places(rec):
    """Where an open duration of a record can stand so that every least and greatest value of a statistic by
    rank is met: at zero, at each finite duration that is known, above them all, and at infinity."""
    seen = sorted({d for ds in rec.values() for d in ds if d is not None and d != INF})
    return [0] + seen + [seen[-1] + 1 if seen else 1, INF]


def six(stat, before, after, x):
    """Within and mix in the two orders, and their averages, with x for the open duration of the second
    period: each a difference of two statistics, None where it is not defined."""
    start = stat(at_mix(before, before, 0))
    end = stat(at_mix(after, after, x))
    a = stat(at_mix(after, before, x))                                        # the second period at the first mix
    b = stat(at_mix(before, after, 0))
    first = less(a, start), less(end, a)
    second = less(end, b), less(b, start)
    average = tuple(None if None in (p, q) else Fraction(1, 2) * (p + q) for p, q in zip(first, second))
    return first + second + average


class AKnownInfinityBesideAnOpenDuration(unittest.TestCase):
    """Section 4 of the review: held() and the worked mean."""

    BEFORE = {"web": [100], "cache": [10, 10]}
    AFTER = {"web": [INF], "cache": [None]}

    def test_the_record_of_the_review_by_counting(self):
        # The web class is above 5 % at either mix: after, the p95 is infinite whatever the cache took.
        for x in (0, 10, 500, INF):
            self.assertEqual(counted(at_mix(self.AFTER, self.AFTER, x)), INF)
            self.assertEqual(counted(at_mix(self.AFTER, self.BEFORE, x)), INF)
            self.assertEqual(six(counted, self.BEFORE, self.AFTER, x), (INF, None, INF, 0, INF, None))

    def test_the_record_of_the_review_as_printed(self):
        text = printed(mix.components, self.BEFORE, self.AFTER, mix.p95, "p95", "distributions")
        self.assertEqual(text, "p95, 100 -> inf ms: +inf ms\n"                # Version 68: ValueError after this
                               "  open after: 1 of 2 durations\n"
                               "  class distributions first, via inf ms:\n"
                               "    within +inf, mix not defined\n"
                               "  mix first, via 100 ms:\n"
                               "    within +inf, mix +0\n"
                               "  average of the two orders:\n"
                               "    within +inf, mix not defined\n")
        text = printed(mix.components, self.AFTER, self.BEFORE, mix.p95, "p95", "distributions")
        self.assertEqual(text, "p95, inf -> 100 ms: -inf ms\n"
                               "  open before: 1 of 2 durations\n"
                               "  class distributions first, via 100 ms:\n"
                               "    within -inf, mix +0\n"
                               "  mix first, via inf ms:\n"
                               "    within -inf, mix not defined\n"
                               "  average of the two orders:\n"
                               "    within -inf, mix not defined\n")

    def compare(self, stat, counting, before, after):
        if stat is mix.mean:
            rows = [mix.of_mean(*mix.periods(before, after), f) for f in (0, 1, Fraction(1, 2))]
        else:
            rows = mix.of_rank(stat, *mix.periods(before, after))
        got = [span for _, inner, mixed in rows for span in (inner, mixed)]
        seen = [six(counting, before, after, x) for x in places(after)]
        for i, span in enumerate(got):
            low, top, void = extent([row[i] for row in seen])
            if low is None:                                                   # no completion defines it
                self.assertTrue(span[0] != span[0] and span[1] != span[1], (before, after, i, span))
                continue
            if counting is averaged and top != low and top != INF and low != -INF:
                # A mean is linear in the open duration: beyond the places it rises or falls without end.
                rising = [row[i] for row in seen if row[i] is not None]
                top, low = (INF, low) if rising[-1] > rising[0] else (top, -INF)
            self.assertEqual((span[0], span[1]), (low, top), (before, after, i))
            self.assertEqual(isinstance(span, mix.Void), void, (before, after, i))

    def records(self, rng, infinite):
        names = "abc"[:rng.randint(2, 3)]
        before = {c: [rng.randint(1, 9) * 10 for _ in range(rng.randint(1, 3))] for c in names}
        after = {c: [rng.randint(1, 9) * 10 for _ in range(rng.randint(1, 3))] for c in names}
        lost = rng.choice(names)
        after[lost][rng.randrange(len(after[lost]))] = None
        if infinite:
            c = rng.choice(names)
            known = [i for i, d in enumerate(after[c]) if d is not None]
            if known:
                after[c][rng.choice(known)] = INF
        return before, after

    def test_a_percentile_against_counting(self):
        rng = random.Random(6802)
        stopped = 0
        for infinite in (False, True):
            for _ in range(300):
                before, after = self.records(rng, infinite)
                self.compare(mix.p95, counted, before, after)
                first = six(counted, before, after, 0)
                stopped += infinite and first[1] is None
        self.assertGreater(stopped, 20)                                       # records that stopped Version 68

    def test_a_mean_against_fractions(self):
        rng = random.Random(6803)
        marked = 0
        for infinite in (False, True):
            for _ in range(300):
                before, after = self.records(rng, infinite)
                self.compare(mix.mean, averaged, before, after)
                marked += isinstance(mix.of_mean(*mix.periods(before, after), 0)[2], mix.Void)
        self.assertGreater(marked, 100)

    QUARTER = {"lone": [100], "fast": [100, 100], "slow": [200]}

    def later(self, x):
        return {"lone": [x], "fast": [100], "slow": [200, 200]}

    def test_the_worked_mean_by_hand(self):
        # With x for the lost duration: 125 before, 125 + x / 4 after; on the way of the first order
        # 100 + x / 4, and on that of the second 150.
        for x in (0, 40, 10 ** 9):
            got = six(averaged, self.QUARTER, self.later(None), x)
            self.assertEqual(got, (F(x, 4) - 25, 25) * 3)
        self.assertEqual(six(averaged, self.QUARTER, self.later(None), INF), (INF, None, INF, 25, INF, None))

    def test_the_worked_mean_as_printed(self):
        self.assertEqual(printed(mix.components, self.QUARTER, self.later(None)).splitlines(),
                         ["mean, 125 -> 125 ms or more: +0 ms or more",
                          "  open after: 1 of 4 durations",
                          "  class means first, via 100 ms or more:",
                          "    within -25 or more, mix +25 if defined",      # Version 68: mix +25
                          "  mix first, via 150 ms:",
                          "    within -25 or more, mix +25",
                          "  average of the two orders:",
                          "    within -25 or more, mix +25 if defined"])
        # The same request stored as infinity says what the open one says at its upper end.
        self.assertEqual(printed(mix.components, self.QUARTER, self.later(INF)).splitlines(),
                         ["mean, 125 -> inf ms: +inf ms",
                          "  class means first, via inf ms:",                # Version 68: not determined
                          "    within +inf, mix not defined",
                          "  mix first, via 150 ms:",
                          "    within +inf, mix +25",
                          "  average of the two orders:",
                          "    within +inf, mix not defined"])
        self.assertEqual(printed(mix.components, self.QUARTER, self.later(300)).splitlines()[1:],
                         ["  class means first, via 175 ms:", "    within +50, mix +25",
                          "  mix first, via 150 ms:", "    within +50, mix +25",
                          "  average of the two orders:", "    within +50, mix +25"])

    def test_the_docstrings_say_what_is_not_defined(self):
        self.assertIn("their difference is not defined", " ".join(mix.of_mean.__doc__.split()))
        self.assertTrue(all(len(line) <= 50 for line in printed(
            mix.components, self.BEFORE, self.AFTER, mix.p95, "p95", "distributions").splitlines()))


def bernoulli(m):
    """B_0 to B_m as fractions, from the recurrence in which the binomial sums of them vanish."""
    b = [Fraction(1)]
    for n in range(1, m + 1):
        b.append(-sum(math.comb(n + 1, k) * b[k] for k in range(n)) / (n + 1))
    return b


BERNOULLI = bernoulli(40)


def ln_gamma(z, shift=60):
    """ln Gamma(z) for a decimal z above zero: Stirling's series at z + shift, where its twenty terms leave
    less than 1e-55, less the logarithms of the factors z to z + shift - 1."""
    w = z + shift
    s = (w - D("0.5")) * w.ln() - w + (2 * PI).ln() / 2
    for k in range(1, 21):
        s += dec(BERNOULLI[2 * k]) / (2 * k * (2 * k - 1) * w ** (2 * k - 1))
    for k in range(shift):
        s -= (z + k).ln()
    return s


def far_tail(t, dof):
    """Student's upper tail far out, in decimals, for the numbers that t and dof are: with a = dof / 2 and
    x = dof / (dof + t^2) it is x^a / (2 a B(a, 1/2)), and what this leaves out is of the order of x."""
    a = dec(dof) / 2
    x = dec(dof) / (dec(dof) + dec(t) ** 2)
    ln_beta = ln_gamma(a) + PI.sqrt().ln() - ln_gamma(a + D("0.5"))
    return (a * x.ln() - a.ln() - ln_beta).exp() / 2


def far_quantile(tail, dof):
    """The t at which far_tail() is `tail`."""
    a = dec(dof) / 2
    ln_beta = ln_gamma(a) + PI.sqrt().ln() - ln_gamma(a + D("0.5"))
    ln_x = ((2 * dec(tail) * a).ln() + ln_beta) / a
    return ((dec(dof).ln() - ln_x) / 2).exp()


class TheCeilingFromTheMiddle(unittest.TestCase):
    """Section 5 of the review: the two searches at the ceiling."""

    TAIL = 0.3525081422567532
    P = 0.2949837154864936

    def test_the_reference(self):
        with localcontext() as ctx:
            ctx.prec = 80
            self.assertLess(abs(ln_gamma(D("0.5")) - PI.sqrt().ln()), D("1e-55"))   # Gamma(1/2) is sqrt(pi)
            self.assertLess(abs(ln_gamma(D(5)) - D(24).ln()), D("1e-55"))            # Gamma(5) is 4!
            at = far_tail(CEILING, 0.001)
            self.assertEqual(str(at)[:23], "0.352508142256753146237")         # the figure of the review
            self.assertGreater(dec(self.TAIL), at)                            # so the quantile is below 1e150
            q = far_quantile(self.TAIL, 0.001)
            self.assertEqual(f"{q:.17E}", "9.99999999999851514E+149")
        with localcontext() as ctx:
            ctx.prec = 240                                                    # the closed form cancels 80 digits
            # For two degrees of freedom the far tail is 1 / (2 t^2): the closed form (1 - t / sqrt(t^2 + 2)) / 2.
            # The two agree as far as the sixty digits of pi above carry.
            t = D(10) ** 40
            closed = (1 - t / (t * t + 2).sqrt()) / 2
            self.assertLess(abs(far_tail(10 ** 40, 2) / closed - 1), D("1e-55"))

    def test_the_two_calls_of_the_review(self):
        with localcontext() as ctx:
            ctx.prec = 80
            want = far_quantile(self.TAIL, 0.001)
            got = dist.t_isf(self.TAIL, 0.001)                                # Version 68: quantile too large
            self.assertLessEqual(got, CEILING)
            self.assertLess(abs(dec(got) / want - 1), D("2e-9"))              # 1e-12 of the tail, over dof
            # The probability between -t and t is p; the upper tail is 1/2 - p/2 as the number that it is.
            want = far_quantile(F(1, 2) - F(self.P) / 2, 0.001)
            got = Result(0, 1, dof=0.001).k(self.P)
            self.assertLessEqual(got, CEILING)
            self.assertLess(abs(dec(got) / want - 1), D("2e-9"))

    def test_the_last_quantile_that_is_accepted_from_the_middle(self):
        low, high = 0.3525, 0.35251
        self.assertEqual(refusal(dist.t_isf, low, 0.001), "ValueError: quantile too large")
        self.assertIsNone(refusal(dist.t_isf, high, 0.001))
        while math.nextafter(low, high) < high:
            mid = low + (high - low) / 2
            if refusal(dist.t_isf, mid, 0.001):
                low = mid
            else:
                high = mid
        with localcontext() as ctx:
            ctx.prec = 80
            ceiling = dec(CEILING)
            self.assertGreater(far_quantile(low, 0.001), ceiling)             # what is refused lies beyond
            self.assertLess(far_quantile(low, 0.001), ceiling * (1 + D("2e-9")))
            tail = high
            for _ in range(60):
                got = dist.t_isf(tail, 0.001)
                self.assertLessEqual(got, CEILING)
                self.assertLess(abs(dec(got) / far_quantile(tail, 0.001) - 1), D("2e-9"), tail)
                tail = math.nextafter(tail, 1.0)
            # The tail of the ceiling itself, rounded to a float either way, is accepted.
            at = float(far_tail(CEILING, 0.001))
            for tail in (at, math.nextafter(at, 1.0), math.nextafter(at, 0.0)):
                self.assertIsNone(refusal(dist.t_isf, tail, 0.001), tail)
                self.assertIsNone(refusal(dist.t_mid, 0.5 - tail, 0.001), tail)

    def test_far_beyond_is_refused_on_both_routes(self):
        self.assertEqual(refusal(dist.t_isf, 0.35, 0.001), "ValueError: quantile too large")
        self.assertEqual(refusal(dist.t_mid, 0.15, 0.001), "ValueError: quantile too large")
        self.assertEqual(refusal(Result(0, 1, dof=0.001).k, 0.3), "ValueError: quantile too large")
        self.assertEqual(refusal(dist.t_isf, 1e-152, 1), "ValueError: quantile too large")
        self.assertIsNone(refusal(Result(0, 1, dof=0.001).k, 0.29))

    def test_numbers_of_fewer_bits(self):
        self.assertEqual(dist.t_sf(2.0, F32(4)), dist.t_sf(2.0, 4.0))
        self.assertEqual(dist.t_sf(2.0, F32(0.1)), dist.t_sf(2.0, float(F32(0.1))))
        self.assertEqual(dist.t_isf(F32(0.125), F32(4)), dist.t_isf(0.125, 4.0))
        self.assertEqual(dist.t_ppf(F32(0.875), 4), dist.t_ppf(0.875, 4.0))
        self.assertEqual(dist.t_within(F32(0.25), 4), dist.t_within(0.25, 4.0))
        self.assertEqual(dist.t_mid(F16(0.125), Wrap8(4)), dist.t_mid(0.125, 4.0))
        self.assertEqual(Result(0, 1, dof=4).k(F32(0.5)), Result(0, 1, dof=4).k(0.5))
        for bad in ("4", None, True, 2 + 3j):
            self.assertEqual(kind(dist.t_sf, 2.0, bad), "TypeError", repr(bad))
            self.assertEqual(kind(dist.t_isf, 0.1, bad), "TypeError", repr(bad))
            self.assertEqual(kind(dist.t_mid, 0.1, bad), "TypeError", repr(bad))
        self.assertEqual(kind(dist.t_isf, "0.4", 1), "TypeError")
        self.assertEqual(kind(dist.t_ppf, True, 4), "TypeError")

    def test_a_probability_in_decimals(self):
        # Python divides no decimal by a float. Next to the middle, where the quantile is found by
        # proportion, Version 68 stopped at that division for a decimal probability, and so did this
        # version in the new step at the ceiling. Found on rereading the listing, and not by the review.
        # With 4 degrees of freedom the density at 0 is 3/8, and the quantile next to 0 is 8/3 of the gap.
        for gap in (D("1e-30"), F(1, 10 ** 30), 1e-30):
            self.assertLess(abs(dist.t_mid(gap, 4) / (8e-30 / 3) - 1), 1e-12, repr(gap))
            self.assertLess(abs(dist.t_within(gap, 4) / (4e-30 / 3) - 1), 1e-12, repr(gap))
        for p in (D("1e-310"), F(1, 10 ** 310), 1e-310):
            self.assertLess(abs(dist.t_within(p, 4) / 1e-310 * 3 / 4 - 1), 1e-12, repr(p))
        self.assertEqual(dist.t_within(D("1e-400"), 4), 0.0)                  # below every float
        gap = 0.1474918577432468                                              # half the p of the review
        for same in (D(repr(gap)), F(repr(gap)), F(gap)):
            self.assertEqual(dist.t_mid(same, 0.001), 1e150, repr(same))
        self.assertEqual(dist.t_isf(D("0.125"), 4), dist.t_isf(0.125, 4))
        self.assertEqual(dist.t_ppf(D("0.875"), 4), dist.t_ppf(0.875, 4))
        for bad in (D("NaN"), D("Infinity"), D("-Infinity"), math.nan, INF):
            for search in (dist.t_mid, dist.t_isf, dist.t_within, dist.t_ppf):
                self.assertEqual(kind(search, bad, 4), "ValueError", (search.__name__, bad))

    def test_a_flag_after_the_number_that_it_equals(self):
        # Found by the full run of this version and not by the review: both
        # searches keep their answers, True equals 1 and has its hash, and
        # the answer kept for 1 was handed to True in place of the refusal.
        # The order of the calls is the test.
        for search in (dist.t_mid, dist.t_isf):
            self.assertIsInstance(search(0.1, 1), float)
            self.assertEqual(kind(search, 0.1, True), "TypeError")
            self.assertEqual(search(0.1, 1.0), search(0.1, 1))
            self.assertEqual(search(0.125, F32(1)), search(0.125, 1))
        # With one degree of freedom the quantile of an upper tail of 1/8 is
        # the tangent of 3 pi / 8, which is 1 + the square root of 2.
        self.assertAlmostEqual(Result(0, 1, dof=1).k(0.75), 1 + 2 ** 0.5, places=12)
        self.assertEqual(kind(Result, 0, 1, dof=True), "TypeError")

    def test_the_docstring_names_one_rule(self):
        text = " ".join(dist._capped.__doc__.split())
        self.assertIn("Both searches ask here, in the same quantity", text)
        self.assertIn("_capped()", dist.t_mid.__doc__)
        self.assertIn("_capped()", dist.t_isf.__doc__)


if __name__ == "__main__":
    unittest.main()
