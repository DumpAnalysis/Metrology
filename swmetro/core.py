"""Measurement results in GUM terms."""
import math
import statistics as st
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction

from .dist import t_cdf, t_isf, t_within
from .stats import mean


_ENDS = (math.inf, -math.inf)


def _units(u, d):
    """u in units of the d-th decimal place,
    rounded once, half to even, from the
    number that u is: as a float, u * 10**d
    is 16.5 for a u just above 1.65, and
    gives 16 where 17 is right."""
    return round(Fraction(u) * Fraction(10) ** d)


def _exact(x):
    """x as the fraction that it is. A text is
    no number here, although Fraction() would
    read one."""
    if isinstance(x, str):
        raise TypeError("a text is no number")
    return Fraction(x)


def _ratio(a, b, u):
    """(a - b) / u from the exact numbers that
    the three are, as a float rounded once.
    A ratio beyond the range of a float is
    the largest float on its side: the
    distance is finite, and does not get the
    probability of an infinite one."""
    z = (_exact(a) - _exact(b)) / _exact(u)
    try:
        return float(z)
    except OverflowError:
        top = math.nextafter(math.inf, 0)
        return top if z > 0 else -top


def _exp10(u):
    """The e with 10 ** e <= u < 10 ** (e + 1)
    for a u above zero, in whole numbers: a
    logarithm would pass through a float."""
    u = Fraction(u)
    bits = u.numerator.bit_length()
    bits -= u.denominator.bit_length()
    e = bits * 30103 // 100000
    while u < Fraction(10) ** e:
        e -= 1
    while u >= Fraction(10) ** (e + 1):
        e += 1
    return e


def decimals(u, digits=2):
    """Decimal places that keep `digits`
    significant digits of the uncertainty u,
    counted after rounding, so that 0.0996 is
    shown as 0.10. Negative for the tens."""
    if not u > 0 or u == math.inf:
        return 3
    d = digits - 1 - _exp10(u)
    if _units(u, d) >= 10 ** digits:
        d -= 1
    return d


def resolved(at, spread):
    """Refuse a spread that a float of the
    size of `at` cannot show: the step from
    one float to the next has to be at most
    a thousandth of it. A time stamp of 2023
    in nanoseconds from 1970, 1.7e18, has a
    step of 256 ns as a float, and one near
    1e18 a step of 128 ns: an uncertainty of
    1 ns added to it, or drawn around it, is
    lost. The remedy is to subtract an origin
    first, in whole numbers. The step and the
    spread are compared exactly."""
    step = 1000 * Fraction(math.ulp(float(at)))
    if spread > 0 and step > _exact(spread):
        raise ValueError(
            "too fine for a float of this size")


def fixed(value, d):
    """The value with d decimal places,
    rounded half to even from the number
    that it is. A whole number or a fraction
    does not pass through a float: a float
    does not hold every time stamp of
    nineteen digits, and Python 3.11 has no
    such format for a fraction. A value that
    rounds to zero is shown without a sign."""
    if value != value or value in _ENDS:
        return f"{value}"
    n = round(Fraction(value) * 10 ** d)
    sign = "-" if n < 0 else ""
    whole, part = divmod(abs(n), 10 ** d)
    if not d:
        return f"{sign}{whole}"
    return f"{sign}{whole}.{part:0{d}d}"


def full(value):
    """A value in full, for a result with no
    uncertainty. A fraction that ends after
    some decimal places is written with
    them, 1/8 as 0.125, and one that does
    not end is written as the ratio that it
    is, 1/3."""
    if not isinstance(value, Fraction):
        return f"{value}"
    q, d = value.denominator, 0
    for prime in (2, 5):
        while q % prime == 0:
            q //= prime
    if q > 1:
        return f"{value}"
    while value * 10 ** d % 1:
        d += 1
    return fixed(value, d)


def fmt(value, u, unit="", digits=2):
    """Concise form 12.3110(97) ms: the digits
    in parentheses are the standard uncertainty
    in units of the last digits shown. With no
    uncertainty there is no place to round
    the value to, and it is shown in full:
    99.9996(0) ms. The digits of u are those
    of the float: 0.165, stored just above
    itself, gives (17), and 0.155 gives (15)."""
    d = decimals(u, digits)
    if u == 0:
        text = f"{full(value)}(0)"
    elif d > 0:
        n = _units(u, d)
        text = f"{fixed(value, d)}({n})"
    else:
        scale = 10 ** -d
        v = round(Fraction(value) / scale)
        w = _units(u, d)
        text = f"{v * scale}({w * scale})"
    return f"{text} {unit}".strip()


def _level(p):
    """A coverage probability: a decimal p as
    the fraction that it is."""
    if isinstance(p, Decimal):
        p = Fraction(p)
    if not 0 <= p < 1:
        raise ValueError("need 0 <= p < 1")
    return p


@dataclass(frozen=True)
class Result:
    """Measured quantity value, standard
    uncertainty u, unit, degrees of freedom.

    The interval methods are a reporting aid.
    They assume that the result is described
    by a normal distribution or, for finite
    dof, a scaled Student t distribution. A
    standard uncertainty alone does not fix
    a coverage interval: where one bounded or
    skewed component dominates, propagate the
    distributions instead. The default dof
    treats u as exactly known, which is itself
    an assumption. A value or a u beyond the
    range of a float, 1.8e308, is refused.
    With finite dof the model is value + u T
    for a Student T: u is its scale, as an
    estimated standard error is in a t
    interval, and not its standard deviation,
    which at 4 degrees of freedom is u times
    the square root of 2."""
    value: float
    u: float
    unit: str = ""
    dof: float = math.inf

    def __post_init__(self):
        u = self.u
        if not math.isfinite(self.value):
            raise ValueError("value is no number")
        if not (u >= 0 and math.isfinite(u)):
            raise ValueError("u must be >= 0")
        if not self.dof > 0:
            raise ValueError("dof must be > 0")
        resolved(self.value, u)

    def k(self, p=0.95):
        """Coverage factor for probability p,
        found from the tail (1 - p) / 2: a p
        next to 1 keeps its digits there, and
        1/2 + p/2 would round it to 1. Below
        one half it is found from p / 2, the
        distance from the middle: there the
        tail would round to 1/2, and the
        factor of a small p to zero. Below
        1e-300, where the factor is in
        proportion to p, it is found for p
        times a power of two, near 2 ** -600,
        and divided by that power. As a float,
        a factor below the least float is 0.0:
        the widths are formed from _k(). A p
        whose tail is below the least normal
        float, 2.2e-308, is refused: such a
        tail has lost its digits."""
        return float(self._k(p))

    def _k(self, p):
        """The factor as a fraction."""
        p = _level(p)
        if not p:
            return Fraction(0)
        if p < 1e-300:
            p = Fraction(p)
            e = p.denominator.bit_length() - 600
            e -= p.numerator.bit_length()
            p = float(p * 2 ** e)
            return self._k(p) / 2 ** e
        if p < 0.5:
            return Fraction(t_within(p, self.dof))
        tail = (1 - p) / 2
        if tail < 2 ** -1022:
            raise ValueError("p is too near 1")
        return Fraction(t_isf(tail, self.dof))

    def _half(self, p):
        """k * u as the fraction that it is.
        With u = 0 it is zero for every level,
        and no factor is looked for."""
        if not (_level(p) and self.u):
            return Fraction(0)
        return self._k(p) * _exact(self.u)

    def expanded(self, p=0.95):
        """Expanded uncertainty U = k * u: the
        exact product, as a float. A product
        beyond the range of a float is refused
        here: interval() and report() take it
        as the fraction that it is."""
        try:
            return float(self._half(p))
        except OverflowError:
            pass
        raise OverflowError("U is beyond a float")

    def interval(self, p=0.95):
        """Coverage interval for probability p.
        With u = 0, and at p = 0, both limits
        are the value as it is stored, and no
        factor is found. Else they are the
        value less k * u and plus k * u as
        fractions, the numbers that they are.
        A limit rounded to a float moves with
        the origin of the clock, by up to
        64 ns near 1e18 ns, and prob_below()
        would not find at it the probability
        that k stands for."""
        w = self._half(p)
        if not w:
            return self.value, self.value
        v = _exact(self.value)
        return v - w, v + w

    def prob_below(self, limit):
        """Probability that the measurand is at
        most `limit`, from the same model as
        the coverage interval. With u = 0 the
        result is a point mass: the probability
        is 1 for a limit at or above the value
        and 0 for a limit below it. A limit
        that is no number is refused, and an
        infinite one has the probability 1 or
        0 whatever the distribution. The
        distance to a finite limit is that of
        _ratio(): 10 ** 18 is one limit as a
        whole number and as a float."""
        if limit != limit:
            raise ValueError("limit is no number")
        if limit in _ENDS:
            return float(limit > 0)
        if self.u == 0:
            v = _exact(self.value)
            return float(_exact(limit) >= v)
        z = _ratio(limit, self.value, self.u)
        return t_cdf(z, self.dof)

    def report(self, p=0.95):
        """(value ± U) unit, with k and p. U
        keeps two significant digits and the
        value is rounded to the same place,
        which may be the tens or above: both
        half to even from the numbers that
        they are, whatever the rounding of
        the decimal context. With no
        uncertainty, and at p = 0, U is 0 and
        the value is shown in full, since
        there is no such place. Without
        uncertainty no factor is looked for,
        and "u = 0" stands in its place. A
        factor has two decimal places, or two
        significant digits where that takes
        more. p is printed as it was given: a
        float as the shortest decimal that
        reads back as the same float, a
        fraction in full. Six digits would
        show 0.9999999, with its k of 5.33,
        as p = 1."""
        v, w = self.value, self._half(p)
        d = decimals(w)
        if d < 0:
            scale, d = 10 ** -d, 0
            v = round(_exact(v) / scale) * scale
            w = round(w / scale) * scale
        if not w:
            body = f"({full(v)} ± 0)"
        else:
            v, w = fixed(v, d), fixed(w, d)
            body = f"({v} ± {w})"
        tail = "u = 0"
        if self.u:
            k = self._k(p)
            d = max(2, decimals(k)) if k else 2
            tail = f"k = {fixed(k, d)}"
        if isinstance(p, float):
            p = repr(p)
        tail = f"{tail}, p = {full(p)}"
        head = f"{body} {self.unit}".strip()
        return f"{head}, {tail}"

    def __str__(self):
        return fmt(self.value, self.u, self.unit)


def type_a(xs, unit=""):
    """Type A evaluation: the mean and the
    experimental standard deviation of the
    mean of n independent indications. The
    sum is exact and the mean is rounded
    once; a whole mean of whole numbers
    stays a whole number, see mean()."""
    n = len(xs)
    u = st.stdev(xs) / math.sqrt(n)
    return Result(mean(xs), u, unit, n - 1)


def _width(w):
    """A half-width or a step as a float, if
    it is finite and not negative: a whole
    number, a fraction or a decimal as well."""
    if not w >= 0 or w == math.inf:
        raise ValueError("need a width >= 0")
    return float(w)


def u_rect(half_width):
    """Type B: standard deviation of a
    rectangular distribution, an assigned
    model for a value known only by bounds."""
    return _width(half_width) / math.sqrt(3)


def rect_interval(center, half_width, p=0.95):
    """Central coverage interval of a
    rectangular distribution: p of its width,
    whatever its standard deviation, for
    0 <= p <= 1. The limits are the center
    less and plus the exact product of p and
    the half-width, as fractions: rounded to
    floats, they can leave out a part of the
    width, also at p = 1. With no width both
    limits are the center as it was given."""
    if not 0 <= p <= 1:
        raise ValueError("need 0 <= p <= 1")
    _width(half_width)
    w = _exact(p) * _exact(half_width)
    if not w:
        return center, center
    resolved(center, w)
    v = _exact(center)
    return v - w, v + w


def u_tri(half_width):
    """Type B: triangular distribution."""
    return _width(half_width) / math.sqrt(6)


def u_resolution(step):
    """Type B: indication quantized in steps."""
    return _width(step) / math.sqrt(12)


def _scaled(us):
    """The values divided by a power of two
    near the largest, and its exponent. The
    division changes no digit, and it keeps
    the squares and fourth powers of very
    small or very large values within what
    a float can hold."""
    top = max(map(abs, us), default=0)
    e = math.frexp(top)[1]
    return [math.ldexp(u, -e) for u in us], e


def combine(*us):
    """Root sum of squares of uncorrelated
    standard uncertainties."""
    us, e = _scaled(us)
    root = math.sqrt(sum(u * u for u in us))
    return math.ldexp(root, e)


def welch_satterthwaite(us, dofs):
    """Effective degrees of freedom of the
    combined standard uncertainty. Every
    uncertainty needs its degrees of freedom:
    one that is left out would count as
    infinitely many. math.inf says so where
    that is meant."""
    us, dofs = list(us), list(dofs)
    if len(dofs) != len(us):
        raise ValueError("one dof for each u")
    if not all(d > 0 for d in dofs):
        raise ValueError("dof must be > 0")
    if not all(0 <= u < math.inf for u in us):
        raise ValueError("u must be >= 0")
    us, _ = _scaled(us)
    den = sum(u ** 4 / d for u, d in zip(us, dofs)
              if u > 0 and math.isfinite(d))
    if not den:
        return math.inf
    return combine(*us) ** 4 / den


class Budget:
    """Uncertainty budget: named components,
    all in the unit of the measurand. The
    components are combined as independent;
    correlated inputs need `propagate.lpu`."""

    def __init__(self, unit=""):
        self.unit, self.rows = unit, []

    def add(self, name, kind, u, dof=math.inf):
        """kind: 'A' or 'B' (evaluation type).
        u must be finite and not negative."""
        if kind not in ("A", "B"):
            raise ValueError("kind is 'A' or 'B'")
        if not (u >= 0 and math.isfinite(u)):
            raise ValueError("u must be >= 0")
        if not dof > 0:
            raise ValueError("dof must be > 0")
        self.rows.append((name, kind, u, dof))
        return self

    def u_c(self):
        """Combined standard uncertainty."""
        return combine(*(r[2] for r in self.rows))

    def dof(self):
        """Effective degrees of freedom."""
        us = [r[2] for r in self.rows]
        dofs = [r[3] for r in self.rows]
        return welch_satterthwaite(us, dofs)

    def table(self):
        """Text table with variance shares."""
        uc = self.u_c()
        out = [f"{'component':<21}{'type':<5}"
               f"{'u':>9}{'share':>9}"]
        for name, kind, u, _ in self.rows:
            share = 0.0
            if uc:
                share = 100 * (u / uc) ** 2
            row = f"{name:<21}{kind:<5}{u:>9.4f}"
            out.append(f"{row}{share:>7.1f} %")
        last = "combined u_c"
        out.append(f"{last:<26}{uc:>9.4f}")
        return "\n".join(out)
