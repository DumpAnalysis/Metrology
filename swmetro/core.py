"""Measurement results in GUM terms."""
import math
import statistics as st
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction

from .dist import t_cdf, t_isf, t_within
from .stats import mean


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
    An infinite a keeps its sign, and a ratio
    beyond the range of a float is the
    infinity on its side."""
    if a in (math.inf, -math.inf):
        return float(a)
    z = (_exact(a) - _exact(b)) / _exact(u)
    try:
        return float(z)
    except OverflowError:
        return math.inf if z > 0 else -math.inf


def decimals(u, digits=2):
    """Decimal places that keep `digits`
    significant digits of the uncertainty u,
    counted after rounding, so that 0.0996 is
    shown as 0.10. Negative for the tens."""
    if not u > 0 or not math.isfinite(u):
        return 3
    exp = math.floor(math.log10(u))
    d = digits - 1 - exp
    if _units(u, d) >= 10 ** digits:
        d -= 1
    return d


def resolved(at, spread):
    """Refuse a spread that a float of the
    size of `at` cannot show: the step from
    one float to the next has to be at most
    a thousandth of it. A time stamp that
    counts nanoseconds from 1970 has a step
    of 256 ns as a float, and an uncertainty
    of 1 ns added to it, or drawn around it,
    is lost. The remedy is to subtract an
    origin first, in whole numbers."""
    step = math.ulp(float(at))
    if spread > 0 and 1000 * step > spread:
        raise ValueError(
            "too fine for a float of this size")


def fixed(value, d):
    """The value with d decimal places. A
    float is formatted as a float. A whole
    number or a fraction is rounded as the
    number that it is, half to even, and
    does not pass through a float: a float
    does not hold a time stamp of nineteen
    digits, and Python 3.11 has no such
    format for a fraction."""
    if isinstance(value, float):
        return f"{value:.{d}f}"
    n = round(Fraction(value) * 10 ** d)
    sign = "-" if value < 0 else ""
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
    if not 0 <= p < 1:
        raise ValueError("need 0 <= p < 1")


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
    an assumption."""
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
        factor of a small p to zero. A decimal
        p is the fraction that it is."""
        _level(p)
        if isinstance(p, Decimal):
            p = Fraction(p)
        if not p:
            return 0.0
        if p < 0.5:
            return t_within(p, self.dof)
        return t_isf((1 - p) / 2, self.dof)

    def expanded(self, p=0.95):
        """Expanded uncertainty U = k * u: the
        exact product, as a float."""
        ku = Fraction(self.k(p)) * _exact(self.u)
        return float(ku)

    def interval(self, p=0.95):
        """Coverage interval for probability p.
        With u = 0, and at p = 0, both limits
        are the value as it is stored. Else
        the half-width is added and subtracted
        exactly: the limits are floats where a
        float holds the value, and fractions
        where it does not, as for a time stamp
        of nineteen digits."""
        _level(p)
        w = self.expanded(p)
        if not w:
            return self.value, self.value
        v, w = _exact(self.value), Fraction(w)
        if float(v) == v:
            return float(v - w), float(v + w)
        return v - w, v + w

    def prob_below(self, limit):
        """Probability that the measurand is at
        most `limit`, from the same model as
        the coverage interval. With u = 0 the
        result is a point mass: the probability
        is 1 for a limit at or above the value
        and 0 for a limit below it. A limit
        that is no number is refused. The
        distance to the limit is that of
        _ratio(): 10 ** 18 is one limit as a
        whole number and as a float."""
        if limit != limit:
            raise ValueError("limit is no number")
        if self.u == 0:
            at_most = limit >= self.value
            return 1.0 if at_most else 0.0
        z = _ratio(limit, self.value, self.u)
        return t_cdf(z, self.dof)

    def report(self, p=0.95):
        """(value ± U) unit, with k and p. U
        keeps two significant digits and the
        value is rounded to the same place,
        which may be the tens or above. With
        no uncertainty the value is shown in
        full, since there is no such place. p
        is printed as it was given: a float as
        the shortest decimal that reads back
        as the same float, a fraction in full.
        Six digits would show 0.9999999, with
        its k of 5.33, as p = 1."""
        v, w = self.value, self.expanded(p)
        d = decimals(w)
        if d < 0:
            v, w, d = round(v, d), round(w, d), 0
        if self.u == 0:
            body = f"({full(v)} ± 0)"
        else:
            body = f"({fixed(v, d)} ± {w:.{d}f})"
        tail = f"k = {self.k(p):.2f}"
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
    """A half-width or a step, if it is
    finite and not negative."""
    if not 0 <= w < math.inf:
        raise ValueError("need a width >= 0")
    return w


def u_rect(half_width):
    """Type B: standard deviation of a
    rectangular distribution, an assigned
    model for a value known only by bounds."""
    return _width(half_width) / math.sqrt(3)


def rect_interval(center, half_width, p=0.95):
    """Central coverage interval of a
    rectangular distribution: p of its width,
    whatever its standard deviation, for
    0 <= p <= 1. With no width both limits
    are the center as it was given."""
    if not 0 <= p <= 1:
        raise ValueError("need 0 <= p <= 1")
    w = p * _width(half_width)
    if w == 0:
        return center, center
    resolved(center, w)
    return center - w, center + w


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
