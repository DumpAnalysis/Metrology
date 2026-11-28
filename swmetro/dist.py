"""Normal and Student t helpers. They compute
with floats of 64 bits. Degrees of freedom
are held as such a float, and so is a
probability unless it is a Fraction or a
Decimal, which is compared as the number
that it is. What is a number, and in which
range, is settled where a result is formed."""
import math
from decimal import Decimal
from fractions import Fraction
from functools import lru_cache
from numbers import Real
from statistics import NormalDist

_N = NormalDist()
phi = _N.cdf        # standard normal CDF
z_ppf = _N.inv_cdf  # standard normal quantile
_TINY = 1e-300
_T_MAX = 1e150
_LEAST = 2.0 ** -1022  # the least normal float
_FEW = 1e-9
_MANY = 1e25


def _float(x, exact=False):
    """x as a float of 64 bits: a float of 32
    bits would carry its width into every
    product. With `exact`, a Fraction or a
    finite Decimal is the fraction that it
    is: Python compares a fraction with a
    float exactly, and does not divide a
    decimal by one. A text and a truth value
    are no numbers. A Fraction is formed
    anew from ints: its two parts can be of
    a fixed width, and they are then no key
    under which an answer can be kept."""
    if type(x) is float:
        return x
    real = isinstance(x, (Real, Decimal))
    if isinstance(x, bool) or not real:
        raise TypeError(f"no number: {x!r}")
    if isinstance(x, Fraction):
        n, d = x.numerator, x.denominator
        x = Fraction(int(n), int(d))
    kinds = (Fraction, Decimal)
    if exact and isinstance(x, kinds):
        finite = getattr(x, "is_finite", None)
        if finite is None or finite():
            return Fraction(x)
    return float(x)


def _plain(t):
    """A decimal as the fraction that it is,
    or as the float for an infinity and a
    NaN, away from the decimal context of
    the caller. A text and a truth value
    are no numbers."""
    if type(t) is float:
        return t
    real = isinstance(t, (Real, Decimal))
    if isinstance(t, bool) or not real:
        raise TypeError(f"no number: {t!r}")
    if isinstance(t, Decimal):
        return _float(t, True)
    return t


def _fix(x):
    return x if abs(x) > _TINY else _TINY


def _frac(a, b, x, y, lam):
    """Continued fraction of the incomplete
    beta function in the form of Didonato
    and Morris, by the modified Lentz
    method; y is 1 - x, and lam is
    a - (a + b) x. With b / (a + b) and the
    exponential of _lead() before it, it is
    I_x(a, b). The usual fraction is given
    x alone. For a large a next to x = 1 it
    answers the rounding of x with an error
    a times as large, about 1e-10 of the
    result at a million degrees of freedom:
    lam carries what 1 - x knows."""
    f = c = _fix((1 + lam) / (a + 1))
    d = 0.0
    for n in range(1, 1000):
        s = a + 2 * n - 1
        w = n * (b - n) * x
        if n == 1:
            p = (a + b) / (a + 1)
            num = (b - 1) * x * x * p / (a + 1)
        else:
            p = (a + n - 1) / s
            num = w * x * p * (a + b + n - 1) / s
        den = 1 + lam + n * (1 + y)
        den = n + w / s + den * (a + n) / (s + 2)
        d = 1 / _fix(den + num * d)
        c = _fix(den + num / c)
        f *= c * d
        if abs(c * d - 1) < 3e-16:
            break
    return 1 / f


def _log(x, y):
    """log(x), where y is 1 - x. Near 1 it is
    taken from y, which is then the more
    exact of the two."""
    if x < 0.5:
        return math.log(x)
    return math.log1p(-y)


def _stirling(z):
    """What the leading terms of Stirling's
    series leave of log Gamma(z), for a z of
    32 or more: four terms, below 1e-17 in
    what is left out of a difference."""
    q = 1 / (z * z)
    s = 1 / 1260 - q / 1680
    return (1 / 12 - (1 / 360 - s * q) * q) / z


def _lead(a, b, x, y):
    """log of x^a y^b Gamma(a + b + 1) /
    (Gamma(a + 1) Gamma(b + 1)), where y is
    1 - x: what stands before the continued
    fraction, but for a ratio of the shapes.
    The gamma function is taken one above
    each argument, where its logarithm is
    not large next to 0. For a shape of 32
    or more the logarithms of Gamma(a + b + 1)
    and of Gamma(a + 1) are large as well,
    six million at a million degrees of
    freedom, and their difference would keep
    nine digits. It is then taken from
    Stirling's series, in which the large
    parts cancel before anything is rounded,
    and the logarithm of a + b, which that
    series leaves, is taken together with
    that of y, whose size it offsets."""
    if a < b:
        a, b, x, y = b, a, y, x
    ln = a * _log(x, y)
    if a < 32:
        ln += math.lgamma(a + b + 1)
        ln -= math.lgamma(a + 1)
        ln -= math.lgamma(b + 1)
        return ln + b * _log(y, x)
    z = a + 1
    ln += (z - 0.5) * math.log1p(b / z) - b
    ln += _stirling(z + b) - _stirling(z)
    ln -= math.lgamma(b + 1)
    return ln + b * math.log((z + b) * y)


def betainc(a, b, x, y=None):
    """Regularized incomplete beta function
    I_x(a, b) for a, b > 0 and 0 <= x <= 1.
    Give y = 1 - x as well where it is known
    more exactly than the subtraction leaves
    it, as it is for an x next to 1. It is
    written for Student's t, where one shape
    is one half, and checked there. Two
    large shapes cost it digits, three at a
    thousand each, and a result next to 0
    that is found as 1 less the other part
    keeps what that subtraction leaves."""
    a, b, x = _float(a), _float(b), _float(x)
    y = 1 - x if y is None else _float(y)
    if x <= 0:
        return 0.0
    if y <= 0:
        return 1.0
    ln = _lead(a, b, x, y)
    # The fraction is short while x is below
    # (a + 1) / (a + b + 2), and it serves
    # up to a / (a + b), where lam is 0 and
    # the answer near one half: beyond it
    # the lesser part is found, from the
    # other end. Each question is put to
    # the lesser of x and y: the one next
    # to 1 has lost its digits.
    if x < 0.5:
        lam = a - (a + b) * x
        near = x * (a + b + 2) < a + 1
    else:
        lam = (a + b) * y - b
        near = y * (a + b + 2) > b + 1
    # The ratio of the shapes goes to the
    # fraction first: with the exponential
    # it could fall below the least normal
    # floating-point number before the
    # fraction raises it again.
    if near or lam >= 0:
        f = _frac(a, b, x, y, lam)
        return math.exp(ln) * (f * (b / (a + b)))
    f = _frac(b, a, y, x, -lam)
    return 1 - math.exp(ln) * (f * (a / (a + b)))


def _split(t, dof):
    """The two parts of 1 that the incomplete
    beta function needs: t^2 / (dof + t^2) and
    its complement, each formed directly, so
    that neither is lost next to 0 or far
    out, with very few degrees of freedom or
    very many."""
    # The cap keeps the square a float. Up
    # to t = 1e153 * sqrt(dof) nothing is
    # cut off, and beyond it t_sf() takes
    # the tail from the logarithm of t.
    r = min(t / math.sqrt(dof), 1e153)
    q = r * r
    return q / (1 + q), 1 / (1 + q)


def _tail(t, dof):
    """Upper tail where t^2 / dof is beyond a
    float: the first term of the series of
    the incomplete beta function, all of it
    there, from the logarithm of t. Below
    _FEW degrees of freedom it is formed as
    in _few(), of which this is the far
    end."""
    n, d = t.as_integer_ratio()
    a = dof / 2
    ln = math.log(dof) + 2 * math.log(d)
    ln = a * (ln - 2 * math.log(n))
    if dof < _FEW:
        ln -= dof * math.log(2)
        return 0.5 * math.exp(ln)
    ln += math.lgamma(a + 0.5)
    ln -= math.lgamma(a + 1)
    return 0.5 * math.exp(ln) / math.sqrt(math.pi)


def _few(dof, x, y):
    """Upper tail below _FEW degrees of freedom,
    from x = dof / (dof + t^2) and y = 1 - x.
    With a = dof / 2 next to 0 the incomplete
    beta function I_x(a, 1/2) is x^a times
    1 - 2 a log(1 + sqrt(y)), and what that
    leaves out is of the order of a^2, below
    the rounding of a float. The tail is
    half of it: it falls as t rises, and
    never exceeds one half. The least float
    has no half, and needs none here."""
    ln = _log(x, y) / 2 - math.log1p(math.sqrt(y))
    return 0.5 * math.exp(dof * ln)


def t_sf(t, dof):
    """Upper tail P(T > t) of Student's t for
    t >= 0, accurate far into the tail, and
    next to t = 0 as well. Nothing lies
    above infinity. t is a float, or an
    exact number of any size. One method
    serves every number of degrees of
    freedom up to 1e25. Beyond it the
    normal tail is used, an approximation
    within the accuracy measured here.
    Against references in decimals, from
    1e-9 degrees of freedom upward, the
    relative error was below 1e-12 down to
    tails of 1e-300, and below 1e-13 for
    tails above 1e-12. Below that accuracy
    the tail is not monotonic: between two
    adjacent values of t it can rise by
    rounding."""
    dof, t = _float(dof), _plain(t)
    if t == math.inf:
        return 0.0
    if dof > _MANY:
        # The normal tail. Beyond 1000 it
        # is 0 as a float.
        z = float(min(t, 1e3))
        return 0.5 * math.erfc(z / math.sqrt(2))
    if t > 1e153 * math.sqrt(dof):
        return _tail(t, dof)
    y, x = _split(float(t), dof)
    if dof < _FEW:
        return _few(dof, x, y)
    return 0.5 * betainc(dof / 2, 0.5, x, y)


def t_cdf(t, dof):
    """CDF of Student's t distribution."""
    t = _plain(t)
    if t < 0:
        return t_sf(-t, dof)
    return 1 - t_sf(t, dof)


def _bisect(below, lo, hi):
    """The t at which `below` turns from true
    to false: hi is doubled, up to 1e150,
    until it is past that point, and the
    interval is then halved as far as floats
    go. A point beyond 1e150 is refused."""
    while below(hi):
        if hi >= _T_MAX:
            raise ValueError("quantile too large")
        lo, hi = hi, min(2 * hi, _T_MAX)
    for _ in range(200):
        mid = (lo + hi) / 2
        if mid in (lo, hi):
            break
        if below(mid):
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _capped(tail, dof):
    """True if the upper tail at the ceiling
    of 1e150 exceeds `tail` by 1e-12 of it or
    less, the accuracy measured for a tail.
    The quantile is then taken as 1e150, so
    that the rounding of a probability does
    not refuse a quantile below the ceiling.
    Both searches ask here, in the same
    quantity: the search from the middle
    with one half less its probability."""
    near = t_sf(_T_MAX, dof) / (1 + 1e-12)
    return near <= tail


def _center(t, dof):
    """P(0 < T < t) of Student's t for t >= 0.
    Next to t = 0 it keeps its relative
    accuracy, which 1/2 less the upper tail
    does not."""
    x, y = _split(t, dof)
    return 0.5 * betainc(0.5, dof / 2, x, y)


def _z_mid(gap):
    """Normal quantile at 1/2 + gap, from the
    gap. For a small gap the sum 1/2 + gap
    would lose its digits, and a series in
    the gap is used."""
    if gap > 1e-3:
        return z_ppf(0.5 + gap)
    g = gap * gap * math.pi
    z = 1 + g / 3 * (1 + 0.7 * g)
    return math.sqrt(2 * math.pi) * gap * z


def t_mid(gap, dof):
    """The t >= 0 with P(0 < T < t) = gap, for
    0 < gap < 0.5 and dof > 0: the quantile of
    Student's t at 1/2 + gap, found from the
    gap itself, so that a probability next to
    one half keeps its distance from one half
    and its quantile is not read as zero.
    Next to t = 0 the probability is
    proportional to t, and the quantile is
    found by that proportion. Checked for
    dof of 0.001 and more; with fewer its
    accuracy is not claimed. A quantile
    beyond 1e150 is refused, and one at that
    ceiling is taken as 1e150 by _capped().
    The two arguments are taken as _float()
    gives them before an answer is kept or
    looked up: True is no key for the answer
    of 1."""
    gap, dof = _float(gap, True), _float(dof)
    return _t_mid(gap, dof)


@lru_cache(maxsize=None)
def _t_mid(gap, dof):
    """t_mid() for numbers of Python's own,
    with its answers kept."""
    if not 0 < gap < 0.5:
        raise ValueError("need 0 < gap < 0.5")
    if not dof > 0:
        raise ValueError("dof must be positive")
    if dof > _MANY:
        return _z_mid(gap)
    lo = 1e-9 * min(1.0, math.sqrt(dof))
    at = _center(lo, dof)
    if gap <= at:
        return gap / at * lo

    def below(t):
        return _center(t, dof) < gap
    try:
        return _bisect(below, lo, 2 * lo)
    except ValueError:
        tail = Fraction(1, 2) - Fraction(gap)
        if not _capped(tail, dof):
            raise
    return _T_MAX


def t_within(p, dof):
    """The t >= 0 with P(-t < T < t) = p, for
    0 < p < 0.5: t_mid() at p / 2. Half the
    least positive float is 0. Below 1e-300,
    where t is proportional to p, the
    quantile is found for 2**52 p and
    divided by 2**52."""
    p = _float(p, True)
    if p < 1e-300:
        big = t_mid(p * 2 ** 51, dof)
        return big / 2.0 ** 52
    return t_mid(p / 2, dof)


def t_isf(tail, dof):
    """Inverse upper tail of Student's t: the
    t >= 0 with P(T > t) = tail, for dof > 0
    and a tail from the least normal float,
    2.2e-308, to 0.5, by bisection on t. A
    smaller tail is refused: as a float it
    has lost digits, and no tail that is
    computed can be compared with it. The
    tail is a float, or a Fraction or a
    Decimal, which is compared as the number
    that it is. Checked against the closed
    forms for 1 and 2 degrees of freedom
    down to tails of 1e-100 and against
    tables. Above a tail of 1/4 the quantile
    is found from the distance of the tail
    from 1/2, which is formed exactly, and
    beyond 1e25 degrees of freedom it is the
    normal quantile. A quantile beyond 1e150
    is refused, and one at that ceiling is
    taken as 1e150, see _capped()."""
    tail, dof = _float(tail, True), _float(dof)
    return _t_isf(tail, dof)


@lru_cache(maxsize=None)
def _t_isf(tail, dof):
    """t_isf() for numbers of Python's own,
    with its answers kept."""
    if not _LEAST <= tail <= 0.5:
        raise ValueError("need 2**-1022"
                         " <= tail <= 0.5")
    if not dof > 0:
        raise ValueError("dof must be positive")
    if tail == 0.5:
        return 0.0
    if tail > 0.25:
        gap = Fraction(1, 2) - Fraction(tail)
        return t_mid(float(gap), dof)
    if dof > _MANY:
        return -z_ppf(float(tail))

    def below(t):
        return t_sf(t, dof) > tail
    try:
        return _bisect(below, 0.0, 1.0)
    except ValueError:
        if not _capped(tail, dof):
            raise
    return _T_MAX


def t_ppf(p, dof):
    """Quantile of Student's t for 0 < p < 1
    and dof > 0. The lower half is found from
    the tail probability p itself and not
    from 1 - p, which would lose the digits
    of a small p. The upper half is found
    from 1 - p, which is formed exactly. The
    tails are those of t_isf()."""
    p = _float(p, True)
    if not 0 < p < 1:
        raise ValueError("p must be in (0, 1)")
    if p < 0.5:
        return -t_isf(p, dof)
    return t_isf(1 - Fraction(p), dof)
