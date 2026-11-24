"""Normal and Student t helpers."""
import math
from functools import lru_cache
from statistics import NormalDist

_N = NormalDist()
phi = _N.cdf        # standard normal CDF
z_ppf = _N.inv_cdf  # standard normal quantile
_TINY = 1e-300
_T_MAX = 1e150


def _fix(x):
    return x if abs(x) > _TINY else _TINY


def _betacf(a, b, x):
    """Continued fraction of the incomplete
    beta function, modified Lentz method."""
    qab, qap, qam = a + b, a + 1, a - 1
    c = 1.0
    d = 1 / _fix(1 - qab * x / qap)
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        num = m * (b - m) * x
        num /= (qam + m2) * (a + m2)
        d = 1 / _fix(1 + num * d)
        c = _fix(1 + num / c)
        h *= d * c
        num = -(a + m) * (qab + m) * x
        num /= (a + m2) * (qap + m2)
        d = 1 / _fix(1 + num * d)
        c = _fix(1 + num / c)
        h *= d * c
        if abs(d * c - 1) < 1e-15:
            break
    return h


def _log(x, y):
    """log(x), where y is 1 - x. Near 1 it is
    taken from y, which is then the more
    exact of the two."""
    if x < 0.5:
        return math.log(x)
    return math.log1p(-y)


def betainc(a, b, x, y=None):
    """Regularized incomplete beta function
    I_x(a, b) for a, b > 0 and 0 <= x <= 1.
    Give y = 1 - x as well where it is known
    more exactly than the subtraction leaves
    it, as it is for an x next to 1."""
    if y is None:
        y = 1 - x
    if x <= 0:
        return 0.0
    if y <= 0:
        return 1.0
    ln = math.lgamma(a + b) - math.lgamma(a)
    ln -= math.lgamma(b)
    ln += a * _log(x, y) + b * _log(y, x)
    front = math.exp(ln)
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    return 1 - front * _betacf(b, a, y) / b


def _far(z, v, sign):
    """Between a normal quantile z and the
    quantile t of Student's distribution for
    many degrees of freedom v: the first
    terms of the asymptotic expansion in 1/v.
    With sign 1 it gives t from z, three
    terms. With -1 it gives z from t, by the
    reversed series, two terms."""
    q = z * z
    g1 = z * (q + 1) / 4
    if sign > 0:
        g2 = ((5 * q + 16) * q + 3) * z / 96
        g3 = ((3 * q + 19) * q + 17) * q - 15
        g2 += g3 * z / (384 * v)
    else:
        g2 = ((13 * q + 8) * q + 3) * z / 96
    return z + (sign * g1 + g2 / v) / v


def _split(t, dof):
    """The two parts of 1 that the incomplete
    beta function needs: t^2 / (dof + t^2) and
    its complement, each formed directly, so
    that neither is lost next to 0 or far
    out, with very few degrees of freedom or
    very many."""
    # The cap keeps the square a float. Up
    # to t = 1e153 * sqrt(dof) nothing is
    # cut off, and beyond it the tail is
    # overstated, never understated.
    r = min(t / math.sqrt(dof), 1e153)
    q = r * r
    return q / (1 + q), 1 / (1 + q)


def t_sf(t, dof):
    """Upper tail P(T > t) of Student's t for
    t >= 0, accurate far into the tail, and
    next to t = 0 as well."""
    if dof > 1e6:
        # Beyond 1000 the tail is 0 as a float.
        z = _far(min(t, 1e3), dof, -1)
        return 0.5 * math.erfc(z / math.sqrt(2))
    y, x = _split(t, dof)
    return 0.5 * betainc(dof / 2, 0.5, x, y)


def t_cdf(t, dof):
    """CDF of Student's t distribution."""
    if t < 0:
        return t_sf(-t, dof)
    return 1 - t_sf(t, dof)


def _bisect(below, lo, hi):
    """The t at which `below` turns from true
    to false: hi is doubled until it is past
    that point, and the interval is then
    halved as far as floats go. A point
    beyond 1e150 is refused."""
    while below(hi):
        lo, hi = hi, 2 * hi
        if hi > _T_MAX:
            raise ValueError("quantile too large")
    for _ in range(200):
        mid = (lo + hi) / 2
        if mid in (lo, hi):
            break
        if below(mid):
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _centre(t, dof):
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


@lru_cache(maxsize=None)
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
    accuracy is not claimed."""
    if not 0 < gap < 0.5:
        raise ValueError("need 0 < gap < 0.5")
    if not dof > 0:
        raise ValueError("dof must be positive")
    if dof > 1e6:
        return _far(_z_mid(gap), dof, 1)
    lo = 1e-9 * math.sqrt(dof)
    at = _centre(lo, dof)
    if gap <= at:
        return gap / at * lo

    def below(t):
        return _centre(t, dof) < gap
    return _bisect(below, lo, 2 * lo)


def t_within(p, dof):
    """The t >= 0 with P(-t < T < t) = p, for
    0 < p < 0.5: t_mid() at p / 2. Half the
    least positive float is 0. Below 1e-300,
    where t is proportional to p, the
    quantile is found for 2**52 p and
    divided by 2**52."""
    if p < 1e-300:
        big = t_mid(p * 2.0 ** 51, dof)
        return big / 2.0 ** 52
    return t_mid(p / 2, dof)


@lru_cache(maxsize=None)
def t_isf(tail, dof):
    """Inverse upper tail of Student's t: the
    t >= 0 with P(T > t) = tail, for
    0 < tail <= 0.5 and dof > 0, by bisection
    on t. Checked against the closed forms
    for 1 and 2 degrees of freedom down to
    tails of 1e-100 and against tables. A
    quantile beyond 1e150 is refused. Above
    10**6 degrees of freedom the normal
    quantile is corrected by an expansion in
    the reciprocal of their number. Above a
    tail of 1/4 the quantile is found from
    the distance of the tail from 1/2."""
    if not 0 < tail <= 0.5:
        raise ValueError("need 0 < tail <= 0.5")
    if not dof > 0:
        raise ValueError("dof must be positive")
    if tail == 0.5:
        return 0.0
    if tail > 0.25:
        return t_mid(0.5 - tail, dof)
    if dof > 1e6:
        return _far(-z_ppf(tail), dof, 1)

    def below(t):
        return t_sf(t, dof) > tail
    return _bisect(below, 0.0, 1.0)


def t_ppf(p, dof):
    """Quantile of Student's t for 0 < p < 1
    and dof > 0. The lower half is found from
    the tail probability p itself and not
    from 1 - p, which would lose the digits
    of a small p."""
    if not 0 < p < 1:
        raise ValueError("p must be in (0, 1)")
    if p < 0.5:
        return -t_isf(p, dof)
    return t_isf(1 - p, dof)
