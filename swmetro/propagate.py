"""Propagation of uncertainty: the GUM law
(first order) and the Monte Carlo method."""
import math
import random
import statistics as st
from fractions import Fraction

from .core import resolved
from .stats import _floor, _root, exact_q, mean
from .stats import native


def correlations(corr, n):
    """The correlation matrix of n inputs from
    {(i, j): r}, as exact fractions: 1 on the
    diagonal, r for a pair, entered once or
    alike in both orders, and 0 for a pair
    that is left out, which is an entry like
    any other. A float is read as the decimal
    that Python prints for it, so that 0.62
    is 62/100. The coefficients are checked
    together: a set from which some
    combination of the inputs would get a
    negative variance is refused, whichever
    combination is wanted. The matrix is
    reduced one input at a time, and what is
    left on its diagonal may not fall below
    zero. Zero is allowed: two readings can
    share all of their error. On that edge
    the rounding of one coefficient can
    decide, and the joint measurements are
    what to keep."""
    m = [[Fraction(i == j) for j in range(n)]
         for i in range(n)]
    seen = set()
    for (i, j), r in (corr or {}).items():
        if not (0 <= i < n and 0 <= j < n):
            raise ValueError("no such input")
        if isinstance(r, float):
            r = repr(r)
        r = Fraction(r)
        if (i, j) in seen and m[i][j] != r:
            raise ValueError("two values for "
                             "one pair")
        if i == j and r != 1:
            raise ValueError("an input with "
                             "itself has 1")
        seen |= {(i, j), (j, i)}
        m[i][j] = m[j][i] = r
    rest = [row[:] for row in m]
    left = list(range(n)) if seen else []
    while left:
        k = max(left, key=lambda i: rest[i][i])
        left.remove(k)
        d = rest[k][k]
        tied = any(rest[k][j] for j in left)
        if d < 0 or not d and tied:
            raise ValueError("the correlations "
                             "cannot hold "
                             "together")
        if not d:
            continue
        for i in left:
            c = rest[i][k] / d
            for j in left:
                rest[i][j] -= c * rest[k][j]
    return m


def lpu(f, xs, us, corr=None):
    """Law of propagation of uncertainty.

    f     measurement function of a list
    xs    estimates of the input quantities
    us    their standard uncertainties, one
          each, finite and not below zero
    corr  {(i, j): r} correlation coefficients,
          checked by correlations() before
          anything is propagated

    Returns y, u_c(y) and the list of c_i * u_i,
    with each c_i * u_i taken as half the change
    of f between x_i - u_i and x_i + u_i. The
    two points are formed as exact fractions,
    and f is called with fractions there. As
    a float, an estimate can be too large to
    show its uncertainty: a time stamp that
    counts nanoseconds from 1970, with 1 ns
    added, is the same float, and the
    difference of two such stamps would come
    out with an uncertainty of zero. The
    squares and the products of the c_i * u_i
    are summed as fractions as well, and the
    root is rounded once: what two fully
    correlated contributions of opposite
    sign leave can lie far below the last
    digit of their squares. The estimates and
    the uncertainties are taken as native()
    gives them.
    """
    xs = [native(x) for x in xs]
    us = [native(u) for u in us]
    if len(us) != len(xs):
        raise ValueError("one u for each x")
    if not all(0 <= u < math.inf for u in us):
        raise ValueError("u must be >= 0")
    m = correlations(corr, len(xs))
    y = f(xs)
    exact = [Fraction(x) for x in xs]
    cu = []
    for i, u in enumerate(us):
        hi, lo = list(exact), list(exact)
        hi[i] += Fraction(u)
        lo[i] -= Fraction(u)
        cu.append(Fraction(f(hi) - f(lo)) / 2)
    var = sum(m[i][j] * a * b
              for i, a in enumerate(cu)
              for j, b in enumerate(cu))
    return y, _root(var), [float(c) for c in cu]


def normal(mu, sigma):
    """Sampler for a normal distribution. The
    draws are floats, and a spread that a
    float of the size of mu cannot show is
    refused, see resolved(). With no spread
    every draw is mu as it was given."""
    if sigma == 0:
        return lambda rng: mu
    resolved(mu, sigma)
    return lambda rng: rng.gauss(mu, sigma)


def rect(lo, hi):
    """Sampler for a rectangular distribution,
    with the same refusal, and with every
    draw lo where hi is lo."""
    if hi == lo:
        return lambda rng: lo
    resolved(max(abs(lo), abs(hi)), hi - lo)
    return lambda rng: rng.uniform(lo, hi)


def monte_carlo(f, samplers, n=100_000, seed=1):
    """Propagate distributions: draw the inputs,
    evaluate f, return the sorted outputs."""
    rng = random.Random(seed)
    samplers = list(samplers)
    ys = [f([s(rng) for s in samplers])
          for _ in range(n)]
    ys.sort()
    return ys


def summarize(ys, p=0.95):
    """Mean, standard deviation and the
    probabilistically symmetric coverage
    interval of sorted Monte Carlo outputs.
    The limits are the draws of the ranks
    r and r + q of Supplement 1 to the GUM,
    7.7.2, counted from 1: q is p * n, to the
    nearest whole number, and r is half of
    n - q, rounded up. p is read as the
    decimal that it is written as, so that
    the ranks do not depend on its binary
    form. The q steps between the two draws
    hold q / (n + 1) of a continuous
    distribution on average, and the range
    of ten draws 9/11 of it, whatever p was
    asked for. Where p * n rounds to n, the
    draws are too few for p and are
    refused."""
    ys = [native(y) for y in ys]
    n = len(ys)
    q = _floor(exact_q(p) * n + Fraction(1, 2))
    if q >= n:
        raise ValueError("more draws needed")
    r = (n - q + 1) // 2
    lo, hi = ys[r - 1], ys[r + q - 1]
    return mean(ys), st.stdev(ys), lo, hi
