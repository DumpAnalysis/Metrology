# replay: exact
"""Chapter 5: propagation of uncertainty."""
import math
from fractions import Fraction

from swmetro import u_rect
from swmetro.dist import z_ppf
from swmetro.propagate import lpu, monte_carlo
from swmetro.propagate import normal, rect
from swmetro.propagate import summarize
from swmetro.stats import enclosed, fraction

N = 200_000  # Monte Carlo trials
RATE = Fraction(1, 10 ** 9)  # u of a clock rate


def speedup(x):
    """S = T_old / T_new."""
    return x[0] / x[1]


def rate(x):
    """Requests per second: R = N / T."""
    return x[0] / x[1]


def lognormal_ratio(m1, s1, m2, s2):
    """Mean, standard deviation and central
    95 % interval of X / Y for independent
    lognormal X and Y with the given means
    and standard deviations. The ratio is
    lognormal as well, and its moments
    exist."""
    v1 = math.log1p((s1 / m1) ** 2)
    v2 = math.log1p((s2 / m2) ** 2)
    mu = math.log(m1 / m2) - (v1 - v2) / 2
    sd = math.sqrt(v1 + v2)
    mean = math.exp(mu + sd * sd / 2)
    u = mean * math.sqrt(math.expm1(sd * sd))
    z = z_ppf(0.975)
    return (mean, u, math.exp(mu - z * sd),
            math.exp(mu + z * sd))


def shared_factor():
    """Two durations, 100 ms and 80 ms, read
    with a clock that runs by a factor of
    1.0 with a standard uncertainty of 0.1.
    First as two inputs with a correlation
    of 1, then with the factor as the one
    input of a model in which it cancels."""
    xs, us = [100.0, 80.0], [10.0, 8.0]
    y, uc, cu = lpu(speedup, xs, us,
                    {(0, 1): 1.0})
    print("a factor of 1.0(1) on both durations:")
    print(f"r = 1.0  S = {y:.4f}  u_c = {uc:.4f}")
    print(f"c*u(T_old) = {cu[0]:+.4f}  "
          f"c*u(T_new) = {cu[1]:+.4f}")

    def model(g):
        return speedup([100 * g[0], 80 * g[0]])
    y, uc, _ = lpu(model, [1.0], [0.1])
    print("the factor as the input:  "
          f"u_c = {uc:.4f}")


def shared_rate(first=0, last=10 ** 9, u=RATE):
    """A shared error that does not cancel.
    Two stamps of one clock, `first` and
    `last` ns after a reading of 10**17 ns,
    share one constant error of its rate: a
    fraction of the time since the clock was
    set, with the standard uncertainty u.
    Each stamp is then uncertain by u of
    itself, with a correlation of 1, and
    their difference by u of itself: 1 ns
    for 1 s at 1e-9, wherever on the clock
    the second lies. The uncertainties of
    the stamps are kept as fractions. As
    whole nanoseconds, half a second would
    keep 0 ns or 1 ns. Returns the
    difference, its u_c with the stamps as
    the inputs, their contributions, and
    its u_c with the rate as the one input.
    The model is linear, the law is exact
    for it, and the two forms agree."""
    u = fraction(u)
    xs = [10 ** 17 + first, 10 ** 17 + last]
    us = [x * u for x in xs]
    y, uc, cu = lpu(lambda t: t[1] - t[0],
                    xs, us, {(0, 1): 1.0})

    def model(g):
        return xs[1] * g[0] - xs[0] * g[0]
    return y, uc, cu, lpu(model, [1], [u])[1]


def clock_rate():
    y, uc, cu, one = shared_rate()
    print("one constant error of the rate of a")
    print("clock, u = 1e-9; two stamps, 1 s")
    print("apart, 10**17 ns after it was set:")
    print(f"r = 1.0  D = {y} ns  "
          f"u_c = {uc:.4f} ns")
    print(f"c*u(start) = {cu[0]:+.2f} ns")
    print(f"c*u(end)   = {cu[1]:+.2f} ns")
    print("the rate as the input:  "
          f"u_c = {one:.4f} ns")
    half = shared_rate(750_000_000,
                       1_250_000_000)
    print("half a second, 0.75 s later:")
    print(f"c*u(start) = {half[2][0]:+.2f} ns")
    print(f"c*u(end)   = {half[2][1]:+.2f} ns")
    print(f"u_c = {half[1]:.4f} ns; the rate"
          " as the")
    print(f"input:  u_c = {half[3]:.4f} ns")
    limit = shared_rate(u=u_rect(float(RATE)))
    print("1e-9 as a limit, rectangular, on")
    print(f"the second:  u_c = {limit[3]:.4f} ns")


def stages():
    """Three stage durations of 10 ms, each
    with a standard uncertainty of 1 ms, and
    their sum. The first two correlate with
    0.9, and so do the last two. A set of
    correlations has to hold together: with
    nothing entered for the first and the
    third, the first less the second plus
    the third would have a variance of
    3 - 1.8 - 1.8 ms squared."""
    xs, us = [10, 10, 10], [1, 1, 1]
    corr = {(0, 1): 0.9, (1, 2): 0.9}
    print("three stages of 10(1) ms, summed;")
    print("r = 0.9 for the first two and the")
    print("last two; first and third left out:")
    try:
        lpu(sum, xs, us, corr)
    except ValueError as err:
        print(f"  {err}")
    for r in (0.62, 0.81, 1.0):
        corr[0, 2] = r
        y, uc, _ = lpu(sum, xs, us, corr)
        print(f"first and third r = {r:.2f}:  "
              f"u_c = {uc:.3f} ms")


def linear_case():
    xs, us = [842.0, 655.0], [12.0, 15.0]
    for r in (0.0, 0.6):
        corr = {(0, 1): r}
        y, uc, cu = lpu(speedup, xs, us, corr)
        print(f"r = {r:.1f}  S = {y:.4f}  "
              f"u_c = {uc:.4f}")
    print(f"c*u(T_old) = {cu[0]:+.4f}  "
          f"c*u(T_new) = {cu[1]:+.4f}")
    shared_factor()
    clock_rate()
    stages()
    # Independent normal inputs, as in the
    # line for r = 0. The figures summarize
    # the draws: a ratio with a normal
    # denominator has no mean of its own.
    draws = [normal(842, 12), normal(655, 15)]
    ys = monte_carlo(speedup, draws, N, 5)
    m, s, lo, hi = summarize(ys)
    print(f"MC, independent normal, {N} draws:")
    print(f"  sample mean {m:.4f}"
          f"  sample sd {s:.4f}")
    lo, hi = enclosed(lo, hi, 3)
    print(f"  95 % of draws  [{lo}, {hi}]")
    m, s, lo, hi = lognormal_ratio(842, 12,
                                   655, 15)
    print("exact, independent lognormal:")
    print(f"  mean {m:.4f}  sd {s:.4f}")
    lo, hi = enclosed(lo, hi, 3)
    print(f"  central 95 %  [{lo}, {hi}]")


def nonlinear_case():
    """T is known only to lie in [1.2, 2.8] s."""
    half = 0.8
    u_t = half / 3 ** 0.5
    y, uc, _ = lpu(rate, [1000, 2.0], [0, u_t])
    print(f"LPU    R = {y:.0f}  u_c = {uc:.0f}")
    lo, hi = enclosed(y - 2 * uc, y + 2 * uc, 0)
    print(f"LPU k=2  [{lo}, {hi}]")
    draws = [lambda rng: 1000, rect(1.2, 2.8)]
    ys = monte_carlo(rate, draws, N, 5)
    m, s, lo, hi = summarize(ys)
    lo, hi = enclosed(lo, hi, 0)
    print(f"MC     R = {m:.0f}  u = {s:.0f}")
    print(f"MC 95 %  [{lo}, {hi}]")
    # The same without a simulation: T is
    # uniform on [a, b], and R = 1000 / T
    # falls as T rises.
    a, b = 1.2, 2.8
    m = 1000 * math.log(b / a) / (b - a)
    s = math.sqrt(1e6 / (a * b) - m * m)
    lo, hi = (1000 / (b - (b - a) * p)
              for p in (0.025, 0.975))
    lo, hi = enclosed(lo, hi, 2)
    print(f"exact  R = {m:.2f}  u = {s:.2f}")
    print(f"exact 95 %  [{lo}, {hi}]")


def main():
    linear_case()
    nonlinear_case()


if __name__ == "__main__":
    main()
