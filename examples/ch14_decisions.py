# replay: exact
"""Chapter 14: decisions under uncertainty."""
import math
import random
import statistics as st

from swmetro.dist import phi, t_ppf, z_ppf
from swmetro.stats import downward, upward

LIMIT = 200.0  # upper specification limit, ms
U = 3.0        # standard uncertainty, ms
RISK = 0.05    # tolerated risk of a wrong call


def conformity():
    """Probability that the measurand is below
    the limit, given the measured value."""
    guard = z_ppf(1 - RISK) * U
    low = downward(LIMIT - guard, 2)
    high = upward(LIMIT + guard, 2)
    print(f"accept up to {low},"
          f" reject above {high} ms")
    print(f"{'measured':>9}{'P(conform)':>12}"
          "  decision")
    for y in (188, 192, 195, 197, 200, 203, 206):
        p = phi((LIMIT - y) / U)
        if p >= 1 - RISK:
            verdict = "accept"
        elif p < RISK:
            verdict = "reject"
        else:
            verdict = "undecided"
        print(f"{y:>6} ms{p:>12.3f}  {verdict}")


def experiment(n, shift, rng):
    """Two arms of n runs each; times in % of
    the baseline mean, run-to-run sd 1.5."""
    a = [rng.gauss(100, 1.5) for _ in range(n)]
    b = [rng.gauss(100 + shift, 1.5)
         for _ in range(n)]
    diff = st.fmean(b) - st.fmean(a)
    var = st.variance(a) + st.variance(b)
    return diff, math.sqrt(var / n)


def flag_rates(n, shift, rng, trials=2000):
    """How often each rule reports a slowdown:
    'more than 1 %' against 'more than the
    one-sided 95 % limit of the difference'."""
    naive = aware = 0
    k = t_ppf(0.95, 2 * n - 2)
    for _ in range(trials):
        diff, u = experiment(n, shift, rng)
        naive += diff > 1.0
        aware += diff > k * u
    return naive / trials, aware / trials


def regressions():
    rng = random.Random(14)
    print(f"{'runs':>4}{'true shift':>12}"
          f"{'fixed 1 %':>11}{'k * u':>8}")
    for n in (5, 20):
        for shift in (0.0, 2.0):
            a, b = flag_rates(n, shift, rng)
            print(f"{n:>4}{shift:>10.0f} %"
                  f"{a:>11.3f}{b:>8.3f}")


def main():
    conformity()
    regressions()


if __name__ == "__main__":
    main()
