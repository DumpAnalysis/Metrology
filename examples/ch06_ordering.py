# replay: exact
"""Chapter 6: can timestamps order two events?"""
import math
import random

from swmetro.dist import phi

SIGMA = 2.0  # standard unc. of each clock, ms


def misordered(gap_ms, n, rng):
    """Share of trials in which the stamps of
    two hosts reverse the true order."""
    bad = 0
    for _ in range(n):
        t1 = rng.gauss(0, SIGMA)
        t2 = gap_ms + rng.gauss(0, SIGMA)
        bad += t2 < t1
    return bad / n


def reversed_of(n, p, shared):
    """Chance of each number, 0 to n, of
    reversed pairs among n with one true gap,
    each reversed with the chance p. With
    clock errors drawn anew for every pair
    the number is binomial. With one offset
    kept by each clock for the capture, and
    no other error, all the pairs go one way,
    and of no pairs none is reversed."""
    if shared and n:
        return [1 - p] + [0.0] * (n - 1) + [p]
    return [math.comb(n, k) * p ** k
            * (1 - p) ** (n - k)
            for k in range(n + 1)]


def ten_pairs(p):
    print("ten pairs 1 ms apart, reversed:")
    print(f"{'':<19}{'none':>9}{'all ten':>12}"
          f"{'mean':>6}")
    for name, shared in (("errors drawn anew", 0),
                         ("one offset kept", 1)):
        ps = reversed_of(10, p, shared)
        mean = sum(k * q
                   for k, q in enumerate(ps))
        print(f"  {name:<17}{100 * ps[0]:>7.2f} %"
              f"{100 * ps[10]:>10.5f} %"
              f"{mean:>6.2f}")


def main():
    rng = random.Random(6)
    u_gap = SIGMA * math.sqrt(2)
    print(f"u(gap) = {u_gap:.2f} ms")
    print(f"{'true gap':>9}{'simulated':>11}"
          f"{'model':>11}")
    for gap in (0.5, 1.0, 2.0, 5.0, 10.0):
        sim = 100 * misordered(gap, 50_000, rng)
        model = 100 * phi(-gap / u_gap)
        print(f"{gap:>6.1f} ms{sim:>9.3f} %"
              f"{model:>9.3f} %")
    need = 1.645 * u_gap
    print(f"gap for 95 % order: {need:.1f} ms")
    ten_pairs(phi(-1.0 / u_gap))


if __name__ == "__main__":
    main()
