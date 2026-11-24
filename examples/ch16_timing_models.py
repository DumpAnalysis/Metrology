# replay: exact
"""Chapter 16: the timing model decides."""
import math
import random
import statistics as st

from swmetro.stats import linfit

TICK = 1.0  # ms; stamps are truncated to it
N = 20      # occurrences of the periodic message


def truncation_error(frac, n, rng):
    """Standard deviation of the error of a
    duration of whole ticks plus `frac`, read
    from two truncated stamps of one clock,
    when the start phase varies uniformly."""
    d = 2345 + frac
    errs = []
    for _ in range(n):
        t0 = rng.uniform(0, TICK)
        seen = math.floor(t0 + d) - math.floor(t0)
        errs.append(seen - d)
    return st.pstdev(errs)


def occurrences(n, period, sigma, walk, rng):
    """Times of a periodic message. With walk,
    every interval varies and the errors
    accumulate; without it, each time jitters
    about its nominal value."""
    if not walk:
        return [k * period + rng.gauss(0, sigma)
                for k in range(n)]
    out, t = [], 0.0
    for _ in range(n):
        out.append(t)
        t += period + rng.gauss(0, sigma)
    return out


def estimator_spread(walk, rng, runs=4000):
    """Spread of two estimators of the period:
    mean of the gaps, and fitted slope."""
    gaps, slopes = [], []
    for _ in range(runs):
        ts = occurrences(N, 7.5, 0.05, walk, rng)
        gaps.append((ts[-1] - ts[0]) / (N - 1))
        slopes.append(linfit(range(N), ts)[1])
    return st.stdev(gaps), st.stdev(slopes)


def main():
    rng = random.Random(16)
    print("error of a truncated duration, ticks")
    print(f"{'fraction':>9}{'simulated':>11}"
          f"{'model':>8}")
    for frac in (0.0, 0.25, 0.5, 0.6):
        sd = truncation_error(frac, 20_000, rng)
        model = math.sqrt(frac * (1 - frac))
        print(f"{frac:>9.2f}{sd:>11.3f}"
              f"{model:>8.3f}")
    mix = 1 / math.sqrt(6)
    print(f"fraction unknown: {mix:.3f}")
    print("spread of a period estimate, s")
    print(f"{'timing noise':<14}{'gaps':>8}"
          f"{'slope':>8}")
    for name, walk in (("phase jitter", False),
                       ("accumulating", True)):
        g, s = estimator_spread(walk, rng)
        print(f"{name:<14}{g:>8.4f}{s:>8.4f}")


if __name__ == "__main__":
    main()
