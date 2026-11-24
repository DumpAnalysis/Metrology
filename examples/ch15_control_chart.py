# replay: exact
"""Chapter 15: is the measurement stable?"""
import random
import statistics as st

MU, SIGMA = 100.0, 1.5  # nightly benchmark, ms


def series(rng, base=30, after=40, shift=1.5):
    """A baseline, then a shift of the mean."""
    xs = [rng.gauss(MU, SIGMA)
          for _ in range(base)]
    xs += [rng.gauss(MU + shift, SIGMA)
           for _ in range(after)]
    return xs


def xmr(baseline):
    """Center and sigma of an individuals
    chart from the mean moving range."""
    pairs = zip(baseline, baseline[1:])
    mr = [abs(b - a) for a, b in pairs]
    sigma = st.fmean(mr) / 1.128
    return st.fmean(baseline), sigma


def shewhart(zs):
    """Index of the first point beyond 3 sigma."""
    for i, z in enumerate(zs):
        if abs(z) > 3:
            return i
    return None


def cusum(zs, k=0.5, h=5.0):
    """Index of the first CUSUM signal."""
    hi = lo = 0.0
    for i, z in enumerate(zs):
        hi = max(0.0, hi + z - k)
        lo = max(0.0, lo - z - k)
        if max(hi, lo) > h:
            return i
    return None


def two_in_a_row(zs, limit=2.0):
    """Index of the second of two consecutive
    points beyond `limit` on the same side:
    a signal that waits for confirmation."""
    prev = 0
    for i, z in enumerate(zs):
        side = (z > limit) - (z < -limit)
        if side and side == prev:
            return i
        prev = side
    return None


def arl(chart, shift, rng, runs=1000):
    """Average run length: points until the
    first signal, for a mean off by `shift`
    standard deviations."""
    total = 0
    for _ in range(runs):
        zs = (shift + rng.gauss(0, 1)
              for _ in range(100_000))
        total += chart(zs) + 1
    return total / runs


def main():
    rng = random.Random(15)
    xs = series(rng)
    center, sigma = xmr(xs[:30])
    lo = center - 3 * sigma
    hi = center + 3 * sigma
    print(f"center {center:.2f} ms")
    print(f"sigma  {sigma:.2f} ms")
    print(f"limits [{lo:.2f}, {hi:.2f}] ms")
    zs = [(x - center) / sigma for x in xs[30:]]
    print("points from the shift to a signal:")
    for name, chart in (("Shewhart", shewhart),
                        ("CUSUM", cusum)):
        i = chart(zs)
        text = "none in 40"
        if i is not None:
            text = str(i + 1)
        print(f"  {name:<9}{text}")
    print(f"{'ARL, shift':<12}{'0':>6}{'1 s':>7}"
          f"{'2 s':>7}")
    charts = (("Shewhart", shewhart),
              ("CUSUM", cusum),
              ("2 in a row", two_in_a_row))
    for name, chart in charts:
        cells = "".join(
            f"{arl(chart, d, rng):>7.1f}"
            for d in (0, 1, 2))
        print(f"{name:<11}{cells}")


if __name__ == "__main__":
    main()
