# replay: exact
"""Chapter 22: which p95, across runs?"""
import math
import random
import statistics as st

from swmetro.stats import ecdf_quantile

S_B = 2.5    # sd of the run p95 between runs, ms
S_W = 1.72   # sampling sd within a run, ms
LEVEL = 182.0  # long-run mean of the run p95


def observed(rng):
    """The p95 that one run reports: its own
    level plus the sampling error of its
    2000 requests."""
    level = LEVEL + rng.gauss(0, S_B)
    return level + rng.gauss(0, S_W)


def spread(m, rng, reps=20_000):
    """Simulated standard deviations of the
    error of the mean of m runs, taken as an
    estimate of (a) the long-run mean and
    (b) the p95 that the next run reports."""
    a, b = [], []
    for _ in range(reps):
        est = st.fmean(observed(rng)
                       for _ in range(m))
        a.append(est - LEVEL)
        b.append(est - observed(rng))
    return st.pstdev(a), st.pstdev(b)


def main():
    rng = random.Random(22)
    s = math.hypot(S_B, S_W)
    print("error of the mean of m runs, sd, ms")
    print(f"{'target':<16}{'m = 1':>8}"
          f"{'4':>7}{'16':>7}")
    sims = [spread(m, rng) for m in (1, 4, 16)]
    rows = (("long-run mean", 0, 0.0),
            ("next run", 1, 1.0))
    for name, i, extra in rows:
        sim = "".join(f"{x[i]:>7.2f}"
                      for x in sims)
        mod = ""
        for m in (1, 4, 16):
            v = s * math.sqrt(extra + 1 / m)
            mod += f"{v:>7.2f}"
        print(f"{name:<16} {sim}")
        print(f"{'  model':<16} {mod}")
    # Ten pairs of runs, each run with requests
    # of one latency only: (latency, count).
    cases = (
        ("all 200 ms", 200.0, 50, 200.0, 50),
        ("100 and 300 ms", 100.0, 50,
         300.0, 50),
        ("300 ms runs short", 100.0, 98,
         300.0, 2),
    )
    print("mean of run p95s, with runs weighted")
    print("equally or by requests; pooled p95")
    print(f"{'runs':<18}{'equal':>7}"
          f"{'by requests':>13}{'pooled':>8}")
    for name, x, n, y, m in cases:
        runs = [[x] * n, [y] * m] * 10
        each = [ecdf_quantile(sorted(r), 0.95)
                for r in runs]
        sizes = [len(r) for r in runs]
        total = sum(p * k
                    for p, k in zip(each, sizes))
        weighted = total / sum(sizes)
        pooled = sorted(v for r in runs
                        for v in r)
        both = ecdf_quantile(pooled, 0.95)
        print(f"{name:<18}{st.fmean(each):>7.0f}"
              f"{weighted:>13.0f}{both:>8.0f}")


if __name__ == "__main__":
    main()
