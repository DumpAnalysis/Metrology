# replay: exact
"""Chapter 2: one record, three targets."""
import random

from swmetro.stats import ecdf_quantile
from swmetro.stats import enclosed, quantile_ranks


def record(n, rng):
    """n latencies, in ms, of a process that
    does not change while it is observed."""
    xs = [rng.lognormvariate(4.6, 0.4)
          for _ in range(n)]
    return sorted(xs)


def main():
    xs = record(400, random.Random(2))
    n = len(xs)
    # 1. A property of the captured record.
    p90 = ecdf_quantile(xs, 0.9)
    # 2. A parameter of the process, assuming
    #    independent draws from it. Ranks are
    #    whole numbers, and the pair that is
    #    used covers a little more than 95 %.
    lo, hi, held = quantile_ranks(n, 0.9)
    # 3. One future observation. Averaged over
    #    samples and over the new draw, it
    #    falls below the k-th of n ordered
    #    values with probability k / (n + 1).
    a, b = 10, 391
    cover = 100 * (b - a) / (n + 1)
    print(f"record of {n} latencies, ms")
    print(f"p90 of the record        {p90:7.1f}")
    low, top = enclosed(xs[lo - 1], xs[hi - 1], 1)
    print(f"p90 of the process, 95 % "
          f"[{low}, {top}]")
    print(f"  ranks {lo} and {hi}, "
          f"coverage {100 * held:.2f} %")
    print(f"next latency, ranks {a} to {b}")
    low, top = enclosed(xs[a - 1], xs[b - 1], 1)
    print(f"{'':25}[{low}, {top}]")
    print(f"  coverage {b - a}/{n + 1}"
          f" = {cover:.1f} %")


if __name__ == "__main__":
    main()
