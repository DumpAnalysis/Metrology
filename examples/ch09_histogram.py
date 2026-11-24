# replay: exact
"""Chapter 9: quantiles from buckets."""
import bisect
import math
import random

from swmetro.stats import exact_q, known, quantile
from swmetro.stats import lower_rank, order_stat

BOUNDS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25,
          0.5, 1.0, 2.5, 5.0, 10.0]  # seconds
TOP = math.log(math.nextafter(math.inf, 0))
LEAST = 2.0 ** -1022  # the least normal double


def bucket_quantile(xs, q, bounds=BOUNDS):
    """Quantile from cumulative bucket counts,
    interpolated linearly inside the bucket,
    as Prometheus histogram_quantile does.
    Without observations the result is NaN.
    When the rank falls in the open-ended
    last bucket the last finite bound is
    returned, and the quantile is only known
    to exceed it. The first bucket begins at
    0: the values pass known(), and one below
    0 is refused. For q = 0 the result is the
    lower end of the first bucket that holds
    a value. The bounds are finite, rise, and
    begin at 0 or above it."""
    bounds = list(bounds)
    ends = [*bounds, math.inf]
    rise = all(a < b for a, b
               in zip(ends, ends[1:]))
    if not (bounds and rise
            and bounds[0] >= 0):
        raise ValueError("no rising bounds >= 0")
    xs = known(xs)
    if not xs:
        return math.nan
    if min(xs) < 0:
        raise ValueError("need values >= 0")
    counts = [0] * (len(bounds) + 1)
    for x in xs:
        counts[bisect.bisect_left(bounds, x)] += 1
    rank, seen = exact_q(q) * len(xs), 0
    for i, c in enumerate(counts):
        if c and seen + c >= rank:
            if i == len(bounds):
                break
            lo = bounds[i - 1] if i else 0.0
            width = bounds[i] - lo
            part = float((rank - seen) / c)
            return lo + width * part
        seen += c
    return bounds[-1]


class LogSketch:
    """Buckets with relative accuracy alpha:
    bucket i covers (g**(i-1), g**i]. For
    positive, finite values only. A quantile
    is the lower order statistic at rank
    floor(q * (n - 1)), within alpha, for an
    alpha above 0 and below 1 that leaves g
    above 1 as a float. The value reported
    for a bucket, g**i * (1 - alpha), is
    formed from its logarithm and held at
    the largest float. Within alpha is the
    guarantee in real numbers, for the
    relative error |reported - x| / x. In
    the tests the floats of this class kept
    it to alpha + 1e-12 for normal doubles:
    a tolerance observed, no derived bound.
    Below the normal doubles add() refuses."""

    def __init__(self, alpha):
        if not 0 < alpha < 1:
            raise ValueError("need 0 < alpha < 1")
        self.g = (1 + alpha) / (1 - alpha)
        if self.g == 1:
            raise ValueError("alpha too small")
        self.lg = (math.log1p(alpha)
                   - math.log1p(-alpha))
        self.low = math.log1p(-alpha)
        self.counts, self.n = {}, 0

    def add(self, x):
        known([x])
        if not LEAST <= x < math.inf:
            raise ValueError(
                "need 2**-1022 <= x < inf")
        i = math.ceil(math.log(x) / self.lg)
        self.counts[i] = self.counts.get(i, 0) + 1
        self.n += 1

    def merge(self, other):
        if other.lg != self.lg:
            raise ValueError("different accuracy")
        mine = self.counts
        for i, c in other.counts.items():
            mine[i] = mine.get(i, 0) + c
        self.n += other.n

    def quantile(self, q):
        rank, seen = lower_rank(q, self.n), 0
        for i in sorted(self.counts):
            seen += self.counts[i]
            if seen > rank:
                e = i * self.lg + self.low
                return math.exp(min(e, TOP))


def latencies(n, median, sigma, rng):
    """Log-normal latencies, in seconds."""
    mu = math.log(median)
    return [rng.lognormvariate(mu, sigma)
            for _ in range(n)]


def sketch_of(xs, alpha=0.01):
    sk = LogSketch(alpha)
    for x in xs:
        sk.add(x)
    return sk


def compare():
    rng = random.Random(9)
    xs = latencies(100_000, 0.08, 0.7, rng)
    xs.sort()
    sk = sketch_of(xs)
    print(f"{'q':<6}{'interp.':>8}{'order':>8}"
          f"{'buckets':>9}{'sketch':>8}   s")
    for q in (0.5, 0.9, 0.99, 0.999):
        print(f"{q:<6}{quantile(xs, q):>8.4f}"
              f"{order_stat(xs, q):>8.4f}"
              f"{bucket_quantile(xs, q):>9.4f}"
              f"{sk.quantile(q):>8.4f}")
    two = [1.0, 1000.0]
    print("two values, 1 and 1000, q = 0.5:")
    rows = (("interpolated", quantile(two, 0.5)),
            ("order statistic",
             order_stat(two, 0.5)),
            ("sketch",
             sketch_of(two).quantile(0.5)))
    for name, value in rows:
        print(f"  {name:<16}{value:>7.2f}")


def averaging():
    """Three instances; the slow one serves
    a tenth of the traffic."""
    rng = random.Random(9)
    fleet = [latencies(45_000, 0.05, 0.4, rng),
             latencies(45_000, 0.05, 0.4, rng),
             latencies(10_000, 0.40, 0.4, rng)]
    p99 = [quantile(sorted(xs), 0.99)
           for xs in fleet]
    merged = sorted(x for xs in fleet for x in xs)
    total = LogSketch(0.01)
    for xs in fleet:
        total.merge(sketch_of(xs))
    text = "  ".join(f"{p:.3f}" for p in p99)
    print(f"p99 by instance  {text}")
    print(f"mean of the p99s {sum(p99) / 3:.3f}")
    print(f"p99 of all data  "
          f"{quantile(merged, 0.99):.3f}")
    print(f"merged sketches  "
          f"{total.quantile(0.99):.3f}")


def main():
    compare()
    averaging()


if __name__ == "__main__":
    main()
