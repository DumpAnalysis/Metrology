# replay: exact
"""Chapter 9: tail quantiles of dependent data."""
import random
import statistics as st

from swmetro.stats import block_bootstrap, upward
from swmetro.stats import bootstrap, quantile
from swmetro.stats import quantile_interval
from swmetro.stats import downward, run_length

N, BLOCK = 2000, 100


def bursty(n, rng):
    """Latencies, in ms, with bursts: a slow
    regime starts with probability 1/200 per
    request and lasts 20 requests on average,
    so neighbors in time are alike."""
    out, slow = [], False
    for _ in range(n):
        flip = 1 / 20 if slow else 1 / 200
        if rng.random() < flip:
            slow = not slow
        x = rng.lognormvariate(3.9, 0.3)
        out.append(4 * x if slow else x)
    return out


def p99(xs):
    """By linear interpolation."""
    return quantile(sorted(xs), 0.99)


def block_sd(trace, block):
    """Standard deviation of the p99 over
    moving-block bootstrap replicates."""
    reps = block_bootstrap(trace, p99, block,
                           300, 9)
    return st.stdev(reps)


def enough(q, miss):
    """Fewest independent values of which
    the largest lies above the q-quantile of
    a continuous distribution, except with a
    chance of `miss` at most: all n of them
    lie below it with the chance q ** n.
    Where no number of values is enough, as
    for q = 1, run_length() says so."""
    return run_length(q, miss)


def upper_ends():
    """What a finite upper end costs, for a
    one-sided bound at 95 % and for the
    interval with 2.5 % in each tail."""
    over = 100 * (1 - 0.999 ** 500)
    print("  the largest of them is above the")
    print(f"  p99.9 in {over:.2f} % of samples")
    print("values for an upper end at 95 %,")
    print("one-sided and with equal tails:")
    for name, q in (("p90", 0.9), ("p95", 0.95),
                    ("p99", 0.99),
                    ("p99.9", 0.999)):
        one = enough(q, 0.05)
        two = enough(q, 0.025)
        print(f"  {name:<6}{one:>6}{two:>7}")


def main():
    rng = random.Random(9)
    trace = bursty(N, rng)
    iid = st.stdev(bootstrap(trace, p99, 300, 9))
    blk = block_sd(trace, BLOCK)
    runs = [p99(bursty(N, rng))
            for _ in range(300)]
    truth = st.stdev(runs)
    rows = (("p99 of one trace", p99(trace)),
            ("sd, iid bootstrap", iid),
            ("sd, block bootstrap", blk),
            ("sd, 300 traces", truth))
    for name, ms in rows:
        print(f"{name:<21}{ms:>7.1f} ms")
    print("sd by block length:")
    for block in (10, 25, 50, 100, 200, 400):
        sd = block_sd(trace, block)
        print(f"  {block:>4}{sd:>8.1f} ms")
    calm = sorted(rng.lognormvariate(3.9, 0.3)
                  for _ in range(500))
    lo, hi = quantile_interval(calm, 0.999)
    top = "none" if hi is None else upward(hi, 1)
    print("p99.9 from 500 independent values:")
    print(f"  lower bound {downward(lo, 1)} ms,"
          f" upper bound {top}")
    upper_ends()


if __name__ == "__main__":
    main()
