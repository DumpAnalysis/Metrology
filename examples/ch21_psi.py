# replay: exact
"""Chapter 21: a drift score needs a null."""
import math
import random
import statistics as st

from swmetro.dist import phi, z_ppf
from swmetro.stats import as_count, quantile

B = 10            # bins fixed in advance
CHI2_95 = 16.919  # chi-square, 9 dof, 0.95


def psi(ref, cur):
    """Population stability index from two
    lists of counts in the same bins, whole
    numbers that are not negative. Half a
    count is added to every bin, so that an
    empty bin gives no infinite term, and
    half the number of bins to each total,
    so that the shares of each sample still
    sum to one."""
    ref = [as_count(r) for r in ref]
    cur = [as_count(c) for c in cur]
    bins = len(ref)
    if not bins or len(cur) != bins:
        raise ValueError("need the same bins")
    n, m = sum(ref), sum(cur)
    if not (n and m):
        raise ValueError("empty sample")
    out = 0.0
    for r, c in zip(ref, cur):
        p = (r + 0.5) / (n + 0.5 * bins)
        q = (c + 0.5) / (m + 0.5 * bins)
        out += (q - p) * math.log(q / p)
    return out


def counts(n, weights, rng):
    """Bin counts of n draws."""
    draw = rng.choices(range(B), weights, k=n)
    return [draw.count(i) for i in range(B)]


def shifted(delta):
    """Bin probabilities when the mean moves
    by delta standard deviations."""
    edges = [z_ppf(i / B) for i in range(1, B)]
    cdf = [0.0] + [phi(e - delta) for e in edges]
    cdf.append(1.0)
    return [b - a for a, b in zip(cdf, cdf[1:])]


def simulate(n, weights, rng, reps=300):
    """Sorted PSI values of `reps` pairs of
    samples of size n."""
    out = [psi(counts(n, None, rng),
               counts(n, weights, rng))
           for _ in range(reps)]
    return sorted(out)


def main():
    rng = random.Random(21)
    print("no drift at all; the columns > 0.1")
    print("and > crit. count pairs out of 1000:")
    print(f"{'n':>6}{'mean':>7}{'model':>7}"
          f"{'> 0.1':>8}{'crit.':>7}"
          f"{'> crit.':>9}")
    for n in (100, 200, 500, 2000):
        xs = simulate(n, None, rng, 1000)
        crit = CHI2_95 * 2 / n
        over = sum(x > 0.1 for x in xs)
        past = sum(x > crit for x in xs)
        model = (B - 1) * 2 / n
        print(f"{n:>6}{st.fmean(xs):>7.3f}"
              f"{model:>7.3f}{over:>8}"
              f"{crit:>7.3f}{past:>9}")
    print("mean moved by 0.2 sd:")
    for n in (500, 5000):
        xs = simulate(n, shifted(0.2), rng)
        crit = CHI2_95 * 2 / n
        hit = sum(x > crit for x in xs)
        print(f"{n:>6}  median PSI "
              f"{quantile(xs, 0.5):.3f}, above "
              f"crit. {hit} of {len(xs)}")


if __name__ == "__main__":
    main()
