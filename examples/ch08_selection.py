# replay: exact
"""Chapter 8: sampling that depends on the
outcome."""
import random
import statistics as st
from fractions import Fraction

from swmetro.stats import weighted_quantile

SLOW = 300.0  # ms; slower traces are all kept
KEEP = Fraction(1, 50)  # of the other traces
ALL, NONE = Fraction(1), Fraction(0)


def requests(n, rng):
    """(latency in ms, is a cache hit). Cache
    hits, 30 % of requests, take about 5 ms
    and are never traced."""
    out = []
    for _ in range(n):
        if rng.random() < 0.30:
            x = rng.lognormvariate(1.6, 0.3)
            out.append((x, True))
        else:
            x = rng.lognormvariate(4.4, 0.6)
            out.append((x, False))
    return out


def kept(rows, rng):
    """Traces that survive, each with its
    inclusion probability, a fraction: its
    reciprocal is the weight of the trace,
    and it has to be exact."""
    out, some = [], float(KEEP)
    for x, hit in rows:
        p, bar = KEEP, some
        if hit:
            p, bar = NONE, 0.0
        elif x > SLOW:
            p, bar = ALL, 1.0
        if rng.random() < bar:
            out.append((x, p))
    return out


def summary(pairs):
    """Weighted mean, inverse-ECDF p95 and
    share above SLOW of (value, weight). All
    sums are sums of fractions: the mean is
    rounded once, and nothing changes when
    every weight is multiplied exactly by
    one number; see weighted_quantile()."""
    pairs = [(x, Fraction(w)) for x, w in pairs]
    total = sum(w for _, w in pairs)
    top = sum(Fraction(x) * w for x, w in pairs)
    slow = sum(w for x, w in pairs if x > SLOW)
    p95 = weighted_quantile(pairs, 0.95)
    return (float(top / total), p95,
            100 * float(slow / total))


def redrawn(rows, rng, reps=400):
    """Weighted summaries over repeated
    retention draws from the same requests:
    the variation that selection alone
    causes, for this fixed record."""
    out = []
    for _ in range(reps):
        sample = kept(rows, rng)
        out.append(summary([(x, 1 / p)
                            for x, p in sample]))
    return list(zip(*out))


def main():
    rng = random.Random(8)
    rows = requests(40_000, rng)
    traced = [r for r in rows if not r[1]]
    sample = kept(rows, rng)
    print(f"{len(rows)} requests,"
          f" {len(sample)} traces kept")
    print(f"{'':<20}{'mean':>7}{'p95':>7}"
          f"{'slow':>8}")
    table = (
        ("traceable requests",
         [(x, 1) for x, _ in traced]),
        ("kept, unweighted",
         [(x, 1) for x, _ in sample]),
        ("kept, weighted 1/p",
         [(x, 1 / p) for x, p in sample]),
        ("all requests",
         [(x, 1) for x, _ in rows]),
    )
    for name, pairs in table:
        m, q, s = summary(pairs)
        print(f"{name:<20}{m:>7.1f}{q:>7.1f}"
              f"{s:>6.1f} %")
    print("weighted, over 400 retention draws")
    cols = redrawn(rows, rng)
    for name, f in (("  average", st.fmean),
                    ("  sd", st.stdev)):
        m, q, s = (f(c) for c in cols)
        print(f"{name:<20}{m:>7.1f}{q:>7.1f}"
              f"{s:>6.2f} %")


if __name__ == "__main__":
    main()
