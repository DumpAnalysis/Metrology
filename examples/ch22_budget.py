# replay: exact
"""Chapter 22: one percentile, two records."""
import random
import statistics as st
from fractions import Fraction

from examples import ch09_histogram as h9
from swmetro import Budget, Result, u_rect
from swmetro.stats import bootstrap, known, wilson
from swmetro.stats import ecdf_quantile, ecdf_rank

LIMIT = 200.0  # objective: p95 <= 200 ms
PROBE = 0.9    # blank indication of the probe, ms
STEP = 1.0     # durations are stored in whole ms
COARSE = [50, 100, 250, 500]
FITTED = [50, 100, 200, 250, 500]


def record(rng):
    """2000 durations as the probe stores them:
    its overhead included, rounded to 1 ms."""
    xs = h9.latencies(2000, 0.120, 0.25, rng)
    return [float(round(1000 * x + PROBE))
            for x in xs]


def p95(xs):
    """The p95 of the objective, by the inverse
    empirical CDF convention: the smallest
    recorded value with at least 95 % of the
    values at or below it. The durations
    enter through known() of the package,
    which refuses one that is missing and
    whatever is no number."""
    return ecdf_quantile(sorted(known(xs)), 0.95)


def from_raw(ms):
    """Target: the mean, over runs under
    these conditions, of the p95 of a run,
    estimated from one run. Rounding commutes
    with order statistics, so the stored step
    bounds the error it causes by half a
    step; the bound is modeled rectangular.
    'Between runs' is a variance component
    with the within-run sampling removed,
    taken from an earlier study of 8 runs.
    The readings are taken as the exact
    numbers they are, a decimal as a decimal,
    and the correction as nine tenths."""
    ms = known(ms)
    reps = bootstrap(ms, p95, 500, seed=22)
    spread = st.stdev(Fraction(r) for r in reps)
    bud = Budget("ms")
    bud.add("sampling in the run", "A", spread)
    bud.add("between runs", "A", 2.5, 7)
    bud.add("probe correction", "B", 0.30)
    bud.add("stored step", "B", u_rect(STEP / 2))
    probe = Fraction(str(PROBE))
    value = float(Fraction(p95(ms)) - probe)
    return bud, Result(value, bud.u_c(), "ms",
                       bud.dof())


def bracket(xs, bounds, q=0.95):
    """Bucket that holds the q-quantile of the
    record, from cumulative counts alone: the
    first bucket whose cumulative count
    reaches the rank of the inverse-ECDF
    quantile. A quantile defined by another
    convention need not lie in it. The first
    bucket has no lower end: a corrected
    indication can be zero or below it. The
    ends are the bounds as they were given:
    turned into floats, a bound next to the
    limit can become the limit itself."""
    xs, bounds = known(xs), tuple(bounds)
    pairs = zip([float("-inf"), *bounds], bounds)
    if not all(a < b for a, b in pairs):
        raise ValueError("bounds do not rise")
    need = ecdf_rank(q, len(xs))
    lo = float("-inf")
    for b in bounds:
        if sum(x <= b for x in xs) >= need:
            return lo, b
        lo = b
    return lo, float("inf")


def verdict(lo, hi):
    """For the record at hand. Unresolved: the
    counts do not say, and finer ones would."""
    if hi <= LIMIT:
        return "met"
    if lo >= LIMIT:
        return "missed"
    return "unresolved"


def main():
    ms = known(record(random.Random(22)))
    bud, r = from_raw(ms)
    print("from the raw durations")
    print(bud.table())
    print(f"result {r.report()}")
    p = r.prob_below(LIMIT)
    print(f"P(mean run p95 <= {LIMIT:.0f} ms)"
          f" = {p:.3f}")
    xs = [x - PROBE for x in ms]
    print("from bucket counts")
    for bounds in (COARSE, FITTED):
        lo, hi = bracket(xs, bounds)
        print(f"bounds {bounds}")
        print(f"  p95 of the record in ({lo:.0f},"
              f" {hi:.0f}] ms: {verdict(lo, hi)}")
    lo, hi = bracket(xs, COARSE)
    est = h9.bucket_quantile(xs, 0.95, COARSE)
    share = (LIMIT - lo) / (hi - lo)
    print(f"interpolated in the wide bucket: "
          f"{est:.0f} ms")
    print(f"if rectangular there (assumed): "
          f"P = {share:.2f}")
    k = sum(x <= LIMIT for x in xs)
    a, b = wilson(k, len(xs))
    frac = k / len(xs)
    print(f"share <= {LIMIT:.0f} ms: {frac:.3f},"
          f" 95 % [{a:.3f}, {b:.3f}]")


if __name__ == "__main__":
    main()
