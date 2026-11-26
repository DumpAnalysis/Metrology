# replay: exact
"""Chapter 10: a report that drops timeouts."""
import random

from swmetro.stats import ecdf_quantile, known
from swmetro.stats import ecdf_rank

TIMEOUT = 1000.0  # ms; the client gives up here


def week(n, stuck, median, rng):
    """n requests, as the client records them:
    a latency, or None for a timeout. A share
    `stuck` never answers at all."""
    out = []
    for _ in range(n):
        if rng.random() < stuck:
            out.append(None)
        else:
            z = rng.lognormvariate(0, 0.5)
            x = median * z
            out.append(x if x < TIMEOUT else None)
    return out


def completions(xs):
    """Completions in order, all below TIMEOUT."""
    done = sorted(known(x for x in xs
                        if x is not None))
    if not all(0 <= x < TIMEOUT for x in done):
        raise ValueError("need 0 <= x < TIMEOUT")
    return done


def p95_completed(xs):
    """p95 of the requests that completed."""
    return ecdf_quantile(completions(xs), 0.95)


def p95_all(xs):
    """p95 of all requests, by the inverse
    empirical CDF convention. A timeout is
    right-censored: at least TIMEOUT, and so
    ranked after every completion. Returns
    the value and a flag, True when the order
    statistic of that rank is censored and
    the value only a bound. A NaN is no
    timeout: completions() refuses it."""
    done = completions(xs := list(xs))
    k = ecdf_rank(0.95, len(xs))
    if k <= len(done):
        return done[k - 1], False
    return TIMEOUT, True


def main():
    rng = random.Random(10)
    w1 = week(10_000, 0.00, 200, rng)
    w2 = week(10_000, 0.08, 150, rng)
    print(f"{'':<8}{'timeouts':>9}"
          f"{'p95 done':>10}{'p95 all':>10}  ms")
    for name, w in ("week 1", w1), ("week 2", w2):
        lost = sum(x is None for x in w)
        value, bound = p95_all(w)
        text = f"{value:.0f}"
        if bound:
            text = ">= " + text
        print(f"{name:<8}{lost:>9}"
              f"{p95_completed(w):>10.0f}"
              f"{text:>10}")


if __name__ == "__main__":
    main()
