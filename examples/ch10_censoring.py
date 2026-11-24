# replay: exact
"""Chapter 10: a report that drops timeouts."""
import random

from examples.ch22_budget import known
from swmetro.stats import ecdf_quantile
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


def p95_completed(xs):
    """p95 of the requests that completed."""
    done = sorted(known(x for x in xs
                        if x is not None))
    return ecdf_quantile(done, 0.95)


def p95_all(xs):
    """p95 of all requests, by the inverse
    empirical CDF convention. A timed-out
    request is right-censored: its latency is
    known only to be at least TIMEOUT, so it
    is ranked after every completed request.
    Returns the value and a flag that is True
    when the order statistic of that rank is
    censored and the value is only a bound.
    A NaN is no timeout: it sets no bound,
    and known() of Listing 22.1 refuses it."""
    done = sorted(known(x for x in xs
                        if x is not None))
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
    weeks = (("week 1", w1), ("week 2", w2))
    for name, xs in weeks:
        lost = sum(x is None for x in xs)
        value, bound = p95_all(xs)
        text = f"{value:.0f}"
        if bound:
            text = ">= " + text
        print(f"{name:<8}{lost:>9}"
              f"{p95_completed(xs):>10.0f}"
              f"{text:>10}")


if __name__ == "__main__":
    main()
