# replay: exact
"""Chapter 23: has the service slowed down?"""
import math
import random
import statistics as st
from fractions import Fraction

from examples import ch09_histogram as h9
from swmetro import Result, type_a
from swmetro.core import fixed
from swmetro.dist import phi, z_ppf
from swmetro.stats import as_count, bootstrap
from swmetro.stats import downward, ecdf_quantile
from swmetro.stats import known, quantile_interval
from swmetro.stats import quiet_bound, rate_limits
from swmetro.stats import run_length, upward

# Telemetry configuration before and after the
# deployment: where the duration starts, and
# the histogram bounds, ms.
OLD = {"start": "handler",
       "bounds": [50, 100, 150, 200, 250, 500]}
NEW = {"start": "accept",
       "bounds": [100, 250, 500, 1000]}
WINDOW = 100.0  # seconds observed per period
Z = z_ppf(0.95)  # one-sided, 5 % at each end


def period(rng):
    """Raw trace of one period: per request,
    the queueing delay before the handler and
    the handler time, ms. The simulated
    service is the same in both periods."""
    rows, t = [], 0.0
    while True:
        t += rng.expovariate(20.0)
        if t >= WINDOW:
            return rows
        queue = rng.expovariate(1 / 6.0)
        z = rng.lognormvariate(0, 0.25)
        rows.append((queue, 120 * z))


def elapsed(xs):
    """Times that elapsed: durations that
    known() admits, and none below zero."""
    xs = known(xs)
    if min(xs, default=0) < 0:
        raise ValueError("a time below zero")
    return xs


def pair(x, y):
    """The two fields of a row before they are
    added. Two floats stay floats: their sum
    is the duration that a configuration
    records. Anything else is taken as the
    exact fractions that the two are, so that
    200.3 ms after no wait is still on a
    bucket bound of 200.3 ms."""
    if type(x) is type(y) is float:
        return x, y
    return Fraction(x), Fraction(y)


def exact(x, y, sign=1):
    """x + y, or x - y for a sign of -1, as
    the number that it is: a float where the
    two are floats and a float holds the
    result, and a fraction otherwise. A bound
    that is put to a margin is not rounded
    across it first. As floats, 1.3 - 0.1 is
    1.2, below the decimal 1.20, and the two
    that are stored differ by more than it."""
    total = Fraction(x) + sign * Fraction(y)
    if type(x) is type(y) is float:
        close = x + sign * y
        if close == total:
            return close
    return total


def recorded(rows, config):
    """Durations as a configuration records
    them, which names one of two starts. From
    handler entry it is the handler time, and
    the queueing delay is not read. From
    accept the two are added by pair(). A
    time that is read has elapsed()."""
    start, rows = config["start"], list(rows)
    if start not in ("handler", "accept"):
        raise ValueError(f"no start: {start!r}")
    hs = elapsed(h for _, h in rows)
    if start == "handler":
        return hs
    qs = elapsed(q for q, _ in rows)
    both = (pair(q, h) for q, h in zip(qs, hs))
    return [q + h for q, h in both]


def reported(rows, config):
    """The dashboard's p95, from buckets."""
    xs = recorded(rows, config)
    bounds = config["bounds"]
    return h9.bucket_quantile(xs, 0.95, bounds)


def shown(p95_ms):
    """A reported percentile as the dashboard
    should show it. A window without requests
    has no percentile, and a number in its
    place would come from the bounds alone.
    A bound that came as a fraction is
    written by fixed(), under Python 3.11
    as under the later ones."""
    if math.isnan(p95_ms):
        return "no observations"
    return fixed(p95_ms, 2)


def arrival(n, seconds=WINDOW):
    """Arrival rate from n requests in a fully
    observed window, under the Poisson model
    of Chapter 16. Without a request the
    estimate is zero, and its uncertainty
    by that model would be zero too: the
    upper bound is stated instead. With so
    few that the symmetric interval of the
    estimate would reach below zero, the
    exact limits are stated instead."""
    n = as_count(n)
    if not seconds > 0:
        raise ValueError("no exposure")
    if n == 0:
        top = upward(quiet_bound(seconds), 5)
        return f"at most {top} req/s at 95 %"
    u = math.sqrt(n) / seconds
    rate = Result(n / seconds, u, "req/s")
    if rate.interval()[0] > 0:
        return str(rate)
    lo, hi = rate_limits(n, seconds)
    return f"{lo:.5f} to {hi:.5f} req/s at 95 %"


def canary(config, n, rng):
    """Check workload: a stub whose handler
    takes 50 ms, measured n times."""
    rows = [(rng.expovariate(1 / 6.0),
             50 + rng.gauss(0, 0.5))
            for _ in range(n)]
    return type_a(recorded(rows, config), "ms")


def p95(xs):
    """Inverse empirical CDF convention, of
    durations that are all known: known() of
    the package refuses a missing one, and
    elapsed() one below zero."""
    xs = sorted(elapsed(xs))
    return ecdf_quantile(xs, 0.95)


def handler_change(before, after):
    """Change of the p95 of the handler time
    between the periods, with a bootstrap
    standard uncertainty. The replicates are
    taken as the exact numbers they are, and
    the two percentiles are subtracted by
    exact(): no change is rounded."""
    hb = [h for _, h in before]
    ha = [h for _, h in after]
    ub, ua = (st.stdev(map(Fraction, bootstrap(
        hs, p95, 300, 23))) for hs in (hb, ha))
    change = exact(p95(ha), p95(hb), -1)
    return Result(change,
                  math.hypot(ua, ub), "ms")


def upper_bound(d):
    """One-sided upper bound at 95 %, by the
    normal approximation, u taken as known.
    The half-width is the exact product of
    the float Z and u: rounded to a float, it
    can fall below a margin that it exceeds."""
    half = Fraction(Z) * Fraction(d.u)
    return Fraction(d.value) + half


def within(upper, margin):
    """Non-inferiority within a margin fixed
    before the measurement: shown only when
    the upper bound is smaller than the
    margin. A bound equal to the margin is
    not enough."""
    return upper < margin


def rank_bound(before, after):
    """Upper bound for the change of the p95
    of the handler time that needs no normal
    approximation: the upper end of the
    order-statistic interval of Chapter 9
    for the later period, less the lower end
    for the earlier one. Each end fails with
    a chance of 2.5 % at most, and one of the
    two with 5 % at most, whether or not they
    depend on each other. The requests of a
    period are still taken as independent.
    The two ends and the bound. An end is
    None where its period is too short for
    it, and there is no bound then."""
    hb = sorted(elapsed(h for _, h in before))
    ha = sorted(elapsed(h for _, h in after))
    low = quantile_interval(hb, 0.95)[0]
    high = quantile_interval(ha, 0.95)[1]
    if low is None or high is None:
        return low, high, None
    return low, high, exact(high, low, -1)


def margins(upper):
    """What an upper bound shows of margins
    of slowdown that would have had to be
    fixed before the measurement."""
    for margin in (5.0, 2.0):
        text = "not established"
        if within(upper, margin):
            text = "slowdown below the margin"
        print(f"margin {margin:.0f} ms: {text}")


def ranked(before, after):
    """The lines of the bound by ranks and of
    its margins. A period is listed first: it
    may be a stream. A missing end is named,
    with the two counts and the least count
    that has such an end: at 2.5 % a side, 2
    requests for a lower end, 72 for an upper."""
    before, after = list(before), list(after)
    low, high, top = rank_bound(before, after)
    print("the p95 by ranks:")
    if top is None:
        print(f"  requests: {len(before)} before,"
              f" {len(after)} after")
    if low is None:
        need = run_length(0.05, 0.025)
        print("  no lower end with fewer"
              f" than {need} before")
    if high is None:
        need = run_length(0.95, 0.025)
        print("  no upper end with fewer"
              f" than {need} after")
    if top is None:
        return
    print("  before, at least"
          f" {downward(low, 2)} ms at 97.5 %")
    print("  after, at most"
          f" {upward(high, 2)} ms at 97.5 %")
    print("  change at most"
          f" {upward(top, 2)} ms at 95 %")
    margins(top)


def side_by_side(n=100, offset=1):
    """The follow-up for the instrument, in
    small: n requests reported by both
    configurations, the second above the
    first by `offset` ms, added exactly. As
    two periods, the change of the p95 has
    the spread of two separate bootstraps.
    Resampled in pairs, one draw of requests
    for both reports, every replicate here
    is the offset: the spread is that of the
    change on the same requests."""
    one = [(0, 100 + i) for i in range(n)]
    two = [(0, exact(h, offset)) for _, h in one]

    def change(ids):
        new = p95(two[i][1] for i in ids)
        old = p95(one[i][1] for i in ids)
        return exact(new, old, -1)
    d = handler_change(one, two)
    top = upward(upper_bound(d), 2)
    reps = bootstrap(range(n), change, 300, 23)
    u = st.stdev(map(Fraction, reps))
    print(f"{n} requests reported twice,"
          f" {offset} ms apart:")
    print(f"  as two periods: {d}")
    print(f"    at most {top} ms at 95 %")
    print("  as pairs, one draw for both:"
          f" {Result(d.value, u, 'ms')}")


def short_trace(n=20):
    """Two periods of n requests of 100 ms.
    No resample differs from another, and
    the change and its bound are 0 ms. Yet
    if one request in ten of the second
    period took 200 ms, n of them would miss
    every slow one with the first chance
    printed, and the largest of n values of a
    continuous population lies above its p95
    with the second. The bound says nothing of
    a tail that the trace did not reach, and
    ranks give no bound for so few: ranked()
    says which end is missing."""
    rows = [(0.0, 100.0)] * n
    d = handler_change(rows, rows)
    miss, reach = 0.9 ** n, 1 - 0.95 ** n
    print(f"{n} requests of 100 ms, twice:")
    print(f"  difference {d}")
    print("  no slow one among them:"
          f" {100 * miss:.2f} %")
    print("  the largest above the p95:"
          f" {100 * reach:.2f} %")
    need = run_length(0.95, 0.05)
    print(f"  requests for 95 %: {need}")
    ranked(rows, rows)


def main():
    rng = random.Random(23)
    before, after = period(rng), period(rng)
    a = reported(before, OLD)
    b = reported(after, NEW)
    print(f"reported p95: {shown(a)}"
          f" -> {shown(b)} ms")
    idle = [shown(reported([], c))
            for c in (OLD, NEW)]
    print("empty window: " + " -> ".join(idle))
    print(f"  its arrival rate {arrival(0)}")
    print(f"  one request: {arrival(1)}")
    for name, rows in (("before", before),
                       ("after", after)):
        rate = arrival(len(rows))
        print(f"arrival rate {name:<7}{rate}")
    # The same requests through all four
    # combinations of the two settings.
    print("after period, reported p95, ms")
    print("start     old bounds  new bounds")
    for start in ("handler", "accept"):
        cells = ""
        for cfg in (OLD, NEW):
            mixed = dict(cfg, start=start)
            v = reported(after, mixed)
            cells += f"{v:>12.2f}"
        print(f"{start:<8}{cells}")
    c0 = canary(OLD, 200, rng)
    c1 = canary(NEW, 200, rng)
    shift = Result(c1.value - c0.value,
                   math.hypot(c0.u, c1.u), "ms")
    print(f"canary before {c0}")
    print(f"canary after  {c1}")
    print(f"placement adds {shift}")
    hb = [h for _, h in before]
    ha = [h for _, h in after]
    wait = st.fmean(q for q, _ in after)
    whole = p95(recorded(after, NEW))
    gap = whole - p95(ha)
    print(f"after: mean wait     {wait:.2f} ms")
    print(f"after: raw p95 shift {gap:.2f} ms")
    d = handler_change(before, after)
    print(f"handler p95: {p95(hb):.1f} ms"
          f" -> {p95(ha):.1f} ms")
    print(f"difference {d}")
    # One-sided test, normal approximation,
    # u taken as known.
    limit = 2 * Z * d.u
    power = phi(5 / d.u - Z)
    upper = upper_bound(d)
    lo, hi = d.interval()
    print(f"detection limit {limit:.1f} ms")
    print(f"power at 5 ms: {power:.2f}")
    print("upper bound, 95 %:"
          f" {upward(upper, 1)} ms")
    print("interval, 95 %:"
          f" [{fixed(lo, 1)}, {fixed(hi, 1)}] ms")
    margins(upper)
    ranked(before, after)
    short_trace()
    side_by_side()


if __name__ == "__main__":
    main()
