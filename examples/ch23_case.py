# replay: exact
"""Chapter 23: has the service slowed down?"""
import math
import random
import statistics as st

from examples import ch09_histogram as h9
from examples.ch22_budget import known
from swmetro import Result, type_a
from swmetro.dist import phi, z_ppf
from swmetro.stats import as_count, bootstrap
from swmetro.stats import ecdf_quantile
from swmetro.stats import quiet_bound, rate_limits

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


def recorded(rows, config):
    """Durations as a configuration records
    them: from accept, or from handler entry."""
    if config["start"] == "accept":
        return [q + h for q, h in rows]
    return [h for _, h in rows]


def reported(rows, config):
    """The dashboard's p95, from buckets."""
    xs = recorded(rows, config)
    bounds = config["bounds"]
    return h9.bucket_quantile(xs, 0.95, bounds)


def shown(p95_ms):
    """A reported percentile as the dashboard
    should show it. A window without requests
    has no percentile, and a number in its
    place would come from the bounds alone."""
    if math.isnan(p95_ms):
        return "no observations"
    return f"{p95_ms:.2f}"


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
        top = quiet_bound(seconds)
        return f"at most {top:.5f} req/s at 95 %"
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
    Listing 22.1 refuses a missing one."""
    return ecdf_quantile(sorted(known(xs)), 0.95)


def handler_change(before, after):
    """Change of the p95 of the handler time
    between the periods, with a bootstrap
    standard uncertainty."""
    hb = [h for _, h in before]
    ha = [h for _, h in after]
    ub = st.stdev(bootstrap(hb, p95, 300, 23))
    ua = st.stdev(bootstrap(ha, p95, 300, 23))
    return Result(p95(ha) - p95(hb),
                  math.hypot(ua, ub), "ms")


def upper_bound(d):
    """One-sided upper bound at 95 %, by the
    normal approximation, u taken as known."""
    return d.value + Z * d.u


def within(upper, margin):
    """Non-inferiority within a margin fixed
    before the measurement: shown only when
    the upper bound is smaller than the
    margin. A bound equal to the margin is
    not enough."""
    return upper < margin


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
    print(f"upper bound, 95 %: {upper:.1f} ms")
    print(f"interval, 95 %: "
          f"[{lo:.1f}, {hi:.1f}] ms")
    # Margins of slowdown that would have had
    # to be fixed before the measurement.
    for margin in (5.0, 2.0):
        text = "not established"
        if within(upper, margin):
            text = "slowdown below the margin"
        print(f"margin {margin:.0f} ms: {text}")


if __name__ == "__main__":
    main()
