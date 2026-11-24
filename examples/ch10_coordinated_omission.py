# replay: exact
"""Chapter 10: coordinated omission."""
import statistics as st

from swmetro.stats import ecdf_quantile, quantile
from swmetro.stats import known

MS = 1_000_000        # nanoseconds in 1 ms
INTERVAL = 10 * MS    # intended spacing
SERVICE = 1 * MS      # normal service time
DURATION = 60_000 * MS
STALLS = {20_000 * MS: 1000 * MS,
          40_000 * MS: 1000 * MS}


def resume(start):
    """When the server takes up a request that
    reaches it at `start`: at once, or at the
    end of a stall."""
    for begin, length in STALLS.items():
        if begin <= start < begin + length:
            return begin + length
    return start


def closed_loop():
    """The tester waits for each response and
    sends at the next tick it can meet, the
    tick of the response if it comes on one.
    Latency runs from the actual send. Times
    are whole nanoseconds and ticks whole
    numbers; the latencies returned are ms."""
    tick, out = 0, []
    while tick * INTERVAL < DURATION:
        sent = tick * INTERVAL
        done = resume(sent) + SERVICE
        out.append((done - sent) / MS)
        up = (done + INTERVAL - 1) // INTERVAL
        tick = max(tick + 1, up)
    return sorted(out)


def open_loop(busy=None):
    """A request is due at every tick. Latency
    runs from the intended send time. Service
    takes `busy`, if given, where a next
    request of the run is due by its start."""
    out, free, tick = [], 0, 0
    while tick * INTERVAL < DURATION:
        due = tick * INTERVAL
        start = resume(max(due, free))
        nxt = due + INTERVAL
        late = (busy is not None and nxt <= start
                and nxt < DURATION)
        free = start + (busy if late else SERVICE)
        out.append((free - due) / MS)
        tick += 1
    return sorted(out)


def back_fill(xs):
    """Add the samples a stalled tester did
    not take: L - i, L - 2i, ... for L > i,
    down to the last one above zero."""
    xs = list(xs)
    out = list(xs)
    for x in xs:
        extra = round(x * MS) - INTERVAL
        while extra > 0:
            out.append(extra / MS)
            extra -= INTERVAL
    return sorted(out)


def row(name, xs):
    """Interpolated percentiles and the mean."""
    qs = [quantile(xs, q)
          for q in (0.5, 0.99, 0.999, 0.9999)]
    cells = "".join(f"{q:>7.0f}" for q in qs)
    print(f"{name:<12}{len(xs):>5}{cells}"
          f"{st.fmean(xs):>5.1f}")


def conventions(xs):
    """One percentile under the rule of the
    table and as the inverse of the
    empirical distribution function."""
    a = quantile(xs, 0.9997)
    b = ecdf_quantile(xs, 0.9997)
    print("p99.97 as measured: interpolated "
          f"{a:.1f} ms,")
    print("  inverse of the empirical CDF "
          f"{b:.0f} ms")


def over(xs, limit=500.0):
    """Entries above the limit, and the share
    of the entries that are within it. Every
    entry has to be a duration, and so has
    the limit. A missing one that is kept as
    a NaN is neither above the limit nor
    within it, and would be counted as
    within: a record with unknown durations
    is for quantile_verdict() of Listing 16.3."""
    xs = known(xs)
    if not xs:
        raise ValueError("no entries")
    if min(xs + known([limit], "limit")) < 0:
        raise ValueError("need durations >= 0")
    n = sum(x > limit for x in xs)
    return n, 100 * (len(xs) - n) / len(xs)


def counted(rows):
    head = f"{'over 500 ms':<12}{'n':>5}"
    print(head + f"{'over':>6}{'within':>11}")
    for name, xs in rows:
        n, share = over(xs)
        print(f"{name:<12}{len(xs):>5}{n:>6}"
              f"{share:>9.4f} %")


def main():
    print(f"{'latency, ms':<12}{'n':>5}{'p50':>7}"
          f"{'p99':>7}{'p99.9':>7}{'p99.99':>7}"
          f"{'mean':>5}")
    measured = closed_loop()
    rows = (("as measured", measured),
            ("back-filled", back_fill(measured)),
            ("intended", open_loop()))
    for name, xs in rows:
        row(name, xs)
    conventions(measured)
    busy = ("5 ms if busy", open_loop(5 * MS))
    counted(rows + (busy,))
    row(*busy)


if __name__ == "__main__":
    main()
