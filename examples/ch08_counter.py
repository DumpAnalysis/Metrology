# replay: exact
"""Chapter 8: a counter that resets or wraps."""

WRAP = 2 ** 32  # a 32-bit counter


def rates(values, step, rule, known=()):
    """Rates from successive readings of a
    cumulative counter, `step` seconds apart.
    naive:  plain difference.
    reset:  a decrease is read as a restart
            from zero, so the new value is a
            lower bound of the increase.
    modulo: a decrease is read as one wrap.
    `known` lists the intervals in which the
    start time of the process shows a restart;
    the new value is then a lower bound too."""
    out = []
    pairs = zip(values, values[1:])
    for i, (a, b) in enumerate(pairs):
        d = b - a
        if d < 0 and rule == "reset":
            d = b
        if d < 0 and rule == "modulo":
            d += WRAP
        if i in known:
            d = b
        out.append(d / step)
    return out


def increments(first, last, most, modulus=WRAP):
    """What two readings of a counter that is
    kept modulo `modulus` allow for the
    increase between them, when nothing reset
    the counter and the increase was at most
    `most`: their difference modulo `modulus`,
    and that plus every whole number of
    wraps. `most` is a whole number of counts
    for the longest gap that the record of
    the two times allows. It is formed in
    whole numbers, as counts a second times
    milliseconds // 1000: a product of floats
    can fall just short of a whole bound and
    lose the largest increase within it."""
    for reading in (first, last):
        if not 0 <= reading < modulus:
            raise ValueError("a reading outside "
                             "the counter")
    if not isinstance(most, int):
        raise ValueError("need a whole bound")
    d = (last - first) % modulus
    return list(range(d, most + 1, modulus))


def restarted():
    """120 events/s, read every 10 s. Four
    seconds after the fifth reading the
    process restarts and is down for 1 s."""
    before = [1200 * i for i in range(1, 6)]
    after = [600 + 1200 * i for i in range(5)]
    return before + after


def wrapping(step, n):
    """Bytes sent at 50 MB/s, counted in 32
    bits, read every `step` seconds."""
    return [50_000_000 * step * i % WRAP
            for i in range(n)]


def main():
    xs = restarted()
    naive = rates(xs, 10, "naive")[4]
    reset = rates(xs, 10, "reset")[4]
    print("a restart between two readings,")
    print("events/s in that interval")
    print(f"  events handled   {108.0:>8.1f}")
    print(f"  naive difference {naive:>8.1f}")
    print(f"  reset rule    >= {reset:>8.1f}")
    # The reading was 1000; 500 events, a
    # restart, and 1500 more events follow.
    ys = [1000, 1500]
    naive = rates(ys, 10, "naive")[0]
    reset = rates(ys, 10, "reset")[0]
    start = rates(ys, 10, "reset", {0})[0]
    print("a restart that leaves no decrease")
    print(f"  events handled   {200.0:>8.1f}")
    print(f"  naive difference {naive:>8.1f}")
    print(f"  reset rule       {reset:>8.1f}")
    print(f"  start time    >= {start:>8.1f}")
    print("a 32-bit byte counter at 50 MB/s")
    ys = wrapping(10, 12)
    k = next(i for i in range(11)
             if ys[i + 1] < ys[i])
    print("read every 10 s, at the wrap, MB/s")
    for rule in ("naive", "reset", "modulo"):
        r = rates(ys, 10, rule)[k] / 1e6
        print(f"  {rule:<17}{r:>8.1f}")
    zs = rates(wrapping(100, 12), 100, "modulo")
    lo, hi = min(zs) / 1e6, max(zs) / 1e6
    print("read every 100 s, all intervals, MB/s")
    print(f"  modulo     {lo:>8.1f} to {hi:.1f}")
    a, c = wrapping(100, 2)
    b = wrapping(50, 2)[1]
    most = 60_000_000 * 100
    print(f"readings {a} and {c}, at most")
    print("60 MB/s for 100 s: an increase of")
    for d in increments(a, c, most):
        print(f"  {d:>10} bytes, "
              f"{d / 1e8:.2f} MB/s")
    print(f"with the reading {b} at 50 s")
    for x, y in ((a, b), (b, c)):
        (d,) = increments(x, y, most // 2)
        print(f"  {d:>10} bytes, "
              f"{d / 5e7:.2f} MB/s")
    print("one answer for every gap below "
          f"{WRAP / 6e7:.2f} s")
    print("readings 0 and 0, the gap rounded to")
    for shown, ms in (("71.58", 71_585),
                      ("71.57", 71_575)):
        most = 60_000_000 * ms // 1000
        ds = increments(0, 0, most)
        text = " or ".join(str(d) for d in ds)
        print(f"  {shown} s: {text} bytes")


if __name__ == "__main__":
    main()
