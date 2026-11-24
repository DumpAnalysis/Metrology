# replay: live
"""Chapter 7: the probe changes the measurand."""
import statistics as st
import sys
import time

N, REPEATS = 10_000, 15
now = time.perf_counter_ns


def square(i):
    return i * i


def work():
    """The operation under measurement."""
    return sum(square(i) for i in range(20))


def blank():
    """An empty operation: what the probe
    indicates when there is nothing to time."""


def external(fn):
    """Two clock readings around N calls."""
    t0 = now()
    for _ in range(N):
        fn()
    return (now() - t0) / N


def probed(fn):
    """Two clock readings around every call.
    Returns the mean indication and the mean
    cost per call of the probed loop."""
    total = 0
    t0 = now()
    for _ in range(N):
        a = now()
        fn()
        total += now() - a
    return total / N, (now() - t0) / N


def hooked(fn):
    """A profile hook that counts events."""
    count = [0]

    def hook(frame, event, arg):
        count[0] += 1

    sys.setprofile(hook)
    try:
        return external(fn)
    finally:
        sys.setprofile(None)


def main():
    # The four procedures are interleaved round
    # by round, so that slow drifts of the
    # machine affect all of them alike.
    rounds = []
    for _ in range(REPEATS):
        base = external(work)
        ind, loop = probed(work)
        zero, _ = probed(blank)
        hook = hooked(work)
        row = (base, ind, zero, loop, hook)
        rounds.append(row)
    cols = [st.median(c) for c in zip(*rounds)]
    base, ind, zero, loop, hook = cols
    rows = (("unprobed loop", base),
            ("probe indication", ind),
            ("blank indication", zero),
            ("corrected", ind - zero),
            ("probed loop", loop),
            ("profile hook", hook))
    for name, ns in rows:
        print(f"{name:<18}{ns:>9.0f} ns"
              f"  x{ns / base:.2f}")


if __name__ == "__main__":
    main()
