# replay: live
"""Chapter 6: the clock as an instrument."""
import time

CLOCKS = ("perf_counter", "monotonic", "time")


def min_step(clock, rounds=200):
    """Smallest positive increment observed
    between consecutive readings. It bounds
    the spacing of readings as executed, not
    the granularity of the clock itself."""
    best = 0
    for _ in range(rounds):
        xs = [clock() for _ in range(1000)]
        for a, b in zip(xs, xs[1:]):
            d = b - a
            if d > 0 and (best == 0 or d < best):
                best = d
    return best


def read_cost(clock, n=200_000):
    """Amortized cost of one reading inside
    a loop, ns, loop overhead included."""
    t0 = time.perf_counter_ns()
    for _ in range(n):
        clock()
    return (time.perf_counter_ns() - t0) / n


def main():
    print(f"{'clock':<13}{'declared':>9}"
          f"{'incr.':>9}{'cost':>8}  ns")
    for name in CLOCKS:
        info = time.get_clock_info(name)
        clock = getattr(time, name + "_ns")
        step = min_step(clock)
        cost = read_cost(clock)
        res = info.resolution * 1e9
        print(f"{name:<13}{res:>9.0f}"
              f"{step:>9d}{cost:>8.1f}")
    for name in CLOCKS:
        info = time.get_clock_info(name)
        impl = info.implementation
        print(f"env: {name}: {impl}")


if __name__ == "__main__":
    main()
