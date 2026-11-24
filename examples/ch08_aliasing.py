# replay: exact
"""Chapter 8: aliasing of a scraped gauge."""
import math
import random

PERIOD = 16.0  # period of the signal, s
SCRAPE = 15.0  # sampling interval, s


def gauge(t, period=PERIOD):
    """Queue depth that oscillates in time."""
    w = 2 * math.pi / period
    return 50 + 40 * math.sin(w * t)


def alias(period, interval):
    """Signed period of the alias produced by
    sampling at a fixed interval."""
    f, fs = 1 / period, 1 / interval
    fa = f - round(f / fs) * fs
    return math.inf if fa == 0 else 1 / fa


def spikes_seen(width, interval, n, rng):
    """Share of spikes of a given width that
    a periodic scrape lands on."""
    seen = 0
    for _ in range(n):
        start = rng.uniform(0, interval)
        seen += start + width > interval
    return seen / n


def main():
    ta = alias(PERIOD, SCRAPE)
    print(f"signal period   {PERIOD:.0f} s")
    print(f"scrape interval {SCRAPE:.0f} s")
    print(f"alias period    {abs(ta):.0f} s")
    print(f"{'t, s':>6}{'scraped':>9}"
          f"{'alias':>8}")
    for i in range(0, 17, 2):
        t = i * SCRAPE
        print(f"{t:>6.0f}{gauge(t):>9.1f}"
              f"{gauge(t, ta):>8.1f}")
    rng = random.Random(8)
    seen = spikes_seen(2.0, SCRAPE, 20_000, rng)
    print(f"2 s spikes seen: {100 * seen:.1f} %")


if __name__ == "__main__":
    main()
