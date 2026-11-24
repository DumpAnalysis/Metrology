# replay: exact
"""Chapter 11: is this process leaking?"""
import random

from swmetro.stats import linfit

RATE = 0.05  # true growth, MiB/min
NOISE = 3.0  # sd of one reading, MiB


def readings(minutes, rng):
    """One reading of memory in use a minute."""
    ts = list(range(minutes + 1))
    ms = [512 + RATE * t + rng.gauss(0, NOISE)
          for t in ts]
    return ts, ms


def main():
    rng = random.Random(11)
    ts, ms = readings(240, rng)
    # Normal critical values are used with an
    # estimated u: an approximation.
    print(f"{'window':>7}{'slope':>8}{'u':>8}"
          f"{'upper':>8}{'L_D':>8}  shown")
    for w in (20, 60, 120, 240):
        fit = linfit(ts[:w + 1], ms[:w + 1])
        slope, u = fit[1], fit[3]
        shown = ("no", "yes")[slope > 1.645 * u]
        upper = slope + 1.645 * u
        print(f"{w:>3} min{slope:>8.4f}{u:>8.4f}"
              f"{upper:>8.4f}{3.29 * u:>8.4f}"
              f"  {shown}")


if __name__ == "__main__":
    main()
