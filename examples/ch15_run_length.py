# replay: exact
"""Chapter 15: what a Shewhart chart predicts,
with known limits and with estimated ones."""
import math
import random

from examples.ch15_control_chart import (
    MU, SIGMA, series, xmr)


def wait(lo, hi, mean, sigma=SIGMA):
    """Mean number of independent normal
    results to the first outside lo and hi.
    The two tails are summed as they are: 1
    less what lies between would lose them."""
    s = sigma * math.sqrt(2)
    p = (math.erfc((mean - lo) / s)
         + math.erfc((hi - mean) / s)) / 2
    return 1 / p if p else math.inf


def main():
    xs = series(random.Random(15))
    center, sigma = xmr(xs[:30])
    rows = ((MU - 3 * SIGMA, MU + 3 * SIGMA),
            (center - 3 * sigma,
             center + 3 * sigma))
    print("Shewhart, exact ARL at a shift of")
    print(f"{'limits, ms':<17}{'0':>7}{'1 s':>7}"
          f"{'2 s':>7}")
    for lo, hi in rows:
        ws = [wait(lo, hi, MU + d * SIGMA)
              for d in (0, 1, 2)]
        cells = "".join(f"{w:>7.1f}" for w in ws)
        print(f"[{lo:.2f}, {hi:.2f}]".ljust(17)
              + cells)
    print("no signal in 40 nights at 1 s")
    for lo, hi in rows:
        p = 1 / wait(lo, hi, MU + SIGMA)
        print(f"[{lo:.2f}, {hi:.2f}]".ljust(17)
              + f"{100 * (1 - p) ** 40:>7.1f} %")


if __name__ == "__main__":
    main()
