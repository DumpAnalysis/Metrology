# replay: exact
"""Chapter 12: calibrating a latency probe."""
import math
import random
import statistics as st

from swmetro import Result
from swmetro.stats import linfit

REFS = [0, 5, 10, 20, 50, 100]  # delays, ms
# One additive offset is common to all six
# reference delays; its standard unc., ms:
U_REF = 0.02
REPEATS = 5


def probe(delay_ms, rng):
    """Simulated probe: offset, gain error
    and noise on top of the true delay."""
    noise = rng.gauss(0, 0.15)
    return 0.85 + 1.012 * delay_ms + noise


def calibrate(rng):
    """Indications for each reference delay
    and the fitted calibration line."""
    xs = [r for r in REFS for _ in range(REPEATS)]
    ys = [probe(x, rng) for x in xs]
    return xs, linfit(xs, ys)


def sources(ind, xs, fit):
    """The corrected delay for an indication
    and two standard uncertainties, in ms:
    that of one indication, and that of the
    fitted line at this point."""
    a, b, _, _, s = fit
    x = (ind - a) / b
    mx = st.fmean(xs)
    sxx = sum((v - mx) ** 2 for v in xs)
    var = 1 / len(xs) + (x - mx) ** 2 / sxx
    return x, s / b, s / b * math.sqrt(var)


def measure(ind, xs, fit, m=1, times=1):
    """Corrected delay for the mean of m new
    indications, with its standard
    uncertainty. The offset of the references
    shifts the whole line, so its uncertainty
    enters whole, and no repeat reduces it.
    `times` above 1: so many times the
    calibration readings, at the same
    settings, slope and scatter as fitted."""
    x, one, line = sources(ind, xs, fit)
    var = one ** 2 / m + line ** 2 / times
    u = math.hypot(math.sqrt(var), U_REF)
    return Result(x, u, "ms", times * len(xs) - 2)


def main():
    rng = random.Random(12)
    xs, fit = calibrate(rng)
    a, b, ua, ub, s = fit
    print(f"ind = {a:.3f} + {b:.4f} * delay")
    print(f"u(a) = {ua:.3f} ms, u(b) = {ub:.5f}")
    print(f"residual s = {s:.3f} ms")
    ind = 37.90
    r = measure(ind, xs, fit)
    print(f"indication        {ind:.2f} ms")
    corr = r.value - ind
    print(f"correction        {corr:+.2f} ms")
    print(f"measured value    {r}")
    print(f"result  {r.report()}")
    _, one, line = sources(ind, xs, fit)
    print(f"one indication {one:.3f}, line "
          f"{line:.3f}, ref. {U_REF:.3f} ms")
    print("u, ms, with calibration readings:"
          "   30    150")
    for m in (1, 5, 30):
        us = [measure(ind, xs, fit, m, k).u
              for k in (1, 5)]
        print(f"  indications in the mean: {m:>2}"
              f"    {us[0]:.3f}  {us[1]:.3f}")
    us = [measure(ind, xs, fit, math.inf, k).u
          for k in (1, 5)]
    print("  limit for many indications:    "
          f"{us[0]:.3f}  {us[1]:.3f}")


if __name__ == "__main__":
    main()
