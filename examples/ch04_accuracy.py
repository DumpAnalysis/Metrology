# replay: exact
"""Chapter 4: trueness, precision, resolution."""
import random
import statistics as st

from swmetro import rect_interval, u_rect
from swmetro import u_resolution

REF = 100.0  # reference quantity value, ms
DEVICES = {  # name: (bias, sigma), in ms
    "A true, precise": (0.0, 0.2),
    "B true, imprecise": (0.0, 2.0),
    "C biased, precise": (3.0, 0.2),
    "D biased, imprecise": (3.0, 2.0),
}


def indications(bias, sigma, n, rng):
    """n indications of the reference value."""
    return [REF + bias + rng.gauss(0, sigma)
            for _ in range(n)]


def trueness_and_precision():
    rng = random.Random(4)
    print(f"{'instrument':<21}{'bias':>7}"
          f"{'s':>7}")
    for name, (bias, sigma) in DEVICES.items():
        xs = indications(bias, sigma, 50, rng)
        est = st.fmean(xs) - REF
        print(f"{name:<21}{est:>7.2f}"
              f"{st.stdev(xs):>7.2f}")


def zero_variance_trap():
    """A timer that truncates to whole ms."""
    true_ms, step = 12.3449, 1.0
    xs = [true_ms // step * step] * 10
    fixed = xs[0] + step / 2
    rows = (
        ("indication (x10)", xs[0]),
        ("Type A, s", st.stdev(xs)),
        ("Type B, step", u_resolution(step)),
        ("error", xs[0] - true_ms),
        ("corrected value", fixed),
        ("error after", fixed - true_ms),
    )
    for name, ms in rows:
        print(f"{name:<18}{ms:>8.3f} ms")


def rectangular_coverage():
    """A value known only to lie within 1 of
    an estimate: what is its 95 % interval?"""
    u = u_rect(1.0)
    lo, hi = map(float, rect_interval(0.0, 1.0))
    wide = 1.96 * u
    print(f"rectangular on [-1, 1]: u = {u:.3f}")
    print(f"1.96 u = {wide:.3f}, past the bound")
    print(f"central 95 % interval: +-{hi:.3f},"
          f" k = {hi / u:.3f}")


def main():
    trueness_and_precision()
    zero_variance_trap()
    rectangular_coverage()


if __name__ == "__main__":
    main()
