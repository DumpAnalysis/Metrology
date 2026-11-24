# replay: exact
"""Chapter 19: floating-point resolution."""
import math
import random
import struct

MAX32 = (2 - 2.0 ** -23) * 2.0 ** 127


def f32(x):
    """Round a Python float to IEEE binary32."""
    packed = struct.pack("f", x)
    return struct.unpack("f", packed)[0]


def ulp32(x):
    """Spacing of binary32 values at x. Below
    2**-126 they are subnormal, 2**-149
    apart down to zero. Beyond the largest
    one, and for a NaN, there is none."""
    if not abs(x) <= MAX32:
        raise ValueError("no binary32 value")
    e = math.frexp(x)[1] if x else -125
    return 2.0 ** (max(e, -125) - 24)


def resolution():
    print(f"{'value':>12}{'step, f32':>12}"
          f"{'step, f64':>12}")
    for x in (1.0, 1000.0, 2.0 ** 24, 1e9):
        print(f"{x:>12.0f}{ulp32(x):>12.3g}"
              f"{math.ulp(x):>12.3g}")
    c = f32(2.0 ** 24)
    stuck = f32(c + 1.0) == c
    print(f"f32: 2**24 + 1 == 2**24 is {stuck}")
    ns = math.ulp(1.8e9) * 1e9
    print(f"f64 time in 2027: step {ns:.0f} ns")
    stamps()


def stamps():
    """Whole nanoseconds since 1970, each
    converted to binary64 before any
    subtraction: two pairs of stamps that
    lie 1 ns apart, counted from one day."""
    day = 17 * 10 ** 17  # ns; 14 November 2023
    step = math.ulp(float(day))
    print("f64 ns since 1970:"
          f" step {step:.0f} ns")
    print("  stamps 1 ns apart, as f64:")
    for k in (0, 128):
        gap = (float(day + k + 1)
               - float(day + k))
        print(f"    +{k} and +{k + 1} ns:"
              f" {gap:.0f} ns")


def accumulate(n):
    """Add 0.1 n times in binary32 and 64."""
    acc32, acc64, step = 0.0, 0.0, f32(0.1)
    for _ in range(n):
        acc32 = f32(acc32 + step)
        acc64 += 0.1
    return acc32, acc64


def loop_sum(xs):
    """Left-to-right sum, one rounding each."""
    acc = 0.0
    for x in xs:
        acc += x
    return acc


def order():
    """The same numbers in three orders,
    against their exactly rounded sum."""
    rng = random.Random(19)
    xs = [rng.lognormvariate(0, 4)
          for _ in range(10_000)]
    exact = math.fsum(xs)
    unit = math.ulp(exact)
    cases = (("as recorded", xs),
             ("reversed", xs[::-1]),
             ("ascending", sorted(xs)))
    for name, ys in cases:
        err = (loop_sum(ys) - exact) / unit
        print(f"sum {name:<12}{err:>+6.0f} ulp")


def main():
    resolution()
    a32, a64 = accumulate(1_000_000)
    print(f"10**6 sums of 0.1, f32: {a32:.2f}")
    print(f"10**6 sums of 0.1, f64: {a64:.9f}")
    order()


if __name__ == "__main__":
    main()
