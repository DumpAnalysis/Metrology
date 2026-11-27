# replay: exact
"""Chapter 8: what a sample can resolve."""
import math
import random
import statistics as st

from swmetro.dist import z_ppf
from swmetro.stats import as_count, enclosed
from swmetro.stats import exact_q, run_length
from swmetro.stats import wilson

UNEVEN = [180] + [5] * 34  # on 35 traces


def profile(p_true, n, rng):
    """Take n stack samples; count those that
    land in the function of interest."""
    return sum(rng.random() < p_true
               for _ in range(n))


def samples_needed(p1, p2, power=0.8):
    """Samples per profile to tell p1 from p2
    with a two-sided 5 % test."""
    z = z_ppf(0.975) + z_ppf(power)
    var = p1 * (1 - p1) + p2 * (1 - p2)
    return math.ceil(z * z * var / (p1 - p2) ** 2)


def units(n_true, group=1):
    """Sizes of the units that get one sampling
    decision each. `group` is the size of a
    unit, and a last, smaller unit holds
    what is left over; or `group` is the
    list of the sizes themselves."""
    n_true = as_count(n_true, "n_true")
    if isinstance(group, (list, tuple)):
        sizes = [as_count(m, "size", 1)
                 for m in group]
        if sum(sizes) != n_true:
            raise ValueError("the sizes must "
                             "add up to n_true")
        return sizes
    group = as_count(group, "group", 1)
    full, rest = divmod(n_true, group)
    return [group] * full + [rest] * (rest > 0)


def sampled_count(n_true, rate, rng, group=1):
    """Estimate a count from a head-sampled
    trace: events kept, divided by the rate.
    One independent decision is made for
    each unit of units(), as for the failed
    children of one trace."""
    if not 0 < rate <= 1:
        raise ValueError("need 0 < rate <= 1")
    kept = sum(m for m in units(n_true, group)
               if rng.random() < rate)
    return kept / rate


def none_kept(n_true, rate, group=1):
    """The chance that such a record holds no
    event: every decision goes against."""
    if not 0 <= rate <= 1:
        raise ValueError("need 0 <= rate <= 1")
    return (1 - rate) ** len(units(n_true, group))


def spread(n_true, rate, group=1):
    """Standard uncertainty of sampled_count().
    A unit of m events is kept whole or not
    at all, and it adds m * m * (1 - rate)
    / rate to the variance."""
    squares = sum(m * m
                  for m in units(n_true, group))
    return math.sqrt(squares * (1 - rate) / rate)


def estimated(kept, rate):
    """From one capture: the estimate of the
    count and its standard uncertainty, for
    units that were kept independently with
    the probability `rate`. `kept` lists the
    events on each unit of the capture, which
    may be none. A unit of m events adds
    m / rate to the estimate and (1 - rate)
    * (m / rate) ** 2 to an unbiased estimate
    of its variance. For a capture without
    an event that sum is 0. If every unit
    was kept, 0 is the count and its
    uncertainty. If not, it is the
    uncertainty of nothing: it is left
    unknown, and most_units() says what such
    a capture supports."""
    if not 0 < rate <= 1:
        raise ValueError("need 0 < rate <= 1")
    sizes = [as_count(m, "size") for m in kept]
    squares = sum(m * m for m in sizes)
    if not squares and rate < 1:
        return 0.0, None
    var = (1 - rate) * squares / rate ** 2
    return sum(sizes) / rate, math.sqrt(var)


def most_units(rate, miss=0.05):
    """After a record with no event: the most
    units that leave such a record with a
    chance of `miss` or more, if each unit is
    kept independently with the probability
    `rate`. With more units than that, a
    record as empty would have been rarer
    than `miss`."""
    miss = exact_q(miss)
    if not 0 < miss < 1:
        raise ValueError("need 0 < miss < 1")
    left = 1 - exact_q(rate)
    n = run_length(left, miss)
    return n if left ** n == miss else n - 1


def most_of_quota(total, kept, miss=0.05):
    """The same bound for a collector that
    keeps exactly `kept` of `total` units,
    every such choice being equally likely.
    With g units that hold an event, the
    record is empty when all that are kept
    come from the other total - g."""
    total = as_count(total, "total", 1)
    kept = as_count(kept, "kept", 1)
    miss = exact_q(miss)
    if kept > total or not 0 < miss < 1:
        raise ValueError("need kept <= total "
                         "and 0 < miss < 1")
    a, b = miss.as_integer_ratio()
    whole, g = a * math.comb(total, kept), 0
    while (b * math.comb(total - g - 1, kept)
           >= whole):
        g += 1
    return g


def profiler():
    rng = random.Random(8)
    print("profiler, true share 7.00 %")
    for n in (100, 1000, 10_000, 100_000):
        k = profile(0.07, n, rng)
        share = 100 * k / n
        lo, hi = (100 * v for v in wilson(k, n))
        lo, hi = enclosed(lo, hi, 2)
        print(f"n = {n:>6}: {share:5.2f} %  "
              f"[{lo:>5}, {hi:>5}]")
    need = samples_needed(0.07, 0.08)
    print(f"to tell 7 % from 8 %: n = {need}")


def trace_sampling():
    rng, n = random.Random(8), 350
    print(f"sampled trace, {n} true errors")
    cases = ((0.01, 1), (0.1, 1), (0.01, 10),
             (0.01, UNEVEN))
    for rate, g in cases:
        est = [sampled_count(n, rate, rng, g)
               for _ in range(2000)]
        m, s = st.fmean(est), st.stdev(est)
        model = spread(n, rate, g)
        none = 100 * est.count(0) / len(est)
        zero = 100 * none_kept(n, rate, g)
        if g == 10:
            print("one decision for 10 errors,")
        if g is UNEVEN:
            print("one trace of 180, 34 of 5,")
        print(f"rate {rate:.2f}: mean {m:.0f}, "
              f"s {s:.0f}, model {model:.0f}")
        print("  runs with no error kept: "
              f"{none:.1f} %, model {zero:.1f} %")


def one_capture():
    print("20 errors kept at rate 0.10, on")
    for sizes in ([1] * 20, [10] * 2, [20]):
        est, u = estimated(sizes, 0.1)
        print(f"  traces of {sizes[0]:>2}: "
              f"estimate {est:.0f}, u = {u:.1f}")


def empty_record():
    print("an empty record: at 95 %, at most")
    for rate in (0.01, 0.1):
        g = most_units(rate)
        print(f"  rate {rate:.2f}: {g:>3} units")
    g = most_of_quota(100, 10)
    print(f"  10 of 100 kept: {g} units")


def main():
    profiler()
    trace_sampling()
    one_capture()
    empty_record()


if __name__ == "__main__":
    main()
