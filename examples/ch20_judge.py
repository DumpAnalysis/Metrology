# replay: exact
"""Chapter 20: an LLM judge as an instrument."""
import math
import random

from swmetro.propagate import lpu
from swmetro.stats import as_count, cohen_kappa

# Simulation truth, unknown to the analyst:
SENS, SPEC = 0.90, 0.85  # judge's hit rates
PASS_TRUE = 0.60         # system under test


def judge(truth, rng):
    """Verdict of the judge on one output."""
    p = SENS if truth else 1 - SPEC
    return rng.random() < p


def corrected(x):
    """Pass rate corrected for the judge's
    errors: x = [observed, sens, spec]. The
    divisor sens + spec - 1 has to be far
    from 0, in either direction, for the
    correction to be stable. This function
    takes a judge whose verdicts go with the
    reference, a divisor above 0. At 0 the
    judged rate does not depend on the true
    one. Below 0 the verdicts go against the
    reference, see mirrored().
    The estimate is not confined to [0, 1].
    With few outputs sampling alone can take
    it below 0 or above 1, even when the
    error rates are known and the model fits,
    and whoever reads it has to see that."""
    obs, sens, spec = x
    j = sens + spec - 1
    if j == 0:
        raise ValueError("uninformative judge")
    if j < 0:
        raise ValueError("reversed judge")
    return (obs + spec - 1) / j


def mirrored(k, n, sens, spec):
    """A judge with sens + spec below 1 read
    with its two verdicts exchanged: k passes
    of n become n - k, and the error rates
    become 1 - sens and 1 - spec. A stable
    judge of this kind carries as much
    information as its mirror image, and is
    a reason to check how the verdicts were
    labeled."""
    return n - k, n, 1 - sens, 1 - spec


def feasible(y):
    """A rate confined to [0, 1], and where
    the unconfined one lies."""
    if y < 0:
        return 0.0, "below 0"
    if y > 1:
        return 1.0, "above 1"
    return y, "within 0 and 1"


def binom_u(p, n):
    n = as_count(n, least=1)
    return math.sqrt(p * (1 - p) / n)


def tails(k, n, p):
    """P(X <= k) and P(X > k) for k of n
    independent trials that each succeed with
    probability p, each summed from its own
    terms: a tail next to zero would lose its
    digits as 1 less the other. The terms are
    built outward from the largest one by
    their ratios and divided by their sum, so
    that a large n neither overflows nor
    underflows."""
    n = as_count(n)
    if not 0 <= p <= 1:
        raise ValueError("need 0 <= p <= 1")
    if k < 0:
        return 0.0, 1.0
    if p == 0 or k >= n:
        return 1.0, 0.0
    if p == 1:
        return 0.0, 1.0
    mode = min(n, int((n + 1) * p))
    odds = p / (1 - p)
    parts = [0.0, 0.0]  # up to k, above k
    term, i = 1.0, mode
    while i >= 0 and term > 0:
        parts[i > k] += term
        term *= i / ((n - i + 1) * odds)
        i -= 1
    term, i = 1.0, mode
    while i < n and term > 0:
        term *= (n - i) * odds / (i + 1)
        i += 1
        parts[i > k] += term
    total = parts[0] + parts[1]
    return parts[0] / total, parts[1] / total


def at_most(k, n, p):
    """P(X <= k), see tails()."""
    return tails(k, n, p)[0]


def exact_interval(k, n, level=0.95):
    """Exact interval for a proportion from k
    successes in n trials: the proportions
    at which a count this low, or this high,
    still has probability (1 - level) / 2.
    Without a trial there is no interval, as
    for wilson(), and both counts are whole
    numbers. Each limit is found from its own
    tail, and the proportion is halved in
    until its ends are neighboring floats."""
    n, k = as_count(n), as_count(k, "k")
    if not n:
        raise ValueError("no trials")
    if k > n:
        raise ValueError("need 0 <= k <= n")
    if not 0 < level < 1:
        raise ValueError("need 0 < level < 1")
    a = (1 - level) / 2

    def solve(too_low):
        lo, hi = 0.0, 1.0
        while True:
            mid = (lo + hi) / 2
            if mid in (lo, hi):
                return mid
            if too_low(mid):
                lo = mid
            else:
                hi = mid
    low, high = 0.0, 1.0
    if k > 0:
        low = solve(lambda p:
                    tails(k - 1, n, p)[1] < a)
    if k < n:
        high = solve(lambda p:
                     tails(k, n, p)[0] > a)
    return low, high


def few_outputs(k, n, sens, spec):
    """The correction for k judged passes in n
    outputs with known error rates: the
    unconfined estimate, the exact interval
    for the judged rate, and what that
    interval leaves of the true rate. The
    judge model allows judged rates from
    1 - spec to sens only. The part of the
    interval within them is carried through
    the correction. If no part is, the
    result is None: the interval and the
    model have no rate in common."""
    seen = exact_interval(k, n)
    y = corrected([k / n, sens, spec])
    if seen[1] < 1 - spec or seen[0] > sens:
        return y, seen, None
    ends = []
    for p in seen:
        end = corrected([p, sens, spec])
        ends.append(feasible(end)[0])
    return y, seen, ends


def kappa_at(share, sens, spec):
    """Cohen's kappa of a judge with these
    hit rates when the given share of the
    reference labels are passes: agreement
    above that expected by chance, over the
    disagreement expected by chance. The
    disagreement is summed from its two
    parts. Taken as one less the chance
    agreement, it is lost to rounding when
    nearly all labels are the same. At a
    share of one half it is one half
    whatever the judge says, and kappa is
    sens + spec - 1. If reference and judge
    both give every item the same one label
    there is none, and kappa is not defined,
    as in cohen_kappa()."""
    passed = (share * sens
              + (1 - share) * (1 - spec))
    failed = (share * (1 - sens)
              + (1 - share) * spec)
    apart = (share * failed
             + (1 - share) * passed)
    if not apart:
        raise ValueError("kappa needs two labels")
    above = (2 * share * (1 - share)
             * (sens + spec - 1))
    return above / apart


def main():
    rng = random.Random(20)
    # 1. Calibration against human labels.
    ref = [i % 2 == 0 for i in range(400)]
    ver = [judge(t, rng) for t in ref]
    pos = sum(ref)
    neg = len(ref) - pos
    pairs = list(zip(ref, ver))
    sens = sum(r and v for r, v in pairs) / pos
    tn = sum(not (r or v) for r, v in pairs)
    spec = tn / neg
    kappa = cohen_kappa(ref, ver)
    print(f"sensitivity {sens:.3f}, "
          f"specificity {spec:.3f}")
    print(f"Cohen's kappa {kappa:.3f}")
    # 2. The judge scores 5000 outputs.
    n = 5000
    obs = sum(judge(rng.random() < PASS_TRUE, rng)
              for _ in range(n)) / n
    xs = [obs, sens, spec]
    j = sens + spec - 1
    print(f"sens + spec - 1 = {j:.3f}")
    low = kappa_at(0.1, sens, spec)
    print("kappa with 10 % reference passes:"
          f" {low:.3f}")
    for size in (n, 10 * n):
        us = [binom_u(obs, size),
              binom_u(sens, pos),
              binom_u(spec, neg)]
        y, u, cu = lpu(corrected, xs, us)
        what = ("projected", "judged")[size == n]
        print(f"outputs {what}: {size}")
        print(f"  observed {obs:.3f}, "
              f"corrected {y:.3f}, u {u:.3f}")
        parts = ", ".join(f"{abs(c):.4f}"
                          for c in cu)
        print(f"  from obs, sens, spec: {parts}")
    # 3. Few outputs, and error rates taken as
    # known: sampling alone crosses the bound.
    low = 0.04
    seen = low * SENS + (1 - low) * (1 - SPEC)
    chance = at_most(2, 20, seen)
    print(f"true rate {low}: judged {seen:.2f},")
    print("  P(at most 2 of 20 pass)"
          f" {chance:.4f}")
    for k, n in ((2, 20), (0, 25), (40, 40)):
        y, seen, ends = few_outputs(k, n, SENS,
                                    SPEC)
        print(f"{k} of {n} judged passes:")
        where = feasible(y)[1]
        print(f"  corrected {y:+.4f}, {where}")
        if ends:
            a, b = ends
            print("  95 % interval"
                  f" {a:.3f} to {b:.3f}")
            continue
        a, b = seen
        print("  judged rate"
              f" {a:.3f} to {b:.3f},")
        print("  no overlap with the judge")
        print(f"  model, {1 - SPEC:.2f} to"
              f" {SENS:.2f}, at 95 %")
    # 4. A judge that goes against the labels.
    odd = (40, 100, 0.10, 0.15)
    try:
        corrected([odd[0] / odd[1], *odd[2:]])
    except ValueError as err:
        print(f"sens 0.10, spec 0.15: {err}")
    k, n, sens, spec = mirrored(*odd)
    y, _, (a, b) = few_outputs(k, n, sens, spec)
    print(f"  verdicts exchanged, {k} of {n}:")
    print(f"  corrected {y:+.4f},")
    print(f"  95 % interval {a:.3f} to {b:.3f}")


if __name__ == "__main__":
    main()
