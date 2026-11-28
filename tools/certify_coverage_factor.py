"""An optional check of the coverage factor, outside the unit tests.

The checker of swmetro/record.py computes the Student t quantile at
1 - risk from the risk as the record states it, an exact decimal. Up to a
risk of one quarter it works from the upper tail, and above it from the
distance of the risk from one half, which it keeps exactly. It then allows
a relative error of DELTA = 1e-6 for the computed factor.

This program certifies, in 80-digit arithmetic, what record._factor returns.

- A computed factor: the true factor lies within one hundredth of the
  allowance of it. For a computed k and a = DELTA / 100 it verifies, for a
  risk up to one quarter,

      P(T > k (1 - a))  >=  risk  >=  P(T > k (1 + a)),

  and for a risk above one quarter, with gap = 1/2 - risk,

      P(0 < T < k (1 - a))  <=  gap  <=  P(0 < T < k (1 + a)),

  which brackets the true quantile without finding it. The bounds that the
  checker uses, k (1 - DELTA) and k (1 + DELTA), are a hundred times wider.
- A factor reported as too large to compute: the quantile is indeed beyond
  5 x 10**149, the least point at which a search for it gives up, and so
  beyond FAR = 10**149, the lower bound that the checker uses for it since
  Version 15. No upper bound is claimed.
- A factor that is only bounded from below (a risk above one quarter with
  fewer than 0.001 degrees of freedom): the bound, twice the gap, is indeed
  below the true factor.
- The one factor that is known exactly is 1.

It needs mpmath, which the examples and the tests do not:

    python -m pip install mpmath==1.4.1
    python tools/certify_coverage_factor.py           # a few minutes
    python tools/certify_coverage_factor.py 5000 3    # more cases, seed 3

The risks are exact decimals from 1e-300 to 1/2 - 1e-300, on a grid and at
random; the degrees of freedom run from 1e-30 to 1e300.

A second family, added with Version 15, lies next to the end of the
computed range. A quantile between 10**146.5 and 10**152.5 and a number of
degrees of freedom between 0.0003 and 3 are drawn, the risk is the true tail
at that quantile to 60 digits, and the same four statements are verified.
There the first archive of Version 15 claimed FAR for factors that were
smaller, because the tail was cut off at 10**150 sqrt(dof); swmetro/dist.py
now cuts it off at 10**153 sqrt(dof), beyond the range of the search.
"""
import math
import random
import sys
from fractions import Fraction
from pathlib import Path

import mpmath as mp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from swmetro.record import DELTA, FAR, LEAST, _factor     # noqa: E402

mp.mp.dps = 80
MARGIN = 100
HALF = Fraction(1, 2)
DOFS = ("1e-30", "1e-21", "1e-19", "1e-9", "0.0009", "0.001", "0.004", "0.01", "0.05", "0.3", "0.5", "1", "1.5", "2", "3", "4", "4.79",
        "7", "12.3", "15.7", "30", "100", "1e3", "1e4", "1e5", "5e5", "9.9e5", "1e6", "1000000.1", "2e6", "1e7", "1e9",
        "1e15", "1e300")
RISKS = [Fraction(10) ** -n for n in (300, 200, 100, 30, 12, 9, 6, 4, 3, 2)]
RISKS += [Fraction(x) for x in ("0.025", "0.05", "0.1", "0.2", "0.24", "0.25", "0.2500000001", "0.26", "0.3", "0.4",
                                "0.45", "0.49", "0.499", "0.4999")]
RISKS += [HALF - Fraction(10) ** -n for n in (6, 8, 9, 10, 12, 15, 16, 17, 20, 30, 50, 100, 150, 200, 300)]


def rational(q):
    return mp.mpf(q.numerator) / mp.mpf(q.denominator)


def center(t, dof):
    """P(0 < T < t) for Student's t. The incomplete beta function is taken in
    the form whose argument is formed without loss, and the complement only
    where t^2 is so far above dof that the direct argument would round to 1."""
    t = mp.mpf(t)
    if dof > Fraction(10) ** 40:      # indistinguishable from the normal at 80 digits
        return mp.erf(t / mp.sqrt(2)) / 2
    v = rational(dof)
    if t * t <= v * mp.mpf(10) ** 30:
        return mp.betainc(mp.mpf(1) / 2, v / 2, 0, t * t / (v + t * t), regularized=True) / 2
    return mp.mpf(1) / 2 - upper_tail(t, dof)


def upper_tail(t, dof):
    """P(T > t), from the form that is accurate far in the tail."""
    t = mp.mpf(t)
    if dof > Fraction(10) ** 40:
        return mp.erfc(t / mp.sqrt(2)) / 2
    v = rational(dof)
    return mp.betainc(v / 2, mp.mpf(1) / 2, 0, v / (v + t * t), regularized=True) / 2


def brackets(risk, dof, lo, hi):
    """True if the factor for this exact risk lies between lo and hi."""
    gap = HALF - risk
    if gap < risk:
        return center(lo, dof) <= rational(gap) <= center(hi, dof)
    return upper_tail(lo, dof) >= rational(risk) >= upper_tail(hi, dof)


GIVES_UP = mp.mpf(5) * mp.mpf(10) ** 149


def judge(risk, dof):
    """The kind of answer _factor gives for this risk, and None or a failure."""
    k, low, high = _factor(risk, dof)
    gap = HALF - risk
    if k is None and low == FAR:              # too large to compute
        # Beyond GIVES_UP is beyond FAR, so the one test shows the need and the bound.
        if high is not None or rational(FAR) >= GIVES_UP or brackets(risk, dof, 0, GIVES_UP):
            return "refused", ("refused without need", float(risk), float(dof))
        return "refused", None
    if k is None and low is None:             # no bound at all: not returned since Version 15
        return "refused", ("no bound", float(risk), float(dof))
    if k is None:                             # bounded from below only
        sound = (high is None and low == 2 * gap and gap < risk and dof < Fraction(1, 1000)
                 and center(rational(low), dof) <= rational(gap))
        return "bounded", None if sound else ("a bound that does not hold", str(risk)[:40], float(dof))
    if low == k == high:                      # the one factor that is known
        known = k == 1 and dof == 1 and risk == Fraction(1, 4)
        return "exact", None if known else ("called exact", float(risk), float(dof))
    a = rational(DELTA) / MARGIN
    inside = (low, high) == (k * (1 - DELTA), k * (1 + DELTA)) and k > 0
    if inside and brackets(risk, dof, rational(k) * (1 - a), rational(k) * (1 + a)):
        return "certified", None
    return "certified", (str(risk)[:40], float(dof), float(k))


def report(title, cases):
    counts = dict.fromkeys(("certified", "exact", "bounded", "refused"), 0)
    failed = []
    for risk, dof in cases:
        kind, failure = judge(risk, dof)
        if failure is None:
            counts[kind] += 1
        else:
            failed.append(failure)
    print(f"{title}{len(cases)}: certified {counts['certified']}, known exactly {counts['exact']}, "
          f"bounded from below {counts['bounded']}, rightly refused {counts['refused']}, failed {len(failed)}")
    for f in failed[:20]:
        print("  ", f)
    return len(failed)


def range_end(rng, n):
    """Risks whose factor lies next to the end of the computed range."""
    cases = []
    while len(cases) < n:
        dof = Fraction(str(float(mp.mpf(10) ** rng.uniform(-3.5, 0.5))))
        tail = upper_tail(mp.mpf(10) ** rng.uniform(146.5, 152.5), dof)
        risk = Fraction(mp.nstr(tail, 60))
        if LEAST <= risk and LEAST <= HALF - risk:
            cases.append((risk, dof))
    return cases


def main(argv):
    extra = int(argv[1]) if len(argv) > 1 else 1500
    seed = int(argv[2]) if len(argv) > 2 else 13
    rng = random.Random(seed)
    cases = [(r, Fraction(d)) for d in DOFS for r in RISKS]
    for _ in range(extra):
        digits = rng.randrange(1, 25)
        if rng.random() < 0.5:        # a risk next to one half, with many nines
            risk = HALF - Fraction(rng.randrange(1, 10 ** 6), 10 ** (6 + rng.randrange(0, 40)))
        else:
            risk = Fraction(rng.randrange(1, 5 * 10 ** digits), 10 ** (digits + 1 + rng.randrange(0, 12)))
        dof = Fraction(str(math.exp(rng.uniform(math.log(0.004), math.log(1e8)))))
        if 0 < risk < HALF:
            cases.append((risk, dof))
    failures = report("cases ", cases)
    failures += report("next to the end of the range ", range_end(random.Random(seed + 1), extra))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
