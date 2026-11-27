"""An optional check of the Student t quantile, outside the unit tests.

In Versions 12 and 13 the checker of swmetro/record.py allowed a relative
error of

    DELTA / (1 - 2 risk),    DELTA = 1e-6,

for the quantile t_isf(risk, dof) that swmetro/dist.py computes from a
float of the risk. Since Version 14 the checker computes the factor for a
risk above one quarter from the distance of the risk from one half and
allows DELTA at every risk; tools/certify_coverage_factor.py certifies
that. This program is kept as a check of t_isf itself for a float tail. It
certifies, in 60-digit arithmetic, that the true quantile lies within one
hundredth of the allowance above of the computed one: for a computed k and
a = allowance / 100 it verifies

    P(T > k (1 - a))  >=  risk  >=  P(T > k (1 + a)),

which brackets the true quantile without finding it. Where t_isf refuses a
quantile as too large, it verifies that the quantile is indeed beyond 1e150,
the ceiling that t_isf states. Until Version 66 the search stopped at 2**498,
the last doubling below that ceiling, and so did this check.

It needs mpmath, which the examples and the tests do not:

    python -m pip install mpmath==1.4.1
    python tools/certify_t_quantile.py            # about a minute
    python tools/certify_t_quantile.py 5000 3     # more cases, seed 3

It covers tail probabilities from 1e-300 to 0.5 - 1e-15 and degrees of
freedom from 0.004 to infinity, on a grid and at random.
"""
import math
import random
import sys
from pathlib import Path

import mpmath as mp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from swmetro.dist import t_isf          # noqa: E402
from swmetro.record import DELTA        # noqa: E402

mp.mp.dps = 60
MARGIN = 100
DOFS = (0.004, 0.01, 0.05, 0.1, 0.3, 0.5, 1, 1.5, 2, 3, 4, 4.79, 7, 12.3, 30, 100, 1e3, 1e4, 3e4, 1e5, 3e5,
        5e5, 9.9e5, 1e6, 1.0000001e6, 2e6, 1e7, 1e9, 1e15, 1e300, math.inf)
TAILS = (1e-300, 1e-200, 1e-100, 1e-30, 1e-12, 1e-9, 1e-6, 1e-4, 0.001, 0.01, 0.025, 0.05, 0.1, 0.2, 0.25,
         0.3, 0.4, 0.45, 0.49, 0.499, 0.4999, 0.499999, 0.49999999, 0.4999999999, 0.5 - 1e-13, 0.5 - 1e-15)


def upper_tail(t, dof):
    """P(T > t) for Student's t, to the working precision of mpmath."""
    t = mp.mpf(t)
    if dof > 1e40:      # indistinguishable from the normal at 60 digits
        return mp.erfc(t / mp.sqrt(2)) / 2
    v = mp.mpf(dof)
    return mp.betainc(v / 2, mp.mpf(1) / 2, 0, v / (v + t * t), regularized=True) / 2


def main(argv):
    extra = int(argv[1]) if len(argv) > 1 else 1500
    rng = random.Random(int(argv[2]) if len(argv) > 2 else 11)
    cases = [(p, v) for v in DOFS for p in TAILS]
    for _ in range(extra):
        low = rng.choice((1e-300, 1e-12, 1e-4))
        cases.append((math.exp(rng.uniform(math.log(low), math.log(0.4999))),
                      math.exp(rng.uniform(math.log(0.004), math.log(1e8)))))
    certified = refused = 0
    failed = []
    for p, v in cases:
        try:
            k = mp.mpf(t_isf(p, v))
        except ValueError:
            refused += 1
            if not upper_tail(mp.mpf(10) ** 150, v) > p:
                failed.append(("refused without need", p, v))
            continue
        a = mp.mpf(DELTA.numerator) / DELTA.denominator / (1 - 2 * mp.mpf(p)) / MARGIN
        if upper_tail(k * max(0, 1 - a), v) >= mp.mpf(p) >= upper_tail(k * (1 + a), v):
            certified += 1
        else:
            failed.append((p, v, float(k)))
    print(f"cases {len(cases)}: certified {certified}, rightly refused {refused - sum(len(f) == 3 and f[0] == 'refused without need' for f in failed)}, "
          f"failed {len(failed)}")
    for f in failed[:20]:
        print("  ", f)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
