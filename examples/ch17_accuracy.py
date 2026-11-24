# replay: exact
"""Chapter 17: accuracy as a measured value."""
import math
import random
from decimal import Decimal
from fractions import Fraction

from swmetro.stats import paired_exact, wilson


def interval_widths():
    """The same observed accuracy on test sets
    of different sizes."""
    print("observed accuracy 87.0 %")
    for n in (500, 2000, 10_000):
        lo, hi = wilson(round(0.87 * n), n)
        half = 50 * (hi - lo)
        print(f"n = {n:>5}: [{100 * lo:.1f}, "
              f"{100 * hi:.1f}] %  (±{half:.1f})")


def two_models(n, rng):
    """Correctness of models A and B on the
    same n items. Hard items are hard for
    both, so their errors are correlated."""
    a, b = [], []
    for _ in range(n):
        hard = rng.gauss(0, 1)
        a.append(hard + rng.gauss(0, 0.6) < 1.30)
        b.append(hard + rng.gauss(0, 0.6) < 1.38)
    return a, b


def same_scores():
    """The scores of two_models() again, 1722
    and 1758 right answers of 2000, with
    other joint outcomes: both right on 1500
    items for 1610, both wrong on 20 for 130."""
    a = [1] * 1722 + [0] * 278
    b = [1] * 1500 + [0] * 222 + [1] * 258
    return a, b + [0] * 20


def two_sided(only_a, only_b):
    """Exact probabilities for the pairs that
    differ, as fractions: B ahead as far as
    it is, and a division as uneven either
    way."""
    one = paired_exact(only_a, only_b)
    two = min(one, paired_exact(only_b, only_a))
    return one, min(Fraction(1), 2 * two)


def shown(p):
    """A probability to print, rounded from
    the fraction itself: six decimals, or
    three digits where those would show fewer."""
    if p >= Fraction(1, 10000) or not p:
        n = round(p * 10 ** 6)
        return f"{n // 10 ** 6}.{n % 10 ** 6:06d}"
    q = p.denominator // p.numerator
    k = Decimal(q).adjusted() + 1  # digits of q
    n = round(p * 10 ** (k + 2))
    if n == 1000:
        n, k = 100, k - 1
    return f"{n // 100}.{n % 100:02d}e-{k:02d}"


def limits(a, b, n):
    """The least and the greatest two-sided
    probability that right answers on a and
    on b of the same n items allow: with the
    most items right for both, and the fewest."""
    lead = abs(b - a)
    low = Fraction(1, 2 ** max(0, lead - 1))
    h = max(0, a + b - n)
    return low, two_sided(a - h, b - h)[1]


def partly(known, a, b, n, parts=None):
    """Each two-sided probability that right
    answers on a and on b of n items allow,
    from the most items right for both, when
    `known` pairs are verified: right for
    both, for A alone, B alone and neither.
    `parts` are batches that keep the pairs
    of the other items together: of each its
    size and the right answers of A and B."""
    known = tuple(known)
    rows = (None if parts is None
            else [tuple(row) for row in parts])
    if any(type(k) is not int or k < 0
           for k in (*known, a, b, n,
                     *sum(rows or [], ()))):
        raise ValueError("need whole counts >= 0")
    both, only_a, only_b, _ = known
    rest = n - sum(known)
    a -= both + only_a
    b -= both + only_b
    if rows is None:
        rows = [(rest, a, b)]
    if any(len(row) != 3 for row in rows):
        raise ValueError("counts do not fit")
    sums = [sum(row[i] for row in rows)
            for i in range(3)]
    if sums != [rest, a, b] or any(
            min(x, y, m - x, m - y) < 0
            for m, x, y in rows):
        raise ValueError("counts do not fit")
    most = sum(min(x, y) for m, x, y in rows)
    least = sum(max(0, x + y - m)
                for m, x, y in rows)
    return [two_sided(only_a + a - j,
                      only_b + b - j)[1]
            for j in range(most, least - 1, -1)]


def compare(a=None, b=None):
    """Models A and B on the same items, by
    default those of two_models(). A
    difference of shares is in percentage
    points. The two-sided test is the same
    whichever model is called B."""
    if a is None and b is None:
        rng = random.Random(17)
        a, b = two_models(2000, rng)
    elif a is None or b is None:
        raise ValueError("both lists, or none")
    n = len(a)
    if len(b) != n or not n:
        raise ValueError("one answer an item")
    pa, pb = sum(a) / n, sum(b) / n
    pairs = list(zip(a, b))
    only_a = sum(x and not y for x, y in pairs)
    only_b = sum(y and not x for x, y in pairs)
    diff = pb - pa
    va, vb = pa * (1 - pa), pb * (1 - pb)
    var = va + vb
    r = "undefined"
    if va * vb:
        cov = sum(x and y for x, y in pairs) / n
        r = (cov - pa * pb) / math.sqrt(va * vb)
        r = f"{r:+.3f}"
    u_unpaired = math.sqrt(var / n)
    disc = (only_a + only_b) / n
    u_paired = math.sqrt((disc - diff ** 2) / n)
    print(f"accuracy A {100 * pa:.2f} %, "
          f"B {100 * pb:.2f} %")
    print(f"only A right {only_a}, "
          f"only B right {only_b}")
    print(f"correlation of right answers {r}")
    print(f"difference B - A  {100 * diff:.2f}"
          " percentage points")
    for name, u in (("unpaired", u_unpaired),
                    ("paired", u_paired)):
        ratio = "undefined"
        if u:
            ratio = f"= {diff / u:.2f}"
        print(f"u, {name:<9}{100 * u:6.2f} "
              f"points, difference / u {ratio}")
    one, two = two_sided(only_a, only_b)
    one, two = shown(one), shown(two)
    print("exact test for the pairs that differ:")
    print(f"  B ahead, one-sided  p = {one}")
    print(f"  two-sided           p = {two}")


def main():
    interval_widths()
    compare()
    print("the same scores, other pairs:")
    compare(*same_scores())
    heads = ("two-sided p that 1722 and b allow:",
             "and with B level or behind:")
    rows = ((1727, 1758, 1767, 1768),
            (1722, 1676, 1672))
    for head, bs in zip(heads, rows):
        print(head)
        for b in bs:
            lo, hi = limits(1722, b, 2000)
            print(f"  b = {b}: {shown(lo)}"
                  f" to {shown(hi)}")
    one, two = two_sided(0, 23)
    print("23 items to B alone, none to A alone:")
    print(f"  p = {shown(one)}, two-sided "
          f"{shown(two)}")
    print("1950 pairs verified, 50 not:")
    for tag, a, w in (("as they are:", 0, 0),
                      ("6 wrong for both:", 0, 6),
                      ("10 for A alone:", 10, 0)):
        known = (1560, 132 + a, 163, 95 + w)
        ps = partly(known, 1722, 1758, 2000)
        print(f"  {tag:<18}{shown(ps[0])}"
              f" to {shown(ps[-1])}")
    known = (1560, 132, 163, 95)
    print("the 50 in two batches of 25:")
    for b1, b2 in ((23, 12), (13, 22),
                   (22, 13), (14, 21)):
        parts = [(25, 23, b1), (25, 7, b2)]
        p = partly(known, 1722, 1758, 2000, parts)
        print(f"  A 23, 7; B {b1}, {b2}: "
              f"{shown(p[0])} to {shown(p[-1])}")


if __name__ == "__main__":
    main()
