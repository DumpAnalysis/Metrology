# replay: exact
"""Chapter 20: error bars for an evaluation."""
import math
import random
import statistics as st

from swmetro.stats import exchange_exact

PASSAGES, PER_PASSAGE = 60, 5


def evaluate(rng):
    """(passage, score A, score B) for each
    question. Questions on one passage share
    its difficulty; both models see the same
    questions."""
    rows = []
    for c in range(PASSAGES):
        shift = rng.gauss(0, 1.0)
        for _ in range(PER_PASSAGE):
            item = shift + rng.gauss(0, 0.7)
            a = item + rng.gauss(0, 0.5) < 0.8
            b = item + rng.gauss(0, 0.5) < 1.0
            rows.append((c, int(a), int(b)))
    return rows


def naive_se(values):
    """Standard error if all questions were
    independent draws."""
    n = len(values)
    return st.stdev(values) / math.sqrt(n)


def clustered_se(values, clusters):
    """Cluster-robust standard error: sum the
    residuals inside each cluster first. The
    factor g / (g - 1) corrects for the finite
    number g of clusters."""
    values = list(values)
    clusters = list(clusters)
    if len(clusters) != len(values):
        raise ValueError("one cluster a value")
    mean = st.fmean(values)
    tot = {}
    for v, c in zip(values, clusters):
        tot[c] = tot.get(c, 0.0) + v - mean
    ss = sum(t * t for t in tot.values())
    g, n = len(tot), len(values)
    if g < 2:
        raise ValueError("need two clusters")
    return math.sqrt(ss * g / (g - 1)) / n


def by_passage(rows):
    """For each passage, the answers that B
    has right less those that A has."""
    net = {}
    for c, a, b in rows:
        net[c] = net.get(c, 0) + b - a
    return list(net.values())


def tails(diffs):
    """The exact test by exchange of labels,
    for B ahead and two-sided, with the
    numbers of differences against B, for
    it, and of none."""
    diffs = list(diffs)
    up = exchange_exact(diffs)
    down = exchange_exact([-d for d in diffs])
    both = min(1, 2 * min(up, down))
    split = [sum(d < 0 for d in diffs),
             sum(d > 0 for d in diffs),
             sum(d == 0 for d in diffs)]
    return split, float(up), float(both)


def exchanged(rows):
    """Is the gain of B more than the labels
    explain? Three exact tests, which differ
    in what is exchanged under the
    hypothesis that the labels mean
    nothing. By question: the two labels of
    any one question, as if no question
    shared anything with another. By
    passage: the labels of all questions of
    a passage together, which keeps what
    they share, with the gain still counted
    in answers. Passages won: the same
    exchange, with a passage counted once
    for the model that has more of it
    right, which is another quantity than
    the gain."""
    rows = list(rows)
    net = by_passage(rows)
    print("B - A in answers, by passage:")
    cells = []
    for v in sorted(set(net)):
        sign = "+" if v > 0 else ""
        k = net.count(v)
        cells.append(f"{sign}{v}: {k}")
    print("  " + "  ".join(cells))
    won = [(d > 0) - (d < 0) for d in net]
    cases = (("question", [b - a
                           for _, a, b in rows]),
             ("passage, gain in answers", net),
             ("passage, won or lost", won))
    print("labels exchanged, exact tests:")
    for name, diffs in cases:
        split, up, both = tails(diffs)
        print(f"  by {name}")
        print("    for A {}, for B {}, none {}"
              .format(*split))
        print(f"    B ahead {up:.5f}, "
              f"two-sided {both:.5f}")


def main():
    rows = evaluate(random.Random(20))
    cl = [r[0] for r in rows]
    a = [r[1] for r in rows]
    b = [r[2] for r in rows]
    d = [y - x for x, y in zip(a, b)]
    n = len(rows)
    print(f"{n} questions, {PASSAGES} passages")
    print(f"{'':<10}{'score':>7}{'naive':>8}"
          f"{'clustered':>11}")
    cases = (("model A", a), ("model B", b),
             ("B - A", d))
    for name, v in cases:
        print(f"{name:<10}{st.fmean(v):>7.3f}"
              f"{naive_se(v):>8.3f}"
              f"{clustered_se(v, cl):>11.3f}")
    unpaired = math.hypot(clustered_se(a, cl),
                          clustered_se(b, cl))
    print(f"B - A, unpaired u  {unpaired:.3f}")
    exchanged(rows)


if __name__ == "__main__":
    main()
