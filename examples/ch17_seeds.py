# replay: exact
"""Chapter 17: how far do seeds move a score?"""
import math
import random
import statistics as st

D, N_TRAIN = 4, 300
W_TRUE = [1.5, -1.0, 0.6, 0.0]
SEEDS = range(20)


def dot(w, x):
    acc = 0.0
    for a, b in zip(w, x):
        acc += a * b
    return acc


def sample(n, rng):
    """n labeled points from a logistic model."""
    rows = []
    for _ in range(n):
        x = [rng.gauss(0, 1) for _ in range(D)]
        p = 1 / (1 + math.exp(-dot(W_TRUE, x)))
        rows.append((x, int(rng.random() < p)))
    return rows


def train(rows, epochs, rng, lr=0.05):
    """Logistic regression by plain SGD. The
    seed sets the start and the data order."""
    w = [rng.gauss(0, 0.5) for _ in range(D)]
    rows = list(rows)
    for _ in range(epochs):
        rng.shuffle(rows)
        for x, y in rows:
            g = 1 / (1 + math.exp(-dot(w, x))) - y
            w = [a - lr * g * b
                 for a, b in zip(w, x)]
    return w


def accuracy(w, rows):
    ok = sum((dot(w, x) > 0) == bool(y)
             for x, y in rows)
    return ok / len(rows)


def scores(data, epochs, resplit):
    """Test accuracy for each seed. With
    resplit, the seed also draws the split."""
    out = []
    for seed in SEEDS:
        rng = random.Random(seed)
        rows = list(data)
        if resplit:
            rng.shuffle(rows)
        w = train(rows[:N_TRAIN], epochs, rng)
        out.append(accuracy(w, rows[N_TRAIN:]))
    return out


def main():
    data = sample(600, random.Random(100))
    for resplit in (False, True):
        a = scores(data, 1, resplit)
        b = scores(data, 3, resplit)
        d = [y - x for x, y in zip(a, b)]
        u = st.stdev(d) / math.sqrt(len(d))
        wins = sum(v > 0 for v in d)
        loses = sum(v < 0 for v in d)
        what = ("seed", "seed + split")[resplit]
        print(f"varying: {what}")
        print(f"  A, 1 epoch   {st.fmean(a):.3f}"
              f"  sd {st.stdev(a):.3f}")
        print(f"  B, 3 epochs  {st.fmean(b):.3f}"
              f"  sd {st.stdev(b):.3f}")
        print(f"  B - A        {st.fmean(d):+.4f}"
              f"  u {u:.4f}")
        print(f"  B ahead in {wins} seeds, "
              f"behind in {loses}, of {len(d)}")


if __name__ == "__main__":
    main()
