# replay: exact
"""Chapter 13: within-run and between-run."""
import math
import random
import statistics as st

RUNS, ITERS = 8, 10
SIGMA_B, SIGMA_W = 4.0, 1.5  # ms


def benchmark(rng):
    """Each process launch draws its own
    offset (layout, caches, placement);
    iterations inside a launch add noise."""
    data = []
    for _ in range(RUNS):
        base = 250 + rng.gauss(0, SIGMA_B)
        data.append([base + rng.gauss(0, SIGMA_W)
                     for _ in range(ITERS)])
    return data


def components(data):
    """One-way random-effects analysis of
    variance for a balanced design, two runs
    or more with the same two iterations or
    more in each: mean squares between and
    within runs, and the variance between.
    For other designs see unbalanced()."""
    sizes = {len(run) for run in data}
    if (len(data) < 2 or len(sizes) != 1
            or min(sizes) < 2):
        raise ValueError("need two runs or more, "
                         "all of one length >= 2")
    n = sizes.pop()
    means = [st.fmean(run) for run in data]
    grand = st.fmean(means)
    ms_b = n * st.variance(means)
    ms_w = st.fmean(st.variance(r) for r in data)
    var_b = max(0.0, (ms_b - ms_w) / n)
    return grand, ms_b, ms_w, var_b


def unbalanced(data):
    """The same analysis for runs of unequal
    length: the squares between runs weighted
    by the lengths, those within pooled, and
    an effective length in place of the
    common one. The mean is that of all
    iterations. The order of the runs does
    not matter."""
    sizes = [len(run) for run in data]
    k, total = len(sizes), sum(sizes)
    if k < 2 or total <= k or min(sizes) < 1:
        raise ValueError("need two runs or more,"
                         " two iterations in one")
    grand = sum(map(sum, data)) / total
    means = [st.fmean(run) for run in data]
    between = sum(n * (m - grand) ** 2
                  for n, m in zip(sizes, means))
    within = sum((x - m) ** 2
                 for run, m in zip(data, means)
                 for x in run)
    ms_b = between / (k - 1)
    ms_w = within / (total - k)
    squares = sum(n * n for n in sizes)
    size = (total - squares / total) / (k - 1)
    var_b = max(0.0, (ms_b - ms_w) / size)
    return grand, ms_b, ms_w, var_b


def u_plan(var_b, var_w, runs, iters):
    """Standard uncertainty of the grand mean
    for a planned design."""
    return math.sqrt(var_b / runs
                     + var_w / (runs * iters))


def budget(var_b, var_w, startup, total):
    """The balanced design with the smallest
    uncertainty of the grand mean in the time
    of `total` iterations, where a launch
    costs that of `startup` iterations before
    its first: runs, iterations, and u."""
    best = None
    for iters in range(1, total - startup + 1):
        runs = total // (startup + iters)
        if runs:
            u = u_plan(var_b, var_w, runs, iters)
            if best is None or u < best[2]:
                best = runs, iters, u
    return best


def main():
    data = benchmark(random.Random(13))
    grand, ms_b, ms_w, var_b = components(data)
    flat = [x for run in data for x in run]
    n = len(flat)
    naive = st.stdev(flat) / math.sqrt(n)
    true_u = math.sqrt(ms_b / n)
    icc = var_b / (var_b + ms_w)
    s_w, s_b = ms_w ** 0.5, var_b ** 0.5
    rows = (("grand mean, ms", grand),
            ("repeatability s_w, ms", s_w),
            ("between-run s_b, ms", s_b),
            ("intraclass correlation", icc),
            ("naive u (n = 80), ms", naive),
            ("u from run means, ms", true_u))
    for name, value in rows:
        print(f"{name:<24}{value:>8.2f}")
    print("design, runs x iterations: u, ms")
    plans = ((8, 10), (40, 2), (80, 1))
    for runs, iters in plans:
        u = u_plan(var_b, ms_w, runs, iters)
        print(f"{runs:>4} x {iters:<3}{u:>6.2f}")
    print("a launch costs 16 iterations, "
          "323 in all:")
    best = budget(var_b, ms_w, 16, 323)[:2]
    for runs, iters in ((19, 1), best, (12, 10)):
        u = u_plan(var_b, ms_w, runs, iters)
        print(f"{runs:>4} x {iters:<3}{u:>7.3f}")
    uneven = ([9, 11], [11, 11, 13, 13],
              [13, 13, 13, 15, 15, 15])
    all_its, _, _, between = unbalanced(uneven)
    by_run = st.fmean(map(st.fmean, uneven))
    print("runs of 2, 4 and 6 iterations:")
    print(f"  mean of runs {by_run:.2f} ms, of "
          f"iterations {all_its:.2f} ms")
    print("  variance between runs "
          f"{between:.2f} ms^2")


if __name__ == "__main__":
    main()
