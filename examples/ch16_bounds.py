# replay: exact
"""Chapter 16: deciding from an incomplete
capture."""
from fractions import Fraction

from swmetro.stats import as_count, ecdf_rank
from swmetro.stats import enclosed, exact_q


def share_bounds(total, captured, failures):
    """Bounds on the failure share of `total`
    requests when only `captured` outcomes
    were recorded: either every missing
    request succeeded, or every one failed.
    Nothing is assumed about how they were
    lost. The three counts are whole, and
    each is a part of the one before. If
    more outcomes were captured than
    requests counted, as after a counter
    was reset, the counts have to be put
    right first: the difference would
    otherwise pass for a bound. The bounds
    are exact fractions: as floats, a share
    at its limit could fall beside it."""
    total = as_count(total, "total", 1)
    captured = as_count(captured, "captured")
    failures = as_count(failures, "failures")
    if not failures <= captured <= total:
        raise ValueError("a part above its whole")
    missing = total - captured
    low = Fraction(failures, total)
    high = Fraction(failures + missing, total)
    return low, high


def verdict(low, high, limit):
    """An objective of at most `limit`, read
    as exact_q() reads a probability."""
    limit = exact_q(limit)
    if high <= limit:
        return "met"
    if low > limit:
        return "missed"
    return "unresolved"


def rank_bounds(total, captured, q):
    """Ranks among the captured latencies, in
    ascending order and counted from 1, that
    bound the q-quantile of all `total`
    requests, by the inverse convention of
    Chapter 9, when the other latencies are
    unknown: either every missing one is the
    shortest, or every one is the longest. A
    bound that the capture cannot give is
    None."""
    total = as_count(total, "total", 1)
    captured = as_count(captured, "captured")
    if captured > total:
        raise ValueError("a part above its whole")
    rank = ecdf_rank(q, total)
    low = rank - (total - captured)
    return (low if low >= 1 else None,
            rank if rank <= captured else None)


def quantile_verdict(total, q, within, unknown):
    """An objective that the q-quantile of
    `total` durations be at most a threshold,
    when `within` of them are known to be at
    or below it and `unknown` of them could be
    on either side. The others are known to
    exceed it: by a completion above it, or
    because the request was still outstanding
    at the threshold or later."""
    total = as_count(total, "total", 1)
    within = as_count(within, "within")
    unknown = as_count(unknown, "unknown")
    if within + unknown > total:
        raise ValueError("a part above its whole")
    need = ecdf_rank(q, total)
    if within >= need:
        return "met"
    if within + unknown < need:
        return "missed"
    return "unresolved"


def main():
    low, high = share_bounds(1000, 960, 20)
    print("1000 requests, 960 outcomes captured,")
    print("20 of them failures")
    lo, hi = enclosed(100 * low, 100 * high, 0)
    print(f"failure share: {lo} % to {hi} %")
    for limit in (0.01, 0.03, 0.07):
        text = verdict(low, high, limit)
        pc = 100 * limit
        print(f"  at most {pc:.0f} %: {text}")
    print("p95 of the 1000 from the latencies")
    print("captured, by their rank:")
    for captured in (960, 949):
        low, high = rank_bounds(1000, captured,
                                0.95)
        top = ", no upper bound"
        if high is not None:
            top = f" to rank {high}"
        print(f"  {captured} captured:"
              f" rank {low}{top}")
    print("p95 at most 200 ms: 910 within,")
    print("20 beyond, 70 with no completion")
    rows = (
        ("all 70 unknown", 70),
        ("40 known outstanding at 500 ms", 30),
    )
    for title, unknown in rows:
        text = quantile_verdict(1000, 0.95, 910,
                                unknown)
        print(f"  {title}:")
        print(f"    910 to {910 + unknown}"
              f" within, {text}")
    # Buckets: 928 up to 180 ms, 25 from 180
    # to 250 ms, 35 known beyond, 12 lost.
    text = quantile_verdict(1000, 0.95, 928, 37)
    print("928 within 180 ms, 25 between 180")
    print(f"and 250 ms, 12 lost: {text}")
    print("a bound at 200 ms, with k of the 25")
    print("at or below it:")
    spans = {}
    for k in range(26):
        text = quantile_verdict(1000, 0.95,
                                928 + k, 12)
        spans.setdefault(text, []).append(k)
    for text, ks in spans.items():
        print(f"  k {ks[0]} to {ks[-1]}: {text}")


if __name__ == "__main__":
    main()
