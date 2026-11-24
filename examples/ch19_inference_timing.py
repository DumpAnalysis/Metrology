# replay: live
"""Chapter 19: cold start and steady state."""
import statistics as st
import time

TOKENS = 64  # tokens "generated" per call
CALLS, WARM_UP = 60, 10
_table = None


def infer():
    """Stand-in for a model call. The first
    call builds a table, as a real one loads
    weights or compiles a graph."""
    global _table
    if _table is None:
        _table = [i * i % 251
                  for i in range(300_000)]
    acc = 0
    for i in range(0, 300_000, 7):
        acc += _table[i]
    return acc


def main():
    now = time.perf_counter_ns
    lat = []
    for _ in range(CALLS):
        t0 = now()
        infer()
        lat.append((now() - t0) / 1e6)
    warm = lat[WARM_UP:]
    med = st.median(warm)
    mad = st.median(abs(x - med) for x in warm)
    # u of a median: asymptotic, for independent
    # calls with a roughly normal spread
    u = 1.2533 * 1.4826 * mad / len(warm) ** 0.5
    rate = TOKENS / med * 1000
    total = TOKENS * len(warm) / sum(warm) * 1000
    mean = st.fmean(lat)
    for name, ms in (("first call", lat[0]),
                     ("mean of all calls", mean),
                     ("steady median", med),
                     ("u(median)", u)):
        print(f"{name:<19}{ms:>9.3f} ms")
    u_rate = rate * u / med
    for name, v in (("rate at median", rate),
                    ("u(rate)", u_rate),
                    ("throughput", total)):
        print(f"{name:<15}{v:>9.0f} tokens/s")


if __name__ == "__main__":
    main()
