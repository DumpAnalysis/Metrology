# replay: live
"""Chapter 11: one list, four measurands."""
import sys
import tracemalloc

N = 100_000


def main():
    tracemalloc.start()
    before, _ = tracemalloc.get_traced_memory()
    data = [float(i) for i in range(N)]
    after, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    shallow = sys.getsizeof(data)
    inner = sum(map(sys.getsizeof, data))
    deep = shallow + inner
    live, top = after - before, peak - before
    rows = (("container only", shallow),
            ("container + elements", deep),
            ("traced, still live", live),
            ("traced, peak", top))
    for name, size in rows:
        kib = size / 1024
        print(f"{name:<22}{kib:>9.1f} KiB")


if __name__ == "__main__":
    main()
