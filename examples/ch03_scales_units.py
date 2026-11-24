# replay: exact
"""Chapter 3: scale types and units."""
import statistics as st

from swmetro.units import Q

LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR",
          "CRITICAL"]
SVC_A = ["INFO"] * 8 + ["CRITICAL"] * 2
SVC_B = ["WARNING"] * 6 + ["ERROR"] * 4
CODES = {"ranks": [1, 2, 3, 4, 5],
         "weights": [1, 2, 3, 4, 50]}


def ordinal():
    """Means depend on the coding of levels;
    medians depend only on their order."""
    for name, code in CODES.items():
        num = dict(zip(LEVELS, code))
        ma = st.fmean(num[x] for x in SVC_A)
        mb = st.fmean(num[x] for x in SVC_B)
        worse = "A" if ma > mb else "B"
        print(f"{name:<8}mean A {ma:4.1f}  "
              f"mean B {mb:4.1f}  worse: {worse}")
    for name, svc in (("A", SVC_A), ("B", SVC_B)):
        ordered = sorted(svc, key=LEVELS.index)
        mid = ordered[(len(svc) - 1) // 2]
        print(f"median level of {name}: {mid}")


def prefixes():
    """Binary and decimal prefixes differ."""
    pairs = (("KiB", "kB"), ("MiB", "MB"),
             ("GiB", "GB"), ("TiB", "TB"))
    for binary, decimal in pairs:
        x = float(Q(1, binary).to(decimal).value)
        gap = 100 * (x - 1)
        print(f"1 {binary} = {x:.6g} {decimal}"
              f"  (+{gap:.1f} %)")
    print("64 GiB =", Q(64, "GiB").to("GB"))
    try:
        Q(5, "ms") + Q(4, "KiB")
    except TypeError as err:
        print("5 ms + 4 KiB:", err)


def main():
    ordinal()
    prefixes()


if __name__ == "__main__":
    main()
