# replay: exact
"""Chapter 6: what matched exchanges between
two hosts leave open about their clocks."""
import math
from fractions import Fraction
from itertools import product

# A sends, B receives, B replies, A receives:
# the stamps of two exchanges, in ms.
EXCHANGES = ((100, 98, 99, 105),
             (200, 196.5, 197.5, 204.5))
RATE = 0.020  # most that the offset moves, ms/s
# A third host C exchanges with B and with A.
B_TO_C = (40, 42, 43, 42)
A_TO_C = (60, 61, 62, 65)


def exact(x):
    """A number as it is printed, 0.1 as 1/10:
    no float holds every offset of two clocks."""
    return Fraction(str(x))


def offset(a1, b2, b3, a4, err=0):
    """The least and the greatest offset of
    clock B from clock A, B less A at one
    instant, that an exchange allows. A
    stamped its request a1 and the reply a4,
    B the request b2 and its reply b3. The
    four events follow one another, the
    clocks keep one rate, and each stamp
    lies within `err` of its event."""
    stamps = (a1, b2, b3, a4, err)
    if not all(map(math.isfinite, stamps)):
        raise ValueError("need finite numbers")
    a1, b2, b3, a4, err = map(exact, stamps)
    if err < 0:
        raise ValueError("err must be >= 0")
    w = 2 * err
    lo = max(b2, b3) - a4 - w
    hi = min(b2, b3) - a1 + w
    if a1 - a4 > w or b2 - b3 > w or lo > hi:
        raise ValueError("no offset fits the "
                         "four events")
    return lo, hi


def both(spans):
    """What the intervals share, or None."""
    lo = max(s[0] for s in spans)
    hi = min(s[1] for s in spans)
    return (lo, hi) if lo <= hi else None


def gap(stamp_a, stamp_b, span, err=0):
    """The true time from an event stamped on
    A to one stamped on B: the difference of
    the stamps less each end of `span`, with
    each stamp within `err` of its event. A
    negative `err` or a span with its ends
    reversed would give holds() an order
    that is not known, and is refused."""
    d = exact(stamp_b) - exact(stamp_a)
    lo, hi = map(exact, span)
    w = 2 * exact(err)
    if w < 0:
        raise ValueError("err must be >= 0")
    if lo > hi:
        raise ValueError("no offset in the span")
    return d - hi - w, d - lo + w


def holds(stamp_a, stamp_b, span, err=0,
          rate=RATE):
    """Seconds after the exchanges below which
    two events still have one order, if the
    offset moves by `rate` at most and no
    clock is stepped: the room that gap()
    leaves beside zero, over the rate."""
    if not 0 <= rate < math.inf:
        raise ValueError("rate must be >= 0")
    lo, hi = gap(stamp_a, stamp_b, span, err)
    room = max(lo, -hi, 0)
    if not rate:
        return math.inf if room else 0.0
    return room / exact(rate)


def late(delay):
    """The bound for stamps written after
    their events, by `delay` at most: with
    half of it taken off every stamp, the
    errors are half of it either way."""
    if not 0 <= delay < math.inf:
        raise ValueError("delay must be >= 0")
    return delay / 2


def chain(first, second):
    """The offset of a third clock from a
    first, by way of a second: the two
    offsets added, end to end."""
    return (exact(first[0]) + exact(second[0]),
            exact(first[1]) + exact(second[1]))


def routes(a_to_c=A_TO_C, err=0):
    """The offset of clock C from clock A by
    their own exchange, by way of B, and by
    both, or None for that, with every stamp
    within `err` of its event."""
    spans = [offset(*e, err) for e in EXCHANGES]
    direct = offset(*a_to_c, err)
    b_c = offset(*B_TO_C, err)
    via_b = chain(both(spans), b_c)
    return direct, via_b, both([direct, via_b])


def miss(sets):
    """How far exchanges A-B, A-B, B-C and A-C
    are from one offset for B and one for C:
    the most by which a reply comes before
    its request or two ends of intervals pass
    each other. Zero or less: they agree."""
    spans, out = [], []
    for stamps in sets:
        a1, b2, b3, a4 = map(exact, stamps)
        lo = max(b2, b3) - a4
        hi = min(b2, b3) - a1
        spans.append((lo, hi))
        out += [b2 - b3, lo - hi]
    lo = max(spans[0][0], spans[1][0])
    hi = min(spans[0][1], spans[1][1])
    out += [lo - hi,
            spans[3][0] - hi - spans[2][1],
            lo + spans[2][0] - spans[3][1]]
    return max(out)


def suspects(a_to_c):
    """Each stamp that alone can be at fault,
    the other fifteen being exact: exchange,
    place in it, and the nearest time for it.
    A stamp moves an end by its own amount
    or not at all, so that time lies as far
    off as miss() says, or there is none."""
    sets = [*EXCHANGES, B_TO_C, a_to_c]
    far, found = miss(sets), []
    if far <= 0:
        return found
    for i, j in product(range(4), repeat=2):
        for move in (-far, far):
            new = [list(s) for s in sets]
            new[i][j] = exact(new[i][j]) + move
            if miss(new) <= 0:
                found.append((i, j, new[i][j]))
    return found


def third(a_to_c=A_TO_C, err=0):
    """Host C against A: each offset of
    routes(), with the time from an event
    stamped 300 ms on A to one stamped 299 ms
    on C."""
    spans = routes(a_to_c, err)
    names = ("direct", "via B", "both")
    for name, span in zip(names, spans):
        if span is None:
            print(f"{name:<12}no offset fits")
            continue
        lo, hi = map(float, span)
        g = map(float, gap(300, 299, span, err))
        print(f"{name:<12}{lo:>7.2f}{hi:>7.2f}"
              + "".join(f"{x:>+8.2f}" for x in g))


def main():
    spans = [offset(*e) for e in EXCHANGES]
    rows = list(zip(("exchange 1", "exchange 2"),
                    spans))
    rows.append(("both", both(spans)))
    print("offset of B from A, ms")
    print(f"{'':<12}{'from':>7}{'to':>7}"
          f"{'middle':>8}")
    for name, span in rows:
        lo, hi = map(float, span)
        print(f"{name:<12}{lo:>7.2f}{hi:>7.2f}"
              f"{(lo + hi) / 2:>8.2f}")
    print("stamped 3 ms earlier on B than on A:")
    print("B after A by, ms")
    for name, span in rows:
        lo, hi = map(float, gap(0, -3, span))
        print(f"{name:<12}{lo:>+7.2f}{hi:>+7.2f}"
              f"{(lo + hi) / 2:>+8.2f}")
    wide = both([offset(*e, err=0.5)
                 for e in EXCHANGES])
    lo, hi = map(float, gap(0, -3, wide, 0.5))
    print("stamps within 0.5 ms of their events:")
    a, b = map(float, wide)
    print(f"offset, both{a:>7.2f}{b:>7.2f}")
    print(f"B after A by{lo:>+7.2f}{hi:>+7.2f}")
    print("an offset that moves by 0.020 ms/s;")
    print("stamps within   B is the later")
    for e in (0, 0.1, 0.125):
        span = both([offset(*x, err=e)
                     for x in EXCHANGES])
        s = float(holds(0, -3, span, e))
        print(f"  {e:.3f} ms      " + (
            f"for less than {s:.1f} s" if s
            else "order open at once"))
    print("B after A by, stamps off by 0.2 ms")
    for name, e in (("either way", 0.2),
                    ("late only", late(0.2))):
        span = both([offset(*x, err=e)
                     for x in EXCHANGES])
        lo, hi = map(float, gap(0, -3, span, e))
        print(f"{name:<12}{lo:>+7.2f}{hi:>+7.2f}")
    print("C: offset from A and C after A by, ms")
    third()
    other = A_TO_C[:3] + (62,)
    print("the last stamp of A at 62 ms for 65:")
    third(other)
    print("all stamps within 0.05 ms:")
    third(err=0.05)
    print("one stamp at fault, with 62 ms:")
    names = ("A-B 1", "A-B 2", "B-C", "A-C")
    sets = [*EXCHANGES, B_TO_C, other]
    for i, j, t in suspects(other):
        print(f"{names[i]}, stamp {j + 1}: "
              f"{sets[i][j]} to {float(t)} ms")
    print("such stamps, by the last stamp of A:")
    counts = [len(suspects(A_TO_C[:3] + (last,)))
              for last in (59, 61, 62, 63, 64)]
    print("  59, 61, 62, 63, 64 ms:", *counts)


if __name__ == "__main__":
    main()
