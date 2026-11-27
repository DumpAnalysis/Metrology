# replay: exact
"""Chapter 23: a change of request mix."""
import math
import random
import statistics as st
from decimal import Decimal
from fractions import Fraction

from swmetro.stats import as_count, downward
from swmetro.stats import exact_q, known, upward
from swmetro.stats import weighted_quantile

# (class, requests, mean duration in ms)
BEFORE = (("fast", 900, 100.0),
          ("slow", 100, 500.0))
AFTER = (("fast", 100, 90.0),
         ("slow", 900, 450.0))
SIGNS = (">", ">=")  # of a bound on a duration
ENDS = (math.inf, -math.inf)


def durations(n, mean, rng):
    """n durations with a log-normal shape, at
    this mean within the rounding of floats."""
    zs = [rng.lognormvariate(0, 0.3)
          for _ in range(n)]
    scale = mean / st.fmean(zs)
    return [z * scale for z in zs]


def record(spec, rng):
    """{class: durations} of one period."""
    return {name: durations(n, mean, rng)
            for name, n, mean in spec}


def absent(rec, mix):
    """Classes that carry weight in `mix` and
    have no observation in `rec`, with their
    shares of the mix."""
    total = sum(len(xs) for xs in mix.values())
    return {name: Fraction(len(xs), total)
            for name, xs in mix.items()
            if xs and not rec.get(name)}


def added(rec, mix):
    """Classes observed in `rec` that carry no
    weight in `mix`. At that mix they count
    for nothing, which is right for a class
    that did not exist then, and wrong for
    one that was renamed or went unrecorded."""
    rec, mix = both(rec, mix)
    return [name for name, xs in rec.items()
            if xs and not mix.get(name)]


def weighted(rec, mix=None, partial=False,
             fill=None):
    """(duration, weight) pairs. Without `mix`
    every request weighs 1, the workload as
    observed. With `mix`, the classes that
    carry weight in it are reweighted to its
    class counts, and a class it does not
    have is left out. A class of the mix
    without observations cannot be
    reweighted, and the result at that mix
    is then not determined: the call fails,
    unless `partial` asks for the observed
    classes alone, which describe those
    classes and nothing more, or `fill`
    gives a duration at which such a class
    is to stand with its whole weight. With
    -inf and inf for it the two results
    enclose whatever the missing durations
    were, for a statistic that does not fall
    when a duration rises."""
    if mix is None:
        return [(x, Fraction(1))
                for xs in classes(rec).values()
                for x in xs]
    rec, mix = both(rec, mix)
    if not any(mix.values()):
        raise ValueError("empty reference mix")
    gone = absent(rec, mix)
    if gone and not partial and fill is None:
        names = ", ".join(gone)
        raise ValueError("no observations for "
                         + names)
    out = []
    for name, ref in mix.items():
        xs = rec.get(name)
        if ref and xs:
            w = Fraction(len(ref), len(xs))
            out += [(x, w) for x in xs]
        elif ref and fill is not None:
            out.append((fill, Fraction(len(ref))))
    return out


def mean(pairs):
    """The weighted mean: one division of two
    exact sums, rounded once, to a float, if
    a duration is a float. Summed as floats,
    three durations of 200 ms at a weight of
    1/3 have a mean below 200 ms. An infinite
    fill is the mean, and with a duration
    that is no number there is none."""
    pairs = [(x, Fraction(w)) for x, w in pairs]
    total = sum(w for _, w in pairs)
    if not total:
        raise ValueError("no requests")
    xs = [x for x, _ in pairs]
    odd = [x for x in xs if x != x or x in ENDS]
    if odd:
        return sum(map(float, odd))
    top = sum(Fraction(x) * w for x, w in pairs)
    if any(isinstance(x, float) for x in xs):
        return float(top / total)
    return top / total


def p95(pairs):
    """Inverse empirical CDF, with weights,
    which are summed exactly whatever their
    type; see weighted_quantile()."""
    pairs = list(pairs)
    if not sum(w for _, w in pairs):
        raise ValueError("no requests")
    return weighted_quantile(pairs, 0.95)


def unknown(x):
    """A request that is in the record with
    no duration: None, or not a number."""
    if isinstance(x, Decimal):
        return x.is_nan()
    return x is None or x != x


def observed(xs):
    """The entries of a class as a list, the
    list itself if it is one: a class that can
    be read only once is read here, and none
    of its requests is lost to the check. Each
    entry is a duration that known() admits: a
    flag is none, and Python would count False
    as 0. What side() reads otherwise stays:
    None, a NaN, an infinity, and a bound, a
    pair of ">" or ">=" and such a duration.
    A duration has elapsed: none is below 0."""
    if not isinstance(xs, list):
        xs = list(xs)
    form = "a bound is > or >="
    for x in xs:
        if isinstance(x, tuple):
            if len(x) != 2 or x[0] not in SIGNS:
                raise ValueError(form)
            x = known([x[1]], "bound")[0]
        elif unknown(x) or x == math.inf:
            continue
        if known([x])[0] < 0:
            raise ValueError("a time below zero")
    return xs


def cursors(rec):
    """Every class as it is, if it is a list,
    and anything else as the one cursor that
    iter() gives for it. A cursor under two
    names would be empty under the second:
    it is refused before it is read."""
    cs = {k: v if isinstance(v, list) else iter(v)
          for k, v in rec.items()}
    once = [id(v) for v in cs.values()
            if not isinstance(v, list)]
    if len(set(once)) < len(once):
        raise ValueError("stream in two classes")
    return cs


def classes(rec):
    """The record with every class as a list,
    held to observed(): every class, also one
    without weight in a mix, each read once."""
    return {k: observed(v)
            for k, v in cursors(rec).items()}


def periods(*recs):
    """The periods of an analysis that passes
    over them more than once: each is read
    here, once, by classes(). A cursor in
    two periods would be empty in the
    second, and is refused as well."""
    cs = [cursors(rec) for rec in recs]
    once = [id(v) for c in cs for v in c.values()
            if not isinstance(v, list)]
    if len(set(once)) < len(once):
        raise ValueError("stream in two periods")
    return [classes(c) for c in cs]


def size(ref, seen, ahead):
    """How many entries: the count kept in
    `seen` for the cursor of `ref`, or the
    length in `ahead`, or one reading of
    the cursor, which is then kept."""
    cur = iter(ref)
    if id(cur) in seen:
        return seen[id(cur)][1]
    if id(ref) in ahead:
        return ahead[id(ref)]
    seen[id(cur)] = cur, sum(1 for _ in cur)
    return seen[id(cur)][1]


def both(rec, mix):
    """The record by classes(), and the mix
    as ranges of its class counts: entries
    of the mix stand for requests and are
    only counted. An object and a cursor are
    counted once and kept, for every class
    that names the one or reads the other.
    The lengths of the mix are taken before
    the record is read: a view of a stream
    may count only what is left of it."""
    cs = cursors(rec)
    ahead = {id(x): len(x) for x in mix.values()
             if hasattr(x, "__len__")}
    held = classes(cs)
    seen = {id(x): (x, len(held[k])) for k in rec
            for x in (rec[k], cs[k])}
    sizes = {}
    for name, ref in mix.items():
        if id(ref) not in seen:
            n = size(ref, seen, ahead)
            seen[id(ref)] = ref, n
        sizes[name] = range(seen[id(ref)][1])
    return held, sizes


def side(x, limit):
    """True if the duration x is at or below
    `limit`, False if it is above it, and
    None if the record leaves that open: x
    is unknown, or only a bound for it is
    known, a pair of ">" or ">=" and a time.
    Every duration is at or below a limit of
    infinity, an unknown one too, and none
    is at or below a limit below zero."""
    if limit == math.inf:
        return True
    if limit < 0:
        return False
    if unknown(x):
        return None
    if not isinstance(x, tuple):
        return bool(x <= limit)
    sign, bound = x
    if sign == ">":
        over = bound >= limit
    elif sign == ">=":
        over = bound > limit
    else:
        raise ValueError("a bound is > or >=")
    return False if over else None


def within(rec, mix, limit):
    """Bounds on the share of requests at or
    below `limit` at the class counts of
    `mix`. A class contributes what its
    durations show. A request that side()
    leaves open may lie on either side, and
    a class without observations is one
    request of that kind: it contributes
    anything from none of its share to all
    of it."""
    rec, mix = both(rec, mix)
    total = sum(len(xs) for xs in mix.values())
    if not total:
        raise ValueError("empty reference mix")
    if unknown(limit):
        raise ValueError("no limit")
    if limit not in ENDS:
        known([limit], "limit")
    low = high = 0
    for name, ref in mix.items():
        xs = rec.get(name) or [None]
        size = len(xs) * total
        sides = [side(x, limit) for x in xs]
        ok = sum(s is True for s in sides)
        maybe = ok + sum(s is None for s in sides)
        low += Fraction(ok * len(ref), size)
        high += Fraction(maybe * len(ref), size)
    return low, high


def most_within(n, mean, limit, cap=None):
    """The most of n durations, none below
    zero, that can lie at or below `limit`
    when their mean is known and, if `cap`
    is given, none of them is above it. The
    numbers are read as they print, before
    any of them is compared."""
    n = as_count(n, least=1)
    mean, limit = (Fraction(str(x))
                   for x in (mean, limit))
    if min(mean, limit) < 0:
        raise ValueError("a time below zero")
    if cap is not None:
        cap = Fraction(str(cap))
    if cap is not None and cap < mean:
        raise ValueError("a mean above its cap")
    if mean <= limit:
        return n
    if cap is None:
        return n - 1
    return n * (cap - mean) // (cap - limit)


def verdict(low, high, need=0.95):
    """A percentile objective, from bounds on
    the share within its limit: the p95 is
    at most the limit if at least 95 % of
    the requests are. `low` and `high` are
    exact shares, the fractions that within()
    returns; a float among them is the binary
    number that it is. `need` is read as
    exact_q() reads it. For a need of 0 the
    percentile is the least duration, and
    some share has to be within the limit."""
    need = exact_q(need)
    if low >= need and low > 0:
        return "met"
    if high < need or not high:
        return "missed"
    return "unresolved"


def with_one(x, mean, n=900):
    """most_within() for n durations with this
    mean, none above 810 ms, when one of them
    is known to be x: that one, if it is
    within 200 ms, and the most of the others
    at the mean that x leaves them. The
    numbers are read as they print."""
    n = as_count(n, least=1)
    x, mean = (Fraction(str(v))
               for v in (x, mean))
    if not 0 <= x <= 810:
        raise ValueError("x beyond the cap")
    if n == 1:
        if x != mean:
            raise ValueError("x is not the mean")
        return int(x <= 200)
    rest = (n * mean - x) / (n - 1)
    return (x <= 200) + most_within(
        n - 1, rest, 200, 810)


def lost_class(before, after):
    """The second period without the records
    of its slow class."""
    before, after = periods(before, after)
    part = {"fast": after["fast"]}
    gone = absent(part, before)
    for name, share in gone.items():
        pct = f"{100 * float(share):.0f} %"
        print(f"after, no records of the {name}")
        print(f"class, {pct} of the mix"
              " of before:")
    print("  mean and p95 at that mix:")
    print("    not determined")
    alone = weighted(part, before, partial=True)
    print("  fast class alone:")
    print(f"    mean {mean(alone):.0f},"
          f" p95 {p95(alone):.0f}")
    cases = (("all slow requests at 450 ms",
              [450.0] * 900),
             ("all slow requests at 900 ms",
              [900.0] * 900),
             ("half at 300 ms, half at 600 ms",
              [300.0] * 450 + [600.0] * 450))
    for text, slow in cases:
        full = weighted(dict(part, slow=slow),
                        before)
        print(f"  {text}:")
        print(f"    mean {mean(full):.0f},"
              f" p95 {p95(full):.0f}")
    low, high = within(part, before, 200)
    print("  within 200 ms: "
          f"{100 * float(low):.1f} % to"
          f" {100 * float(high):.1f} %")
    print("  p95 at most 200 ms:"
          f" {verdict(low, high)}")
    slow, share = len(after["slow"]), gone["slow"]
    print("  with the slow mean of 450 ms and")
    for cap in (None, 800, 810):
        most = most_within(slow, 450, 200, cap)
        top = low + share * most / slow
        text = ("no other limit" if cap is None
                else f"none above {cap} ms")
        print(f"  {text}:")
        print(f"    to {100 * float(top):.2f} %,"
              f" {verdict(low, top)}")
    print("  none above 810 ms, one recovered")
    for x in (100, 110, 700):
        most = with_one(x, 450, slow)
        top = low + share * most / slow
        print(f"    at {x} ms: to"
              f" {100 * float(top):.2f} %,"
              f" {verdict(low, top)}")
    print("  at 100 ms, the mean as it prints")
    for digits in (2, 1, 0):
        half = Fraction(5, 10 ** (digits + 1))
        most = with_one(100, 450 - half, slow)
        top = low + share * most / slow
        print(f"    {450:.{digits}f} ms: to"
              f" {100 * float(top):.2f} %,"
              f" {verdict(low, top)}")
    print("  720 ms, of one request or of all")
    rows = (("one", with_one(720, 450, slow)),
            ("the longest",
             most_within(slow, 450, 200, 720)))
    for text, most in rows:
        top = low + share * most / slow
        print(f"    {text}: to"
              f" {100 * float(top):.2f} %,"
              f" {verdict(low, top)}")
    low, gap = Fraction(98, 100), Fraction(2, 100)
    print("  98 % known within, 2 % missing:"
          f" {verdict(low, low + gap)}")


def one_unknown():
    """Ninety fast and ten slow requests, one
    of the slow without a duration, then
    with a bound for it, and with infinity
    in the place of the bound."""
    rest = [100.0] * 4 + [300.0] * 5
    cases = (("of the slow without a duration:",
              None, (200,)),
             ("and known only to exceed 200 ms:",
              (">", 200), (200, 250)),
             ("and stored as infinity:",
              math.inf, (250,)))
    print("90 fast requests and 10 slow, one")
    for text, last, limits in cases:
        rec = {"fast": [100.0] * 90,
               "slow": rest + [last]}
        print(text)
        for limit in limits:
            low, high = within(rec, rec, limit)
            print(f"  within {limit} ms: "
                  f"{100 * float(low):.0f} % to"
                  f" {100 * float(high):.0f} %")
            print(f"  p95 at most {limit} ms:"
                  f" {verdict(low, high)}")


def two_missing():
    """Eighteen fast requests and two slow
    ones without a duration: the slow class
    as a list, as an iterator, which can be
    read once, and with its two rows lost on
    the way. What is missing is not what was
    discarded."""
    gone = [None, None]
    cases = (("in a list:", gone),
             ("in an iterator:", iter(gone)),
             ("dropped:", None))
    print("18 fast requests and 2 slow without")
    print("a duration, within 200 ms")
    for text, slow in cases:
        rec = {"fast": [100.0] * 18}
        if slow is not None:
            rec["slow"] = slow
        rec = classes(rec)
        low, high = within(rec, rec, 200)
        print(f"  {text:<16}"
              f"{100 * float(low):.0f} % to"
              f" {100 * float(high):.0f} %,"
              f" {verdict(low, high)}")


def new_class(before, after):
    """The second period with a class that the
    first did not have: 100 bulk requests of
    1000 ms."""
    before, after = periods(before, after)
    more = dict(after, bulk=[1000.0] * 100)
    for name in added(more, before):
        print(f"after, with a new class, {name}:")
    print(f"{'workload':<12}{'mean':>9}"
          f"{'p95':>9}{'<=200 ms':>10}")
    rows = (("after", after, None),
            ("with bulk", more, None),
            ("at old mix", more, before))
    for text, rec, mix in rows:
        pairs = weighted(rec, mix)
        total = sum(w for _, w in pairs)
        ok = sum(w for x, w in pairs if x <= 200)
        share = 100 * float(ok / total)
        print(f"{text:<12}{mean(pairs):>9.2f}"
              f"{p95(pairs):>9.2f}"
              f"{share:>7.2f} %")
    components(before, more)


def exact(pairs):
    """The pairs with every finite duration as
    the fraction that it is, so that a
    statistic of them is rounded neither
    before it is compared nor before it is
    subtracted. What is no finite number
    stays what it is."""
    out = []
    for x, w in pairs:
        try:
            x = Fraction(x)
        except (TypeError, ValueError,
                OverflowError):
            pass
        out.append((x, w))
    return out


def shown(x, plus=""):
    """x in whole milliseconds, rounded once,
    half to even, from the number that it
    is. With plus="+" the sign is written,
    and a change that rounds to zero keeps
    its own."""
    if not isinstance(x, Fraction):
        return f"{x:{plus}.0f}"
    n = round(x)
    if plus and not n:
        return "-0" if x < 0 else "+0"
    return f"{n:{plus}d}"


def is_open(x):
    """A request whose duration the record
    leaves open: unknown, or a bound."""
    return unknown(x) or isinstance(x, tuple)


def ends(pairs):
    """The pairs twice: with every open
    duration at its lower bound, and at
    infinity. The lower bound of an unknown
    duration is 0, and that of a bound is
    its time, which a strict bound does not
    attain."""
    low, top = [], []
    for x, w in pairs:
        least = x
        if unknown(x):
            least = 0
        elif isinstance(x, tuple):
            least = x[1]
        most = math.inf if is_open(x) else x
        low.append((least, w))
        top.append((most, w))
    return low, top


def reach(stat, pairs):
    """(lower bound, upper bound) of a
    statistic that does not fall when a
    duration rises, over everything that the
    open durations can have been: the same
    number twice where they do not matter.
    An end that is no number, as the mean of
    an infinity above and one below, is no
    limit."""
    low, top = ends(pairs)
    least = most = stat(exact(low))
    if top != low:
        most = stat(exact(top))
    if least != least:
        least = -math.inf
    if most != most:
        most = math.inf
    return least, most


def limits(low, top):
    """Two ends as a span. An infinity less
    an infinity is no number. As one end it
    is no limit on that side. Where neither
    end is a number the difference is not
    defined, which is not the same as a
    difference that can be anything, and
    both ends stay what they are."""
    if low != low and top != top:
        return low, top
    if low != low:
        low = -math.inf
    if top != top:
        top = math.inf
    return low, top


def apart(new, old):
    """The span of the change from one span
    to another."""
    return limits(new[0] - old[1],
                  new[1] - old[0])


def one(span):
    """Is the span one finite number?"""
    return span[0] == span[1] not in ENDS


def edge(x, up, plus=""):
    """A limit of a span as text, to the
    hundredth of a millisecond, rounded to
    its safe side: a lower limit downward
    and an upper limit upward, so that what
    is printed excludes nothing that the
    record allows. Rounded to the nearest, a
    lower limit of 199.95 ms would print as
    200 ms. Zeros at the end are left out."""
    if x in ENDS:
        return shown(x, plus)
    text = (upward if up else downward)(x, 2)
    text = text.rstrip("0").rstrip(".")
    if plus and text[0] != "-":
        text = plus + text
    return text


def spanned(span, plus="", unit=""):
    """A span as text: one number, rounded as
    shown() rounds it, or its limits, each
    by edge()."""
    low, top = span
    if low != low:
        return "not defined"
    if low == top:
        return shown(low, plus) + unit
    if (low, top) == (-math.inf, math.inf):
        return "no limit"
    a, b = edge(low, 0, plus), edge(top, 1, plus)
    if top == math.inf:
        return f"{a}{unit} or more"
    if low == -math.inf:
        return f"{b}{unit} or less"
    return f"{a} to {b}{unit}"


def shares(rec):
    """The mix of a period: the share of each
    class in its requests."""
    total = sum(len(xs) for xs in rec.values())
    return {name: Fraction(len(xs), total)
            for name, xs in rec.items() if xs}


def blend(parts):
    """The span of a sum of class means, each
    a span with a coefficient. A mean with
    the coefficient zero drops out, whatever
    is known of it. A class without requests
    has no mean, and with any other
    coefficient the sum is not determined:
    None."""
    low = top = 0
    for k, span in parts:
        if not k:
            continue
        if span is None:
            return None
        a, b = sorted((k * span[0], k * span[1]))
        low, top = low + a, top + b
    return limits(low, top)


def of_mean(before, after, f):
    """For the mean: the statistic on the way,
    the change within the classes and the
    change of the mix, each a span or None,
    with the share f of the later mix in the
    weights of the classes. f = 0 changes
    the class means first and f = 1 the mix;
    f = 1/2 is the average of the two orders
    and has no way. Each is a sum of class
    means with coefficients, in which every
    open duration stands once, and one whose
    coefficient is zero not at all: a class
    that keeps its share adds nothing to the
    change of the mix, whatever its
    durations were."""
    s, t = shares(before), shares(after)
    m, n = ({k: reach(mean, [(x, 1) for x in v])
             for k, v in rec.items() if v}
            for rec in (before, after))
    via, inner, mixed = [], [], []
    for c in {*s, *t}:
        a, b = s.get(c, 0), t.get(c, 0)
        x, y = m.get(c), n.get(c)
        w = (1 - f) * a + f * b
        via.append((b, x) if f else (a, y))
        inner += [(w, y), (-w, x)]
        mixed += [((1 - f) * (b - a), y),
                  (f * (b - a), x)]
    return blend(via), blend(inner), blend(mixed)


def components(before, after, stat=mean,
               name="mean", what="means"):
    """The change of a statistic of all the
    requests as a change within the classes
    plus a change of the mix, in the two
    orders in which the two can be changed,
    and the average of the two. Each order
    passes through the statistic of one
    period's durations at the mix of the
    other. A mean needs the class means for
    that, and a percentile the durations of
    the classes. Where a class of that mix
    has no durations, the statistic is taken
    with the class below every duration and
    above every duration. If the two agree,
    the missing durations do not matter, as
    they may not for a percentile. If they
    differ, as they do for a mean, the order
    is not determined. A request that is in
    the record with its duration unknown, or
    known only to exceed a time, is still a
    request of its class: the statistic is
    taken with every such duration at its
    lower bound and at infinity, and what is
    printed is the span between the two.
    With the same mix in both periods the
    whole change is within the classes,
    whatever the durations were. A mean is
    a sum over the classes, and of_mean()
    gives each part of its change as such a
    sum. A percentile depends on ranks: an
    order whose way is a span is not
    determined for it. The spans enclose
    only a statistic that does not fall when
    a duration rises. A statistic of another
    quantity, such as the distance of a
    duration from a target, has to be given
    records of that quantity. Every
    statistic is taken from exact durations,
    the differences and their average are
    exact, and each number is rounded where
    it is printed, and not before."""
    before, after = periods(before, after)
    start = reach(stat, weighted(before))
    end = reach(stat, weighted(after))
    change = apart(end, start)
    print(f"{name}, {spanned(start)} ->"
          f" {spanned(end, unit=' ms')}:"
          f" {spanned(change, '+', ' ms')}")
    for rec, when in ((before, "before"),
                      (after, "after")):
        xs = [x for v in rec.values() for x in v]
        k = sum(map(is_open, xs))
        if k:
            print(f"  open {when}: {k} of"
                  f" {len(xs)} durations")
    if (shares(before) == shares(after)
            and not (one(start) and one(end))):
        inner = spanned(change, "+")
        print("  the mix is the same in both"
              " periods:")
        print(f"    within {inner}, mix +0")
        return
    orders = ((f"class {what} first",
               after, before, "after"),
              ("mix first",
               before, after, "before"))
    found = []
    for f, (text, rec, mix, when) in enumerate(
            orders):
        if stat is mean:
            via, inner, mixed = of_mean(
                before, after, f)
            fixed = via and via[0] not in ENDS
        else:
            via = tuple(
                reach(stat, weighted(rec, mix,
                                     fill=x))[i]
                for i, x in enumerate(ENDS[::-1]))
            inner = apart(via, start)
            mixed = apart(end, via)
            if rec is before:
                inner, mixed = mixed, inner
            fixed = one(via)
        if not fixed:
            gone = ", ".join(absent(rec, mix))
            kind = "open"
            if via and via[0] == via[1]:
                kind = "infinite"
            why = f"a duration {when} is {kind}"
            if gone:
                why = f"no {gone} requests {when}"
            print(f"  {text}: not determined,")
            print(f"    {why}")
            continue
        if stat is mean or (one(inner)
                            and one(mixed)):
            found.append((inner[0], mixed[0]))
        way = spanned(via, unit=" ms")
        inner = spanned(inner, "+")
        mixed = spanned(mixed, "+")
        print(f"  {text}, via {way}:")
        print(f"    within {inner}, mix {mixed}")
    if len(found) == 2:
        if stat is mean:
            inner, mixed = of_mean(
                before, after, Fraction(1, 2))[1:]
        else:
            inner, mixed = ((sum(xs) / 2,) * 2
                            for xs in zip(*found))
        print("  average of the two orders:")
        print(f"    within {spanned(inner, '+')},"
              f" mix {spanned(mixed, '+')}")


def one_lost():
    """Twenty requests of one class in each
    period, all of 100 ms, and the duration
    of one of them lost in the second. The
    request is in the record: the mix has
    not changed. The mean is open upward,
    and the percentile is settled, because
    the other nineteen requests all took
    100 ms."""
    before = {"web": [100] * 20}
    after = {"web": [100] * 19 + [None]}
    print("20 web requests of 100 ms in each")
    print("period, one duration lost after:")
    components(before, after)
    components(before, after, p95, "p95",
               "distributions")


def one_bulk():
    """A common class of 99 requests in both
    periods and one bulk request in the
    second. At the mix of the second period
    the percentile of the first is settled
    without a bulk duration, and its mean is
    not."""
    before = {"common": [80.0] * 92
              + [100.0] * 6 + [180.0]}
    after = {"common": [70.0] * 92
             + [90.0] * 6 + [160.0],
             "bulk": [500.0]}
    print("99 common requests in each period,")
    print("one bulk request of 500 ms after:")
    components(before, after)
    components(before, after, p95, "p95",
               "distributions")


def at_share(rec, slow, of=1000):
    """The 95th percentile at a mix with the
    given share of slow requests, slow in
    `of`, and the rest fast. A reference mix
    is used for its class counts alone."""
    ref = {"fast": range(of - slow),
           "slow": range(slow)}
    return p95(weighted(rec, ref))


def less_one(rec, slow):
    """The 95th percentile at the given mix
    with each fast request in turn left out,
    in the order of the durations: what one
    request decides."""
    xs = sorted(rec["fast"])
    out = []
    for i in range(len(xs)):
        rest = xs[:i] + xs[i + 1:]
        out.append(at_share(dict(rec, fast=rest),
                            slow))
    return out


def one_mix(before, after):
    """Both periods at the same mix, for
    three mixes that neither had."""
    before, after = periods(before, after)
    print("p95 of both periods at one mix,")
    print("by its share of slow requests:")
    for slow in (40, 41, 47):
        b = at_share(before, slow)
        a = at_share(after, slow)
        print(f"  {slow / 10:.1f} %:"
              f" {b:.2f} -> {a:.2f} ms")
    top = sorted(after["fast"])[-2:]
    print("the two largest of the 100 fast")
    print("requests after:"
          f" {top[0]:.2f}, {top[1]:.2f} ms")
    each = less_one(after, 41)
    print("at 4.1 %, with one fast request")
    print("of the 100 after left out:")
    print(f"  the largest: {each[-1]:.2f} ms")
    for a in sorted(set(each[:-1])):
        print(f"  any other:   {a:.2f} ms")


def class_p95(xs):
    """The 95th percentile of one class."""
    return p95([(x, 1) for x in xs])


def main():
    rng = random.Random(23)
    before = record(BEFORE, rng)
    after = record(AFTER, rng)
    print("            before             after")
    print("class     n  mean  p95"
          "      n  mean  p95")
    for name in before:
        line = f"{name:<7}"
        for xs in (before[name], after[name]):
            line += (f"{len(xs):>4}"
                     f"{st.fmean(xs):>6.0f}"
                     f"{class_p95(xs):>5.0f}   ")
        print(line.rstrip())
    obs_b = weighted(before)
    obs_a = weighted(after)
    std_a = weighted(after, before)
    for name, f in (("mean", mean), ("p95", p95)):
        b, a = f(obs_b), f(obs_a)
        print(f"{name}, as observed"
              f"{b:>8.0f} ->{a:>5.0f}")
        print(f"{name}, after, at the mix of"
              f" before{f(std_a):>6.0f}")
    each = [class_p95(xs)
            for xs in after.values()]
    print("mean of the class p95s, after"
          f"{st.fmean(each):>8.0f}")
    components(before, after)
    components(before, after, p95, "p95",
               "distributions")
    one_mix(before, after)
    lost_class(before, after)
    one_unknown()
    two_missing()
    new_class(before, after)
    one_bulk()
    one_lost()


if __name__ == "__main__":
    main()
