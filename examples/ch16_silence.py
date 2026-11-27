# replay: exact
"""Chapter 16: what a silence establishes."""
import itertools
import math
from fractions import Fraction
from pathlib import Path

from examples import ch16_trace_measurements as tm
from examples.ch20_judge import exact_interval
from swmetro.stats import as_count, enclosed
from swmetro.stats import least, not_above
from swmetro.stats import paired_exact
from swmetro.stats import poisson_within
from swmetro.stats import quiet_bound
from swmetro.stats import rate_limits, rate_upper
from swmetro.stats import sum_within, upward

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "data" / "app_trace.log"


def merged(spans):
    """Stretches of time as disjoint (from,
    to) in order. Stretches that overlap or
    touch are one stretch, and the time they
    share is there once. The times are taken
    as the fractions that they are, and the
    stretches come back as fractions. A time
    stamp that counts nanoseconds from 1970
    and a duration given as a float can then
    be subtracted from one another: as
    floats, such stamps lie 256 ns apart."""
    out = []
    for x, y in sorted((Fraction(x), Fraction(y))
                       for x, y in spans):
        if not x < y:
            raise ValueError("empty stretch")
        if out and x <= out[-1][1]:
            x, top = out.pop()
            y = max(y, top)
        out.append((x, y))
    return out


def starts(spans, delta):
    """The start times from which a request
    leaves a message in the given stretches
    of time, as merged (from, to) intervals.
    The request writes a start message and,
    delta later, an end message. Its start
    message is seen if it starts in one of
    the stretches, and its end message if it
    starts in a copy of one moved back by
    delta. A request longer than a stretch
    can start before it and end after it,
    and leave neither."""
    if delta < 0:
        raise ValueError("need delta >= 0")
    spans, delta = merged(spans), Fraction(delta)
    back = [(x - delta, y - delta)
            for x, y in spans]
    return merged(spans + back)


def plain(f):
    """A fraction as an int if it is whole,
    and as the nearest float if it is not:
    a length as it is printed, and not as a
    bound is formed from it."""
    if f.denominator == 1:
        return f.numerator
    return float(f)


def reach(spans, delta):
    """Total length of starts(). For a single
    window of length T it is T + min(T, delta).
    The summed length of the stretches plus
    delta would lose where they lie. The sum
    is exact and is returned as the fraction
    that it is: a bound is formed from it,
    and a length rounded upward to a float
    would give a bound that is too low.
    plain() rounds it where it is printed."""
    found = starts(spans, delta)
    return sum(y - x for x, y in found)


def paired_bound(spans, delta, level=0.95):
    """Upper bound for the rate of messages
    that come in pairs, when the stretches
    held none: requests start as a Poisson
    process, before the stretches as well as
    in them, and each stands for two
    messages. A message rate is twice the
    rate of requests, and half the exposure
    gives its bound in one step, see
    quiet_bound()."""
    exposure = reach(spans, delta)
    return quiet_bound(exposure / 2, level)


def mean_reach(spans, shares):
    """Start times covered when requests
    differ in duration: shares is a list of
    (share, duration), the shares summing to
    one. Each request has its duration
    independently of the starts and of the
    other requests. The requests of each
    duration are then a Poisson process of
    their own, the stretches stay empty only
    if none of them leaves a message, and
    the exposure of the starts is the mean
    of reach() over the durations. That is
    not reach() at the mean duration, which
    for a single window is never smaller and
    can give too low a bound. The shares are
    taken as the numbers that they are, and
    the mean is a fraction."""
    spans, shares = list(spans), list(shares)
    total = sum(p for p, _ in shares)
    if (abs(total - 1) > 1e-9
            or min(p for p, _ in shares) < 0):
        raise ValueError("shares must sum to 1")
    return sum(Fraction(p) * reach(spans, d)
               for p, d in shares)


def by_messages(spans, delta):
    """Lengths of the start times from which
    a request brings two messages into the
    stretches, and one: where both its
    start and its end are seen, and where
    only one of them is. The stretches are
    merged first: time that two of them
    share was observed once. Both lengths
    are returned as the fractions that they
    are."""
    spans = merged(spans)
    seen = sum(y - x for x, y in spans)
    union = sum(y - x
                for x, y in starts(spans, delta))
    return 2 * seen - union, 2 * (union - seen)


def partial(x, n):
    """The sums of x^i / i! up to i = 0, 1 and
    so on to n, as fractions, for a fraction
    x."""
    out, term = [Fraction(1)], Fraction(1)
    for i in range(1, n + 1):
        term = term * x / i
        out.append(out[-1] + term)
    return out


def paired_upper(n, spans, delta, level=0.95):
    """Upper bound for the rate of messages
    that come in pairs, from n messages in
    the stretches. A request brings two
    messages if it starts where both are
    seen, and one if it starts where only
    its start or only its end is. The two
    sets of start times do not overlap, so
    the requests in them are independent
    Poisson counts. The bound is the rate at
    which so few messages still have
    probability 1 - level; for n = 0 it is
    paired_bound(). With the expected counts
    a and b of the two sets, that
    probability is exp(-a - b) times a sum
    of products of a^j / j! and b^i / i!,
    which is formed exactly. The float that
    is returned is the least at which the
    probability is known to be 1 - level or
    less, see least()."""
    n = as_count(n)
    if not 0 < level < 1:
        raise ValueError("need 0 < level < 1")
    if not n:
        return paired_bound(spans, delta, level)
    two, one = by_messages(spans, delta)
    if not two + one > 0:
        raise ValueError("no exposure")
    tail = 1 - Fraction(level)

    def test(rate, digits):
        half = Fraction(rate) / 2
        a, b = half * two, half * one
        sums = partial(b, n)
        c, term = 0, Fraction(1)
        for j in range(n // 2 + 1):
            c += term * sums[n - 2 * j]
            term = term * a / (j + 1)
        ends = sum_within([(c, a + b)], digits)
        return not_above(ends, tail)
    return least(test)


def kept_starts(spans, delta, keep, unit):
    """Start times covered when the recorder
    keeps the share `keep` of the messages
    of requests that write two, delta
    apart: each length of by_messages()
    times the chance that a request which
    starts there leaves a kept message. The
    unit is what the recorder decides on.
    "message": each message is kept or
    dropped on its own, and of two in the
    stretches one or both stay with
    probability keep * (2 - keep).
    "request": one decision holds for both
    messages, as when traces are sampled by
    their identifier, and that probability
    is keep. The decisions are independent
    of the times and of one another. The
    share is taken as the number that it is,
    and the length is a fraction."""
    if not 0 < keep <= 1:
        raise ValueError("need 0 < keep <= 1")
    keep = Fraction(keep)
    two, one = by_messages(spans, delta)
    both = {"message": keep * (2 - keep),
            "request": keep}[unit]
    return keep * one + both * two


def searched_bound(seconds, windows, level=0.95):
    """Upper bound for the rate when the
    observation is that at least one of
    several disjoint windows, found by
    looking through all of them, held no
    message: the rate at which such a search
    still succeeds with probability
    1 - level. Then every window holds a
    message with probability `level`: the
    chance of that is the chance for one
    window, 1 less that of no event, to the
    power of their number. It is enclosed
    and compared exactly, and the float
    that is returned is the least at which
    it is known to reach the level, see
    least(). For one window that is
    quiet_bound()."""
    windows = as_count(windows, "windows", 1)
    if not 0 < seconds < math.inf:
        raise ValueError("no exposure")
    if not 0 < level < 1:
        raise ValueError("need 0 < level < 1")
    seconds = Fraction(seconds)
    level = Fraction(level)

    def test(rate, digits):
        mean = Fraction(rate) * seconds
        lo, hi = poisson_within(0, mean, digits)
        if (1 - hi) ** windows >= level:
            return True
        if (1 - lo) ** windows < level:
            return False
        return None
    return least(test)


def gap_somewhere(rate, total, length):
    """Chance that the single messages of a
    homogeneous Poisson process leave a gap
    of `length` somewhere in a record of
    `total`: before the first message,
    between two, or after the last.
    Windows that overlap are not independent.
    The sum is the chance for the largest of
    the gaps, see gap_given(), taken over the
    Poisson count. It is summed in floats
    from gap_terms(), with signs that
    alternate, and is good to some thirteen
    digits."""
    terms = gap_terms(rate, total, length)
    return sum(float(c) * math.exp(-x)
               for c, x in terms)


def gap_terms(rate, total, length):
    """The chance of gap_somewhere() as pairs
    (c, x), of which it is the sum of
    c * exp(-x). The three arguments are
    taken as the numbers that they are, and
    the pairs are fractions."""
    if not 0 < length <= total < math.inf:
        raise ValueError("no room for the gap")
    if total > 20 * length:
        raise ValueError("need 20 gaps or fewer")
    if not 0 <= rate < math.inf:
        raise ValueError("need rate >= 0")
    rate, total, length = map(
        Fraction, (rate, total, length))
    out = []
    for k in range(1, int(total // length) + 1):
        m = rate * (total - k * length)
        c = m ** (k - 1) * (m / k + 1)
        c /= math.factorial(k - 1)
        out.append((c if k % 2 else -c,
                    k * rate * length))
    return out


def gap_given(n, total, length):
    """Chance of a gap of `length` or more
    somewhere in a record of `total`, for a
    homogeneous Poisson process and given
    exactly n messages. Their times are
    then n independent uniform times, put
    in order, with n + 1 gaps between them
    and to the two ends. Another kind of
    source has other gaps at the same rate
    and count: a heartbeat leaves none
    above its period. The terms alternate
    and can be far larger than their sum,
    which is taken in fractions. No rate
    enters, and no bound for a source comes
    out."""
    n = as_count(n)
    if not 0 < length <= total:
        raise ValueError("no room for the gap")
    share = Fraction(length) / Fraction(total)
    out = Fraction(0)
    for k in range(1, n + 2):
        rest = 1 - k * share
        if rest < 0:
            break
        term = math.comb(n + 1, k) * rest ** n
        out += term if k % 2 else -term
    return float(out)


RISE = ((150, Fraction(1, 10)),
        (150, Fraction(3, 10)))
QUIET = ((60, 90), (210, 240))


def expected(t, pieces=RISE):
    """Messages expected up to time t from a
    reference rate that is constant on each
    of its pieces, (seconds, rate). On this
    scale the messages of a Poisson process
    with that rate are uniform points again,
    given their number, and gap_given()
    holds for a gap in expected messages."""
    t, out = Fraction(t), Fraction(0)
    for seconds, rate in pieces:
        out += min(max(t, 0), seconds) * rate
        t -= seconds
    return out


def changing(n=60, spans=QUIET):
    """The same 30 s without a message where
    the reference rate is low and where it
    is high, in a record that holds n
    messages: as a window chosen in advance,
    and as the largest gap found anywhere,
    both in expected messages."""
    (s1, r1), (s2, r2) = RISE
    whole = expected(s1 + s2)
    print(f"  a rate of {float(r1):.2f} msg/s"
          f" for {s1} s, then")
    print(f"  {float(r2):.2f} msg/s for {s2} s,"
          f" {n} counted:")
    for lo, hi in spans:
        a = expected(hi) - expected(lo)
        empty = float((1 - a / whole) ** n)
        found = gap_given(n, whole, a)
        print(f"    {lo} s to {hi} s spans"
              f" {plain(a)} expected messages")
        print("      chosen in advance, empty:"
              f" {100 * empty:.4g} %")
        print(f"      a gap of {plain(a)} or"
              f" more: {100 * found:.4g} %")


def depleted(k, n=60, m=60, here=0):
    """Chance that the first of two records
    has `here` or fewer of the k messages
    that a window holds in both together,
    where the first has n messages in all
    and the second m. The two records are
    independent. Under the hypothesis that
    they are Poisson processes with
    proportional rate profiles, and given
    the two totals and k, every division of
    the n + m messages between them that
    keeps the totals is equally likely. The
    one-sided exact test of Fisher. Two
    copies of one stream of events are not
    independent: each has half of the k."""
    k, n = as_count(k, "k"), as_count(n)
    m, here = as_count(m, "m"), as_count(here)
    if k > n + m:
        raise ValueError("k is at most n + m")
    part = 0
    for i in range(min(here, k) + 1):
        ways = math.comb(n, i)
        part += ways * math.comb(m, k - i)
    return Fraction(part, math.comb(n + m, k))


def second(counts=(3, 6, 9)):
    """The window of changing() with a second
    record of 60 messages as the reference,
    in place of a rate taken as known."""
    print("  a second record of 60 messages as")
    print("  the reference, k in the window,")
    print("  and none in it here:")
    for k in counts:
        share = 100 * float(depleted(k))
        print(f"    k = {k}: {share:.4g} %")


def matched(before=9, now=3, n=60):
    """The same n requests in two runs, with
    `before` of them in the window in the
    reference and `now` in the incident.
    Those in it both times did not change.
    They are at least before + now - n, or
    there would be fewer than no requests
    outside both times, and at most the
    smaller of the two counts. Of the rest,
    the number that moved into the window
    is binomial with 1/2 if the pairs are
    independent and either move is equally
    likely, see paired_exact(): the chance
    of so few is the one-sided test for a
    fall, and is large after a rise. Two
    independent records with the same
    counts give the last line."""
    n = as_count(n)
    before = as_count(before, "before")
    now = as_count(now, "now")
    if max(before, now) > n:
        raise ValueError("counts are at most n")
    print(f"  the same {n} requests twice,"
          f" {before}")
    print(f"  in the window and then {now}; by")
    print("  those in it both times:")
    least = max(0, before + now - n)
    most = min(before, now)
    for both in range(least, most + 1):
        moved = now - both, before - both
        share = 100 * float(paired_exact(*moved))
        print(f"    {both}: {share:.2f} %")
    p = depleted(before + now, n, n, now)
    print("  two independent records:"
          f" {100 * float(p):.2f} %")


def gap_bound(total, length, level=0.95):
    """Upper bound for the rate when the
    observation is that the record held a
    gap of `length` somewhere: the rate at
    which one is found with probability
    1 - level. The chance is enclosed from
    gap_terms(), and the float that is
    returned is the least at which it is
    known to be 1 - level or less, see
    least()."""
    if not 0 < level < 1:
        raise ValueError("need 0 < level < 1")
    tail = 1 - Fraction(level)

    def test(rate, digits):
        terms = gap_terms(rate, total, length)
        ends = sum_within(terms, digits)
        return not_above(ends, tail)
    return least(test)


def selections(windows, delta):
    """For every selection of one or more of
    the windows: +1 or -1 as it holds an
    odd or an even number of them, and its
    reach(), the start times from which a
    request brings a message into one of
    its windows. Ten windows have 1023
    selections; more than sixteen are
    refused."""
    windows = list(windows)
    if not 0 < len(windows) <= 16:
        raise ValueError("need 1 to 16 windows")
    out = []
    for k in range(1, len(windows) + 1):
        for some in itertools.combinations(
                windows, k):
            out.append(((-1) ** (k + 1),
                        reach(some, delta)))
    return out


def some_empty(rate, terms):
    """Probability that at least one window
    holds no message of a paired source with
    the given rate of messages, from the
    terms of selections(). All windows of a
    selection are empty at once if no
    request starts in its reach, and the
    sum over the selections, with their
    signs, counts each way in which some
    window is empty once. Single messages
    need no such sum: their counts in
    disjoint windows are independent. Those
    of pairs are not, since a request that
    starts in one window can end in the
    next."""
    return sum(sign * math.exp(-rate / 2 * a)
               for sign, a in terms)


def searched_pairs(windows, delta, level=0.95):
    """Upper bound for the rate of messages
    that come in pairs, delta apart, when at
    least one of the windows, found by
    looking through all of them, held none:
    the rate at which such a search still
    succeeds with probability 1 - level. The
    chance of some_empty() is enclosed, with
    the selections of one reach taken
    together: the 1023 selections of ten
    windows in a row have 24 reaches. The
    float that is returned is the least at
    which it is known to be 1 - level or
    less, see least()."""
    if not 0 < level < 1:
        raise ValueError("need 0 < level < 1")
    net = {}
    for sign, a in selections(windows, delta):
        net[a] = net.get(a, 0) + sign
    tail = 1 - Fraction(level)

    def test(rate, digits):
        half = Fraction(rate) / 2
        pairs = [(c, half * a)
                 for a, c in net.items()]
        ends = sum_within(pairs, digits)
        return not_above(ends, tail)
    return least(test)


def up(rate):
    """A bound as it is printed: upward."""
    return upward(rate, 5) + " msg/s"


def most(rate):
    """An upper bound with its level."""
    return f"at most {up(rate)} at 95 %"


def nominal(rate):
    """A rate drawn at 95 % from a mix that
    is not known to be that of the source."""
    return up(rate) + " at a nominal 95 %"


def seconds(parts):
    """Times in ms, in s."""
    return [(p[0] / 1000, p[1] / 1000)
            for p in parts]


def inside(rows, comp, found):
    """Messages of one component among the
    lines of the given runs(). A count
    belongs to the time in which it was
    taken: a line that lies in a gap is in
    the record and not in the verified time.
    The lines of the verified time are told
    by their sequence numbers. The first and
    the last line of a run belong to it, and
    a line beyond a gap does not, whatever
    its stamp."""
    own = {seq for _, _, part in found
           for seq in part}
    return sum(c == comp and seq in own
               for seq, _, c, _ in rows)


def stamped(rows, comp, found, lo, hi, span):
    """With time stamps alone: the least and
    the most messages of one component in
    the verified time of a window, when its
    runs() were found in these lines and the
    sequence numbers are no longer at hand.
    A stamp inside the time of a run settles
    it. At an end of a run one line of the
    run is stamped, and another line of that
    stamp may lie beyond a loss. If the
    record has one line of that stamp, it is
    the line of the run. If it has more,
    all but one of them are in doubt, or all
    but two where two runs meet. The range
    always holds the count by number. It is
    that count where no two lines share a
    stamp, and it can be wider than the
    stamps require where they do."""
    a, b = tm.window(lo, hi, span)
    spans = [p[:2] for p in found]
    ends = [t for p in spans for t in p]
    held = [(ms - span[0], c)
            for _, ms, c, _ in rows
            if a <= ms < b]
    least = most = 0
    for t in {t for t, _ in held}:
        same = [c for u, c in held if u == t]
        own = same.count(comp)
        if any(x < t < y for x, y in spans):
            least += own
        elif t in ends:
            spare = len(same) - ends.count(t)
            least += max(0, own - spare)
        else:
            own = 0
        most += own
    return least, most


def models(rows, span, counter=tm.COUNTER):
    """The second half minute of the service
    component under two models of what writes
    its messages. The count that enters a
    bound is the count in the verified time."""
    rows = list(rows)
    lo, hi = 30, 60
    n = tm.count(rows, "svc", lo, hi, span)
    found = tm.runs(rows, lo, hi, span, counter)
    live = seconds(found)
    t = sum(y - x for x, y in live)
    k = inside(rows, "svc", found)
    noun = "message" if n == 1 else "messages"
    print(f"svc, {lo}-{hi} s: {n} {noun},"
          f" {t:g} s verified")
    if not t:
        print("  exposure unknown, no bound")
        return
    print(f"  in the verified time: {k}")
    print("  messages arriving singly:")
    print("    " + most(rate_upper(k, t)))
    try:
        d = tm.time_delta(rows).value / 1000
    except ValueError:
        print("  no request to take the time")
        print("  between a start and an end from")
        return
    covered = reach(live, d)
    print("  requests arriving, each with a")
    print(f"  start and an end {d:g} s apart:")
    print("    starts covered:"
          f" {plain(covered):g} s")
    print("    " + most(paired_upper(k, live, d)))
    if k == 0:
        one = quiet_bound(t)
        still = math.exp(-one / 2 * covered)
        print("    silent at the first bound:"
              f" {100 * still:.1f} %")


def gap(at):
    """Five lines of a capture that lost
    lines on either side of 45 s, with one
    service message at the given second,
    written after the other line of that
    second if there is one."""
    marks = [(t, "pool")
             for t in (30, 40, 50, 60)]
    marks.append((at, "svc"))
    seq, rows = 0, []
    for t, comp in sorted(marks):
        seq += 4 if 40 < t <= 50 else 1
        rows.append((seq, 1000 * t, comp, "x"))
    return rows


TIED = ((1, 30), (2, 40), (5, 40), (9, 50),
        (10, 60))
LONE = ((1, 30), (2, 40), (5, 50), (6, 60))


def edge(own, marks=TIED):
    """Lines of a capture, as (number,
    second), of which the one with the given
    number is the service's. TIED lost lines
    before and after its line 5, which bears
    the stamp of line 2. In LONE no two
    lines share a stamp."""
    return [(n, 1000 * t,
             "svc" if n == own else "pool", "x")
            for n, t in marks]


def told(marks, owns):
    """The count in the verified time of 30
    to 60 s by sequence number and by time
    stamp alone, with each of the given
    lines in turn as the service's."""
    half = (0, 60_000)

    def series(xs):
        xs = [str(x) for x in xs]
        head = ", ".join(xs[:-1])
        return head + " and " + xs[-1]
    print("lines " + series(n for n, _ in marks)
          + ", stamped")
    print(series(t for _, t in marks) + " s:")
    found = tm.runs(edge(0, marks), 30, 60, half)
    names = [f"{own[0]} to {own[-1]}"
             if len(own) > 1 else f"{own[0]}"
             for _, _, own in found]
    print("  verified time has lines "
          + " and ".join(names))
    print("  the line of the service, counted")
    print("  by number, and by stamp alone:")
    for own in owns:
        rows = edge(own, marks)
        k = inside(rows, "svc", found)
        low, top = stamped(rows, "svc", found,
                           30, 60, half)
        text = f"{low} to {top}"
        if low == top:
            text = f"{low}"
        print(f"    line {own}: {k}, and {text}")


def counted():
    """One message in a gap, one in the
    verified time beside it, and one at the
    stamp at which a run ends."""
    half = (0, 60_000)
    print("verified 30 to 40 and 50 to 60 s,")
    print("one message in the record:")
    for at in (45, 35):
        print(f"at {at} s")
        models(gap(at), half, None)
    told(TIED, (2, 5, 10))
    told(LONE, (2,))


def pair_count():
    """One message of a paired source, and
    one window described as two stretches
    that overlap."""
    whole = [(0, 30)]
    print("one message in 30 s, from pairs")
    for d in (2.346, 60, 10):
        top = paired_upper(1, whole, d)
        print(f"  {d:g} s apart:")
        print("    " + most(top))
    parts = [(0, 20), (10, 30)]
    top = paired_upper(1, parts, 10)
    print("  10 s apart, the 30 s given as")
    print("  0 to 20 s and 10 to 30 s:")
    print("    " + most(top))


def spacing(rows, span):
    """Where the verified time lies matters
    for pairs: a request longer than the
    window, and the first half minute with
    its gap."""
    rows = list(rows)
    whole = [(0, 30)]
    print("requests of 60 s, window of 30 s:")
    print("  starts covered:"
          f" {plain(reach(whole, 60)):g} s")
    print("    " + most(paired_bound(whole, 60)))
    d = tm.time_delta(rows).value / 1000
    live = seconds(tm.stretches(
        rows, 0, 30, span, tm.COUNTER))
    text = " and ".join(f"{x:g} to {y:g}"
                        for x, y in live)
    t = sum(y - x for x, y in live)
    print("first half minute, verified from")
    print(f"  {text} s:")
    print(f"  {t:g} s, and for pairs {d:g} s")
    print("  apart, starts covered:"
          f" {plain(reach(live, d)):g} s")


def durations():
    """A silent window of 30 s and three
    sources of requests with one mean
    duration, 30 s."""
    whole = [(0, 30)]
    cases = (
        ("all of 30 s", [(1, 30)]),
        ("half of 10 s, half of 50 s",
         [(0.5, 10), (0.5, 50)]),
        ("0.9 of 1 s, 0.1 of 291 s",
         [(0.9, 1), (0.1, 291)]))
    print("silence of 30 s, requests with a")
    print("mean duration of 30 s:")
    for text, shares in cases:
        a = mean_reach(whole, shares)
        print(f"  {text}:")
        print("    starts covered:"
              f" {plain(a):g} s")
        print("    " + most(quiet_bound(a / 2)))
    low = paired_bound(whole, 30)
    a = mean_reach(whole, cases[1][1])
    still = math.exp(-low / 2 * a)
    print("  at the first bound the second")
    print(f"  stays silent: {100 * still:.2f} %")


def eligible(window, rule, d):
    """Length of the start times from which
    a request of duration d enters a sample
    taken with one window. The rule says
    where the request has to fall: "start"
    if it starts in the window, "end" if it
    ends in it, "both" if it starts and ends
    in it, "either" if its start or its end
    lies in it. A request ends in the window
    if it starts in a stretch of the same
    length that lies d earlier, so that the
    first two rules give every duration the
    length of the window. That stretch
    begins before the window opens, and for
    a request shorter than the window it
    ends inside: "end" takes a request
    wherever its start falls. Taking only
    those that started before the window
    would be another rule, with min(window,
    d) of start times."""
    return {"start": window,
            "end": window,
            "both": max(0, window - d),
            "either": plain(
                reach([(0, window)], d))
            }[rule]


def selected(shares, window, rule):
    """The durations as a sample of requests
    shows them, when a request enters the
    sample by the rule: a duration is seen
    in proportion to its share times the
    length of the start times that select
    it. The sample has the shares of the
    source if that length is the same for
    every duration. It is for "start" and
    for "end". For "either" it is if no
    duration is shorter than the window,
    and for "both" only if all durations
    are equal. The starts are taken to come
    at one steady rate, as everywhere in
    this listing."""
    seen = [(p * eligible(window, rule, d), d)
            for p, d in shares]
    total = sum(w for w, _ in seen)
    if not total:
        raise ValueError("no request selected")
    return [(w / total, d) for w, d in seen]


def sampling():
    """One source, half its requests of 10 s
    and half of 50 s, and four samples of
    its durations taken with windows of
    30 s. The mix of each is put to the
    silent window of durations() as if it
    were the mix of the source. The rate
    drawn from it has its 95 % in name
    only, and the line below it gives how
    often the source is silent at that
    rate."""
    whole = [(0, 30)]
    source = [(0.5, 10), (0.5, 50)]
    rules = (("start", "starts in a window"),
             ("end", "ends in a window"),
             ("both", "starts and ends in it"),
             ("either", "starts or ends in it"))
    print("durations sampled with windows of")
    print("30 s; a request is taken if it")
    for rule, text in rules:
        room = " and ".join(
            f"{eligible(30, rule, d):g} s"
            for _, d in source)
        seen = selected(source, 30, rule)
        mix = ", ".join(
            f"{100 * p:.0f} % of {d} s"
            for p, d in seen)
        a = mean_reach(whole, seen)
        top = quiet_bound(a / 2)
        true = mean_reach(whole, source)
        still = math.exp(-top / 2 * true)
        print(f"  {text}:")
        print(f"    from {room} of start times")
        print(f"    {mix}")
        print("    starts covered:"
              f" {plain(a):g} s")
        print("    " + nominal(top))
        print("    the source is silent at"
              f" that: {100 * still:.2f} %")


def calibrated(k=2, n=20):
    """A silent window of 30 s when the mix
    of two durations, 10 s and 50 s, is
    known from a sample only: k short
    requests among n that were taken by
    their start and followed to their end.
    The risk of 5 % is divided before the
    data are seen. With at most 2.5 % the
    share of the short requests lies above
    its exact upper limit, and at the limit
    the starts covered are fewest. With
    2.5 % a window stays silent at the rate
    drawn from those starts. The bound fails
    only if one of the two does, in at most
    5 % of cases, whether or not sample and
    window are independent. The last case
    needs no sample: every request short."""
    def covered(p):
        p = Fraction(p)
        a = mean_reach([(0, 30)], [(p, 10),
                                   (1 - p, 50)])
        print("    starts covered:"
              f" {plain(a):g} s")
        return a
    risk = 0.05 / 2
    top = exact_interval(k, n, 1 - 2 * risk)[1]
    print("durations of 10 s and 50 s, the mix")
    print(f"from {n} requests taken by their")
    print(f"start: {k} of 10 s."
          " A silence of 30 s:")
    print(f"  the {100 * k / n:g} %"
          " taken as known:")
    known = covered(Fraction(k, n))
    known = quiet_bound(known / 2)
    print("    " + nominal(known))
    share = upward(100 * top, 3)
    print(f"  at most {share} % of 10 s,"
          " at 97.5 %:")
    a = covered(top)
    still = math.exp(-known / 2 * a)
    print("    silent at the nominal rate:"
          f" {100 * still:.2f} %")
    held = quiet_bound(a / 2, 1 - risk)
    print("    " + most(held))
    print("  no sample, all of 10 s:")
    print("    " + most(quiet_bound(
        covered(1) / 2)))


def first_count():
    """The first count above zero."""
    c = tm.current(1, 30)
    print(f"one message in 30 s: {c}")
    lo, hi = enclosed(*c.interval(), 5)
    print(f"  normal, 95 %: {lo} to {hi}")
    lo, hi = enclosed(*rate_limits(1, 30), 5)
    print(f"  exact, 95 %: {lo} to {hi}")
    print("  " + most(rate_upper(1, 30)))


def exposure(span):
    """A capture of 60 s without a message:
    what the bound owes to the verified time
    and to the share of messages kept, and
    what verifies the time of an empty
    record."""
    print("silence in a capture of 60 s, by")
    print("verified time and share kept:")
    for t, keep in ((60, 1), (40, 1), (40, 0.25),
                    (20, 1)):
        top = quiet_bound(t * keep)
        print(f"  {t} s, {100 * keep:3.0f} %:"
              f" {t * keep:g} s,")
        print("    " + most(top))
    print("an empty record of that capture:")
    for text, counter in (
            ("counter at 1, and at 1", (1, 1)),
            ("counter at 1, and at 41", (1, 41)),
            ("no counter readings", None)):
        t = tm.verified([], 0, 60, span, counter)
        what = "exposure unknown, no bound"
        if t:
            what = f"{t:g} s verified"
        print(f"  {text}:")
        print(f"    {what}")
    back = [(i + 1, 1000 * t, "x", "")
            for i, t in enumerate((1, 26, 6, 29))]
    t = tm.verified(back, 0, 30, (0, 30_000))
    what = "exposure unknown, no bound"
    if t:
        what = f"{t:g} s verified"
    print("four consecutive lines stamped 1, 26,")
    print(f"6 and 29 s: {what}")


def retained():
    """A silent window of 30 s, chosen in
    advance and observed in full, and a
    recorder that keeps one message in four.
    Three sources send the same number of
    messages on average: single messages,
    and requests of 10 s that write two,
    kept by the message or by the request.
    The last line of a source gives how
    often it is silent at the bound for
    single messages."""
    whole, keep = [(0, 30)], 0.25
    single = quiet_bound(30 * keep)
    print("silence of 30 s, one message in")
    print("four kept; effective exposure:")
    print(f"  single messages: {30 * keep:g} s")
    print("    " + most(single))
    for unit, text in (
            ("message", "each message"),
            ("request", "both messages")):
        a = kept_starts(whole, 10, keep, unit) / 2
        still = math.exp(-single * a)
        print("  pairs 10 s apart, a decision")
        print(f"  for {text}: {plain(a):g} s")
        print("    " + most(quiet_bound(a)))
        print("    silent at the first bound:"
              f" {100 * still:.2f} %")


def search():
    """An empty window of 30 s, by how it was
    come by: for single messages, all of
    them kept, and for requests of 10 s that
    write two. The ten windows follow one
    another; a gap of 30 s may begin
    anywhere."""
    one = quiet_bound(30)
    print("an empty window of 30 s:")
    print("  chosen in advance:")
    print("    " + most(one))
    print("  found by a search of ten:")
    single = searched_bound(30, 10)
    print("    " + most(single))
    print("  all ten empty:")
    print("    " + most(quiet_bound(300)))
    found = 1 - 0.95 ** 10
    print("  at the first bound a search of ten")
    print("  finds an empty window:"
          f" {100 * found:.2f} %")
    print("  a gap of 30 s anywhere in 300 s:")
    print("    " + most(gap_bound(300, 30)))
    found = gap_somewhere(single, 300, 30)
    print("  at the second bound such a gap")
    print("  is found somewhere:"
          f" {100 * found:.2f} %")
    print("  with 60 messages counted, a gap of")
    for length in (30, 35, 40, 45):
        found = gap_given(60, 300, length)
        print(f"    {length} s or more:"
              f" {100 * found:.2f} %")
    changing()
    second()
    matched()
    ten = [(30 * i, 30 * i + 30)
           for i in range(10)]
    print("the same for requests of 10 s that")
    print("write two messages:")
    print("  chosen in advance:")
    first = paired_bound(ten[:1], 10)
    print("    " + most(first))
    print("  found by a search of ten:")
    print("    " + most(searched_pairs(ten, 10)))
    print("  all ten empty:")
    print("    " + most(paired_bound(ten, 10)))
    found = some_empty(single,
                       selections(ten, 10))
    print("  at the single-message search bound")
    print("  a search of ten finds an empty one:"
          f" {100 * found:.2f} %")


def main():
    text = LOG.read_text(encoding="utf-8")
    rows = [tm.parse(x)
            for x in text.splitlines()]
    span = [tm.stamp(x) for x in tm.CAPTURE]
    models(rows, span)
    spacing(rows, span)
    durations()
    sampling()
    calibrated()
    first_count()
    pair_count()
    counted()
    exposure(span)
    retained()
    search()


if __name__ == "__main__":
    main()
