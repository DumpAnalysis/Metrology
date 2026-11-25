# replay: exact
"""Chapter 16: measurements read from a trace."""
import math
from fractions import Fraction
from pathlib import Path

from swmetro import Result, type_a, u_tri
from swmetro.stats import as_count, linfit
from swmetro.stats import quiet_bound, upward

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "data" / "app_trace.log"
STEP = 1.0  # timestamp resolution, ms
# Metadata of the capture, which no line of
# the trace carries: when the recorder was
# started and when it was stopped, and the
# sequence number that the tracer would have
# assigned next at each of the two moments.
CAPTURE = ("12:26:00.000", "12:27:00.000")
COUNTER = (1, 800)
PAIR = ("request 17 start", "request 17 end")


def stamp(clock):
    """HH:MM:SS.mmm as ms since midnight, with
    no date: the lines must be of one day.
    Past midnight the stamps go back, as
    after a step of the clock."""
    h, m, s = clock.split(":")
    ms = (int(h) * 60 + int(m)) * 60_000
    return ms + round(float(s) * 1000)


def parse(line):
    """seq, time in ms, component, message."""
    seq, _, clock, comp, msg = line.split(None, 4)
    ms = stamp(clock)
    return int(seq), ms, comp, msg.strip()


def time_delta(rows, pair=PAIR, comp=None):
    """Interval between two significant events.
    Model: both stamps are truncated at
    independent, uniformly distributed
    phases, so the error of their difference
    is triangular on (-STEP, STEP). The trace
    must hold each event once, the start
    first: after a second start the lines do
    not say which attempt the end closes.
    The events are known by their text, which
    must name one request in all the rows:
    where workers number their own requests,
    pass one as `comp` to read its lines."""
    own = [(m, ms) for _, ms, c, m in rows
           if m in pair and comp in (None, c)]
    if tuple(m for m, _ in own) != tuple(pair):
        raise ValueError("no single pair")
    delta = own[1][1] - own[0][1]
    if delta < 0:
        raise ValueError("stamps go back")
    return Result(delta, u_tri(STEP), "ms")


def ticks(seconds):
    """Seconds as ms, exactly. A float is read
    as the decimal that prints for it: in
    floats 1000 * 2.007 is 2007.0000000000002,
    and a line stamped 2007 would be counted
    on the wrong side of that end."""
    if isinstance(seconds, float):
        seconds = repr(float(seconds))
    q = 1000 * Fraction(seconds)
    return int(q) if q == int(q) else q


def window(lo, hi, span):
    """[lo, hi) s after the start of the
    capture, in ms of the trace clock. The
    start is metadata: the first message
    does not mark it, and a capture without
    a message still has one. A window that
    leaves the capture was not observed."""
    a = span[0] + ticks(lo)
    b = span[0] + ticks(hi)
    if not span[0] <= a < b <= span[1]:
        raise ValueError("outside the capture")
    return a, b


def count(rows, comp, lo, hi, span):
    """Messages of one component in a window."""
    a, b = window(lo, hi, span)
    return sum(c == comp and a <= ms < b
               for _, ms, c, _ in rows)


def runs(rows, lo, hi, span, counter=None):
    """The parts of a window for which there
    is evidence that no message was lost,
    each as (from, to, lines): its time in
    ms after the start of the capture, and
    the sequence numbers of the lines that
    belong to it. Lines with consecutive
    sequence numbers form a run, and are
    evidence for the time from its first
    line to its last. Readings of the
    tracer's counter at the start and the
    stop of the capture are evidence for the
    time before the first line and after the
    last. The span of the capture alone is
    evidence of nothing. Sequence numbers
    give the order of the lines and not
    their times: where a time stamp lies
    before that of the line before it, the
    stamps are no evidence either, and
    nothing is verified. Nor is it where a
    number does not exceed the one before
    it, after a restart of the counter or a
    reordering of the lines. The lines of a
    part are those of its run that lie in
    the window. They are known by their
    numbers and not by their stamps: a line
    outside the run can bear the stamp of
    its last line. What numbered() refuses
    is refused here."""
    a, b = window(lo, hi, span)
    rows = list(rows)
    numbered(rows, counter)
    marks = [(r[0], r[1], 1) for r in rows]
    if counter:
        head = (counter[0] - 1, span[0], 0)
        marks.insert(0, head)
        marks.append((counter[1], span[1], 0))
    chains = [marks[:1]]
    for p, q in zip(marks, marks[1:]):
        if q[1] < p[1] or q[0] <= p[0]:
            return []
        if q[0] - p[0] != 1:
            chains.append([])
        chains[-1].append(q)
    out = []
    for chain in chains:
        if len(chain) < 2:
            continue
        x = max(chain[0][1], a) - span[0]
        y = min(chain[-1][1], b) - span[0]
        own = [n for n, t, line in chain
               if line and a <= t < b]
        if x < y:
            out.append((x, y, own))
    return out


def stretches(rows, lo, hi, span, counter=None):
    """The times of runs() alone, as (from,
    to). Parts that touch are one stretch of
    time."""
    out = []
    for x, y, _ in runs(rows, lo, hi, span,
                        counter):
        if out and out[-1][1] == x:
            x = out.pop()[0]
        out.append((x, y))
    return out


def verified(rows, lo, hi, span, counter=None):
    """Seconds of a window that are verified,
    by the stamps, wherever in it they lie."""
    found = stretches(rows, lo, hi, span, counter)
    ms = sum(y - x for x, y in found)
    return float(Fraction(ms) / 1000)


def current(n, seconds):
    """Rate from n messages counted in a time.
    The count of the record is exact; the
    uncertainty is that of the rate of a
    homogeneous Poisson process estimated
    from it. For n = 0 it would be 0, as if
    silence had shown the rate to be exactly
    nothing. The estimate of zero is stated
    with its upper bound, quiet_bound().
    Listing 16.4 takes the small counts up."""
    n = as_count(n)
    if not n:
        raise ValueError("no messages: a bound")
    if not seconds > 0:
        raise ValueError("no exposure")
    u = math.sqrt(n) / seconds
    return Result(n / seconds, u, "msg/s")


def period(rows, text):
    """Period of a recurring message, twice:
    mean of the gaps (Type A on the gaps) and
    slope of time against occurrence index."""
    ts = [ms / 1000 for _, ms, _, m in rows
          if m == text]
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    fit = linfit(range(len(ts)), ts)
    slope = Result(fit[1], fit[3], "s")
    return type_a(gaps, "s"), slope


def numbered(rows, counter=None):
    """The numbers of the lines and the two
    readings of the counter, checked once
    for counts and verified time. One that
    is not whole is refused, as is a counter
    that runs back and a line beyond it.
    Without readings the ends are the least
    number and one past the greatest."""
    low = -math.inf
    seqs = [as_count(r[0], "number", low)
            for r in rows]
    ends = counter or (min(seqs, default=0),
                       max(seqs, default=-1) + 1)
    first, end = (as_count(x, "reading", low)
                  for x in ends)
    if end < first:
        raise ValueError("counter runs back")
    if seqs and not (first <= min(seqs)
                     and max(seqs) < end):
        raise ValueError("beyond the counter")
    return seqs, first, end


def completeness(rows, counter=None):
    """Messages lost, messages sent, and the
    time span of the largest gap in the
    sequence numbers, None if no number is
    missing between two lines and NaN if
    the stamps go back, when the counts
    stand and the time is not known. Without
    the readings of the counter the counts
    are those of the span from the first
    line to the last. Numbers that do not
    rise are a restart or another stream,
    and are refused, as is all that
    numbered() refuses."""
    rows = list(rows)
    seqs, first, end = numbered(rows, counter)
    steps = [q - p
             for p, q in zip(seqs, seqs[1:])]
    if min(steps, default=1) < 1:
        raise ValueError("numbers do not rise")
    if not (counter or seqs):
        raise ValueError("no line, no counter")
    gap_ms, wide = None, max(steps, default=1)
    if wide > 1:
        i = steps.index(wide)
        gap_ms = rows[i + 1][1] - rows[i][1]
        times = [r[1] for r in rows]
        if times != sorted(times):
            gap_ms = math.nan
    sent = end - first
    return sent - len(seqs), sent, gap_ms


def main():
    text = LOG.read_text(encoding="utf-8")
    rows = [parse(x) for x in text.splitlines()]
    print("Time Delta:", time_delta(rows))
    span = tuple(stamp(x) for x in CAPTURE)
    for lo, hi in ((0, 30), (30, 60)):
        n = count(rows, "db", lo, hi, span)
        c = current(n, hi - lo)
        print(f"db current, {lo:>2}-{hi} s: {c}")
    n = count(rows, "svc", 30, 60, span)
    t = verified(rows, 30, 60, span, COUNTER)
    print(f"svc, 30-60 s: {n} messages,"
          f" {t:g} s verified")
    if n == 0 and t > 0:
        top = upward(quiet_bound(t), 5)
        print("  if messages arrive singly:")
        print(f"  at most {top} msg/s"
              " at 95 %")
    naive, slope = period(rows, "E1042 timeout")
    print("E1042 period, mean of gaps:", naive)
    print("E1042 period, fitted slope:", slope)
    lost, sent, gap = completeness(rows, COUNTER)
    share = 100 * lost / sent
    print(f"lost {lost} of {sent} messages"
          f" ({share:.1f} %)")
    if gap is not None:
        print(f"sequence gap spans {gap} ms")
    t = verified(rows, 0, 30, span, COUNTER)
    print(f"verified time, 0-30 s: {t:g} s")
    n = count(rows, "db", 0, 30, span)
    c = current(n + lost, 30)
    print("db current, 0-30 s, if all the lost")
    print(f"  messages were db: {c}")


if __name__ == "__main__":
    main()
