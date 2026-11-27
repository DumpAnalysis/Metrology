"""Regression tests added after the review of Version 45.

The review found that the caption of Figure 6.5 called a distance what the dashed line shows below zero as an
overlap, and that the text gave 0.5 - 6e for the least time between two events without the range in which it
holds; that Listing 10.1, with a spacing of 0.1 or 0.3 ms, lost requests of the closed loop and gained one in
the open loop, because its times were floats; and that shown() of Listing 17.1 failed for a probability whose
reciprocal has more than 4300 digits. It asked which single stamp can explain the contradiction of the three
hosts, for a server whose recovery the closed loop cannot tell from another, and for the range of the exact
test when the pairs of fifty items of two thousand are lost, which Figure 17.2 now draws.

Each test computes what it expects independently of the code under test: the three hosts by placing the events
of every exchange in order for offsets on a grid, in whole quarters of a millisecond where many cases are
tried; the two generators by loops in whole microseconds, and the two servers by a simulation that steps through
the minute a millisecond at a time with a queue; the exact test from binomial coefficients over every table
that the counts allow; printed probabilities by decimal arithmetic.
"""
import collections
import contextlib
import decimal
import io
import math
import random
import statistics
import sys
import unittest
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from examples import ch06_exchange as exchange
from examples import ch10_coordinated_omission as omission
from examples import ch17_accuracy as accuracy

ROOT = Path(__file__).resolve().parent.parent
OTHER = exchange.A_TO_C[:3] + (62,)                    # the exchange of A and C with 62 ms for its last stamp


def expected(name):
    return (ROOT / "expected" / f"{name}.txt").read_text(encoding="utf-8")


def printed(function, *args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        function(*args)
    return out.getvalue()


@contextlib.contextmanager
def constants(module, **values):
    """The named constants of a module at other values, and back at the old ones whatever happens."""
    old = {name: getattr(module, name) for name in values}
    try:
        for name, value in values.items():
            setattr(module, name, value)
        yield
    finally:
        for name, value in old.items():
            setattr(module, name, value)


def in_order(stamps, theta, e):
    """Whether the four events of an exchange can follow one another when each stamp is within e of its event
    and the clock of the host that replies is `theta` ahead: every event as early as those before it allow."""
    a1, b2, b3, a4 = (Fraction(str(x)) for x in stamps)
    at = a1 - e
    for stamp in (b2 - theta, b3 - theta, a4):
        at = max(at, stamp - e)
        if at > stamp + e:
            return False
    return True


def offsets_of_c(a_to_c, e, step, reach=10):
    """The offsets of C from A on a grid: by their exchange, by way of B, and by all four exchanges."""
    grid = [k * step for k in range(int(-reach / step), int(reach / step) + 1)]
    of_b = [b for b in grid if all(in_order(stamps, b, e) for stamps in exchange.EXCHANGES)]
    alone = [c for c in grid if in_order(a_to_c, c, e)]
    through = sorted({c for b in of_b for c in grid if in_order(exchange.B_TO_C, c - b, e)})
    return alone, through, sorted(set(alone) & set(through))


class WhatTheDashedLineMeasures(unittest.TestCase):
    def test_the_numbers_of_the_review(self):
        e = Fraction(3, 10)
        alone, through, common = offsets_of_c(OTHER, e, Fraction(1, 10))
        self.assertEqual((alone[0], alone[-1]), (Fraction(-3, 5), Fraction(8, 5)))
        self.assertEqual((through[0], through[-1]), (Fraction(-31, 5), Fraction(-3, 10)))
        self.assertEqual((common[0], common[-1]), (Fraction(-3, 5), Fraction(-3, 10)))
        direct, via_b, fit = exchange.routes(OTHER, e)
        self.assertEqual((direct, via_b, fit), ((alone[0], alone[-1]), (through[0], through[-1]), (common[0], common[-1])))
        self.assertEqual(direct[0] - via_b[1], Fraction(-3, 10))                 # the dashed line, below zero

    def test_apart_above_zero_and_overlapping_below(self):
        for e, step in ((Fraction(0), Fraction(1, 4)), (Fraction(1, 5), Fraction(1, 10)), (Fraction(1, 4), Fraction(1, 4)),
                        (Fraction(3, 10), Fraction(1, 10)), (Fraction(1, 2), Fraction(1, 4))):
            alone, through, common = offsets_of_c(OTHER, e, step)
            line = Fraction(3, 2) - 6 * e
            direct, via_b, _ = exchange.routes(OTHER, e)
            self.assertEqual(direct[0] - via_b[1], line, e)
            apart = min(abs(a - b) for a in alone for b in through)             # the distance of two sets
            shared = common[-1] - common[0] if common else 0
            self.assertEqual((apart, shared), (max(line, 0), max(-line, 0)), e)
            self.assertEqual(bool(common), line <= 0, e)
        e = Fraction(2)                                                          # far beyond the figure
        common = offsets_of_c(OTHER, e, Fraction(1, 4), reach=15)[2]
        self.assertEqual((common[-1] - common[0], 6 * e - Fraction(3, 2)), (9, Fraction(21, 2)))   # the overlap is less there

    def test_the_key_of_the_figure(self):
        source = (ROOT / "figures" / "make_figures.py").read_text(encoding="utf-8")
        for piece in ('"the two offsets of C apart,"', '"below zero overlapping"'):
            self.assertIn(piece, source)
        self.assertNotIn('"the two offsets of C, apart"', source)                # the key of Version 45


class TheRangeOfTheFormula(unittest.TestCase):
    def test_the_least_and_the_greatest_time_for_any_bound(self):
        for k in range(0, 61):
            e = Fraction(k, 20)
            fit = exchange.routes(exchange.A_TO_C, e)[2]
            least, most = exchange.gap(300, 299, fit, e)
            self.assertEqual(least, max(Fraction(1, 2) - 6 * e, -2 - 4 * e), e)
            self.assertEqual(most, 2 + 4 * e, e)
            self.assertEqual(least == Fraction(1, 2) - 6 * e, e <= Fraction(5, 4), e)

    def test_from_the_grid(self):
        for e in (Fraction(1), Fraction(5, 4), Fraction(3, 2), Fraction(2)):
            common = offsets_of_c(exchange.A_TO_C, e, Fraction(1, 4), reach=15)[2]
            least, most = (299 - common[-1] - e) - (300 + e), (299 - common[0] + e) - (300 - e)
            self.assertEqual((least, most), exchange.gap(300, 299, (common[0], common[-1]), e), e)
            self.assertEqual((least, most), (max(Fraction(1, 2) - 6 * e, -2 - 4 * e), 2 + 4 * e), e)
        self.assertEqual((least, most), (-10, 10))                               # at a bound of 2 ms
        self.assertEqual(Fraction(1, 2) - 6 * 2, Fraction(-23, 2))               # what the first expression gives there

    def test_which_exchange_sets_the_upper_end_of_the_offset(self):
        for e, by in ((Fraction(1), "via B"), (Fraction(5, 4), "both"), (Fraction(3, 2), "direct")):
            direct, via_b, fit = exchange.routes(exchange.A_TO_C, e)
            self.assertEqual({"via B": via_b[1] < direct[1], "both": via_b[1] == direct[1], "direct": via_b[1] > direct[1]}[by], True)
            self.assertEqual(fit[1], min(direct[1], via_b[1]))


NAMES = ("AB1", "AB2", "BC", "AC")
QUARTERS = range(-60, 61)                              # offsets of -15 to +15 ms, in quarters of a millisecond


def quarters(stamps):
    out = tuple(Fraction(str(x)) * 4 for x in stamps)
    assert all(x.denominator == 1 for x in out)
    return tuple(int(x) for x in out)


def follows(stamps, theta):
    a1, b2, b3, a4 = stamps
    return a1 <= b2 - theta <= b3 - theta <= a4


def solve(sets):
    """The offsets of C from A, in quarters, that exact exchanges allow: `sets` maps some of the four names to
    stamps in quarters. None if no pair of offsets, of B and of C, fits them."""
    of_b = [b for b in QUARTERS if all(follows(sets[n], b) for n in ("AB1", "AB2") if n in sets)]
    of_c = [c for c in QUARTERS
            if ("AC" not in sets or follows(sets["AC"], c))
            and any("BC" not in sets or follows(sets["BC"], c - b) for b in of_b)]
    return (of_c[0], of_c[-1]) if of_c else None


def the_four(a_to_c):
    return dict(zip(NAMES, map(quarters, (*exchange.EXCHANGES, exchange.B_TO_C, a_to_c))))


def one_stamp(a_to_c, reach=20):
    """For each stamp that alone can be moved so that the four exchanges agree: its nearest such time, in ms."""
    four, found = the_four(a_to_c), []
    for i, name in enumerate(NAMES):
        for j in range(4):
            good = []
            for move in range(-reach, reach + 1):
                stamps = list(four[name])
                stamps[j] += move
                if solve({**four, name: tuple(stamps)}) is not None:
                    good.append(move)
            if good:
                nearest = min(good, key=abs)
                assert [m for m in good if abs(m) == abs(nearest)] == [nearest]
                found.append((i, j, Fraction(four[name][j] + nearest, 4)))
    return found


class WhichStampCanBeAtFault(unittest.TestCase):
    def test_three_exchanges_show_the_contradiction(self):
        four = the_four(OTHER)
        without = {name: solve({n: s for n, s in four.items() if n != name}) for name in NAMES}
        self.assertEqual(without, {"AB1": None, "AB2": (0, 0), "BC": (0, 4), "AC": (-20, -6)})   # in quarters
        bad = []
        for size in range(5):
            for names in __import__("itertools").combinations(NAMES, size):
                if solve({n: four[n] for n in names}) is None:
                    bad.append(names)
        self.assertEqual(bad, [("AB2", "BC", "AC"), NAMES])                      # of the sixteen subsets

    def test_the_three_stamps_of_the_review(self):
        found = one_stamp(OTHER)
        self.assertEqual(found, [(1, 0, Fraction(397, 2)), (2, 0, Fraction(77, 2)), (3, 3, Fraction(127, 2))])
        self.assertEqual(exchange.suspects(OTHER), found)
        four = the_four(OTHER)
        for i, j, t in found:
            self.assertEqual(abs(t - Fraction(four[NAMES[i]][j], 4)), Fraction(3, 2))   # the two offsets miss by 1.5 ms

    def test_the_other_end_of_a_message_cannot_move_far_enough(self):
        with self.assertRaises(ValueError):
            exchange.offset(200, 198, 197.5, 204.5)                              # B would reply before it received
        four = the_four(OTHER)
        self.assertIsNone(solve({**four, "AB2": quarters((200, 198, 197.5, 204.5))}))
        self.assertIsNone(solve({**four, "AB2": quarters((200, 197.5, 197.5, 204.5))}))   # as far as it can go

    def test_nothing_is_at_fault_where_the_exchanges_agree(self):
        self.assertEqual(exchange.suspects(exchange.A_TO_C), [])
        self.assertEqual(one_stamp(exchange.A_TO_C, reach=0), [(i, j, Fraction(the_four(exchange.A_TO_C)[n][j], 4))
                                                              for i, n in enumerate(NAMES) for j in range(4)])

    def test_other_exchanges_of_a_and_c(self):
        for a_to_c in ((60, 61, 62, 61.5), (60, 61, 62, 63), (60, 60.5, 60.5, 61), (60, 57, 57.5, 65), (60, 53, 54, 65)):
            agree = solve(the_four(a_to_c)) is not None
            found = [] if agree else one_stamp(a_to_c)                           # nothing is at fault where they agree
            self.assertEqual(exchange.suspects(a_to_c), found, a_to_c)
            self.assertEqual(agree, a_to_c == (60, 57, 57.5, 65))
        self.assertEqual(exchange.suspects((60, 61, 62, 61.5)), [(2, 0, 38), (3, 3, Fraction(127, 2))])
        self.assertEqual(exchange.suspects((60, 53, 54, 65)), [(2, 3, 44), (3, 0, 58)])   # an offset below those by way of B

    def test_routes_with_other_exchanges(self):
        # Version 46 gave routes() the other exchanges as arguments, for suspects(). Since Version 47 suspects()
        # measures how far the exchanges miss by itself (tests/test_review46.py), and routes() takes two again.
        with self.assertRaises(TypeError):
            exchange.routes(OTHER, 0, exchange.EXCHANGES, exchange.B_TO_C)
        self.assertEqual(exchange.routes(OTHER), ((0, 1), (-5, Fraction(-3, 2)), None))

    def test_the_transcript(self):
        text = expected("ch06_exchange")
        lines = text.splitlines()
        self.assertEqual(len(lines), 39)
        rows = []
        four = the_four(OTHER)
        for i, j, t in one_stamp(OTHER):
            was = Fraction(four[NAMES[i]][j], 4)
            rows.append(f"{('A-B 1', 'A-B 2', 'B-C', 'A-C')[i]}, stamp {j + 1}: {int(was)} to {float(t)} ms")
        self.assertEqual(lines[33:37], ["one stamp at fault, with 62 ms:"] + rows)
        self.assertEqual(rows, ["A-B 2, stamp 1: 200 to 198.5 ms", "B-C, stamp 1: 40 to 38.5 ms", "A-C, stamp 4: 62 to 63.5 ms"])
        self.assertEqual(printed(exchange.main), text)


def version_45(service, spacing, duration):
    """The numbers of requests that the two loops of Version 45 send, with times in floats."""
    t, closed = 0.0, 0
    while t < duration:
        done = t + service
        closed += 1
        t = max(t + spacing, math.ceil(done / spacing) * spacing)
    due, opened = 0.0, 0
    while due < duration:
        opened += 1
        due += spacing
    return closed, opened


def fcfs(slow, closed, stalls=((20_000, 1000), (40_000, 1000))):
    """One minute at one server, a millisecond at a time: the (send, completion) of every request. Requests wait
    in a queue and are served in the order of their arrival; no service starts in a stall; a service takes
    `slow` ms if another request waits as it starts, and 1 ms if none does. The requests are due every 10 ms
    (closed false), or come from a tester that sends at the first of those moments at which it has its response
    and has not yet sent (closed true)."""
    queue, pairs = collections.deque(), []
    current, busy_until, next_send, last_send = None, 0, 0, None
    for t in range(0, 64_000):
        if current is not None and t == busy_until:
            pairs.append((current, t))
            current = None
            if closed:
                next_send = max(last_send + 10, -(-t // 10) * 10)
        if t < 60_000 and (t == next_send if closed else t % 10 == 0):
            queue.append(t)
            last_send, next_send = t, None
        if current is None and queue and not any(begin <= t < begin + length for begin, length in stalls):
            current = queue.popleft()
            busy_until = t + (slow if queue else 1)
    assert current is None and not queue
    return pairs


def p99(xs):
    return statistics.quantiles(xs, n=100, method="inclusive")[98]


class TicksInWholeNanoseconds(unittest.TestCase):
    def test_the_table_of_the_review(self):
        table = {0.1: (487, 600, 600), 0.25: (240, 240, 240), 0.3: (177, 201, 200)}
        for spacing, (closed, opened, due) in table.items():
            microseconds = round(spacing * 1000)
            self.assertEqual(version_45(spacing, spacing, 60.0), (closed, opened), spacing)   # the table of the text
            self.assertEqual(len(range(0, 60_000, microseconds)), due)
            step = microseconds * 1000                                           # the listing counts nanoseconds
            with constants(omission, INTERVAL=step, SERVICE=step, DURATION=60 * omission.MS, STALLS={}):
                self.assertEqual((len(omission.closed_loop()), len(omission.open_loop())), (due, due), spacing)
                self.assertEqual(set(omission.closed_loop()), {step / omission.MS})

    def test_what_the_text_says_of_floats(self):
        self.assertEqual(0.1 + 0.2, 0.30000000000000004)                         # the completion that the text names
        self.assertEqual(math.ceil((0.2 + 0.1) / 0.1), 4)                        # the fourth scheduled moment, for the third
        self.assertEqual((0.25).hex(), "0x1.0000000000000p-2")                   # a quarter is a number that a float holds
        self.assertNotEqual(Fraction(0.1), Fraction(1, 10))
        self.assertEqual(version_45(10.0, 10.0, 60_000.0), (6000, 6000))         # whole milliseconds did no harm
        source = (ROOT / "examples" / "ch10_coordinated_omission.py").read_text(encoding="utf-8")
        self.assertNotIn("math.ceil", source)

    def test_whole_numbers_throughout(self):
        self.assertEqual(omission.MS, 1_000_000)
        times = [omission.INTERVAL, omission.SERVICE, omission.DURATION, *omission.STALLS, *omission.STALLS.values()]
        self.assertTrue(all(type(t) is int for t in times))
        self.assertEqual([t // omission.MS for t in times], [10, 1, 60_000, 20_000, 40_000, 1000, 1000])
        self.assertEqual((omission.resume(19_999 * omission.MS), omission.resume(20_000 * omission.MS),
                          omission.resume(20_999 * omission.MS + 999_999), omission.resume(21_000 * omission.MS)),
                         (19_999 * omission.MS, 21_000 * omission.MS, 21_000 * omission.MS, 21_000 * omission.MS))

    def test_other_services_and_spacings(self):
        rng = random.Random(45)
        for _ in range(200):
            service, spacing = rng.randrange(1, 3000), rng.randrange(1, 1500)   # microseconds
            sends, t = 0, 0
            while t < 100_000:
                sends += 1
                done, t = t + service, t + spacing
                while t < done:
                    t += spacing
            with constants(omission, INTERVAL=spacing * 1000, SERVICE=service * 1000, DURATION=100 * omission.MS, STALLS={}):
                self.assertEqual(len(omission.closed_loop()), sends, (service, spacing))
                self.assertEqual(len(omission.open_loop()), len(range(0, 100_000, spacing)))

    def test_back_filling_in_whole_nanoseconds(self):
        self.assertNotEqual(0.35 - 0.1, 0.25)                                    # a float loses the quarter
        with constants(omission, INTERVAL=100_000):                              # a spacing of 0.1 ms
            self.assertEqual(omission.back_fill([0.35]), [0.05, 0.15, 0.25, 0.35])
            self.assertEqual(omission.back_fill([0.1, 0.2]), [0.1, 0.1, 0.2])
        self.assertEqual(omission.back_fill([25.0, 5.0]), [5.0, 5.0, 15.0, 25.0])

    def test_the_durations_of_the_book(self):
        measured = sorted(float(done - sent) for sent, done in fcfs(1, True))
        self.assertEqual((len(measured), measured[-3:]), (5800, [1.0, 1001.0, 1001.0]))
        self.assertEqual(omission.closed_loop(), measured)
        self.assertEqual(omission.open_loop(), sorted(float(done - due) for due, done in fcfs(1, False)))


class TwoServersOneTrace(unittest.TestCase):
    def test_the_table_of_the_review(self):
        rows = []
        for slow in (1, 5):
            xs = [done - due for due, done in fcfs(slow, False)]
            over = sum(x > 500 for x in xs)
            rows.append((len(xs), over, f"{100 * (len(xs) - over) / len(xs):.2f}", f"{p99(xs):.2f}"))
        self.assertEqual(rows, [(6000, 112, "98.13", "731.09"), (6000, 202, "96.63", "855.05")])

    def test_the_open_loop_of_the_listing(self):
        for busy, slow in ((None, 1), (omission.SERVICE, 1), (5 * omission.MS, 5), (3 * omission.MS, 3)):
            want = sorted(float(done - due) for due, done in fcfs(slow, False))
            self.assertEqual(omission.open_loop(busy), want, slow)
        self.assertEqual(omission.over(omission.open_loop(5 * omission.MS)), (202, 100 * 5798 / 6000))
        self.assertEqual(omission.over(omission.open_loop()), (112, 100 * 5888 / 6000))

    def test_the_closed_loop_cannot_tell_them_apart(self):
        first, second = fcfs(1, True), fcfs(5, True)
        self.assertEqual(first, second)                                          # every send and every completion
        self.assertEqual(len(first), 5800)
        self.assertEqual(sorted(float(done - sent) for sent, done in second), omission.closed_loop())
        self.assertNotEqual(fcfs(1, False), fcfs(5, False))

    def test_an_objective_that_one_server_meets(self):
        within = [omission.over(omission.open_loop(busy))[1] for busy in (None, 5 * omission.MS)]
        self.assertEqual([f"{share:.2f}" for share in within], ["98.13", "96.63"])
        self.assertTrue(within[0] >= 98 > within[1])
        repaired = omission.over(omission.back_fill(omission.closed_loop()))[1]   # one repair for both servers
        self.assertEqual(f"{repaired:.2f}", "98.30")

    def test_the_transcript(self):
        text = expected("ch10_coordinated_omission")
        lines = text.splitlines()
        self.assertEqual(len(lines), 12)
        xs = sorted(done - due for due, done in fcfs(5, False))
        over = sum(x > 500 for x in xs)
        self.assertEqual(lines[10], f"{'5 ms if busy':<12}{len(xs):>5}{over:>6}{100 * (len(xs) - over) / len(xs):>9.4f} %")
        self.assertEqual(lines[10], "5 ms if busy 6000   202  96.6333 %")
        cuts = statistics.quantiles(xs, n=10_000, method="inclusive")            # interpolated around rank q(n - 1)
        cells = "".join(f"{q:>7.0f}" for q in (cuts[4999], cuts[9899], cuts[9989], cuts[9998]))
        self.assertEqual(lines[11], f"{'5 ms if busy':<12}{len(xs):>5}{cells}{statistics.fmean(xs):>5.1f}")
        self.assertEqual(lines[11], "5 ms if busy 6000      1    855    990   1005 34.8")
        self.assertEqual(printed(omission.main), text)


def two_sided(only_a, only_b):
    """The exact two-sided probability of a division at least as uneven, from binomial coefficients."""
    n, low = only_a + only_b, min(only_a, only_b)
    return min(Fraction(1), 2 * Fraction(sum(math.comb(n, k) for k in range(low + 1)), 2 ** n))


def every_table(known, right_a, right_b, n):
    """Each table of joint outcomes for all n items that the verified pairs and the two totals allow: the cells
    of the items that are not verified are tried one by one."""
    rest = n - sum(known)
    out = []
    for both in range(rest + 1):
        for only_a in range(rest - both + 1):
            for only_b in range(rest - both - only_a + 1):
                cells = (known[0] + both, known[1] + only_a, known[2] + only_b, known[3] + rest - both - only_a - only_b)
                if cells[0] + cells[1] == right_a and cells[0] + cells[2] == right_b:
                    out.append(cells)
    return out


def in_decimals(p, digits):
    """A fraction as Transcript 17.1 prints a probability, by decimal arithmetic with that many digits."""
    with decimal.localcontext() as context:
        context.prec = digits
        context.rounding = decimal.ROUND_HALF_EVEN
        value = Decimal(p.numerator) / Decimal(p.denominator)
        if p == 0 or p >= Fraction(1, 10_000):
            return str(value.quantize(Decimal("0.000001")))
        figures, power = f"{value:.2e}".split("e")
        return f"{figures}e-{-int(power):02d}"


class PairsThatLostTheirIdentifiers(unittest.TestCase):
    KNOWN, A, B, N = (1560, 132, 163, 95), 1722, 1758, 2000

    def test_the_numbers_of_the_review(self):
        self.assertEqual(sum(self.KNOWN), 1950)
        tables = every_table(self.KNOWN, self.A, self.B, self.N)
        self.assertEqual(len(tables), 16)
        self.assertEqual({(t[0] + t[1], t[0] + t[2]) for t in tables}, {(1722, 1758)})
        self.assertEqual(sorted(t[0] for t in tables), list(range(1575, 1591)))  # right for both, of all 2000
        ps = {t[0]: two_sided(t[1], t[2]) for t in tables}
        self.assertEqual((f"{float(ps[1590]):.7f}", f"{float(ps[1575]):.7f}"), ("0.0431285", "0.0538539"))
        self.assertTrue(ps[1581] < Fraction(1, 20) < ps[1580])
        self.assertEqual((f"{float(ps[1581]):.6f}", f"{float(ps[1580]):.6f}"), ("0.049510", "0.050229"))
        self.assertEqual(accuracy.partly(self.KNOWN, self.A, self.B, self.N), [ps[k] for k in range(1590, 1574, -1)])

    def test_what_a_few_verified_pairs_settle(self):
        tables = every_table(self.KNOWN, self.A, self.B, self.N)
        six = [t for t in tables if t[3] - self.KNOWN[3] >= 6]                   # six of the 50 wrong for both
        ten = [t for t in tables if t[1] - self.KNOWN[1] >= 10]                  # ten of them right for A alone
        self.assertEqual((len(six), len(ten)), (10, 6))
        self.assertEqual((min(t[0] for t in six), max(t[0] for t in ten)), (1581, 1580))
        for some, more in ((six, (0, 0, 0, 6)), (ten, (0, 10, 0, 0))):
            known = tuple(k + m for k, m in zip(self.KNOWN, more))
            self.assertEqual(sorted(every_table(known, self.A, self.B, self.N)), sorted(some))
            want = sorted(two_sided(t[1], t[2]) for t in some)
            self.assertEqual(accuracy.partly(known, self.A, self.B, self.N), want)
        level = Fraction(1, 20)
        self.assertTrue(max(two_sided(t[1], t[2]) for t in six) < level < min(two_sided(t[1], t[2]) for t in ten))

    def test_no_pair_verified_is_the_range_of_two_scores(self):
        for right_b in (1727, 1758, 1768, 1722, 1676):
            ps = accuracy.partly((0, 0, 0, 0), self.A, right_b, self.N)
            self.assertEqual((ps[0], ps[-1]), accuracy.limits(self.A, right_b, self.N))
            self.assertEqual(ps, sorted(ps))
        self.assertEqual(len(accuracy.partly((0, 0, 0, 0), self.A, self.B, self.N)), 1722 - 1480 + 1)

    def test_small_cases_against_every_table(self):
        rng = random.Random(17)
        tried = refused = 0
        while tried < 400:
            n = rng.randint(1, 14)
            cuts = sorted(rng.randint(0, rng.randint(0, n)) for _ in range(3))
            known = (cuts[0], cuts[1] - cuts[0], cuts[2] - cuts[1], rng.randint(0, n - cuts[2]))
            right_a, right_b = rng.randint(0, n), rng.randint(0, n)
            tables = every_table(known, right_a, right_b, n)
            if not tables:
                refused += 1
                with self.assertRaises(ValueError) as caught:
                    accuracy.partly(known, right_a, right_b, n)
                self.assertEqual(str(caught.exception), "counts do not fit")
                continue
            tried += 1
            tables.sort(reverse=True)                                            # from the most items right for both
            got = accuracy.partly(known, right_a, right_b, n)
            self.assertEqual(got, [two_sided(t[1], t[2]) for t in tables], (known, right_a, right_b, n))
            self.assertEqual(got, sorted(got))
        self.assertGreater(refused, 100)

    def test_the_transcript(self):
        text = expected("ch17_accuracy")
        lines = text.splitlines()
        self.assertEqual(len(lines), 43)
        rows = []
        for name, more in (("as they are:", (0, 0, 0, 0)), ("6 wrong for both:", (0, 0, 0, 6)), ("10 for A alone:", (0, 10, 0, 0))):
            known = tuple(k + m for k, m in zip(self.KNOWN, more))
            ps = [two_sided(t[1], t[2]) for t in every_table(known, self.A, self.B, self.N)]
            rows.append(f"  {name:<18}{in_decimals(min(ps), 3000)} to {in_decimals(max(ps), 3000)}")
        self.assertEqual(lines[34:38], ["1950 pairs verified, 50 not:"] + rows)
        self.assertEqual(rows, ["  as they are:      0.043128 to 0.053854", "  6 wrong for both: 0.043128 to 0.049510",
                                "  10 for A alone:   0.050229 to 0.053854"])
        self.assertEqual(printed(accuracy.main), text)


class ProbabilitiesBeyond4300Digits(unittest.TestCase):
    def test_what_version_45_did(self):
        limit = sys.get_int_max_str_digits()
        if not 0 < limit <= 4300:
            self.skipTest("the limit for the text of an integer is not the default")
        self.assertEqual(len(str(2 ** 14284)), 4300)
        with self.assertRaises(ValueError):
            len(str(2 ** 14285))                                                 # the line of Version 45

    def test_the_table_of_the_review(self):
        table = {23: "1.19e-07", 14284: "1.22e-4300", 14285: "6.12e-4301"}
        for power, text in table.items():
            p = Fraction(1, 2 ** power)
            self.assertEqual(accuracy.shown(p), text)
            self.assertEqual(in_decimals(p, 22_000), text)

    def test_longer_ones(self):
        rng = random.Random(4300)
        ps = [Fraction(1, 2 ** power) for power in (1075, 14286, 20_000, 33_219, 33_220)]
        ps += [Fraction(1, 10 ** k) for k in (5, 4299, 4300, 4301, 6000)]
        ps += [Fraction(rng.randrange(1, 10 ** 9), 10 ** rng.randrange(4290, 4320)) for _ in range(20)]
        ps += [Fraction(9_995, 10 ** 4400), Fraction(99_949, 10 ** 4400), Fraction(99_950, 10 ** 4400)]
        for p in ps:
            self.assertEqual(accuracy.shown(p), in_decimals(p, 22_000))
        self.assertEqual([accuracy.shown(p) for p in ps[5:10]], ["1.00e-05", "1.00e-4299", "1.00e-4300", "1.00e-4301", "1.00e-6000"])

    def test_the_digits_of_a_whole_number(self):
        rng = random.Random(10)
        for _ in range(2000):
            q = rng.randrange(1, 10 ** rng.randrange(1, 400))
            self.assertEqual(Decimal(q).adjusted() + 1, len(str(q)))
        for k in (1, 2, 28, 29, 4299, 4300):                                     # more digits than a context keeps
            self.assertEqual((Decimal(10 ** k - 1).adjusted() + 1, Decimal(10 ** k).adjusted() + 1), (k, k + 1))

    def test_the_context_is_left_alone(self):
        before = (decimal.getcontext().prec, decimal.getcontext().rounding)
        accuracy.shown(Fraction(1, 2 ** 20_000))
        self.assertEqual((decimal.getcontext().prec, decimal.getcontext().rounding), before)
        self.assertEqual(before[0], 28)                                          # no test of the run has changed it
        for digits in (5, 28, 60):                                               # and whatever it holds, the text is one
            with decimal.localcontext() as context:
                context.prec = digits
                self.assertEqual(accuracy.shown(Fraction(1, 2 ** 14285)), "6.12e-4301")

    def test_the_probabilities_of_the_book_are_printed_as_they_were(self):
        lines = expected("ch17_accuracy").splitlines()
        self.assertEqual(lines[32:34], ["23 items to B alone, none to A alone:", "  p = 1.19e-07, two-sided 2.38e-07"])
        # Until Version 69 both ends of a range were printed to the nearest digit. Since Version 70 the least
        # probability is rounded downward and the greatest upward, so that the printed range holds them all
        # (tests/test_review69.py). The upper ends of two rows read 0.124738 and 0.046195: the greatest
        # probabilities are 0.1247383875... and 0.0461954783...
        self.assertEqual(lines[25:28], ["  b = 1758: 2.91e-11 to 0.124739", "  b = 1767: 5.68e-14 to 0.051495",
                                        "  b = 1768: 2.84e-14 to 0.046196"])


class TheTwentySeventhFigure(unittest.TestCase):
    def test_the_script(self):
        source = (ROOT / "figures" / "make_figures.py").read_text(encoding="utf-8")
        self.assertEqual(source.count("\n    save(fig, "), 32)
        for piece in ("def unlinked_pairs():", "c17.partly(known, 1722, 1758, 2000)", 'save(fig, "fig-17-2-unlinked-pairs.png")',
                      "known = (1560, 132 + only_a, 163, 95 + neither)", "assert counts == [16, 10, 6]",
                      "    joint_outcomes()\n    unlinked_pairs()\n    batch_labels()\n    reliability()\n"):
            self.assertIn(piece, source)

    def test_what_it_shows(self):
        rows = []
        for only_a, neither in ((0, 0), (0, 6), (10, 0)):
            known = (1560, 132 + only_a, 163, 95 + neither)
            rows.append([float(two_sided(t[1], t[2])) for t in every_table(known, 1722, 1758, 2000)])
        self.assertEqual([len(row) for row in rows], [16, 10, 6])                # the dots of each row
        self.assertTrue(all(0.0425 < p < 0.0545 for row in rows for p in row))   # within the axis
        self.assertTrue(max(rows[1]) < 0.05 < min(rows[2]))                      # on either side of the dashed line
        self.assertEqual(sorted(rows[1] + rows[2]), sorted(rows[0]))


if __name__ == "__main__":
    unittest.main()
