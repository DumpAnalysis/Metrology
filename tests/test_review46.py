"""Regression tests added after the review of Version 46.

The review found that the text of Chapter 10 called the first row of Transcript 10.1 a measurement of service
time, where the listing records the time from a send to its completion; that open_loop() of Listing 10.1 let
the last request of a run be served slowly for a request that never arrives; that suspects() of Listing 6.3
failed where an exchange by itself allows no offset, although one stamp can repair it; and that partly() of
Listing 17.1 accepted a negative count of verified pairs. It asked how many stamps can each explain a record as
one stamp moves, which Figure 6.6 now draws, and what two batches that survive an export settle about fifty
pairs that did not.

Each test computes what it expects independently of the code under test: the generator and the server by a
simulation that steps through time a millisecond at a time with a queue, and keeps the start of every service;
the stamps by trying offsets, and moves of each stamp, in whole quarters of a millisecond; the exact test from
binomial coefficients over every table that the counts of each batch allow.
"""
import collections
import contextlib
import io
import itertools
import math
import random
import unittest
from fractions import Fraction
from pathlib import Path

from examples import ch06_exchange as exchange
from examples import ch10_coordinated_omission as omission
from examples import ch17_accuracy as accuracy

ROOT = Path(__file__).resolve().parent.parent
MS = omission.MS


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


def served(spacing, window, stalls, slow, closed=False, service=1):
    """One server and its queue, a millisecond at a time: (send, start of service, completion) of every
    request. Requests are due every `spacing` ms while the time is below `window` (closed false), or come from
    a tester that sends at the first of those moments at which it has its response and has not yet sent (closed
    true). No service starts in a stall. A service takes `slow` ms if another request waits in the queue as
    it starts, and `service` ms if none does."""
    queue, out = collections.deque(), []
    current, busy_until, next_send, last_send = None, 0, 0, None
    t = 0
    while t < window or current is not None or queue:
        if current is not None and t == busy_until:
            out.append(current + (t,))
            current = None
            if closed:
                next_send = max(last_send + spacing, -(-t // spacing) * spacing)
        if t < window and (t == next_send if closed else t % spacing == 0):
            queue.append(t)
            last_send, next_send = t, None
        if current is None and queue and not any(begin <= t < begin + length for begin, length in stalls):
            current = (queue.popleft(), t)
            busy_until = t + (slow if queue else service)
        t += 1
    return out


BOOK = ((20_000, 1000), (40_000, 1000))


class TheFirstRowIsResponseTime(unittest.TestCase):
    def test_the_decomposition_of_the_review(self):
        rows = served(10, 60_000, BOOK, 1, closed=True)
        self.assertEqual(len(rows), 5800)
        self.assertEqual({done - start for _, start, done in rows}, {1})         # every request gets 1 ms of service
        waited = [(send, start, done) for send, start, done in rows if start > send]
        self.assertEqual(waited, [(20_000, 21_000, 21_001), (40_000, 41_000, 41_001)])
        self.assertEqual([(start - send, done - start, done - send) for send, start, done in waited], [(1000, 1, 1001)] * 2)

    def test_the_listing_records_send_to_completion(self):
        rows = served(10, 60_000, BOOK, 1, closed=True)
        self.assertEqual(omission.closed_loop(), sorted(float(done - send) for send, _, done in rows))
        self.assertNotEqual(omission.closed_loop(), [float(done - start) for _, start, done in rows])
        self.assertEqual(sorted(set(omission.closed_loop())), [1.0, 1001.0])
        self.assertEqual(omission.resume(20_000 * MS) - 20_000 * MS, 1000 * MS)  # the wait of the listing

    def test_fewer_requests_during_the_stalls(self):
        sends = [send for send, _, _ in served(10, 60_000, BOOK, 1, closed=True)]
        by_second = collections.Counter(send // 1000 for send in sends)
        self.assertEqual({second: n for second, n in by_second.items() if n != 100}, {20: 1, 21: 99, 40: 1, 41: 99})
        self.assertEqual(sum(by_second.values()), 6000 - 200)

    def test_the_transcript_is_unchanged(self):
        text = expected("ch10_coordinated_omission")
        self.assertEqual(len(text.splitlines()), 12)
        self.assertEqual(printed(omission.main), text)


class ANextRequestHasToArrive(unittest.TestCase):
    def test_the_table_of_the_review(self):
        rows = served(10, 30, ((0, 30),), 5)
        self.assertEqual(rows, [(0, 30, 35), (10, 35, 40), (20, 40, 41)])
        self.assertEqual([done - send for send, _, done in rows], [35, 30, 21])
        with constants(omission, DURATION=30 * MS, STALLS={0: 30 * MS}):
            self.assertEqual(omission.open_loop(5 * MS), [21.0, 30.0, 35.0])     # Version 46: 25.0 for the last
            self.assertEqual(omission.open_loop(), [31.0, 22.0, 13.0][::-1])

    def test_what_version_46_did(self):
        out, free = [], 0
        for due in (0, 10, 20):
            start = max(due, free, 30)
            late = due + 10 <= start                                             # also for a request that never comes
            free = start + (5 if late else 1)
            out.append(free - due)
        self.assertEqual(out, [35, 30, 25])

    def test_finite_runs_against_the_queue(self):
        cases = 0
        for spacing, ticks, stall, slow in itertools.product((5, 10), (1, 2, 3, 5, 8), (0, 7, 30, 55), (2, 5, 12)):
            want = sorted(float(done - send) for send, _, done in served(spacing, spacing * ticks, ((0, stall),), slow))
            stalls = {0: stall * MS} if stall else {}
            with constants(omission, INTERVAL=spacing * MS, DURATION=spacing * ticks * MS, STALLS=stalls):
                self.assertEqual(omission.open_loop(slow * MS), want, (spacing, ticks, stall, slow))
            cases += 1
        self.assertEqual(cases, 120)

    def test_random_runs_against_the_queue(self):
        rng = random.Random(46)
        for _ in range(200):
            spacing, ticks = rng.randint(2, 12), rng.randint(1, 12)
            begin, length = rng.randrange(0, spacing * ticks), rng.randint(0, 60)
            slow = rng.randint(1, 15)
            want = sorted(float(done - send) for send, _, done in served(spacing, spacing * ticks, ((begin, length),), slow))
            with constants(omission, INTERVAL=spacing * MS, DURATION=spacing * ticks * MS, STALLS={begin * MS: length * MS}):
                self.assertEqual(omission.open_loop(slow * MS), want, (spacing, ticks, begin, length, slow))

    def test_a_service_of_no_time_is_a_service_time(self):
        with constants(omission, DURATION=30 * MS, STALLS={0: 30 * MS}):
            self.assertEqual(omission.open_loop(0), [11.0, 20.0, 30.0])          # 30, 30 and 31 ms after 0: the last waits for none
            self.assertEqual(omission.open_loop(None), omission.open_loop())
        self.assertNotEqual(omission.open_loop(0), omission.open_loop())         # Version 46 took 0 for none given

    def test_the_runs_of_the_book(self):
        for busy, slow in ((None, 1), (5 * MS, 5)):
            want = sorted(float(done - send) for send, _, done in served(10, 60_000, BOOK, slow))
            self.assertEqual(omission.open_loop(busy), want)
        self.assertEqual((omission.over(omission.open_loop())[0], omission.over(omission.open_loop(5 * MS))[0]), (112, 202))


NAMES = ("AB1", "AB2", "BC", "AC")
QUARTERS = range(-80, 81)                              # offsets of -20 to +20 ms, in quarters of a millisecond


def quarters(stamps):
    out = tuple(Fraction(str(x)) * 4 for x in stamps)
    assert all(x.denominator == 1 for x in out)
    return tuple(int(x) for x in out)


def follows(stamps, theta):
    a1, b2, b3, a4 = stamps
    return a1 <= b2 - theta <= b3 - theta <= a4


def agree(four):
    """Whether offsets of B and of C exist, on the grid, with which the events of all four exchanges follow."""
    of_b = [b for b in QUARTERS if follows(four["AB1"], b) and follows(four["AB2"], b)]
    return any(follows(four["AC"], c) and any(follows(four["BC"], c - b) for b in of_b) for c in QUARTERS)


def the_four(a_to_c, others=None):
    return dict(zip(NAMES, map(quarters, (*(others or (*exchange.EXCHANGES, exchange.B_TO_C)), a_to_c))))


def one_stamp(four, reach=24):
    """Each stamp that alone can be moved so that the four exchanges agree, with its nearest such time in ms."""
    found = []
    for i, name in enumerate(NAMES):
        for j in range(4):
            good = []
            for move in range(-reach, reach + 1):
                stamps = list(four[name])
                stamps[j] += move
                if agree({**four, name: tuple(stamps)}):
                    good.append(move)
            if good:
                nearest = min(good, key=abs)
                assert [m for m in good if abs(m) == abs(nearest)] == [nearest]
                found.append((i, j, Fraction(four[name][j] + nearest, 4)))
    return found


def last(stamp):
    return exchange.A_TO_C[:3] + (stamp,)


class OneStampWhateverIsWrong(unittest.TestCase):
    def test_a_reply_before_its_request(self):
        with self.assertRaises(ValueError):
            exchange.offset(*last(59))                                           # where suspects() of Version 46 stopped
        four = the_four(last(59))
        self.assertFalse(agree(four))
        found = one_stamp(four)
        self.assertEqual(found, [(3, 3, Fraction(127, 2))])                      # 59 to 63.5 ms, and no other stamp
        self.assertEqual(exchange.suspects(last(59)), found)
        repaired = the_four(last(63.5))
        of_c = [c for c in QUARTERS if follows(repaired["AC"], c) and agree(repaired)]
        self.assertEqual(of_c[0], -6)                                            # C - A = -1.5 ms, in quarters

    def test_how_many_by_the_last_stamp(self):
        counts = {}
        for q in range(59 * 4, 65 * 4 + 1):
            stamp = q / 4
            four = the_four(last(stamp))
            found = [] if agree(four) else one_stamp(four)
            self.assertEqual(exchange.suspects(last(stamp)), found, stamp)
            counts[stamp] = len(found)
        want = {stamp: 1 if stamp < 61 else 2 if stamp < 62 else 3 if stamp < 62.5 else 6 if stamp < 63.5 else 0
                for stamp in counts}
        self.assertEqual(counts, want)
        self.assertEqual([counts[s] for s in (59, 61, 62, 63, 64)], [1, 2, 3, 6, 0])   # the line of the transcript

    def test_which_stamps(self):
        self.assertEqual([(i, j) for i, j, _ in exchange.suspects(last(61))], [(2, 0), (3, 3)])
        self.assertEqual(exchange.suspects(last(63)),
                         [(1, 0, Fraction(399, 2)), (1, 1, 197), (2, 0, Fraction(79, 2)), (2, 1, Fraction(85, 2)),
                          (3, 2, Fraction(123, 2)), (3, 3, Fraction(127, 2))])
        self.assertEqual(exchange.suspects(last(63)), one_stamp(the_four(last(63))))
        # At a miss of 1 ms the other end of a message can still move: the host that answers receives as it replies.
        self.assertEqual([(i, j) for i, j, _ in exchange.suspects(last(62.5))], [(1, 0), (1, 1), (2, 0), (2, 1), (3, 2), (3, 3)])
        self.assertEqual([t for _, _, t in exchange.suspects(last(62.5))][1::2][:2], [Fraction(395, 2), 43])
        self.assertEqual([(i, j) for i, j, _ in exchange.suspects(last(62.25))], [(1, 0), (2, 0), (3, 3)])

    def test_miss_against_the_grid(self):
        rng = random.Random(6)
        book = [*exchange.EXCHANGES, exchange.B_TO_C, exchange.A_TO_C]
        tried = repaired = 0
        while tried < 60:
            sets = [tuple(float(Fraction(str(x)) + Fraction(rng.randint(-6, 6), 4) * (rng.random() < 0.3)) for x in s)
                    for s in book]
            four = dict(zip(NAMES, map(quarters, sets)))
            far = exchange.miss(sets)
            self.assertEqual(far <= 0, agree(four), sets)
            if far > 5:
                continue
            tried += 1
            if far <= 0:
                continue
            found = one_stamp(four)
            for i, j, t in found:
                self.assertEqual(abs(t - Fraction(four[NAMES[i]][j], 4)), far, sets)   # no nearer time than the miss
            with constants(exchange, EXCHANGES=tuple(sets[:2]), B_TO_C=sets[2]):
                self.assertEqual(exchange.suspects(sets[3]), found, sets)
            repaired += bool(found)
        self.assertGreater(repaired, 10)

    def test_miss_of_the_book(self):
        book = [*exchange.EXCHANGES, exchange.B_TO_C]
        self.assertEqual(exchange.miss(book + [exchange.A_TO_C]), -1)            # every reply 1 ms or more after its request
        self.assertEqual(exchange.miss(book + [last(62)]), Fraction(3, 2))       # the two offsets of C
        self.assertEqual(exchange.miss(book + [last(59)]), Fraction(9, 2))
        self.assertEqual(exchange.miss(book + [last(63.5)]), 0)
        self.assertEqual(exchange.suspects(exchange.A_TO_C), [])
        self.assertEqual(exchange.suspects(last(63.5)), [])

    def test_routes_takes_two_arguments_again(self):
        with self.assertRaises(TypeError):
            exchange.routes(last(62), 0, exchange.EXCHANGES)
        self.assertEqual(exchange.routes(last(62), 0), ((0, 1), (-5, Fraction(-3, 2)), None))

    def test_the_transcript(self):
        text = expected("ch06_exchange")
        lines = text.splitlines()
        self.assertEqual(len(lines), 39)
        counts = []
        for stamp in (59, 61, 62, 63, 64):
            four = the_four(last(stamp))
            counts.append(0 if agree(four) else len(one_stamp(four)))
        self.assertEqual(lines[37:], ["such stamps, by the last stamp of A:", "  59, 61, 62, 63, 64 ms: " + " ".join(map(str, counts))])
        self.assertEqual(lines[38], "  59, 61, 62, 63, 64 ms: 1 2 3 6 0")
        self.assertEqual(printed(exchange.main), text)


def two_sided(only_a, only_b):
    """The exact two-sided probability of a division at least as uneven, from binomial coefficients."""
    n, low = only_a + only_b, min(only_a, only_b)
    return min(Fraction(1), 2 * Fraction(sum(math.comb(n, k) for k in range(low + 1)), 2 ** n))


def joint(size, right_a, right_b):
    """Each number of items of a batch that both models can have right: the cells of a table are tried."""
    return [both for both in range(size + 1)
            if min(right_a - both, right_b - both, size - right_a - right_b + both) >= 0]


def by_batches(known, batches):
    """The probability of every total of items right for both that the batches allow, from the most."""
    both, only_a, only_b, _ = known
    right_a = sum(a for _, a, _ in batches)
    right_b = sum(b for _, _, b in batches)
    totals = sorted({sum(choice) for choice in itertools.product(*(joint(*batch) for batch in batches))}, reverse=True)
    return totals, [two_sided(only_a + right_a - j, only_b + right_b - j) for j in totals]


class BatchesThatSurvived(unittest.TestCase):
    KNOWN = (1560, 132, 163, 95)

    def test_the_table_of_the_review(self):
        level = Fraction(1, 20)
        table = (([(25, 23, 23), (25, 7, 12)], (21, 30), ("0.043128", "0.049510")),
                 ([(25, 23, 13), (25, 7, 22)], (15, 20), ("0.050229", "0.053854")))
        for batches, among, shown in table:
            totals, ps = by_batches(self.KNOWN, batches)
            self.assertEqual(totals, list(range(among[1], among[0] - 1, -1)))    # every whole number between the ends
            self.assertEqual((1560 + totals[-1], 1560 + totals[0]), (1560 + among[0], 1560 + among[1]))
            self.assertEqual((f"{float(ps[0]):.6f}", f"{float(ps[-1]):.6f}"), shown)
            self.assertEqual(accuracy.partly(self.KNOWN, 1722, 1758, 2000, batches), ps)
            self.assertEqual((accuracy.shown(ps[0]), accuracy.shown(ps[-1])), shown)
            self.assertTrue(all(p < level for p in ps) or all(p > level for p in ps))
        self.assertLess(by_batches(self.KNOWN, table[0][0])[1][-1], level)       # the first export: below for every pairing
        self.assertGreater(by_batches(self.KNOWN, table[1][0])[1][0], level)     # the second: above for every pairing

    def test_the_ends_within_each_batch(self):
        self.assertEqual((joint(25, 23, 23)[0], joint(25, 7, 12)[0]), (21, 0))   # at least 21 in batch 1
        self.assertEqual((joint(25, 23, 13)[-1], joint(25, 7, 22)[-1]), (13, 7))  # at most 13 and 7
        for size, a, b in ((25, 23, 23), (25, 7, 12), (25, 23, 13), (25, 7, 22), (9, 0, 9), (9, 9, 9), (1, 1, 0)):
            self.assertEqual(joint(size, a, b), list(range(max(0, a + b - size), min(a, b) + 1)))

    def test_one_batch_is_no_batch(self):
        whole = accuracy.partly(self.KNOWN, 1722, 1758, 2000)
        self.assertEqual(accuracy.partly(self.KNOWN, 1722, 1758, 2000, [(50, 30, 35)]), whole)
        self.assertEqual(accuracy.partly(self.KNOWN, 1722, 1758, 2000, None), whole)
        self.assertEqual(len(whole), 16)
        split = accuracy.partly(self.KNOWN, 1722, 1758, 2000, [(25, 15, 20), (25, 15, 15)])
        self.assertTrue(set(split) <= set(whole))                               # batches can only narrow the range
        self.assertEqual(len(split), len(by_batches(self.KNOWN, [(25, 15, 20), (25, 15, 15)])[0]))

    def test_batches_that_do_not_add_up(self):
        for parts in ([(25, 23, 23), (25, 7, 11)], [(25, 23, 23), (24, 7, 12)], [(25, 26, 23), (25, 4, 12)],
                      [(25, 23, 23), (25, 7, 12), (0, 0, 1)], [(50, 31, 35)]):
            with self.assertRaises(ValueError) as caught:
                accuracy.partly(self.KNOWN, 1722, 1758, 2000, parts)
            self.assertEqual(str(caught.exception), "counts do not fit", parts)
        # A batch of -10 items was refused as one that does not fit until Version 48. Since Version 49 every
        # count of a batch is first a whole number that is not negative, as the other counts were already.
        with self.assertRaises(ValueError) as caught:
            accuracy.partly(self.KNOWN, 1722, 1758, 2000, [(60, 40, 35), (-10, -10, 0)])
        self.assertEqual(str(caught.exception), "need whole counts >= 0")

    def test_small_cases_against_every_table(self):
        rng = random.Random(1746)
        for _ in range(300):
            batches = []
            for _ in range(rng.randint(1, 3)):
                size = rng.randint(0, 6)
                batches.append((size, rng.randint(0, size), rng.randint(0, size)))
            known = tuple(rng.randint(0, 4) for _ in range(4))
            right_a = known[0] + known[1] + sum(a for _, a, _ in batches)
            right_b = known[0] + known[2] + sum(b for _, _, b in batches)
            n = sum(known) + sum(size for size, _, _ in batches)
            totals, ps = by_batches(known, batches)
            self.assertEqual(accuracy.partly(known, right_a, right_b, n, batches), ps, (known, batches))
            self.assertEqual(totals, list(range(totals[0], totals[-1] - 1, -1)))
            self.assertEqual(ps, sorted(ps))

    def test_the_transcript(self):
        text = expected("ch17_accuracy")
        lines = text.splitlines()
        self.assertEqual(len(lines), 43)
        rows = []
        for right_b in ((23, 12), (13, 22)):
            _, ps = by_batches(self.KNOWN, [(25, 23, right_b[0]), (25, 7, right_b[1])])
            rows.append(f"  A 23, 7; B {right_b[0]}, {right_b[1]}: {float(min(ps)):.6f} to {float(max(ps)):.6f}")
        self.assertEqual(lines[38:41], ["the 50 in two batches of 25:"] + rows)
        self.assertEqual(rows, ["  A 23, 7; B 23, 12: 0.043128 to 0.049510", "  A 23, 7; B 13, 22: 0.050229 to 0.053854"])
        self.assertEqual(printed(accuracy.main), text)

    def test_the_two_values_of_the_caption_of_figure_17_2(self):
        six = accuracy.partly((1560, 132, 163, 101), 1722, 1758, 2000)
        ten = accuracy.partly((1560, 142, 163, 95), 1722, 1758, 2000)
        self.assertEqual((accuracy.shown(six[-1]), accuracy.shown(ten[0])), ("0.049510", "0.050229"))
        self.assertTrue(six[-1] < Fraction(1, 20) < ten[0])


class CountsThatAreNotCounts(unittest.TestCase):
    def test_the_table_of_the_review(self):
        self.assertEqual(accuracy.partly((0, 0, 0, 0), 1, 7, 7), [Fraction(1, 32)])
        # What Version 46 made of a count of -1: an eighth item, and a second table with one item for A alone.
        self.assertEqual([float(two_sided(0, 6)), float(two_sided(1, 7))], [0.03125, 0.0703125])
        with self.assertRaises(ValueError) as caught:
            accuracy.partly((0, 0, 0, -1), 1, 7, 7)
        self.assertEqual(str(caught.exception), "need whole counts >= 0")

    def test_whole_numbers_and_none_below_zero(self):
        good = ((1560, 132, 163, 95), 1722, 1758, 2000)
        self.assertEqual(len(accuracy.partly(*good)), 16)
        for place in range(7):
            for bad in (-1, 1.0, 2.5, True, "3", None, Fraction(3)):
                flat = [*good[0], *good[1:]]
                flat[place] = bad
                with self.assertRaises(ValueError) as caught:
                    accuracy.partly(tuple(flat[:4]), *flat[4:])
                self.assertEqual(str(caught.exception), "need whole counts >= 0", (place, bad))

    def test_the_margins_are_checked_as_before(self):
        for known, a, b, n in (((5, 0, 0, 0), 4, 5, 10), ((0, 0, 0, 0), 11, 5, 10), ((3, 3, 3, 3), 6, 6, 11), ((1, 2, 0, 0), 3, 0, 5)):
            with self.assertRaises(ValueError) as caught:
                accuracy.partly(known, a, b, n)
            self.assertEqual(str(caught.exception), "counts do not fit", (known, a, b, n))

    def test_valid_small_cases_are_what_they_were(self):
        cases = 0
        for n in range(1, 8):
            for right_a in range(n + 1):
                for right_b in range(n + 1):
                    for verified in range(0, n + 1, 2):
                        known = (min(right_a, right_b, verified), 0, 0, 0)
                        if known[0] > verified or right_a - known[0] > n - known[0] or right_b - known[0] > n - known[0]:
                            continue
                        want = by_batches(known, [(n - known[0], right_a - known[0], right_b - known[0])])[1]
                        self.assertEqual(accuracy.partly(known, right_a, right_b, n), want)
                        cases += 1
        self.assertEqual(cases, 689)


class TheTwentyEighthFigure(unittest.TestCase):
    def test_the_script(self):
        source = (ROOT / "figures" / "make_figures.py").read_text(encoding="utf-8")
        self.assertEqual(source.count("\n    save(fig, "), 32)
        for piece in ("def stamps_at_fault():", "c6x.suspects(c6x.A_TO_C[:3] + (last,))", 'save(fig, "fig-06-6-stamps-at-fault.png")',
                      "[(59, 1), (61, 2), (62, 3), (62.5, 6), (63.5, 0)]",
                      "    three_host_bound()\n    stamps_at_fault()\n    aliasing()\n"):
            self.assertIn(piece, source)

    def test_what_it_shows(self):
        runs, before = [], None
        for k in range(49):                                                      # 59 to 65 ms in steps of 1/8 ms
            stamp = 59 + k / 8
            n = len(exchange.suspects(last(stamp)))
            if n != before:
                runs.append((stamp, n))
                before = n
        self.assertEqual(runs, [(59, 1), (61, 2), (62, 3), (62.5, 6), (63.5, 0)])   # each line begins with a filled circle
        for stamp, n in runs[:-1]:                                               # and the grid says the same of each start
            four = the_four(last(stamp))
            self.assertEqual(len(one_stamp(four)), n)
        self.assertTrue(agree(the_four(last(63.5))) and not agree(the_four(last(63.25))))


if __name__ == "__main__":
    unittest.main()
