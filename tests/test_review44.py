"""Regression tests added after the review of Version 44.

The review found that closed_loop() of Listing 10.1 never sent at the moment of a response, so that a service
time equal to the spacing halved the requests of a tester that may send then; that back_fill() goes down to the
last value above zero where a widely used rule stops at one interval; that Transcript 17.1 printed positive
probabilities as 0.000000 and that limits() could return its two ends in the wrong order; and that the text of
Chapter 17 gave its numbers for a lead of B without saying so, and none for a tie. It asked for the values in
the description of Figure 6.4, for the bound on the stamps that the order of two events on three hosts bears,
which Figure 6.5 now draws, for the plan of launches under the variances that generated the pilot data, and for
the uncertainties of the two means of launches of unequal length.

Each test computes what it expects independently of the code under test: sends by a loop in whole milliseconds
under each rule; added entries by a loop of its own; the exact test from binomial coefficients over every table
that two totals allow; printed probabilities by decimal arithmetic; the three hosts by placing the four events
of every exchange in order for offsets on a grid; plans by trying every one; the two uncertainties in fractions.
"""
import contextlib
import decimal
import io
import math
import random
import statistics
import unittest
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from examples import ch06_exchange as exchange
from examples import ch10_coordinated_omission as omission
from examples import ch13_variance_components as launches
from examples import ch17_accuracy as accuracy

ROOT = Path(__file__).resolve().parent.parent


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


def sends(service, spacing, duration, on_the_tick, stalls=()):
    """The durations that a tester measures which waits for each response, in whole milliseconds. The next
    request goes at the first scheduled moment that is not before the response (on_the_tick) or that is after
    it, and never at the moment of the request before. A request that starts in a stall of (begin, length)
    is answered `service` after the end of the stall."""
    t, out = 0, []
    while t < duration:
        done = t + service
        for begin, length in stalls:
            if begin <= t < begin + length:
                done = begin + length + service
        out.append(done - t)
        t += spacing
        while t < done or (t == done and not on_the_tick):
            t += spacing
    return sorted(out)


class ARequestThatEndsOnATick(unittest.TestCase):
    def test_the_table_of_the_review(self):
        table = {9: (6000, 6000), 10: (3000, 6000), 11: (3000, 3000), 20: (2000, 3000)}
        for service, (after, at) in table.items():
            self.assertEqual((len(sends(service, 10, 60_000, False)), len(sends(service, 10, 60_000, True))), (after, at))
            with constants(omission, SERVICE=service * omission.MS, STALLS={}):     # nanoseconds since Version 46
                durations = omission.closed_loop()
            self.assertEqual(len(durations), at, service)                        # Version 44 sent `after`
            self.assertEqual(set(durations), {float(service)})

    def test_other_services_and_spacings(self):
        rng = random.Random(44)
        for _ in range(300):
            service, spacing = rng.randint(1, 90), rng.randint(1, 40)
            want = sends(service, spacing, 4000, True)
            with constants(omission, SERVICE=service * omission.MS, INTERVAL=spacing * omission.MS,
                           DURATION=4000 * omission.MS, STALLS={}):
                self.assertEqual(omission.closed_loop(), [float(x) for x in want], (service, spacing))

    def test_the_division_is_exact_for_whole_milliseconds(self):
        rng = random.Random(10)
        for _ in range(20_000):
            done, spacing = rng.randrange(0, 10 ** 8), rng.randrange(1, 10 ** 5)
            self.assertEqual(math.ceil(float(done) / float(spacing)), -(-done // spacing), (done, spacing))
        self.assertEqual(math.ceil((0.1 + 0.2) / 0.1), 4)                        # three tenths as floats hold them
        # Version 45 divided floats, with a comment that said for which times that is exact. Since Version 46
        # the listing counts whole nanoseconds and rounds a quotient of whole numbers up (tests/test_review45.py).
        source = (ROOT / "examples" / "ch10_coordinated_omission.py").read_text(encoding="utf-8")
        self.assertIn("up = (done + INTERVAL - 1) // INTERVAL\n        tick = max(tick + 1, up)", source)
        self.assertNotIn("math.ceil(done / INTERVAL)", source)

    def test_a_request_is_never_sent_with_the_one_before(self):
        with constants(omission, SERVICE=0, STALLS={}):                          # a response at the moment of the send
            self.assertEqual(len(omission.closed_loop()), 6000)

    def test_the_values_of_the_book(self):
        stalls = ((20_000, 1000), (40_000, 1000))
        after, at = sends(1, 10, 60_000, False, stalls), sends(1, 10, 60_000, True, stalls)
        self.assertEqual(after, at)                                              # no response of the book is on a tick
        self.assertEqual((len(at), at[-3:]), (5800, [1, 1001, 1001]))
        self.assertEqual(omission.closed_loop(), [float(x) for x in at])
        self.assertEqual({(d + 1) % 10 for d in (0, 20_000, 59_990)}, {1})       # a response comes 1 ms after a tick

    def test_the_transcript(self):
        text = expected("ch10_coordinated_omission")
        self.assertEqual(printed(omission.main), text)
        lines = text.splitlines()
        self.assertEqual(len(lines), 12)
        self.assertEqual(lines[1].split()[:3], ["as", "measured", "5800"])


def filled(xs, spacing, kept):
    """xs with L - i, L - 2i, ... after each L, for as long as `kept` says of the value to add."""
    out = list(xs)
    for x in xs:
        value = x - spacing
        while kept(value):
            out.append(value)
            value -= spacing
    return sorted(out)


def p99(xs):
    return statistics.quantiles(xs, n=100, method="inclusive")[98]


class WhereBackFillingStops(unittest.TestCase):
    def series(self):
        measured = [float(x) for x in sends(1, 10, 60_000, True, ((20_000, 1000), (40_000, 1000)))]
        to_zero = filled(measured, 10.0, lambda value: value > 0)                # the listing
        to_one_interval = filled(measured, 10.0, lambda value: value >= 10.0)    # the other rule
        return measured, to_zero, to_one_interval, omission.open_loop()

    def test_the_table_of_the_review(self):
        rows = [(len(xs), sum(x > 500 for x in xs), f"{p99(xs):.1f}") for xs in self.series()]
        self.assertEqual(rows, [(5800, 2, "1.0"), (6000, 102, "701.1"), (5998, 102, "701.3"), (6000, 112, "731.1")])

    def test_what_each_rule_adds(self):
        measured, to_zero, to_one_interval, _ = self.series()
        for xs, count, least in ((to_zero, 200, 1.0), (to_one_interval, 198, 11.0)):
            added = sorted(xs)
            for x in measured:
                added.remove(x)
            self.assertEqual((len(added), added[0], added[1], added[-1]), (count, least, least, 991.0))
        self.assertEqual(filled([1001.0], 10.0, lambda value: value > 0)[:2], [1.0, 11.0])
        self.assertEqual(filled([1001.0], 10.0, lambda value: value >= 10.0)[:2], [11.0, 21.0])

    def test_the_listing_goes_to_the_last_value_above_zero(self):
        measured, to_zero, to_one_interval, _ = self.series()
        self.assertEqual(omission.back_fill(measured), to_zero)
        self.assertEqual(omission.back_fill(omission.closed_loop()), to_zero)
        self.assertNotEqual(to_zero, to_one_interval)
        self.assertEqual(omission.back_fill([10.0, 20.0, 25.0]), [5.0, 10.0, 10.0, 15.0, 20.0, 25.0])
        self.assertIn("down to the last one above zero", omission.back_fill.__doc__)

    def test_the_two_rules_in_the_table_of_the_transcript(self):
        _, to_zero, to_one_interval, _ = self.series()
        row = expected("ch10_coordinated_omission").splitlines()[2].split()
        self.assertEqual(row[:4], ["back-filled", "6000", "1", f"{p99(to_zero):.0f}"])
        self.assertEqual((f"{p99(to_zero):.0f}", f"{p99(to_one_interval):.0f}"), ("701", "701"))


def tails(only_a, only_b):
    """The exact test for the items that one model alone has right, from binomial coefficients, as fractions:
    B ahead at least as far as it is, and a division at least as uneven either way."""
    n = only_a + only_b

    def at_least(k):
        return Fraction(sum(math.comb(n, i) for i in range(k, n + 1)), 2 ** n)
    one = at_least(only_b)
    return one, min(Fraction(1), 2 * min(one, at_least(only_a)))


def every_probability(right_a, right_b, n):
    """The two-sided probability for each number of items that both models can have right, from the fewest."""
    return [tails(right_a - both, right_b - both)[1] for both in range(max(0, right_a + right_b - n), min(right_a, right_b) + 1)]


def in_decimals(p, rounding=decimal.ROUND_HALF_EVEN):
    """A fraction as Transcript 17.1 prints a probability, by decimal arithmetic with 6000 digits: six decimals
    from 0.0001, and three digits with a power of ten below that. Since Version 70 the least probability of a
    range is rounded downward and the greatest upward, which the second argument names."""
    with decimal.localcontext() as context:
        context.prec = 6000
        context.rounding = rounding
        value = Decimal(p.numerator) / Decimal(p.denominator)
        if p == 0 or p >= Fraction(1, 10_000):
            return str(value.quantize(Decimal("0.000001")))
        digits, power = f"{value:.2e}".split("e")
        return f"{digits}e-{-int(power):02d}"


class ExactProbabilitiesForMatchedPairs(unittest.TestCase):
    def test_what_version_44_printed(self):
        one, two = tails(0, 23)
        self.assertEqual((one, two), (Fraction(1, 8_388_608), Fraction(1, 4_194_304)))
        self.assertEqual((f"{float(one):.6f}", f"{float(two):.6f}"), ("0.000000", "0.000000"))   # Version 44
        self.assertEqual(accuracy.two_sided(0, 23), (one, two))
        self.assertEqual((accuracy.shown(one), accuracy.shown(two)), ("1.19e-07", "2.38e-07"))

    def test_compare_for_the_example_of_the_review(self):
        a = [True] * 1977 + [False] * 23
        b = [True] * 2000
        lines = printed(accuracy.compare, a, b).splitlines()
        self.assertEqual(lines[-2:], ["  B ahead, one-sided  p = 1.19e-07", "  two-sided           p = 2.38e-07"])
        self.assertNotIn("0.000000", "\n".join(lines[-2:]))

    def test_fractions_from_the_function(self):
        for only_a in range(0, 31):
            for only_b in range(0, 31):
                got = accuracy.two_sided(only_a, only_b)
                self.assertEqual(got, tails(only_a, only_b), (only_a, only_b))
                self.assertTrue(all(isinstance(p, Fraction) for p in got))
        self.assertEqual(accuracy.two_sided(0, 0), (1, 1))                       # no item tells the models apart

    def test_the_two_ends_in_their_order(self):
        low, high = accuracy.limits(0, 1076, 1076)
        self.assertEqual((low, high), (Fraction(1, 2 ** 1075), Fraction(1, 2 ** 1075)))
        self.assertEqual((float(low), float(high)), (0.0, 0.0))                  # Version 44: (2 ** -1075, 0.0)
        self.assertEqual(accuracy.shown(low), "2.47e-324")
        for n in range(1, 15):
            for right_a in range(n + 1):
                for right_b in range(n + 1):
                    ps = every_probability(right_a, right_b, n)
                    low, high = accuracy.limits(right_a, right_b, n)
                    self.assertEqual((low, high), (min(ps), max(ps)), (right_a, right_b, n))
                    self.assertLessEqual(low, high)
        for n in (1075, 1076, 3000):
            for right_a, right_b in ((0, n), (n, 0), (1, n), (n // 2, n)):
                low, high = accuracy.limits(right_a, right_b, n)
                self.assertTrue(0 < low <= high <= 1, (right_a, right_b, n))

    def test_the_printed_form(self):
        rng = random.Random(45)
        ps = [Fraction(1, 2 ** k) for k in list(range(0, 80)) + [1000, 1074, 1075, 1076, 2999, 4000]]
        ps += [Fraction(0), Fraction(1), Fraction(1, 10_000), Fraction(9_999, 100_000_000), Fraction(1, 128),
               Fraction(9_995, 10 ** 8), Fraction(99_995, 10 ** 9), Fraction(1_235, 10 ** 7), Fraction(1_245, 10 ** 7),
               Fraction(1_005, 10 ** 9), Fraction(1_015, 10 ** 9), Fraction(1, 10 ** 5), Fraction(1, 10 ** 30)]
        ps += [Fraction(rng.randrange(1, 10 ** 6), 10 ** rng.randrange(6, 40)) for _ in range(400)]
        ps += [Fraction(rng.randrange(1, 2 ** 60), 2 ** rng.randrange(60, 1200)) for _ in range(400)]
        ps += [tails(a, b)[k] for a, b, k in ((246, 282, 0), (246, 282, 1), (0, 36, 1), (232, 278, 1), (5, 500, 0))]
        for p in ps:
            self.assertEqual(accuracy.shown(p), in_decimals(p), p)
        self.assertEqual([accuracy.shown(p) for p in ps[80:86]],
                         ["9.33e-302", "4.94e-324", "2.47e-324", "1.24e-324", "1.63e-903", "7.59e-1205"])
        self.assertEqual(accuracy.shown(Fraction(9_995, 10 ** 8)), "1.00e-04")   # a tie, to the even digit
        self.assertEqual(accuracy.shown(Fraction(1, 128)), "0.007812")           # and one at six decimals

    def test_the_transcript(self):
        text = expected("ch17_accuracy")
        lines = text.splitlines()
        self.assertEqual(len(lines), 43)
        rows = []
        for right_b in (1722, 1676, 1672):
            ps = every_probability(1722, right_b, 2000)
            # Until Version 69 both ends of a range were printed to the nearest digit. Since Version 70 the least
            # probability is rounded downward and the greatest upward, so that the printed range holds them all
            # (tests/test_review69.py). Two rows read 2.84e-14 to 0.066555 and 1.78e-15 to 0.046448.
            low = in_decimals(min(ps), decimal.ROUND_FLOOR)
            rows.append(f"  b = {right_b}: {low} to {in_decimals(max(ps), decimal.ROUND_CEILING)}")
        self.assertEqual(lines[28:32], ["and with B level or behind:"] + rows)
        self.assertEqual(rows, ["  b = 1722: 1.000000 to 1.000000", "  b = 1676: 2.84e-14 to 0.066556",
                                "  b = 1672: 1.77e-15 to 0.046448"])
        one, two = tails(0, 23)
        self.assertEqual(lines[32:34], ["23 items to B alone, none to A alone:",
                                      f"  p = {in_decimals(one)}, two-sided {in_decimals(two)}"])
        for line in lines:
            if " p = " in line:
                self.assertNotIn("p = 0.000000", line)
        self.assertEqual(printed(accuracy.main), text)


class WhichModelLeads(unittest.TestCase):
    A, N = 1722, 2000

    def ends(self, right_b):
        ps = every_probability(self.A, right_b, self.N)
        self.assertEqual(accuracy.limits(self.A, right_b, self.N), (min(ps), max(ps)))
        return min(ps), max(ps)

    def test_the_table_of_the_review(self):
        for right_b, most in ((1768, "0.046195"), (1676, "0.066555"), (1672, "0.046448")):
            self.assertEqual(f"{float(self.ends(right_b)[1]):.6f}", most)

    def test_a_lead_of_b(self):
        level = Fraction(1, 20)
        self.assertGreaterEqual(self.ends(self.A + 5)[0], level)                 # up to 5 ahead: never below
        self.assertLess(self.ends(self.A + 6)[0], level)
        self.assertGreaterEqual(self.ends(self.A + 45)[1], level)
        self.assertLess(self.ends(self.A + 46)[1], level)                        # from 46 ahead: always below
        first = next(d for d in range(1, 200) if accuracy.limits(self.A, self.A + d, self.N)[1] < level)
        self.assertEqual(first, 46)

    def test_b_behind(self):
        level = Fraction(1, 20)
        most = [f"{float(self.ends(self.A - d)[1]):.4f}" for d in (46, 47, 48, 49, 50)]
        self.assertEqual(most, ["0.0666", "0.0609", "0.0557", "0.0509", "0.0464"])
        first = next(d for d in range(1, 200) if accuracy.limits(self.A, self.A - d, self.N)[1] < level)
        self.assertEqual(first, 50)
        self.assertEqual(self.ends(self.A - 46)[0], self.ends(self.A + 46)[0])   # the least is the same either way
        self.assertGreaterEqual(self.ends(self.A - 5)[0], level)
        self.assertLess(self.ends(self.A - 6)[0], level)

    def test_why_the_two_directions_differ(self):
        # With the fewest items right for both, the same difference of 46 is spread over more items that differ.
        def differ(right_b):
            both = max(0, self.A + right_b - self.N)
            return self.A - both, right_b - both
        self.assertEqual((differ(1768), differ(1676)), ((232, 278), (324, 278)))
        self.assertEqual(sum(differ(1768)), 2 * self.N - self.A - 1768)          # wrong for A, and wrong for B
        self.assertEqual(sum(differ(1676)), 2 * self.N - self.A - 1676)

    def test_a_tie(self):
        self.assertEqual(self.ends(self.A), (1, 1))
        self.assertEqual(self.ends(self.A + 1)[0], 1)
        self.assertEqual(self.ends(self.A - 1)[0], 1)
        for lead in range(0, 60):
            least = min(Fraction(1), Fraction(2, 2 ** lead))                     # the smaller of 1 and 2 / 2 ** lead
            self.assertEqual(accuracy.limits(self.A, self.A + lead, self.N)[0], least)
            self.assertEqual(accuracy.limits(self.A, self.A - lead, self.N)[0], least)


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


def offsets_of_c(a_to_c, e, step):
    """The offsets of C from A on a grid from -10 to 10 ms: by their exchange, by way of B, and by all four."""
    grid = [k * step for k in range(int(-10 / step), int(10 / step) + 1)]
    of_b = [b for b in grid if all(in_order(stamps, b, e) for stamps in exchange.EXCHANGES)]
    alone = [c for c in grid if in_order(a_to_c, c, e)]
    through = sorted({c for b in of_b for c in grid if in_order(exchange.B_TO_C, c - b, e)})
    return alone, through, sorted(set(alone) & set(through))


def after_a(offsets, e):
    """The least and the greatest time from the event stamped 300 ms on A to the one stamped 299 ms on C."""
    return (299 - offsets[-1] - e) - (300 + e), (299 - offsets[0] + e) - (300 - e)


class ABoundForEveryStamp(unittest.TestCase):
    OTHER = exchange.A_TO_C[:3] + (62,)

    def test_the_numbers_of_the_review(self):
        want = {Fraction(0): (Fraction(1, 2), Fraction(2)), Fraction(1, 20): (Fraction(1, 5), Fraction(11, 5)),
                Fraction(1, 12): (Fraction(0), Fraction(7, 3)), Fraction(1, 10): (Fraction(-1, 10), Fraction(12, 5))}
        for e, times in want.items():
            step = e if e else Fraction(1, 4)
            both = offsets_of_c(exchange.A_TO_C, e, step)[2]
            self.assertEqual(after_a(both, e), times, e)
            span = exchange.routes(exchange.A_TO_C, e)[2]
            self.assertEqual(span, (both[0], both[-1]), e)
            self.assertEqual(exchange.gap(300, 299, span, e), times, e)

    def test_the_two_lines(self):
        for k in range(0, 121):
            e = Fraction(k, 400)
            direct, via_b, both = exchange.routes(exchange.A_TO_C, e)
            self.assertEqual(direct, (-3 - 2 * e, 1 + 2 * e))
            self.assertEqual(via_b, (-5 - 4 * e, Fraction(-3, 2) + 4 * e))
            self.assertEqual(exchange.gap(300, 299, both, e), (Fraction(1, 2) - 6 * e, 2 + 4 * e))
        below = Fraction(1, 12) - Fraction(1, 10 ** 9)
        self.assertGreater(exchange.gap(300, 299, exchange.routes(exchange.A_TO_C, below)[2], below)[0], 0)
        at = Fraction(1, 12)
        self.assertEqual(exchange.gap(300, 299, exchange.routes(exchange.A_TO_C, at)[2], at)[0], 0)

    def test_no_bound_is_the_exchange_of_version_44(self):
        self.assertEqual(exchange.routes(), exchange.routes(exchange.A_TO_C, 0))
        direct, via_b, both = exchange.routes()
        self.assertEqual((direct, via_b, both), ((-3, 1), (-5, Fraction(-3, 2)), (-3, Fraction(-3, 2))))
        self.assertEqual(exchange.routes(self.OTHER), ((0, 1), (-5, Fraction(-3, 2)), None))

    def test_a_bound_with_which_the_exchanges_agree(self):
        for e, step, want in ((Fraction(0), Fraction(1, 4), None), (Fraction(1, 5), Fraction(1, 20), None),
                              (Fraction(1, 4), Fraction(1, 4), (Fraction(-1, 2), Fraction(-1, 2))),
                              (Fraction(3, 10), Fraction(1, 20), (Fraction(-3, 5), Fraction(-3, 10)))):
            alone, through, both = offsets_of_c(self.OTHER, e, step)
            self.assertEqual((both[0], both[-1]) if both else None, want, e)
            self.assertEqual(exchange.routes(self.OTHER, e)[2], want, e)
            self.assertEqual(alone[0] - through[-1], Fraction(3, 2) - 6 * e, e)  # how far apart the two ways are
        first = next(k for k in range(0, 400) if exchange.routes(self.OTHER, Fraction(k, 400))[2])
        self.assertEqual(Fraction(first, 400), Fraction(1, 4))

    def test_the_transcript(self):
        text = expected("ch06_exchange")
        lines = text.splitlines()
        self.assertEqual(len(lines), 39)
        e = Fraction(1, 20)
        rows = []
        for name, offsets in zip(("direct", "via B", "both"), offsets_of_c(exchange.A_TO_C, e, e)):
            least, most = after_a(offsets, e)
            rows.append(f"{name:<12}{float(offsets[0]):>7.2f}{float(offsets[-1]):>7.2f}{float(least):>+8.2f}{float(most):>+8.2f}")
        self.assertEqual(lines[29:33], ["all stamps within 0.05 ms:"] + rows)
        self.assertEqual(rows[2], "both          -3.10  -1.30   +0.20   +2.20")
        self.assertEqual(printed(exchange.main), text)

    def test_the_description_of_figure_6_4(self):
        # Its values are those of the rows for no bound: either order by the exchange alone, C after A otherwise.
        alone, through, both = offsets_of_c(exchange.A_TO_C, Fraction(0), Fraction(1, 4))
        self.assertEqual([after_a(offsets, 0) for offsets in (alone, through, both)],
                         [(-2, 2), (Fraction(1, 2), 4), (Fraction(1, 2), 2)])
        self.assertEqual(expected("ch06_exchange").splitlines()[22:25],
                         ["direct        -3.00   1.00   -2.00   +2.00", "via B         -5.00  -1.50   +0.50   +4.00",
                          "both          -3.00  -1.50   +0.50   +2.00"])


def every_plan(var_b, var_w, startup, total):
    """The square of the standard uncertainty of the grand mean for every balanced design within the time."""
    return {(runs, iters): var_b / runs + var_w / (runs * iters)
            for runs in range(1, total + 1) for iters in range(1, total + 1) if runs * (startup + iters) <= total}


class ThePlanAndTheVariancesBehindIt(unittest.TestCase):
    def fitted(self):
        _, _, ms_w, var_b = launches.components(launches.benchmark(random.Random(13)))
        return var_b, ms_w

    def test_the_table_of_the_review(self):
        var_b, var_w = self.fitted()
        self.assertEqual((f"{var_b:.4f}", f"{var_w:.4f}"), ("4.8594", "2.7110"))
        made = launches.SIGMA_B ** 2, launches.SIGMA_W ** 2
        self.assertEqual(made, (16.0, 2.25))
        for (b, w), row in (((var_b, var_w), ("0.631", "0.582", (17, 3))), (made, ("0.980", "0.993", (19, 1)))):
            plans = every_plan(b, w, 16, 323)
            best = min(plans, key=plans.get)
            self.assertEqual((f"{math.sqrt(plans[19, 1]):.3f}", f"{math.sqrt(plans[17, 3]):.3f}", best), row)
            self.assertEqual(launches.budget(b, w, 16, 323)[:2], best)
        self.assertEqual((f"{var_w / var_b:.3f}", f"{made[1] / made[0]:.3f}"), ("0.558", "0.141"))

    def test_where_the_two_plans_change_places(self):
        b = Fraction(16)
        plans = every_plan(b, Fraction(3), 16, 323)                              # within / between = 3/16
        least = min(plans.values())
        self.assertEqual(sorted(plan for plan, u2 in plans.items() if u2 == least), [(17, 3), (19, 1)])
        for w, best in ((Fraction(299, 100), (19, 1)), (Fraction(301, 100), (17, 3))):
            plans = every_plan(b, w, 16, 323)
            self.assertEqual(min(plans, key=plans.get), best)
        self.assertLess(Fraction(9, 4) / 16, Fraction(3, 16))                    # the simulator is below it


class TwoMeansAndTheirUncertainties(unittest.TestCase):
    RUNS = ([9, 11], [11, 11, 13, 13], [13, 13, 13, 15, 15, 15])

    def components(self):
        """The variances between and within the launches, from the sums of squares, in fractions."""
        sizes = [len(run) for run in self.RUNS]
        total, k = sum(sizes), len(sizes)
        means = [Fraction(sum(run), len(run)) for run in self.RUNS]
        grand = Fraction(sum(map(sum, self.RUNS)), total)
        within = sum((x - m) ** 2 for run, m in zip(self.RUNS, means) for x in run) / (total - k)
        between = sum(n * (m - grand) ** 2 for n, m in zip(sizes, means)) / (k - 1)
        size = (total - Fraction(sum(n * n for n in sizes), total)) / (k - 1)
        return (between - within) / size, within, sizes, means

    def test_the_fitted_variances(self):
        var_b, var_w, sizes, means = self.components()
        self.assertEqual((var_b, var_w, sizes, means), (Fraction(36, 11), Fraction(4, 3), [2, 4, 6], [10, 12, 14]))
        _, _, ms_w, got_b = launches.unbalanced(self.RUNS)
        self.assertAlmostEqual(got_b, float(var_b), places=12)
        self.assertAlmostEqual(ms_w, float(var_w), places=12)

    def squares(self):
        var_b, var_w, sizes, means = self.components()
        by_launch = [Fraction(1, 3)] * 3
        by_iteration = [Fraction(n, 12) for n in sizes]
        out = []
        for weights in (by_launch, by_iteration):
            self.assertEqual(sum(weights), 1)
            mean = sum(w * m for w, m in zip(weights, means))
            u2 = sum(w * w * (var_b + var_w / n) for w, n in zip(weights, sizes))
            out.append((mean, u2))
        return out

    def test_the_table_of_the_review(self):
        (mean_1, u2_1), (mean_2, u2_2) = self.squares()
        self.assertEqual((mean_1, mean_2), (12, Fraction(38, 3)))
        self.assertEqual((u2_1, u2_2), (Fraction(1093, 891), Fraction(137, 99)))
        self.assertEqual((f"{float(mean_1):.2f}", f"{float(mean_2):.2f}"), ("12.00", "12.67"))
        self.assertEqual((f"{math.sqrt(u2_1):.3f}", f"{math.sqrt(u2_2):.3f}"), ("1.108", "1.176"))

    def test_by_simulation(self):
        var_b, var_w, sizes, _ = self.components()
        s_b, s_w = math.sqrt(var_b), math.sqrt(var_w)
        rng = random.Random(13)
        by_launch, by_iteration = [], []
        for _ in range(40_000):
            runs = []
            for n in sizes:
                level = rng.gauss(0.0, s_b)
                runs.append([level + rng.gauss(0.0, s_w) for _ in range(n)])
            by_launch.append(statistics.fmean(map(statistics.fmean, runs)))
            by_iteration.append(statistics.fmean(x for run in runs for x in run))
        (_, u2_1), (_, u2_2) = self.squares()
        self.assertAlmostEqual(statistics.pvariance(by_launch) / float(u2_1), 1.0, delta=0.03)
        self.assertAlmostEqual(statistics.pvariance(by_iteration) / float(u2_2), 1.0, delta=0.03)
        self.assertGreater(statistics.pvariance(by_iteration), statistics.pvariance(by_launch))

    def test_the_transcript_has_both_means(self):
        lines = expected("ch13_variance_components").splitlines()
        self.assertEqual(lines[15], "  mean of runs 12.00 ms, of iterations 12.67 ms")


class TheTwentySixthFigure(unittest.TestCase):
    def test_the_script(self):
        source = (ROOT / "figures" / "make_figures.py").read_text(encoding="utf-8")
        self.assertEqual(source.count("\n    save(fig, "), 32)
        for piece in ("def three_host_bound():", "c6x.routes(c6x.A_TO_C, e)[2]", "c6x.routes(other, e)",
                      'save(fig, "fig-06-5-three-host-bound.png")', "float(c17.two_sided(",
                      "    third_host()\n    three_host_bound()\n    stamps_at_fault()\n    aliasing()\n"):
            self.assertIn(piece, source)

    def test_what_it_shows(self):
        other = exchange.A_TO_C[:3] + (62,)

        def least(e):
            return exchange.gap(300, 299, exchange.routes(exchange.A_TO_C, e)[2], e)[0]

        def apart(e):
            direct, via_b, _ = exchange.routes(other, e)
            return direct[0] - via_b[1]
        bounds = [Fraction(k, 400) for k in range(121)]
        for value, start, zero in ((least, Fraction(1, 2), Fraction(1, 12)), (apart, Fraction(3, 2), Fraction(1, 4))):
            self.assertEqual([value(e) for e in bounds], [start - 6 * e for e in bounds])   # two parallel lines
            self.assertEqual(value(zero), 0)
            self.assertTrue(0 < zero < Fraction(305, 1000))                      # each circle is within the axis
        self.assertEqual((f"{1 / 12:.3f} ms", f"{1 / 4:.3f} ms"), ("0.083 ms", "0.250 ms"))   # the two labels
        self.assertTrue(-0.45 < float(apart(bounds[-1])) and float(least(0)) < 1.6)


if __name__ == "__main__":
    unittest.main()
