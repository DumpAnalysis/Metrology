"""Regression tests added after the review of Version 49.

The review found that the text of Chapter 23 had a request that is known only to have exceeded a limit recorded
with an infinite duration, which decides a later question that the trace leaves open; that most_within() of
Listing 23.2 compared its arguments before it read them as they print and before it checked them; that within()
gave a verdict for a limit that is not a number; and that weighted_quantile() returned None for a single value
of None. It asked what one recovered duration settles of a class known by its mean and an upper limit, which
Figure 23.2 now draws, and which batch labels of Chapter 17 are worth verifying.

Each test computes what it expects independently of the code under test: the side of a bounded duration from
completions that satisfy the bound, the bounds of a record from every way of completing it, the most durations
within a limit by trying each count from the largest down against the sums that it allows and by enumerating
vectors of durations, and the labels worth verifying from every pair of answers that can have changed places.
"""
import contextlib
import io
import itertools
import math
import random
import unittest
from fractions import Fraction
from pathlib import Path

from examples import ch17_accuracy as accuracy
from examples import ch23_mix as mix
from swmetro.stats import weighted_quantile

ROOT = Path(__file__).resolve().parent.parent
FAST = [100.0] * 90
SLOW = [100.0] * 4 + [300.0] * 5
REF = {"fast": range(90), "slow": range(10)}


def expected(name):
    return (ROOT / "expected" / f"{name}.txt").read_text(encoding="utf-8")


def printed(function, *args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        function(*args)
    return out.getvalue()


def record(last):
    return {"fast": FAST, "slow": SLOW + [last]}


def missing(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def completions(x, limit):
    """Durations that a record of one request allows, enough of them to show each side of `limit` that it can
    be on: the number itself, anything for a request of which nothing is known, and for a bound the bound if it
    is allowed, times just above it, the limit and times beyond the limit."""
    near = [limit - 1, limit, limit + Fraction(1, 1000), limit + 1, 10 ** 9, 0]
    if missing(x):
        return near
    if isinstance(x, tuple):
        sign, bound = x
        tries = near + [bound, bound + Fraction(1, 1000), bound + 1]
        return [t for t in tries if (t > bound if sign == ">" else t >= bound)]
    return [x]


def sides(x, limit):
    return {t <= limit for t in completions(x, limit)}


def every_share(rec, reference, limit):
    """Each share within the limit that a record allows at the class counts of the reference: every request is
    put on each side that its completions can be on, and a class without requests anywhere."""
    total = sum(len(xs) for xs in reference.values())
    choices = []
    for name, ref in reference.items():
        xs = rec.get(name) or []
        weight = Fraction(len(ref), total)
        if not xs:
            choices.append([Fraction(0), weight])
            continue
        sure = sum(1 for x in xs if sides(x, limit) == {True})
        open_ = sum(1 for x in xs if sides(x, limit) == {True, False})
        choices.append([weight * Fraction(sure + more, len(xs)) for more in range(open_ + 1)])
    return [sum(choice) for choice in itertools.product(*choices)]


def naive_p95(xs):
    """The smallest value with at least 95 % of the values at or below it."""
    xs = sorted(xs)
    return next(x for i, x in enumerate(xs) if 100 * (i + 1) >= 95 * len(xs))


class ABoundKeptWithTheRequest(unittest.TestCase):
    BOUND = (">", 200)

    def test_the_table_of_the_review(self):
        pct = lambda pair: tuple(round(100 * float(p), 6) for p in pair)
        rec = record(self.BOUND)
        self.assertEqual(pct(mix.within(rec, REF, 200)), (94, 94))
        self.assertEqual(pct(mix.within(rec, REF, 250)), (94, 95))
        self.assertEqual(mix.verdict(*mix.within(rec, REF, 200)), "missed")
        self.assertEqual(mix.verdict(*mix.within(rec, REF, 250)), "unresolved")
        rows = ((210.0, (95, 95), "met", 210.0, "111.1"), (700.0, (94, 94), "missed", 300.0, "116.0"),
                (math.inf, (94, 94), "missed", 300.0, "inf"))
        for last, at_250, verdict, p95, mean in rows:
            rec = record(last)
            self.assertEqual(pct(mix.within(rec, REF, 200)), (94, 94))
            self.assertEqual(pct(mix.within(rec, REF, 250)), at_250)
            self.assertEqual(mix.verdict(*mix.within(rec, REF, 250)), verdict)
            self.assertEqual(naive_p95(FAST + SLOW + [last]), p95)
            self.assertEqual(mix.p95(mix.weighted(rec)), p95)
            self.assertEqual(f"{float(mix.mean(mix.weighted(rec))):.1f}", mean)
            self.assertEqual(f"{sum(FAST + SLOW + [last]) / 100:.1f}", mean)

    def test_infinity_decides_what_the_trace_leaves_open(self):
        open_ = mix.within(record(self.BOUND), REF, 250)
        closed = mix.within(record(math.inf), REF, 250)
        self.assertEqual((mix.verdict(*open_), mix.verdict(*closed)), ("unresolved", "missed"))
        self.assertTrue(open_[0] == closed[0] == closed[1] < open_[1])
        self.assertIs(mix.side(math.inf, 10 ** 12), False)                       # beyond every limit, as it should be
        self.assertIs(mix.side(-math.inf, -10 ** 12), True)

    def test_every_completion_above_the_bound(self):
        shares, p95s = set(), set()
        for k in range(200 * 8 + 1, 900 * 8):                                    # above 200 ms, in eighths
            x = k / 8
            rec = record(x)
            low, high = mix.within(rec, REF, 250)
            self.assertEqual(low, high)
            shares.add(low)
            p95s.add(naive_p95(FAST + SLOW + [x]))
            self.assertEqual(naive_p95(FAST + SLOW + [x]), min(x, 300.0))        # the smaller of the two
            self.assertEqual(mix.within(rec, REF, 200), (Fraction(94, 100),) * 2)
        self.assertEqual(sorted(shares), list(mix.within(record(self.BOUND), REF, 250)))
        self.assertEqual((min(p95s), max(p95s)), (200.125, 300.0))               # above 200 and up to 300 ms
        self.assertTrue(all(200 < p <= 300 for p in p95s))

    def test_the_sign_of_the_bound(self):
        table = {(">", 199): False, (">", 200): False, (">", 201): None,
                 (">=", 199): False, (">=", 200): None, (">=", 201): None}
        for (sign, limit), want in table.items():
            self.assertIs(mix.side((sign, 200), limit), want, (sign, limit))
            both = sides((sign, 200), limit)
            self.assertEqual(want, None if both == {True, False} else both.pop(), (sign, limit))
        # A timeout of Chapter 10 is known to be at least the time that the client waited.
        rec = {"all": [100.0] * 94 + [(">=", 200)] * 6}
        ref = {"all": range(100)}
        self.assertEqual(mix.within(rec, ref, 200), (Fraction(94, 100), Fraction(1)))
        self.assertEqual(mix.within(rec, ref, 199.5), (Fraction(94, 100),) * 2)
        rec = {"all": [100.0] * 94 + [(">", 200)] * 6}
        self.assertEqual(mix.within(rec, ref, 200), (Fraction(94, 100),) * 2)
        self.assertEqual(mix.within(rec, ref, 200.5), (Fraction(94, 100), Fraction(1)))

    def test_side_against_completions(self):
        rng = random.Random(4923)
        kinds = (lambda: rng.choice([50, 150.0, 200, 250.5, 400]), lambda: None, lambda: math.nan,
                 lambda: (">", rng.choice([100, 200, 250, 300])), lambda: (">=", rng.choice([100, 200, 250, 300])))
        for _ in range(2000):
            x, limit = rng.choice(kinds)(), rng.choice([100, 150, 200, 250, 300, 350])
            both = sides(x, limit)
            want = None if both == {True, False} else next(iter(both))
            self.assertIs(mix.side(x, limit), want, (x, limit))

    def test_a_bound_has_one_of_two_signs(self):
        for bad in (("<", 200), ("<=", 200), ("=", 200), (200, ">"), ("> ", 200), (">", 200, 0), (">",), ()):
            with self.assertRaises(ValueError):
                mix.side(bad, 250)
        with self.assertRaises(ValueError) as caught:
            mix.side(("<", 200), 250)
        self.assertEqual(str(caught.exception), "a bound is > or >=")

    def test_random_records_against_every_completion(self):
        rng = random.Random(2349)
        values = [50.0, 150.0, 250.0, 200, None, math.nan, (">", 100), (">", 200), (">=", 200), (">=", 250), (">", 300)]
        for _ in range(500):
            names = ["a", "b", "c"][:rng.randint(1, 3)]
            ref = {name: range(rng.randint(0, 5)) for name in names}
            if not any(len(r) for r in ref.values()):
                ref[names[0]] = range(1)
            rec = {name: [rng.choice(values) for _ in range(rng.randint(0, 5))] for name in names if rng.random() < 0.8}
            for limit in (200, 250):
                shares = every_share(rec, ref, limit)
                self.assertEqual(mix.within(rec, ref, limit), (min(shares), max(shares)), (rec, ref, limit))

    def test_records_without_bounds_are_as_before(self):
        rng = random.Random(2350)
        for _ in range(200):
            rec = {name: [rng.choice([rng.uniform(0, 400), None, math.nan]) for _ in range(rng.randint(0, 20))] for name in "ab"}
            ref = {name: range(rng.randint(1, 9)) for name in "ab"}
            low = high = Fraction(0)
            for name in "ab":
                xs = rec[name] or [None]
                sure = sum(1 for x in xs if not missing(x) and x <= 200)
                open_ = sum(1 for x in xs if missing(x))
                low += Fraction(sure * len(ref[name]), len(xs) * sum(len(r) for r in ref.values()))
                high += Fraction((sure + open_) * len(ref[name]), len(xs) * sum(len(r) for r in ref.values()))
            self.assertEqual(mix.within(rec, ref, 200), (low, high))

    def test_the_transcript(self):
        lines = expected("ch23_mix").splitlines()
        self.assertEqual(len(lines), 108)                                        # 96 in Version 50, 100 in 51, 103 until 57
        at = lines.index("90 fast requests and 10 slow, one")
        rows = ["90 fast requests and 10 slow, one"]
        cases = (("of the slow without a duration:", None, (200,)), ("and known only to exceed 200 ms:", (">", 200), (200, 250)),
                 ("and stored as infinity:", math.inf, (250,)))
        for text, last, limits in cases:
            rows.append(text)
            for limit in limits:
                shares = every_share(record(last), REF, limit)
                low, high = min(shares), max(shares)
                verdict = "met" if low >= Fraction(95, 100) else "missed" if high < Fraction(95, 100) else "unresolved"
                rows += [f"  within {limit} ms: {100 * float(low):.0f} % to {100 * float(high):.0f} %",
                         f"  p95 at most {limit} ms: {verdict}"]
        self.assertEqual(lines[at:at + 12], rows)
        self.assertEqual(rows[5:], ["  within 200 ms: 94 % to 94 %", "  p95 at most 200 ms: missed",
                                    "  within 250 ms: 94 % to 95 %", "  p95 at most 250 ms: unresolved",
                                    "and stored as infinity:", "  within 250 ms: 94 % to 94 %", "  p95 at most 250 ms: missed"])
        self.assertEqual(lines[at + 12], "18 fast requests and 2 slow without")   # five lines of Version 58
        self.assertEqual(lines[at + 17], "after, with a new class, bulk:")
        self.assertEqual(printed(mix.one_unknown).splitlines(), rows)
        self.assertEqual(printed(mix.main), expected("ch23_mix"))


class NoLimitNoVerdict(unittest.TestCase):
    def test_a_limit_that_is_not_a_number(self):
        rec = {"all": [100.0] * 5}
        # Version 49 compared: nothing is at or below a limit that is not a number, 0 % to 0 %, "missed".
        self.assertEqual(sum(x <= math.nan for x in rec["all"]), 0)
        for limit in (math.nan, float("nan"), None):
            with self.assertRaises(ValueError) as caught:
                mix.within(rec, rec, limit)
            self.assertEqual(str(caught.exception), "no limit")
        self.assertEqual(mix.within(rec, rec, 100), (Fraction(1),) * 2)
        self.assertEqual(mix.within(rec, rec, math.inf), (Fraction(1),) * 2)     # an infinite limit is a limit
        self.assertEqual(mix.within(rec, rec, 99.9), (Fraction(0),) * 2)


def allowed(k, n, total, limit, cap):
    """Whether k of n durations, none below zero and none above the cap, can be at or below the limit and
    the others above it, with this sum: the sums that such durations can have are compared with it."""
    rest = n - k
    if rest == 0:
        return 0 <= total <= n * (limit if cap is None else min(limit, cap))
    if cap is not None and cap <= limit:
        return False
    if not total > rest * limit:
        return False
    return cap is None or total <= k * limit + rest * cap


def most_by_trial(n, total, limit, cap):
    """The largest count that is allowed, each tried from n down, or None if no count is."""
    return next((k for k in range(n, -1, -1) if allowed(k, n, total, limit, cap)), None)


def attaining(k, n, total, limit, cap):
    """Durations that show an allowed count: k at or below the limit, the others above it."""
    rest = n - k
    if rest == 0:
        return [Fraction(total, n)] * n
    low = min(Fraction(limit), Fraction(total - rest * limit, 2 * k)) if k else Fraction(0)
    high = Fraction(total - k * low, rest)
    if cap is not None and high > cap:
        high = Fraction(cap)
        low = Fraction(total - rest * high, k)
    return [low] * k + [high] * rest


def in_halves(k, n, total, limit, cap):
    """Whether durations in halves from 0 to 6 can show that count: k of them up to the limit, the others from
    half a unit above it, and every sum in halves between the least and the greatest."""
    rest = n - k
    top = 6 if cap is None else min(cap, 6)
    if rest and top <= limit:
        return False
    return rest * (limit + Fraction(1, 2)) <= total <= k * min(limit, top) + rest * top


class MeanAndCapBeforeCounting(unittest.TestCase):
    def test_the_table_of_the_review(self):
        with self.assertRaises(ValueError) as caught:
            mix.most_within(900, 450, 500, 400)
        self.assertEqual(str(caught.exception), "a mean above its cap")
        self.assertEqual(mix.most_within(20, 0.1, Fraction(1, 10)), 20)
        self.assertEqual(mix.most_within(20, 0.1, Fraction(1, 10), 0.1), 20)
        with self.assertRaises(ValueError) as caught:
            mix.most_within(0, 450, 200)
        self.assertEqual(str(caught.exception), "need a whole n >= 1")
        # The float 0.1 is a binary number above one tenth, and it is read as the 0.1 that it prints.
        self.assertTrue(0.1 > Fraction(1, 10))
        self.assertEqual(Fraction(str(0.1)), Fraction(1, 10))
        self.assertIsNone(most_by_trial(900, 900 * 450, 500, 400))               # no durations fit that record
        self.assertEqual(most_by_trial(20, 2, Fraction(1, 10), Fraction(1, 10)), 20)

    def test_the_printed_maxima_are_what_they_were(self):
        for cap, most in ((None, 899), (800, 525), (810, 531), (Fraction(33200, 41), 531)):
            self.assertEqual(mix.most_within(900, 450, 200, cap), most)
            self.assertEqual(most_by_trial(900, 900 * 450, 200, cap), most)
        self.assertEqual(mix.most_within(900, 450.0, 200.0, 810.0), 531)
        self.assertEqual(mix.most_within(900, "450", "200", "809.75"), 530)

    def test_a_count_of_durations(self):
        for n in (0, -1, 2.5, None, "many", math.nan, Fraction(7, 2)):
            with self.assertRaises(ValueError):
                mix.most_within(n, 450, 200, 810)
        self.assertEqual(mix.most_within(1, 450, 200), 0)
        self.assertEqual(mix.most_within(1, 450, 200, 810), 0)
        self.assertEqual(mix.most_within(1, 200, 200, 810), 1)
        self.assertEqual(mix.most_within(3.0, 450, 200), 2)
        self.assertIs(type(mix.most_within(900, 450, 200, 810)), int)

    def test_times_below_zero_and_times_that_are_no_numbers(self):
        for mean, limit in ((-1, 200), (450, -1), (-0.5, -0.5)):
            with self.assertRaises(ValueError) as caught:
                mix.most_within(10, mean, limit)
            self.assertEqual(str(caught.exception), "a time below zero")
        for args in ((10, math.nan, 200), (10, 450, math.nan), (10, 450, 200, math.nan), (10, 450, 200, math.inf),
                     (10, None, 200), (10, 450, None)):
            with self.assertRaises(ValueError):
                mix.most_within(*args)
        self.assertEqual(mix.most_within(10, 0, 0), 10)
        self.assertEqual(mix.most_within(10, 0, 0, 0), 10)

    def test_the_cap_is_checked_whatever_the_limit(self):
        for n, mean, limit, cap in ((900, 450, 500, 400), (5, 3, 3, 2), (5, 3, 9, 2.999), (1, 0.3, 0.3, 0.29)):
            self.assertIsNone(most_by_trial(n, n * Fraction(str(mean)), Fraction(str(limit)), Fraction(str(cap))))
            with self.assertRaises(ValueError) as caught:
                mix.most_within(n, mean, limit, cap)
            self.assertEqual(str(caught.exception), "a mean above its cap")
        self.assertEqual(mix.most_within(5, 3, 3, 3), 5)                         # all at the mean, which is the cap

    def test_decimals_are_compared_as_decimals(self):
        cases = 0
        tenths = [k / 10 for k in range(0, 13)]
        for n, mean, limit, cap in itertools.product((1, 2, 3, 7, 20), tenths, tenths[::2], [None] + tenths[::3]):
            total = n * Fraction(str(mean))
            want = most_by_trial(n, total, Fraction(str(limit)), None if cap is None else Fraction(str(cap)))
            for as_given in ((mean, limit, cap), (Fraction(str(mean)), limit, cap), (mean, Fraction(str(limit)), cap),
                             (str(mean), str(limit), cap if cap is None else str(cap))):
                if cap is not None and Fraction(str(cap)) < Fraction(str(mean)):
                    with self.assertRaises(ValueError):
                        mix.most_within(n, *as_given)
                else:
                    self.assertEqual(mix.most_within(n, *as_given), want, (n, as_given))
            cases += 1
        self.assertEqual(cases, 5 * 13 * 7 * 6)

    def test_small_cases_by_trial_with_durations_that_attain_them(self):
        cases = 0
        for n in range(1, 9):
            for mean, limit, cap in itertools.product((0, 1, 2, 3, 5, 8), (0, 2, 4), (None, 3, 5, 6, 9)):
                if cap is not None and cap < mean:
                    continue
                k = most_by_trial(n, n * mean, limit, cap)
                self.assertEqual(mix.most_within(n, mean, limit, cap), k, (n, mean, limit, cap))
                xs = attaining(k, n, n * mean, limit, cap)
                self.assertEqual((len(xs), sum(xs), sum(x <= limit for x in xs)), (n, n * mean, k))
                self.assertTrue(all(x >= 0 and (cap is None or x <= cap) for x in xs), (n, mean, limit, cap, xs))
                self.assertFalse(k < n and allowed(k + 1, n, n * mean, limit, cap))
                cases += 1
        self.assertEqual(cases, 8 * 3 * sum(1 for m in (0, 1, 2, 3, 5, 8) for c in (None, 3, 5, 6, 9) if c is None or c >= m))

    def test_vectors_of_durations_in_halves(self):
        grid = [Fraction(k, 2) for k in range(13)]                               # 0 to 6 in halves
        best = {}
        for n in range(1, 5):
            for xs in itertools.combinations_with_replacement(grid, n):
                for limit, cap in itertools.product((1, 2, 3), (None, 4, 6)):
                    if cap is not None and xs[-1] > cap:
                        continue
                    key = (n, sum(xs), limit, cap)
                    best[key] = max(best.get(key, 0), sum(x <= limit for x in xs))
        both = set()
        for (n, total, limit, cap), count in best.items():
            most = mix.most_within(n, Fraction(total, n), limit, cap)
            self.assertLessEqual(count, most, (n, total, limit, cap))            # no vector does better
            shown = in_halves(most, n, total, limit, cap)                        # and where halves up to 6 can show
            self.assertEqual(count == most, shown, (n, total, limit, cap))       # the most, the enumeration has it
            self.assertEqual(count, max(k for k in range(n + 1) if in_halves(k, n, total, limit, cap)))
            both.add(shown)
        self.assertEqual(both, {True, False})
        # Sums in halves from 0 to 6 n without a cap and with one of 6, and to 4 n with a cap of 4; three limits.
        self.assertEqual(len(best), 3 * sum((12 * n + 1) * 2 + (8 * n + 1) for n in range(1, 5)))


def most_with_one(x):
    """The most of the 900 slow durations within 200 ms, with a mean of 450 ms and none above 810 ms, when one
    of them is known to be x: each count of the other 899 is tried against the sums that it allows."""
    return int(x <= 200) + most_by_trial(899, 900 * 450 - x, 200, 810)


class OneRecoveredDuration(unittest.TestCase):
    FAST = Fraction(891, 1000)

    def share(self, k):
        return self.FAST + Fraction(1, 10) * Fraction(k, 900)

    def test_the_allowance(self):
        self.assertEqual(531 * 200 + 369 * 810, 405090)                          # 90 ms more than the mean allows
        self.assertEqual(100 + 530 * 200 + 369 * 810, 404990)                    # with one request at 100 ms: too little
        self.assertEqual(110 + 530 * 200 + 369 * 810, 405000)
        self.assertEqual(531 * 200 + 368 * 810 + 700, 404980)
        self.assertEqual(531 * 200 + 368 * 810 + 720, 405000)

    def test_the_two_lines_of_the_review(self):
        for x in (0, 50, 100, 109, 110, 150, 200, 201, 400, 700, 719, 720, 800, 810):
            rest = Fraction(405000 - x, 899)
            self.assertEqual(int(x <= 200) + mix.most_within(899, rest, 200, 810), most_with_one(x), x)

    def test_where_the_objective_is_settled(self):
        runs, before = [], None
        for k in range(810 * 4 + 1):                                             # 0 to 810 ms in quarters
            x = Fraction(k, 4)
            most = most_with_one(x)
            self.assertEqual(int(x <= 200) + mix.most_within(899, Fraction(405000 - x, 899), 200, 810), most)
            if most != before:
                runs.append((x, most))
                before = most
        self.assertEqual(runs, [(0, 530), (110, 531), (Fraction(801, 4), 530), (720, 531)])
        self.assertEqual([most_with_one(x) for x in (110, 200, 720, 810)], [531] * 4)   # these ends belong to 531
        self.assertEqual([most_with_one(x) for x in (Fraction(10999, 100), Fraction(20001, 100), Fraction(71999, 100))], [530] * 3)

    def test_the_shares_and_the_durations_that_attain_them(self):
        self.assertEqual(f"{100 * float(self.share(530)):.4f}", "94.9889")
        self.assertEqual(f"{100 * float(self.share(531)):.4f}", "95.0000")
        self.assertEqual(mix.verdict(self.FAST, self.share(530)), "missed")
        self.assertEqual(mix.verdict(self.FAST, self.share(531)), "unresolved")
        last = Fraction(29910, 37)
        self.assertEqual(100 + 529 * 200 + 370 * last, 405000)
        self.assertTrue(200 < last <= 810)
        self.assertEqual(f"{float(last):.2f}", "808.38")
        self.assertEqual(f"{float(Fraction(405000 - 110, 899)):.2f}", "450.38")   # the others equal: one within

    def test_the_transcript(self):
        lines = expected("ch23_mix").splitlines()
        at = lines.index("  none above 810 ms, one recovered")
        rows = []
        for x in (100, 110, 700):
            top = self.share(most_with_one(x))
            verdict = "missed" if top < Fraction(95, 100) else "unresolved"
            rows.append(f"    at {x} ms: to {100 * float(top):.2f} %, {verdict}")
        self.assertEqual(lines[at + 1:at + 4], rows)
        self.assertEqual(rows, ["    at 100 ms: to 94.99 %, missed", "    at 110 ms: to 95.00 %, unresolved",
                                "    at 700 ms: to 94.99 %, missed"])
        self.assertEqual(lines[at - 1], "    to 95.00 %, unresolved")
        self.assertEqual(lines[at + 11], "  98 % known within, 2 % missing: met")   # after four lines of Version 51 and three of 52


class TheThirtiethFigure(unittest.TestCase):
    def test_the_script(self):
        source = (ROOT / "figures" / "make_figures.py").read_text(encoding="utf-8")
        self.assertEqual(source.count("\n    save(fig, "), 32)
        for piece in ("def one_recovered():", 'save(fig, "fig-23-2-one-recovered.png")',
                      "return c23.with_one(x, 450)", "ends = [0, 110, 200, 720, 810]",
                      "[530, 531, 530, 531]", "    guard_values()\n    one_mix()\n    one_recovered()\n"):
            self.assertIn(piece, source)

    def test_what_it_draws(self):
        ends = [0, 110, 200, 720, 810]
        for (a, b), want in zip(zip(ends, ends[1:]), (530, 531, 530, 531)):
            inside = {most_with_one(Fraction(k, 4)) for k in range(4 * a + 1, 4 * b)}
            self.assertEqual(inside, {want})
        # A filled circle where the end has the count of its line: 0 on the first, both ends on each line of 531.
        self.assertEqual([most_with_one(x) for x in ends], [530, 531, 531, 531, 531])


class NoneHasNoPlaceEither(unittest.TestCase):
    def test_one_value_of_none(self):
        # With a single pair there was nothing to compare it with, and Version 49 returned None.
        for pairs in ([(None, 1)], [(None, 3), (None, 2)][:1], [(None, Fraction(1, 2))]):
            with self.assertRaises(ValueError) as caught:
                weighted_quantile(pairs, 0.5)
            self.assertEqual(str(caught.exception), "a value is no number")

    def test_none_among_numbers(self):
        values = [5.0, 1.0, 4.0, 2.0, 3.0]
        for place in range(6):
            for nothing in (None, math.nan):
                xs = values[:place] + [nothing] + values[place:]
                for q in (0, 0.5, 0.95, 1):
                    with self.assertRaises(ValueError) as caught:
                        weighted_quantile([(x, 1) for x in xs], q)
                    self.assertEqual(str(caught.exception), "a value is no number")

    def test_numbers_are_ordered_as_before(self):
        rng = random.Random(4924)
        for _ in range(300):
            pairs = [(rng.choice([rng.uniform(-5, 5), rng.randint(-3, 3), math.inf, -math.inf]), rng.randint(0, 4))
                     for _ in range(rng.randint(1, 12))]
            if not any(w for _, w in pairs):
                pairs.append((0.5, 1))
            q = Fraction(rng.randint(0, 20), 20)
            ordered = sorted(p for p in pairs if p[1])                           # since Version 51 a value of no weight
            total = sum(w for _, w in ordered)                                   # is not the quantile of rank zero
            want = next(x for i, (x, _) in enumerate(ordered)
                        if sum(w for _, w in ordered[:i + 1]) * q.denominator >= q.numerator * total)
            self.assertEqual(weighted_quantile(pairs, q), want, (pairs, q))

    def test_the_percentile_of_the_listing(self):
        for last in (None, math.nan, (">", 200), (">=", 1000)):
            with self.assertRaises(ValueError) as caught:
                mix.p95(mix.weighted(record(last)))
            self.assertEqual(str(caught.exception), "a value is no number")

    def test_whatever_is_no_number(self):
        # Values of one kind compare with each other, and a record of bounds alone had a bound for a percentile.
        self.assertEqual(sorted([(">", 300), (">", 200)])[0], (">", 200))
        for pairs in ([((">", 200), 1)], [((">", 200), 1), ((">", 300), 1)], [("slow", 1), ("fast", 2)], [([1], 1)],
                      [(1.0, 1), ("2", 1)], [({1}, 1)]):
            with self.assertRaises(ValueError) as caught:
                weighted_quantile(pairs, 0.5)
            self.assertEqual(str(caught.exception), "a value is no number")
        from decimal import Decimal
        self.assertEqual(weighted_quantile([(Decimal("1.5"), 1), (Fraction(1, 3), 1), (2, 1), (True, 1)], 0.5), True)
        self.assertEqual(weighted_quantile([(Decimal("1.5"), 1), (Fraction(7, 3), 1), (2, 1)], 0.5), 2)


def moved(b1, b2, i_right, j_right):
    """The right answers of B in the two batches if an answer of the first batch and one of the second carry
    each other's label: put back, a right answer and a wrong one move the counts by one each."""
    if i_right == j_right:
        return b1, b2
    return (b1 - 1, b2 + 1) if i_right else (b1 + 1, b2 - 1)


class LabelsWorthVerifying(unittest.TestCase):
    KNOWN = (1560, 132, 163, 95)
    LEVEL = Fraction(1, 20)

    def ps(self, b1, b2):
        return accuracy.partly(self.KNOWN, 1722, 1758, 2000, [(25, 23, b1), (25, 7, b2)])

    def adverse(self, b1, b2):
        """Every pair of an answer of B in the first batch and one in the second that, had they changed
        places, allows a probability on the other side of the level; answers are numbered, the right ones
        first, in each batch."""
        below = max(self.ps(b1, b2)) < self.LEVEL
        pairs = set()
        for i, j in itertools.product(range(25), repeat=2):
            ps = self.ps(*moved(b1, b2, i < b1, j < b2))
            if (max(ps) > self.LEVEL) if below else (min(ps) < self.LEVEL):
                pairs.add((i, j))
        return pairs

    def test_the_two_groups_of_the_review(self):
        first = self.adverse(23, 12)
        self.assertEqual(first, set(itertools.product(range(23), range(12, 25))))        # 23 right in 1, 13 wrong in 2
        self.assertEqual(len(first), 23 * 13)
        self.assertEqual(len(first), 299)
        second = self.adverse(13, 22)
        self.assertEqual(second, set(itertools.product(range(13, 25), range(22))))       # 12 wrong in 1, 22 right in 2
        self.assertEqual(len(second), 12 * 22)
        self.assertEqual(len(second), 264)

    def test_the_smallest_audit(self):
        rng = random.Random(1749)
        for (b1, b2), in_first, need in (((23, 12), False, 13), ((13, 22), True, 12)):
            pairs = self.adverse(b1, b2)
            ones, twos = {i for i, _ in pairs}, {j for _, j in pairs}
            self.assertEqual(min(len(ones), len(twos)), need)
            small = ones if in_first else twos
            self.assertEqual(len(small), need)
            # Verified in its batch, an answer has not changed places: the smaller group excludes every pair.
            self.assertFalse([p for p in pairs if p[0 if in_first else 1] not in small])
            # With fewer labels verified, whichever they are, an answer of each group is left, and their pair.
            records = [("one", i) for i in range(25)] + [("two", j) for j in range(25)]
            for _ in range(500):
                checked = set(rng.sample(records, need - 1))
                left = [(i, j) for i, j in pairs if ("one", i) not in checked and ("two", j) not in checked]
                self.assertTrue(left)
            for size in range(need, 51):                                         # and enough labels need not be the right ones
                checked = set(rng.sample(records, size))
                left = [(i, j) for i, j in pairs if ("one", i) not in checked and ("two", j) not in checked]
                covers = ones <= {i for name, i in checked if name == "one"} or twos <= {j for name, j in checked if name == "two"}
                self.assertEqual(not left, covers)


if __name__ == "__main__":
    unittest.main()
