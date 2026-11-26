"""Regression tests added after the review of Version 62.

The review found that both() of Listing 23.2 kept the count of an observed class under the identity of the object
alone, so that the cursor of that object, named in the reference, was read at its end and counted as empty: a
record that misses its objective was reported as met; that Listing 18.1 took a label by its truth value, a missing
one as negative and a NaN as positive, fitted a NaN among the logits, and returned infinity from a search without
an upper end; that rect_interval() formed its limits in floats and left out a part of the width, also at a
coverage of 1; that prob_below() of an infinite limit took the probability of a capped distance; that expanded()
and report() of a result without uncertainty looked for a factor; that a level next to 1 raised an error of the
normal quantile or lost digits; that a small factor was printed as 0.00 beside a width above zero, and a zero
with a sign in one form and without it in the other; that the half-width of a Wilson interval was printed with the
sign reserved for expanded uncertainty; that the two ends of an interval for a temperature were printed to the
nearest, inside the interval; and that p95_all() and two scores read a stream and then asked for its length. It
proposed one more outcome for the exercise on missing pairs.

Each test computes what it expects independently of the code under test: shares and percentiles by counting and
sorting, temperatures from closed forms and from binomial tails summed here, Student's tail from its density, the
normal quantile next to 1 from the asymptotic series of its tail in logarithms, and printed forms by the decimal
module.
"""
import contextlib
import datetime
import io
import itertools
import math
import random
import unittest
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from statistics import NormalDist

from examples import ch10_censoring as censoring
from examples import ch18_calibration as cal
from examples import ch23_mix as mix
from swmetro import Result, core, dist, rect_interval, u_rect, u_resolution, u_tri
from swmetro.propagate import lpu

ROOT = Path(__file__).resolve().parent.parent
D = Decimal
KINDS = (int, float, Decimal, Fraction)


def expected(name):
    return (ROOT / "expected" / f"{name}.txt").read_text(encoding="utf-8")


def printed(function, *args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        function(*args)
    return out.getvalue()


def far_tail(exponent, dof):
    """Upper tail of Student's distribution at t = 10 ** exponent, far out: the density there is
    c * (t * t / dof) ** (-(dof + 1) / 2), and its integral from t on is c * sqrt(dof) * (t * t / dof) ** (-dof / 2)
    / dof. The power is taken in decimals of 50 digits (added with Version 64)."""
    c = math.gamma((dof + 1) / 2) / (math.sqrt(math.pi * dof) * math.gamma(dof / 2))
    with localcontext() as ctx:
        ctx.prec = 50
        power = ((D(10) ** (2 * exponent)) / D(dof)).ln() * (-D(dof) / 2)
        return float(D(c) * D(dof).sqrt() / D(dof) * power.exp())


def p95(xs):
    """Inverse empirical CDF: the least value with at least 95 % of the values at or below it."""
    xs = sorted(xs)
    for x in xs:
        if 100 * sum(y <= x for y in xs) >= 95 * len(xs):
            return x


class Capture:
    """One object around one cursor: __iter__() returns the cursor that it was given."""
    def __init__(self, cursor):
        self.cursor = cursor

    def __iter__(self):
        return self.cursor


class Once:
    """A stream that opens once and refuses a second request."""
    def __init__(self, values):
        self.values, self.opened = values, 0

    def __iter__(self):
        self.opened += 1
        if self.opened > 1:
            raise RuntimeError("opened twice")
        return iter(self.values)


class Fresh:
    """An object without a length that gives a new cursor at each request."""
    def __init__(self, n):
        self.n = n

    def __iter__(self):
        return iter(range(self.n))


class ASlowStreamInTheRecordAndInTheReference(unittest.TestCase):
    FAST = [100] * 18

    def slow(self, record, reference):
        cursor = iter([500, 500])
        wrapper = Capture(cursor)
        made = {"list": [500, 500], "cursor": cursor, "wrapper": wrapper, "other wrapper": Capture(cursor),
                "range": range(2)}
        if record == reference == "list":
            return made["list"], made["list"]
        return made[record], made[reference]

    def test_the_case_of_the_review(self):
        slow = iter([500, 500])
        record = {"fast": [100] * 18, "slow": Capture(slow)}
        ref = {"fast": range(18), "slow": slow}
        bounds = mix.within(record, ref, 200)
        self.assertEqual((bounds, mix.verdict(*bounds)), ((Fraction(9, 10), Fraction(9, 10)), "missed"))   # Version 62: met
        self.assertEqual((sum(x <= 200 for x in self.FAST + [500, 500]), p95(self.FAST + [500, 500])), (18, 500))

    def test_every_way_of_naming_the_same_two_requests(self):
        ways = (("list", "range"), ("list", "list"), ("cursor", "range"), ("cursor", "cursor"), ("cursor", "wrapper"),
                ("wrapper", "range"), ("wrapper", "wrapper"), ("wrapper", "cursor"), ("wrapper", "other wrapper"))
        checked = 0
        for (record, reference), rec_first, ref_first in itertools.product(ways, (True, False), (True, False)):
            for call in ("within", "weighted"):
                rec_slow, ref_slow = self.slow(record, reference)
                rec = {"fast": list(self.FAST), "slow": rec_slow}
                ref = {"fast": range(18), "slow": ref_slow}
                if not rec_first:
                    rec = dict(reversed(rec.items()))
                if not ref_first:
                    ref = dict(reversed(ref.items()))
                if call == "within":
                    bounds = mix.within(rec, ref, 200)
                    self.assertEqual((bounds, mix.verdict(*bounds)), ((Fraction(9, 10),) * 2, "missed"), (record, reference))
                else:
                    pairs = mix.weighted(rec, ref)
                    self.assertEqual((sum(w for _, w in pairs), mix.p95(pairs)), (20, 500), (record, reference))
                checked += 1
        self.assertEqual(checked, 9 * 2 * 2 * 2)

    def test_two_reference_classes_through_one_cursor(self):
        # Ten entries read through two wrappers: both classes have the ten, in either order of the keys.
        for order in (("fast", "slow"), ("slow", "fast")):
            cursor = iter(range(10))
            ref = {name: Capture(cursor) for name in order}
            rec = {"fast": [100] * 10, "slow": [500] * 10}
            bounds = mix.within(rec, ref, 200)
            self.assertEqual((bounds, mix.verdict(*bounds)), ((Fraction(1, 2), Fraction(1, 2)), "missed"), order)
            held, sizes = mix.both({"fast": [100] * 10, "slow": [500] * 10},
                                   {name: Capture(c) for c in [iter(range(10))] for name in order})
            self.assertEqual({name: len(r) for name, r in sizes.items()}, {"fast": 10, "slow": 10})

    def test_a_stream_is_asked_for_its_cursor_once(self):
        slow = Once([500, 500])
        rec = {"fast": [100] * 18, "slow": slow}
        self.assertEqual(mix.within(rec, {"fast": range(18), "slow": slow}, 200), (Fraction(9, 10),) * 2)
        self.assertEqual(slow.opened, 1)
        ref = Once(range(7))
        held, sizes = mix.both({"fast": [100] * 18, "slow": [500] * 2}, {"fast": ref, "slow": ref})
        self.assertEqual(({name: len(r) for name, r in sizes.items()}, ref.opened), ({"fast": 7, "slow": 7}, 1))
        stream = (x for x in range(5))
        held, sizes = mix.both({"a": [1], "b": [2]}, {"a": stream, "b": stream})
        self.assertEqual({name: len(r) for name, r in sizes.items()}, {"a": 5, "b": 5})

    def test_a_count_belongs_to_the_cursor_that_was_read(self):
        # Fifty reference classes, each with a cursor of its own that nothing else holds: a count kept under
        # the number of an object that was let go could be taken for that of the next one.
        sizes = [3 + (7 * i) % 23 for i in range(50)]
        rec = {f"c{i}": [100] for i in range(50)}
        ref = {f"c{i}": Fresh(n) for i, n in enumerate(sizes)}
        held, counted = mix.both(rec, ref)
        self.assertEqual([len(counted[f"c{i}"]) for i in range(50)], sizes)
        pairs = mix.weighted(rec, {f"c{i}": Fresh(n) for i, n in enumerate(sizes)})
        self.assertEqual(sum(w for _, w in pairs), sum(sizes))

    def test_what_was_right_stays(self):
        slow = Capture(iter([500, 500]))
        with self.assertRaisesRegex(ValueError, "stream in two classes"):
            mix.within({"fast": [100] * 18, "a": slow, "b": slow}, {"fast": range(18), "a": range(1), "b": range(1)}, 200)
        again = [500, 500]
        held, sizes = mix.both({"fast": [100] * 18, "slow": again}, {"fast": range(18), "slow": again})
        self.assertIs(held["slow"], again)
        self.assertEqual(len(sizes["slow"]), 2)
        self.assertEqual(len(expected("ch23_mix").splitlines()), 108)
        self.assertEqual(printed(mix.main), expected("ch23_mix"))


class ACaseHasAFiniteLogitAndALabel(unittest.TestCase):
    @property
    def SCORES(self):
        """The fit, the scores, and the check that they share, which is new in Version 63."""
        return [getattr(cal, name) for name in ("fit_temperature", "nll", "brier", "ece", "bins_of", "checked")]

    def test_the_table_of_the_review(self):
        base = [(40.0, 1)] * 99
        for negative in (0, False, 0.0):
            self.assertAlmostEqual(cal.fit_temperature(base + [(40.0, negative)]), 40 / math.log(99), places=9)
            self.assertAlmostEqual(cal.nll(base + [(40.0, negative)]), 0.400, places=12)
        self.assertAlmostEqual(40 / math.log(99), 8.7048872074, places=10)
        for positive in (1, True, 1.0):
            with self.assertRaisesRegex(ValueError, "no finite T: the loss falls as T falls"):
                cal.fit_temperature(base + [(40.0, positive)])
            self.assertAlmostEqual(cal.nll(base + [(40.0, positive)]) / math.log1p(math.exp(-40.0)), 1, places=12)
        # A label that is missing, or no label: Version 62 took None as negative and a NaN, "0" and 2 as positive.
        for label in (None, math.nan, "0", "1", 2, -1, 0.5, D("0.5"), (), [1]):
            for score in self.SCORES:
                with self.assertRaisesRegex(ValueError, "a label is 0 or 1"):
                    score(base + [(40.0, label)])
                with self.assertRaisesRegex(ValueError, "a label is 0 or 1"):
                    score([(40.0, label)] + base)

    def test_a_logit_is_a_finite_number(self):
        self.assertAlmostEqual(cal.fit_temperature([(2, 1), (1, 0), (1.5, 1)]),
                               cal.fit_temperature([(2.0, True), (1.0, False), (1.5, True)]), places=12)
        # Version 64: a logit is held to known(), which also refuses a flag and a number beyond a float, and the
        # message is the one of known(); Version 63 said "no finite logit".
        for logit in (math.nan, math.inf, -math.inf, D("NaN"), D("Infinity")):
            for score in self.SCORES:                                          # Version 62: 0.20000000000000007
                with self.assertRaisesRegex(ValueError, "a logit is missing or too large"):
                    score([(2, 1), (1, 0), (logit, 1)])
        for logit in (None, "40", (40,)):
            for score in self.SCORES:
                with self.assertRaises((TypeError, ValueError)):
                    score([(2, 1), (1, 0), (logit, 1)])
        # Nothing is scored before every case is checked: the bad case is the last of a stream.
        for score in self.SCORES:
            stream = iter([(2.0, 1)] * 5 + [(1.0, None)])
            with self.assertRaises(ValueError):
                score(stream)
            self.assertEqual(list(stream), [])

    def test_a_search_has_two_finite_ends(self):
        for negative in (1, 2, 3, 5):
            data = [(40, 1)] * (100 - negative) + [(40, 0)] * negative
            want = 40 / math.log((100 - negative) / negative)                   # the stated 0.99 is the frequency
            self.assertAlmostEqual(cal.fit_temperature(data, hi=100), want, places=9)
            for lo, hi in ((0.2, math.inf), (0.2, math.nan), (math.nan, 10), (0, 10), (-1, 10), (10, 10), (20, 10)):
                with self.assertRaisesRegex(ValueError, "need 0 < lo < hi < inf"):   # Version 62: inf for hi = inf
                    cal.fit_temperature(data, lo, hi)
        self.assertAlmostEqual(40 / math.log(49), 10.277966847, places=9)
        with self.assertRaisesRegex(ValueError, "the minimum is above hi"):
            cal.fit_temperature([(40, 1)] * 98 + [(40, 0)] * 2)

    def test_a_flat_loss_leaves_the_temperature_open(self):
        data = [(0, 1), (0, 0), (0.0, 1)]
        with self.assertRaisesRegex(ValueError, "the loss is the same at every T"):
            cal.fit_temperature(data)
        losses = {round(cal.nll(data, t), 12) for t in (0.2, 1, 7.5, 10, 1000)}
        self.assertEqual(losses, {round(math.log(2), 12)})                      # every T has the loss of a half
        text = " ".join(cal.fit_temperature.__doc__.split())
        self.assertIn("every T has the same loss: the data leave T unidentified", text)
        self.assertIn("both finite", text)

    def test_ninety_nine_labels_and_one_that_is_missing(self):
        known = [(40.0, 1)] * 99
        with self.assertRaisesRegex(ValueError, "a label is 0 or 1"):
            cal.fit_temperature(known + [(40.0, None)])
        self.assertAlmostEqual(cal.fit_temperature(known + [(40.0, 0)]), 8.7048872074, places=9)
        with self.assertRaisesRegex(ValueError, "no finite T"):
            cal.fit_temperature(known + [(40.0, 1)])
        self.assertEqual(f"{40 / math.log(99):.2f}", "8.70")
        self.assertIn("  T = 8.70, NLL 0.400 -> 0.056\n", expected("ch18_calibration"))

    def test_scores_of_a_stream(self):
        rng = random.Random(62)
        data = [(rng.gauss(0, 3), rng.randint(0, 1)) for _ in range(400)]
        for score in (cal.ece, cal.nll, cal.brier, cal.bins_of):               # Version 62: TypeError for two of them
            self.assertEqual(score(iter(data)), score(data))
            self.assertEqual(score(row for row in data), score(data))
        self.assertAlmostEqual(cal.ece([(1.0, 1), (-1.0, 0)]), 1 / (1 + math.e), places=12)
        # Version 64: a bin has a fourth entry, the sum of its residuals; Version 63 had three.
        self.assertEqual(sum(b[0] for b in cal.bins_of(data)), 400)


class ARectangularIntervalKeepsItsWidth(unittest.TestCase):
    ORIGIN = 10 ** 18

    def test_the_two_cases_of_the_review(self):
        low, high = rect_interval(self.ORIGIN, 150_050, 1.0)
        self.assertEqual((low - self.ORIGIN, high - self.ORIGIN), (-150_050, 150_050))    # Version 62: 150 016
        self.assertEqual(int(float(self.ORIGIN + 150_050)) - self.ORIGIN, 150_016)
        self.assertAlmostEqual(100 * 150_016 / 150_050, 99.9773409, places=7)
        low, high = rect_interval(self.ORIGIN, 200_064, 0.75)
        self.assertEqual((low - self.ORIGIN, high - self.ORIGIN), (-150_048, 150_048))
        self.assertEqual(rect_interval(self.ORIGIN, 200_064, Fraction(3, 4)), (low, high))
        # The share of a rectangular distribution at or below the upper limit: seven eighths, and not 0.87492.
        below = Fraction(high - (self.ORIGIN - 200_064), 2 * 200_064)
        self.assertEqual(below, Fraction(7, 8))
        self.assertAlmostEqual(0.5 + 150_016 / 200_064 / 2, 0.8749200, places=7)
        self.assertAlmostEqual(100 * 150_016 / 200_064, 74.9840051, places=7)

    def test_the_limits_do_not_move_with_the_origin(self):
        rng = random.Random(62)
        for origin in (0, 10 ** 18, 10 ** 18 + 1, 17 * 10 ** 17 + 128, 2 ** 63 + 1024, -(10 ** 18) - 64):
            for _ in range(40):
                half = rng.randint(5_000_000, 10 ** 7) + rng.choice((0, 0.5, 0.25))   # wide enough for 2 ** 63
                p = rng.choice((1, 1.0, 0.75, 0.95, D("0.95"), Fraction(19, 20), 0.5))
                near, far = rect_interval(0, half, p), rect_interval(origin, half, p)
                width = Fraction(p) * Fraction(half)
                self.assertEqual(near, (-width, width))
                self.assertEqual((far[0] - origin, far[1] - origin), near, (origin, half, p))
                self.assertEqual([type(x) for x in far], [Fraction, Fraction])

    def test_every_type_of_half_width_and_level(self):
        for half, p in itertools.product((1, 1.0, D("1"), Fraction(1)), (0.95, D("0.95"), Fraction(19, 20), 1, 1.0)):
            low, high = rect_interval(0, half, p)                              # Version 62: TypeError for a decimal
            self.assertEqual((low, high), (-Fraction(p), Fraction(p)), (half, p))
        for center in (5, 5.5, D("5.5"), Fraction(11, 2)):
            for half, p in ((0, 0.95), (0.0, 1), (3, 0), (3, 0.0), (D("3"), D("0"))):
                got = rect_interval(center, half, p)
                self.assertEqual((got, type(got[0]), type(got[1])), ((center, center), type(center), type(center)))
        for half, p in ((-1, 0.5), (math.inf, 0.5), (math.nan, 0.5), (1, -0.1), (1, 1.1), (1, math.nan)):
            with self.assertRaises(ValueError):
                rect_interval(0, half, p)

    def test_the_example_of_chapter_4(self):
        low, high = rect_interval(0.0, 1.0)
        self.assertEqual((float(low), float(high)), (-0.95, 0.95))
        self.assertIn("central 95 % interval: +-0.950, k = 1.645\n", expected("ch04_accuracy"))
        self.assertAlmostEqual(0.95 / (1 / math.sqrt(3)), 1.645, places=3)

    def test_the_three_standard_deviations(self):
        for kind in KINDS:                                                     # Version 62: TypeError for a decimal
            self.assertEqual((u_rect(kind(1)), u_tri(kind(1)), u_resolution(kind(1))),
                             (1 / math.sqrt(3), 1 / math.sqrt(6), 1 / math.sqrt(12)))
            self.assertEqual([type(f(kind(1))) for f in (u_rect, u_tri, u_resolution)], [float] * 3)
        rng = random.Random(62)
        for _ in range(500):                                                   # floats as before, to the last bit
            x = rng.uniform(0, 10) * 10 ** rng.randint(-9, 9)
            self.assertEqual((u_rect(x), u_tri(x), u_resolution(x)), (x / math.sqrt(3), x / math.sqrt(6), x / math.sqrt(12)))
        self.assertEqual(u_rect(Fraction(1, 3)), (1 / 3) / math.sqrt(3))
        for f in (u_rect, u_tri, u_resolution):
            for bad in (-1, math.inf, math.nan, D("-1"), D("Infinity")):
                with self.assertRaises(ValueError):
                    f(bad)


def student_tail(t, dof):
    """P(T > t) far out: the leading term of the tail of Student's density."""
    c = math.gamma((dof + 1) / 2) / (math.sqrt(dof * math.pi) * math.gamma(dof / 2))
    return c * dof ** ((dof + 1) / 2) * t ** -dof / dof


class AnInfiniteLimit(unittest.TestCase):
    def test_the_table_of_the_review(self):
        for dof in (1, 0.01, 0.001, 0.5, 5, 40, math.inf):
            for value, u in ((0, 1), (195.0, 3), (0, 1e-300), (0, 1e300), (D("1.5"), D("3")), (Fraction(1, 3), 2)):
                r = Result(value, u, dof=dof)
                self.assertEqual((r.prob_below(-math.inf), r.prob_below(math.inf)), (0.0, 1.0), (dof, u))
                self.assertEqual((r.prob_below(D("-Infinity")), r.prob_below(D("Infinity"))), (0.0, 1.0))
            self.assertEqual((dist.t_sf(math.inf, dof), dist.t_cdf(math.inf, dof), dist.t_cdf(-math.inf, dof)), (0.0, 1.0, 0.0))
        # What Version 62 returned was the probability at the cap of the distance, 1e153 times sqrt(dof).
        for dof, was in ((1, 3.1831e-154), (0.01, 0.01465471695), (0.001, 0.35129272243)):
            self.assertAlmostEqual(student_tail(1e153 * math.sqrt(dof), dof) / was, 1, places=4)

    def test_a_finite_limit_keeps_a_finite_distance(self):
        # A limit of 1e308 at u = 1e-300 is 1e608 standard uncertainties away, beyond the range of a float.
        # It is not at infinity. Version 64: the tail is taken at that distance, from its logarithm. Version 63
        # returned the tail at the cap of the distance, 0.35129272243 and 0.01465471695, which overstated it:
        # the review of Version 63 gave 0.87720826 for the first probability, where Version 63 had 0.64870728.
        for dof, capped in ((0.001, 0.35129272243), (0.01, 0.01465471695)):
            r = Result(0, 1e-300, dof=dof)
            tail = far_tail(608, dof)
            self.assertLess(tail, capped)
            self.assertAlmostEqual(r.prob_below(1e308), 1 - tail, places=9)
            self.assertAlmostEqual(r.prob_below(-1e308), tail, places=9)
            self.assertLess(r.prob_below(1e308), r.prob_below(math.inf))
            self.assertAlmostEqual(r.prob_below(Fraction(10) ** 400), 1 - far_tail(700, dof), places=9)
        self.assertEqual(f"{Result(0, 1e-300, dof=0.001).prob_below(1e308):.8f}", "0.87720826")
        self.assertEqual(Result(0, 1e-300).prob_below(1e308), 1.0)             # the normal tail there is 0 as a float
        with self.assertRaises(ValueError):
            Result(0, 1).prob_below(math.nan)


class AResultWithoutUncertaintyHasNoFactor(unittest.TestCase):
    def test_the_case_of_the_review(self):
        r = Result(195, 0, dof=0.01)
        with self.assertRaisesRegex(ValueError, "quantile too large"):
            Result(195, 1, dof=0.01).k(0.99)                                   # the factor that would be looked for
        self.assertEqual((r.interval(0.99), r.expanded(0.99)), ((195, 195), 0.0))    # Version 62: expanded() raised
        self.assertEqual(r.report(0.99), "(195 ± 0), u = 0, p = 0.99")
        self.assertEqual(Result(195, 0, "ms").report(), "(195 ± 0) ms, u = 0, p = 0.95")
        self.assertEqual(str(Result(195, 0, "ms", 0.01)), "195(0) ms")

    def test_every_level_and_every_number_of_degrees(self):
        for value, dof in itertools.product((195, 99.9996, D("195.0004"), Fraction(1, 3)), (0.01, 1, 3, math.inf)):
            for p in (0, 0.5, 0.95, 0.99, D("0.999"), 1 - Fraction(1, 10 ** 400)):
                r = Result(value, 0, "ms", dof)
                self.assertEqual((r.expanded(p), r.interval(p)), (0.0, (value, value)))
                self.assertEqual(r.report(p).split(", ")[1], "u = 0")
            for p in (1, 1.5, -0.1, D("1")):                                   # the level is still held to its range
                for call in (Result(value, 0, "ms", dof).expanded, Result(value, 0, "ms", dof).report):
                    with self.assertRaises(ValueError):
                        call(p)
        # With an uncertainty and no coverage the factor is zero, and is printed.
        self.assertEqual(Result(195, 3, "ms").report(0), "(195 ± 0) ms, k = 0.00, p = 0")


def normal_quantile(log_tail):
    """z with log P(Z > z) = log_tail, far out, from the asymptotic series of the tail by bisection."""
    def log_q(z):
        w = 1 / (z * z)
        series = 1 - w + 3 * w ** 2 - 15 * w ** 3 + 105 * w ** 4 - 945 * w ** 5
        return -z * z / 2 - math.log(z) - math.log(2 * math.pi) / 2 + math.log(series)
    lo, hi = 5.0, 60.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if log_q(mid) > log_tail:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


class ALevelNextToOne(unittest.TestCase):
    def test_the_two_levels_of_the_review(self):
        # The factors exist: 38.52686726 and 38.46709544, from the logarithm of the tail.
        self.assertAlmostEqual(normal_quantile(math.log(5) - 325 * math.log(10)), 38.52686726, places=7)
        self.assertAlmostEqual(normal_quantile(math.log(5) - 324 * math.log(10)), 38.46709544, places=7)
        with localcontext() as context:
            context.prec = 400
            decimal_level = D(1) - D("1e-310")                                  # at 28 digits this would be 1
        self.assertLess(decimal_level, 1)
        for p in (1 - Fraction(1, 10 ** 324), 1 - Fraction(1, 10 ** 323), decimal_level, 1 - Fraction(1, 10 ** 5000)):
            r = Result(0, 1, "ms")
            for call in (r.k, r.expanded, r.interval, r.report):               # Version 62: an error of inv_cdf, or
                with self.assertRaisesRegex(ValueError, "p is too near 1"):    # 38.46740562 for the second
                    call(p)

    def test_down_to_the_least_normal_float(self):
        least = Fraction(2) ** -1022
        self.assertEqual(float(least), 2.2250738585072014e-308)
        r = Result(0, 1)
        for tail in (least, 2 * least, Fraction(5, 10 ** 301), Fraction(1, 10 ** 200), Fraction(1, 10 ** 50)):
            want = normal_quantile(math.log(tail.numerator) - math.log(tail.denominator))
            self.assertAlmostEqual(r.k(1 - 2 * tail) / want, 1, places=9, msg=float(tail))
        self.assertAlmostEqual(r.k(1 - Fraction(1, 10 ** 300)), 37.06578788, places=7)
        with self.assertRaisesRegex(ValueError, "p is too near 1"):
            r.k(1 - 2 * least + Fraction(1, 10 ** 400))
        # A float next to 1 is far from that: its tail is 5.6e-17.
        self.assertAlmostEqual(0.5 * math.erfc(r.k(1 - 2 ** -53) / math.sqrt(2)) / 2 ** -54, 1, places=9)
        text = " ".join(Result.k.__doc__.split())
        self.assertIn("least normal float, 2.2e-308, is refused", text)


class WhatAReportShows(unittest.TestCase):
    def test_a_small_factor_keeps_two_digits(self):
        r = Result(5, 50, "ms")
        self.assertEqual(r.report(0.001), "(5.000 ± 0.063) ms, k = 0.0013, p = 0.001")       # Version 62: k = 0.00
        self.assertAlmostEqual(r.k(0.001), NormalDist().inv_cdf(0.5005), places=12)
        self.assertAlmostEqual(r.k(0.001), 0.0012533145, places=10)
        for p, shown in ((0.95, "1.96"), (0.99, "2.58"), (0.6827, "1.00"), (0.5, "0.67"), (0.1, "0.13"),
                         (0.05, "0.063"), (0.01, "0.013"), (0.0001, "0.00013")):
            k = NormalDist().inv_cdf(0.5 + p / 2)
            digits = max(2, 1 - math.floor(math.log10(k)))                      # two decimals, or two digits
            self.assertEqual(f"{k:.{digits}f}", shown)
            self.assertEqual(r.report(p).split(", ")[1], f"k = {shown}", p)
        self.assertEqual(Result(5, 50, "ms", 3).report().split(", ")[1], "k = 3.18")
        self.assertEqual(Result(5, 50, "ms", 3).report(0.999).split(", ")[1], "k = 12.92")
        self.assertEqual(Result(5, 50, "ms").report(0).split(", ")[1], "k = 0.00")

    def test_a_zero_has_no_sign(self):
        r = Result(-0.1, 10, "ms")
        self.assertEqual((r.report(), str(r)), ("(0 ± 20) ms, k = 1.96, p = 0.95", "0(10) ms"))   # Version 62: (-0 ± 20)
        for value in (-0.0004, Fraction(-1, 2500), D("-0.0004")):
            r = Result(value, 0.57, "ms")
            self.assertEqual((r.report(), str(r)), ("(0.0 ± 1.1) ms, k = 1.96, p = 0.95", "0.00(57) ms"))
        self.assertEqual([core.fixed(x, 2) for x in (-0.004, -0.0, 0.0, -0.006, Fraction(-1, 1000), -1, D("-0.001"))],
                         ["0.00", "0.00", "0.00", "-0.01", "0.00", "-1.00", "0.00"])
        self.assertEqual([core.fixed(x, 0) for x in (-0.5, -0.4, -1.5, Fraction(-1, 2), -0.6)], ["0", "0", "-2", "0", "-1"])
        self.assertEqual(str(Result(-0.26, 0.5)), "-0.26(50)")
        self.assertEqual(Result(-0.26, 0.5).report(), "(-0.26 ± 0.98), k = 1.96, p = 0.95")

    def test_one_rounding_for_every_type(self):
        rng = random.Random(62)
        for _ in range(4000):
            x = rng.uniform(-1.5, 1.5) * 10 ** rng.randint(-6, 12)
            for places in (0, 1, 2, 5):
                with_sign = D(x).quantize(D(1).scaleb(-places))                # half to even, by the decimal module
                want = format(abs(with_sign) if not with_sign else with_sign, "f")
                self.assertEqual(core.fixed(x, places), want, (x, places))
                self.assertEqual(core.fixed(Fraction(x), places), want)
        self.assertEqual([core.fixed(x, 2) for x in (math.inf, -math.inf)], ["inf", "-inf"])
        self.assertEqual(core.fixed(math.nan, 2), "nan")
        self.assertEqual((core.fixed(0.125, 2), core.fixed(0.375, 2), core.fixed(2.675, 2)), ("0.12", "0.38", "2.67"))
        text = " ".join(core.fixed.__doc__.split())
        self.assertIn("does not hold every time stamp of nineteen digits", text)
        self.assertEqual(int(float(10 ** 18)), 10 ** 18)                       # this one a float holds
        self.assertNotEqual(int(float(10 ** 18 + 1)), 10 ** 18 + 1)


def wilson_limits(k, n):
    """The score interval at 95 %, from its two roots."""
    z = NormalDist().inv_cdf(0.975)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


class TheHalfWidthOfAWilsonInterval(unittest.TestCase):
    def test_the_first_rows_of_the_transcript(self):
        lines = expected("ch17_accuracy").splitlines()
        self.assertEqual(lines[0], "observed accuracy 87.0 %")
        for line, n in zip(lines[1:4], (500, 2000, 10_000)):
            low, high = wilson_limits(round(0.87 * n), n)
            self.assertEqual(line, f"n = {n:>5}: [{100 * low:.1f}, {100 * high:.1f}] %, half-width {50 * (high - low):.1f} points")
        self.assertNotIn("±", expected("ch17_accuracy"))                        # the sign of an expanded uncertainty
        low, high = wilson_limits(435, 500)
        self.assertEqual((round(100 * low, 4), round(100 * high, 4), round(50 * (low + high), 4)), (83.7679, 89.6679, 86.7179))
        self.assertLess(50 * (low + high), 87.0)                                # the middle is not the observed share
        self.assertTrue(all(len(line) <= 50 for line in lines))


class TheStepOf256ns(unittest.TestCase):
    def test_from_when_to_when(self):
        epoch = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)

        def at(nanoseconds):
            return epoch + datetime.timedelta(microseconds=nanoseconds // 1000)

        def step(year, month, day):
            seconds = (datetime.datetime(year, month, day, tzinfo=datetime.timezone.utc) - epoch).total_seconds()
            return math.ulp(float(int(seconds) * 10 ** 9))
        self.assertEqual((at(2 ** 60).date(), at(2 ** 61).date()), (datetime.date(2006, 7, 14), datetime.date(2043, 1, 25)))
        self.assertEqual((math.ulp(float(2 ** 60)), math.ulp(math.nextafter(2.0 ** 61, 0)), math.ulp(2.0 ** 61)), (256, 256, 512))
        self.assertEqual([step(2006, 1, 1), step(2006, 7, 14), step(2006, 7, 15)], [128, 128, 256])
        self.assertEqual([step(2043, 1, 25), step(2043, 1, 26), step(2043, 2, 1)], [256, 512, 512])
        self.assertEqual(step(2023, 11, 14), 256)                               # the stamp of Listing 19.1


class StreamsInChapters10And18(unittest.TestCase):
    def test_all_requests_from_a_stream(self):
        rows = [100.0] * 19 + [None]
        self.assertEqual(censoring.p95_all(x for x in rows), (100.0, False))   # Version 62: TypeError
        self.assertEqual(censoring.p95_all(iter([100.0] * 18 + [None, None])), (censoring.TIMEOUT, True))
        rng = random.Random(62)
        for _ in range(200):
            n = rng.randint(1, 60)
            rows = [rng.choice((None, rng.uniform(1, 999))) if rng.random() < 0.2 else rng.uniform(1, 999) for _ in range(n)]
            want = censoring.p95_all(rows)
            self.assertEqual(censoring.p95_all(iter(rows)), want)
            ordered = sorted(x if x is not None else math.inf for x in rows)
            at_rank = ordered[math.ceil(Fraction(95 * n, 100)) - 1]
            self.assertEqual(want, (censoring.TIMEOUT, True) if at_rank == math.inf else (at_rank, False))
        self.assertEqual(censoring.p95_completed(x for x in [100.0] * 19 + [None]), 100.0)
        self.assertEqual(printed(censoring.main), expected("ch10_censoring"))


def tail(n, k, p, upper):
    """P(X >= k) if upper, else P(X <= k), for X binomial with n trials and probability p."""
    terms = range(k, n + 1) if upper else range(0, k + 1)
    return math.fsum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in terms)


def exact_limits(k, n):
    """Clopper and Pearson at 95 %: the proportions at which the two tails are 2.5 %, by halving."""
    out = []
    for upper in (True, False):
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if (tail(n, k, mid, upper) < 0.025) == upper:
                lo = mid
            else:
                hi = mid
        out.append((lo + hi) / 2)
    return out


class TheEndsOfATemperaturePointOutward(unittest.TestCase):
    TABLE = {1: "4.83 to 14.02", 2: "6.64 to 15.50", 3: "7.88 to 16.85", 5: "9.77 to 19.40"}
    NEAREST = {1: "4.83 to 14.01", 2: "6.65 to 15.50", 3: "7.89 to 16.85", 5: "9.78 to 19.40"}

    def test_the_table_of_the_review(self):
        for negative, shown in self.TABLE.items():
            p_low, p_high = exact_limits(100 - negative, 100)
            low, high = 40 / math.log(p_high / (1 - p_high)), 40 / math.log(p_low / (1 - p_low))
            got = cal.temperatures(40, 100 - negative, 100)
            self.assertAlmostEqual(got[0] / low, 1, places=9)
            self.assertAlmostEqual(got[1] / high, 1, places=9)
            down = D(low).quantize(D("0.01"), rounding=ROUND_FLOOR)
            up = D(high).quantize(D("0.01"), rounding=ROUND_CEILING)
            self.assertEqual((cal.worded(got), f"{down} to {up}"), (shown, shown))
            self.assertTrue(down <= D(low) and D(high) <= up)                   # the print encloses the interval
            self.assertEqual(f"{low:.2f} to {high:.2f}", self.NEAREST[negative])   # what Version 62 printed
        # One negative label: 14.012 is inside the interval and beyond 14.01.
        high = cal.temperatures(40, 99, 100)[1]
        self.assertAlmostEqual(high, 14.01393582, places=8)
        self.assertTrue(14.01 < 14.012 < high < 14.02)
        low = cal.temperatures(40, 98, 100)[0]
        self.assertTrue(6.64 < low < 6.648 < 6.65)

    def test_the_transcript(self):
        lines = expected("ch18_calibration").splitlines()
        self.assertEqual((lines[16], lines[23]), ("  95 % interval for T: 4.83 to 14.02", "  1 positive at -40: 4.83 to 14.02"))
        self.assertEqual(lines[21], "  100 positive: T up to 12.20")
        self.assertEqual(lines[20], "  55 positive: T from 64.76, no upper end")
        self.assertEqual(printed(cal.main), expected("ch18_calibration"))


class OneDurationRecoveredAt103(unittest.TestCase):
    A = [100] * 100
    B = [101] * 94

    def test_the_objective_and_the_percentile(self):
        rec = {"requests": self.B + [103] + [None] * 5}
        low, high = mix.within(rec, rec, 105)
        self.assertEqual((low, high, mix.verdict(low, high)), (Fraction(95, 100), 1, "met"))
        seen = set()
        for five in itertools.combinations_with_replacement((0, 100, 101, 102, 102.5, 103, 104, 201, 10 ** 6), 5):
            at95 = p95(self.B + [103] + list(five))
            self.assertTrue(101 <= at95 <= 103, five)
            seen.add(at95)
        self.assertEqual((min(seen), max(seen)), (101, 103))
        self.assertEqual(p95(self.B + [103] + [101] * 5) - p95(self.A), 1)
        self.assertEqual(p95(self.B + [103] + [201] * 5) - p95(self.A), 3)

    def test_before_the_recovery(self):
        for common in (102, 105, 10 ** 3, 10 ** 9):                            # no upper end for the change
            self.assertEqual(p95(self.B + [common] * 6) - p95(self.A), common - 100)
        rec = {"requests": self.B + [None] * 6}
        self.assertEqual(mix.verdict(*mix.within(rec, rec, 105)), "unresolved")


class WhatIsSquared(unittest.TestCase):
    def test_a_contribution_is_the_uncertainty_times_its_coefficient(self):
        y, u, parts = lpu(lambda x: 2 * x[0], [10.0], [3.0])
        self.assertEqual((y, u, parts), (20.0, 6.0, [6.0]))
        y, u, parts = lpu(lambda x: 2 * x[0] + x[1], [10.0, 5.0], [3.0, 8.0])
        self.assertEqual((u, parts), (10.0, [6.0, 8.0]))                        # the root of 36 + 64, not of 9 + 64
        self.assertNotAlmostEqual(u, math.hypot(3.0, 8.0))

    def test_u_is_the_scale_of_a_student_result(self):
        r = Result(0, 3, "ms", 4)
        self.assertAlmostEqual(r.expanded() / 3, 2.7764451, places=6)           # t at 0.975 with 4 degrees of freedom
        self.assertAlmostEqual(3 * math.sqrt(4 / (4 - 2)), 3 * math.sqrt(2), places=12)
        self.assertAlmostEqual(3 * math.sqrt(2), 4.2426, places=4)
        # The variance of T with four degrees of freedom, by Simpson's rule on t^2 times its density: 2.
        c = math.gamma(2.5) / (math.sqrt(4 * math.pi) * math.gamma(2))
        steps, top = 200_000, 2000.0
        h = top / steps
        ys = [(i * h) ** 2 * c * (1 + (i * h) ** 2 / 4) ** -2.5 for i in range(steps + 1)]
        inner = 2 * h / 3 * (ys[0] + ys[-1] + 4 * sum(ys[1:-1:2]) + 2 * sum(ys[2:-1:2]))
        self.assertAlmostEqual(inner + 2 * c * 32 / (2 * top ** 2), 2, places=5)   # and the tail beyond, c 4^2.5 / (2 t^2)
        text = " ".join(Result.__doc__.split())
        self.assertIn("u is its scale", text)
        self.assertIn("at 4 degrees of freedom is u times the square root of 2", text)


if __name__ == "__main__":
    unittest.main()
