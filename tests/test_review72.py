"""Regression tests after the review of Version 72.

The review found that Listing 22.4 rounded one reading in two ways, as a float and as a decimal; that the
conversion of a unit kept forty digits and could put a value on a limit that it exceeds; that a rounded estimate
was recorded with an increment of zero where the increment lies below the least float; that the listing printed
a guard band for a record that the checks had stopped; that the decimal context of the caller could change the
number of gaps that Listing 16.4 admits, and that under the trap decimal.FloatOperation a bound of the package
and the decision of the stored record failed; that a decimal of more than 4300 digits could be decided and not
shown; that two functions of Listing 16.4 took a flag and a text for a time and a negative rate for a rate; that
a probability was returned as the nearest float where two floats were left; that a number on a limit which is no
decimal cannot be printed as equal to it; and that the replay compared printed numbers as floats and collapsed
white space where the text has to be identical.

Beyond the review, the hostile context of tests/test_review71.py did not set decimal.FloatOperation, although the
README called it a context with every trap set; a decimal NaN was refused with one error or another as the traps
of the caller said; combine() took the absolute value of a decimal in the context of the caller, and the tails of
Student's distribution compared a decimal with floats of their own; chance() returned 2.0 for terms that sum to 2,
after 5120 digits; full() and the text of a quantity printed a long whole number through a text of its own;
resolved() and the upper tail took a flag for a number; and the interval of the stored record was rounded to the
nearest and not outward.

The references of these tests are their own: arithmetic in fractions and whole numbers, texts put together from
digits, and logarithms and exponentials in a decimal context of 80 digits that the tests state themselves.
"""
import contextlib
import copy
import decimal
import io
import math
import random
import struct
import subprocess
import sys
import unittest
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import run_all
from examples import ch09_histogram as h9
from examples import ch16_silence as silence
from examples import ch18_calibration as calibration
from examples import ch22_budget as budget
from examples import ch22_record_check as check
from examples import ch23_case as case
from swmetro import core, dist, propagate, record, stats, units
from swmetro.record import decide, guard, in_unit, invalid, read

try:
    import numpy
except ImportError:                                                        # the package needs no NumPy
    numpy = None

ROOT = Path(__file__).resolve().parent.parent
D = Decimal
F = Fraction
C = decimal.Context(prec=80, rounding=decimal.ROUND_HALF_EVEN, Emax=decimal.MAX_EMAX, Emin=decimal.MIN_EMIN,
                    traps=[decimal.InvalidOperation, decimal.DivisionByZero, decimal.Overflow])
SIGNALS = (decimal.Inexact, decimal.Rounded, decimal.Underflow, decimal.Subnormal, decimal.Overflow,
           decimal.InvalidOperation, decimal.DivisionByZero, decimal.Clamped, decimal.FloatOperation)
FOUR_BYTES = struct.unpack("f", struct.pack("f", 0.95))[0]                 # the float that 0.95 is in four bytes
ROUNDING = ("unresolved", ["rounding decides: wants working values"])


def expected(name):
    return (ROOT / "expected" / f"{name}.txt").read_text(encoding="utf-8")


def printed(function, *args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        function(*args)
    return out.getvalue()


def refusal(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except Exception as error:                                            # the kind and the words of a refusal
        return f"{type(error).__name__}: {error}"
    return None


def hostile():
    """A decimal context of three digits, exponents from -10 to 10, rounding away from zero, with the trap of
    every signal that the module has, decimal.FloatOperation among them."""
    return decimal.Context(prec=3, Emax=10, Emin=-10, rounding=decimal.ROUND_UP, traps=list(SIGNALS))


def loose():
    """A decimal context of five digits that rounds toward zero and traps nothing."""
    return decimal.Context(prec=5, rounding=decimal.ROUND_DOWN, traps=[])


def stated(x):
    """The number that a float, an int or a decimal of a record states, as a fraction: a float as the decimal
    that Python prints for it."""
    return F(repr(x)) if isinstance(x, float) else F(x)


def half_even(x, places):
    """The fraction x rounded to `places` decimals, a half to the even digit, in whole numbers."""
    scaled = F(x) * F(10) ** places
    low = scaled.numerator // scaled.denominator
    rest = scaled - low
    n = low + (rest > F(1, 2) or (rest == F(1, 2) and low % 2 == 1))
    return n * F(10) ** -places


def least_float(x):
    """The least float that is not below the fraction x."""
    f = float(x)
    return f if F(f) >= x else math.nextafter(f, math.inf)


def text(n, places):
    """The whole number n with `places` digits after the point."""
    whole, part = divmod(abs(n), 10 ** places)
    tail = f".{part:0{places}d}" if places else ""
    return ("-" if n < 0 else "") + f"{whole}{tail}"


def reference(x, places, limit, way):
    """x rounded upward, downward or to the nearest with a half to even, at the fewest places from `places` at
    which the rounded number stands to the limit as x does."""
    x = F(x)
    while True:
        scaled = x * 10 ** places
        low = scaled.numerator // scaled.denominator
        rest = scaled - low
        if way == "up":
            n = low + (rest > 0)
        elif way == "down":
            n = low
        else:
            n = low + (rest > F(1, 2) or (rest == F(1, 2) and low % 2 == 1))
        y = F(n, 10 ** places)
        if (y > limit) == (x > limit) and (y < limit) == (x < limit):
            return text(n, places)
        places += 1


def long_decimal(places, head="0.2"):
    """`head`, then zeros, then a 1 in the given decimal place: a text, a decimal and a fraction of it."""
    text = head + "0" * (places - len(head.partition(".")[2]) - 1) + "1"
    return text, D(text), F(D(head)) + F(1, 10 ** places)


class Record(unittest.TestCase):
    def setUp(self):
        self.rec = check.load("measurement_record.json")

    def one(self, value, u, dof, risk, limit, digits="all", step=None, unit=None):
        """The published record with one component, the uncertainty of the result equal to it."""
        out = copy.deepcopy(self.rec)
        part = copy.deepcopy(out["components"][1])
        part["u"], part["dof"] = u, dof
        out["components"] = [part]
        result = out["result"]
        result["value"], result["u"], result["dof"] = value, u, dof
        result["significant_digits"] = digits
        if step is not None:
            result["value_step"] = step
        out["decision"]["risk"], out["decision"]["limit"] = risk, limit
        if unit is not None:
            out["measurand"]["unit"] = unit
        return out

    def band(self, rec):
        """What near() prints of the estimate plus its band, or None where it prints no such line."""
        lines = printed(check.near, "a title", rec, rec["result"]["value"]).splitlines()
        last = lines[-1]
        if not last.startswith("  estimate + band: "):
            return None
        return last.split(": ")[1]


class OneReadingOneRounding(Record):
    """Section "One reading rounds two ways"."""

    def test_the_record_of_the_review(self):
        # One component of 0.10 with one degree of freedom at a risk of 0.25: the coverage factor is 1, and the
        # working values give 2.675 + 0.10 = 2.775, below the limit of 2.785. Rounded, the estimate is 2.68 for
        # 2.675 to 2.685, the uncertainty 0.10 for 0.0995 to 0.105, and the factor lies about 1: 2.685 + 0.105
        # is above the limit and 2.675 + 0.0995 below it, so the rounded record cannot decide.
        for value in (2.675, D("2.675")):
            rec = self.one(value, D("0.10"), D(1), D("0.25"), D("2.785"))
            self.assertEqual(decide(rec), ("accept", []))
            shown = check.displayed(rec)
            self.assertEqual(stated(shown["result"]["value"]), F(268, 100), type(value))   # Version 72: 2.67
            self.assertIs(type(shown["result"]["value"]), type(value))
            self.assertEqual(stated(shown["result"]["value_step"]), F(1, 100))
            self.assertEqual(invalid(shown), [])
            self.assertEqual(decide(shown), ROUNDING, type(value))         # Version 72: accept for the float
        self.assertEqual(round(2.675, 2), 2.67)                            # the binary number lies below the half
        self.assertLess(F(2.675), F(2675, 1000))

    def test_a_float_is_rounded_as_the_decimal_that_it_states(self):
        cases = [(2.675, 2), (2.665, 2), (1.005, 2), (0.125, 2), (0.375, 2), (1.05, 1), (0.355, 2), (2.5, 0),
                 (3.5, 0), (-2.675, 2), (-0.135, 2), (1234.5, -1), (1250.0, -2), (1350.0, -2), (5e-324, 323),
                 (1.5e-323, 323), (0.1 + 0.2, 16), (1e22, -21), (0.0, 3), (7.0, 0), (194.65, 1)]
        rng = random.Random(72)
        for _ in range(1500):                                              # half of these are ties
            j = rng.randint(1, 9)
            cases.append((float(F(5 * rng.randrange(-10 ** 6, 10 ** 6), 10 ** j)), j - 1))
        differ = 0
        for x, places in cases:
            want = half_even(stated(x), places)
            got = record.rounded(x, places)
            self.assertIs(type(got), float, (x, places))
            self.assertEqual(stated(got), want, (x, places))
            as_decimal = record.rounded(D(repr(x)), places)
            self.assertIs(type(as_decimal), D)
            self.assertEqual(F(as_decimal), want, (x, places))
            self.assertEqual(as_decimal.as_tuple().exponent, -places, (x, places))
            differ += stated(round(x, places)) != want
        self.assertGreater(differ, 300)                                    # round() of the float: another number
        # A whole number stays one, with the tens for places below zero.
        self.assertEqual((record.rounded(7, -1), record.rounded(25, -1), record.rounded(35, -1), record.rounded(12, 3)),
                         (10, 20, 40, 12))
        self.assertIs(type(record.rounded(25, -1)), int)

    def test_where_no_float_states_the_rounded_number(self):
        # The largest float is 1.7976931348623157e308, and its two leading digits round to 1.8e308, which is
        # above it: no float prints as that number.
        top = sys.float_info.max
        got = record.rounded(top, -307)
        self.assertIs(type(got), D)
        self.assertEqual(F(got), 18 * F(10) ** 307)
        self.assertEqual(half_even(stated(top), -307), 18 * F(10) ** 307)

    def test_the_uncertainties_and_the_degrees_of_freedom(self):
        # Two significant digits of the decimal that is stated. As binary numbers 1.05 lies above its half and
        # 0.355 below it; as decimals both are ties and go to the even digit.
        for x, want in ((1.05, F(10, 10)), (0.355, F(36, 100)), (0.345, F(34, 100)), (2.675, F(27, 10)),
                        (0.0995, F(10, 100)), (15.5, F(16)), (16.5, F(16)), (0.10, F(1, 10))):
            self.assertEqual(stated(check.two(x)), want, x)
            self.assertEqual(F(check.two(D(repr(x)))), want, x)
        self.assertEqual((round(1.05, 1), round(0.355, 2)), (1.1, 0.35))   # what the binary numbers give
        # One record, written with floats and with decimals, is displayed as one record.
        floats = self.one(194.65, 0.355, 16.5, 0.05, 200.0)
        decimals = self.one(D("194.65"), D("0.355"), D("16.5"), D("0.05"), D("200.0"))
        a, b = check.displayed(floats), check.displayed(decimals)
        for key in ("value", "u", "dof", "value_step"):
            self.assertEqual(stated(a["result"][key]), stated(b["result"][key]), key)
        self.assertEqual(stated(a["components"][0]["u"]), stated(b["components"][0]["u"]))
        self.assertEqual([stated(a["result"][key]) for key in ("value", "u", "dof", "value_step")],
                         [F(19465, 100), F(36, 100), F(16), F(1, 100)])
        self.assertEqual(decide(a), decide(b))


class AConvertedValueInFull(Record):
    """Section "Unit conversion can turn rejection into acceptance"."""

    def test_the_record_of_the_review(self):
        text, value, exactly = long_decimal(41)                            # 0.2 s and 10 to the power -41 s
        rec = self.one(value, D(0), D(1), D("0.25"), D("0.2"), unit="s")
        self.assertEqual(decide(rec), ("reject", []))
        ms, u = in_unit(rec, "ms")
        self.assertEqual(F(ms), 1000 * exactly)                            # Version 72: 200
        self.assertEqual(str(ms), "200." + "0" * 37 + "1")
        self.assertEqual(u, 0)
        self.assertEqual(in_unit(rec, "s")[0], value)
        there = self.one(ms, u, D(1), D("0.25"), D(200), unit="ms")
        self.assertEqual(decide(there), ("reject", []))                    # Version 72: accept

    def test_every_conversion_is_exact(self):
        seconds = {"ns": F(1, 10 ** 9), "us": F(1, 10 ** 6), "ms": F(1, 1000), "s": F(1), "ks": F(1000)}
        bits = {"bit": F(1), "B": F(8), "KiB": F(8 * 2 ** 10), "MiB": F(8 * 2 ** 20), "kB": F(8000),
                "Mbit": F(10 ** 6)}
        values = [long_decimal(41)[1], long_decimal(60, "241.4")[1], D("0.2414"), D("9007199254740993"),
                  D("1E-300"), D("123456789012345678901234567890.123456789012345678901234567890"), 0.2414, 7]
        for table in (seconds, bits):
            for here, there in ((a, b) for a in table for b in table):
                for value in values:
                    rec = self.one(value, value, D(1), D("0.25"), D(1), unit=here)
                    got, u = in_unit(rec, there)
                    want = stated(value) * table[here] / table[there]
                    self.assertEqual(F(got), want, (here, there, value))
                    self.assertEqual(F(u), want)
                    self.assertIs(type(got), D)

    def test_what_is_unknown_stays_unknown(self):
        rec = self.one(D("0.2414"), None, D(1), D("0.25"), D(1), unit="s")
        self.assertEqual(in_unit(rec, "ms"), (D("241.4"), None))


class AnIncrementThatTheFormatHas(Record):
    """Section "A rounded estimate is recorded as exact"."""

    def test_the_record_of_the_review(self):
        rec = self.one(D("2.345E-322"), D("1.0E-323"), D(1), D("0.25"), D("2.446E-322"))
        self.assertEqual(invalid(rec), [])
        # Version 72 issued the estimate 2.34E-322 with an increment of 0.0, and accepted it.
        self.assertEqual(refusal(check.displayed, rec), "ValueError: no such increment")
        self.assertEqual(float(D("1E-324")), 0.0)                          # the increment is no float above zero
        # The format has no such number either: the checker refuses it where it is written.
        stated_step = self.one(D("2.34E-322"), D("1.0E-323"), D("1.0"), D("0.25"), D("2.446E-322"), 2, D("1E-324"))
        self.assertEqual(invalid(stated_step), ["result.value_step is not a finite number >= 0"])

    def test_the_increment_is_the_float_that_prints_as_it(self):
        # An uncertainty of 1.0 times a power of ten puts the estimate at the next place. The increment is
        # formed from the decimal, and no longer as 10.0 ** -place, a power that is left to the C library.
        for place in range(1, 324):
            for u in (D((0, (1, 0), -place)), float(f"1e-{place - 1}")):
                rec = self.one(type(u)(0), u, D(1), D("0.25"), D(1))
                shown = check.displayed(rec)
                step = shown["result"]["value_step"]
                self.assertIs(type(step), float)
                self.assertEqual(stated(step), F(1, 10 ** place), place)
                self.assertEqual(step, float(f"1e-{place}"), place)
                self.assertEqual(invalid(shown), [], place)
        for u in (D((0, (1, 0), -324)), float("1e-323")):                  # the last one has no increment
            rec = self.one(type(u)(0), u, D(1), D("0.25"), D(1))
            self.assertEqual(refusal(check.displayed, rec), "ValueError: no such increment")
        # The tens, and no uncertainty.
        shown = check.displayed(self.one(D("12345"), D("240"), D(4), D("0.05"), D(20000)))
        self.assertEqual((shown["result"]["value"], stated(shown["result"]["value_step"])), (D("1.234E+4"), 10))
        shown = check.displayed(self.one(D("12.345"), D(0), D(4), D("0.05"), D(20)))
        self.assertEqual((shown["result"]["value"], shown["result"]["value_step"]), (D("12.345"), 0))


class NoBandWhereTheChecksStop(Record):
    """Section "A failed record check still prints a guard band"."""

    def stops(self):
        shown = check.displayed(self.rec)
        between = ("components", 1, "u")
        return (
            ("off its grid", check.altered(shown, {("result", "value"): 194.65}),
             ("invalid", ["result.value is not a multiple of its step"])),
            ("a lost record", check.altered(self.rec, {("capture", "records_lost"): 1}),
             ("unresolved", ["records were lost"])),
            ("loss unknown", check.altered(self.rec, {("capture", "records_lost"): None}),
             ("unresolved", ["wants capture.records_lost"])),
            ("another format", check.altered(self.rec, {("format",): "measurement record, version 4"}),
             ("unresolved", ["format not supported"])),
            ("another procedure", check.altered(self.rec, {("decision", "procedure"): "by eye"}),
             ("unresolved", ["procedure not supported"])),
            ("correlated", check.altered(self.rec, {("covariances",): [{"between": [0, 1], "r": 0.5}]}),
             ("unresolved", ["correlations not supported"])),
            ("a component changed", check.altered(self.rec, {between: 5.0}),
             ("invalid", ["result.u does not follow from the components"])),
            ("not evaluated", check.altered(self.rec, {("not_evaluated",): ["probe drift"]}),
             ("unresolved", ["not evaluated: probe drift"])),
            ("a negative uncertainty", check.altered(self.rec, {("result", "u"): -10.0}), None),
            ("the template", check.load("record_template.json"), None),
        )

    def test_a_record_that_the_checks_stop_has_no_guarded_values(self):
        for name, rec, want in self.stops():
            found = record.stopped(rec)
            self.assertIsNotNone(found, name)
            self.assertEqual(found, decide(rec), name)
            self.assertIn(found[0], ("invalid", "unresolved"), name)
            if want is not None:
                self.assertEqual(found, want, name)
            # guard() of Version 72: a band for seven of the ten, a TypeError for the three others.
            self.assertEqual(refusal(guard, rec), "ValueError: no decision is formed", name)
            out = printed(check.near, name, rec, rec["result"].get("value"))
            self.assertNotIn("estimate + band", out, name)                 # Version 72, the first: 199.91 to 200.17
            self.assertEqual(out, printed(check.report, name, rec), name)
        self.assertEqual(record.stopped(check.load("record_template.json"))[0], "unresolved")

    def test_a_record_that_passes_them_has(self):
        shown = check.displayed(self.rec)
        k = dist.t_isf(0.05, 2.0)                                          # a computed factor, about 2.92
        passing = (
            ("the stored record", self.rec, ("accept", [])),
            ("rounded, 194.6", check.altered(shown, {("result", "value"): 194.6}), ROUNDING),
            ("rounded, 194.8", check.altered(shown, {("result", "value"): 194.8}), ("undecided", [])),
            ("above the limit", check.altered(self.rec, {("result", "value"): 212.0}), None),
            ("at the limit", self.one(D(100), D(10), D(2), D("0.05"), D(repr(100 + 10 * k))),
             ("unresolved", ["at the limit: the coverage factor decides"])),
            ("no factor", self.one(D(100), D("1E-200"), D(1), D("1E-160"), D(200)),
             ("unresolved", ["coverage factor not computed"])),
        )
        for name, rec, want in passing:
            self.assertIsNone(record.stopped(rec), name)
            if want is not None:
                self.assertEqual(decide(rec), want, name)
            (lo, hi), (low, top) = guard(rec)
            self.assertLessEqual(lo, hi, name)
            self.assertLessEqual(low, top, name)
            self.assertIsNotNone(self.band(rec), name)
        # The rounded record whose rounding alone decides: the two ends lie on either side of the limit.
        lo, hi = guard(passing[1][1])[0]
        self.assertLess(lo, 200)
        self.assertGreater(hi, 200)

    def test_the_transcript(self):
        self.assertEqual(printed(check.main), expected("ch22_record_check"))


class TheSettingsOfTheCaller(Record):
    """Section "The caller's decimal settings can change a result"."""

    def under(self, context, call):
        with decimal.localcontext(context):
            try:
                return call()
            except Exception as error:                                    # the kind and the words of a refusal
                return f"{type(error).__name__}: {error}"

    def same(self, calls):
        """The outcomes of the calls, which have to be the same under every context."""
        usual = [self.under(decimal.Context(), call) for call in calls]
        for context in (hostile(), loose()):
            for i, call in enumerate(calls):
                self.assertEqual(repr(self.under(context, call)), repr(usual[i]), i)
        return usual

    def test_every_trap_is_every_trap(self):
        self.assertEqual(set(SIGNALS), set(decimal.getcontext().flags))    # the nine signals of the module
        with decimal.localcontext(hostile()):
            self.assertEqual(refusal(lambda: D("0.5") < 0.5), "FloatOperation: [<class 'decimal.FloatOperation'>]")
            self.assertEqual(refusal(lambda: D(1) / D(3)), "Inexact: [<class 'decimal.Inexact'>, "
                             "<class 'decimal.Rounded'>]")

    def test_twenty_gaps_are_twenty_gaps(self):
        # 20.02 is 20 times 1.001. In three digits the product is rounded to 20.0, and Version 72 refused.
        self.assertEqual(F("20.02"), 20 * F("1.001"))
        args = D("0.4"), D("20.02"), D("1.001")
        usual = silence.gap_terms(*args)
        self.assertEqual(len(usual), 20)
        self.assertEqual(usual[0], (F(2, 5) * (F("20.02") - F("1.001")) + 1, F(2, 5) * F("1.001")))
        for digits in (3, 5, 28, 60):
            for rounding in (decimal.ROUND_UP, decimal.ROUND_DOWN, decimal.ROUND_HALF_EVEN):
                with decimal.localcontext(decimal.Context(prec=digits, rounding=rounding)):
                    self.assertEqual(silence.gap_terms(*args), usual, (digits, rounding))
                    self.assertEqual(refusal(silence.gap_terms, D("0.4"), D("20.03"), D("1.001")),
                                     "ValueError: need 20 gaps or fewer")
                    self.assertEqual(refusal(silence.gap_terms, D("0.4"), D("20.01"), D("1.001")), None)
        got = self.same([lambda: silence.gap_somewhere(*args), lambda: silence.gap_terms(*args),
                         lambda: silence.gap_bound(D("20.02"), D("1.001"))])
        self.assertEqual(got[1], usual)

    def test_the_bound_and_the_record_of_the_review(self):
        # The rate at which 30 s without an event has the probability 1 - 0.95: ln(1 / tail) / 30.
        tail = 1 - F(0.95)                                                 # the level is the float 0.95
        log = C.ln(C.divide(D(tail.denominator), D(tail.numerator)))
        want = least_float(F(C.divide(log, D(30))))
        with decimal.localcontext(hostile()):
            self.assertEqual(stats.quiet_bound(D("30")), want)             # Version 72: FloatOperation
            self.assertEqual(decide(self.rec), ("accept", []))             # Version 72: FloatOperation
            self.assertEqual(invalid(self.rec), [])
            self.assertEqual(record.stopped(self.rec), None)
        self.assertEqual(stats.quiet_bound(D("30")), want)

    def calls(self):
        rec = self.rec
        big = D("12345.678901234567890123456789")
        spans = [(D("0"), D("30.5")), (D("40.25"), D("70"))]
        return [
            lambda: stats.quiet_bound(D("30")),
            lambda: stats.quiet_bound(D("30.5"), D("0.9")),
            lambda: stats.rate_upper(2, D("30.5"), D("0.9")),
            lambda: stats.rate_limits(3, D("30.5"), D("0.9")),
            lambda: stats.as_level(D("0.95")),
            lambda: stats.poisson_within(3, D("2.5")),
            lambda: stats.binomial_within(2, 20, D("0.1")),
            lambda: stats.sum_within([(D("1.5"), D("0.25")), (-1, D("2"))]),
            lambda: stats.chance([(1, D("0")), (-1, D("2.5"))]),
            lambda: stats.wilson(3, 20, D("0.95")),
            lambda: stats.weighted_quantile([(D("1.5"), D("0.5")), (big, 2), (D("3"), F(1, 3))], D("0.5")),
            lambda: stats.upward(big, 3, D("12345.679")),
            lambda: (decide(rec), guard(rec), record.combined(rec), record.spread(rec), invalid(rec)),
            lambda: in_unit(rec, "s"),
            lambda: printed(check.main),
            lambda: check.displayed(rec),
            lambda: record.rounded(big, 5),
            lambda: core.combine(D("1.2345"), big),
            lambda: core.welch_satterthwaite([D("1.5"), D("0.25")], [D("3"), 4]),
            lambda: (core.decimals(big), core.fixed(big, 3), core.fmt(big, D("0.0123456"), "ms")),
            lambda: (core.u_rect(D("0.5")), core.u_tri(big), core.u_resolution(D("0.001"))),
            lambda: core.rect_interval(big, D("0.5"), D("0.95")),
            lambda: core.Result(big, D("0.0123456"), "ms", D("15.7")).prob_below(D("12345.7")),
            lambda: core.Result(big, D("0.0123456"), "ms", D("15.7")).report(D("0.95")),
            lambda: propagate.lpu(lambda t: F(t[1]) - F(t[0]), [D(10), big], [D("0.5"), D("0.25")]),
            lambda: (dist.t_sf(D("1.5"), D(4)), dist.t_cdf(D("-1.5"), D(4)), dist.betainc(D(2), D("0.5"), D("0.3"))),
            lambda: (silence.starts(spans, D("10")), silence.reach(spans, D("10"))),
            lambda: silence.paired_bound(spans, D("10"), D("0.9")),
            lambda: silence.kept_starts(spans, D("10"), D("0.25"), "message"),
            lambda: silence.searched_bound(D("30.5"), 3, D("0.9")),
            lambda: silence.gap_given(3, D("20.02"), D("11.5")),
            lambda: silence.expected(D("200.5"), [(D("150"), D("0.1")), (D("150"), D("0.3"))]),
            lambda: calibration.at_one_logit(D("2"), D("0.7")),
            lambda: case.arrival(3, D("30.5")),
            lambda: case.arrival(0, D("30.5")),
            lambda: h9.bucket_quantile([D("0.3"), D("0.7"), D("0.9")], D("0.5"), [D("0.5"), D("1")]),
            lambda: budget.bracket([D("100")] * 19 + [D("210")], [D("50"), D("100"), D("200")]),
            lambda: self.sketched(),
        ]

    @staticmethod
    def sketched():
        sketch = h9.LogSketch(0.01)
        for x in (D("2.5"), D("0.004"), D("17")):
            sketch.add(x)
        return sketch.n, sorted(sketch.counts.items())

    def test_decimals_alone_under_every_trap(self):
        usual = self.same(self.calls())
        for i, got in enumerate(usual):
            self.assertFalse(isinstance(got, str) and "Error" in got.split(":")[0], (i, got))
        self.assertEqual(usual[14], expected("ch22_record_check"))
        self.assertEqual(usual[0], stats.quiet_bound(30))                  # one number, however it is written
        self.assertEqual(usual[34], case.arrival(0, 30.5))
        self.assertEqual(usual[30], float(4 * (1 - F("11.5") / F("20.02")) ** 3))
        self.assertEqual(usual[31], F(15) + F("50.5") * F(3, 10))

    def test_what_is_no_finite_number_is_refused_in_one_way(self):
        # A decimal NaN in a comparison raises decimal.InvalidOperation or gives False as the trap of the caller
        # says, and Python compares a decimal infinity with the float math.inf under no trap but one.
        nan, inf = D("NaN"), D("Infinity")
        calls = []
        for bad in (nan, inf, -inf, D("sNaN")):
            calls += [
                lambda bad=bad: stats.quiet_bound(bad),
                lambda bad=bad: stats.quiet_bound(D(30), bad),
                lambda bad=bad: stats.rate_limits(2, bad),
                lambda bad=bad: stats.as_level(bad),
                lambda bad=bad: stats.poisson_within(2, bad),
                lambda bad=bad: stats.binomial_within(2, 20, bad),
                lambda bad=bad: stats.wilson(3, 20, bad),
                lambda bad=bad: stats.sum_within([(1, bad)]),
                lambda bad=bad: core.u_rect(bad),
                lambda bad=bad: core.rect_interval(bad, D(1)),
                lambda bad=bad: core.rect_interval(D(1), D(1), bad),
                lambda bad=bad: core.welch_satterthwaite([D(1), bad], [3, 4]),
                lambda bad=bad: core.combine(D(1), bad),
                lambda bad=bad: core.Result(D(1), bad, "ms"),
                lambda bad=bad: core.Result(bad, D(1), "ms"),
                lambda bad=bad: core.Budget("ms").add("a", "B", bad) and None,
                lambda bad=bad: core.decimals(bad),
                lambda bad=bad: core.fixed(bad, 2),
                lambda bad=bad: core.Result(D(1), D(1), "ms").prob_below(bad),
                lambda bad=bad: propagate.lpu(lambda t: t[0], [D(1)], [bad]),
                lambda bad=bad: silence.starts([(0, 30)], bad),
                lambda bad=bad: silence.gap_terms(bad, 20, 1),
                lambda bad=bad: silence.gap_terms(D("0.4"), bad, 1),
                lambda bad=bad: silence.gap_given(2, bad, 1),
                lambda bad=bad: silence.gap_given(2, 20, bad),
                lambda bad=bad: silence.expected(bad),
                lambda bad=bad: silence.expected(10, [(bad, 1)]),
                lambda bad=bad: silence.expected(10, [(1, bad)]),
                lambda bad=bad: silence.searched_bound(bad, 3),
                lambda bad=bad: silence.kept_starts([(0, 30)], 10, bad, "message"),
                lambda bad=bad: case.arrival(3, bad),
                lambda bad=bad: calibration.at_one_logit(bad, D("0.7")),
                lambda bad=bad: calibration.at_one_logit(D(2), bad),
            ]
        usual = self.same(calls)
        per = len(calls) // 4
        self.assertEqual(per, 33)
        # combine() returns what the floats give, decimals() has no place to round to, fixed() prints the
        # number, and an infinite limit has a probability. Every other call refuses all four, as a
        # ValueError: Version 72 raised decimal.InvalidOperation for a NaN under the usual context. An infinite
        # time of expected() is no fraction, and is refused as Python refuses to make one of it.
        answers = (12, 16, 17, 18)
        for k in range(4):
            for i in range(per):
                got = usual[k * per + i]
                if i not in answers or k == 3:                             # a signaling NaN is refused everywhere
                    self.assertIsInstance(got, str, (k, i))
                    kinds = ("ValueError", "OverflowError") if i == 25 else ("ValueError",)
                    self.assertIn(got.split(":")[0], kinds, (k, i, got))
        # A decimal NaN and a decimal infinity are refused in the words of the float.
        self.assertEqual(usual[0], refusal(stats.quiet_bound, float("nan")))
        self.assertEqual(usual[per + 0], refusal(stats.quiet_bound, float("inf")))
        self.assertEqual(usual[21], refusal(silence.gap_terms, float("nan"), 20, 1))
        self.assertEqual(usual[per + 30], refusal(case.arrival, 3, float("inf")))
        self.assertEqual(usual[16], 3)                                     # decimals(): no place to round to
        self.assertEqual(core.decimals(float("nan")), 3)
        self.assertEqual((usual[17], usual[per + 17], usual[2 * per + 17]), ("NaN", "Infinity", "-Infinity"))
        self.assertEqual((usual[per + 18], usual[2 * per + 18]), (1.0, 0.0))   # an infinite limit

    def test_a_decimal_beside_a_float_is_the_one_refusal_of_the_caller(self):
        # Python compares a decimal with a float exactly. Under the trap FloatOperation it refuses to, and a
        # list that holds both ends there: that is what the caller has asked for. With decimals alone, or with
        # floats alone, the trap changes nothing.
        mixed = [(D("1.5"), 1), (2.5, 1), (D("3.5"), 1)]
        self.assertEqual(stats.weighted_quantile(mixed, 0.5), 2.5)
        with decimal.localcontext(hostile()):
            self.assertEqual(refusal(stats.weighted_quantile, mixed, 0.5),
                             "FloatOperation: [<class 'decimal.FloatOperation'>]")
            self.assertEqual(stats.weighted_quantile([(D("1.5"), 1), (D("2.5"), 1), (D("3.5"), 1)], D("0.5")),
                             D("2.5"))
            self.assertEqual(stats.weighted_quantile([(1.5, 1), (2.5, 1), (3.5, 1)], 0.5), 2.5)

    def test_the_examples_under_every_trap(self):
        # Every example that is replayed exactly, in a process of its own in which the context of three digits
        # with all nine traps is set before anything of the repository is imported.
        names = sorted(path.stem for path in (ROOT / "examples").glob("ch*.py")
                       if path.read_text(encoding="utf-8").startswith("# replay: exact"))
        self.assertEqual(len(names), 39)
        code = (
            "import contextlib, decimal, importlib, io, pathlib\n"
            "signals = list(decimal.getcontext().flags)\n"
            "assert decimal.FloatOperation in signals and len(signals) == 9\n"
            "decimal.setcontext(decimal.Context(prec=3, Emax=10, Emin=-10, rounding=decimal.ROUND_UP,\n"
            "                                   traps=signals))\n"
            f"for name in {names!r}:\n"
            "    out = io.StringIO()\n"
            "    with contextlib.redirect_stdout(out):\n"
            "        importlib.import_module('examples.' + name).main()\n"
            "    want = pathlib.Path('expected', name + '.txt').read_text(encoding='utf-8')\n"
            "    print(name, out.getvalue() == want)\n")
        out = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr[-400:])
        self.assertEqual(out.stdout.split(), [word for name in names for word in (name, "True")])


class ALongDecimal(Record):
    """Section "A valid long decimal can be decided but cannot be displayed"."""

    def setUp(self):
        super().setUp()
        self.addCleanup(sys.set_int_max_str_digits, sys.get_int_max_str_digits())

    def test_the_record_of_the_review(self):
        digits, value, exactly = long_decimal(4301)                        # 0.2 and 10 to the power -4301
        rec = self.one(value, D(0), D(1), D("0.25"), D("0.2"), unit="s")
        literal = read('{"value": %s}' % digits)["value"]
        self.assertEqual((type(literal), literal), (D, value))
        self.assertEqual(decide(rec), ("reject", []))
        (lo, hi), (low, top) = guard(rec)                                  # Version 72: ValueError
        self.assertEqual({F(lo), F(hi), F(low), F(top)}, {exactly})
        self.assertEqual(self.band(rec), digits)                           # above its limit, in 4301 places
        self.assertEqual(F(in_unit(rec, "ms")[0]), 1000 * exactly)
        self.assertEqual(str(in_unit(rec, "s")[0]), digits)

    def test_no_whole_number_is_printed_as_a_text(self):
        # The limit of 4300 digits can be set lower, down to 640. Nothing here depends on it.
        digits, value, exactly = long_decimal(5000, "1.25")
        for most in (640, 4300, 0):
            sys.set_int_max_str_digits(most)
            rec = self.one(value, D(0), D(1), D("0.25"), D("1.25"))
            self.assertEqual(self.band(rec), digits, most)
            self.assertEqual(str(record.rounded(value, 4999)), "1.25" + "0" * 4997, most)
            self.assertEqual(str(record.rounded(value, 1)), "1.3", most)   # above the tie, which is 1.25
            self.assertEqual(str(record.rounded(D("1.25"), 1)), "1.2", most)
            self.assertEqual(stats.upward(value, 2), "1.26", most)
            self.assertEqual(stats.downward(value, 5000), digits, most)
            self.assertEqual(stats.nearest(value, 5001), digits + "0", most)
            # Downward, two places would put the lower end on the limit that the value is above.
            self.assertEqual(stats.enclosed(value, value, 2, D("1.25")), (digits, "1.26"), most)
            whole = 10 ** 5000 + 1                                         # a whole number of 5001 digits
            self.assertEqual(stats.downward(whole, 0), "1" + "0" * 4999 + "1", most)
            self.assertEqual(stats.upward(-whole, 1), "-1" + "0" * 4999 + "1.0", most)
            self.assertEqual(core.fixed(whole, 0), "1" + "0" * 4999 + "1", most)
            # A quarter of it: 25 and 4998 zeros, 5000 digits, and a quarter.
            self.assertEqual(core.fixed(F(whole, 4), 2), "25" + "0" * 4998 + ".25", most)
            self.assertEqual(stats.tens(whole, -5000), D("1." + "0" * 4999 + "1"), most)
            self.assertEqual(core.full(whole), "1" + "0" * 4999 + "1", most)
            self.assertEqual(core.full(F(whole, 3)), "1" + "0" * 4999 + "1/3", most)    # 10 ** 5000 + 1 is 2 mod 3
            self.assertEqual(repr(units.Q(-whole, "ns")), "-1" + "0" * 4999 + "1 ns", most)
        sys.set_int_max_str_digits(640)
        self.assertTrue(refusal(str, 10 ** 700).startswith("ValueError: Exceeds the limit"))

    def test_a_root_of_long_numbers(self):
        # One degree of freedom at a risk of 0.25: the factor is 1 and the band is the root itself. Two
        # components of 3 and 4 units of a place far down combine to 5 of them, exactly.
        sys.set_int_max_str_digits(640)
        a = 1 + F(1, 10 ** 900)                                            # a unit of 901 digits

        def times(n):
            return D(f"{n}." + "0" * 899 + f"{n}")
        rec = self.one(D(100), times(5), D(1), D("0.25"), D(200))
        rec["components"] = [dict(rec["components"][0], u=times(3)), dict(rec["components"][0], u=times(4))]
        self.assertEqual(decide(rec), ("accept", []))
        (lo, hi), (low, top) = guard(rec)                                  # Version 72, at this limit: ValueError
        self.assertEqual((F(lo), F(hi), F(low), F(top)), (100 + 5 * a, 100 + 5 * a, 100 - 5 * a, 100 - 5 * a))
        # A root that is no fraction is enclosed: two components of one unit combine to the root of 2 units.
        rec = self.one(D(100), D("1.4142135623731"), D(1), D("0.25"), D(200))
        rec["components"] = [dict(rec["components"][0], u=times(1)), dict(rec["components"][0], u=times(1))]
        self.assertEqual(decide(rec), ("accept", []))
        for lo, hi in (map(F, record.spread(rec)), (F(x) - 100 for x in guard(rec)[0])):
            self.assertLess(lo ** 2, 2 * a ** 2)
            self.assertGreater(hi ** 2, 2 * a ** 2)
            self.assertLessEqual(hi - lo, F(1, 10 ** 39))


class GapsAndExpectedMessages(unittest.TestCase):
    """Section "gap_given() and expected() accept inconsistent inputs"."""

    def test_a_gap_of_more_than_half_the_record(self):
        # n uniform times leave n + 1 gaps, each of which is at least the share g of the record with the
        # probability (1 - g) ** n. Two gaps cannot both be more than half of it, so for g above one half the
        # chance of some such gap is (n + 1) (1 - g) ** n.
        for n in (0, 1, 2, 5, 12):
            for total, length in ((2, 1.5), (D("3"), D("2")), (F(7, 2), F(2)), (D("20.02"), D("11.5")),
                                  (FOUR_BYTES, 0.6), (10 ** 30, 10 ** 30 - 1)):
                g = F(length) / F(total)
                want = float(min(F(1), (n + 1) * (1 - g) ** n))
                self.assertEqual(silence.gap_given(n, total, length), want, (n, total, length))
        self.assertEqual(silence.gap_given(1, 2, 1), 1.0)                  # one time leaves a gap of half or more

    def test_what_is_no_time_is_refused(self):
        for total, length in ((2, True), (True, 1), (2, "1"), ("2", 1), (None, 1), (2, 1 + 0j)):
            got = refusal(silence.gap_given, 1, total, length)             # Version 72: 1.0 for the first
            self.assertTrue(got.startswith("TypeError: no number"), (total, length, got))
        for total, length in ((2, 0), (2, -1), (1, 2), (math.inf, 1), (2, math.nan), (math.nan, 1), (0, 0)):
            self.assertEqual(refusal(silence.gap_given, 1, total, length), "ValueError: no room for the gap")
        self.assertEqual(refusal(silence.gap_given, True, 2, 1), "ValueError: need a whole n >= 0")

    @unittest.skipIf(numpy is None, "NumPy is not installed")
    def test_times_of_four_bytes(self):
        four = numpy.float32
        g = F(float(four(1.7))) / F(float(four(2.9)))
        self.assertEqual(silence.gap_given(3, four(2.9), four(1.7)), float(4 * (1 - g) ** 3))   # Version 72: TypeError
        self.assertEqual(silence.expected(four(100.5)), F(float(four(100.5))) / 10)
        self.assertEqual(silence.expected(100, [(four(150), four(0.25))]), 25)
        self.assertEqual(silence.expected(numpy.int8(100)), 10)

    def test_messages_expected_of_a_rate_in_pieces(self):
        # 150 s at 0.1 per second and 150 s at 0.3 per second.
        for t, want in ((0, 0), (100, 10), (150, 15), (200, 30), (300, 60), (1000, 60), (-5, 0),
                        (D("100.5"), F(1005, 100)), (F(1, 3), F(1, 30)), (0.5, F(1, 20)), (10 ** 40, 60)):
            got = silence.expected(t)
            self.assertEqual(got, want, t)
            self.assertIs(type(got), F)
        pieces = [(D("1.5"), D("0.25")), (2, F(1, 3)), (0.5, 0)]
        self.assertEqual(silence.expected(D("2.5"), pieces), F(3, 8) + F(1, 3))   # Version 72: TypeError
        self.assertEqual(silence.expected(10, pieces), F(3, 8) + F(2, 3))
        self.assertEqual(silence.expected(1, [(1, D("1E-400"))]), F(1, 10 ** 400))

    def test_what_is_no_piece_is_refused(self):
        for t in ("100", True, None):
            self.assertTrue(refusal(silence.expected, t).startswith("TypeError: no number"), t)   # 72: 10 for the text
        for t in (math.nan, math.inf, -math.inf, D("NaN"), D("Infinity"), D("sNaN")):     # no time, as elsewhere
            self.assertIn(refusal(silence.expected, t).split(":")[0], ("ValueError", "OverflowError"), t)
        for piece in ((1, -1), (1, D("-0.5")), (0, 1), (-1, 1), (math.inf, 1), (1, math.inf), (1, math.nan),
                      (math.nan, 1)):
            self.assertEqual(refusal(silence.expected, 1, [piece]), "ValueError: no such piece", piece)   # 72: -1 first
        for piece in ((1, True), ("1", 1), (1, "0.5"), (None, 1)):
            self.assertTrue(refusal(silence.expected, 1, [piece]).startswith("TypeError: no number"), piece)

    def test_the_transcript(self):
        self.assertEqual(printed(silence.main), expected("ch16_silence"))


class AProbabilityThatIsNotPlaced(unittest.TestCase):
    """Section "The probability returned at the precision ceiling can be the wrong float"."""

    def test_the_sum_of_the_review(self):
        # One half and 2 to the power -54 lies halfway between the float 0.5 and the next one. The second term,
        # e to the power -120000, is above zero and puts the sum above the middle, but it is below every power
        # of ten that 5120 digits enclose, so the two ends stay on either side.
        middle = F(1, 2) + F(1, 2 ** 54)
        self.assertEqual(middle, (F(0.5) + F(math.nextafter(0.5, 1))) / 2)
        self.assertLess(-120000 / math.log(10), -10 * stats.DEEPEST)       # e ** -120000 < 10 ** -51200
        terms = [(middle, 0), (1, 120000)]
        self.assertEqual(refusal(stats.chance, terms), "ValueError: not decided in 5120 digits")   # 72: 0.5

    def test_a_sum_that_is_placed(self):
        above = math.nextafter(0.5, 1)
        middle = F(1, 2) + F(1, 2 ** 54)
        # With e to the power -100, about 3.7e-44, the sum is above the middle by more than forty digits show.
        self.assertEqual(stats.chance([(middle, 0), (1, 100)]), above)
        self.assertEqual(stats.chance([(middle, 0), (-1, 100)]), 0.5)
        self.assertEqual(stats.chance([(F(1, 2), 0)]), 0.5)
        self.assertEqual(stats.chance([(1, 0), (-1, 1)]), float(C.subtract(D(1), C.exp(D(-1)))))
        self.assertEqual(stats.chance([(F(1, 3), D("0.25")), (F(1, 7), 2)]),
                         float(C.add(C.divide(C.exp(D("-0.25")), D(3)), C.divide(C.exp(D(-2)), D(7)))))
        self.assertEqual(stats.chance([(2, 0)]), 1.0)                      # not above 1; Version 72: 2.0
        self.assertEqual(stats.chance([(-1, 0)]), 0.0)                     # not below 0
        self.assertEqual(silence.gap_somewhere(0.4, 20, 1), stats.chance(silence.gap_terms(0.4, 20, 1)))


class ALimitThatIsNoDecimal(unittest.TestCase):
    """Section "A decimal display cannot preserve equality to every rational limit"."""

    def test_a_third_on_a_limit_of_a_third(self):
        third = F(1, 3)
        self.assertEqual(stats.nearest(third, 2, third), "0.33")           # below the limit: no text is on it
        self.assertEqual(stats.enclosed(third, third, 2, third), ("0.33", "0.34"))
        self.assertEqual((stats.upward(third, 2, third), stats.downward(third, 2, third)), ("0.34", "0.33"))
        self.assertEqual(stats.upward(third, 0, third), "1")

    def test_a_decimal_on_its_limit_is_printed_on_it(self):
        for x in (F(1, 8), D("200.003"), F(3, 2 ** 20), 0.1, 12, F(7, 5 ** 9)):
            for way in (stats.upward, stats.downward, stats.nearest):
                got = way(x, 2, x)
                self.assertEqual(F(got), F(x), (x, way.__name__))
            low, top = stats.enclosed(x, x, 2, x)
            self.assertEqual((F(low), F(top)), (F(x), F(x)))

    def test_beside_a_limit_that_is_no_decimal(self):
        third = F(1, 3)
        ways = ((stats.upward, "up"), (stats.downward, "down"), (stats.nearest, "nearest"))
        for x in (F(33, 100), F(333, 1000), F(3334, 10000), F(34, 100), third - F(1, 10 ** 9),
                  third + F(1, 10 ** 9), F(1, 4), F(2, 3), 0.3333333333333333, D("0.33333333333333333334")):
            for way, name in ways:
                got = way(x, 2, third)
                self.assertEqual(got, reference(x, 2, third, name), (x, name))
                self.assertEqual((F(got) > third, F(got) < third), (F(x) > third, F(x) < third), (x, name))
        # Upward, 0.333 would print as 0.34, above the limit that it is below, and takes a third place.
        self.assertEqual((stats.upward(F(333, 1000), 2, third), stats.nearest(third + F(1, 10 ** 9), 2, third)),
                         ("0.333", "0.333333334"))


class TheReplayComparesDecimals(unittest.TestCase):
    """Section "Exact-mode replay can miss a change across a decision limit"."""

    def test_the_two_numbers_of_the_review(self):
        above, below = "200.00000000000000004", "199.99999999999999998"
        self.assertEqual(float(above), float(below))                       # one float, on a limit of 200
        diffs = run_all.compare("exact", f"sum {above}\n", f"sum {below}\n")
        self.assertEqual(diffs, [f"line 1: {above} expected, {below} produced"])   # Version 72: no difference

    def test_one_unit_of_the_last_place_and_no_more(self):
        for old, new, same in (("0.10000000000000000001", "0.10000000000000000002", True),
                               ("0.10000000000000000001", "0.10000000000000000003", False),
                               ("0.105", "0.106", True), ("0.105", "0.104", True), ("0.105", "0.107", False),
                               ("9.99", "10.00", True), ("-0.01", "0.00", True), ("-0.01", "0.01", False),
                               ("1.5e-320", "1.6e-320", True), ("1.5e-320", "1.7e-320", False),
                               ("1.5e400", "1.6e400", True), ("1.5e400", "1.7e400", False),
                               ("5800", "5800", True), ("5800", "5801", False),
                               ("9007199254740993", "9007199254740992", False)):
            diffs = run_all.compare("exact", f"x {old}\n", f"x {new}\n")
            self.assertEqual(diffs == [], same, (old, new, diffs))
        self.assertEqual(float("9007199254740993"), float("9007199254740992"))
        self.assertEqual(run_all.last_place("1.50e-3"), F(1, 10 ** 5))
        self.assertEqual(run_all.value("-1.50e-3"), F(-3, 2000))

    def test_white_space_is_text(self):
        for old, new in (("a  b 1.5", "a b 1.5"), ("a b 1.5", "a b 1.5 "), ("  x 2", " x 2"), ("x\t2", "x 2"),
                         ("n =  9.99", "n = 10.00")):
            diffs = run_all.compare("exact", old + "\n", new + "\n")
            self.assertEqual(len(diffs), 1, (old, new))                    # Version 72: none
            self.assertIn("text differs", diffs[0])
            self.assertEqual(run_all.compare("live", old + "\n", new + "\n"), [])
        self.assertEqual(run_all.compare("exact", "a  b 1.5 \n", "a  b 1.5 \n"), [])

    def test_a_long_whole_number(self):
        self.addCleanup(sys.set_int_max_str_digits, sys.get_int_max_str_digits())
        sys.set_int_max_str_digits(640)
        long = "9" * 5000
        self.assertEqual(run_all.compare("exact", f"n {long}\n", f"n {long}\n"), [])
        self.assertEqual(len(run_all.compare("exact", f"n {long}\n", f"n {long[:-1]}8\n")), 1)
        self.assertEqual(run_all.compare("exact", f"n {long}.5\n", f"n {long}.6\n"), [])

    def test_every_transcript_is_its_own_replay(self):
        names = sorted(path.stem for path in (ROOT / "expected").glob("ch*.txt"))
        self.assertEqual(len(names), 43)
        for name in names:
            text = expected(name)
            mode = run_all.mode_of(ROOT / "examples" / f"{name}.py")
            self.assertEqual(run_all.compare(mode, text, text), [], name)


class BeyondTheReview(Record):
    def test_a_number_for_a_comparison(self):
        for x, want in ((0.1, F(0.1)), (D("0.1"), F(1, 10)), (7, F(7)), (F(2, 3), F(2, 3)), (10 ** 400, F(10 ** 400)),
                        (D("1E-5000"), F(1, 10 ** 5000)), (FOUR_BYTES, F(FOUR_BYTES))):
            got = stats.number(x)
            self.assertEqual((type(got), got), (F, want), x)
        for x in (math.inf, -math.inf, D("Infinity"), D("-Infinity")):
            got = stats.number(x)
            self.assertEqual((type(got), got), (float, float(x)))
        for x in (math.nan, D("NaN")):
            got = stats.number(x)
            self.assertTrue(type(got) is float and got != got)
        self.assertEqual(refusal(stats.number, D("sNaN")), "ValueError: cannot convert signaling NaN to float")
        for x in (True, "1", None, 1 + 0j):
            self.assertTrue(refusal(stats.number, x).startswith("TypeError: no number"), x)

    def test_a_flag_or_a_text_in_three_more_places(self):
        for call in (lambda: core.resolved(True, 1), lambda: core.resolved("1", 1), lambda: dist.t_sf(True, 4),
                     lambda: dist.t_cdf("1", 4), lambda: dist.t_cdf(None, 4), lambda: stats.tens(True, 0),
                     lambda: stats.tens("15", 0), lambda: stats.tens(D(5), 0), lambda: stats.tens(1.0, 0)):
            self.assertTrue(refusal(call).startswith("TypeError"), refusal(call))     # Version 72 took the first three
        # With one degree of freedom the distribution is Cauchy's: a quarter lies above 1 and a quarter below -1.
        self.assertAlmostEqual(dist.t_sf(1, 1), 0.25, delta=1e-15)
        self.assertAlmostEqual(dist.t_cdf(F(-1), 1), 0.25, delta=1e-15)
        self.assertAlmostEqual(dist.t_cdf(D("-1"), 1), 0.25, delta=1e-15)
        # At 1e18 a float steps by 128, and a spread is resolved from a thousand steps on.
        self.assertEqual(math.ulp(1e18), 128.0)
        self.assertIsNone(core.resolved(1e18, 128000))
        self.assertIsNone(core.resolved(D("1E+18"), D("128000")))
        self.assertEqual(refusal(core.resolved, 1e18, F(127999)), "ValueError: too fine for a float of this size")

    def test_a_decimal_from_its_digits(self):
        for n, e in ((0, 0), (0, -2), (5, -2), (-12, 3), (10 ** 5000 + 1, -5000), (-(10 ** 700), 700)):
            with decimal.localcontext(hostile()):
                got = stats.tens(n, e)
            self.assertEqual(F(got), n * F(10) ** e, (n, e))
            self.assertEqual(got.as_tuple().exponent, e)

    def test_combine_and_the_tails_with_decimals(self):
        us = D("1.2345"), D("2.3456"), D("0.000123456789")
        want = stats._root(sum(F(u) ** 2 for u in us))
        with decimal.localcontext(hostile()):                              # three digits, every trap
            got = core.combine(*us)                                        # Version 72: decimal.Inexact, from abs()
        self.assertAlmostEqual(got / want, 1, delta=1e-15)
        self.assertEqual(got, core.combine(*map(float, us)))
        with decimal.localcontext(hostile()):
            self.assertEqual(dist.t_cdf(D("-1.5"), 4), dist.t_cdf(-1.5, 4))    # Version 72: FloatOperation
            self.assertEqual(dist.t_sf(D("1.5"), 4), dist.t_sf(1.5, 4))
            self.assertEqual(dist.betainc(D(2), D("0.5"), D("0.3")), dist.betainc(2.0, 0.5, 0.3))
        self.assertEqual(dist.t_sf(D("Infinity"), 4), 0.0)
        self.assertAlmostEqual(dist.t_cdf(D("0"), 4), 0.5, delta=1e-15)

    def test_bounds_of_buckets_in_decimals(self):
        values = [D("0.3"), D("0.7"), D("0.9")]
        with decimal.localcontext(hostile()):
            # Rank 1.5 of 3 lies in the second bucket, 0.5 to 1, which holds two values after one below it.
            self.assertEqual(h9.bucket_quantile(values, D("0.5"), [D("0.5"), D("1")]), 0.5 + 0.5 * 0.25)
            self.assertEqual(budget.bracket([D(100)] * 19 + [D(210)], [D(50), D(100), D(200)]), (D(50), D(100)))
            for bad in ([D("NaN"), D(100)], [D("-Infinity"), D(50)], [D(100), D(50)], [D(50), D(50)]):
                self.assertEqual(refusal(budget.bracket, [D(100)] * 20, bad), "ValueError: bounds do not rise")
        self.assertEqual(budget.bracket([100.0] * 20, [50, 100, math.inf]), (50, 100))
        self.assertTrue(refusal(budget.bracket, [100.0] * 20, [True, 50]).startswith("TypeError: no number"))

    def test_the_interval_of_the_stored_record(self):
        # A lower limit is rounded downward and an upper one upward, to a tenth of a millisecond.
        _, result = budget.from_raw(budget.record(random.Random(22)))
        lo, hi = result.interval(0.95)
        want = [math.floor(10 * lo) / 10, math.ceil(10 * hi) / 10]
        self.assertEqual([float(x) for x in self.rec["result"]["interval_95"]], want)
        self.assertLessEqual(F(str(want[0])), lo)
        self.assertGreaterEqual(F(str(want[1])), hi)


if __name__ == "__main__":
    unittest.main()
