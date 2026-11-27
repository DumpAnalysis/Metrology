"""Regression tests added after the review of Version 13.

The review found a false acceptance for a risk just below one half: the
risk 0.49999999999999999999 became the float 0.5 inside the quantile
calculation, the coverage factor became zero, and a record at its limit
with a positive uncertainty was accepted. The factor is now computed from
the distance of the risk from one half, which the checker keeps exactly.
The review also found that a result with zero uncertainty was printed to
three decimal places whatever its value, and asked for the condition under
which a timed-out request can be ranked after the completed ones, for a
worked case with three kinds of evidence, and for the calibration model of
Chapter 12 to be stated.

The records are complete files, opened with the loader printed in
Listing 22.4. Each test computes what it expects independently of the code
under test: closed forms of the t quantile for one and two degrees of
freedom, a second method on the other side of a threshold, enumeration of
every possible outcome, a fit repeated with shifted references.
"""
import itertools
import json
import math
import random
import tempfile
import unittest
from decimal import Context, Decimal
from fractions import Fraction
from pathlib import Path
from statistics import NormalDist

from examples import ch12_calibration, ch16_bounds, ch22_record_check
from swmetro import dist
from swmetro import record as checker
from swmetro.core import Result, fmt
from swmetro.dist import t_isf, t_mid
from swmetro.record import decide, guard, read
from swmetro.stats import ecdf_quantile, ecdf_rank, linfit

ROOT = Path(__file__).resolve().parent.parent
STORED = (ROOT / "data" / "measurement_record.json").read_text(encoding="utf-8")
FACTOR = ["at the limit: the coverage factor decides"]
NINES = "0.49999999999999999999"     # the risk of the review: one half less 1e-20


def file_text(value, limit, u="1", dof="1", risk="0.05"):
    """The text of a complete record file with one component, in which the
    given tokens stand exactly as they are passed."""
    rec = json.loads(STORED)
    rec["components"] = [{"name": "c0", "type": "B", "kind": "stipulated", "u": "@u@"}]
    rec["result"].update(value="@value@", u="@stated@", dof="@dof@")
    rec["decision"].update(limit="@limit@", risk="@risk@")
    text = json.dumps(rec, indent=1)
    for key, token in {"value": value, "stated": u, "u": u, "dof": dof, "limit": limit, "risk": risk}.items():
        assert text.count(f'"@{key}@"') == 1, key
        text = text.replace(f'"@{key}@"', token)
    return text


def opened(text):
    """The record of a file with this text, opened with the printed loader."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "record.json"
        path.write_text(text, encoding="utf-8")
        return ch22_record_check.load(str(path))


def decimal_text(q, places=60):
    """A fraction as a decimal token with `places` decimals, every one of them right:
    the default context of 28 digits would round a risk of thirty nines to one half."""
    exact = Context(prec=places + 50).divide(Decimal(q.numerator), Decimal(q.denominator))
    return f"{exact:.{places}f}"


class RiskNextToOneHalf(unittest.TestCase):
    def test_the_record_of_the_review(self):
        self.assertEqual(float(NINES), 0.5)                       # what a float makes of the risk
        self.assertEqual(Fraction(1, 2) - Fraction(NINES), Fraction(1, 10 ** 20))
        rec = opened(file_text("100", "100", risk=NINES))
        self.assertEqual(checker.invalid(rec), [])
        # A positive band on both sides of the limit: neither accepted nor rejected.
        self.assertEqual(decide(rec), ("undecided", []))
        # At the usual risk the same record was already undecided.
        self.assertEqual(decide(opened(file_text("100", "100", risk="0.05"))), ("undecided", []))
        # With no uncertainty there is no band, and equality is accepted at either risk.
        for risk in (NINES, "0.05"):
            self.assertEqual(decide(opened(file_text("100", "100", u="0", risk=risk))), ("accept", []))

    def test_the_factor_of_the_review(self):
        # One degree of freedom: the upper-tail factor is tan(pi (1/2 - risk)).
        want = math.tan(math.pi * 1e-20)
        self.assertAlmostEqual(want / 3.14159e-20, 1.0, places=5)
        k, low, high = checker._factor(Fraction(NINES), Fraction(1))
        self.assertAlmostEqual(float(k) / want, 1.0, places=12)
        self.assertTrue(0 < low < k < high)
        (lo, hi), (down, up) = guard(opened(file_text("100", "100", risk=NINES)))
        self.assertTrue(down < 100 < lo)                          # the two ends of the review
        self.assertAlmostEqual(float(lo - 100) / want, 1.0, places=12)
        self.assertAlmostEqual(float(100 - down) / want, 1.0, places=12)

    def test_the_sign_settles_equality_at_every_supported_risk(self):
        for n in (1, 2, 6, 9, 12, 16, 17, 20, 30, 100, 299, 300):
            risk = decimal_text(Fraction(1, 2) - Fraction(1, 10 ** n), 310)
            for dof in ("1", "2", "15.7", "1e7"):
                rec = opened(file_text("100", "100", dof=dof, risk=risk))
                self.assertEqual(decide(rec), ("undecided", []), (n, dof))
                rec = opened(file_text("100", "100", u="0", dof=dof, risk=risk))
                self.assertEqual(decide(rec), ("accept", []), (n, dof))

    def test_a_small_factor_still_decides_on_both_sides(self):
        # One degree of freedom, u = 1: the band is tan(pi gap), about pi gap.
        for n in (6, 12, 20, 40):
            gap = Fraction(1, 10 ** n)
            risk = decimal_text(Fraction(1, 2) - gap)
            band = Fraction(math.tan(math.pi * float(gap)))
            for part, want in ((Fraction(2), "accept"), (Fraction(101, 100), "accept"), (Fraction(99, 100), "undecided"),
                               (Fraction(1, 2), "undecided"), (Fraction(0), "undecided"), (Fraction(-99, 100), "undecided"),
                               (Fraction(-101, 100), "reject"), (Fraction(-2), "reject")):
                value = decimal_text(Fraction(100) - part * band)       # headroom = part x band
                rec = opened(file_text(value, "100", risk=risk))
                self.assertEqual(decide(rec), (want, []), (n, float(part)))

    def test_closed_forms_from_the_gap(self):
        inside = float(checker.DELTA) / 100
        for gap in (1e-300, 1e-200, 1e-100, 1e-20, 1e-12, 1e-9, 1e-6, 1e-3, 0.01, 0.1, 0.2, 0.2499):
            one = math.tan(math.pi * gap)                               # 1 degree of freedom
            two = 2 * gap * math.sqrt(2) / math.sqrt(1 - 4 * gap * gap)  # 2: P(0 < T < t) = t / (2 sqrt(2 + t^2))
            self.assertLess(abs(t_mid(gap, 1) / one - 1), inside, gap)
            self.assertLess(abs(t_mid(gap, 2) / two - 1), inside, gap)
            # Many degrees of freedom: the normal quantile, checked through erf.
            z = t_mid(gap, math.inf)
            self.assertLess(abs(0.5 * math.erf(z / math.sqrt(2)) / gap - 1), 1e-13, gap)
            # In between: the probability of the quantile found is the gap asked for. (Below a gap
            # of 1e-100 the square of the quantile is beyond a float, and t_mid works by proportion.)
            for dof in (0.001, 0.5, 4.79, 15.7, 300.0, 9.9e5):
                try:
                    t = t_mid(gap, dof)
                except ValueError:       # with 0.001 degrees of freedom the quantile at 0.7 is beyond 1e150
                    self.assertTrue(dof == 0.001 and gap >= 0.2, (gap, dof))
                    continue
                if gap >= 1e-100:
                    self.assertLess(abs(dist._centre(t, dof) / gap - 1), 1e-12, (gap, dof))
                else:
                    self.assertAlmostEqual(t / t_mid(1e-100, dof), gap / 1e-100, delta=gap * 1e88, msg=(gap, dof))

    def test_two_methods_meet_at_a_quarter(self):
        # Up to a tail of 1/4 the quantile comes from the upper tail, above it from the gap.
        for dof in (0.5, 1, 2, 4.79, 15.7, 300.0, 1e5, 9.9e5, 2e6):
            below, above = t_isf(0.25, dof), t_mid(0.25 * (1 - 1e-13), dof)
            self.assertAlmostEqual(above / below, 1.0, places=8, msg=dof)
            # The upper tail and the centre add up to one half.
            for t in (1e-12, 1e-3, 0.5, 3.0):
                self.assertAlmostEqual(dist.t_sf(t, dof) + dist._centre(t, dof), 0.5, places=9, msg=(t, dof))
        self.assertEqual(t_isf(0.4, 5.0), t_mid(0.5 - 0.4, 5.0))
        self.assertAlmostEqual(t_mid(0.25, 1.0), 1.0, places=14)

    def test_many_degrees_of_freedom_next_to_one_half(self):
        for gap in (1e-300, 1e-30, 1e-9, 1e-4, 9e-4, 1.1e-3, 0.01, 0.2):
            z = t_mid(gap, math.inf)
            if gap > 1e-3:
                self.assertEqual(z, NormalDist().inv_cdf(0.5 + gap))
            self.assertAlmostEqual(z / (math.sqrt(2 * math.pi) * gap), 1.0, delta=1.2 * gap * gap + 1e-15)
            self.assertGreater(t_mid(gap, 1e7), z)                      # heavier tails, larger quantile
            self.assertAlmostEqual(t_mid(gap, 1e7) / z, 1.0, places=6)
        # The series and the library quantile agree where they change over.
        self.assertAlmostEqual(dist._z_mid(1e-3) / dist._z_mid(1e-3 * (1 + 1e-12)), 1.0, places=11)

    def test_the_allowance_is_one_part_in_a_million_at_every_risk(self):
        for risk, dof in (("0.05", 10), ("0.3", 10), ("0.49", 10), ("0.4999", 3.5)):
            k, low, high = checker._factor(Fraction(risk), Fraction(str(dof)))
            self.assertEqual((low, high), (k * (1 - checker.DELTA), k * (1 + checker.DELTA)))
            band = k * 2                                                # u = 2
            for rel, want in ((Fraction(1, 10 ** 5), ("accept", [])), (Fraction(-1, 10 ** 5), ("undecided", [])),
                              (Fraction(1, 10 ** 8), ("unresolved", FACTOR)), (Fraction(-1, 10 ** 8), ("unresolved", FACTOR))):
                value = decimal_text(Fraction(200) - band * (1 + rel), 40)
                rec = opened(file_text(value, "200", u="2", dof=str(dof), risk=risk))
                self.assertEqual(decide(rec), want, (risk, float(rel)))

    def test_the_two_ends_of_the_supported_risks(self):
        edge = Fraction(1, 10 ** 300)
        for risk, supported in ((edge, True), (edge / 10, False), (Fraction(1, 2) - edge, True),
                                (Fraction(1, 2) - edge / 10, False)):
            rec = opened(file_text("50", "100", dof="1e7", risk=decimal_text(risk, 320)))
            self.assertEqual(checker.invalid(rec), [])
            if supported:
                self.assertIn(decide(rec)[0], ("accept", "undecided"), float(risk))
            else:
                self.assertEqual(decide(rec), ("unresolved", ["risk not supported"]))

    def test_very_few_degrees_of_freedom_next_to_one_half(self):
        # Below 0.001 degrees of freedom the factor is not computed from the gap. It is known to
        # exceed twice the gap, because no density of Student's t exceeds 1 / sqrt(2 pi) at zero.
        for dof in (1e-4, 1e-9, 1e-30):
            for gap in (Fraction(1, 10), Fraction(1, 10 ** 6), Fraction(1, 10 ** 20)):
                risk = Fraction(1, 2) - gap
                self.assertEqual(checker._factor(risk, Fraction(str(dof))), (None, 2 * gap, None))
                token = decimal_text(risk)
                at = opened(file_text("100", "100", dof=str(dof), risk=token))
                self.assertEqual(decide(at), ("undecided", []), (dof, gap))          # the sign settles it
                self.assertEqual(decide(opened(file_text("100", "100", u="0", dof=str(dof), risk=token))), ("accept", []))
                near = opened(file_text(decimal_text(Fraction(100) - gap), "100", dof=str(dof), risk=token))
                self.assertEqual(decide(near), ("undecided", []), (dof, gap))        # headroom below twice the gap
                far = opened(file_text(decimal_text(Fraction(100) - 3 * gap), "100", dof=str(dof), risk=token))
                self.assertEqual(decide(far), ("unresolved", ["coverage factor not computed"]), (dof, gap))
                (lo, hi), (down, up) = guard(at)
                self.assertEqual((lo, hi, down, up), (Decimal(100) + 2 * Decimal(decimal_text(gap)), Decimal("Infinity"),
                                                      Decimal("-Infinity"), Decimal(100) - 2 * Decimal(decimal_text(gap))))
        # The bound itself: f(0) <= 1 / sqrt(2 pi) for every number of degrees of freedom.
        for dof in (1e-6, 0.001, 0.5, 1, 2, 30, 1e4):
            f0 = math.exp(math.lgamma((dof + 1) / 2) - math.lgamma(dof / 2)) / math.sqrt(dof * math.pi)
            self.assertLess(f0, 1 / math.sqrt(2 * math.pi))
            self.assertGreater(1 / f0, 2.5)
        # Up to a risk of one quarter a factor beyond the range has no upper bound, and since
        # Version 15 a lower bound of 1e149.
        rec = opened(file_text("50", "100", dof="1e-9", risk="0.05"))
        self.assertEqual(decide(rec), ("undecided", []))
        # In full since Version 71 (tests/test_review70.py); until Version 70 it was rounded to forty digits, 1E+149.
        self.assertEqual(guard(rec)[0], (Decimal(10 ** 149 + 50), Decimal("Infinity")))

    def test_the_tail_is_not_lost_far_out_with_few_degrees_of_freedom(self):
        # With 1e-30 degrees of freedom the upper tail stays near one half as far as floats go,
        # and a quantile for a tail of one quarter is refused, not invented.
        for t in (1.0, 1e100, 1e140, 1e150):
            self.assertGreater(dist.t_sf(t, 1e-30), 0.49)
        for dof in (1e-30, 1e-20, 1e-9):
            with self.assertRaises(ValueError):
                t_isf(0.25, dof)


class ResultWithoutUncertainty(unittest.TestCase):
    ROWS = ((99.9996, 1.0), (100.0, 1.0), (100.0004, 0.0))

    def test_the_three_rows_of_the_review(self):
        shown = set()
        for value, prob in self.ROWS:
            r = Result(value, 0.0, "ms")
            short, long = str(r), r.report()
            self.assertEqual(short, f"{value!r}(0) ms")
            # Version 62 and before printed "k = 1.96" here. Since Version 63 no factor is looked for where
            # there is no uncertainty to multiply, and "u = 0" says why the width is zero.
            self.assertEqual(long, f"({value!r} ± 0) ms, u = 0, p = 0.95")
            # Either printed form gives the stored estimate back.
            self.assertEqual(float(short.split("(")[0]), value)
            self.assertEqual(float(long[1:].split(" ±")[0]), value)
            self.assertEqual(r.prob_below(100.0), prob)
            self.assertEqual(r.interval(), (value, value))
            shown.add((short, long))
        self.assertEqual(len(shown), 3)
        self.assertEqual([str(Result(v, 0.0, "ms")) for v, _ in self.ROWS],
                         ["99.9996(0) ms", "100.0(0) ms", "100.0004(0) ms"])
        # What Version 13 printed for all three.
        self.assertEqual({f"{v:.3f}(0) ms" for v, _ in self.ROWS}, {"100.000(0) ms"})

    def test_other_values_without_uncertainty(self):
        self.assertEqual(str(Result(2 ** 53 + 1, 0, "ns")), "9007199254740993(0) ns")
        self.assertEqual(Result(2 ** 53 + 1, 0, "ns").report(), "(9007199254740993 ± 0) ns, u = 0, p = 0.95")
        self.assertEqual(str(Result(1e-07, 0.0, "s")), "1e-07(0) s")
        self.assertEqual(str(Result(0.1 + 0.2, 0.0)), "0.30000000000000004(0)")
        self.assertEqual(Result(5.5, 0.0, "ms", 3).report(), "(5.5 ± 0) ms, u = 0, p = 0.95")     # Version 62: k = 3.18
        self.assertEqual(Result(5.5, 0.0).report(), "(5.5 ± 0), u = 0, p = 0.95")
        self.assertEqual(fmt(5.5, 0.0), "5.5(0)")

    def test_results_with_an_uncertainty_print_as_before(self):
        self.assertEqual(str(Result(12.311, 0.0097, "ms")), "12.3110(97) ms")
        self.assertEqual(Result(12.311, 0.0097, "ms").report(), "(12.311 ± 0.019) ms, k = 1.96, p = 0.95")
        self.assertEqual(str(Result(1230.0, 240.0, "ms")), "1230(240) ms")
        self.assertEqual(Result(1230.0, 240.0, "ms").report(), "(1230 ± 470) ms, k = 1.96, p = 0.95")
        self.assertEqual(Result(1.5, 0.25).report(), "(1.50 ± 0.49), k = 1.96, p = 0.95")
        text = (ROOT / "expected" / "ch12_calibration.txt").read_text(encoding="utf-8")
        self.assertIn("measured value    36.55(16) ms\n", text)
        self.assertIn("result  (36.55 ± 0.32) ms, k = 2.05, p = 0.95\n", text)


class TimedOutRequests(unittest.TestCase):
    def test_the_table_of_the_review(self):
        # 910 within 200 ms, 20 above it, 40 still outstanding at 500 ms, 30 lost: 1000 in all.
        self.assertEqual(910 + 20 + 40 + 30, 1000)
        self.assertEqual(ecdf_rank(0.95, 1000), 950)
        self.assertEqual(ch16_bounds.quantile_verdict(1000, 0.95, 910, 70), "unresolved")   # 910 to 980
        self.assertEqual(ch16_bounds.quantile_verdict(1000, 0.95, 910, 30), "missed")       # 910 to 940
        self.assertEqual(ch16_bounds.quantile_verdict(1000, 0.95, 950, 0), "met")
        self.assertEqual(ch16_bounds.quantile_verdict(1000, 0.95, 949, 0), "missed")
        self.assertEqual(ch16_bounds.quantile_verdict(1000, 0.95, 949, 1), "unresolved")
        text = (ROOT / "expected" / "ch16_bounds.txt").read_text(encoding="utf-8")
        for line in ("  all 70 unknown:", "    910 to 980 within, unresolved", "  40 known outstanding at 500 ms:",
                     "    910 to 940 within, missed"):
            self.assertIn(line + "\n", text)

    def test_the_verdict_against_every_possible_outcome(self):
        # Small cases, with every way the unknown durations could fall on either side of 200 ms.
        for total in (4, 10, 20):
            for q in (0.5, 0.9, 0.95):
                for within in range(total + 1):
                    for unknown in range(total - within + 1):
                        beyond = total - within - unknown
                        outcomes = set()
                        for fast in range(unknown + 1):
                            every = sorted([100.0] * (within + fast) + [300.0] * (beyond + unknown - fast))
                            outcomes.add(ecdf_quantile(every, q) <= 200.0)
                        want = "met" if outcomes == {True} else "missed" if outcomes == {False} else "unresolved"
                        self.assertEqual(ch16_bounds.quantile_verdict(total, q, within, unknown), want,
                                         (total, q, within, unknown))

    def test_a_timeout_bounds_only_its_own_request(self):
        # Three completions, and a fourth request still running at its own deadline of 50 ms.
        done = [20.0, 80.0, 90.0]
        medians = {ecdf_quantile(sorted(done + [late]), 0.5) for late in (60.0, 100.0)}
        self.assertEqual(medians, {60.0, 80.0})          # three quarters completed, the median is open
        # One timeout for all, later than every completion: the request ranks last, whatever it took.
        medians = {ecdf_quantile(sorted(done + [late]), 0.5) for late in (100.0, 250.0, math.inf)}
        self.assertEqual(medians, {80.0})


class CalibrationModel(unittest.TestCase):
    IND = 37.90

    def corrected(self, shift):
        """The corrected delay when the true delay of each reference setting is its nominal
        value plus shift[setting], and the line is fitted against the nominal values."""
        rng = random.Random(12)
        xs = [r for r in ch12_calibration.REFS for _ in range(ch12_calibration.REPEATS)]
        ys = [0.85 + 1.012 * (x + shift[x]) + rng.gauss(0, 0.15) for x in xs]
        a, b = linfit(xs, ys)[:2]
        return (self.IND - a) / b

    def test_one_shared_offset_or_six_independent_ones(self):
        refs = ch12_calibration.REFS
        base = self.corrected({r: 0.0 for r in refs})
        self.assertAlmostEqual(base, 36.55, places=2)
        h = 1e-6
        each = [(self.corrected({q: h if q == r else 0.0 for q in refs}) - base) / h for r in refs]
        together = (self.corrected({q: h for q in refs}) - base) / h
        # A shared offset moves the corrected value by the whole offset.
        self.assertAlmostEqual(abs(together), 1.0, places=2)
        self.assertAlmostEqual(sum(each), together, places=6)
        shared = ch12_calibration.U_REF * abs(together)
        independent = ch12_calibration.U_REF * math.sqrt(sum(c * c for c in each))
        self.assertEqual((round(shared, 2), round(independent, 4)), (0.02, 0.0083))

    def test_the_listing_carries_the_offset_whole(self):
        rng = random.Random(12)
        xs, fit = ch12_calibration.calibrate(rng)
        r = ch12_calibration.measure(self.IND, xs, fit)
        a, b, _, _, s = fit
        mx = sum(xs) / len(xs)
        sxx = sum((v - mx) ** 2 for v in xs)
        u_fit = s / b * math.sqrt(1 + 1 / len(xs) + (r.value - mx) ** 2 / sxx)
        self.assertAlmostEqual(r.u, math.hypot(u_fit, 0.02), places=12)
        self.assertEqual(r.report(), "(36.55 ± 0.32) ms, k = 2.05, p = 0.95")


class FigureLabel(unittest.TestCase):
    def test_the_lower_row_of_figure_22_1_names_the_rounded_record(self):
        source = (ROOT / "figures" / "make_figures.py").read_text(encoding="utf-8")
        self.assertIn('"rounded\\nrecord"', source)
        self.assertNotIn("rounded to\\ntwo digits", source)


if __name__ == "__main__":
    unittest.main()
