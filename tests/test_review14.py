"""Regression tests added after the review of Version 14.

The review found that a histogram with no observations reported its last
finite bound as a percentile, so that a change of bounds alone doubled the
latency of an empty window; that the correction for a judge's error rates
clipped a negative estimate to zero in silence, and blamed the model for
what sampling alone does; and that a coverage factor too large to compute
was treated as unbounded even where a finite guard band would have settled
the record. It also suggested a worked case in which one more histogram
bound decides an objective.

Each test computes what it expects independently of the code under test:
binomial sums, a second route to the exact interval through the incomplete
beta function, exact coverage by enumeration, every assignment of the
unknown durations, the closed form of the quantile for one degree of
freedom.
"""
import json
import math
import tempfile
import unittest
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from examples import ch09_histogram, ch16_bounds, ch20_judge, ch22_budget, ch22_record_check, ch23_case
from swmetro import record as checker
from swmetro.dist import betainc
from swmetro.propagate import lpu
from swmetro.record import decide, guard
from swmetro.stats import ecdf_quantile, ecdf_rank

ROOT = Path(__file__).resolve().parent.parent
STORED = (ROOT / "data" / "measurement_record.json").read_text(encoding="utf-8")
NOT_COMPUTED = ["coverage factor not computed"]


def expected(name):
    return (ROOT / "expected" / f"{name}.txt").read_text(encoding="utf-8")


def file_text(value, limit, u="1", dof="1", risk="0.05", digits='"all"', step=None):
    """The text of a complete record file with one component, in which the
    given tokens stand exactly as they are passed."""
    rec = json.loads(STORED)
    rec["components"] = [{"name": "c0", "type": "B", "kind": "stipulated", "u": "@u@"}]
    rec["result"].update(value="@value@", u="@stated@", dof="@dof@", significant_digits="@digits@")
    rec["decision"].update(limit="@limit@", risk="@risk@")
    tokens = {"value": value, "stated": u, "u": u, "dof": dof, "limit": limit, "risk": risk, "digits": digits}
    if step is not None:
        rec["result"]["value_step"] = "@step@"
        tokens["step"] = step
    text = json.dumps(rec, indent=1)
    for key, token in tokens.items():
        assert text.count(f'"@{key}@"') == 1, key
        text = text.replace(f'"@{key}@"', token)
    return text


def opened(text):
    """The record of a file with this text, opened with the printed loader."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "record.json"
        path.write_text(text, encoding="utf-8")
        return ch22_record_check.load(str(path))


class EmptyHistogram(unittest.TestCase):
    def test_the_two_empty_rows_of_the_review(self):
        # No observations: no sample percentile, whatever the bounds.
        self.assertTrue(math.isnan(ch09_histogram.bucket_quantile([], 0.95)))
        for q in (0.0, 0.5, 0.95, 1.0):
            for bounds in (ch09_histogram.BOUNDS, [50, 100, 150, 200, 250, 500], [100, 250, 500, 1000], [1.0]):
                self.assertTrue(math.isnan(ch09_histogram.bucket_quantile([], q, bounds)), (q, bounds))
        # What Version 14 returned instead: the last finite bound of each configuration.
        self.assertEqual((ch09_histogram.BOUNDS[-1], ch23_case.OLD["bounds"][-1], ch23_case.NEW["bounds"][-1]),
                         (10.0, 500, 1000))

    def test_a_rank_in_the_open_ended_bucket(self):
        # The third row, kept apart: the last finite bound, which the percentile exceeds.
        two = [650.0, 800.0]
        self.assertEqual(ch09_histogram.bucket_quantile(two, 0.95, [100, 500]), 500)
        self.assertEqual(ecdf_quantile(two, 0.95), 800.0)
        self.assertEqual(ch22_budget.bracket(two, [100, 500]), (500, math.inf))

    def test_the_dashboard_shows_no_observations(self):
        for config in (ch23_case.OLD, ch23_case.NEW):
            value = ch23_case.reported([], config)
            self.assertTrue(math.isnan(value))
            self.assertEqual(ch23_case.shown(value), "no observations")
        self.assertEqual(ch23_case.shown(191.5701), "191.57")
        text = expected("ch23_case")
        self.assertIn("reported p95: 191.57 -> 241.35 ms\n", text)
        self.assertIn("empty window: no observations -> no observations\n", text)

    def test_one_observation_is_enough_for_a_number(self):
        self.assertTrue(0.25 < ch09_histogram.bucket_quantile([0.3], 0.95) <= 0.5)
        self.assertEqual(ch23_case.reported([(0.0, 120.0)], ch23_case.OLD), 100 + 50 * 0.95)

    def test_nonempty_histograms_report_as_before(self):
        text = expected("ch09_histogram")
        for line in ("0.5     0.0802  0.0802   0.0833  0.0796", "0.99    0.4107  0.4107   0.4712  0.4106",
                     "0.999   0.7127  0.7127   0.9043  0.7189"):
            self.assertIn(line + "\n", text)
        xs = [0.5] * 19 + [7.0]
        self.assertEqual(ch09_histogram.bucket_quantile(xs, 0.95, [1.0, 5.0, 10.0]), 1.0)


class JudgeCorrection(unittest.TestCase):
    SENS, SPEC = 0.90, 0.85

    def test_the_estimate_is_no_longer_confined_in_silence(self):
        y = ch20_judge.corrected([2 / 20, self.SENS, self.SPEC])
        self.assertAlmostEqual(y, -0.05 / 0.75, places=12)
        self.assertEqual(f"{100 * y:.2f}", "-6.67")
        self.assertEqual(ch20_judge.feasible(y), (0.0, "below 0"))
        high = ch20_judge.corrected([19 / 20, self.SENS, self.SPEC])
        self.assertGreater(high, 1.0)
        self.assertEqual(ch20_judge.feasible(high), (1.0, "above 1"))
        self.assertEqual(ch20_judge.feasible(0.25), (0.25, "within 0 and 1"))
        self.assertEqual(ch20_judge.feasible(0.0)[1], "within 0 and 1")
        with self.assertRaises(ValueError):
            ch20_judge.corrected([0.5, 0.5, 0.5])

    def test_sampling_alone_crosses_the_bound(self):
        # A true pass rate of 4 %: the judge passes 18 % on average.
        seen = 0.04 * self.SENS + 0.96 * (1 - self.SPEC)
        self.assertAlmostEqual(seen, 0.18, places=12)
        self.assertAlmostEqual(ch20_judge.corrected([seen, self.SENS, self.SPEC]), 0.04, places=12)
        # In 20 outputs, at most two judged passes with probability 27.48 %.
        chance = sum(math.comb(20, k) * seen ** k * (1 - seen) ** (20 - k) for k in range(3))
        self.assertEqual(f"{100 * chance:.2f}", "27.48")
        self.assertAlmostEqual(ch20_judge.at_most(2, 20, seen), chance, places=14)
        # Each of those samples gives a negative estimate; three passes give exactly none.
        for k in range(3):
            self.assertLess(ch20_judge.corrected([k / 20, self.SENS, self.SPEC]), 0, k)
        self.assertAlmostEqual(ch20_judge.corrected([3 / 20, self.SENS, self.SPEC]), 0.0, places=12)

    def test_the_exact_interval_by_another_route(self):
        # The same bounds as quantiles of beta distributions: P(X >= k) = I_p(k, n - k + 1).
        def bisect(f, target):
            lo, hi = 0.0, 1.0
            for _ in range(80):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if f(mid) < target else (lo, mid)
            return (lo + hi) / 2
        for k, n in ((2, 20), (1, 20), (7, 20), (19, 20), (3, 7), (50, 400)):
            low, high = ch20_judge.exact_interval(k, n)
            self.assertAlmostEqual(low, bisect(lambda p: betainc(k, n - k + 1, p), 0.025), places=10, msg=(k, n))
            self.assertAlmostEqual(high, bisect(lambda p: betainc(k + 1, n - k, p), 0.975), places=10, msg=(k, n))
        low, high = ch20_judge.exact_interval(2, 20)
        self.assertEqual((round(low, 4), round(high, 4)), (0.0123, 0.3170))
        # No successes and no failures have closed forms.
        self.assertEqual(ch20_judge.exact_interval(0, 20)[0], 0.0)
        self.assertAlmostEqual(ch20_judge.exact_interval(0, 20)[1], 1 - 0.025 ** (1 / 20), places=12)
        self.assertEqual(ch20_judge.exact_interval(20, 20)[1], 1.0)
        self.assertAlmostEqual(ch20_judge.exact_interval(20, 20)[0], 0.025 ** (1 / 20), places=12)

    def test_the_exact_interval_covers_at_least_95_percent(self):
        # Exact coverage, by summing the probabilities of the counts whose interval holds p.
        n = 20
        intervals = [ch20_judge.exact_interval(k, n) for k in range(n + 1)]
        for p in (0.01, 0.04, 0.18, 0.3, 0.5, 0.77, 0.99):
            cover = sum(math.comb(n, k) * p ** k * (1 - p) ** (n - k)
                        for k, (low, high) in enumerate(intervals) if low <= p <= high)
            self.assertGreaterEqual(cover, 0.95, p)

    def test_the_interval_of_the_review(self):
        # Since Version 16 the interval of the judged rate is returned as well.
        y, seen, (low, high) = ch20_judge.few_outputs(2, 20, self.SENS, self.SPEC)
        self.assertEqual(seen, ch20_judge.exact_interval(2, 20))
        self.assertAlmostEqual(y, -1 / 15, places=12)
        self.assertEqual((low, f"{100 * high:.1f}"), (0.0, "22.3"))
        # The unconfined ends: the lower one is below zero and is brought to the bound.
        a, b = ch20_judge.exact_interval(2, 20)
        self.assertLess((a + self.SPEC - 1) / (self.SENS + self.SPEC - 1), 0)
        self.assertAlmostEqual(high, (b + self.SPEC - 1) / (self.SENS + self.SPEC - 1), places=12)
        # The true rate of the example lies in it.
        self.assertTrue(low <= 0.04 <= high)

    def test_the_transcript(self):
        text = expected("ch20_judge")
        for line in ("  observed 0.593, corrected 0.592, u 0.024", "  from obs, sens, spec: 0.0094, 0.0174, 0.0141",
                     "true rate 0.04: judged 0.18,", "  P(at most 2 of 20 pass) 0.2748", "2 of 20 judged passes:",
                     "  corrected -0.0667, below 0", "  95 % interval 0.000 to 0.223"):
            self.assertIn(line + "\n", text)

    def test_the_interior_case_and_its_budget_are_unchanged(self):
        xs, us = [0.593, 0.895, 0.845], [0.00695, 0.0217, 0.0256]
        y, u, cu = lpu(ch20_judge.corrected, xs, us)
        j = xs[1] + xs[2] - 1
        self.assertAlmostEqual(y, (xs[0] + xs[2] - 1) / j, places=12)
        self.assertAlmostEqual(cu[0], us[0] / j, places=6)
        self.assertTrue(0 < y < 1)


class ExtraBoundary(unittest.TestCase):
    """1000 requests: 928 within 180 ms, 25 between 180 and 250 ms, 20 above 250 ms, 15 still
    outstanding at 500 ms, 12 with no completion information; p95 at most 200 ms."""

    def test_the_counts(self):
        self.assertEqual(928 + 25 + 20 + 15 + 12, 1000)
        self.assertEqual(ecdf_rank(0.95, 1000), 950)
        # Without a bound at 200 ms: between 928 and 965 within the threshold.
        self.assertEqual(ch16_bounds.quantile_verdict(1000, 0.95, 928, 25 + 12), "unresolved")
        self.assertTrue(928 < 950 <= 928 + 25 + 12 == 965)

    def test_the_table_of_the_review_against_every_assignment(self):
        ranges = {"missed": range(0, 10), "unresolved": range(10, 22), "met": range(22, 26)}
        for k in range(26):
            # Every number of the 12 lost requests that could have been within 200 ms.
            outcomes = {928 + k + lost_fast >= 950 for lost_fast in range(13)}
            want = "met" if outcomes == {True} else "missed" if outcomes == {False} else "unresolved"
            self.assertEqual(ch16_bounds.quantile_verdict(1000, 0.95, 928 + k, 12), want, k)
            self.assertIn(k, ranges[want])
            self.assertEqual((928 + k, 940 + k), (928 + k, 928 + k + 12))

    def test_the_same_through_the_durations_themselves(self):
        # One arrangement for each end of each verdict, with actual durations.
        def p95(k, lost_fast):
            within = [150.0] * 928 + [190.0] * k + [100.0] * lost_fast
            beyond = [220.0] * (25 - k) + [300.0] * 20 + [600.0] * 15 + [900.0] * (12 - lost_fast)
            return ecdf_quantile(sorted(within + beyond), 0.95)
        self.assertGreater(p95(9, 12), 200.0)        # missed even if every lost request was fast
        self.assertLessEqual(p95(10, 12), 200.0)     # unresolved: met if the lost were fast,
        self.assertGreater(p95(10, 0), 200.0)        # missed if they were slow
        self.assertGreater(p95(21, 0), 200.0)
        self.assertLessEqual(p95(22, 0), 200.0)      # met even if every lost request was slow

    def test_the_transcript(self):
        text = expected("ch16_bounds")
        for line in ("928 within 180 ms, 25 between 180", "and 250 ms, 12 lost: unresolved", "  k 0 to 9: missed",
                     "  k 10 to 21: unresolved", "  k 22 to 25: met", "  960 captured: rank 910 to rank 950",
                     "    910 to 940 within, missed"):
            self.assertIn(line + "\n", text)


class FactorNotComputed(unittest.TestCase):
    def test_the_three_estimates_of_the_review(self):
        # One degree of freedom, risk 1e-160: the factor is cot(pi risk), beyond the range computed.
        factor = 1 / math.tan(math.pi * 1e-160)
        self.assertAlmostEqual(factor / 3.1831e159, 1.0, places=4)
        self.assertGreater(factor, 1e150)
        self.assertAlmostEqual(factor * 1e-160, 0.31831, places=5)          # the band, in ms
        rows = (("99", ("unresolved", NOT_COMPUTED)),      # the rule gives accept: 99 + 0.318 <= 100
                ("100", ("undecided", [])),                # the sign of the band settles equality
                ("101", ("unresolved", NOT_COMPUTED)))     # the rule gives reject: 101 - 0.318 > 100
        for value, want in rows:
            rec = opened(file_text(value, "100", u="1e-160", risk="1e-160"))
            self.assertEqual(checker.invalid(rec), [])
            self.assertEqual(decide(rec), want, value)
            # What is shown: the known lower bound of the band, and no upper one.
            (lo, hi), (down, up) = guard(rec)
            self.assertEqual((lo, hi), (Decimal(value) + Decimal("1E-11"), Decimal("Infinity")))
            self.assertEqual((down, up), (Decimal("-Infinity"), Decimal(value) - Decimal("1E-11")))
        # Version 14 called all three undecided.

    def test_with_no_uncertainty_the_factor_does_not_matter(self):
        for value, want in (("99", "accept"), ("100", "accept"), ("101", "reject")):
            rec = opened(file_text(value, "100", u="0", risk="1e-160"))
            self.assertEqual(decide(rec), (want, []), value)

    def test_the_lower_bound_decides_where_it_can(self):
        k, low, high = checker._factor(Fraction(1, 10 ** 160), Fraction(1))
        self.assertEqual((k, low, high), (None, checker.FAR, None))
        self.assertEqual(checker.FAR, 10 ** 149)
        # The bound holds: for one degree of freedom P(T > 1e149) = atan(1e-149) / pi, far above the risk.
        self.assertGreater(math.atan(1e-149) / math.pi, 1e-160)
        # An ordinary uncertainty: the band exceeds 1e149 ms, and every estimate is undecided, as before.
        for value in ("-1e100", "99", "100", "101", "1e100"):
            self.assertEqual(decide(opened(file_text(value, "100", u="1", risk="1e-160"))), ("undecided", []), value)
        # Headroom below and above the bound of the band, 1e149 x 1e-160 = 1e-11 ms.
        for value, want in (("99.999999999995", ("undecided", [])), ("100.000000000005", ("undecided", [])),
                            ("99.99999999998", ("unresolved", NOT_COMPUTED)), ("100.00000000002", ("unresolved", NOT_COMPUTED))):
            rec = opened(file_text(value, "100", u="1e-160", risk="1e-160"))
            self.assertEqual(decide(rec), want, value)

    def test_the_reason_names_the_factor_and_not_the_rounding(self):
        rec = opened(file_text("99", "100", u="1e-160", risk="1e-160", digits="2", step="0.1"))
        self.assertEqual(decide(rec), ("unresolved", NOT_COMPUTED))
        # A computed factor at the limit keeps its own reason.
        k = checker._factor(Fraction("0.05"), Fraction(10))[0]
        value = Fraction(200) - 2 * k
        text = f"{Decimal(value.numerator) / Decimal(value.denominator):.15f}"
        rec = opened(file_text(text, "200", u="2", dof="10"))
        self.assertEqual(decide(rec), ("unresolved", ["at the limit: the coverage factor decides"]))

    def test_other_factors_that_are_not_computed(self):
        # Very few degrees of freedom: the upper tail (bound 1e149) and next to one half (twice the gap).
        self.assertEqual(checker._factor(Fraction("0.05"), Fraction("1e-9")), (None, checker.FAR, None))
        gap = Fraction(1, 10 ** 6)
        self.assertEqual(checker._factor(Fraction(1, 2) - gap, Fraction("1e-9")), (None, 2 * gap, None))
        self.assertEqual(decide(opened(file_text("50", "100", dof="1e-9"))), ("undecided", []))

    def test_the_bound_holds_next_to_the_end_of_the_range(self):
        # Far out the upper tail of Student's t is the first term of its series,
        # P(T > t) = x^a / (2 a B(a, 1/2)) with a = dof / 2 and x = dof / (dof + t^2), so the quantile for a
        # risk has a closed form there. It gives the factor without the incomplete beta function.
        def quantile(risk, dof):
            a = dof / 2
            log_ab = math.lgamma(a + 1) + math.lgamma(0.5) - math.lgamma(a + 0.5)
            return math.exp((math.log(dof) - (math.log(2 * risk) + log_ab) / a) / 2)

        def risk_at(t, dof):
            a = dof / 2
            log_ab = math.lgamma(a + 1) + math.lgamma(0.5) - math.lgamma(a + 0.5)
            return 0.5 * math.exp(a * (math.log(dof) - 2 * math.log(t)) - log_ab)
        # The form against a known case: one degree of freedom, where the tail far out is 1 / (pi t).
        self.assertAlmostEqual(quantile(1e-140, 1.0) * math.pi * 1e-140, 1.0, places=12)
        # Four factors between 6e148 and 5e149 with few degrees of freedom, on both sides of a risk of 1/4.
        # The tail used to be cut off at 1e150 sqrt(dof); two of these were then refused although they
        # lie below FAR, and the bound claimed for them was false.
        cases = ((0.0036245335443339027, 0.14246988848051192), (0.001989087053765291, 0.25097656379736094),
                 (0.04862157896774435, 2.390242313147182e-08), (0.0013841049378146353, 0.3092740270285795))
        below_far = 0
        for dof, risk in cases:
            true = quantile(risk, dof)
            self.assertTrue(6e148 < true < 5e149, (dof, true))
            self.assertLess(1e150 * math.sqrt(dof), true)                 # beyond the old cut-off
            k, low, high = checker._factor(Fraction(risk), Fraction(dof))
            self.assertIsNotNone(k, (dof, risk))
            self.assertAlmostEqual(float(k) / true, 1.0, places=8, msg=(dof, risk))
            self.assertEqual((low, high), (k * (1 - checker.DELTA), k * (1 + checker.DELTA)))
            below_far += true < checker.FAR
        self.assertEqual(below_far, 2)
        # Beyond the range the factor is refused, and it does exceed the bound.
        for dof in (0.0036245335443339027, 0.04862157896774435, 0.5, 1.0, 1.5):
            for t in (2e150, 1e152, 1e160):
                risk = risk_at(t, dof)
                if risk < 1e-300:
                    continue
                self.assertEqual(checker._factor(Fraction(risk), Fraction(dof)), (None, checker.FAR, None), (dof, t))
                self.assertGreater(quantile(risk, dof), checker.FAR)
        # And just inside it the factor is computed.
        for dof in (0.0036245335443339027, 0.5, 1.0, 1.5):
            risk = risk_at(3e149, dof)
            k = checker._factor(Fraction(risk), Fraction(dof))[0]
            self.assertAlmostEqual(float(k) / 3e149, 1.0, places=8, msg=dof)


if __name__ == "__main__":
    unittest.main()
