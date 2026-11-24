"""Regression tests added after the accuracy review of Version 1.

Each test checks the code against an independent fact (an analytic value, a
published table, a stated convention), not against its own saved output.
"""
import json
import math
import random
import statistics as st
import unittest
from pathlib import Path

from examples import ch09_histogram as h9
from examples import ch22_budget
from swmetro import Budget, Result, rect_interval, u_rect
from swmetro.dist import betainc, t_cdf, t_ppf, t_sf
from swmetro.stats import block_bootstrap, bootstrap, order_stat
from swmetro.stats import quantile, quantile_interval

ROOT = Path(__file__).resolve().parent.parent


class StudentTails(unittest.TestCase):
    def test_cauchy_identity(self):
        # One degree of freedom is the Cauchy distribution: t = tan(pi (p - 1/2)).
        for p in (0.6, 0.975, 0.9995, 0.99995, 0.9999995):
            exact = math.tan(math.pi * (p - 0.5))
            self.assertAlmostEqual(t_ppf(p, 1) / exact, 1.0, places=9)

    def test_two_degrees_of_freedom_closed_form(self):
        for p in (0.75, 0.975, 0.9995):
            exact = (2 * p - 1) / math.sqrt(2 * p * (1 - p))
            self.assertAlmostEqual(t_ppf(p, 2) / exact, 1.0, places=9)

    def test_upper_tail_table(self):
        # Two-sided 0.001 points of Student's t (standard tables).
        table = {5: 6.869, 10: 4.587, 30: 3.646, 120: 3.373}
        for dof, expected in table.items():
            self.assertAlmostEqual(t_ppf(0.9995, dof), expected, places=3)

    def test_round_trip_and_symmetry(self):
        for dof in (1, 2.5, 7, 40):
            for p in (0.51, 0.9, 0.999, 0.999999):
                self.assertAlmostEqual(t_cdf(t_ppf(p, dof), dof), p, places=10)
                self.assertAlmostEqual(t_sf(t_ppf(p, dof), dof) / (1 - p), 1.0, places=6)
            self.assertAlmostEqual(t_ppf(0.2, dof), -t_ppf(0.8, dof), places=12)

    def test_domain_is_validated(self):
        for p in (0.0, 1.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                t_ppf(p, 5)
        for dof in (0, -1):
            with self.assertRaises(ValueError):
                t_ppf(0.9, dof)

    def test_incomplete_beta_special_cases(self):
        self.assertAlmostEqual(betainc(1, 1, 0.3), 0.3, places=12)       # uniform
        self.assertAlmostEqual(betainc(2, 1, 0.5), 0.25, places=12)      # x**2
        self.assertAlmostEqual(betainc(0.5, 0.5, 0.5), 0.5, places=12)   # arcsine


class CoverageModel(unittest.TestCase):
    def test_rectangular_interval_follows_the_distribution(self):
        lo, hi = rect_interval(10.0, 1.0)
        self.assertAlmostEqual(hi - lo, 1.9)
        self.assertGreaterEqual(lo, 9.0)
        self.assertLessEqual(hi, 11.0)
        # The normal factor applied to the same standard uncertainty overshoots
        # the bounds: this is why Result is documented as an approximation.
        self.assertGreater(1.96 * u_rect(1.0), 1.0)
        rng = random.Random(1)
        inside = sum(lo <= rng.uniform(9, 11) <= hi for _ in range(40_000)) / 40_000
        self.assertAlmostEqual(inside, 0.95, delta=0.005)

    def test_probability_and_interval_share_one_model(self):
        for dof in (4, 15.3, math.inf):
            r = Result(100.0, 2.0, "ms", dof)
            lo, hi = r.interval(0.95)
            self.assertAlmostEqual(r.prob_below(hi), 0.975, places=9)
            self.assertAlmostEqual(r.prob_below(lo), 0.025, places=9)
            self.assertAlmostEqual(r.prob_below(100.0), 0.5, places=12)

    def test_invalid_uncertainties_are_rejected(self):
        for bad in (-1.0, math.nan, math.inf):
            with self.assertRaises(ValueError):
                Result(1.0, bad)
            with self.assertRaises(ValueError):
                Budget().add("x", "A", bad)
        with self.assertRaises(ValueError):
            Budget().add("x", "C", 1.0)
        with self.assertRaises(ValueError):
            Budget().add("x", "A", 1.0, dof=0)


class BudgetEdges(unittest.TestCase):
    def test_all_zero_budget_has_a_defined_table(self):
        bud = Budget("ms").add("a", "A", 0.0, 9).add("b", "B", 0.0)
        self.assertEqual(bud.u_c(), 0.0)
        self.assertEqual(bud.dof(), math.inf)
        table = bud.table()
        self.assertIn("0.0 %", table)
        self.assertNotIn("nan", table)


class QuantileConventions(unittest.TestCase):
    def test_two_point_sample(self):
        xs = [1.0, 1000.0]
        self.assertEqual(quantile(xs, 0.5), 500.5)        # interpolated convention
        self.assertEqual(order_stat(xs, 0.5), 1.0)        # order-statistic convention
        est = h9.sketch_of(xs, 0.01).quantile(0.5)
        self.assertLessEqual(abs(est - 1.0) / 1.0, 0.01 + 1e-12)   # the stated guarantee

    def test_sketch_guarantee_against_order_statistics(self):
        rng = random.Random(5)
        for alpha in (0.05, 0.01):
            xs = sorted(rng.lognormvariate(0, 3) for _ in range(3000))
            sk = h9.sketch_of(xs, alpha)
            for q in (0.0, 0.01, 0.5, 0.9, 0.999, 1.0):
                exact = order_stat(xs, q)
                self.assertLessEqual(abs(sk.quantile(q) - exact) / exact, alpha * 1.0001)

    def test_sketch_bin_edges_small_and_large_values(self):
        sk = h9.LogSketch(0.01)
        values = [sk.g ** i for i in (-2000, -1, 0, 1, 2000)] + [1e-12, 1e12]
        for x in values:
            one = h9.sketch_of([x], 0.01)
            self.assertLessEqual(abs(one.quantile(0.5) - x) / x, 0.0101)

    def test_sketch_rejects_unsupported_input(self):
        sk = h9.LogSketch(0.01)
        for bad in (0.0, -1.0):
            with self.assertRaises(ValueError):
                sk.add(bad)
        with self.assertRaises(ValueError):
            sk.merge(h9.LogSketch(0.02))

    def test_order_statistic_interval(self):
        xs = [float(i) for i in range(1, 401)]
        lo, hi = quantile_interval(xs, 0.9)
        self.assertLess(lo, 360.0)
        self.assertGreater(hi, 360.0)
        # Coverage of the true quantile under independent draws.
        rng = random.Random(3)
        true_q, hits, trials = 1.2815515655, 0, 400
        for _ in range(trials):
            sample = sorted(rng.gauss(0, 1) for _ in range(200))
            a, b = quantile_interval(sample, 0.9)
            hits += a <= true_q <= b
        self.assertGreater(hits / trials, 0.93)
        # Too small a sample cannot bound an extreme quantile from above.
        self.assertIsNone(quantile_interval(xs[:500], 0.999)[1])


class Dependence(unittest.TestCase):
    def test_block_bootstrap_sees_autocorrelation(self):
        rng = random.Random(8)
        xs, x = [], 0.0
        for _ in range(3000):                       # AR(1), coefficient 0.9
            x = 0.9 * x + rng.gauss(0, 1)
            xs.append(x)
        iid = st.stdev(bootstrap(xs, st.fmean, 300, seed=1))
        blk = st.stdev(block_bootstrap(xs, st.fmean, 150, 300, seed=1))
        # For AR(1) the standard deviation of the mean is about
        # sqrt((1 + r) / (1 - r)) = 4.4 times the iid value.
        self.assertGreater(blk / iid, 2.5)
        white = [rng.gauss(0, 1) for _ in range(3000)]
        iid = st.stdev(bootstrap(white, st.fmean, 300, seed=1))
        blk = st.stdev(block_bootstrap(white, st.fmean, 150, 300, seed=1))
        self.assertLess(abs(blk / iid - 1), 0.35)


class ProvenanceRecord(unittest.TestCase):
    def test_record_agrees_with_the_example_it_describes(self):
        rec = json.loads((ROOT / "data" / "measurement_record.json").read_text())
        _, r = ch22_budget.from_raw(ch22_budget.record(random.Random(22)))
        lo, hi = r.interval()
        self.assertAlmostEqual(rec["result"]["value"], r.value, delta=0.05)
        self.assertAlmostEqual(rec["result"]["u"], r.u, delta=0.005)
        self.assertAlmostEqual(rec["result"]["dof"], r.dof, delta=0.05)
        self.assertAlmostEqual(rec["result"]["interval_95"][0], lo, delta=0.05)
        self.assertAlmostEqual(rec["result"]["interval_95"][1], hi, delta=0.05)
        self.assertEqual(rec["measurand"]["unit"], "ms")
        self.assertEqual(rec["decision"]["limit"], ch22_budget.LIMIT)
        probe = [c for c in rec["components"] if c["name"] == "probe correction"]
        self.assertEqual([c["value"] for c in probe], [-ch22_budget.PROBE])
        for key in ("measurand", "capture", "components", "result", "decision",
                    "not_evaluated", "replay"):
            self.assertIn(key, rec)


if __name__ == "__main__":
    unittest.main()
