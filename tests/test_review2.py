"""Regression tests added after the accuracy review of Version 2.

As before, each test checks the code against an independent fact: a closed
form, a definition evaluated by brute force, or a simulated coverage.
"""
import importlib.util
import json
import math
import random
import unittest
from fractions import Fraction
from pathlib import Path

from examples import ch02_targets, ch08_counter, ch08_selection
from examples import ch10_censoring as c10
from examples import ch22_budget as c22
from examples import ch22_runs
from swmetro import Result
from swmetro.dist import phi, t_isf, t_ppf, t_sf, z_ppf
from swmetro.stats import ecdf_quantile, ecdf_rank, quantile

ROOT = Path(__file__).resolve().parent.parent


class LowerTail(unittest.TestCase):
    def test_cauchy_lower_tail(self):
        # One degree of freedom: t = -1 / tan(pi p), which never forms 1 - p.
        for p in (1e-3, 1e-12, 1e-16, 1e-20, 1e-100):
            exact = -1 / math.tan(math.pi * p)
            self.assertAlmostEqual(t_ppf(p, 1) / exact, 1.0, places=11)

    def test_two_degrees_of_freedom_lower_tail(self):
        # Two degrees of freedom: t = (1 - 2 q) / sqrt(2 q (1 - q)) for tail q.
        for q in (1e-6, 1e-30, 1e-100):
            exact = (1 - 2 * q) / math.sqrt(2 * q * (1 - q))
            self.assertAlmostEqual(t_isf(q, 2) / exact, 1.0, places=11)
            self.assertAlmostEqual(-t_ppf(q, 2) / exact, 1.0, places=11)

    def test_inverse_survival_round_trip(self):
        for dof in (1, 3.5, 12, 200):
            for tail in (0.4, 1e-3, 1e-9, 1e-40):
                self.assertAlmostEqual(t_sf(t_isf(tail, dof), dof) / tail, 1.0, places=9)

    def test_centre_and_domain(self):
        self.assertEqual(t_ppf(0.5, 7), 0.0)
        for tail in (0.0, 0.6, -1e-9):
            with self.assertRaises(ValueError):
                t_isf(tail, 5)
        with self.assertRaises(ValueError):
            t_isf(0.1, 0)
        # A valid probability whose quantile lies beyond the supported range
        # is refused; it is not answered with a wrong number.
        with self.assertRaises(ValueError):
            t_ppf(1e-200, 1)


class PointMass(unittest.TestCase):
    def test_zero_uncertainty_probability(self):
        r = Result(5.0, 0.0)
        self.assertEqual(r.prob_below(4.0), 0.0)
        self.assertEqual(r.prob_below(5.0), 1.0)
        self.assertEqual(r.prob_below(6.0), 1.0)
        self.assertEqual(r.interval(), (5.0, 5.0))


class QuantileContract(unittest.TestCase):
    def test_rank_is_the_definition(self):
        # Smallest k with k / n >= q, found by exact arithmetic.
        for n in (1, 2, 3, 7, 20, 400, 2000):
            for q in ("0", "0.01", "0.5", "0.9", "0.95", "0.99", "0.999", "1"):
                want = next((k for k in range(1, n + 1) if Fraction(k, n) >= Fraction(q)), n)
                self.assertEqual(ecdf_rank(float(q), n), want, (q, n))

    def test_quantile_and_share_say_the_same(self):
        rng = random.Random(1)
        for _ in range(300):
            n = rng.randint(1, 60)
            xs = sorted(float(rng.randint(1, 30)) for _ in range(n))
            q = rng.choice((0.5, 0.9, 0.95, 0.99))
            limit = float(rng.randint(1, 30))
            share = Fraction(sum(x <= limit for x in xs), n)
            self.assertEqual(ecdf_quantile(xs, q) <= limit, share >= Fraction(str(q)))

    def test_reviewers_counterexample(self):
        xs = [100.0] * 19 + [5000.0]
        bounds = [100, 200, 1000, 5000]
        self.assertEqual(c22.p95(xs), 100.0)
        self.assertEqual(c22.bracket(xs, bounds), (-math.inf, 100.0))   # (0.0, 100.0) until Version 53: no lower end
        self.assertEqual(c22.verdict(*c22.bracket(xs, bounds)), "met")
        # The linear convention is another statistic and gives another answer.
        self.assertAlmostEqual(quantile(sorted(xs), 0.95), 345.0)

    def test_rank_on_a_bucket_boundary(self):
        xs = [10.0 * i for i in range(1, 21)]          # 10, 20, ..., 200
        self.assertEqual(c22.p95(xs), 190.0)
        self.assertEqual(c22.bracket(xs, [100, 190, 200]), (100.0, 190.0))
        self.assertEqual(c22.bracket(xs, [100, 180, 200]), (180.0, 200.0))
        self.assertEqual(c22.bracket(xs, [50, 100]), (100.0, math.inf))

    def test_bracket_always_holds_the_quantile(self):
        rng = random.Random(2)
        for _ in range(400):
            n = rng.randint(1, 80)
            xs = [float(rng.randint(1, 400)) for _ in range(n)]
            bounds = sorted(rng.sample(range(1, 450), rng.randint(1, 6)))
            q = rng.choice((0.5, 0.9, 0.95, 0.99))
            lo, hi = c22.bracket(xs, bounds, q)
            value = ecdf_quantile(sorted(xs), q)
            self.assertTrue(lo < value <= hi, (xs, bounds, q))


class Censoring(unittest.TestCase):
    def test_straddling_ranks(self):
        # Rank 19 of 20 is observed: the value is exact whatever the last one is.
        self.assertEqual(c10.p95_all([100.0] * 19 + [None]), (100.0, False))
        # Rank 19 of 20 is censored: only a bound.
        self.assertEqual(c10.p95_all([100.0] * 18 + [None] * 2), (c10.TIMEOUT, True))

    def test_all_completed_and_all_censored(self):
        xs = [float(i) for i in range(1, 41)]
        self.assertEqual(c10.p95_all(xs), (ecdf_quantile(xs, 0.95), False))
        self.assertEqual(c10.p95_all([None] * 7), (c10.TIMEOUT, True))

    def test_result_holds_for_every_completion(self):
        rng = random.Random(3)
        for _ in range(200):
            n = rng.randint(1, 50)
            xs = [rng.uniform(1, 900) if rng.random() < 0.8 else None for _ in range(n)]
            value, bound = c10.p95_all(xs)
            for late in (c10.TIMEOUT, 5000.0, math.inf):
                full = sorted(late if x is None else x for x in xs)
                true = ecdf_quantile(full, 0.95)
                if bound:
                    self.assertGreaterEqual(true, value)
                else:
                    self.assertEqual(true, value)


class Prediction(unittest.TestCase):
    def test_coverage_is_an_average_over_samples(self):
        rng = random.Random(4)
        hits, reps = 0, 4000
        for _ in range(reps):
            xs = ch02_targets.record(400, rng)
            new = rng.lognormvariate(4.6, 0.4)
            hits += xs[9] <= new <= xs[390]
        self.assertAlmostEqual(hits / reps, 381 / 401, delta=0.012)


class Detection(unittest.TestCase):
    def test_power_at_the_detection_limit(self):
        z = z_ppf(0.95)
        for u in (0.5, 2.95, 40.0):
            limit = 2 * z * u
            self.assertAlmostEqual(phi(limit / u - z), 0.95, places=12)
        # Half the detection limit is detected about one time in two.
        self.assertAlmostEqual(phi(z - z), 0.5, places=12)


class OptionalExamples(unittest.TestCase):
    def test_counter_rules(self):
        xs = ch08_counter.restarted()
        self.assertEqual(ch08_counter.rates(xs, 10, "naive")[4], -540.0)
        self.assertEqual(ch08_counter.rates(xs, 10, "reset")[4], 60.0)
        ys = ch08_counter.wrapping(10, 12)
        self.assertTrue(all(r == 50e6 for r in ch08_counter.rates(ys, 10, "modulo")))
        # More than one wrap between readings cannot be undone.
        zs = ch08_counter.wrapping(100, 12)
        self.assertTrue(all(r < 50e6 / 5 for r in ch08_counter.rates(zs, 100, "modulo")))

    def test_weighting_recovers_only_what_can_be_sampled(self):
        rng = random.Random(5)
        rows = ch08_selection.requests(60_000, rng)
        sample = ch08_selection.kept(rows, rng)
        traced = [(x, 1.0) for x, hit in rows if not hit]
        want = ch08_selection.summary(traced)[0]
        plain = ch08_selection.summary([(x, 1.0) for x, _ in sample])[0]
        weighted = ch08_selection.summary([(x, 1 / p) for x, p in sample])[0]
        everything = ch08_selection.summary([(x, 1.0) for x, _ in rows])[0]
        self.assertLess(abs(weighted - want) / want, 0.05)
        self.assertGreater(plain / want, 1.5)
        self.assertGreater((weighted - everything) / everything, 0.2)

    def test_mean_of_run_quantiles_is_not_the_pooled_quantile(self):
        runs = [[100.0] * 50, [300.0] * 50] * 10
        each = [ecdf_quantile(sorted(r), 0.95) for r in runs]
        pooled = sorted(x for r in runs for x in r)
        self.assertEqual(sum(each) / len(each), 200.0)
        self.assertEqual(ecdf_quantile(pooled, 0.95), 300.0)
        rng = random.Random(6)
        a, b = ch22_runs.spread(4, rng, 4000)
        s = math.hypot(ch22_runs.S_B, ch22_runs.S_W)
        self.assertAlmostEqual(a / (s / 2), 1.0, delta=0.05)
        self.assertAlmostEqual(b / (s * math.sqrt(1.25)), 1.0, delta=0.05)


class Record(unittest.TestCase):
    def test_stored_record_is_what_the_program_writes(self):
        spec = importlib.util.spec_from_file_location("make_record", ROOT / "data" / "make_record.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        stored = json.loads((ROOT / "data" / "measurement_record.json").read_text(encoding="utf-8"))
        self.assertEqual(stored, json.loads(module.text()))
        self.assertEqual(stored["measurand"]["convention"], "inverse empirical CDF")
        for key in ("data", "components", "sources"):
            self.assertIn(key, stored)
        self.assertEqual(len("".join(stored["sources"]["sha256"])), 64)


if __name__ == "__main__":
    unittest.main()
