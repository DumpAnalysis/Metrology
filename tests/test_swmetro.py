"""Unit tests for swmetro: python -m unittest discover -s tests"""
import math
import unittest

from swmetro import Budget, Result, combine, fmt, type_a
from swmetro import u_rect, u_resolution, u_tri, welch_satterthwaite
from swmetro.dist import phi, t_cdf, t_ppf, z_ppf
from swmetro.propagate import lpu, monte_carlo, normal, summarize
from swmetro.stats import bootstrap, cohen_kappa, linfit, quantile, wilson
from swmetro.units import Q


class DistTests(unittest.TestCase):
    def test_t_quantiles_match_published_tables(self):
        # Two-sided 95 % points of Student's t (standard tables).
        table = {1: 12.706, 2: 4.303, 5: 2.571, 9: 2.262, 10: 2.228,
                 30: 2.042, 120: 1.980}
        for dof, expected in table.items():
            self.assertAlmostEqual(t_ppf(0.975, dof), expected, places=3)

    def test_t_tends_to_normal(self):
        self.assertAlmostEqual(t_ppf(0.975, math.inf), 1.95996, places=4)
        self.assertAlmostEqual(t_cdf(0.0, 7), 0.5, places=12)
        self.assertAlmostEqual(t_ppf(0.025, 9), -t_ppf(0.975, 9), places=9)

    def test_normal(self):
        self.assertAlmostEqual(phi(1.645), 0.95, places=3)
        self.assertAlmostEqual(z_ppf(0.975), 1.95996, places=4)


class CoreTests(unittest.TestCase):
    def test_type_a(self):
        r = type_a([1.0, 2.0, 3.0, 4.0, 5.0], "ms")
        self.assertAlmostEqual(r.value, 3.0)
        self.assertAlmostEqual(r.u, math.sqrt(2.5 / 5))
        self.assertEqual(r.dof, 4)

    def test_type_b(self):
        self.assertAlmostEqual(u_rect(1.0), 0.57735, places=5)
        self.assertAlmostEqual(u_tri(1.0), 0.40825, places=5)
        self.assertAlmostEqual(u_resolution(1.0), 0.28868, places=5)

    def test_combine_and_effective_dof(self):
        self.assertAlmostEqual(combine(3.0, 4.0), 5.0)
        self.assertEqual(welch_satterthwaite([1.0], [math.inf]), math.inf)
        self.assertAlmostEqual(welch_satterthwaite([1.0, 1.0], [4, 4]), 8.0)

    def test_formats(self):
        self.assertEqual(fmt(12.311, 0.00968, "ms"), "12.3110(97) ms")
        self.assertEqual(fmt(842.0, 12.0, "ms"), "842(12) ms")
        self.assertEqual(fmt(12340.0, 123.0), "12340(120)")
        r = Result(100.0, 1.0, "ms")
        self.assertEqual(r.report(), "(100.0 ± 2.0) ms, k = 1.96, p = 0.95")
        lo, hi = r.interval()
        self.assertAlmostEqual(hi - lo, 2 * 1.95996, places=4)

    def test_budget(self):
        bud = Budget("ms").add("a", "A", 3.0, 9).add("b", "B", 4.0)
        self.assertAlmostEqual(bud.u_c(), 5.0)
        self.assertAlmostEqual(bud.dof(), 5.0 ** 4 / (3.0 ** 4 / 9))
        self.assertIn("36.0 %", bud.table())


class PropagateTests(unittest.TestCase):
    def test_lpu_matches_analytic_ratio(self):
        y, u, cu = lpu(lambda x: x[0] / x[1], [10.0, 2.0], [0.1, 0.02])
        self.assertAlmostEqual(y, 5.0)
        self.assertAlmostEqual(u, 5.0 * math.hypot(0.01, 0.01), places=4)
        self.assertAlmostEqual(cu[0], 0.05, places=6)

    def test_lpu_correlation(self):
        f = lambda x: x[0] - x[1]
        _, u, _ = lpu(f, [5.0, 3.0], [1.0, 1.0], {(0, 1): 1.0})
        self.assertAlmostEqual(u, 0.0)

    def test_monte_carlo_agrees_for_a_linear_model(self):
        ys = monte_carlo(lambda x: x[0] + x[1],
                         [normal(1, 0.3), normal(2, 0.4)], 20_000, seed=3)
        mean, sd, lo, hi = summarize(ys)
        self.assertAlmostEqual(mean, 3.0, delta=0.02)
        self.assertAlmostEqual(sd, 0.5, delta=0.02)
        self.assertLess(lo, 3.0 - 0.9)
        self.assertGreater(hi, 3.0 + 0.9)


class StatsTests(unittest.TestCase):
    def test_quantile(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertEqual(quantile(xs, 0.5), 3.0)
        self.assertAlmostEqual(quantile(xs, 0.9), 4.6)
        self.assertEqual(quantile(xs, 1.0), 5.0)

    def test_wilson(self):
        lo, hi = wilson(50, 100)
        self.assertAlmostEqual(lo, 0.4038, places=4)
        self.assertAlmostEqual(hi, 0.5962, places=4)
        lo, hi = wilson(0, 10)
        self.assertAlmostEqual(lo, 0.0, places=12)
        self.assertGreater(hi, 0.25)

    def test_linfit_exact_line(self):
        xs = [0, 1, 2, 3, 4]
        a, b, ua, ub, s = linfit(xs, [1 + 2 * x for x in xs])
        self.assertAlmostEqual(a, 1.0)
        self.assertAlmostEqual(b, 2.0)
        self.assertAlmostEqual(s, 0.0)
        self.assertAlmostEqual(ub, 0.0)

    def test_bootstrap_is_seeded(self):
        xs = [float(i) for i in range(50)]
        mean = lambda v: sum(v) / len(v)
        self.assertEqual(bootstrap(xs, mean, 50, seed=7),
                         bootstrap(xs, mean, 50, seed=7))

    def test_kappa(self):
        a = [1, 1, 0, 0]
        self.assertAlmostEqual(cohen_kappa(a, a), 1.0)
        self.assertAlmostEqual(cohen_kappa(a, [1, 0, 1, 0]), 0.0)


class UnitsTests(unittest.TestCase):
    def test_prefixes(self):
        self.assertAlmostEqual(Q(1, "GiB").to("GB").value, 1.073741824)
        self.assertAlmostEqual(Q(8, "bit").to("B").value, 1.0)
        self.assertAlmostEqual(Q(1500, "ms").to("s").value, 1.5)
        self.assertAlmostEqual((Q(1, "s") + Q(500, "ms")).value, 1.5)

    def test_kinds_do_not_mix(self):
        with self.assertRaises(TypeError):
            Q(1, "ms") + Q(1, "kB")
        with self.assertRaises(ValueError):
            Q(1, "parsec")


if __name__ == "__main__":
    unittest.main()
