"""Regression tests added after the accuracy review of Version 3.

The subject is the reading of a probability and the rank that follows from it.
Every expected rank is found from the definition by exact rational arithmetic.
"""
import hashlib
import json
import math
import random
import unittest
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from examples import ch08_counter, ch08_selection
from examples import ch09_histogram as h9
from examples import ch22_budget
from swmetro.stats import ecdf_quantile, ecdf_rank, exact_q
from swmetro.stats import lower_rank, order_stat, quantile

ROOT = Path(__file__).resolve().parent.parent
TINY = Fraction(1, 10 ** 30)


def rank_by_definition(q, n):
    """Smallest k in 1..n with k / n >= q, for an exact q."""
    return next((k for k in range(1, n + 1) if Fraction(k, n) >= q), n)


class ReadingOfProbability(unittest.TestCase):
    def test_float_is_read_as_its_printed_decimal(self):
        self.assertEqual(exact_q(0.95), Fraction(19, 20))
        self.assertEqual(exact_q(0.9), Fraction(9, 10))
        self.assertEqual(exact_q(1e-5), Fraction(1, 100000))
        up = math.nextafter(0.5, 1.0)
        self.assertEqual(exact_q(up), Fraction("0.5000000000000001"))
        self.assertGreater(exact_q(up), Fraction(1, 2))

    def test_exact_types_are_taken_as_they_stand(self):
        self.assertEqual(exact_q(Fraction(2, 3)), Fraction(2, 3))
        self.assertEqual(exact_q("0.95"), Fraction(19, 20))
        self.assertEqual(exact_q(Decimal("0.999")), Fraction(999, 1000))
        self.assertEqual(exact_q(1), 1)
        # The binary number nearest to 0.9 lies above nine tenths.
        self.assertGreater(exact_q(Fraction(0.9)), Fraction(9, 10))

    def test_invalid_probabilities_are_rejected(self):
        for bad in (-0.1, 1.0000001, math.nan, math.inf, "x", Fraction(3, 2)):
            with self.assertRaises(ValueError):
                exact_q(bad)
        with self.assertRaises(ValueError):
            ecdf_rank(0.5, 0)


class RankBoundary(unittest.TestCase):
    def test_reviewers_reproducer(self):
        q = math.nextafter(0.5, 1.0)
        xs = [100.0, 5000.0]
        self.assertEqual(ecdf_rank(q, 2), 2)
        self.assertEqual(ecdf_quantile(xs, q), 5000.0)
        table = {0.5: 1, 0.5000000000000001: 2, 0.5000000000001: 2, 0.500000000001: 2}
        for q, want in table.items():
            self.assertEqual(ecdf_rank(q, 2), want, q)

    def test_at_and_on_either_side_of_every_boundary(self):
        for n in range(1, 41):
            for j in range(1, n + 1):
                q = Fraction(j, n)
                self.assertEqual(ecdf_rank(q, n), j)
                self.assertEqual(ecdf_rank(q - TINY, n), j)
                if j < n:
                    self.assertEqual(ecdf_rank(q + TINY, n), j + 1)

    def test_adjacent_floats_at_decimal_boundaries(self):
        for n in (2, 4, 5, 8, 10, 20, 25, 50, 100, 1000, 2000):
            for j in range(1, n):
                q = j / n
                if exact_q(q) != Fraction(j, n):
                    continue                      # j / n has no short decimal form
                self.assertEqual(ecdf_rank(q, n), j, (j, n))
                self.assertEqual(ecdf_rank(math.nextafter(q, 1.0), n), j + 1, (j, n))
                self.assertEqual(ecdf_rank(math.nextafter(q, 0.0), n), j, (j, n))

    def test_rank_meets_the_coverage_condition(self):
        rng = random.Random(1)
        for _ in range(3000):
            n = rng.randint(1, 500)
            q = rng.choice((rng.random(), rng.randint(0, n) / n,
                            math.nextafter(rng.randint(0, n - 1) / n, 1.0),
                            math.nextafter(rng.randint(1, n) / n, 0.0)))
            k, f = ecdf_rank(q, n), exact_q(q)
            self.assertEqual(k, rank_by_definition(f, n), (q, n))
            self.assertGreaterEqual(Fraction(k, n), f)
            self.assertTrue(k == 1 or Fraction(k - 1, n) < f)

    def test_the_two_readings_are_not_interchanged(self):
        self.assertEqual(ecdf_rank(0.9, 10), 9)                 # nine tenths
        self.assertEqual(ecdf_rank("0.9", 10), 9)
        self.assertEqual(ecdf_rank(Fraction(0.9), 10), 10)      # the binary number
        self.assertEqual(ecdf_rank(Fraction(0.95), 20), 19)     # lies below 19/20

    def test_earlier_cases_are_retained(self):
        self.assertEqual(ecdf_rank(0.95, 20), 19)
        self.assertEqual(ecdf_rank(0.95, 2000), 1900)
        for n in (1, 2, 7, 400):
            self.assertEqual(ecdf_rank(0, n), 1)
            self.assertEqual(ecdf_rank(0.0, n), 1)
            self.assertEqual(ecdf_rank(1, n), n)
            self.assertEqual(ecdf_rank(1.0, n), n)
        xs = [100.0] * 19 + [5000.0]
        self.assertEqual(ecdf_quantile(xs, 0.95), 100.0)


class OtherConventions(unittest.TestCase):
    def test_lower_rank_is_an_exact_floor(self):
        # 0.29 * 100 is 28.999999999999996 in binary arithmetic.
        self.assertEqual(int(0.29 * 100), 28)
        self.assertEqual(lower_rank(0.29, 101), 29)
        xs = [float(i) for i in range(101)]
        self.assertEqual(order_stat(xs, 0.29), 29.0)
        for n in range(1, 60):
            for num in range(0, 101):
                q = Fraction(num, 100)
                want = (q * (n - 1)).numerator // (q * (n - 1)).denominator
                self.assertEqual(lower_rank(num / 100, n), want)

    def test_sketch_and_comparator_share_the_rank(self):
        xs = [1.5 ** i for i in range(101)]           # values far apart
        sketch = h9.sketch_of(xs, 0.01)
        for q in (0.29, 0.57, 0.58, 0.5, 0.99):
            exact = order_stat(xs, q)
            self.assertLessEqual(abs(sketch.quantile(q) - exact) / exact, 0.0101, q)

    def test_linear_quantile_is_continuous_at_a_rank(self):
        xs = [float(i * i) for i in range(101)]
        self.assertEqual(quantile(xs, 0.29), xs[29])
        below = quantile(xs, math.nextafter(0.29, 0.0))
        above = quantile(xs, math.nextafter(0.29, 1.0))
        self.assertAlmostEqual(below, xs[29], places=9)
        self.assertAlmostEqual(above, xs[29], places=9)

    def test_bucket_estimate_at_a_boundary_before_an_empty_bucket(self):
        xs = [0.5] * 19 + [7.0]                        # buckets (0, 1] and (5, 10]
        bounds = [1.0, 5.0, 10.0]
        self.assertEqual(h9.bucket_quantile(xs, 0.95, bounds), 1.0)
        above = h9.bucket_quantile(xs, math.nextafter(0.95, 1.0), bounds)
        self.assertAlmostEqual(above, 5.0, places=9)
        self.assertGreater(above, 5.0)


class Additions(unittest.TestCase):
    def test_restart_without_a_decrease(self):
        ys = [1000, 1500]
        self.assertEqual(ch08_counter.rates(ys, 10, "naive"), [50.0])
        self.assertEqual(ch08_counter.rates(ys, 10, "reset"), [50.0])
        self.assertEqual(ch08_counter.rates(ys, 10, "reset", {0}), [150.0])

    def test_weighted_quantile_agrees_with_the_shared_rank(self):
        rng = random.Random(2)
        for _ in range(200):
            xs = [float(rng.randint(1, 500)) for _ in range(rng.randint(1, 80))]
            got = ch08_selection.summary([(x, 1.0) for x in xs])[1]
            self.assertEqual(got, ecdf_quantile(sorted(xs), 0.95))

    def test_selection_alone_makes_the_estimates_vary(self):
        rng = random.Random(3)
        rows = ch08_selection.requests(20_000, rng)
        truth = ch08_selection.summary([(x, 1.0) for x, hit in rows if not hit])[0]
        means = ch08_selection.redrawn(rows, rng, 120)[0]
        mean = sum(means) / len(means)
        sd = math.sqrt(sum((m - mean) ** 2 for m in means) / (len(means) - 1))
        self.assertGreater(sd, 1.0)                       # a real spread, in ms
        self.assertLess(abs(mean - truth), 4 * sd / math.sqrt(len(means)))

    def test_probability_line_names_its_target(self):
        text = (ROOT / "expected" / "ch22_budget.txt").read_text(encoding="utf-8")
        self.assertIn("P(mean run p95 <= 200 ms)", text)
        self.assertNotIn("P(p95 <=", text)
        # (0.0, 100.0) until Version 53: the first bucket has no lower end.
        self.assertEqual(ch22_budget.bracket([100.0] * 19 + [5000.0], [100, 200]), (-math.inf, 100.0))


class RecordProvenance(unittest.TestCase):
    def test_record_carries_origin_and_fingerprint(self):
        rec = json.loads((ROOT / "data" / "measurement_record.json").read_text(encoding="utf-8"))
        between = [c for c in rec["components"] if c["name"] == "between runs"]
        self.assertEqual([c["kind"] for c in between], ["stipulated"])
        src = rec["sources"]
        self.assertIn("data/make_record.py", src["files"])
        self.assertEqual(len(src["files"]), len(src["sha256_16_of_each"]))
        whole = hashlib.sha256()
        for rel, short in zip(src["files"], src["sha256_16_of_each"]):
            text = (ROOT / rel).read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")
            self.assertEqual(hashlib.sha256(text).hexdigest()[:16], short, rel)
            whole.update(text)
        self.assertEqual(whole.hexdigest(), "".join(src["sha256"]))
        self.assertTrue(src["release"])


if __name__ == "__main__":
    unittest.main()
