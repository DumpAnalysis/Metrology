"""Regression tests added after the accuracy review of Version 4.

The review found no code defect. These tests fix, in executable form, the
two facts on which its wording corrections rest, and the two-line example
that the book now prints in Chapter 9.
"""
import random
import unittest
from fractions import Fraction

from examples import ch08_selection as c8
from examples import ch22_budget as c22
from swmetro.stats import ecdf_quantile, ecdf_rank


class SelectionSums(unittest.TestCase):
    def test_slow_numerator_is_fixed_and_total_weight_varies(self):
        rng = random.Random(8)                     # the record of the example
        rows = c8.requests(40_000, rng)
        slow = sum(1 for x, hit in rows if not hit and x > c8.SLOW)
        self.assertEqual(slow, 368)
        totals = set()
        for _ in range(20):
            sample = c8.kept(rows, rng)
            weighted_slow = sum(1 / p for x, p in sample if x > c8.SLOW)
            self.assertEqual(weighted_slow, slow)  # every slow trace is kept, weight 1
            totals.add(sum(1 / p for _, p in sample))
        self.assertGreater(len(totals), 10)        # the denominator does vary


class AtMost(unittest.TestCase):
    def test_counts_at_the_threshold_do_not_establish_below(self):
        xs = [c22.LIMIT] * 40                      # every request takes exactly 200 ms
        share = sum(x <= c22.LIMIT for x in xs) / len(xs)
        self.assertEqual(share, 1.0)
        self.assertEqual(ecdf_quantile(xs, 0.95), c22.LIMIT)       # at most, and not below
        self.assertFalse(ecdf_quantile(xs, 0.95) < c22.LIMIT)
        lo, hi = c22.bracket(xs, c22.FITTED)
        self.assertEqual((lo, hi), (100.0, 200.0))                 # the bucket is (100, 200]
        self.assertEqual(c22.verdict(lo, hi), "met")               # "p95 <= 200 ms"


class PrintedSnippet(unittest.TestCase):
    def test_the_two_readings_printed_in_chapter_9(self):
        self.assertEqual(ecdf_rank(0.9, 10), 9)
        self.assertEqual(ecdf_rank(Fraction(0.9), 10), 10)
        self.assertGreater(Fraction(0.9), Fraction(9, 10))


if __name__ == "__main__":
    unittest.main()
