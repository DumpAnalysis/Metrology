"""Regression tests added after the review of Version 12.

The review found a false acceptance that began before the checker was
called. A reading written in a file as 9007199254740993.0 came back from
the default JSON reader as the float 9007199254740992.0, the limit, and was
accepted. The checker now has its own reader, which keeps a number as the
decimal that is written, and its validation and its arithmetic take such
decimals. The review also asked that what the checker displays keep an
exact reading and survive extreme scales, that the condition on a positive
duration in Chapter 5 be qualified, and for bounds on a percentile when
some latencies of a capture are unknown.

The cases about reading are complete files, written to a temporary
directory and opened with the loader printed in Listing 22.4. Each test
computes what it expects independently of the code under test.
"""
import itertools
import json
import math
import random
import tempfile
import unittest
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from examples import ch16_bounds, ch22_record_check
from swmetro import record as checker
from swmetro.record import ALL, combined, decide, guard, in_unit, invalid, read, spread
from swmetro.stats import ecdf_quantile, ecdf_rank

ROOT = Path(__file__).resolve().parent.parent
STORED = (ROOT / "data" / "measurement_record.json").read_text(encoding="utf-8")
ROUNDING = ["rounding decides: wants working values"]


def file_text(value, limit, us=("0",), stated=None, dof="16", risk="0.05", digits='"all"', step=None, lost="0"):
    """The text of a complete record file in which the given tokens stand
    exactly as they are passed: 9007199254740993.0 stays those characters."""
    rec = json.loads(STORED)
    rec["components"] = [{"name": f"c{i}", "type": "B", "kind": "stipulated", "u": f"@u{i}@"}
                         for i in range(len(us))]
    rec["result"].update(value="@value@", u="@stated@", dof="@dof@", significant_digits="@digits@")
    rec["decision"].update(limit="@limit@", risk="@risk@")
    rec["capture"]["records_lost"] = "@lost@"
    tokens = {"value": value, "stated": us[0] if stated is None else stated, "dof": dof, "limit": limit,
              "risk": risk, "digits": digits, "lost": lost, **{f"u{i}": u for i, u in enumerate(us)}}
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


class ReadingAFile(unittest.TestCase):
    LIMIT = "9007199254740992"   # 2**53 ns

    def test_the_table_of_the_review(self):
        self.assertEqual(int(self.LIMIT), 2 ** 53)
        for token in ("9007199254740993", "9007199254740993.0", "9.007199254740993e15", "9007199254740993e0",
                      "9.007199254740993E+15", "900719925474099.3e1"):
            rec = opened(file_text(token, self.LIMIT))
            # One nanosecond above the limit, however it is written.
            self.assertEqual(Fraction(Decimal(token)) - int(self.LIMIT), 1, token)
            self.assertEqual(checker.exact(rec["result"]["value"]), 2 ** 53 + 1, token)
            self.assertEqual(decide(rec), ("reject", []), token)
            self.assertEqual(invalid(rec), [], token)

    def test_what_the_default_reader_does_to_the_same_files(self):
        # The digit is lost before any check can see it: this is why the module has a reader.
        for token in ("9007199254740993.0", "9.007199254740993e15"):
            text = file_text(token, self.LIMIT)
            self.assertEqual(json.loads(text)["result"]["value"], 9007199254740992.0)
            self.assertEqual(read(text)["result"]["value"], Decimal(token))
        self.assertEqual(json.loads("0.10000000000000001"), 0.1)

    def test_an_ordinary_scale(self):
        for token, want in (("0.10000000000000001", "reject"), ("0.1000000000000000000000000001", "reject"),
                            ("0.1", "accept"), ("0.09999999999999999", "accept"), ("0.0999999999999999999999", "accept")):
            self.assertEqual(decide(opened(file_text(token, "0.1"))), (want, []), token)

    def test_true_equalities_are_still_accepted(self):
        same = (("9007199254740992", self.LIMIT), ("9007199254740992.0", self.LIMIT),
                ("9.007199254740992e15", self.LIMIT), ("9007199254740992", "9.007199254740992E+15"),
                ("9007199254740992.000", "9007199254740992"), ("0.1", "0.1"), ("0.10", "0.1"), ("1e-1", "0.1"),
                ("0.1", "0.1000000000000000000000000000"), ("100e-3", "1E-1"), ("150", "1.5e2"), ("0", "0.0"), ("-0.0", "0"))
        for value, limit in same:
            self.assertEqual(Decimal(value), Decimal(limit))
            self.assertEqual(decide(opened(file_text(value, limit))), ("accept", []), (value, limit))

    def test_a_reading_with_a_guard_band(self):
        # 1 degree of freedom, risk 0.25: the factor is 1 and the band is the uncertainty itself.
        rows = (("100", "50", "150", "accept"), ("100.0", "50.00", "150", "accept"), ("1e2", "5e1", "1.5E2", "accept"),
                ("100.00000000000000000001", "50", "150", "undecided"), ("100", "50.00000000000000000001", "150", "undecided"),
                ("99.99999999999999999999", "50", "150", "accept"), ("200", "50", "150", "undecided"),
                ("200.00000000000000000001", "50", "150", "reject"), ("0.1", "0.05", "0.15", "accept"),
                ("0.10000000000000000001", "0.05", "0.15", "undecided"))
        for value, u, limit, want in rows:
            for dof, risk in (("1", "0.25"), ("1.0", "0.250"), ("1e0", "25e-2")):
                rec = opened(file_text(value, limit, us=(u,), dof=dof, risk=risk))
                self.assertEqual(decide(rec), (want, []), (value, u, limit, dof, risk))
        # The default reader would have accepted the fourth row: it reads 100.00000000000000000001 as 100.
        self.assertEqual(json.loads("100.00000000000000000001"), 100.0)

    def test_a_rounded_record_from_a_file(self):
        # Limit 150 ms, no uncertainty, estimate rounded to the nearest 100 ms, in ms and in s.
        for value, step, limit in (("100", "100", "150"), ("0.1", "0.1", "0.15"), ("1e-1", "1.0e-1", "15e-2")):
            rec = opened(file_text(value, limit, digits="2", step=step))
            self.assertEqual(decide(rec), ("accept", []), (value, step, limit))
        for value, step, limit in (("200", "100", "150"), ("0.2", "0.1", "0.15"), ("0.20", "0.10", "0.150")):
            rec = opened(file_text(value, limit, digits="2", step=step))
            self.assertEqual(decide(rec), ("unresolved", ROUNDING), (value, step, limit))
        # A longer decimal moves the end of the interval off the limit.
        # (Until Version 55 with an estimate of 0.2, which is no multiple of that increment and is now refused.
        # Three increments, and a limit a little below two and a half of them: as floats the lower end of the
        # interval and the limit are both 0.25.)
        rec = opened(file_text("0.2", "0.15", digits="2", step="0.0999999999999999999999"))
        self.assertEqual(decide(rec), ("invalid", ["result.value is not a multiple of its step"]))
        rec = opened(file_text("0.2999999999999999999997", "0.2499999999999999999997", digits="2",
                               step="0.0999999999999999999999"))
        self.assertEqual(decide(rec), ("reject", []))
        self.assertEqual(float("0.24999999999999999999975"), float("0.2499999999999999999997"))
        # The interval of a rounded uncertainty, from decimals as from floats.
        for x in (Decimal("10"), Decimal("1E+1"), Decimal("10.0"), 10, 10.0):
            self.assertEqual(checker.bounds(x, 2), (Fraction("9.95"), Fraction("10.5")), x)
        for x in (Decimal("9.6"), 9.6):
            self.assertEqual(checker.bounds(x, 2), (Fraction("9.55"), Fraction("9.65")), x)
        long = Decimal("0." + "9" * 45)        # not a power of ten, however close
        self.assertEqual(checker.bounds(long, 2), (Fraction(long) - Fraction("0.005"), Fraction(long) + Fraction("0.005")))

    def test_the_stored_record_reads_as_it_always_did(self):
        kept, floats = read(STORED), json.loads(STORED)
        self.assertEqual(decide(kept), decide(floats))
        self.assertEqual(decide(kept), ("accept", []))
        self.assertEqual((invalid(kept), checker.missing(kept)), ([], []))

        def numbers(part):
            if isinstance(part, dict):
                return list(itertools.chain.from_iterable(numbers(v) for v in part.values()))
            if isinstance(part, list):
                return list(itertools.chain.from_iterable(numbers(v) for v in part))
            if isinstance(part, bool) or not isinstance(part, (int, float, Decimal)):
                return []
            return [part]
        a, b = numbers(kept), numbers(floats)
        self.assertEqual(len(a), len(b))
        self.assertGreater(sum(isinstance(x, Decimal) for x in a), 10)
        # A float that Python wrote is read back as the decimal that Python prints for it.
        for x, y in zip(a, b):
            self.assertEqual(checker.exact(x), checker.exact(y))
        self.assertEqual(guard(kept), guard(floats))
        self.assertEqual(combined(kept), combined(floats))

    def test_validation_takes_decimals_and_keeps_its_rules(self):
        self.assertEqual(invalid(opened(file_text("100.5", "150.25", us=("2.5", "0.3"), stated="2.517935662386948"))), [])
        bad = (
            (dict(value="1", limit="2", us=("-0.5",)), "components[0].u is not a finite number >= 0"),
            (dict(value="1", limit="2", us=("0.5",), stated="-0.5"), "result.u is not a finite number >= 0"),
            (dict(value="1", limit="2", dof="0.0"), "result.dof is not a finite number > 0"),
            (dict(value="1", limit="2", risk="0.5"), "decision.risk is not a number in (0, 0.5)"),
            (dict(value="1", limit="2", risk="0.50000000000000000001"), "decision.risk is not a number in (0, 0.5)"),
            (dict(value="1e999", limit="2"), "result.value is not a finite number"),
            (dict(value="1", limit="2", lost="0.0"), "capture.records_lost is not a count"),
            (dict(value="1", limit="2", digits="2.0", step="0"), "result.significant_digits is not all or 1 to 17"),
            (dict(value="1", limit="2", step="0.1"), "result.value_step is not 0 for working values"),
            (dict(value="1", limit="2", digits="2", step="-0.1"), "result.value_step is not a finite number >= 0"),
            (dict(value="true", limit="2"), "result.value is not a finite number"),
            (dict(value='"1.0"', limit="2"), "result.value is not a finite number"),
        )
        for tokens, reason in bad:
            rec = opened(file_text(**tokens))
            self.assertEqual(decide(rec)[0], "invalid", tokens)
            self.assertIn(reason, decide(rec)[1], tokens)
        # Until Version 53 a file with the bare words NaN or Infinity was read and its record was then found invalid. The
        # reader refuses the words now, since they are not JSON; in a record that never was a file they are invalid as before.
        for tokens, key, reason in ((dict(value="NaN", limit="2"), ("result", "value"), "result.value is not a finite number"),
                                    (dict(value="1", limit="Infinity"), ("decision", "limit"),
                                     "decision.limit is not a finite number")):
            with self.assertRaisesRegex(ValueError, "is not JSON"):
                opened(file_text(**tokens))
            rec = opened(file_text("1", "2"))
            rec[key[0]][key[1]] = float(tokens[{"result": "value", "decision": "limit"}[key[0]]])
            self.assertEqual(decide(rec)[0], "invalid", tokens)
            self.assertIn(reason, decide(rec)[1], tokens)
        # A risk just inside the range is a risk.
        self.assertEqual(invalid(opened(file_text("1", "2", risk="0.49999999999999999999"))), [])

    def test_the_range_of_a_float_is_not_extended(self):
        # The digits of a number are read exactly; its size must still be one that a float can hold.
        for token in ("1e-200", "1e200", "4.9e-324", "1.7e308", "123456789012345678901234567890.5"):
            self.assertEqual(invalid(opened(file_text("1", "2", us=(token,)))), [], token)
        for token in ("1e309", "1e999999999", "1e-400", "1e-999999999", "-1e-400"):
            rec = opened(file_text(token, "2"))
            self.assertEqual(decide(rec), ("invalid", ["result.value is not a finite number"]), token)
        # The default reader would have read the small ones as zero.
        self.assertEqual(json.loads("1e-400"), 0.0)

    def test_nothing_depends_on_the_decimal_context_or_on_how_a_number_prints(self):
        import decimal
        texts = [file_text("9007199254740993.0", "9007199254740992"), file_text("100.00000000000000000001", "150", us=("50",),
                                                                                dof="1", risk="0.25"), STORED]
        before = [(decide(read(t)), guard(read(t)), combined(read(t))) for t in texts]
        with decimal.localcontext() as ctx:
            ctx.prec = 3
            ctx.rounding = decimal.ROUND_UP
            after = [(decide(read(t)), guard(read(t)), combined(read(t))) for t in texts]
        self.assertEqual(before, after)
        self.assertEqual([b[0][0] for b in before], ["reject", "undecided", "accept"])

        class Wrapped(float):                # a float of another library prints as it likes
            def __repr__(self):
                return f"Wrapped({float(self)})"

        class Count(int):
            def __repr__(self):
                return f"Count({int(self)})"
        rec = json.loads(file_text("100", "150", us=("50",), dof="1", risk="0.25"))
        rec["result"].update(value=Wrapped(0.1), u=Wrapped(0.05), dof=Count(1))
        rec["components"][0]["u"] = Wrapped(0.05)
        rec["decision"]["limit"] = Wrapped(0.15)
        self.assertEqual(decide(rec), ("accept", []))
        rec["result"]["value"], rec["decision"]["limit"] = Count(2 ** 53 + 1), Count(2 ** 53)
        rec["result"]["u"] = rec["components"][0]["u"] = 0
        self.assertEqual(decide(rec), ("reject", []))

    def test_a_name_written_twice_is_refused(self):
        text = file_text("100", "150")
        twice = text.replace('"value": 100', '"value": 200,\n  "value": 100')
        self.assertEqual(twice.count('"value"'), text.count('"value"') + 1)
        self.assertEqual(json.loads(twice)["result"]["value"], 100)   # the first statement is gone
        with self.assertRaises(ValueError):
            read(twice)
        with self.assertRaises(ValueError):
            opened(twice)
        self.assertEqual(decide(read(text)), ("accept", []))


class WhatIsShown(unittest.TestCase):
    def test_an_exact_reading_is_shown_exactly(self):
        for token in ("9007199254740993", "9007199254740993.0", "9.007199254740993e15"):
            rec = opened(file_text(token, "9007199254740992"))
            self.assertEqual(guard(rec), ((Decimal(2 ** 53 + 1),) * 2,) * 2, token)
            self.assertEqual(str(guard(rec)[0][0]), "9007199254740993")
        # Built in a program, with the reading as an integer.
        rec = json.loads(file_text("9007199254740993", "9007199254740992"))
        self.assertEqual(str(guard(rec)[0][0]), "9007199254740993")
        self.assertEqual(float(2 ** 53 + 1), 9007199254740992.0)          # what a float would show
        rec = opened(file_text("0.10000000000000001", "0.1"))
        self.assertEqual(str(guard(rec)[0][0]), "0.10000000000000001")
        # A rounded estimate with no band: half its increment either way, exactly.
        rec = opened(file_text("0.2", "0.15", digits="2", step="0.1"))
        self.assertEqual(guard(rec), ((Decimal("0.15"), Decimal("0.25")),) * 2)

    def test_a_band_is_added_to_the_reading_exactly(self):
        # Factor 1: the band is the uncertainty, and the sum has every digit of the reading.
        rec = opened(file_text("9007199254740993", "9007199254740994", us=("0.5",), dof="1", risk="0.25"))
        self.assertEqual(guard(rec), ((Decimal("9007199254740993.5"),) * 2, (Decimal("9007199254740992.5"),) * 2))
        self.assertEqual(decide(rec), ("accept", []))
        # A computed factor: the 17 digits of a float for the band.
        rec = ch22_record_check.altered(read(STORED), {("result", "value"): Decimal("194.6")})
        (lo, hi), (down, _) = guard(rec)
        self.assertEqual(lo, hi)
        band = lo - Decimal("194.6")
        self.assertEqual(down, Decimal("194.6") - band)
        self.assertLessEqual(len(band.as_tuple().digits), 17)
        r = rec["result"]
        k = checker.t_isf(0.05, float(r["dof"]))
        u = math.sqrt(sum(float(c["u"]) ** 2 for c in rec["components"]))
        self.assertAlmostEqual(float(band), k * u, places=12)
        self.assertEqual(f"{lo:.5f}", "199.94976")

    def test_an_unbounded_band(self):
        rec = opened(file_text("50", "100", us=("1",), dof="1e-9"))
        (lo, hi), (down, up) = guard(rec)
        # Until Version 14 all four ends were infinite. Since Version 15 a factor that is not
        # computed has its known lower bound, 1e149, and no upper one.
        self.assertEqual((hi, down), (Decimal("Infinity"), Decimal("-Infinity")))
        # Until Version 70 a guarded value of more than forty digits was rounded to forty for display, here to
        # 1E+149 and -1E+149. Since Version 71 guard() returns such a decimal in full, so that two ends next to a
        # limit stay on their sides of it (tests/test_review70.py): the estimate of 50 is in both.
        self.assertEqual((lo, up), (Decimal(10 ** 149 + 50), Decimal(50 - 10 ** 149)))
        self.assertEqual(decide(rec), ("undecided", []))

    def test_extreme_scales(self):
        for token, shown in (("1e-200", "1E-200"), ("1e200", "1E+200"), ("1E-300", "1E-300"), ("2.5e-170", "2.5E-170")):
            rec = opened(file_text("1", "2", us=(token,)))
            self.assertEqual(spread(rec), (Decimal(token),) * 2, token)
            self.assertEqual(combined(rec), (Decimal(token),) * 2, token)
            self.assertEqual(str(combined(rec)[0]), shown)
            # What a float square would have left of them.
            self.assertIn(float(token) * float(token), (0.0, math.inf))
        for us, root in ((("3e-200", "4e-200"), "5E-200"), (("3e200", "4e200"), "5E+200"), (("3", "4"), "5"),
                         (("300", "400"), "500"), (("0.3", "0.4"), "0.5")):
            rec = opened(file_text("1", "2", us=us, stated=root))
            self.assertEqual(str(combined(rec)[0]), root, us)
            self.assertEqual(invalid(rec), [])
            self.assertIn(decide(rec)[0], ("accept", "reject", "undecided"), us)
        # Roots that are not decimals are shown to 40 digits, and they are those of the root.
        rec = opened(file_text("1", "2", us=("1", "1"), stated="1.4142135623730951"))
        root = combined(rec)[0]
        self.assertIn(len(root.as_tuple().digits), (39, 40))   # 40, less a final zero
        self.assertAlmostEqual(float(root), math.sqrt(2), places=15)
        self.assertLess(abs(Fraction(root) ** 2 - 2), Fraction(1, 10 ** 38))

    def test_a_conversion_of_unit_is_exact(self):
        other = dict(ch22_record_check.OTHER)
        self.assertEqual(in_unit(other, "ms"), (Decimal("241.4"), None))
        self.assertEqual(str(in_unit(other, "ms")[0]), "241.4")
        # The product of two floats need not be the converted decimal; the exact one is.
        other["result"] = {"value": 0.1821, "u": 0.00306}
        self.assertEqual((0.1821 * 1e3, 0.00306 * 1e3), (182.10000000000002, 3.0599999999999996))
        self.assertEqual(in_unit(other, "ms"), (Decimal("182.1"), Decimal("3.06")))
        rec = read(file_text("9007199254740993", "9007199254740992", us=("0.25",)))
        rec["measurand"]["unit"] = "ns"
        self.assertEqual(in_unit(rec, "ms"), (Decimal("9007199254.740993"), Decimal("0.00000025")))
        self.assertEqual(in_unit(rec, "ns"), (Decimal("9007199254740993"), Decimal("0.25")))
        with self.assertRaises(TypeError):
            in_unit(rec, "MiB")


class Allowance(unittest.TestCase):
    def test_the_figures_of_the_text(self):
        # Version 13 divided the allowance of one part in a million by 1 - 2 risk (1.11 parts at a
        # risk of 0.05, 50 at 0.49), because the factor next to one half was then known less well.
        # Since Version 14 it is computed from the distance to one half, and the allowance is one
        # part in a million at every risk.
        self.assertEqual(checker.DELTA, Fraction(1, 10 ** 6))
        for risk in ("0.001", "0.05", "0.25", "0.49", "0.4999999999"):
            k, low, high = checker._factor(Fraction(risk), Fraction(16))
            self.assertEqual((low, high), (k * (1 - checker.DELTA), k * (1 + checker.DELTA)), risk)
            self.assertGreater(low, 0)
        # The factor that is known has no allowance.
        self.assertEqual(checker._factor(Fraction(1, 4), Fraction(1)), (1, 1, 1))


class Reciprocal(unittest.TestCase):
    def test_a_positive_duration_can_still_have_no_mean_reciprocal(self):
        # Uniform on (0, 1): the integral of 1/x over each halving, from 2^-(k+1) to 2^-k, is ln 2.
        # The pieces are equal and there is no end of them.
        for k in (0, 1, 10, 100, 1000):
            a, b = Fraction(1, 2 ** (k + 1)), Fraction(1, 2 ** k)
            steps = 2000
            h = (b - a) / steps
            simpson = sum((1 / (a + i * h)) * (1 if i in (0, steps) else 4 if i % 2 else 2)
                          for i in range(steps + 1)) * h / 3
            self.assertAlmostEqual(float(simpson), math.log(2), places=12)
        # A seeded sample shows nothing of it: its mean is finite, as every sample mean is.
        rng = random.Random(5)
        self.assertTrue(math.isfinite(sum(1 / (1 - rng.random()) for _ in range(1000)) / 1000))

    def test_the_reciprocal_of_a_lognormal_duration_has_two_moments(self):
        m, s = 655.0, 15.0
        v = math.log(1 + (s / m) ** 2)
        mu = math.log(m) - v / 2
        # 1/Y is lognormal with parameters -mu and v.
        first, second = math.exp(-mu + v / 2), math.exp(-2 * mu + 2 * v)
        self.assertAlmostEqual(first * m, 1 + (s / m) ** 2, places=12)
        self.assertAlmostEqual(second * m * m, (1 + (s / m) ** 2) ** 3, places=12)
        self.assertTrue(0 < first < math.inf and first ** 2 < second < math.inf)


class MissingLatencies(unittest.TestCase):
    def test_the_ranks_of_the_transcript(self):
        self.assertEqual(ecdf_rank(0.95, 1000), 950)
        self.assertEqual(ch16_bounds.rank_bounds(1000, 960, 0.95), (910, 950))
        self.assertEqual(ch16_bounds.rank_bounds(1000, 949, 0.95), (899, None))
        self.assertEqual(ch16_bounds.rank_bounds(1000, 950, 0.95), (900, 950))
        self.assertEqual(ch16_bounds.rank_bounds(1000, 1000, 0.95), (950, 950))
        self.assertEqual(ch16_bounds.rank_bounds(1000, 50, 0.95), (None, None))
        text = (ROOT / "expected" / "ch16_bounds.txt").read_text(encoding="utf-8")
        self.assertIn("  960 captured: rank 910 to rank 950\n", text)
        self.assertIn("  949 captured: rank 899, no upper bound\n", text)

    def test_the_bounds_hold_for_any_loss_and_are_reached(self):
        rng = random.Random(16)
        for trial in range(200):
            total = rng.choice((20, 100, 1000))
            lost = rng.randrange(0, total // 10 + 1)
            q = rng.choice((0.5, 0.9, 0.95, 0.99))
            every = sorted(round(rng.lognormvariate(4, 1), 3) for _ in range(total))
            truth = ecdf_quantile(every, q)
            low, high = ch16_bounds.rank_bounds(total, total - lost, q)
            losses = [rng.sample(range(total), lost),                # at random
                      list(range(lost)),                             # the shortest
                      list(range(total - lost, total)),              # the longest
                      sorted(range(total), key=lambda i: abs(every[i] - truth))[:lost]]   # around the quantile
            for gone in losses:
                kept = [x for i, x in enumerate(every) if i not in set(gone)]
                if low is not None:
                    self.assertLessEqual(kept[low - 1], truth, (trial, total, lost, q))
                if high is not None:
                    self.assertGreaterEqual(kept[high - 1], truth, (trial, total, lost, q))
            if low is not None:      # reached when the shortest are the ones lost
                self.assertEqual(every[lost:][low - 1], truth)
            if high is not None:     # reached when the longest are the ones lost
                self.assertEqual(every[:total - lost][high - 1], truth)

    def test_fewer_captured_than_the_rank_leave_no_upper_bound(self):
        captured = [float(i) for i in range(1, 950)]                 # 949 latencies, up to 949 ms
        self.assertEqual(ch16_bounds.rank_bounds(1000, len(captured), 0.95), (899, None))
        for longest in (1e3, 1e6, 1e12):                             # the 51 unknown ones can be anything
            every = sorted(captured + [longest] * 51)
            self.assertEqual(ecdf_quantile(every, 0.95), longest)
            self.assertGreater(ecdf_quantile(every, 0.95), max(captured))
        self.assertEqual(ecdf_quantile(sorted([0.0] * 51 + captured), 0.95), captured[899 - 1])

    def test_an_objective_of_200_ms(self):
        def verdict(captured, total=1000, limit=200.0):
            low, high = ch16_bounds.rank_bounds(total, len(captured), 0.95)
            if high is not None and captured[high - 1] <= limit:
                return "met"
            if low is not None and captured[low - 1] > limit:
                return "missed"
            return "unresolved"
        fast = sorted([100.0] * 950 + [900.0] * 10)                  # the 950th captured is 100 ms
        slow = sorted([100.0] * 909 + [900.0] * 51)                  # the 910th captured is 900 ms
        between = sorted([100.0] * 930 + [900.0] * 30)               # 910th at 100 ms, 950th at 900 ms
        self.assertEqual((verdict(fast), verdict(slow), verdict(between)), ("met", "missed", "unresolved"))
        self.assertEqual(verdict(sorted([100.0] * 949)), "unresolved")   # 949 captured: never "met"


if __name__ == "__main__":
    unittest.main()
