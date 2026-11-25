"""Regression tests added after the review of Version 7.

The review found that the record checker accepted records it should have
refused, that two records with unknown clocks compared as equal, that the
second record of the example was a partly converted copy, and that the test
of the margin decisions read a saved transcript.

The review of Version 8 then withdrew the 5 % tolerance between the stated
combined uncertainty and its components. The two tests of that tolerance
were replaced; tests/test_review8.py has the cases of that review.

Each test here states what must be refused, or computes the expected value
independently of the code under test: a quantile from a published table, a
root sum of squares, a normal quantile from the standard library, a
percentile by its definition.
"""
import copy
import json
import math
import random
import statistics as st
import unittest
from pathlib import Path

from examples import ch22_record_check, ch23_case
from swmetro import Result
from swmetro import record as checker
from swmetro.record import comparable, decide, in_unit, invalid, is_unknown, missing
from swmetro.stats import bootstrap

ROOT = Path(__file__).resolve().parent.parent
DECISIONS = ("accept", "reject", "undecided")


def load(name):
    return json.loads((ROOT / "data" / name).read_text(encoding="utf-8"))


def edit(rec, changes):
    """A copy of a record with the values at the given paths replaced."""
    out = copy.deepcopy(rec)
    for path, value in changes.items():
        part = out
        for key in path[:-1]:
            part = part[key]
        part[path[-1]] = value
    return out


def simple(value, u, dof=10, limit=100.0, risk=0.05):
    """A complete record with one component, for arithmetic checks."""
    rec = load("measurement_record.json")
    rec["components"] = [{"name": "only", "type": "A", "kind": "observed", "u": u}]
    rec["result"].update(value=value, u=u, dof=dof)
    rec["decision"].update(limit=limit, risk=risk)
    return rec


class Refusals(unittest.TestCase):
    def setUp(self):
        self.rec = load("measurement_record.json")

    def test_the_stored_record_is_accepted_and_says_so(self):
        self.assertEqual(decide(self.rec), ("accept", []))
        self.assertEqual(self.rec["decision"]["outcome"], "accept")
        self.assertEqual(self.rec["format"], checker.FORMAT)
        self.assertEqual(self.rec["decision"]["procedure"], checker.PROCEDURE)

    def test_the_five_rows_of_the_review(self):
        rows = (
            ({("result", "value"): 212, ("result", "u"): -10}, "invalid", "result.u"),
            ({("decision", "rule"): None}, "unresolved", "wants decision.rule"),
            ({("decision", "rule"): "accept if value - k u >= limit"}, "invalid", "decision.rule"),
            ({("components", 1, "u"): None}, "unresolved", "wants components[1].u"),
            ({("capture", "records_lost"): -1}, "invalid", "capture.records_lost"),
        )
        self.assertEqual(self.rec["components"][1]["name"], "between runs")
        for changes, status, reason in rows:
            got, why = decide(edit(self.rec, changes))
            self.assertEqual(got, status, changes)
            self.assertTrue(any(reason in w for w in why), (changes, why))

    def test_no_corrupted_number_reaches_a_decision(self):
        never = [float("nan"), float("inf"), -float("inf"), True, False, "3", [1], {"v": 1}, 10 ** 400]
        fields = {
            ("result", "value"): [],
            ("result", "u"): [-1e-9, -10],
            ("result", "dof"): [0, -1, -0.5],
            ("decision", "limit"): [],
            ("decision", "risk"): [0, 0.5, 0.7, 1, -0.1],
            ("capture", "records_lost"): [-1, 0.5, 1.0, 0.0],
            ("components", 0, "u"): [-1e-9],
            ("components", 2, "value"): [],
        }
        for path, more in fields.items():
            for bad in never + more:
                got, why = decide(edit(self.rec, {path: bad}))
                self.assertNotIn(got, DECISIONS, (path, bad))
                self.assertTrue(why, (path, bad))
                # An absurd count of lost records is still a count: unresolved.
                absurd = path == ("capture", "records_lost") and bad == 10 ** 400
                self.assertEqual(got, "unresolved" if absurd else "invalid", (path, bad))

    def test_unknown_never_decides_and_is_never_zero(self):
        for unknown in (None, "", "   ", "<a prompt>"):
            self.assertTrue(is_unknown(unknown))
        for known in (0, 0.0, False, "0", [], "x"):
            self.assertFalse(is_unknown(known), known)
        for section, keys in checker.REQUIRED:
            for key in keys.split():
                for unknown in (None, "", "<a prompt>"):
                    rec = edit(self.rec, {(section, key): unknown})
                    got, why = decide(rec)
                    self.assertEqual(got, "unresolved", (section, key, unknown))
                    self.assertIn(f"wants {section}.{key}", why)
                    self.assertIn(f"{section}.{key}", missing(rec))
        for name in ("measurand", "capture", "result", "decision") + checker.LISTS:
            rec = {k: v for k, v in self.rec.items() if k != name}
            self.assertEqual(decide(rec)[0], "unresolved", name)
        for name in checker.LISTS:
            rec = edit(self.rec, {(name,): None})
            self.assertIn(f"wants {name}", decide(rec)[1])

    def test_every_component_is_an_input(self):
        for i, comp in enumerate(self.rec["components"]):
            for key in ("name", "u"):
                got, why = decide(edit(self.rec, {("components", i, key): None}))
                self.assertEqual((got, why), ("unresolved", [f"wants components[{i}].{key}"]))
            if "value" in comp:
                got, why = decide(edit(self.rec, {("components", i, "value"): None}))
                self.assertEqual((got, why), ("unresolved", [f"wants components[{i}].value"]))
        self.assertEqual(decide(edit(self.rec, {("components",): []}))[0], "unresolved")

    def test_combined_uncertainty_must_follow_from_the_components(self):
        us = [c["u"] for c in self.rec["components"]]
        root = math.sqrt(sum(u * u for u in us))
        self.assertAlmostEqual(self.rec["result"]["u"], root, places=9)
        # A component changed, or removed, with the combined value kept.
        stale = edit(self.rec, {("components", 1, "u"): 5.0})
        self.assertEqual(decide(stale), ("invalid", ["result.u does not follow from the components"]))
        fewer = edit(self.rec, {("components",): self.rec["components"][:1]})
        self.assertEqual(decide(fewer)[0], "invalid")
        # 3 and 4 combine to 5. The record holds working values, so the stated
        # value must be that number; the rounded case is in test_review8.py.
        for stated, ok in ((5.0, True), (5.0 + 1e-12, True), (5.2, False), (4.8, False), (0.0, False)):
            rec = simple(90.0, stated)
            rec["components"] = [{"name": "a", "u": 3.0}, {"name": "b", "u": 4.0}]
            self.assertEqual(decide(rec)[0] != "invalid", ok, stated)
        self.assertEqual(decide(simple(90.0, 0.0))[0], "accept")

    def test_guard_band_against_published_quantiles(self):
        # Student t, 10 degrees of freedom: 1.812 at 0.95 and 2.228 at 0.975.
        cases = (
            (0.05, 98.18, "accept"), (0.05, 98.19, "undecided"),
            (0.05, 101.81, "undecided"), (0.05, 101.82, "reject"),
            (0.025, 97.77, "accept"), (0.025, 97.78, "undecided"),
            (0.025, 102.22, "undecided"), (0.025, 102.23, "reject"),
        )
        for risk, value, want in cases:
            self.assertEqual(decide(simple(value, 1.0, risk=risk)), (want, []), (risk, value))
        # The rule accepts at equality and rejects only above the limit.
        self.assertEqual(decide(simple(100.0, 0.0))[0], "accept")
        self.assertEqual(decide(simple(100.0 + 1e-9, 0.0))[0], "reject")
        # A coverage factor too large to compute decides nothing unless u is zero.
        self.assertEqual(decide(simple(50.0, 1.0, dof=1e-9))[0], "undecided")
        self.assertEqual(decide(simple(50.0, 1.0, risk=1e-300))[0], "undecided")
        self.assertEqual(decide(simple(50.0, 0.0, dof=1e-9))[0], "accept")

    def test_what_blocks_a_decision_and_what_does_not(self):
        blocks = (
            ({("not_evaluated",): ["warm-up drift"]}, "not evaluated: warm-up drift"),
            ({("covariances",): [{"between": ["a", "b"], "r": None}]}, "correlations not supported"),
            ({("covariances",): [{"between": ["a", "b"], "r": 0.0}]}, "correlations not supported"),
            ({("decision", "procedure"): "lower limit, t guard band, v1"}, "procedure not supported"),
            ({("format",): "measurement record, version 2"}, "format not supported"),
            ({("capture", "records_lost"): 3}, "records were lost"),
        )
        for changes, reason in blocks:
            self.assertEqual(decide(edit(self.rec, changes)), ("unresolved", [reason]), changes)
        # Assumptions and things outside the claim qualify a decision; they do not veto it.
        more = edit(self.rec, {("assumed",): self.rec["assumed"] + ["one more assumption"],
                               ("not_claimed",): self.rec["not_claimed"] + ["another load"]})
        self.assertEqual(decide(more), ("accept", []))
        self.assertTrue(self.rec["assumed"] and self.rec["not_claimed"])
        self.assertEqual(self.rec["not_evaluated"], [])

    def test_statements_must_be_those_of_the_named_procedure(self):
        for section, key, text in checker.EXPECTS:
            self.assertEqual(self.rec[section][key], text)
            got, why = decide(edit(self.rec, {(section, key): text + " "}))
            self.assertEqual((got, why), ("invalid", [f"{section}.{key} is not that of the procedure"]))

    def test_template_and_description_follow_the_checker(self):
        blank = load("record_template.json")
        self.assertEqual(blank["format"], checker.FORMAT)
        self.assertEqual(decide(blank)[0], "unresolved")
        self.assertEqual(invalid(blank), [])
        for key in self.rec:
            if key not in ("replay",):
                self.assertIn(key, blank)
        text = (ROOT / "data" / "RECORD_FORMAT.md").read_text(encoding="utf-8")
        for needed in (checker.FORMAT, checker.PROCEDURE) + tuple(t for _, _, t in checker.EXPECTS):
            self.assertIn(needed, text)
        for status in DECISIONS + ("unresolved", "invalid"):
            self.assertIn(f"`{status}`", text)


class Clocks(unittest.TestCase):
    def setUp(self):
        self.rec = load("measurement_record.json")
        self.here = self.rec["capture"]["clock"]

    def relation(self, clocks, of="rate", state="established"):
        return {"clocks": clocks, "of": of, "state": state, "evidence": "stated for the test"}

    def test_unknown_clocks_are_not_equal_clocks(self):
        for unknown in (None, "", "  ", "<which clock>"):
            a = edit(self.rec, {("capture", "clock"): unknown})
            self.assertEqual(comparable(a, copy.deepcopy(a))[0], ["clock unknown"], unknown)
            self.assertEqual(comparable(self.rec, a)[0], ["clock unknown"], unknown)
        bare = {k: v for k, v in self.rec.items() if k != "capture"}
        self.assertEqual(comparable(self.rec, bare)[0], ["clock unknown"])

    def test_a_relation_must_name_both_clocks_and_concern_their_rates(self):
        there = "different host, wall clock"
        other = edit(self.rec, {("capture", "clock"): there})
        self.assertEqual(comparable(self.rec, other)[0], ["clock rate relation"])
        fails = (
            [self.relation([self.here, "a third clock"])],
            [self.relation([there, "a third clock"])],
            [self.relation([self.here])],
            [self.relation([self.here, there], of="offset")],
            [self.relation([self.here, there], state="not established")],
            [self.relation([self.here, there], state=None)],
            ["not needed, one clock"],
        )
        for relations in fails:
            b = edit(other, {("capture", "clock_relations"): relations})
            self.assertEqual(comparable(self.rec, b)[0], ["clock rate relation"], relations)
            self.assertEqual(comparable(b, self.rec)[0], ["clock rate relation"], relations)
        for pair in ([self.here, there], [there, self.here]):
            b = edit(other, {("capture", "clock_relations"): [self.relation(pair)]})
            self.assertEqual(comparable(self.rec, b), ([], []))
            self.assertEqual(comparable(b, self.rec), ([], []))
            a = edit(self.rec, {("capture", "clock_relations"): [self.relation(pair)]})
            self.assertEqual(comparable(a, other), ([], []))

    def test_one_clock_needs_no_relation(self):
        self.assertEqual(self.rec["capture"]["clock_relations"], [])
        self.assertEqual(comparable(self.rec, copy.deepcopy(self.rec)), ([], []))


class SecondRecord(unittest.TestCase):
    def test_it_is_written_afresh_and_carries_nothing_stale(self):
        other = ch22_record_check.OTHER
        self.assertEqual(set(other), {"format", "measurand", "data", "capture", "result"})
        self.assertEqual(set(other["result"]), {"value", "u"})
        self.assertEqual(other["measurand"]["unit"], "s")
        self.assertIsNone(other["result"]["u"])
        self.assertEqual(decide(other)[0], "unresolved")
        value, u = in_unit(other, "ms")
        self.assertAlmostEqual(float(value), 241.4, places=9)   # a decimal since Version 13
        self.assertIsNone(u)

    def test_comparison_with_the_stored_record(self):
        rec = load("measurement_record.json")
        reasons, notes = comparable(rec, ch22_record_check.OTHER)
        self.assertEqual(reasons, ["start_event", "convention", "clock rate relation"])
        self.assertEqual(notes, ["1 s = 1000 ms"])

    def test_an_altered_copy_keeps_its_stale_outcome_and_the_checker_ignores_it(self):
        rec = load("measurement_record.json")
        copy_ = ch22_record_check.altered(rec, {("capture", "records_lost"): None})
        self.assertEqual(copy_["decision"]["outcome"], "accept")
        self.assertEqual(decide(copy_)[0], "unresolved")
        self.assertEqual(rec["capture"]["records_lost"], 0)


class Margin(unittest.TestCase):
    Z = st.NormalDist().inv_cdf(0.95)

    def test_strictly_smaller_than_the_margin(self):
        for upper, margin, want in ((4.999, 5.0, True), (5.0, 5.0, False), (5.001, 5.0, False),
                                    (1.999, 2.0, True), (2.0, 2.0, False), (-1.0, 0.0, True)):
            self.assertIs(ch23_case.within(upper, margin), want, (upper, margin))

    def test_upper_bound_is_the_normal_one_sided_bound(self):
        self.assertAlmostEqual(ch23_case.upper_bound(Result(1.0, 2.0, "ms")), 1.0 + self.Z * 2.0, places=9)
        self.assertAlmostEqual(ch23_case.upper_bound(Result(-3.0, 0.0, "ms")), -3.0, places=12)

    def test_the_decisions_of_the_case_recomputed(self):
        def p95(xs):
            """The definition: the smallest value with at least 95 % at or below it."""
            xs = sorted(xs)
            for x in xs:
                if 100 * sum(v <= x for v in xs) >= 95 * len(xs):
                    return x

        rng = random.Random(23)
        before, after = ch23_case.period(rng), ch23_case.period(rng)
        hb = [h for _, h in before]
        ha = [h for _, h in after]
        d = p95(ha) - p95(hb)
        u = math.hypot(st.stdev(bootstrap(ha, ch23_case.p95, 300, 23)),
                       st.stdev(bootstrap(hb, ch23_case.p95, 300, 23)))
        upper = d + self.Z * u
        got = ch23_case.handler_change(before, after)
        self.assertAlmostEqual(got.value, d, places=9)
        self.assertAlmostEqual(ch23_case.upper_bound(got), upper, places=9)
        # The printed bound, and the two decisions that follow from it.
        self.assertEqual(f"{upper:.1f}", "4.7")
        # Since Version 59 the transcript prints an upper bound rounded upward: 4.71 is at most 4.8, not 4.7.
        self.assertTrue(4.7 < upper <= 4.8)
        self.assertTrue(upper < 5.0 and ch23_case.within(upper, 5.0))
        self.assertTrue(upper >= 2.0 and not ch23_case.within(upper, 2.0))
        text = (ROOT / "expected" / "ch23_case.txt").read_text(encoding="utf-8")
        self.assertIn(f"upper bound, 95 %: {math.ceil(upper * 10) / 10:.1f} ms", text)


if __name__ == "__main__":
    unittest.main()
