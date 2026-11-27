"""Writes data/measurement_record.json, the record of the Chapter 22 result,
and data/record_template.json, the same fields with nothing filled in.

    python data/make_record.py

The record is produced by the program that made the measurement, so that its
numbers, its decision and the hash of its sources cannot drift from the code.
The estimate, the uncertainties and the degrees of freedom are written at
working precision, as the program computed them; only the interval, which the
checker does not read, is rounded for display.
tests/test_review2.py checks that the stored file equals a fresh one.
"""
import hashlib
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from examples import ch22_budget  # noqa: E402
from swmetro import record as checker  # noqa: E402

# The files that determine the record: the measurement, the package, and this
# program, which sets the decision risk and builds the decision fields.
SOURCES = ["examples/ch22_budget.py", "examples/ch09_histogram.py",
           "swmetro/core.py", "swmetro/dist.py", "swmetro/stats.py",
           "swmetro/record.py", "data/make_record.py"]
RELEASE = "book Version 71, November 2026"
REPOSITORY = "github.com/DumpAnalysis/Metrology"
RISK = 0.05
# What the record says of each component of the budget beyond its name, its
# evaluation type, its standard uncertainty and its degrees of freedom.
COMPONENTS = {
    "sampling in the run": ("derived", "bootstrap of the run, 500 draws"),
    "between runs": ("stipulated", "earlier study of 8 runs"),
    "probe correction": ("stipulated", "set in the simulation"),
    "stored step": ("derived", "rectangular, half a step"),
}


def _text(rel):
    """A source file as hashed: UTF-8 text with LF line ends."""
    text = (ROOT / rel).read_text(encoding="utf-8")
    return text.replace("\r\n", "\n").encode("utf-8")


def source_hash():
    """SHA-256 of the listed files' text, in order."""
    h = hashlib.sha256()
    for rel in SOURCES:
        h.update(_text(rel))
    digest = h.hexdigest()
    return [digest[:32], digest[32:]]


def file_hashes():
    """First 16 hexadecimal digits of each file's own SHA-256, in the order
    of SOURCES: enough to tell which file changed, and short enough to print."""
    return [hashlib.sha256(_text(rel)).hexdigest()[:16] for rel in SOURCES]


def components(bud):
    """The rows of the budget as the record states them: each u is a
    contribution to the result, in the unit of the record."""
    out = []
    for name, evaluation, u, dof in bud.rows:
        kind, source = COMPONENTS[name]
        row = {"name": name, "type": evaluation, "kind": kind}
        if name == "probe correction":
            row["value"] = -ch22_budget.PROBE
        row["u"] = u
        if math.isfinite(dof):
            row["dof"] = dof
        row["source"] = source
        out.append(row)
    return out


def build():
    bud, r = ch22_budget.from_raw(ch22_budget.record(random.Random(22)))
    lo, hi = map(float, r.interval(0.95))
    expects = {key: text for _, key, text in checker.EXPECTS}
    rec = {
        "format": checker.FORMAT,
        "measurand": {
            "quantity": "p95 of request latency",
            "convention": "inverse empirical CDF",
            "target": "mean of the run p95 over runs",
            "conditions": "test load",
            "start_event": "request accepted",
            "end_event": "last byte sent",
            "unit": "ms",
        },
        "data": {
            "origin": "simulated test run",
            "generator": "examples.ch22_budget.record",
            "seed": 22,
            "runs": 1,
            "requests_per_run": 2000,
            "run_weighting": "equal",
        },
        "capture": {
            "kind": "stipulated",
            "clock": "one host, monotonic",
            "clock_relations": [],
            "stored_step": ch22_budget.STEP,
            "sampling": "all requests",
            "records_lost": 0,
        },
        "components": components(bud),
        "covariances": [],
        "result": {
            "kind": "derived",
            "value": r.value,
            "u": r.u,
            "dof": r.dof,
            "significant_digits": checker.ALL,
            "interval_95": [round(lo, 1), round(hi, 1)],
            "method": expects["method"],
        },
        "decision": {
            "kind": "derived",
            "procedure": checker.PROCEDURE,
            "limit": ch22_budget.LIMIT,
            "rule": expects["rule"],
            "k": expects["k"],
            "risk": RISK,
            "risk_is": expects["risk_is"],
        },
        "assumed": [
            "requests independent within a run",
            "components uncorrelated",
        ],
        "not_claimed": [
            "loads other than the test load",
            "long-run false-acceptance rate",
        ],
        "not_evaluated": [],
        "sources": {
            "repo": REPOSITORY,
            "release": RELEASE,
            "files": SOURCES,
            "sha256_16_of_each": file_hashes(),
            "sha256": source_hash(),
            "of": "the files' text, in order, LF ends",
        },
        "replay": {
            "program": "examples/ch22_budget.py",
            "python": "3.11 or later",
        },
    }
    rec["decision"]["outcome"] = checker.decide(rec)[0]
    return rec


def template():
    """The same fields with nothing filled in. A prompt in angle brackets or a
    null stands for unknown; the checker never reads either as zero. The list
    of covariances is null and not empty: an empty list is the statement that
    none were entered, and the template makes no statement."""
    return {
        "format": checker.FORMAT,
        "measurand": {
            "quantity": "<what is measured>",
            "convention": "<e.g. quantile rule>",
            "target": "<record, process, next run ...>",
            "conditions": "<load, versions, hardware>",
            "start_event": "<where the interval starts>",
            "end_event": "<where the interval ends>",
            "unit": "<unit of every value below>",
        },
        "data": {
            "origin": "<capture or simulation id>",
            "runs": None,
            "requests_per_run": None,
            "run_weighting": "<equal, by requests ...>",
        },
        "capture": {
            "kind": "<observed or stipulated>",
            "clock": "<the one clock of both events>",
            "clock_relations": [
                {"clocks": ["<this clock>", "<another clock>"],
                 "of": "<rate or offset>",
                 "state": "<established, or not>",
                 "evidence": "<how it is known>"},
            ],
            "stored_step": None,
            "sampling": "<rule and probabilities>",
            "records_lost": None,
        },
        "components": [
            {"name": "<component of the budget>",
             "type": "<A or B>",
             "kind": "<observed, derived ...>",
             "u": None,
             "dof": None,
             "source": "<study, runs, date>"},
            {"name": "<a correction, if one is applied>",
             "type": "<A or B>",
             "kind": "<observed, derived ...>",
             "value": None,
             "u": None,
             "source": "<calibration and its date>",
             "valid_for": "<range and conditions>"},
        ],
        "covariances": None,
        "result": {
            "kind": "derived",
            "value": None,
            "u": None,
            "dof": None,
            "significant_digits": "<all, or a count>",
            "value_step": None,
            "interval_95": [None, None],
            "method": "<how u and the interval arise>",
        },
        "decision": {
            "kind": "derived",
            "procedure": "<a procedure the checker has>",
            "limit": None,
            "rule": "<fixed before the measurement>",
            "k": "<coverage factor of the rule>",
            "risk": None,
            "risk_is": "<model probability or rate>",
            "outcome": None,
        },
        "assumed": ["<what the result takes for granted>"],
        "not_claimed": ["<what lies outside the target>"],
        "not_evaluated": ["<required, never evaluated>"],
        "sources": {"repo": "<address>",
                    "release": "<tag or commit>", "files": []},
    }


def text():
    return json.dumps(build(), indent=2) + "\n"


def template_text():
    return json.dumps(template(), indent=2) + "\n"


if __name__ == "__main__":
    for name, body in (("measurement_record.json", text()),
                       ("record_template.json", template_text())):
        out = ROOT / "data" / name
        out.write_text(body, encoding="utf-8")
        print(f"wrote {out.relative_to(ROOT)}")
