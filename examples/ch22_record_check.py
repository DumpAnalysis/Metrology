# replay: exact
"""Chapter 22: checking a record before use."""
import copy
from fractions import Fraction
from pathlib import Path

from swmetro.core import decimals
from swmetro.record import combined, comparable
from swmetro.record import decide, exact, guard
from swmetro.record import in_unit, missing, read
from swmetro.record import rounded, spread, stale
from swmetro.record import stopped
from swmetro.stats import downward, enclosed
from swmetro.stats import nearest, tens

ROOT = Path(__file__).resolve().parent.parent
# A record from other telemetry, written
# afresh with only what a comparison reads.
# Its uncertainty was never evaluated, and it
# has no budget and no decision: the objective
# of the stored record was agreed for another
# start event and another convention.
OTHER = {
    "format": "measurement record, version 5",
    "measurand": {
        "quantity": "p95 of request latency",
        "convention": "interpolated in a bucket",
        "target": "mean of the run p95 over runs",
        "conditions": "test load",
        "start_event": "connection accepted",
        "end_event": "last byte sent",
        "unit": "s",
    },
    "data": {"run_weighting": "equal"},
    "capture": {
        "clock": "gateway host, wall clock",
        "clock_relations": [],
    },
    "result": {"value": 0.2414, "u": None},
}


def load(name):
    """A record of the data directory, with
    its numbers as they are written: the
    module's reader keeps a decimal a decimal
    where the default one would make a float
    of it."""
    path = ROOT / "data" / name
    return read(path.read_text("utf-8"))


def altered(rec, changes):
    """A copy with some values replaced, and
    nothing that depends on them recomputed:
    what a careless edit leaves behind."""
    out = copy.deepcopy(rec)
    for path, value in changes.items():
        part = out
        for key in path[:-1]:
            part = part[key]
        part[path[-1]] = value
    return out


def two(x):
    """The number that x states, with two
    significant digits."""
    return rounded(x, decimals(exact(x)))


def displayed(rec):
    """A copy as a report would print it: the
    uncertainties and the degrees of freedom
    with two significant digits, the estimate
    to the place of its uncertainty, and the
    record saying so. With no uncertainty
    there is no such place, and the estimate
    is left as it is. The increment is one
    unit of that place, as the float that
    prints as it. Below the least float
    there is none, and no record is issued:
    an increment of 0 would call the
    rounded estimate exact."""
    out = copy.deepcopy(rec)
    for c in out["components"]:
        c["u"] = two(c["u"])
    r = out["result"]
    r["value_step"] = 0.0
    if r["u"] > 0:
        place = decimals(exact(r["u"]))
        step = float(tens(1, -place))
        if not step:
            raise ValueError("no such increment")
        r["value"] = rounded(r["value"], place)
        r["value_step"] = step
    r["u"], r["dof"] = two(r["u"]), two(r["dof"])
    r["significant_digits"] = 2
    return out


def report(title, rec):
    status, why = decide(rec)
    print(title)
    n = len(missing(rec))
    print(f"  decision inputs missing: {n}")
    print(f"  status: {status}")
    for reason in why[:2]:
        print(f"    {reason}")
    kept = stale(rec)
    if kept is not None:
        print(f"  stored outcome, stale: {kept}")


def near(title, rec, value):
    """The record with an estimate close to
    the limit, and the bounds of that
    estimate plus its guard band. Two ends
    that differ are two bounds, however
    close: the lower is rounded downward and
    the upper upward, to two places or to
    the second digit of their distance.
    Equal ends are one value, rounded to
    the nearest. The limit is read as the
    decision reads it, and whatever is
    printed stands to it as the number did:
    below it, on it or above it, with more
    places where two would not show that.
    A record that the checks stop has no
    such bounds, and none are printed."""
    change = {("result", "value"): value}
    rec = altered(rec, change)
    report(title, rec)
    if stopped(rec):
        return
    lo, hi = guard(rec)[0]
    limit = exact(rec["decision"]["limit"])
    if hi.is_infinite():
        low = downward(lo, 2, limit)
        text = f"{low} or more"
    elif hi != lo:
        apart = Fraction(hi) - Fraction(lo)
        places = 2
        while apart * 10 ** places < 10:
            places += 1
        ends = enclosed(lo, hi, places, limit)
        text = " to ".join(ends)
    else:
        text = nearest(lo, 2, limit)
    print(f"  estimate + band: {text}")


def main():
    rec = load("measurement_record.json")
    report("stored record", rec)
    between = ("components", 1, "u")
    cases = (
        ("loss count unknown",
         {("capture", "records_lost"): None}),
        ("a component unknown", {between: None}),
        ("a component changed", {between: 5.0}),
        ("negative uncertainty",
         {("result", "value"): 212.0,
          ("result", "u"): -10.0}),
        ("another rule declared",
         {("decision", "rule"):
          "accept if value - k u >= limit"}),
        ("outcome edited",
         {("decision", "outcome"): "reject"}),
    )
    for title, changes in cases:
        report(title, altered(rec, changes))
    # The record rounded for display, and
    # estimates that bring it near the limit.
    shown = displayed(rec)
    report("rounded to two digits", shown)
    rows = (("components allow", spread),
            ("with the stated u", combined))
    for name, f in rows:
        lo, hi = enclosed(*f(shown), 3)
        print(f"  {name}: {lo} to {hi}")
    near("rounded, estimate 194.6", shown, 194.6)
    near("rounded, estimate 194.8", shown, 194.8)
    near("working values, estimate 194.6",
         rec, 194.6)
    # The same telemetry under another load:
    # another quantity, unless the load is
    # the variable under study.
    heavier = {key: copy.deepcopy(rec[key])
               for key in ("format", "measurand",
                           "data", "capture")}
    load_of = heavier["measurand"]
    load_of["conditions"] = "twice the test load"
    print("against a record at twice the load")
    for reason in comparable(rec, heavier)[0]:
        print(f"  differs in: {reason}")
    left, _ = comparable(rec, heavier,
                         across=("conditions",))
    fit = "no" if left else "yes"
    print("  comparable across conditions:", fit)
    reasons, notes = comparable(rec, OTHER)
    print("against a record of other telemetry")
    for reason in reasons:
        print(f"  differs in: {reason}")
    for note in notes:
        print(f"  to convert: {note}")
    value, u = in_unit(OTHER, "ms")
    known = "not evaluated"
    if u is not None:
        known = f"{u:.1f} ms"
    print(f"  its result: {value:.1f} ms,"
          f" u {known}")
    print(f"  its status: {decide(OTHER)[0]}")
    report("blank template",
           load("record_template.json"))


if __name__ == "__main__":
    main()
