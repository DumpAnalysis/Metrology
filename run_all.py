"""Replay every example and compare its output with expected/.

    python run_all.py              replay all examples and compare
    python run_all.py ch05 ch09    replay the examples whose names match
    python run_all.py --update     rewrite the transcripts in expected/
    python run_all.py --lint       check the 50-column rule only

Each example declares its replay mode in its first line:

    # replay: exact   seeded and deterministic; every number must be written
                      in the notation of the transcript (an exponent or
                      none, no added plus sign or leading zeros) and printed
                      to the same place; a whole number must be identical,
                      and a number with decimals must agree to one unit in
                      its last place
    # replay: live    measures this machine; only the shape of the output
                      is compared (numbers masked, 'env:' lines ignored)
"""
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXAMPLES = ROOT / "examples"
EXPECTED = ROOT / "expected"
WIDTH = 50
NUM = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
WHOLE = re.compile(r"[-+]?\d+")


def mode_of(path):
    first = path.read_text(encoding="utf-8").splitlines()[0]
    match = re.match(r"#\s*replay:\s*(exact|live)\s*$", first)
    if not match:
        raise SystemExit(f"{path.name}: first line must declare replay mode")
    return match.group(1)


def run(name):
    env = dict(os.environ, PYTHONUTF8="1", PYTHONHASHSEED="0")
    proc = subprocess.run(
        [sys.executable, "-m", f"examples.{name}"], cwd=ROOT, env=env,
        capture_output=True, encoding="utf-8", timeout=300)
    if proc.returncode != 0:
        raise SystemExit(f"{name} failed:\n{proc.stderr}")
    return proc.stdout


def skeleton(line):
    """The line with numbers masked and white space collapsed."""
    return " ".join(NUM.sub("#", line).split())


def last_place(token):
    """One unit in the last printed digit of a numeric token."""
    mantissa, _, exponent = token.lower().partition("e")
    digits = len(mantissa.partition(".")[2])
    return 10.0 ** (int(exponent or 0) - digits)


def notation(token):
    """How a number is written, apart from its sign and the digits of its
    mantissa: its exponent as text, and leading zeros."""
    mantissa = re.split("[eE]", token)[0]
    whole = mantissa.lstrip("+-").partition(".")[0]
    return token[len(mantissa):], len(whole) > 1 and whole[0] == "0"


def same_notation(x, y):
    """5800, 5800e0, +5800 and 05800 are four notations of one value. A plus
    sign is part of the notation; a minus sign is part of the value."""
    if notation(x) != notation(y):
        return False
    return "-" in (x[0], y[0]) or (x[0] == "+") == (y[0] == "+")


def compare(mode, expected, actual):
    """Return a list of differences (empty when the replay agrees)."""
    exp = expected.splitlines()
    act = actual.splitlines()
    if mode == "live":
        exp = [x for x in exp if not x.startswith("env:")]
        act = [x for x in act if not x.startswith("env:")]
    if len(exp) != len(act):
        return [f"{len(exp)} lines expected, {len(act)} produced"]
    diffs = []
    for i, (e, a) in enumerate(zip(exp, act), 1):
        if skeleton(e) != skeleton(a):
            diffs.append(f"line {i}: text differs\n  - {e}\n  + {a}")
        elif mode == "exact":
            for x, y in zip(NUM.findall(e), NUM.findall(a)):
                if not same_notation(x, y):
                    diffs.append(f"line {i}: {x} expected, {y} produced, "
                                 "written in another notation")
                elif last_place(x) != last_place(y):
                    diffs.append(f"line {i}: {x} expected, {y} produced, "
                                 "printed to another place")
                elif WHOLE.fullmatch(x):
                    if int(x) != int(y):
                        diffs.append(f"line {i}: {x} expected, {y} produced; "
                                     "a whole number must be identical")
                elif abs(float(x) - float(y)) > 1.0001 * last_place(x):
                    diffs.append(f"line {i}: {x} expected, {y} produced")
    return diffs


def lint():
    """Listings and transcripts must fit the page of the book."""
    bad = []
    files = sorted((ROOT / "swmetro").glob("*.py"))
    files += sorted(EXAMPLES.glob("*.py")) + sorted(EXPECTED.glob("*.txt"))
    files.append(ROOT / "data" / "measurement_record.json")
    for path in files:
        text = path.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            if len(line) > WIDTH:
                rel = path.relative_to(ROOT)
                bad.append(f"{rel}:{i}: {len(line)} columns")
    return bad


def main(argv):
    update = "--update" in argv
    names = [a for a in argv if not a.startswith("--")]
    bad = lint()
    if "--lint" in argv:
        print("\n".join(bad) if bad else f"all lines <= {WIDTH} columns")
        return 1 if bad else 0
    failures = 0
    EXPECTED.mkdir(exist_ok=True)
    for path in sorted(EXAMPLES.glob("ch*.py")):
        name = path.stem
        if names and not any(n in name for n in names):
            continue
        mode = mode_of(path)
        actual = run(name)
        transcript = EXPECTED / f"{name}.txt"
        if update:
            transcript.write_text(actual, encoding="utf-8")
            print(f"updated {mode:<5} {name}")
            continue
        expected = transcript.read_text(encoding="utf-8")
        diffs = compare(mode, expected, actual)
        print(f"{'FAIL' if diffs else 'ok':<4} {mode:<5} {name}")
        for d in diffs:
            print("    " + d)
        failures += bool(diffs)
    if bad and not update:
        print("\n".join(bad))
    return 1 if failures or (bad and not update) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
