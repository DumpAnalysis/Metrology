# replay: exact
"""Chapter 20: reference labels that err."""
import math
import random

TASKS = 400
NOISE = 0.15   # a rater's own error rate
TRAPS = 0.25   # share of failing outputs that
#                look right to every rater
FOOLED = 0.90  # chance that a rater passes one


def outputs(n, rng):
    """(truly passes, is a trap) per task. The
    true pass rate is 0.60."""
    out = []
    for _ in range(n):
        ok = rng.random() < 0.60
        trap = not ok and rng.random() < TRAPS
        out.append((ok, trap))
    return out


def rating(ok, trap, rng):
    """One rater's verdict on one output."""
    if trap:
        return rng.random() < FOOLED
    wrong = rng.random() < NOISE
    return ok != wrong


def panel(tasks, k, rng):
    """Majority verdict of k raters per task.
    Returns the pass rate and the shares of
    tasks labeled wrongly, split by cause:
    raters' own noise, or a trap."""
    passed = noise = traps = 0
    for ok, trap in tasks:
        votes = sum(rating(ok, trap, rng)
                    for _ in range(k))
        verdict = votes > k / 2
        passed += verdict
        if verdict != ok:
            traps += trap
            noise += not trap
    n = len(tasks)
    return passed / n, noise / n, traps / n


def agreement(tasks, want_trap, rng):
    """Share of tasks of one kind on which two
    raters give the same verdict."""
    same = total = 0
    for ok, trap in tasks:
        if trap == want_trap:
            a = rating(ok, trap, rng)
            b = rating(ok, trap, rng)
            same += a == b
            total += 1
    return same / total


def main():
    rng = random.Random(20)
    tasks = outputs(TASKS, rng)
    truth = sum(ok for ok, _ in tasks) / TASKS
    u = math.sqrt(truth * (1 - truth) / TASKS)
    print(f"true pass rate, these tasks"
          f" {truth:.3f}")
    print(f"u from task sampling        {u:.3f}")
    print("raters  pass rate  wrong by:")
    print(f"{'':<17}{'noise':>8}{'traps':>8}")
    for k in (1, 3, 9, 27):
        rate, noise, traps = panel(tasks, k, rng)
        print(f"{k:>6}{rate:>11.3f}"
              f"{noise:>8.3f}{traps:>8.3f}")
    print("two raters agree on")
    for name, flag in (("ordinary", False),
                       ("trap", True)):
        a = agreement(tasks, flag, rng)
        print(f"  {name:<9}outputs {a:.2f}")


if __name__ == "__main__":
    main()
