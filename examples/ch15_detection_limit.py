# replay: exact
"""Chapter 15: critical value, detection limit."""
import random

S0 = 2.0  # sd of the blank indication, per min
L_C = 1.645 * S0  # decide "detected" above this
L_D = 3.29 * S0   # detected 95 % of the time


def flagged(true_excess, n, rng):
    """Share of indications above L_C for a
    given true excess error rate."""
    hits = 0
    for _ in range(n):
        ind = true_excess + rng.gauss(0, S0)
        hits += ind > L_C
    return hits / n


def main():
    rng = random.Random(15)
    print(f"critical value  L_C = {L_C:.2f} /min")
    print(f"detection limit L_D = {L_D:.2f} /min")
    cases = (("zero (blank)", 0.0),
             ("equal to L_C", L_C),
             ("equal to L_D", L_D))
    for name, excess in cases:
        p = flagged(excess, 100_000, rng)
        print(f"true excess {name:<13}"
              f"flagged {100 * p:5.1f} %")


if __name__ == "__main__":
    main()
