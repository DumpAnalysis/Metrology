# replay: exact
"""Chapter 18: calibration of probabilities."""
import math
import random
import statistics as st
from fractions import Fraction

from examples.ch20_judge import exact_interval
from swmetro.stats import as_count, known
from swmetro.stats import downward, upward


def sigmoid(z):
    """1 / (1 + exp(-z)), with an exponent that
    is never positive and cannot overflow."""
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    e = math.exp(z)
    return e / (1 + e)


def make(n, scale, rng):
    """(reported logit, label) pairs. The label
    follows the true logit z; the model reports
    scale * z, too sharp when scale > 1."""
    data = []
    for _ in range(n):
        z = rng.gauss(0, 1.5)
        y = int(rng.random() < sigmoid(z))
        data.append((scale * z, y))
    return data


def checked(data, temp=1.0):
    """The cases as a list, at least one, of
    (logit / T, label). The logit and T are
    finite numbers that known() admits, the
    label is 0 or 1, and T is above zero:
    below it the labels would change places,
    and a confident error would score well.
    A quotient beyond a float is refused."""
    data = list(data)
    if not data:
        raise ValueError("no cases")
    temp = float(known([temp], "T")[0])
    if not temp > 0:
        raise ValueError("need T > 0")
    zs = known([z for z, _ in data], "logit")
    for _, y in data:
        if y not in (0, 1):
            raise ValueError("a label is 0 or 1")
    out = [(float(z) / temp, int(y == 1))
           for z, (_, y) in zip(zs, data)]
    if any(abs(z) == math.inf for z, _ in out):
        raise ValueError("logit / T too large")
    return out


def residual(z, y):
    """P(y=1) - y at the scaled logit z, from
    the side on which it keeps its digits: a
    probability that has rounded to 1 would
    leave none for the label 1."""
    return -sigmoid(-z) if y else sigmoid(z)


def bins_of(data, temp=1.0, bins=10):
    """Per bin of stated probability: count,
    sum of probabilities, sum of labels, sum
    of residuals. A probability is in the
    bin where the float that it is lies: on
    the edge of two bins, in the upper one."""
    bins = as_count(bins, "bin count", 1)
    tot = [[0, 0.0, 0, 0.0] for _ in range(bins)]
    for z, y in checked(data, temp):
        p = sigmoid(z)
        n, d = p.as_integer_ratio()
        b = tot[min(n * bins // d, bins - 1)]
        b[0] += 1
        b[1] += p
        b[2] += y
        b[3] += residual(z, y)
    return tot


def ece(data, temp=1.0, bins=10):
    """Expected calibration error of P(y=1),
    the probability of the label 1."""
    tot = bins_of(data, temp, bins)
    gap = sum(abs(b[3]) for b in tot)
    return gap / sum(b[0] for b in tot)


def nll(data, temp=1.0):
    """Mean negative log-likelihood, taken from
    the logit. A probability that has rounded
    to 1 has lost the size of the error: at
    a scaled logit of 40 the loss of a case
    with the label 0 is 40, and the rounded
    probability gives the logarithm of 0.
    The losses are added as fractions: two
    of 1e308 have a mean that is a float,
    and no such sum."""
    data, acc = checked(data, temp), 0
    for z, y in data:
        t = -z if y else z
        soft = math.log1p(math.exp(-abs(t)))
        acc += Fraction(max(t, 0.0) + soft)
    return float(acc / len(data))


def brier(data, temp=1.0):
    data = checked(data, temp)
    return st.fmean(residual(z, y) ** 2
                    for z, y in data)


def margins(data):
    """The logit of each case with the sign of
    its label: positive where the model leans
    to the label that was observed."""
    return [z if y else -z for z, y in data]


def slope(ms, beta):
    """Derivative of the summed loss by beta,
    which is 1 / T. It rises with beta. The
    terms are added as fractions: the sum
    may be beyond a float, and has a sign."""
    return sum(Fraction(-m * sigmoid(-beta * m))
               for m in ms)


def fit_temperature(data, lo=0.2, hi=10.0):
    """The finite T above zero that minimizes
    the NLL on a calibration set, if it lies
    between lo and hi, both finite: they
    bound the search and not the model. A
    minimum beyond them is reported, and not
    replaced by the nearer of the two. The
    shape of the loss is settled
    from the margins before anything is
    searched, because a search among rounded
    losses can stop where they are equal.
    With margins that are all zero every T
    has the same loss: the data leave T
    unidentified. Otherwise, with no margin
    below zero the loss falls all the way as
    T falls, and with margins that sum to
    zero or less, as T rises: there is no
    finite T. These are errors, as is a
    minimum outside the interval. A minimum
    at lo or at hi is returned. Otherwise
    the slope changes its sign once inside,
    and bisection finds where."""
    if not 0 < lo < hi < math.inf > 1 / lo:
        raise ValueError("need 0 < lo < hi < inf")
    ms = margins(checked(data))
    if not any(ms):
        raise ValueError("the loss is the same "
                         "at every T")
    if min(ms) >= 0:
        raise ValueError("no finite T: the loss "
                         "falls as T falls")
    if sum(map(Fraction, ms)) <= 0:
        raise ValueError("no finite T: the loss "
                         "falls as T rises")
    at_lo = slope(ms, 1 / float(lo))
    at_hi = slope(ms, 1 / float(hi))
    if at_lo < 0:
        raise ValueError("the minimum is "
                         "below lo")
    if at_hi > 0:
        raise ValueError("the minimum is "
                         "above hi")
    if not (at_lo and at_hi):
        return hi if at_lo else lo
    a, b = math.log(lo), math.log(hi)
    for _ in range(60):
        mid = (a + b) / 2
        if slope(ms, math.exp(-mid)) > 0:
            a = mid
        else:
            b = mid
    return math.exp((a + b) / 2)


def correction():
    rng = random.Random(18)
    cal = make(2000, 2.0, rng)
    test = make(5000, 2.0, rng)
    temp = fit_temperature(cal)
    print(f"fitted temperature T = {temp:.2f}")
    print(f"{'on test set':<12}{'ECE':>8}"
          f"{'Brier':>8}{'NLL':>8}")
    for name, t in (("as reported", 1.0),
                    ("corrected", temp)):
        print(f"{name:<12}{ece(test, t):>8.3f}"
              f"{brier(test, t):>8.3f}"
              f"{nll(test, t):>8.3f}")
    print("stated   observed   n   (as reported)")
    for n, s, k, _ in bins_of(test, 1.0, 5):
        print(f"{s / n:6.3f}{k / n:>10.3f}{n:>6}")


def estimator_bias():
    """ECE of a perfectly calibrated model."""
    rng = random.Random(18)
    print("perfectly calibrated model:")
    for n in (200, 2000, 20_000):
        reps = [ece(make(n, 1.0, rng))
                for _ in range(20)]
        print(f"  n = {n:>5}: mean ECE "
              f"{st.fmean(reps):.3f}")


def at_one_logit(logit, p):
    """The T above zero that turns one stated
    logit into the logit of the probability
    p. There is one only for p between a half
    and the label that the logit leans to,
    which is 1 for a logit above zero and 0
    for one below: as T rises from 0 without
    limit, the stated probability moves from
    that label to a half and reaches
    neither."""
    side = (logit > 0) - (logit < 0)
    leans = (p > 0.5) - (p < 0.5)
    if not (0 < p < 1 and side and leans == side):
        raise ValueError("no T above zero "
                         "states this p")
    return float(logit) / math.log(p / (1 - p))


def temperatures(logit, positive, n):
    """The T above zero at which `positive`
    labels of n, all at one stated logit, lie
    within the exact 95 % interval of their
    frequency: (low, high), or None where
    there is none. The limits change places.
    The interval is cut to what a T can
    state: where it reaches 1, low is 0,
    which is not attained, and where it
    reaches a half, high is math.inf. A
    logit below zero is the same case with
    the two labels exchanged. At a logit of
    zero every T states a half, and every T
    or none is in the set."""
    if not math.isfinite(logit):
        raise ValueError("need a finite logit")
    if logit < 0:
        logit, positive = -logit, n - positive
    p_low, p_high = exact_interval(positive, n)
    if p_high <= 0.5 and logit:
        return None
    if not (logit or p_low <= 0.5 <= p_high):
        return None
    low, high = 0.0, math.inf
    if logit and p_high < 1:
        low = at_one_logit(logit, p_high)
    if logit and p_low > 0.5:
        high = at_one_logit(logit, p_low)
    return low, high


def worded(ends):
    if ends is None:
        return "no T above zero"
    low, high = ends
    if high == math.inf and not low:
        return "any T above zero"
    down = downward(low, 2)
    if high == math.inf:
        return f"T from {down}, no upper end"
    if not low:
        return f"T up to {upward(high, 2)}"
    return f"{down} to {upward(high, 2)}"


def confident():
    """A hundred cases stated at a logit of 40,
    of which one is negative: the fit, and
    what a hundred labels settle of it."""
    data = [(40.0, 1)] * 99 + [(40.0, 0)]
    temp = fit_temperature(data)
    print("100 cases at logit 40, one negative:")
    print(f"  T = {temp:.2f}, NLL {nll(data):.3f}"
          f" -> {nll(data, temp):.3f}")
    ends = temperatures(40, 99, 100)
    print("  95 % interval for T: "
          + worded(ends))
    two = at_one_logit(40, 0.98)
    print(f"  with two negative: T = {two:.2f}")
    none = 0.99 ** 100
    one = 100 * 0.01 * 0.99 ** 99
    print("  resampled, they hold no negative")
    print(f"  case in {100 * none:.1f} %, one in "
          f"{100 * one:.1f} %")
    for positive in (55, 100, 35):
        ends = temperatures(40, positive, 100)
        print(f"  {positive} positive: "
              + worded(ends))
    ends = temperatures(-40, 1, 100)
    print("  1 positive at -40: " + worded(ends))


def main():
    correction()
    estimator_bias()
    confident()


if __name__ == "__main__":
    main()
