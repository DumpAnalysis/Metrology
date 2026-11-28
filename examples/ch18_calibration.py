# replay: exact
"""Chapter 18: calibration of probabilities."""
import math
import random
import statistics as st
from decimal import Decimal, localcontext
from fractions import Fraction

from examples.ch20_judge import exact_interval
from swmetro.stats import as_count, downward
from swmetro.stats import fraction, known, native
from swmetro.stats import stated, upward

LEAST = 2.0 ** -1022  # the least normal float


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


def scaled(z, temp):
    """z / temp, rounded once. Two floats are
    divided so by the machine. Any other two
    numbers are divided as the fractions that
    they are: as floats, the decimals 3e-324
    and 5e-324 are one number, and 1e-400 is
    zero. A quotient beyond a float is
    refused."""
    if {type(z), type(temp)} == {float}:
        q = z / temp
    else:
        try:
            q = fraction(z) / fraction(temp)
            q = float(q)
        except OverflowError:
            q = math.inf
    if abs(q) == math.inf:
        raise ValueError("logit / T too large")
    return q


def cases(data):
    """The cases as a list, at least one, of
    (logit, label): the logit a finite number
    that known() admits, in the type in which
    it came, and the label 0 or 1."""
    data = list(data)
    if not data:
        raise ValueError("no cases")
    zs = known([z for z, _ in data], "logit")
    for _, y in data:
        if y not in (0, 1):
            raise ValueError("a label is 0 or 1")
    return [(z, int(y == 1))
            for z, (_, y) in zip(zs, data)]


def checked(data, temp=1.0):
    """The cases of cases() as (logit / T,
    label). T is a finite number above zero
    in the type in which it came: below zero
    the labels would change places, and a
    confident error would score well."""
    data = list(data)
    if not data:
        raise ValueError("no cases")
    temp = known([temp], "T")[0]
    if not temp > 0:
        raise ValueError("need T > 0")
    return [(scaled(z, temp), y)
            for z, y in cases(data)]


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
    the edge of two bins, in the upper one.
    The two sums of floats are exact, as
    fractions: a sum of floats that is
    rounded at each step depends on the
    order of the cases, and two residuals of
    one half hide a small third between
    them."""
    bins = as_count(bins, "bin count", 1)
    tot = [[0, 0, 0, 0] for _ in range(bins)]
    for z, y in checked(data, temp):
        p = sigmoid(z)
        n, d = p.as_integer_ratio()
        b = tot[min(n * bins // d, bins - 1)]
        b[0] += 1
        b[1] += Fraction(n, d)
        b[2] += y
        b[3] += Fraction(residual(z, y))
    return tot


def ece(data, temp=1.0, bins=10):
    """Expected calibration error of P(y=1),
    the probability of the label 1."""
    tot = bins_of(data, temp, bins)
    gap = sum(abs(b[3]) for b in tot)
    return float(gap / sum(b[0] for b in tot))


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


def quick(ms, beta):
    """The sign of the slope from floats, or None
    where their rounding could hide it. A term
    -m * sigmoid(-beta * m) is right to within
    2|beta m| + 8 units of its last place, and the
    terms are added without a further error.
    Nothing is concluded from a term below the
    least normal float, or near the last one."""
    try:
        b = float(beta)
        fs = [float(m) for m in ms]
    except OverflowError:
        return None
    terms, slack = [], []
    for f in fs:
        x = b * f
        s = sigmoid(-x)
        t = -f * s
        if min(b, abs(f), s, abs(t)) < LEAST:
            return None
        terms.append(t)
        slack.append(abs(t) * (2 * abs(x) + 8))
    if not sum(slack) < 1e300:
        return None
    total = math.fsum(terms)
    if abs(total) > math.fsum(slack) * 2.0 ** -52:
        return (total > 0) - (total < 0)
    return None


def sure(ms, beta, digits):
    """The sign of the slope from decimals of
    `digits` digits, or None where they do not
    show it. The terms are kept as logarithms and
    added as shares of the largest, apart for the
    margins below zero and above: a margin far out
    or next to zero gives a term too small for a
    float. A share below 10 ** -digits is left out
    and counted in the error."""
    # A context that takes nothing from the
    # decimal settings of the caller.
    with localcontext(stated(digits)):
        b = Decimal(beta.numerator)
        b /= beta.denominator
        logs = []
        for m in ms:
            a = Decimal(abs(m.numerator))
            a /= m.denominator
            x = b * a
            v = a.ln() - ((-x).exp() + 1).ln()
            if m > 0:
                v -= x
            logs.append((v, 3 * x + abs(v) + 8))
        top, most = max(logs)
        unit = Decimal(10) ** (2 - digits)
        sums = [Decimal(0), Decimal(0)]
        slack = Decimal(10) ** -digits * len(ms)
        for m, (v, w) in zip(ms, logs):
            if v - top < -3 * digits:
                continue
            if (w + most) * unit > 0.001:
                return None
            share = (v - top).exp()
            sums[m > 0] += share
            slack += share * (w + most) * unit
        slack += sum(sums) * len(ms) * unit
        up, down = sums
        if abs(up - down) > slack:
            return (up > down) - (up < down)
    return None


def lean(ms, beta):
    """Sign of the derivative of the summed loss
    by beta, which is 1 / T, at an exact beta. The
    derivative rises with beta. Its terms are
    -m * sigmoid(-beta * m), below zero for a
    margin above zero and above zero for one
    below. The sign is that of the sum of the
    terms as they are, and not of their rounded
    values: rounded terms can add up to zero, or
    to the other sign, where the loss is nearly
    flat. Where floats do not show it, it is found
    with 40 decimal digits, and with twice as many
    until it shows. A slope with no sign in 640
    digits is refused. No margin, no slope."""
    ms = [fraction(m) for m in ms if m]
    beta = fraction(beta)
    sign = quick(ms, beta) if ms else 0
    digits = 40
    while sign is None:
        if digits > 640:
            raise ValueError("the slope has no "
                             "sign in 640 digits")
        sign = sure(ms, beta, digits)
        digits *= 2
    return sign


def fit_temperature(data, lo=0.2, hi=10.0):
    """The finite T above zero that minimizes the
    NLL on a calibration set, as the float nearest
    to it, if that float lies between lo and hi,
    both finite: they bound the search and not the
    model. A minimum beyond them is reported, and
    not replaced by the nearer of the two; one
    that rounds to lo or to hi is that float; the
    largest takes all below the middle between it
    and 2 ** 1024, and lean() refuses a tie. The
    shape of the loss is settled from the margins,
    as the numbers that they are, before anything
    is searched, because a search among rounded
    losses can stop where they are equal. With
    margins that are all zero every T has the same
    loss: the data leave T unidentified.
    Otherwise, with no margin below zero the loss
    falls all the way as T falls, and with margins
    that sum to zero or less, as T rises: there is
    no finite T. These are errors. Otherwise the
    slope changes its sign once, between two
    floats that bisection finds."""
    try:
        lo, hi = map(float, known([lo, hi]))
    except ValueError:
        lo = hi = 0.0
    if not 0 < lo < hi:
        raise ValueError("need 0 < lo < hi < inf")
    ms = margins((fraction(z), y)
                 for z, y in cases(data))
    if not any(ms):
        raise ValueError("the loss is the same "
                         "at every T")
    if min(ms) >= 0:
        raise ValueError("no finite T: the loss "
                         "falls as T falls")
    if sum(ms) <= 0:
        raise ValueError("no finite T: the loss "
                         "falls as T rises")

    def above(a, b):
        """Is the minimum above the middle of
        the floats a and b?"""
        mid = (Fraction(a) + Fraction(b)) / 2
        return lean(ms, 1 / mid) > 0

    if not above(math.nextafter(lo, 0.0), lo):
        raise ValueError("the minimum is "
                         "below lo")
    top = min(math.nextafter(hi, math.inf),
              Fraction(2) ** 1024)
    if above(hi, top):
        raise ValueError("the minimum is "
                         "above hi")
    a, b = lo, hi
    while True:
        mid = math.sqrt(a) * math.sqrt(b)
        if not a < mid < b:
            mid = a / 2 + b / 2
        if not a < mid < b:
            break
        if above(mid, mid):
            a = mid
        else:
            b = mid
    return b if above(a, b) else a


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
        print(f"{float(s / n):6.3f}"
              f"{k / n:>10.3f}{n:>6}")


def estimator_bias():
    """ECE of a perfectly calibrated model."""
    rng = random.Random(18)
    print("perfectly calibrated model:")
    for n in (200, 2000, 20_000):
        reps = [ece(make(n, 1.0, rng))
                for _ in range(20)]
        print(f"  n = {n:>5}: mean ECE "
              f"{st.fmean(reps):.3f}")


def logit_of(p):
    """log(p / (1 - p)) for 0 < p < 1, as a
    fraction. Next to one half the quotient
    is next to 1, and 1 - p rounds away half
    of what p differs from one half by: for
    the float below 0.5 it is 0.5. The
    distance d = 2p - 1 is formed from the
    number that p is, and the logit is twice
    the inverse hyperbolic tangent of d,
    which below 1e-9 is d to the last
    place. Beyond one quarter and three
    quarters the lesser of p and 1 - p is
    formed in the same way; below the least
    normal float its logarithm is that of
    its two whole numbers."""
    p = fraction(p)
    d = 2 * p - 1
    if abs(d) < 1e-9:
        return 2 * d
    if abs(d) <= 0.5:
        return Fraction(2 * math.atanh(d))
    r = min(p, 1 - p)
    if r >= LEAST:
        ln = math.log(r) - math.log1p(-r)
    else:
        ln = math.log(r.numerator)
        ln -= math.log(r.denominator)
    return Fraction(ln if d < 0 else -ln)


def at_one_logit(logit, p):
    """The T above zero that turns one stated
    logit into the logit of the probability p.
    There is one only for p between a half and the
    label that the logit leans to, which is 1 for
    a logit above zero and 0 for one below: as T
    rises from 0 without limit, the stated
    probability moves from that label to a half
    and reaches neither. The quotient of the two
    logits is formed from the numbers that they
    are and rounded once; a T that is no finite
    float above zero is refused."""
    logit, p = native(logit), native(p)
    side = (logit > 0) - (logit < 0)
    leans = (p > 0.5) - (p < 0.5)
    if not (0 < p < 1 and side and leans == side):
        raise ValueError("no T above zero "
                         "states this p")
    try:
        t = Fraction(logit) / logit_of(p)
        t = float(t)
    except OverflowError:
        t = math.inf
    if not 0 < t < math.inf:
        raise ValueError("T is beyond a float")
    return t


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
