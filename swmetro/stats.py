"""Small statistics used by the examples."""
import math
import random
import statistics as st
from decimal import Decimal
from fractions import Fraction
from numbers import Integral, Real

from .dist import t_within, z_ppf


def native(x):
    """x as a number of Python's own that is
    equal to it: an int, a float, a Fraction
    or a Decimal. A truth value is no number
    here, although Python counts True as 1:
    a flag in the place of a reading is a
    fault of the record, and it is refused,
    as a text and None are. A whole number
    of a fixed width, as an array holds it,
    wraps around where a sum leaves that
    width: two readings of 100 in eight bits
    have the mean -28. It becomes an int
    before any arithmetic. A float of another
    width becomes the float that is equal to
    it, or else the fraction that it is. A
    Fraction can hold such whole numbers as
    its two parts, and it is then formed anew
    from ints."""
    real = isinstance(x, (Real, Decimal))
    if isinstance(x, bool) or not real:
        raise TypeError(f"no number: {x!r}")
    if isinstance(x, Fraction):
        n, d = x.numerator, x.denominator
        if type(n) is type(d) is int:
            return x
        return Fraction(int(n), int(d))
    if isinstance(x, Decimal):
        return x
    if type(x) in (int, float):
        return x
    if isinstance(x, Integral):
        return int(x)
    if isinstance(x, float):
        return float(x)
    ratio = getattr(x, "as_integer_ratio", None)
    if ratio is None:
        raise TypeError(f"no number: {x!r}")
    try:
        n, d = ratio()
    except (ValueError, OverflowError):
        return float(x)  # a NaN, an infinity
    f = Fraction(int(n), int(d))
    try:
        y = float(f)
    except OverflowError:
        return f
    return y if y == f else f


def natives(xs):
    """The values as a list, each as native()
    gives it. Every value is looked at, once,
    before any of them takes a place in a rank
    or a sum: a flag among nineteen durations
    would stand there as a duration of 0 or 1
    and move the rank of the others. A list of
    ints, floats and decimals alone is handed
    back as it is. What has no length or no
    places is refused. A stream can be read
    once, and whoever takes one reads it into
    a list. A set has no order, and a table of
    names and values has no first value."""
    need = "__len__", "__getitem__"
    rows = all(hasattr(xs, a) for a in need)
    if not rows or hasattr(xs, "keys"):
        raise TypeError("need a list of values")
    own = {int, float, Decimal}
    plain = isinstance(xs, list)
    if plain and set(map(type, xs)) <= own:
        return xs
    return [native(x) for x in xs]


def exact_q(q):
    """A probability as an exact fraction. A
    float is read as the decimal number that
    Python prints for it, so that 0.95 means
    95/100 and not the binary number nearest
    to it. A Fraction, a Decimal, an int or a
    string is taken as it stands; pass one of
    those when another reading is meant. A
    truth value is no probability, and what
    is no string is taken as native() gives
    it."""
    if not isinstance(q, str):
        q = native(q)
    if isinstance(q, float):
        q = repr(q)
    f = Fraction(q)
    if not 0 <= f <= 1:
        raise ValueError("q must be in [0, 1]")
    return f


def as_count(x, name="n", least=0):
    """x as an int, if it is a whole number not
    below `least`. A number that is negative or
    has a fraction is not a count yet. The
    difference of two readings of a counter is
    negative after a reset, and has to be
    interpreted before a rate or a bound is drawn
    from it: a bound of zero for it would claim
    more than a true count of zero does. A truth
    value is no count either: False would pass
    for a count of zero that was observed."""
    real = isinstance(x, (Real, Decimal))
    ok = real and not isinstance(x, bool)
    try:
        ok = ok and x == int(x) and x >= least
    except (TypeError, ValueError,
            OverflowError):
        ok = False
    if not ok:
        raise ValueError(
            f"need a whole {name} >= {least}")
    return int(x)


def known(xs, what="duration"):
    """xs as a list, each of them a finite real
    number as native() gives it: an int, a
    float, a Fraction or a Decimal. A NaN has
    no rank among the others, and sorting a
    record that holds one can make a percentile
    depend on the order of its rows; to drop it
    would be to judge another population. It is
    refused as missing, and so is an infinity or
    a number beyond the range of a float. A
    boolean, a text and None are no numbers."""
    xs = list(xs)
    gone = f"a {what} is missing or too large"
    for i, x in enumerate(xs):
        real = isinstance(x, (Real, Decimal))
        if isinstance(x, bool) or not real:
            raise ValueError(f"no number: {x!r}")
        try:
            there = math.isfinite(float(x))
        except (ValueError, OverflowError):
            there = False
        if not there:
            raise ValueError(gone)
        xs[i] = native(x)
    return xs


def _placed(n, places):
    """The whole number n with its last `places`
    digits after the point. Whole numbers alone
    take part, so that no decimal context can
    round the digits a second time."""
    sign = "-" if n < 0 else ""
    whole, part = divmod(abs(n), 10 ** places)
    if not places:
        return f"{sign}{whole}"
    return f"{sign}{whole}.{part:0{places}d}"


def upward(x, places):
    """x as text with `places` decimals, rounded
    toward +inf in exact arithmetic. An upper
    bound that is printed has to stay one, and
    rounded to the nearest it can lie below the
    bound that was computed. A float is the
    binary number that it is, and `places` is a
    whole number not below 0."""
    places = as_count(places, "number of places")
    x = Fraction(native(x))
    n = math.ceil(x * 10 ** places)
    return _placed(n, places)


def downward(x, places):
    """The same toward -inf, for a lower bound."""
    places = as_count(places, "number of places")
    x = Fraction(native(x))
    n = math.floor(x * 10 ** places)
    return _placed(n, places)


def _floor(f):
    return f.numerator // f.denominator


def quantile(xs, q, exact=False):
    """Quantile of sorted xs by linear
    interpolation between the order statistics
    around rank q * (n - 1), counted from 0. Where
    the rank is a whole number, or the two values
    are the same, there is nothing to interpolate,
    and the value is returned as native() gives
    it. Between two values the interpolation is
    exact. It is rounded once, to a float between
    them, if floats hold both, and it is not
    rounded if a float does not hold one of them,
    see _kept(). With `exact` it is never rounded.
    Between 100 and 101 the 95th percentile is
    2019/20; the float 100.95 lies above that,
    and compared with a limit of 2019/20 it would
    exceed a limit that the percentile meets."""
    xs = natives(xs)
    pos = exact_q(q) * (len(xs) - 1)
    i = _floor(pos)
    j = min(i + 1, len(xs) - 1)
    w, a, b = pos - i, xs[i], xs[j]
    if not w or b == a:
        return a
    f = Fraction(a)
    f += w * (Fraction(b) - f)
    return f if exact else _kept(f, (a, b))


def lower_rank(q, n):
    """Index, counted from 0, of the lower
    order statistic of n values, which is
    floor(q * (n - 1)) in exact arithmetic."""
    return _floor(exact_q(q) * (as_count(n) - 1))


def order_stat(xs, q):
    """Lower order statistic of sorted xs: an
    observed value, with no interpolation,
    and every value as natives() gives it."""
    xs = natives(xs)
    return xs[lower_rank(q, len(xs))]


def ecdf_rank(q, n):
    """Rank, counted from 1, of the q-quantile of
    n values under the inverse empirical CDF
    convention: the smallest k with k / n >= q,
    found in exact arithmetic from the reading of
    q that exact_q gives. For q = 0 it is 1."""
    n = as_count(n, least=1)
    k = -_floor(-exact_q(q) * n)
    return max(k, 1)


def ecdf_quantile(xs, q):
    """Inverse empirical CDF quantile of sorted
    xs: the smallest observed value with at least
    a share q of the values at or below it. It is
    the quantile that cumulative counts determine,
    and for q > 0 "the q-quantile is at most T"
    means exactly "a share q of the values is at
    most T". For q = 0 it is the least value.
    Every value is taken as natives() gives it: a
    flag in the place of a duration that is
    missing would otherwise count as a request
    within the limit."""
    xs = natives(xs)
    return xs[ecdf_rank(q, len(xs)) - 1]


def weighted_quantile(pairs, q):
    """Inverse empirical CDF of (value, weight)
    pairs: the smallest value with at least a
    share q of the weight at or below it, and for
    q = 0 the smallest that has weight. The
    weights are summed as the fractions that they
    are, a float as the binary number that it is.
    Summed as floats, nineteen of twenty weights
    of 4 / 3 can fall a hair short of 95 % of the
    total; the p95 is then the twentieth value. A
    reciprocal of a probability is formed from the
    exact one: 1 / 0.75 as a float is already
    below 4 / 3. A factor on all the weights
    leaves the result alone if it is applied
    exactly: 19 * 0.7 and 0.7 are no longer in the
    ratio 19 to 1. Whatever is no real number is
    refused, None, True or 2 + 3j, and so is a
    float that is not a number. Values and weights
    are taken as native() gives them."""
    pairs = [(x, Fraction(native(w)))
             for x, w in pairs]
    try:
        pairs = [(native(x), w) for x, w in pairs]
        ok = all(x == x for x, _ in pairs)
    except TypeError:
        ok = False
    if not ok:
        raise ValueError("a value is no number")
    pairs.sort()
    total = sum(w for _, w in pairs)
    if not total or min(w for _, w in pairs) < 0:
        raise ValueError("need weights >= 0,"
                         " not all zero")
    need, acc = exact_q(q) * total, 0
    for x, w in pairs:
        acc += w
        if w and acc >= need:
            return x


def run_length(q, miss):
    """Smallest whole n, at least 1, with
    q ** n <= miss, in exact arithmetic. It
    is the number of independent events of
    chance q that all happen with a chance
    of `miss` at most: values that all lie
    below a q-quantile, or sampling
    decisions that all go against. Both
    arguments are read as exact_q() reads a
    probability. No n is large enough for
    q = 1 with miss < 1, nor for miss = 0
    with q > 0, and that is an error. The
    search doubles n and then halves the
    gap, and so needs few powers."""
    q, miss = exact_q(q), exact_q(miss)
    if q <= miss:
        return 1
    if q == 1 or not miss:
        raise ValueError("no finite n: q ** n"
                         " stays above miss")
    lo, hi = 1, 2
    while q ** hi > miss:
        lo, hi = hi, 2 * hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if q ** mid > miss:
            lo = mid
        else:
            hi = mid
    return hi


def _ranks(n, q, a):
    """For B binomial with n and q = s / t: the
    largest rank r with P(B < r) <= a, the
    smallest with P(B >= r) <= a, and the two
    chances at them as whole numbers over
    t ** n, which is returned last; 0 and 0
    where no rank has it. The chances are
    sums of whole numbers and are compared
    with a in whole numbers. Floats summed
    to one tail, with the other taken from
    1, can put a rank on the wrong side of
    a cutoff that it meets exactly, or of
    one that is very small."""
    s, t = q.numerator, q.denominator
    whole = t ** n
    room = a.numerator * whole // a.denominator
    term, total = (t - s) ** n, 0
    lo, hi, low, high = 0, 0, 0, 0
    for r in range(1, n + 1):
        total += term  # whole * P(B < r)
        term = term * ((n - r + 1) * s)
        term //= r * (t - s)
        if total <= room:
            lo, low = r, total
        if whole - total <= room:
            hi, high = r, whole - total
            break
    return lo, hi, low, high, whole


def quantile_ranks(n, q, p=0.95):
    """Ranks lo and hi among n sorted
    independent draws from one continuous
    distribution, such that its q-quantile
    lies below the lo-th draw with a chance
    of (1 - p) / 2 at most, and above the
    hi-th likewise; and the coverage that
    the two attain. It is p or more, because
    ranks are whole numbers, and it is the
    exact ratio, rounded once. A rank is
    None where n draws cannot bound the
    quantile on that side, and that side
    then takes nothing from the coverage.
    The number of draws below the quantile
    is binomial with n and q; for q above
    1/2 the draws above it are counted, so
    that the sums are the short ones. They
    are exact, and take about a second for
    the median of 100 000 draws and a
    hundred times that for a million."""
    q, a = exact_q(q), (1 - exact_q(p)) / 2
    if not 0 < q < 1:
        raise ValueError("need 0 < q < 1")
    n = as_count(n)
    if 2 * q <= 1:
        got = _ranks(n, q, a)
        lo, hi, low, high, whole = got
    else:
        got = _ranks(n, 1 - q, a)
        hi, lo, high, low, whole = got
        lo = lo and n + 1 - lo
        hi = hi and n + 1 - hi
    cover = (whole - low - high) / whole
    return lo or None, hi or None, cover


def quantile_interval(xs, q, p=0.95):
    """Order-statistic interval for the
    q-quantile of the process behind sorted
    xs, assuming independent draws. An end
    is None where the sample is too small
    to bound the quantile on that side. See
    quantile_ranks()."""
    xs = natives(xs)
    lo, hi, _ = quantile_ranks(len(xs), q, p)
    return (xs[lo - 1] if lo else None,
            xs[hi - 1] if hi else None)


def _z(p):
    """Normal quantile for a two-sided level
    p. Below one half it is found from p / 2,
    the distance from the middle, which the
    tail (1 - p) / 2 would round away."""
    if p < 0.5:
        return t_within(p, math.inf)
    return -z_ppf((1 - p) / 2)


def wilson(k, n, p=0.95):
    """Wilson score interval for k successes
    in n independent trials, at level p.
    Without a success the lower limit is 0,
    and without a failure the upper one is
    1. The two are set: the subtraction and
    the addition that give the other limits
    would round them to numbers just inside
    the range, and an interval that has to
    contain 0 or 1 would miss it."""
    n, k = as_count(n), as_count(k, "k")
    if not n:
        raise ValueError("no trials")
    if k > n:
        raise ValueError("need 0 <= k <= n")
    p = native(p)
    if not 0 < p < 1:
        raise ValueError("need 0 < p < 1")
    z = _z(p)
    ph = k / n
    den = 1 + z * z / n
    mid = (ph + z * z / (2 * n)) / den
    var = ph * (1 - ph) / n + z * z / (4 * n * n)
    half = z * math.sqrt(var) / den
    low = mid - half if k else 0.0
    high = mid + half if k < n else 1.0
    return low, high


def _span(exposure, level):
    """An exposure and a level as native()
    gives them: a finite time above zero,
    and a probability between 0 and 1."""
    exposure = native(exposure)
    level = native(level)
    if not 0 < exposure < math.inf:
        raise ValueError("no exposure")
    if not 0 < level < 1:
        raise ValueError("need 0 < level < 1")
    return exposure, level


def _rate(count, exposure):
    """An expected count over an exposure:
    one division of the two numbers that
    they are, rounded once. Divided by a
    float of sixteen bits, the quotient is
    rounded to sixteen bits, and an upper
    bound can then lie below what it bounds
    by more than its printed digits allow.
    A rate beyond the floats is refused:
    inf is no lower limit of a rate."""
    rate = Fraction(count) / Fraction(exposure)
    try:
        return float(rate)
    except OverflowError:
        raise ValueError("no rate in a float")


def quiet_bound(exposure, level=0.95):
    """One-sided upper confidence bound for
    the rate of a homogeneous Poisson process
    when no event was counted: the rate at
    which a silence of this exposure still
    has probability 1 - level. The exposure
    is the time in which an event would have
    been counted, or that time multiplied by
    the probability that an event is kept."""
    exposure, level = _span(exposure, level)
    return _rate(-math.log1p(-level), exposure)


def poisson_tails(n, mean):
    """P(X <= n) and P(X > n) for a Poisson
    count with the given expected value,
    each summed from its own terms: a tail
    next to zero would lose its digits as 1
    less the other. The terms are built
    outward from the largest one by their
    ratios and divided by their sum, so that
    a large mean does not underflow. The
    mean is finite and not negative, and is
    held as a float."""
    mean = float(native(mean))
    if not 0 <= mean < math.inf:
        raise ValueError("need a mean >= 0")
    if n < 0:
        return 0.0, 1.0
    if not mean:
        return 1.0, 0.0
    mode = int(mean)
    parts = [0.0, 0.0]  # up to n, above n
    term, i = 1.0, mode
    while i >= 0 and term > 0:
        parts[i > n] += term
        term *= i / mean
        i -= 1
    term, i = 1.0, mode
    while term > 0:
        i += 1
        term *= mean / i
        parts[i > n] += term
    total = parts[0] + parts[1]
    return parts[0] / total, parts[1] / total


def poisson_at_most(n, mean):
    """P(X <= n), see poisson_tails()."""
    return poisson_tails(n, mean)[0]


def _mean_at(n, tail, above=False):
    """The expected value at which a count of
    at most n has the probability `tail`,
    or, with `above`, a count above n. The
    interval is halved until its ends are
    neighboring floats, however near zero
    the value lies."""
    def small(mean):
        low, high = poisson_tails(n, mean)
        if above:
            return high < tail
        return low > tail
    lo, hi = 0.0, n + 1.0
    while small(hi):
        lo, hi = hi, 2 * hi
    while True:
        mid = (lo + hi) / 2
        if mid in (lo, hi):
            return mid
        if small(mid):
            lo = mid
        else:
            hi = mid


def rate_upper(n, exposure, level=0.95):
    """One-sided upper confidence bound for
    the rate of a homogeneous Poisson process
    from a count of n in the exposure: the
    rate at which a count this low still has
    probability 1 - level. For n = 0 it is
    quiet_bound(). The count is a whole
    number of events, see as_count(). Below
    a level of one half the rate is found
    from the other tail, a count above n
    with probability `level`: 1 - level
    would be rounded there."""
    n = as_count(n)
    exposure, level = _span(exposure, level)
    if not n:
        return quiet_bound(exposure, level)
    if level < 0.5:
        top = _mean_at(n, level, True)
    else:
        top = _mean_at(n, 1 - level)
    return _rate(top, exposure)


def rate_limits(n, exposure, level=0.95):
    """Exact equal-tailed confidence limits for
    the rate of a homogeneous Poisson process
    from a count of n in the exposure. At
    the lower limit a count this high, and at
    the upper one a count this low, has
    probability (1 - level) / 2. For n = 0
    the lower limit is 0. Each limit is
    found from its own tail."""
    n = as_count(n)
    exposure, level = _span(exposure, level)
    a = (1 - level) / 2
    low = 0.0
    if n:
        low = _mean_at(n - 1, a, True)
    high = _mean_at(n, a)
    return (_rate(low, exposure),
            _rate(high, exposure))


def bootstrap(xs, stat, n_boot=1000, seed=1):
    """Sorted bootstrap replicates of stat."""
    rng = random.Random(seed)
    n = len(xs)
    reps = [stat(rng.choices(xs, k=n))
            for _ in range(as_count(n_boot))]
    reps.sort()
    return reps


def block_bootstrap(xs, stat, block,
                    n_boot=1000, seed=1):
    """Moving-block bootstrap: resample runs
    of `block` consecutive values, so that
    dependence within a block is kept."""
    rng = random.Random(seed)
    n = len(xs)
    block = as_count(block, "block", 1)
    pieces = -(-n // block)
    reps = []
    for _ in range(as_count(n_boot)):
        out = []
        for _ in range(pieces):
            i = rng.randrange(n - block + 1)
            out.extend(xs[i:i + block])
        reps.append(stat(out[:n]))
    reps.sort()
    return reps


def _root(f):
    """Square root of a fraction, rounded once
    to a float. The root is taken of a whole
    number: the fraction times a power of
    four that gives it some 128 bits. One
    more bit records what the two floors
    dropped. The whole number is then cut
    to the 53 bits of a float, or to as many
    as the smallest floats have, and rounded
    to the nearer neighbor, at a tie to the
    even one. The square itself never has to
    be a float, and the root is returned
    where it is one and its square is not."""
    if not f:
        return 0.0
    a, b = f.numerator, f.denominator
    k = (128 - a.bit_length()
         + b.bit_length()) // 2
    if k >= 0:
        q, rest = divmod(a << 2 * k, b)
    else:
        q, rest = divmod(a, b << -2 * k)
    r = math.isqrt(q)
    r = 2 * r + (r * r != q or rest != 0)
    cut = max(r.bit_length() - 53, k - 1073)
    r, low = divmod(r, 1 << cut)
    half = 1 << cut - 1
    r += low > half or (low == half and r % 2)
    return math.ldexp(r, cut - k - 1)


def _plain(v):
    """Whether a float holds this reading as
    it stands. Above 2**53 floats no longer
    hold every whole number, and a whole
    reading of that size counts as none of
    theirs, even if one is equal to it."""
    if isinstance(v, float):
        return True
    return abs(v) <= 2 ** 53 and float(v) == v


def _kept(f, readings):
    """A result that was formed exactly and
    lies among the readings: as a float,
    rounded once, if floats hold all of
    them. Among whole time stamps of
    nineteen digits a float can lie outside
    all of them, and the result is returned
    as the whole number or the fraction
    that it is."""
    x = float(f)
    if x == f or all(map(_plain, readings)):
        return x
    return int(f) if f.denominator == 1 else f


def variance(xs):
    """Experimental variance of the readings:
    the sum of their squared distances from
    the mean over n - 1, as the fraction
    that it is. The readings are taken as
    natives() gives them and summed as
    fractions, so that they can be of
    several types. Python leaves a float
    beside a decimal undefined in its
    statistics, and stops at it. There are
    two readings at least, and all of them
    are finite."""
    xs = natives(list(xs))
    try:
        fs = [Fraction(x) for x in xs]
    except (OverflowError, ValueError):
        raise ValueError("no finite readings")
    return st.variance(fs)


def mean(xs):
    """Mean of the readings, each as native()
    gives it, of one type or of several: an
    exact sum, rounded once. A whole mean of
    ints stays an int, and see _kept(). With
    a reading that is no finite number the
    mean is the sum of the floats over their
    number."""
    xs = natives(list(xs))
    try:
        f = st.mean([Fraction(x) for x in xs])
    except (OverflowError, ValueError):
        if not xs:
            raise
        return sum(map(float, xs)) / len(xs)
    if set(map(type, xs)) == {int}:
        if f.denominator == 1:
            return int(f)
    return _kept(f, xs)


def _held(a, b, var, n, sx, sxx):
    """The two coefficients of a fitted line
    as floats, where the line through the
    floats is the fitted line, or lies
    within a thousandth of its standard
    uncertainty everywhere. That is so if
    the squares by which the rounding moves
    the n fitted values sum to a millionth
    of the residual variance or less: no
    fitted value at any x, the intercept at
    x = 0 among them, is then moved by more
    than a thousandth of its uncertainty,
    and neither is the slope. Otherwise the
    intercept, the slope or both are not
    rounded, however the readings are
    stored. A known rounding is an error to
    be left out, not a part of the
    uncertainty: the line would no longer
    be the line of least squares, nor s its
    scatter. A fit with no residual rounds
    nothing that a float does not hold."""
    fa, fb = float(a), float(b)
    for ka, kb in ((fa, fb), (a, fb), (fa, b),
                   (a, b)):
        da = Fraction(ka) - a
        db = Fraction(kb) - b
        moved = n * da * da + 2 * da * db * sx
        moved += db * db * sxx
        if 10 ** 6 * moved <= var:
            break
    return _whole(ka), _whole(kb)


def _whole(c):
    """A float as it is, and a fraction as
    the whole number that it is, if it is
    one."""
    whole = getattr(c, "denominator", 0) == 1
    return int(c) if whole else c


def linfit(xs, ys):
    """Least-squares line y = a + b * x.
    Returns a, b, u(a), u(b) and the residual
    standard deviation s. The points are
    pairs: a y without an x, or an x without
    a y, is refused. So are fewer than three
    points, which leave no residual to
    estimate s from, and xs that are all the
    same, which give no slope. Every reading
    is taken as the fraction that it is, the
    sums are exact, and only the five results
    are rounded, the last three as roots of
    exact squares. Time stamps that count
    nanoseconds from 1970 are the reason:
    as floats they lie 256 ns apart. A mean
    of them taken in floating point loses
    the intervals between them, and with the
    intervals the slope. An intercept that
    is carried back to x = 0 through one
    rounded product of that size loses the
    offset between two clocks. The intercept
    and the slope are rounded only where
    the line through the floats is the
    fitted line, or within a thousandth of
    its uncertainty, see _held(): a float
    slope times such a stamp is tens of
    nanoseconds off. With an origin
    subtracted from the stamps first, in
    whole numbers, readings that scatter
    give floats. A reading that is not a
    finite number is refused."""
    xs = [Fraction(native(x)) for x in xs]
    ys = [Fraction(native(y)) for y in ys]
    n = len(xs)
    if len(ys) != n:
        raise ValueError("one y for each x")
    if n < 3:
        raise ValueError("need three points")
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    syy = sum(y * y for y in ys)
    d = n * sxx - sx * sx
    if not d:
        raise ValueError("xs do not vary")
    m = n * sxy - sx * sy
    b = m / d
    a = (sy - b * sx) / n
    rss = (n * syy - sy * sy - b * m) / n
    var = rss / (n - 2)
    ua = _root(var * sxx / d)
    ub = _root(var * n / d)
    a, b = _held(a, b, var, n, sx, sxx)
    return a, b, ua, ub, _root(var)


def paired_exact(fewer, more):
    """Exact test for matched pairs, the exact
    test of McNemar. Of the pairs that
    changed, `fewer` went one way and `more`
    the other. If the pairs are independent
    and either way is equally likely for
    each, the number that went the first
    way is binomial with 1/2, given how many
    changed. Returns the chance of `fewer`
    or fewer, a fraction: the one-sided
    test, whichever of the two counts is
    the larger. The two-sided test is twice
    the smaller of this and of the chance
    with the two counts exchanged, and at
    most 1: with as many one way as the
    other, the middle is in both tails."""
    fewer = as_count(fewer, "fewer")
    n = fewer + as_count(more, "more")
    part = sum(math.comb(n, i)
               for i in range(fewer + 1))
    return Fraction(part, 2 ** n)


def exchange_exact(diffs):
    """Exact test for paired differences in
    whole numbers, each the result under one
    label less the result under the other.
    To exchange the two labels of a pair is
    to change the sign of its difference.
    If the pairs are independent and the
    exchange leaves the distribution of
    each as it is, every assignment of
    signs to the sizes that were observed
    is equally likely. Returns the chance
    of a sum that is the observed one or
    more, a fraction: the one-sided test
    for a gain. With the differences negated
    it is the test for a loss, and the
    two-sided test is twice the smaller, at
    most 1. A pair may be a group that is
    exchanged as a whole, with the sum of
    its differences. Sizes of 1 alone give
    paired_exact()."""
    ways, total, pairs = {0: 1}, 0, 0
    for d in map(native, diffs):
        size = as_count(abs(d), "difference")
        if not size:
            continue
        total += size if d > 0 else -size
        pairs += 1
        new = {}
        for s, k in ways.items():
            for t in (s - size, s + size):
                new[t] = new.get(t, 0) + k
        ways = new
    part = sum(k for s, k in ways.items()
               if s >= total)
    return Fraction(part, 2 ** pairs)


def cohen_kappa(a, b):
    """Cohen's kappa: chance-corrected
    agreement of two lists of labels, one
    label of each rater for every item. The
    agreement expected by chance comes from
    the shares of the labels, so that kappa
    moves with those shares at fixed error
    rates. It is taken from the counts, in
    whole numbers: the agreements above
    those expected by chance, over the
    disagreements expected by chance. If
    both raters give every item the same
    one label, agreement is complete, by
    chance as well, and kappa is not
    defined."""
    a, b = list(a), list(b)
    n = len(a)
    if len(b) != n:
        raise ValueError("two labels an item")
    if not n:
        raise ValueError("no items")
    same = sum(x == y for x, y in zip(a, b))
    chance = sum(a.count(c) * b.count(c)
                 for c in set(a) | set(b))
    if chance == n * n:
        raise ValueError("kappa needs two labels")
    return ((n * same - chance)
            / (n * n - chance))
