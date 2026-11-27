"""Checks of a measurement record of the form
of Listing 22.3: which inputs of its decision
are missing, which of its values cannot be,
which decision it supports under the one
procedure supported here, and whether two
records describe one quantity. A record is
opened with read(), which keeps its numbers
as they are written. These are checks of
consistency. A record that passes them is not
thereby a record whose model fits the
system."""
import json
import math
from decimal import Context, Decimal
from fractions import Fraction

from .dist import t_isf, t_mid
from .units import Q

FORMAT = "measurement record, version 5"
REQUIRED = (
    ("measurand", "quantity convention target"
     " start_event end_event unit"),
    ("capture", "clock records_lost"),
    ("result", "value u dof method"
     " significant_digits"),
    ("decision", "procedure rule k limit risk"
     " risk_is"),
)
LISTS = ("components", "covariances",
         "not_evaluated")
# The one supported procedure, and what a
# record that names it must also state.
PROCEDURE = "upper limit, t guard band, v1"
EXPECTS = (
    ("result", "method",
     "budget, Student t interval"),
    ("decision", "rule",
     "accept if value + k u <= limit"),
    ("decision", "k",
     "t quantile at 1 - risk, result dof"),
    ("decision", "risk_is",
     "probability under the model"),
)
RANGES = (
    ("result", "value", lambda v: True,
     "a finite number"),
    ("result", "u", lambda v: v >= 0,
     "a finite number >= 0"),
    ("result", "dof", lambda v: v > 0,
     "a finite number > 0"),
    ("decision", "limit", lambda v: True,
     "a finite number"),
    ("decision", "risk",
     lambda v: 0 < v < 0.5,
     "a number in (0, 0.5)"),
)
SAME = (
    ("measurand", "quantity start_event"
     " end_event convention target"
     " conditions"),
    ("data", "run_weighting"),
)
# Of these, what a study may vary. The
# others define the quantity.
VARIED = ("conditions",)
# Names, which are text when they are known.
TEXT = SAME + (("measurand", "unit"),
               ("capture", "clock"))
WANTED = ("rate", "established")
ALL = "all"  # digits: working values, unrounded
# Relative allowances: for two working values
# computed in two ways, and for a coverage
# factor computed numerically. A risk within
# LEAST of 0 or of 1/2 is not supported. With
# fewer than FEW degrees of freedom a factor
# next to zero is bounded and not computed.
# A factor too large to compute is above FAR.
EPS = Fraction(1, 10 ** 12)
DELTA = Fraction(1, 10 ** 6)
LEAST = Fraction(1, 10 ** 300)
FEW = Fraction(1, 1000)
FAR = Fraction(10) ** 149
# What is shown is a decimal: exact if it has
# at most 40 digits. A guard band is shown
# with the 17 digits of a float.
SHOW = Context(prec=40)
BAND = Context(prec=17)


def _once(pairs):
    """An object of the file. A name that
    occurs twice in it is refused: the default
    reader would keep the last value and say
    nothing of the first."""
    part = dict(pairs)
    if len(part) != len(pairs):
        raise ValueError("a name occurs twice")
    return part


def _refuse(token):
    raise ValueError(f"{token} is not JSON")


def read(text):
    """A record from the text of its file. A
    number written with a point or an exponent
    is kept as the decimal that is written,
    and an integer as an integer. The default
    reader makes a float of the first, the
    nearest binary number, and that may be
    another number: 9007199254740993.0 comes
    back as 9007199254740992.0. It also takes
    the bare words NaN, Infinity and
    -Infinity for numbers, which the grammar
    of JSON does not have. They are refused.
    So is a text that is JSON and no object,
    a list or a null: a record is an object."""
    rec = json.loads(text, parse_float=Decimal,
                     parse_constant=_refuse,
                     object_pairs_hook=_once)
    if not isinstance(rec, dict):
        raise ValueError("a record is an object")
    return rec


def is_unknown(v):
    """None, an empty string, or a prompt left
    over from the template stands for unknown.
    Zero does not: zero is a statement."""
    if isinstance(v, str):
        s = v.strip()
        return not s or s.startswith("<")
    return v is None


def _obj(v):
    """An object, or an empty one in the place
    of anything else."""
    return v if isinstance(v, dict) else {}


def _part(rec, section):
    return _obj(_obj(rec).get(section))


def _get(rec, section, key):
    return _part(rec, section).get(key)


def _rows(part, name):
    """Entries of a list, each with its place
    for a message. An entry that is not an
    object counts as an empty one."""
    rows = part.get(name)
    if not isinstance(rows, list):
        return []
    return [(f"{name}[{i}]",
             r if isinstance(r, dict) else {})
            for i, r in enumerate(rows)]


def missing(rec):
    """Inputs of the decision that are not
    known: the required fields, the three
    lists, and in each component its name,
    its u and, for a correction, its value.
    Nothing else in the record is examined."""
    out, rec = [], _obj(rec)
    for section, keys in REQUIRED:
        for key in keys.split():
            v = _get(rec, section, key)
            if is_unknown(v):
                out.append(f"{section}.{key}")
    n = _get(rec, "result", "significant_digits")
    step = _get(rec, "result", "value_step")
    rounded = not is_unknown(n) and n != ALL
    if rounded and is_unknown(step):
        out.append("result.value_step")
    for name in LISTS:
        if not isinstance(rec.get(name), list):
            out.append(name)
    if rec.get("components") == []:
        out.append("components")
    for tag, c in _rows(rec, "components"):
        keys = ["name", "u"]
        if "value" in c:
            keys.append("value")
        for key in keys:
            if is_unknown(c.get(key)):
                out.append(f"{tag}.{key}")
    return out


def _bad(v, test):
    """True if a known value is not a finite
    number that passes the test. A boolean
    is not a number here. Nor is a number
    beyond the range of a float, too large
    for one or too small to differ from zero
    in one: the digits of a decimal are read
    exactly, but its range is not extended."""
    if is_unknown(v):
        return False
    kinds = (int, float, Decimal)
    if isinstance(v, bool):
        return True
    if not isinstance(v, kinds):
        return True
    try:
        f = float(v)
    except (OverflowError, ValueError):
        return True
    if not math.isfinite(f):
        return True
    if f == 0 and v != 0:
        return True
    return not test(v)


def invalid(rec):
    """Values that cannot be, and statements
    that contradict the named procedure. A
    name that is known is text: a number or
    a list in the place of a clock or of an
    event names nothing. In a rounded record
    an uncertainty or the degrees of freedom
    with more digits than it declares, or a
    value between two steps, was not rounded
    as the record says."""
    if not isinstance(rec, dict):
        return ["the record is not an object"]
    out = []
    for section, keys in TEXT:
        for key in keys.split():
            v = _get(rec, section, key)
            if not (is_unknown(v)
                    or isinstance(v, str)):
                out.append(f"{section}.{key}"
                           " is not text")
    for section, key, test, text in RANGES:
        if _bad(_get(rec, section, key), test):
            where = f"{section}.{key}"
            out.append(f"{where} is not {text}")
    lost = _get(rec, "capture", "records_lost")
    if not is_unknown(lost):
        if type(lost) is not int or lost < 0:
            out.append("capture.records_lost"
                       " is not a count")
    n = _get(rec, "result", "significant_digits")
    if not is_unknown(n) and n != ALL:
        if type(n) is not int or not 0 < n < 18:
            out.append(
                "result.significant_digits"
                " is not all or 1 to 17")
    step = _get(rec, "result", "value_step")
    if _bad(step, lambda v: v >= 0):
        out.append("result.value_step is not"
                   " a finite number >= 0")
    elif n == ALL and not is_unknown(step):
        if step != 0:
            out.append("result.value_step is not"
                       " 0 for working values")
    for tag, c in _rows(rec, "components"):
        if _bad(c.get("u"), lambda v: v >= 0):
            out.append(f"{tag}.u is not"
                       " a finite number >= 0")
        if _bad(c.get("value"), lambda v: True):
            out.append(f"{tag}.value is not"
                       " a finite number")
    if not out and type(n) is int:
        out = _unrounded(rec, n, step)
    named = _get(rec, "decision", "procedure")
    if named == PROCEDURE:
        for section, key, text in EXPECTS:
            v = _get(rec, section, key)
            if not is_unknown(v) and v != text:
                out.append(
                    f"{section}.{key} is not"
                    " that of the procedure")
    return out


def _unrounded(rec, n, step):
    """Numbers of a rounded record that its
    rounding cannot have written: an
    uncertainty or degrees of freedom with
    more than n significant digits, trailing
    zeros apart, and an estimate that is no
    multiple of its increment."""
    r, out = _part(rec, "result"), []
    seen = [("result.u", r.get("u")),
            ("result.dof", r.get("dof"))]
    for tag, c in _rows(rec, "components"):
        seen.append((f"{tag}.u", c.get("u")))
    for where, v in seen:
        if is_unknown(v):
            continue
        digits = _dec(v).as_tuple().digits
        text = "".join(map(str, digits))
        if len(text.rstrip("0")) > n:
            out.append(f"{where} is not"
                       f" rounded to {n} digits")
    v = r.get("value")
    if not (is_unknown(v) or is_unknown(step)):
        if step and exact(v) % exact(step):
            out.append("result.value is not"
                       " a multiple of its step")
    return out


def _dec(x):
    """The decimal that a number states: the
    one that was read from the file, an
    integer as it stands, or the decimal that
    Python prints for a float."""
    if isinstance(x, Decimal):
        return x
    if isinstance(x, int):
        return Decimal(int(x))
    return Decimal(float.__repr__(x))


def exact(x):
    """A stated number as an exact fraction,
    so that 0.15 is 3/20 and not the binary
    number next to it."""
    return Fraction(_dec(x))


def bounds(x, digits):
    """Lowest and highest value that rounding
    to `digits` significant digits can have
    turned into x, as exact fractions: half a
    step of the last digit away, except below
    a power of ten, where the steps are ten
    times finer. With two digits 9.6 stands
    for 9.55 to 9.65, and 10 for 9.95 to 10.5.
    Below zero the two are mirrored: -10
    stands for -10.5 to -9.95. A working
    value (ALL) and 0 stand for themselves.
    That x has no more than `digits` digits
    is for the caller to see to; invalid()
    does it for a record."""
    d = _dec(x)
    x = Fraction(d)
    if digits == ALL or x == 0:
        return x, x
    top = d.adjusted()
    step = Fraction(10) ** (top - digits + 1)
    near, size = step / 2, abs(x)
    if size == Fraction(10) ** top:
        near = step / 20
    lo, hi = size - near, size + step / 2
    return (lo, hi) if x > 0 else (-hi, -lo)


def _float(q):
    try:
        return float(q)
    except OverflowError:
        return math.inf if q > 0 else -math.inf


def _tidy(d):
    """A decimal without trailing zeros, and
    written out in full where that takes no
    more than 40 digits: 100 and not 1E+2."""
    d = d.normalize(SHOW)
    if d.as_tuple().exponent > 0:
        if d.adjusted() < SHOW.prec:
            d = d.quantize(1, context=SHOW)
    return d


def _shown(q):
    """A fraction as a decimal, for display:
    exact if it has at most 40 digits."""
    n, d = q.numerator, q.denominator
    return _tidy(SHOW.divide(Decimal(n), d))


def _full(q):
    """A fraction as a decimal for display,
    as _shown() gives it where that is exact,
    and otherwise in full where the fraction
    is a decimal: its denominator then has no
    factors but 2 and 5. Two ends that lie
    half a step of 1e-45 on either side of a
    limit stay two ends, each on its side."""
    d = _shown(q)
    k, twos, fives = q.denominator, 0, 0
    while k % 2 == 0:
        k, twos = k // 2, twos + 1
    while k % 5 == 0:
        k, fives = k // 5, fives + 1
    if Fraction(d) == q or k > 1:
        return d
    e = max(twos, fives)
    whole = q.numerator * 2 ** (e - twos)
    whole *= 5 ** (e - fives)
    return Decimal(f"{whole}e{-e}")


def _root(q):
    """Square root of a fraction, as a decimal
    of at most 40 digits. No float is formed
    on the way, so that the square of 1e-200
    and that of 1e200 keep their roots."""
    return _tidy(SHOW.sqrt(_shown(q)))


def _squares(rec, both=True):
    """Exact bounds of the square of the
    combined standard uncertainty: the sum of
    the squares of the components, each a
    contribution in the unit of the result,
    taken as uncorrelated and as rounded as
    the record says. With `both`, narrowed to
    what the stated value allows as well, or
    None if the two cannot both be true. In a
    rounded record that test is exact. For
    working values it allows the last bits of
    the arithmetic that produced them."""
    r = rec["result"]
    n = r["significant_digits"]
    lo = hi = Fraction(0)
    for _, c in _rows(rec, "components"):
        a, b = bounds(c["u"], n)
        lo, hi = lo + a * a, hi + b * b
    if not both:
        return lo, hi
    s_lo, s_hi = bounds(r["u"], n)
    if n == ALL:
        if abs(s_lo ** 2 - lo) > 2 * EPS * lo:
            return None
        return lo, lo
    if s_hi ** 2 < lo or hi < s_lo ** 2:
        return None
    return max(lo, s_lo ** 2), min(hi, s_hi ** 2)


def spread(rec):
    """Bounds of what the components combine
    to, without regard to the stated value,
    as decimals for display."""
    lo, hi = _squares(rec, both=False)
    return _root(lo), _root(hi)


def combined(rec):
    """Bounds of the combined standard
    uncertainty that the whole record
    supports, or None if the components and
    the stated value cannot both be true, as
    decimals for display. For working values
    the bounds are one number, that of the
    components."""
    both = _squares(rec)
    if both is None:
        return None
    return _root(both[0]), _root(both[1])


def _factor(risk, dof):
    """Coverage factor, the t quantile at
    1 - risk: its computed value, and a lower
    and an upper bound for the decision, as
    fractions. The bounds allow DELTA for the
    accuracy of the computation. In one case
    the factor is known and the three are
    equal: with one degree of freedom and a
    risk of 0.25 it is 1. Above a risk of one
    quarter the factor is computed from the
    distance of the risk from one half, which
    is exact here and which a float of the
    risk can lose altogether. With very few
    degrees of freedom that computation is
    not relied on: the factor is then known
    only to exceed twice that distance, since
    no density of Student's is above
    1 / sqrt(2 pi) at zero. A factor too
    large to compute is known to exceed FAR.
    None stands for what is not computed,
    and as an upper bound for no bound."""
    if dof == 1 and risk == Fraction(1, 4):
        return (Fraction(1),) * 3
    gap = Fraction(1, 2) - risk
    if risk <= gap:
        gap = None
    elif dof < FEW:
        return None, 2 * gap, None
    dof = _float(dof)
    try:
        if gap is None:
            k = t_isf(float(risk), dof)
        else:
            k = t_mid(float(gap), dof)
    except ValueError:
        return None, FAR, None
    k = Fraction(k)
    return k, k * (1 - DELTA), k * (1 + DELTA)


def _parts(rec):
    """What the decision compares, exactly:
    bounds of the headroom, which is the limit
    less the estimate; bounds of the square of
    the combined uncertainty; and the coverage
    factors, with their bounds, for the two
    ends of the degrees of freedom. In a
    rounded record the estimate stands for
    half its stated increment either way. The
    limit and the risk are requirements and
    are read as written."""
    d, r = rec["decision"], rec["result"]
    n = r["significant_digits"]
    h = Fraction(0)
    if n != ALL:
        h = exact(r["value_step"]) / 2
    room = exact(d["limit"]) - exact(r["value"])
    d_lo, d_hi = bounds(r["dof"], n)
    risk = exact(d["risk"])
    ks = _factor(risk, d_hi), _factor(risk, d_lo)
    return (room - h, room + h), _squares(rec), ks


def _over(k, u2, x, strict):
    """True if the guard band k u is above x,
    or at least x when not `strict`, where u2
    is the square of u. The comparison is of
    exact fractions, on the squares. A band is
    never negative, and a k of None is
    unbounded."""
    if x < 0 or (x == 0 and not strict):
        return True
    if u2 == 0:
        return False
    if k is None:
        return True
    left, right = k * k * u2, x * x
    if strict:
        return left > right
    return left >= right


def guard(rec):
    """Bounds, as decimals for display, of the
    two guarded values of a record that has
    passed the checks: the estimate plus the
    guard band, and the estimate less it. The
    band has the 17 digits of a float and is
    added to the estimate exactly, so that
    with no band the estimate appears as it
    is written. The sum is a decimal and is
    returned in full, see _full(). An
    unbounded band gives an infinite
    bound."""
    room, us, ks = _parts(rec)
    limit = exact(rec["decision"]["limit"])
    v = limit - room[1], limit - room[0]
    b = []
    for i, u2 in enumerate(us):
        k = ks[i][0]
        if k is None:
            k = ks[i][i + 1]
        band = Fraction(0)
        if u2 and k is None:
            band = None
        elif u2:
            u = _root(u2)
            band = BAND.multiply(_shown(k), u)
            band = Fraction(band)
        b.append(band)

    def end(x, band, sign):
        if band is None:
            return sign * Decimal("Infinity")
        return _full(x + sign * band)
    return ((end(v[0], b[0], 1),
             end(v[1], b[1], 1)),
            (end(v[0], b[1], -1),
             end(v[1], b[0], -1)))


def decide(rec):
    """Status of a record under the supported
    procedure: accept if value + k u <= limit,
    reject if value - k u > limit, otherwise
    undecided. Here u is combined from the
    components, and k is the t quantile at
    1 - risk for the degrees of freedom of
    the result. Before that, in this order:
    invalid if a value cannot be; unresolved
    if an input is missing, or the format or
    the procedure is another one, or
    components are correlated; invalid if the
    stated u and the components cannot both
    be true, within the rounding that the
    record declares; unresolved if a required
    component was never evaluated or events
    were lost. The comparison itself is of
    exact fractions of the numbers as they
    are written, so that a change of unit
    cannot change it. A decision is given
    only if it holds for every value that a
    rounded record allows, and for a coverage
    factor anywhere within the allowance for
    its computation; otherwise the status is
    unresolved. A factor that is not computed
    is bounded from below only. It can show
    that a record is undecided, and no more.
    The reasons are returned. Unknown is
    never read as zero."""
    why = invalid(rec)
    if why:
        return "invalid", why
    why = [f"wants {m}" for m in missing(rec)]
    if why:
        return "unresolved", why
    d = rec["decision"]
    if rec.get("format") != FORMAT:
        why.append("format not supported")
    if d["procedure"] != PROCEDURE:
        why.append("procedure not supported")
    if rec["covariances"]:
        why.append("correlations not supported")
    risk = exact(d["risk"])
    if min(risk, Fraction(1, 2) - risk) < LEAST:
        why.append("risk not supported")
    if why:
        return "unresolved", why
    if _squares(rec) is None:
        return "invalid", ["result.u does not"
                           " follow from the"
                           " components"]
    for item in rec["not_evaluated"]:
        why.append(f"not evaluated: {item}")
    if rec["capture"]["records_lost"] > 0:
        why.append("records were lost")
    if why:
        return "unresolved", why
    room, (u_lo, u_hi), ks = _parts(rec)
    k_lo, k_hi = ks[0][1], ks[1][2]
    if not _over(k_hi, u_hi, room[0], True):
        return "accept", []
    if not _over(k_hi, u_hi, -room[1], False):
        return "reject", []
    above = _over(k_lo, u_lo, room[1], True)
    held = _over(k_lo, u_lo, -room[0], False)
    if above and held:
        return "undecided", []
    if k_hi is None:
        return "unresolved", ["coverage factor"
                              " not computed"]
    n = rec["result"]["significant_digits"]
    if n == ALL:
        return "unresolved", ["at the limit: the"
                              " coverage factor"
                              " decides"]
    return "unresolved", ["rounding decides:"
                          " wants working values"]


def _rates_related(a, b, pair):
    """True if either record states an
    established relation of the rates of
    exactly these two clocks, each by its
    name as text: the number 7 does not name
    the clock "7"."""
    for rec in (a, b):
        cap = _part(rec, "capture")
        rows = _rows(cap, "clock_relations")
        for _, row in rows:
            clocks = row.get("clocks")
            if not isinstance(clocks, list):
                continue
            if not _text(*clocks):
                continue
            named = sorted(clocks)
            got = row.get("of"), row.get("state")
            if (named, got) == (pair, WANTED):
                return True
    return False


def stale(rec):
    """The outcome stored in a record, if it
    is known and is not the status that
    decide() finds now. The record was then
    changed after the outcome was written,
    or the outcome itself was. The stored
    word takes no part in the decision and
    is reported beside it. Otherwise None."""
    kept = _get(rec, "decision", "outcome")
    if is_unknown(kept):
        return None
    if _differ(kept, decide(rec)[0]):
        return kept
    return None


def _differ(a, b):
    """True unless two values are equal. A
    value that refuses to be compared differs
    from every other."""
    try:
        return bool(a != b)
    except ArithmeticError:
        return True


def _text(*names):
    return all(isinstance(v, str) for v in names)


def comparable(a, b, across=()):
    """Reasons why two records are not results
    for one quantity, and notes on what a
    comparison would have to convert. With no
    reasons, they may be compared. The
    conditions, the load among them, define
    the quantity and must be known and equal,
    unless they are named in `across` as the
    variable under study; a difference there
    is then noted and is no reason. Nothing
    else can be so named: other events,
    another convention, target or weighting
    make another measurand, and results for
    two measurands are not two results for
    one. A mean over runs depends on how the
    runs are weighted, so the weighting must
    be known and equal.
    Each interval is taken on one clock, so
    two different clocks need an established
    relation of their rates that names both.
    Offsets do not enter an elapsed interval
    and are not asked for."""
    reasons, notes = [], []
    a, b = _obj(a), _obj(b)
    if not set(across) <= set(VARIED):
        raise ValueError("only conditions vary")
    for section, keys in SAME:
        for key in keys.split():
            va = _get(a, section, key)
            vb = _get(b, section, key)
            if is_unknown(va) or is_unknown(vb):
                reasons.append(f"{key} unknown")
            elif not _text(va, vb):
                reasons.append(f"{key} is not"
                               " text")
            elif not _differ(va, vb):
                continue
            elif key in across:
                notes.append(f"across {key}")
            else:
                reasons.append(key)
    ua = _get(a, "measurand", "unit")
    ub = _get(b, "measurand", "unit")
    try:
        one = Q(1.0, ub).to(ua)
    except (KeyError, TypeError, ValueError):
        reasons.append("unit")
    else:
        if _differ(ub, ua):
            right = f"{one.value:g} {ua}"
            notes.append(f"1 {ub} = {right}")
    ca = _get(a, "capture", "clock")
    cb = _get(b, "capture", "clock")
    if is_unknown(ca) or is_unknown(cb):
        reasons.append("clock unknown")
    elif not _text(ca, cb):
        reasons.append("clock is not text")
    elif _differ(ca, cb):
        pair = sorted((ca, cb))
        if not _rates_related(a, b, pair):
            reasons.append("clock rate relation")
    return reasons, notes


def in_unit(rec, unit):
    """Value and standard uncertainty of a
    record, converted explicitly to another
    unit of the same kind of quantity, as
    decimals. The factor is the exact ratio
    of the factors of the two units, a whole
    number or a fraction, and it is applied
    to the numbers as they are written, so
    that 0.2414 s is 241.4 ms and nothing
    else. An unknown one stays unknown, and
    nothing else in the record is
    converted."""
    here = Q(1.0, rec["measurand"]["unit"])
    there = here.to(unit)
    f = Fraction(here.factor) / there.factor
    r = _part(rec, "result")
    v, u = r.get("value"), r.get("u")
    if not is_unknown(v):
        v = _shown(exact(v) * f)
    if not is_unknown(u):
        u = _shown(exact(u) * f)
    return v, u
