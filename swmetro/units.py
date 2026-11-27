"""Quantity values with units for time and
information. Decimal (SI) and binary
(IEC 80000-13) prefixes are kept apart.
The factors are whole numbers or fractions:
a billionth is no binary number, and the
float 1e-9 turned one second into
999999999.9999999 ns."""
import math
from fractions import Fraction

from .stats import native

_BIN = {"Ki": 2 ** 10, "Mi": 2 ** 20,
        "Gi": 2 ** 30, "Ti": 2 ** 40}
_SI = {"k": 10 ** 3, "M": 10 ** 6,
       "G": 10 ** 9, "T": 10 ** 12,
       "m": Fraction(1, 10 ** 3),
       "u": Fraction(1, 10 ** 6),
       "n": Fraction(1, 10 ** 9)}
# unit: (kind of quantity, factor to base unit)
_BASE = {"s": ("time", 1),
         "bit": ("information", 1),
         "B": ("information", 8)}


def parse(unit):
    """Return (kind of quantity, factor)."""
    if unit in _BASE:
        return _BASE[unit]
    for table in (_BIN, _SI):
        for prefix, mult in table.items():
            base = unit[len(prefix):]
            if unit.startswith(prefix) \
                    and base in _BASE:
                kind, factor = _BASE[base]
                return kind, mult * factor
    raise ValueError(f"unknown unit: {unit}")


def _whole(x):
    """A fraction, or the whole number that it
    is."""
    if x.denominator == 1:
        return x.numerator
    return x


class Q:
    """A quantity value: a number and a unit.
    The number is held as native() gives it:
    a whole number of a fixed width as an
    int, and a truth value not at all."""

    def __init__(self, value, unit):
        self.value = native(value)
        self.unit = unit
        self.kind, self.factor = parse(unit)

    def to(self, unit):
        """Convert to another unit of the same
        kind of quantity. The ratio of the two
        factors is exact. A whole number or a
        fraction is converted as the number
        that it is, and a float is rounded
        once. Where the ratio is one, the
        value is returned as it was given."""
        kind, factor = parse(unit)
        if kind != self.kind:
            raise TypeError(
                f"{self.kind} is not {kind}")
        ratio = Fraction(self.factor) / factor
        v = self.value
        if ratio == 1:
            return Q(v, unit)
        if isinstance(v, float):
            if not math.isfinite(v):
                return Q(v * float(ratio), unit)
            return Q(float(Fraction(v) * ratio),
                     unit)
        x = Fraction(v) * ratio
        return Q(_whole(x), unit)

    def __add__(self, other):
        """Sum in the unit of the first value.
        Two floats are added as floats. With a
        whole number or a fraction among the
        two the sum is exact: a float zero
        added to a time stamp of nineteen
        digits would move it by up to 128."""
        x = other.to(self.unit).value
        pair = self.value, other.value
        floats = [v for v in pair
                  if isinstance(v, float)]
        if len(floats) == 2 or not all(
                map(math.isfinite, floats)):
            return Q(self.value + x, self.unit)
        x = Fraction(other.value) * other.factor
        x = Fraction(self.value) + x / self.factor
        return Q(_whole(x), self.unit)

    def __repr__(self):
        """Six significant digits of a float or
        a fraction, and a whole number in
        full."""
        v = self.value
        if isinstance(v, Fraction):
            v = float(v)
        if isinstance(v, float):
            return f"{v:g} {self.unit}"
        return f"{v} {self.unit}"
