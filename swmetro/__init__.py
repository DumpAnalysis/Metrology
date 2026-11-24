"""swmetro: small metrology helpers for the
examples of the book. Standard library only."""
from .core import Budget, Result, combine, fmt
from .core import rect_interval, type_a, u_rect
from .core import u_resolution, u_tri
from .core import welch_satterthwaite

__all__ = [
    "Budget", "Result", "combine", "fmt",
    "rect_interval", "type_a", "u_rect",
    "u_resolution", "u_tri",
    "welch_satterthwaite",
]
