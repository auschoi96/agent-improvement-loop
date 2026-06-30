"""Evaluate an arithmetic expression over the calc helpers."""

from calc.ops import apply, precedence
from calc.parser import tokenize

__all__ = ["evaluate"]

# Silence unused-import warnings while the body is a stub; the intended
# implementation uses all three helpers.
_HELPERS = (apply, precedence, tokenize)


def evaluate(expression):
    """Evaluate ``expression`` (integers with ``+``, ``-``, ``*``) to an int.

    Operator precedence: ``*`` binds tighter than ``+`` and ``-``; operators of
    equal precedence are left-associative. Build this on the existing helpers —
    ``calc.parser.tokenize`` to lex, and ``calc.ops.precedence`` / ``calc.ops.apply``
    to combine operands.
    """
    raise NotImplementedError("implement evaluate() using tokenize, precedence, and apply")
