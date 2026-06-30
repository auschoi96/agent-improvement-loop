"""L1 check for ts-impl-02: evaluate() must honor precedence and associativity.

Imports only from the seed ``calc`` package. The seed ``evaluate`` is a stub
that raises ``NotImplementedError`` (so these fail); a correct implementation
over the existing helpers makes them pass.
"""

from calc.evaluate import evaluate


def test_multiplication_binds_tighter_than_addition():
    assert evaluate("2 + 3 * 4") == 14


def test_subtraction_is_left_associative():
    assert evaluate("10 - 2 - 3") == 5
