"""Binary operators and their precedence."""


def add(a, b):
    return a + b


def sub(a, b):
    return a - b


def mul(a, b):
    return a * b


# operator -> (function, precedence). Higher precedence binds tighter.
OPERATORS = {
    "+": (add, 1),
    "-": (sub, 1),
    "*": (mul, 2),
}


def precedence(operator):
    """Binding strength of a binary operator; higher binds tighter."""
    return OPERATORS[operator][1]


def apply(operator, a, b):
    """Apply the named binary operator to two operands."""
    return OPERATORS[operator][0](a, b)
