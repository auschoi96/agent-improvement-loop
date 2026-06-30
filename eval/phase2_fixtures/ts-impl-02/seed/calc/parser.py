"""Lexer for the arithmetic mini-language: produce int and operator tokens."""


def tokenize(expression):
    """Split ``expression`` into a list of int operands and operator strings.

    Whitespace is ignored; multi-digit integers are kept whole. Raises
    ``ValueError`` on any character that is not a digit, an operator, or space.
    """
    tokens = []
    number = ""
    for char in expression:
        if char.isspace():
            if number:
                tokens.append(int(number))
                number = ""
            continue
        if char.isdigit():
            number += char
            continue
        if number:
            tokens.append(int(number))
            number = ""
        if char in OPERATOR_CHARS:
            tokens.append(char)
        else:
            raise ValueError(f"unexpected character: {char!r}")
    if number:
        tokens.append(int(number))
    return tokens


OPERATOR_CHARS = frozenset("+-*")
