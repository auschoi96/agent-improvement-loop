"""L1 check for ts-fix-01: areas computed through the registry must be correct.

Imports only from the seed ``shapes`` package (restored beside it in the arm
workspace). Two seeded bugs make this fail: ``triangle_area`` uses the wrong
formula, and the registry mis-wires the ``"triangle"`` name. Both must be fixed.
"""

import math

from shapes.registry import area


def test_circle_area():
    assert math.isclose(area("circle", 2.0), math.pi * 4.0)


def test_rectangle_area():
    assert math.isclose(area("rectangle", 3.0, 4.0), 12.0)


def test_triangle_area():
    # A triangle with base 6 and height 4 has area 0.5 * 6 * 4 == 12.
    assert math.isclose(area("triangle", 6.0, 4.0), 12.0)
