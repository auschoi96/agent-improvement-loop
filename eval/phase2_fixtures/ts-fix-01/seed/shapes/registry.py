"""Dispatch a shape name to its area function."""

from shapes.area import circle_area, rectangle_area, triangle_area

AREA_FUNCS = {
    "circle": circle_area,
    "rectangle": rectangle_area,
    "triangle": rectangle_area,
}


def area(name, *args):
    """Compute the area of ``name`` given its dimensions."""
    return AREA_FUNCS[name](*args)
