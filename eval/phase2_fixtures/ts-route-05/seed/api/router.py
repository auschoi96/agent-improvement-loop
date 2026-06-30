"""A tiny method + path router.

Patterns are registered against a (method, path) shape. A path segment written
as ``<name>`` captures that segment and is passed to the handler as a keyword
argument. An unmatched request returns a :class:`~api.models.NotFound`.
"""

from api.models import NotFound


def _split(path):
    return tuple(segment for segment in path.strip("/").split("/") if segment)


def _match(pattern, segments):
    """Return the captured params if ``segments`` match ``pattern``, else ``None``."""
    params = {}
    for pat, seg in zip(pattern, segments):
        if pat.startswith("<") and pat.endswith(">"):
            params[pat[1:-1]] = seg
        elif pat != seg:
            return None
    return params


class Router:
    def __init__(self):
        self._routes = []

    def register(self, method, pattern, handler):
        """Register ``handler`` for ``method`` requests matching ``pattern``."""
        self._routes.append((method.upper(), _split(pattern), handler))

    def dispatch(self, method, path):
        """Route ``(method, path)`` to its handler, or return ``NotFound``."""
        segments = _split(path)
        for route_method, pattern, handler in self._routes:
            if route_method != method.upper() or len(pattern) != len(segments):
                continue
            params = _match(pattern, segments)
            if params is not None:
                return handler(**params)
        return NotFound(resource=path)
