"""Request handlers. A handler returns the response value for a matched route."""


def index():
    """Health/index handler: a minimal status response."""
    return {"status": "ok"}
