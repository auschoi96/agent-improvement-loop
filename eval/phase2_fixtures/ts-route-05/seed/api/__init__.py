"""Wire the handlers into a router via build_router()."""

from api.handlers import index
from api.router import Router

__all__ = ["build_router"]


def build_router():
    """Construct the application router with all routes registered."""
    router = Router()
    router.register("GET", "/health", index)
    return router
