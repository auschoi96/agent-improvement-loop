"""L1 check for ts-route-05: GET /users/<id> routes to a get_user handler.

Imports only from the seed ``api`` package. On the seed, ``build_router`` only
registers ``/health``, so a request for a user falls through to ``NotFound`` and
``test_existing_user_is_returned`` fails. Implementing and registering a
``get_user`` handler for ``GET /users/<id>`` makes it pass.
"""

from api import build_router
from api.models import NotFound, User


def test_existing_user_is_returned():
    assert build_router().dispatch("GET", "/users/1") == User(id=1, name="Ada")


def test_missing_user_returns_not_found():
    assert isinstance(build_router().dispatch("GET", "/users/999"), NotFound)
