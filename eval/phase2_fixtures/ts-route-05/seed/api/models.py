"""The User model, an in-memory store, and a lookup helper."""

from dataclasses import dataclass


@dataclass(frozen=True)
class User:
    id: int
    name: str


@dataclass(frozen=True)
class NotFound:
    """Result returned when a requested resource does not exist."""

    resource: str


STORE = {
    1: User(id=1, name="Ada"),
    2: User(id=2, name="Linus"),
}


def get(user_id):
    """Return the :class:`User` with this id, or ``None`` if there is none."""
    return STORE.get(user_id)
