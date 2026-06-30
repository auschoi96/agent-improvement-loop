"""L1 check for ts-config-04: a max_retries setting (int, default 3) enforced.

Imports only from the seed ``app`` package. On the seed, ``Settings`` has no
``max_retries`` field and ``validate`` ignores it, so these fail; adding the
setting to ``config.yaml`` and enforcing it in ``settings.py`` makes them pass.
"""

import pytest

from app.settings import Settings, load, validate


def test_loaded_config_has_max_retries_of_three():
    assert load().max_retries == 3


def test_max_retries_defaults_to_three_when_absent():
    settings = validate({"app_name": "x", "timeout_seconds": 5})
    assert isinstance(settings, Settings)
    assert settings.max_retries == 3


def test_explicit_max_retries_is_honored():
    assert validate({"app_name": "x", "timeout_seconds": 5, "max_retries": 7}).max_retries == 7


def test_invalid_max_retries_raises():
    with pytest.raises(ValueError):
        validate({"app_name": "x", "timeout_seconds": 5, "max_retries": 0})
    with pytest.raises(ValueError):
        validate({"app_name": "x", "timeout_seconds": 5, "max_retries": "lots"})


def test_missing_required_key_still_raises():
    with pytest.raises(ValueError):
        validate({"timeout_seconds": 5})
