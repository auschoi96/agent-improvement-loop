"""Load and validate the application configuration.

The config file is a flat ``key: value`` document parsed with the stdlib only
(no PyYAML), so the fixture has no third-party dependency.
"""

from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"

REQUIRED_KEYS = ("app_name", "timeout_seconds")


@dataclass(frozen=True)
class Settings:
    app_name: str
    timeout_seconds: int


def _parse_config(text):
    """Parse a flat ``key: value`` document into a dict, coercing integers."""
    config = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, sep, value = stripped.partition(":")
        if not sep:
            raise ValueError(f"malformed config line: {line!r}")
        key = key.strip()
        value = value.strip()
        if value.lstrip("-").isdigit():
            config[key] = int(value)
        else:
            config[key] = value
    return config


def validate(raw):
    """Build a :class:`Settings` from a raw config mapping.

    Raises ``ValueError`` if a required key is missing.
    """
    missing = [key for key in REQUIRED_KEYS if key not in raw]
    if missing:
        raise ValueError(f"missing required config key(s): {', '.join(missing)}")
    return Settings(
        app_name=raw["app_name"],
        timeout_seconds=raw["timeout_seconds"],
    )


def load(path=DEFAULT_CONFIG_PATH):
    """Read and validate the config at ``path`` (defaults to the bundled file)."""
    raw = _parse_config(Path(path).read_text(encoding="utf-8"))
    return validate(raw)
