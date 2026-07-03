"""One-command, durable companion bootstrap.

``ail-companion-start`` collapses the four manual deployer steps — mint a static
token, export ``DATABRICKS_HOST``/``DATABRICKS_TOKEN``, set ``AIL_CATALOG``/
``AIL_SCHEMA``, then run ``python -m ail.companion poll ...`` — into a single
supervised loop.

The companion deliberately refuses a ``--profile`` OAuth login internally: it is a
long-lived local process, and an OAuth bearer refreshes ~hourly but cannot persist
that refresh from a background process, so a long poll would die mid-run. This
supervisor sidesteps that by minting a **fresh static token per cycle** from the
CLI profile, handing it to the companion via the environment, and running exactly
one poll pass each cycle. The profile is used ONLY to mint here; it is never passed
down to the inner companion.

Fail-closed: if the mint fails (bad profile, non-zero CLI, empty/malformed token,
missing host) the supervisor exits non-zero and never runs the companion
unauthenticated.
"""

from __future__ import annotations

import argparse
import configparser
import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from types import FrameType

from ail.companion import cli as companion_cli

_TAG = "[ail.companion.bootstrap]"

# What ``signal.signal`` accepts and returns (a Python callable, SIG_DFL/SIG_IGN, or None).
_SigHandler = Callable[[int, FrameType | None], object] | int | None


# ---------------------------------------------------------------------------
# Per-cycle static-token mint (fail-closed).
# ---------------------------------------------------------------------------


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """Thin, monkeypatchable wrapper around the Databricks CLI invocation."""
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _mint_token(profile: str) -> str:
    """Mint a FRESH static bearer via ``databricks auth token`` (fail-closed)."""
    cmd = ["databricks", "auth", "token", "--profile", profile, "--output", "json"]
    try:
        proc = _run(cmd)
    except OSError as exc:  # CLI not installed / not on PATH
        raise SystemExit(
            f"{_TAG} FATAL: could not run the Databricks CLI ({exc}). Install it and "
            "ensure `databricks` is on PATH."
        ) from exc
    if proc.returncode != 0:
        raise SystemExit(
            f"{_TAG} FATAL: `databricks auth token --profile {profile}` failed "
            f"(exit {proc.returncode}). Check the profile is a valid U2M login "
            f"(`databricks auth login --profile {profile}`).\n{proc.stderr.strip()}"
        )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{_TAG} FATAL: could not parse token JSON from the CLI ({exc}).") from exc
    token = str(payload.get("access_token") or "").strip()
    if not token:
        raise SystemExit(
            f"{_TAG} FATAL: the CLI returned an empty access_token for profile {profile!r}."
        )
    return token


def _profile_host(profile: str) -> str:
    """Derive the workspace host from the profile's ``~/.databrickscfg`` section."""
    cfg_path = Path(os.environ.get("DATABRICKS_CONFIG_FILE") or (Path.home() / ".databrickscfg"))
    parser = configparser.ConfigParser()
    try:
        read = parser.read(cfg_path)
    except configparser.Error as exc:
        raise SystemExit(f"{_TAG} FATAL: could not parse {cfg_path} ({exc}).") from exc
    if not read:
        raise SystemExit(f"{_TAG} FATAL: config file {cfg_path} not found or unreadable.")
    if profile not in parser:
        raise SystemExit(f"{_TAG} FATAL: profile {profile!r} not found in {cfg_path}.")
    host = (parser[profile].get("host") or "").strip()
    if not host:
        raise SystemExit(f"{_TAG} FATAL: profile {profile!r} has no host in {cfg_path}.")
    return host


def mint_static_auth(profile: str) -> tuple[str, str]:
    """Return ``(host, static_access_token)`` for a Databricks CLI profile.

    Fail-closed: raises ``SystemExit`` on any error so the supervisor never runs the
    companion unauthenticated.
    """
    token = _mint_token(profile)
    host = _profile_host(profile)
    return host, token


# ---------------------------------------------------------------------------
# Supervisor loop.
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ail-companion-start",
        description=(
            "Durable one-command companion supervisor: per cycle, mint a fresh static "
            "token from --profile, export auth + AIL_CATALOG/AIL_SCHEMA, then run one "
            "`ail.companion poll` pass. Re-minting per cycle keeps a long run alive past "
            "OAuth token expiry. --profile is used only to mint; it is never passed to "
            "the companion."
        ),
    )
    parser.add_argument(
        "--profile",
        required=True,
        help="Databricks CLI profile to MINT a fresh static token from each cycle.",
    )
    parser.add_argument("--experiment", required=True, help="MLflow experiment id.")
    parser.add_argument("--catalog", required=True, help="UC catalog for framework tables.")
    parser.add_argument("--schema", required=True, help="UC schema for framework tables.")
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=60.0,
        help="Seconds to sleep between cycles (default 60).",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=0,
        help="Number of poll cycles to run; 0 (default) runs until interrupted.",
    )
    # Lean pass-through to the poll loop; other knobs fall to their poll defaults/env.
    parser.add_argument("--warehouse-id", default=None, help="Monitoring SQL warehouse id.")
    parser.add_argument("--agent", default="claude_code", help="Agent name to poll for.")
    args = parser.parse_args(argv)
    if args.interval_seconds < 0:
        parser.error("--interval-seconds must be >= 0")
    if args.max_cycles < 0:
        parser.error("--max-cycles must be >= 0")
    return args


def _poll_passthrough(args: argparse.Namespace) -> list[str]:
    """Core poll flags forwarded each cycle. NEVER includes --profile."""
    argv = [
        "--experiment",
        args.experiment,
        "--catalog",
        args.catalog,
        "--schema",
        args.schema,
        "--agent",
        args.agent,
    ]
    if args.warehouse_id:
        argv += ["--warehouse-id", args.warehouse_id]
    return argv


def _export_env(host: str, token: str, catalog: str, schema: str) -> None:
    os.environ["DATABRICKS_HOST"] = host
    os.environ["DATABRICKS_TOKEN"] = token
    os.environ["AIL_CATALOG"] = catalog
    os.environ["AIL_SCHEMA"] = schema
    # An ambient profile would let the SDK fall back to a mid-run-refreshing OAuth,
    # exactly the failure this supervisor exists to avoid. Drop it.
    os.environ.pop("DATABRICKS_CONFIG_PROFILE", None)


def _install_sigterm(stop: threading.Event) -> tuple[bool, _SigHandler]:
    def _handler(signum: int, frame: FrameType | None) -> None:
        print(f"{_TAG} received SIGTERM; shutting down after the current cycle.", file=sys.stderr)
        stop.set()

    try:
        previous = signal.signal(signal.SIGTERM, _handler)
    except ValueError:  # not the main thread; rely on KeyboardInterrupt
        return False, None
    return True, previous


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    poll_passthrough = _poll_passthrough(args)

    stop = threading.Event()
    installed, previous = _install_sigterm(stop)

    print(
        f"{_TAG} starting: profile={args.profile} experiment={args.experiment} "
        f"catalog={args.catalog} schema={args.schema} interval={args.interval_seconds}s "
        f"max_cycles={args.max_cycles or 'unbounded'}"
    )

    # Tracks the worst outcome across cycles: starts clean (0) and only bumps to
    # non-zero on a poll failure. Returned regardless of HOW the loop ended (signal,
    # max-cycles, or KeyboardInterrupt) so a persistent poll failure is never masked
    # as success by a graceful stop.
    worst_rc = 0
    try:
        cycle = 0
        while not stop.is_set():
            cycle += 1
            host, token = mint_static_auth(args.profile)  # fail-closed: SystemExit on failure
            _export_env(host, token, args.catalog, args.schema)
            print(f"{_TAG} cycle {cycle}: minted fresh token; polling host={host}")
            try:
                rc = companion_cli.main(["poll", *poll_passthrough, "--max-iterations", "1"])
                worst_rc = max(worst_rc, rc)
            except Exception as exc:  # noqa: BLE001 - stay durable across transient cycle errors
                worst_rc = max(worst_rc, 1)
                print(f"{_TAG} cycle {cycle} poll error (continuing): {exc}", file=sys.stderr)
            if args.max_cycles and cycle >= args.max_cycles:
                break
            if stop.wait(args.interval_seconds):
                break
    except KeyboardInterrupt:
        print(f"{_TAG} interrupted; shutting down cleanly.")
    finally:
        if installed:
            with contextlib.suppress(ValueError, TypeError):
                signal.signal(signal.SIGTERM, previous if previous is not None else signal.SIG_DFL)

    return worst_rc


if __name__ == "__main__":
    raise SystemExit(main())
