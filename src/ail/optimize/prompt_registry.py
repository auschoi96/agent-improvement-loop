"""Register agent skills/prompts in the MLflow **UC prompt registry** (human gate).

This is the **explicit, human-run promote step** of the loop. GEPA emits a
human-gated CANDIDATE artifact (``artifacts/gepa_candidate*.json`` — see
:mod:`ail.optimize.gepa_runner`) and deliberately does **not** register or promote
anything. After a human reviews that candidate, this module is what they run to
version the evolved skill body — and the seed it started from — in the
Unity-Catalog-backed MLflow Prompt Registry, stamping the *provenance* of why the
version was promoted onto the registered version as tags.

Versioning the bodies in the prompt registry is also what unlocks MLflow's
``mlflow.genai.optimize_prompts`` path, which requires the prompts it optimizes to
live in the prompt registry.

Design constraints (intentional):

* **Human-gated by construction.** Nothing here is called by ``gepa_runner`` or the
  comparison harness. Registration makes a live Databricks/MLflow call **only**
  when a human invokes it at runtime. Importing this module, and every code path
  the test suite exercises, makes no network call: the MLflow prompt-registry
  client is injected through a small :class:`PromptRegistryClient` seam and is
  mocked entirely in tests.
* **No silent production promotion.** Registering a version never sets a
  ``champion``/``production`` alias unless the caller passes ``alias=`` explicitly.
* **Fail-closed honesty.** :func:`register_gepa_candidate` **refuses** to register a
  candidate that did not improve over the seed on the held-out split (``changed`` is
  ``False``, no held-out validation, or a held-out savings delta that does not beat
  seed) unless the caller passes ``force=True`` — and a forced registration records
  *why* it was non-improving on the version itself, so a registered version can
  never silently masquerade as an improvement.

Provenance & license
--------------------
Clean-room: uses only the **public** ``mlflow.genai`` prompt-registry API
(``register_prompt`` / ``set_prompt_alias`` / ``search_prompts`` / ``load_prompt``,
resolved against the installed mlflow 3.14) and the public Databricks-managed MLflow
URI convention (tracking ``databricks``, registry ``databricks-uc``). See
``PROVENANCE.md``.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from ail.optimize.gepa_runner import DEFAULT_COMPONENT, GepaOptimizationResult
from ail.optimize.lever import token_efficiency_skill

__all__ = [
    "DEFAULT_CATALOG",
    "DEFAULT_SCHEMA",
    "DEFAULT_PROMPT_NAME",
    "TRACKING_URI",
    "REGISTRY_URI",
    "PROMPT_TAG_PREFIX",
    "CHAMPION_ALIASES",
    "PromptSource",
    "PromptProvenance",
    "RegisteredPrompt",
    "PromptRegistryClient",
    "NonImprovingCandidateError",
    "resolve_prompt_name",
    "candidate_improvement",
    "register_prompt_body",
    "register_seed_prompt",
    "register_gepa_candidate",
    "search_registered_prompts",
]

#: Default UC catalog the prompt registry writes to (configurable per call).
DEFAULT_CATALOG = "austin_choi_omni_agent_catalog"
#: Default UC schema under :data:`DEFAULT_CATALOG`.
DEFAULT_SCHEMA = "agent_improvement_loop"
#: Default leaf name for the token-efficiency skill. The on-disk skill *slug* is
#: ``token-efficient-execution`` (hyphens), but a Unity Catalog object name is an
#: SQL identifier, so the registered prompt uses the underscore form.
DEFAULT_PROMPT_NAME = "token_efficient_execution"

#: Databricks-managed MLflow URIs (the registry must be ``databricks-uc`` for the
#: UC-backed prompt registry). Used only when a live client is built (``client`` is
#: ``None``); never touched on import or in tests.
TRACKING_URI = "databricks"
REGISTRY_URI = "databricks-uc"

#: Namespace prefix for the provenance tags this module stamps on a registered
#: version (``ail.prompt.<field>``). The single source of truth for the tag schema:
#: :meth:`PromptProvenance.as_tags` writes keys under it and the lineage publish
#: (:mod:`ail.publish_lineage`) reads them back under the same prefix.
PROMPT_TAG_PREFIX = "ail.prompt"

#: Aliases that designate the production champion of a registered prompt. ``champion``
#: is canonical (see ``docs/PROMPT_REGISTRY.md``); ``production`` is accepted as a
#: synonym. The single source of truth for the champion-alias names — the lineage
#: publish and the ``ail-revert`` CLI both import this so the definition cannot drift.
CHAMPION_ALIASES: tuple[str, ...] = ("champion", "production")


class PromptSource(StrEnum):
    """Where a registered prompt body came from."""

    SEED = "seed"
    GEPA_EVOLVED = "gepa-evolved"


class NonImprovingCandidateError(RuntimeError):
    """Raised when :func:`register_gepa_candidate` refuses a non-improving candidate.

    The ``reason`` explains the refusal (identical to seed, no held-out validation,
    or a held-out savings delta that does not beat seed). Pass ``force=True`` to
    register anyway — the reason is then recorded on the version instead of raising.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, kw_only=True)
class PromptProvenance:
    """Why a prompt version was registered — stamped as version tags.

    Every field is optional so the same shape serves a seed registration (just
    ``source``) and a GEPA-evolved one (the full held-out comparison). Values are
    stringified into ``ail.prompt.*`` tags by :meth:`as_tags`; ``None`` fields are
    omitted so a tag is present only when it carries information.
    """

    source: PromptSource
    suite_version: str | None = None
    suite_content_hash: str | None = None
    changed: bool | None = None
    gepa_best_val_score: float | None = None
    gepa_num_candidates: int | None = None
    component_name: str | None = None
    reflection_lm: str | None = None
    holdout_evolved_promote: str | None = None
    holdout_seed_promote: str | None = None
    holdout_evolved_savings_pct: float | None = None
    holdout_seed_savings_pct: float | None = None
    holdout_savings_delta_pct: float | None = None
    candidate_artifact: str | None = None
    improving: bool | None = None
    registration_reason: str | None = None
    forced: bool = False

    def as_tags(self) -> dict[str, str]:
        """Render the provenance as ``{ail.prompt.<field>: <str value>}`` tags."""
        raw: dict[str, object | None] = {
            "source": self.source.value,
            "suite_version": self.suite_version,
            "suite_content_hash": self.suite_content_hash,
            "changed": self.changed,
            "gepa_best_val_score": self.gepa_best_val_score,
            "gepa_num_candidates": self.gepa_num_candidates,
            "component_name": self.component_name,
            "reflection_lm": self.reflection_lm,
            "holdout_evolved_promote": self.holdout_evolved_promote,
            "holdout_seed_promote": self.holdout_seed_promote,
            "holdout_evolved_savings_pct": self.holdout_evolved_savings_pct,
            "holdout_seed_savings_pct": self.holdout_seed_savings_pct,
            "holdout_savings_delta_pct": self.holdout_savings_delta_pct,
            "candidate_artifact": self.candidate_artifact,
            "improving": self.improving,
            "registration_reason": self.registration_reason,
            "forced": self.forced if self.forced else None,
        }
        tags: dict[str, str] = {}
        for key, value in raw.items():
            rendered = _render_tag(value)
            if rendered is not None:
                tags[f"{PROMPT_TAG_PREFIX}.{key}"] = rendered
        return tags


@dataclass(frozen=True)
class RegisteredPrompt:
    """The outcome of a register call: the version that now exists in the registry."""

    name: str
    version: int
    uri: str
    source: PromptSource
    tags: Mapping[str, str]
    alias: str | None = None
    forced: bool = False
    reason: str | None = None


class PromptRegistryClient(Protocol):
    """The slice of the MLflow prompt-registry API this module needs.

    The default implementation (:class:`_GenAIPromptRegistryClient`) delegates to
    ``mlflow.genai`` against the configured Databricks workspace. Tests inject a
    fake exposing these four methods so no live call is ever made.
    """

    def register_prompt(
        self,
        name: str,
        template: str,
        commit_message: str | None,
        tags: dict[str, str] | None,
    ) -> Any:
        """Register a new prompt version; return an object with ``.version``/``.uri``."""
        ...

    def set_prompt_alias(self, name: str, alias: str, version: int) -> None:
        """Point ``alias`` at ``version`` of prompt ``name``."""
        ...

    def search_prompts(self, filter_string: str | None) -> list[Any]:
        """Return prompts matching ``filter_string`` (each has ``.name``)."""
        ...

    def load_prompt(self, name_or_uri: str) -> Any:
        """Load a prompt version by name or ``prompts:/...`` URI."""
        ...


def resolve_prompt_name(
    name: str,
    *,
    catalog: str = DEFAULT_CATALOG,
    schema: str = DEFAULT_SCHEMA,
) -> str:
    """Return the three-level UC name ``catalog.schema.name``.

    If ``name`` is already three-level (contains two dots) it is returned verbatim,
    so a caller can pass a fully-qualified name to override ``catalog``/``schema``.
    """
    if name.count(".") >= 2:
        return name
    if "." in name:
        raise ValueError(
            f"prompt name {name!r} is partially qualified; pass a bare leaf name or a full "
            "catalog.schema.name"
        )
    return f"{catalog}.{schema}.{name}"


def candidate_improvement(result: GepaOptimizationResult) -> tuple[bool, str]:
    """Decide whether a GEPA candidate improved over seed, with a reason.

    The fail-closed gate behind :func:`register_gepa_candidate`. Returns
    ``(is_improving, reason)``. A candidate is an improvement only when **all** hold:

    * it actually changed from the seed (``changed`` is ``True``);
    * it carries a live held-out validation for both the evolved and seed bodies;
    * its held-out realized token-savings delta (evolved minus seed,
      :attr:`~ail.optimize.gepa_runner.GepaOptimizationResult.holdout_savings_delta_pct`)
      is **strictly positive** — i.e. the evolved body beat seed on tasks GEPA never
      trained on.

    Held-out savings are summed over PROMOTE tasks only (see
    :class:`~ail.optimize.phase2.Phase2Artifact`), so a delta ``> 0`` is the honest
    anti-overfit signal — never a train-set or self-reported score.
    """
    if not result.changed:
        return False, "candidate is identical to the seed (changed=False): nothing to promote"

    evolved = result.holdout_evolved
    seed = result.holdout_seed_baseline
    if evolved is None or seed is None:
        return (
            False,
            "no held-out validation present (holdout_evolved/holdout_seed_baseline missing): "
            "cannot prove the candidate beats seed",
        )

    delta = result.holdout_savings_delta_pct
    if delta is None:
        return (
            False,
            "held-out realized savings unavailable for the evolved or seed body: "
            "cannot prove improvement",
        )
    # Trap non-finite deltas (NaN/±inf) explicitly: `nan <= 0` and `inf <= 0` are both
    # False, so a NaN from an empty PROMOTE set or an upstream math error would slip
    # through a bare `delta <= 0` and register as a fake improvement. Fail closed.
    if not math.isfinite(delta) or delta <= 0:
        return (
            False,
            f"held-out savings delta {delta} pct-pts does not beat seed "
            f"(evolved {evolved.realized_token_savings_pct} vs "
            f"seed {seed.realized_token_savings_pct})",
        )
    return (
        True,
        f"held-out savings delta +{delta} pct-pts beats seed "
        f"(evolved {evolved.realized_token_savings_pct} vs "
        f"seed {seed.realized_token_savings_pct})",
    )


def register_prompt_body(
    *,
    body: str,
    provenance: PromptProvenance,
    name: str = DEFAULT_PROMPT_NAME,
    catalog: str = DEFAULT_CATALOG,
    schema: str = DEFAULT_SCHEMA,
    commit_message: str | None = None,
    alias: str | None = None,
    client: PromptRegistryClient | None = None,
    profile: str | None = None,
) -> RegisteredPrompt:
    """Register ``body`` as a new version of the UC prompt ``catalog.schema.name``.

    Each call creates a **new version** (MLflow versions on every register). The
    ``provenance`` is stamped as ``ail.prompt.*`` version tags so the version carries
    *why* it exists. An ``alias`` is set **only** if the caller passes one — there is
    no implicit production promotion.

    Args:
        body: The prompt/skill body to register (the markdown skill body, verbatim).
        provenance: Why this version is being registered; becomes version tags.
        name: Leaf name, or a full ``catalog.schema.name`` to override the prefix.
        catalog: UC catalog (used only when ``name`` is a bare leaf).
        schema: UC schema (used only when ``name`` is a bare leaf).
        commit_message: Optional human-readable commit message for the version.
        alias: Optional alias (e.g. ``"champion"``) to point at the new version.
        client: Injectable prompt-registry client; a live one is built when ``None``.
        profile: Databricks CLI profile, used only when building a live client.

    Returns:
        A :class:`RegisteredPrompt` describing the version that now exists.
    """
    if not body.strip():
        raise ValueError("refusing to register an empty prompt body")

    full_name = resolve_prompt_name(name, catalog=catalog, schema=schema)
    registry = client if client is not None else _new_prompt_client(profile)
    tags = provenance.as_tags()

    version_obj = registry.register_prompt(full_name, body, commit_message, tags)
    version = int(version_obj.version)
    uri = str(getattr(version_obj, "uri", f"prompts:/{full_name}/{version}"))

    if alias is not None:
        registry.set_prompt_alias(full_name, alias, version)

    return RegisteredPrompt(
        name=full_name,
        version=version,
        uri=uri,
        source=provenance.source,
        tags=tags,
        alias=alias,
        forced=provenance.forced,
        reason=provenance.registration_reason,
    )


def register_seed_prompt(
    *,
    body: str | None = None,
    name: str = DEFAULT_PROMPT_NAME,
    catalog: str = DEFAULT_CATALOG,
    schema: str = DEFAULT_SCHEMA,
    suite_version: str | None = None,
    suite_content_hash: str | None = None,
    commit_message: str | None = None,
    alias: str | None = None,
    client: PromptRegistryClient | None = None,
    profile: str | None = None,
) -> RegisteredPrompt:
    """Register the seed skill body as a ``source=seed`` version (the baseline).

    ``body`` defaults to the on-disk token-efficiency seed skill, so the version
    that every GEPA candidate is measured against is itself versioned and tracked.
    """
    seed_body = body if body is not None else token_efficiency_skill().body
    provenance = PromptProvenance(
        source=PromptSource.SEED,
        suite_version=suite_version,
        suite_content_hash=suite_content_hash,
        changed=False,
        component_name=DEFAULT_COMPONENT,
    )
    return register_prompt_body(
        body=seed_body,
        provenance=provenance,
        name=name,
        catalog=catalog,
        schema=schema,
        commit_message=commit_message or "Register token-efficiency seed skill (baseline)",
        alias=alias,
        client=client,
        profile=profile,
    )


def register_gepa_candidate(
    candidate_json_path: str | Path,
    *,
    name: str = DEFAULT_PROMPT_NAME,
    catalog: str = DEFAULT_CATALOG,
    schema: str = DEFAULT_SCHEMA,
    alias: str | None = None,
    force: bool = False,
    commit_message: str | None = None,
    client: PromptRegistryClient | None = None,
    profile: str | None = None,
) -> RegisteredPrompt:
    """Register the evolved skill body from a ``gepa_candidate*.json`` (human promote).

    Reads the human-gated candidate artifact, **refuses** to register it unless it
    improved over seed on the held-out split (see :func:`candidate_improvement`),
    and otherwise registers its ``evolved_skill_body`` as a new
    ``source=gepa-evolved`` version stamped with the GEPA run's provenance: the
    suite content hash, ``changed``, ``gepa_best_val_score``, the held-out PROMOTE
    counts and savings for both arms, and a pointer back to the candidate artifact.

    Args:
        candidate_json_path: Path to the ``GepaOptimizationResult`` JSON GEPA wrote.
        name / catalog / schema: Target UC prompt name (see :func:`register_prompt_body`).
        alias: Optional alias to set on the new version (no implicit promotion).
        force: Register even if the candidate did not beat seed. The refusal reason
            is then recorded on the version (``ail.prompt.forced=true`` +
            ``ail.prompt.registration_reason``) instead of raising — a forced version
            never masquerades as an improvement.
        commit_message: Optional override; a provenance summary is generated otherwise.
        client: Injectable prompt-registry client; a live one is built when ``None``.
        profile: Databricks CLI profile, used only when building a live client.

    Raises:
        NonImprovingCandidateError: if the candidate did not beat seed and ``force``
            is ``False``.
    """
    path = Path(candidate_json_path)
    result = GepaOptimizationResult.model_validate_json(path.read_text(encoding="utf-8"))

    improving, reason = candidate_improvement(result)
    if not improving and not force:
        raise NonImprovingCandidateError(reason)
    forced = not improving and force

    evolved = result.holdout_evolved
    seed = result.holdout_seed_baseline
    provenance = PromptProvenance(
        source=PromptSource.GEPA_EVOLVED,
        suite_version=result.suite_version or None,
        suite_content_hash=result.suite_content_hash or None,
        changed=result.changed,
        gepa_best_val_score=result.gepa_best_val_score,
        gepa_num_candidates=result.gepa_num_candidates,
        component_name=result.component_name,
        reflection_lm=result.reflection_lm,
        holdout_evolved_promote=_promote_ratio(evolved),
        holdout_seed_promote=_promote_ratio(seed),
        holdout_evolved_savings_pct=(evolved.realized_token_savings_pct if evolved else None),
        holdout_seed_savings_pct=(seed.realized_token_savings_pct if seed else None),
        holdout_savings_delta_pct=result.holdout_savings_delta_pct,
        candidate_artifact=str(path),
        improving=improving,
        registration_reason=reason,
        forced=forced,
    )

    forced_prefix = "FORCE-registered non-improving GEPA candidate"
    if commit_message is None:
        prefix = forced_prefix if forced else "Promote GEPA candidate"
        commit_message = f"{prefix}: {reason}"
    elif forced:
        # A forced (non-improving) registration must NEVER record a clean message: prepend
        # the warning (and reason) even to a caller-supplied message, keeping their text
        # after it, so a forced version can't masquerade as genuine in the audit log.
        commit_message = f"{forced_prefix}: {reason}\n\n{commit_message}"

    return register_prompt_body(
        body=result.evolved_skill_body,
        provenance=provenance,
        name=name,
        catalog=catalog,
        schema=schema,
        commit_message=commit_message,
        alias=alias,
        client=client,
        profile=profile,
    )


def search_registered_prompts(
    *,
    catalog: str = DEFAULT_CATALOG,
    schema: str = DEFAULT_SCHEMA,
    client: PromptRegistryClient | None = None,
    profile: str | None = None,
) -> list[Any]:
    """List prompts registered under ``catalog.schema`` (a read convenience).

    Mirrors the verified filter form
    ``search_prompts(filter_string="catalog = '...' AND schema = '...'")``; useful to
    confirm a registration landed.
    """
    registry = client if client is not None else _new_prompt_client(profile)
    filter_string = f"catalog = '{catalog}' AND schema = '{schema}'"
    return list(registry.search_prompts(filter_string))


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _render_tag(value: object | None) -> str | None:
    """Stringify a tag value (MLflow tags are strings); ``None`` means omit the tag."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _promote_ratio(artifact: Any) -> str | None:
    """Render a Phase-2 artifact's PROMOTE/total ratio, e.g. ``"2/3"``."""
    if artifact is None:
        return None
    return f"{artifact.n_promote}/{artifact.n_tasks}"


def _new_prompt_client(profile: str | None) -> PromptRegistryClient:
    """Build a live prompt-registry client against the configured Databricks workspace.

    Built **only** when a human invokes a register/search with no injected ``client``
    — this is the one place a live MLflow call originates. Mirrors the repo's
    ``ail.ingest.mlflow_source`` configuration (lazy import; set tracking +
    ``databricks-uc`` registry URIs; resolve the workspace host from the CLI profile).
    """
    try:
        import mlflow
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "the prompt registry requires mlflow. Install it with: pip install 'mlflow>=3.14,<4'"
        ) from exc

    from ail.ingest.mlflow_source import MLflowTraceSource

    if profile:
        os.environ.setdefault("DATABRICKS_CONFIG_PROFILE", profile)
    MLflowTraceSource._resolve_workspace_host()
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(REGISTRY_URI)
    return _GenAIPromptRegistryClient()


class _GenAIPromptRegistryClient:
    """Default :class:`PromptRegistryClient` delegating to the public ``mlflow.genai`` API."""

    def register_prompt(
        self,
        name: str,
        template: str,
        commit_message: str | None,
        tags: dict[str, str] | None,
    ) -> Any:
        import mlflow.genai

        return mlflow.genai.register_prompt(
            name=name, template=template, commit_message=commit_message, tags=tags
        )

    def set_prompt_alias(self, name: str, alias: str, version: int) -> None:
        import mlflow.genai

        mlflow.genai.set_prompt_alias(name, alias, version)

    def search_prompts(self, filter_string: str | None) -> list[Any]:
        import mlflow.genai

        return list(mlflow.genai.search_prompts(filter_string=filter_string))

    def load_prompt(self, name_or_uri: str) -> Any:
        import mlflow.genai

        return mlflow.genai.load_prompt(name_or_uri)
