"""Framework-**generated** assets *for the target agent* (skills, tools, …).

This is the ``optimize/assets/`` location of the loop (``docs/ARCHITECTURE.md``
§4/§6): the place the framework writes the helper assets it proposes for the
agent under improvement. The first asset is a behavioral **skill**,
``skills/token-efficient-execution/SKILL.md`` — the token-efficiency lever for
Phase 2.

These assets are authored as drop-in Claude Code skills (a ``SKILL.md`` with
``name``/``description`` YAML front-matter), so they are usable verbatim wherever
a Claude Code skill is loaded. The :class:`SkillAsset` model and
:func:`load_skill_asset` read one off disk and expose its body in the two forms
the loop needs: as the original ``SKILL.md`` text, and as a system-prompt section
the :class:`~ail.optimize.lever.SkillInjectionIntervention` injects into a
candidate run (the Claude Code adapter disables ambient skill discovery —
``setting_sources=[]`` — and injects context explicitly, so a comparison can pin
the candidate's behaviour to exactly this asset).

This is **not** a user skill: it is never written to a ``polly`` skills directory
or to ``~/.claude/skills``. It lives in the framework package as a generated
artifact and is loaded by path.

Stage 6 adds the **generator** side of this package: an extensible
:class:`~ail.optimize.assets.base.AssetGenerator` seam + registry
(:mod:`ail.optimize.assets.base`) that turns L3/RLM ranked recommendations into
concrete assets, with the ``metric_view`` generator
(:mod:`ail.optimize.assets.metric_view`) implemented end-to-end and the other
asset types raising a clear ``next`` signal
(:class:`~ail.optimize.assets.base.AssetGeneratorNotImplemented`).
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import yaml

from ail.optimize.assets.asset_contract import (
    DroppedMeasure,
    GeneratedAsset,
    GeneratedMetricView,
    MetricViewDimension,
    MetricViewMeasure,
    MetricViewSpec,
)
from ail.optimize.assets.base import (
    AssetGenerator,
    AssetGeneratorNotImplemented,
    generate_asset,
    get_generator,
    register,
    registered_asset_types,
)
from ail.optimize.assets.l0_contract import (
    L0_CONTRACT,
    L0ColumnContract,
    verify_against_publish,
)
from ail.optimize.assets.metric_view import (
    GENERATOR_VERSION,
    MEASURE_CATALOG,
    MetricViewGenerator,
    SpecValidationError,
    generate_metric_view,
    generate_metric_views_from_report,
    validate_spec,
)

__all__ = [
    # pre-authored skill assets (existing)
    "SkillAsset",
    "load_skill_asset",
    "skill_asset_path",
    "TOKEN_EFFICIENCY_SKILL",
    # generator seam + registry
    "AssetGenerator",
    "AssetGeneratorNotImplemented",
    "register",
    "get_generator",
    "registered_asset_types",
    "generate_asset",
    # metric-view generator (implemented end-to-end)
    "MetricViewGenerator",
    "generate_metric_view",
    "generate_metric_views_from_report",
    "validate_spec",
    "SpecValidationError",
    "MEASURE_CATALOG",
    "GENERATOR_VERSION",
    # typed asset output contract
    "GeneratedAsset",
    "GeneratedMetricView",
    "MetricViewSpec",
    "MetricViewDimension",
    "MetricViewMeasure",
    "DroppedMeasure",
    # real L0 column contract the generator builds on
    "L0_CONTRACT",
    "L0ColumnContract",
    "verify_against_publish",
]

#: Slug of the Phase-2 token-efficiency skill (its directory under ``skills/``).
TOKEN_EFFICIENCY_SKILL = "token-efficient-execution"

_SKILL_FILENAME = "SKILL.md"
_FRONTMATTER_FENCE = "---"


@dataclass(frozen=True, slots=True)
class SkillAsset:
    """A parsed ``SKILL.md`` asset: front-matter identity plus markdown body.

    ``name`` and ``description`` come from the YAML front-matter (the Claude Code
    skill contract). ``body`` is the markdown after the front-matter — the
    instructional content. ``raw`` is the verbatim file text (front-matter
    included) for callers that want the original artifact. ``source_path`` records
    where it was loaded from for provenance.
    """

    slug: str
    name: str
    description: str
    body: str
    raw: str
    source_path: str | None = None

    def as_system_prompt_section(self) -> str:
        """Render the skill as a system-prompt section the agent must follow.

        Used by :class:`~ail.optimize.lever.SkillInjectionIntervention` to inject
        the skill into a candidate run when ambient skill discovery is disabled.
        The body is wrapped in a clear, stable marker so the instruction is
        unambiguous and the injection is auditable in a captured trace.
        """
        return (
            f'<skill name="{self.name}">\n'
            "The following are standing operating instructions you must follow for "
            "this task.\n\n"
            f"{self.body.strip()}\n"
            "</skill>"
        )


def skill_asset_path(slug: str = TOKEN_EFFICIENCY_SKILL) -> Path:
    """Filesystem path to a skill's ``SKILL.md`` (``skills/<slug>/SKILL.md``)."""
    resource = files(__package__).joinpath("skills", slug, _SKILL_FILENAME)
    return Path(str(resource))


def _split_frontmatter(text: str) -> tuple[dict[str, object], str]:
    """Split ``---`` YAML front-matter from the markdown body.

    Returns ``({}, text)`` when there is no leading front-matter block so a
    body-only file still loads (just without a parsed name/description).
    """
    if not text.startswith(_FRONTMATTER_FENCE):
        return {}, text
    parts = text.split(_FRONTMATTER_FENCE, 2)
    if len(parts) < 3:
        return {}, text
    meta = yaml.safe_load(parts[1]) or {}
    if not isinstance(meta, dict):
        meta = {}
    return meta, parts[2].lstrip("\n")


def load_skill_asset(slug: str = TOKEN_EFFICIENCY_SKILL) -> SkillAsset:
    """Load and parse the ``SKILL.md`` for ``slug``.

    Raises:
        FileNotFoundError: if the asset does not exist.
        ValueError: if the front-matter is missing a ``name`` or the body is
            empty — an asset with no name/body is malformed, not silently usable.
    """
    path = skill_asset_path(slug)
    if not path.is_file():
        raise FileNotFoundError(f"no skill asset at {path}")
    raw = path.read_text(encoding="utf-8")
    meta, body = _split_frontmatter(raw)
    name = str(meta.get("name") or "").strip()
    description = " ".join(str(meta.get("description") or "").split())
    if not name:
        raise ValueError(f"skill asset {path} has no 'name' in its front-matter")
    if not body.strip():
        raise ValueError(f"skill asset {path} has an empty body")
    return SkillAsset(
        slug=slug,
        name=name,
        description=description,
        body=body,
        raw=raw,
        source_path=str(path),
    )
