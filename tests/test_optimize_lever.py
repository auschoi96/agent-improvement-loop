"""Tests for the Phase-2 token-efficiency lever (:mod:`ail.optimize.lever` + assets).

Offline: loads the shipped ``SKILL.md`` asset, checks the parsed identity/body,
and asserts the :class:`~ail.optimize.lever.SkillInjectionIntervention` injects the
skill into a candidate task's system prompt **and nothing else** (so a
baseline-vs-candidate comparison isolates the skill's effect). The malformed-asset
guards are exercised against temp files so a broken asset fails loudly instead of
loading silently.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ail.ingest.base import AgentTask
from ail.optimize import (
    BASELINE,
    CANDIDATE,
    SkillInjectionIntervention,
    load_skill_asset,
    token_efficiency_skill,
)
from ail.optimize.assets import SkillAsset, _split_frontmatter, skill_asset_path
from ail.optimize.assets import load_skill_asset as _load_by_slug


class TestSkillAsset:
    def test_loads_token_efficiency_skill(self) -> None:
        asset = load_skill_asset()
        assert asset.slug == "token-efficient-execution"
        assert asset.name == "token-efficient-execution"
        assert asset.description  # non-empty, whitespace-collapsed
        assert "\n" not in asset.description
        # The body carries the behavioural rules the lever is about.
        body = asset.body.lower()
        assert "re-read" in body or "re-read a file" in body
        assert "batch" in body
        assert "boilerplate" in body or "cd" in body
        assert asset.source_path is not None and asset.source_path.endswith("SKILL.md")

    def test_skill_file_exists_under_generated_assets_location(self) -> None:
        # The asset must live in the framework's generated-assets package, never a
        # user skills dir.
        path = skill_asset_path()
        assert path.is_file()
        parts = path.parts
        assert "optimize" in parts and "assets" in parts and "skills" in parts
        assert ".claude" not in parts and "polly" not in parts

    def test_as_system_prompt_section_wraps_body(self) -> None:
        asset = load_skill_asset()
        section = asset.as_system_prompt_section()
        assert section.startswith('<skill name="token-efficient-execution">')
        assert section.rstrip().endswith("</skill>")
        assert asset.body.strip() in section

    def test_missing_asset_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            _load_by_slug("no-such-skill-xyz")

    def test_malformed_asset_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # A front-matter block with no name, or an empty body, is malformed.
        no_name = tmp_path / "SKILL.md"
        no_name.write_text("---\ndescription: x\n---\nbody here\n", encoding="utf-8")
        monkeypatch.setattr("ail.optimize.assets.skill_asset_path", lambda slug="x": no_name)
        with pytest.raises(ValueError, match="no 'name'"):
            _load_by_slug("x")

        empty_body = tmp_path / "SKILL2.md"
        empty_body.write_text("---\nname: x\n---\n   \n", encoding="utf-8")
        monkeypatch.setattr("ail.optimize.assets.skill_asset_path", lambda slug="x": empty_body)
        with pytest.raises(ValueError, match="empty body"):
            _load_by_slug("x")

    def test_split_frontmatter_handles_no_frontmatter(self) -> None:
        meta, body = _split_frontmatter("just a body, no fence")
        assert meta == {}
        assert body == "just a body, no fence"


class TestInterventionIsolation:
    def test_injects_skill_into_empty_system_prompt(self) -> None:
        task = AgentTask(prompt="implement X")
        out = CANDIDATE.intervention.apply(task)
        assert CANDIDATE.intervention is not None
        assert out is not task  # pure: a new task
        assert out.system_prompt is not None and "<skill" in out.system_prompt
        # Everything else is identical — the skill is the only controlled difference.
        assert out.prompt == task.prompt
        assert out.allowed_tools == task.allowed_tools
        assert out.model == task.model
        assert out.params == task.params

    def test_appends_to_existing_system_prompt(self) -> None:
        task = AgentTask(prompt="implement X", system_prompt="You are a careful engineer.")
        out = CANDIDATE.intervention.apply(task)
        assert out.system_prompt is not None
        assert out.system_prompt.startswith("You are a careful engineer.")
        assert "<skill" in out.system_prompt

    def test_does_not_mutate_input_task(self) -> None:
        task = AgentTask(prompt="implement X")
        CANDIDATE.intervention.apply(task)
        assert task.system_prompt is None  # untouched

    def test_intervention_is_named_and_pure(self) -> None:
        asset = SkillAsset(slug="s", name="s", description="d", body="be efficient", raw="raw")
        iv = SkillInjectionIntervention(name="my-skill", skill=asset)
        assert iv.name == "my-skill"
        out = iv.apply(AgentTask(prompt="p"))
        assert "be efficient" in (out.system_prompt or "")


class TestLeverConfigs:
    def test_baseline_has_no_asset(self) -> None:
        assert BASELINE.intervention is None
        assert BASELINE.asset_enabled is False

    def test_candidate_enables_the_skill(self) -> None:
        assert CANDIDATE.intervention is not None
        assert CANDIDATE.asset_enabled is True
        assert CANDIDATE.intervention.skill is token_efficiency_skill()

    def test_skill_is_cached(self) -> None:
        assert token_efficiency_skill() is token_efficiency_skill()
