"""Judge-vs-human agreement on the Human Anchor — a first-class metric.

This is the anti-co-adaptation safeguard of the frozen evaluation wall
(``docs/ARCHITECTURE.md`` §2). A judge is only trustworthy while it still agrees
with people, so agreement against a small human-labeled slice is measured as a
first-class output with a **configurable floor**: when agreement drops below the
floor the judge is flagged ``distrusted`` and the loop must stop trusting its
scores until it is re-aligned and re-measured.

Crucially, this runs on the **Human Anchor** pool only — never the Alignment
Set (which aligns the judge) and never the Task Suite (which compares agents).
Measuring agreement on the same labels the judge was aligned against would just
report how well alignment memorized them; the anchor is held out for exactly
this reason. Alignment (:mod:`ail.judges.alignment`) and agreement run on their
own cadences, decoupled from agent optimization.

Two entry points:

* :func:`compute_agreement` — pure function over ``(judge_value, human_value)``
  pairs. No MLflow, no model: fully offline and unit-testable.
* :func:`score_anchor` — runs a judge over a :class:`~ail.judges.pools.HumanAnchor`
  slice, then delegates to :func:`compute_agreement`. The judge calls are the
  only model-touching part (mock them offline; gate live with
  ``@pytest.mark.live``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ail.judges.contract import AgreementItem, AgreementReport
from ail.judges.pools import AnchorItem, HumanAnchor, ScoreValue

if TYPE_CHECKING:
    from mlflow.genai.judges import Judge

__all__ = [
    "AgreementConfig",
    "ScorePair",
    "coerce_score",
    "compute_agreement",
    "score_anchor",
    "log_agreement",
]

#: Default minimum acceptable judge-vs-human agreement. A deployer tunes this
#: per judge and per risk tolerance; it is deliberately a knob, not a constant,
#: because the right floor for a ship/no-ship guardrail differs from the floor
#: for a monitoring dashboard.
DEFAULT_FLOOR = 0.7


#: Default minimum number of *scored* anchor items below which the measurement
#: is treated as insufficient (and the judge as distrusted). The floor is 1 —
#: zero scored items is an unmeasured judge, which must never read as trusted. A
#: real ship/no-ship guardrail should raise this so a 1–2 item anchor cannot
#: certify a judge; it is deliberately a knob, not a constant.
DEFAULT_MIN_SAMPLES = 1


@dataclass(frozen=True, slots=True)
class AgreementConfig:
    """Knobs for the agreement computation.

    Args:
        floor: Minimum acceptable :attr:`AgreementReport.agreement_rate`. At or
            above it the judge is trusted; below it ``distrusted`` fires.
        numeric_tolerance: For **float** labels, the absolute difference within
            which a judge score and a human label are deemed to agree. Ignored
            for categorical/bool/int labels, which require exact equality.
        case_insensitive: Compare string labels case-insensitively (so a judge
            emitting ``"Yes"`` agrees with a human ``"yes"``). Applied uniformly:
            the agreement decision **and** the Cohen's-kappa discretization /
            label space honour this flag, so kappa never silently case-folds
            labels a deployer asked to keep distinct.
        min_samples: Minimum number of *scored* items required to trust a
            measurement. Below it (an empty anchor, or a judge that scored too
            few items) the report is flagged ``insufficient_data`` and
            ``distrusted`` — an unmeasured judge is never trusted.
    """

    floor: float = DEFAULT_FLOOR
    numeric_tolerance: float = 0.0
    case_insensitive: bool = True
    min_samples: int = DEFAULT_MIN_SAMPLES


@dataclass(frozen=True, slots=True)
class ScorePair:
    """One judge score beside its human gold label, for agreement scoring."""

    item_id: str
    human_value: ScoreValue
    judge_value: ScoreValue | None = None
    error: str | None = None


#: Sentinel for "this object has no ``.value`` attribute" in :func:`coerce_score`.
_MISSING: Any = object()


def coerce_score(value: Any) -> ScoreValue | None:
    """Reduce whatever a judge returns to a comparable :data:`ScoreValue`.

    A judge's ``__call__`` may return a raw scalar, an MLflow ``Feedback`` (whose
    ``.value`` carries the score), a ``CategoricalRating`` / other ``Enum``, or a
    one-element list of feedbacks. This normalizes all of those to a bare
    ``bool``/``int``/``float``/``str`` so agreement compares like with like.
    ``None`` (a judge that produced no value) passes through as ``None``.
    """
    if value is None:
        return None
    # Unwrap an MLflow Feedback (duck-typed: it exposes a ``.value``). Guard
    # against plain objects by only unwrapping when ``value`` itself is absent
    # of being a primitive.
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return coerce_score(value[0])
        raise ValueError(f"cannot coerce a multi-element judge result to a score: {value!r}")
    feedback_value = getattr(value, "value", _MISSING)
    if feedback_value is not _MISSING:
        return coerce_score(feedback_value)
    # Bare enums without a meaningful ``.value`` fall back to their string form.
    return str(value)


def _values_agree(
    judge_value: ScoreValue, human_value: ScoreValue, config: AgreementConfig
) -> bool:
    """Whether one judge value agrees with one human label under ``config``."""
    # Float labels: agree within tolerance (only when neither is a bool, since
    # bool is an int subclass and a yes/no guardrail must match exactly).
    numeric = (isinstance(judge_value, (int, float)) and not isinstance(judge_value, bool)) and (
        isinstance(human_value, (int, float)) and not isinstance(human_value, bool)
    )
    if numeric and (isinstance(judge_value, float) or isinstance(human_value, float)):
        return abs(float(judge_value) - float(human_value)) <= config.numeric_tolerance
    if config.case_insensitive and isinstance(judge_value, str) and isinstance(human_value, str):
        return judge_value.strip().casefold() == human_value.strip().casefold()
    return judge_value == human_value


def _cohen_kappa(
    pairs: Sequence[tuple[ScoreValue, ScoreValue]], *, case_insensitive: bool
) -> float | None:
    """Cohen's kappa over discrete (judge, human) label pairs, or ``None``.

    Chance-corrected agreement: ``(p_o - p_e) / (1 - p_e)``. Returns ``None`` when
    it is undefined or uninformative — no pairs, or perfect expected agreement
    (a single label used by both raters), where the raw rate is the honest
    number to report. Labels are discretized with :func:`_kappa_key` under the
    same ``case_insensitive`` rule used for the agreement decision, so kappa and
    the raw rate measure like with like.
    """
    n = len(pairs)
    if n == 0:
        return None

    def key(value: ScoreValue) -> str:
        return _kappa_key(value, case_insensitive=case_insensitive)

    labels = sorted({key(j) for j, _ in pairs} | {key(h) for _, h in pairs})
    if len(labels) < 2:
        return None
    index = {label: i for i, label in enumerate(labels)}
    observed = sum(1 for j, h in pairs if key(j) == key(h)) / n
    judge_counts = [0.0] * len(labels)
    human_counts = [0.0] * len(labels)
    for j, h in pairs:
        judge_counts[index[key(j)]] += 1
        human_counts[index[key(h)]] += 1
    expected = sum((judge_counts[i] / n) * (human_counts[i] / n) for i in range(len(labels)))
    if expected >= 1.0:
        return None
    return round((observed - expected) / (1.0 - expected), 6)


def _kappa_key(value: ScoreValue, *, case_insensitive: bool) -> str:
    """Discretize a label for kappa / label-space (string form).

    Case-folds only when ``case_insensitive`` is set, matching
    :func:`_values_agree`; with it off, ``"Yes"`` and ``"yes"`` stay distinct
    labels rather than being silently merged.
    """
    text = str(value).strip()
    return text.casefold() if case_insensitive else text


def compute_agreement(
    pairs: Sequence[ScorePair],
    *,
    judge_name: str,
    config: AgreementConfig | None = None,
    generated_at: str | None = None,
) -> AgreementReport:
    """Compute judge-vs-human agreement over precomputed score pairs.

    Pure: no model, no MLflow. Every item contributes to :attr:`n_items`; an item
    whose judge value is missing (``error`` set, or ``judge_value is None``)
    counts as a non-agreement (a judge that cannot score has not agreed) and is
    reflected in :attr:`n_scored`.

    Cohen's kappa is reported for categorical comparisons (no float tolerance in
    play); the **floor is applied to the raw agreement rate**, and
    :attr:`AgreementReport.distrusted` fires when the rate is below it.

    Fail-closed on insufficient data: with fewer than ``config.min_samples``
    *scored* items (an empty anchor is the limiting case), the judge is
    unmeasured. :attr:`AgreementReport.insufficient_data` is set and
    :attr:`AgreementReport.distrusted` fires regardless of the rate — an
    unmeasured judge must never read as trusted.
    """
    cfg = config or AgreementConfig()
    items: list[AgreementItem] = []
    scored_pairs: list[tuple[ScoreValue, ScoreValue]] = []
    used_tolerance = False

    for pair in pairs:
        if pair.error is not None or pair.judge_value is None:
            items.append(
                AgreementItem(
                    item_id=pair.item_id,
                    human_value=pair.human_value,
                    judge_value=pair.judge_value,
                    agree=False,
                    error=pair.error or "judge produced no value",
                )
            )
            continue
        agree = _values_agree(pair.judge_value, pair.human_value, cfg)
        if _is_float_compare(pair.judge_value, pair.human_value):
            used_tolerance = True
        scored_pairs.append((pair.judge_value, pair.human_value))
        items.append(
            AgreementItem(
                item_id=pair.item_id,
                human_value=pair.human_value,
                judge_value=pair.judge_value,
                agree=agree,
            )
        )

    n_items = len(items)
    n_scored = len(scored_pairs)
    n_agreements = sum(1 for item in items if item.agree)
    # The rate is over ALL items (an unscored item is a non-agreement), so a
    # judge that errors on half the anchor cannot look fully trustworthy.
    rate = round(n_agreements / n_items, 6) if n_items else 0.0
    # Fail closed: too few scored items (an empty anchor is the limiting case)
    # means the judge is unmeasured, and an unmeasured judge is never trusted.
    insufficient_data = n_scored < cfg.min_samples
    distrusted = insufficient_data or rate < cfg.floor

    kappa = (
        None
        if used_tolerance
        else _cohen_kappa(scored_pairs, case_insensitive=cfg.case_insensitive)
    )
    label_space = sorted(
        {_kappa_key(p.human_value, case_insensitive=cfg.case_insensitive) for p in pairs}
    )

    notes: list[str] = []
    if insufficient_data:
        if n_items == 0:
            notes.append(
                "empty Human Anchor: judge is unmeasured; flagged distrusted (fail closed)"
            )
        else:
            notes.append(
                f"only {n_scored} scored item(s) < min_samples {cfg.min_samples}: judge is "
                "under-measured; flagged distrusted (fail closed)"
            )
    n_errored = sum(1 for item in items if item.error is not None)
    if n_errored:
        notes.append(f"{n_errored} item(s) had no judge value and count as non-agreements")
    if used_tolerance:
        notes.append(
            f"float labels compared within tolerance {cfg.numeric_tolerance}; "
            "Cohen's kappa omitted (defined for categorical agreement)"
        )

    return AgreementReport(
        judge_name=judge_name,
        n_items=n_items,
        n_scored=n_scored,
        n_agreements=n_agreements,
        agreement_rate=rate,
        floor=cfg.floor,
        distrusted=distrusted,
        insufficient_data=insufficient_data,
        cohen_kappa=kappa,
        numeric_tolerance=cfg.numeric_tolerance if used_tolerance else None,
        label_space=label_space,
        items=items,
        generated_at=generated_at,
        notes=notes,
    )


def _is_float_compare(a: ScoreValue, b: ScoreValue) -> bool:
    numeric = (isinstance(a, (int, float)) and not isinstance(a, bool)) and (
        isinstance(b, (int, float)) and not isinstance(b, bool)
    )
    return numeric and (isinstance(a, float) or isinstance(b, float))


def score_anchor(
    judge: Judge,
    anchor: HumanAnchor,
    *,
    config: AgreementConfig | None = None,
    generated_at: str | None = None,
) -> AgreementReport:
    """Score a judge against a Human-Anchor slice and report agreement.

    For each :class:`~ail.judges.pools.AnchorItem`, calls
    ``judge(inputs=, outputs=, expectations=)``, normalizes the result with
    :func:`coerce_score`, pairs it with the item's ``human_label``, and delegates
    to :func:`compute_agreement`. A judge call that raises is captured per-item
    (recorded as an error, counted as a non-agreement) so one bad item never
    aborts the whole measurement.

    This is the only model-touching function in the module. Offline tests pass a
    mock judge; a live measurement is gated behind ``@pytest.mark.live``.
    """
    pairs = [_score_one(judge, item) for item in anchor.items]
    return compute_agreement(
        pairs,
        judge_name=getattr(judge, "name", "judge"),
        config=config,
        generated_at=generated_at or datetime.now(UTC).isoformat(),
    )


def _score_one(judge: Judge, item: AnchorItem) -> ScorePair:
    try:
        result = judge(
            inputs=item.inputs,
            outputs=item.outputs,
            expectations=item.expectations,
        )
        judge_value = coerce_score(result)
    except Exception as exc:  # noqa: BLE001 - one bad item must not abort the slice
        return ScorePair(item_id=item.item_id, human_value=item.human_label, error=str(exc))
    return ScorePair(
        item_id=item.item_id,
        human_value=item.human_label,
        judge_value=judge_value,
        error=None if judge_value is not None else "judge produced no value",
    )


def log_agreement(report: AgreementReport, *, run_id: str | None = None) -> bool:
    """Log an agreement report to MLflow if a tracking context is available.

    Logs ``judge_human_agreement`` and ``judge_distrusted`` as metrics (so the
    drift trend and the floor breach are queryable) and the full report as a JSON
    artifact. Best-effort: if MLflow is unavailable or there is no active/known
    run, it does nothing and returns ``False`` rather than raising — logging is a
    side effect of measurement, never a precondition for it.

    Returns:
        ``True`` if the report was logged, ``False`` otherwise.
    """
    try:
        import mlflow
    except ImportError:  # pragma: no cover - mlflow is a hard dep, guard anyway
        return False
    try:
        kwargs = {"run_id": run_id} if run_id else {}
        mlflow.log_metric("judge_human_agreement", report.agreement_rate, **kwargs)
        mlflow.log_metric("judge_distrusted", 1.0 if report.distrusted else 0.0, **kwargs)
        mlflow.log_metric(
            "judge_insufficient_data", 1.0 if report.insufficient_data else 0.0, **kwargs
        )
        if report.cohen_kappa is not None:
            mlflow.log_metric("judge_human_cohen_kappa", report.cohen_kappa, **kwargs)
        mlflow.log_dict(
            report.model_dump(),
            f"judge_agreement/{report.judge_name}.json",
            **kwargs,
        )
    except Exception:  # noqa: BLE001 - no active run / offline: logging is optional
        return False
    return True
