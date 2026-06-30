"""The extensible asset-generator seam: an interface, a registry, and dispatch.

Stage 6 of the loop turns the L3/RLM ranked recommendations
(:class:`ail.l3.contract.RankedAsset`) into concrete, deployable assets. This is
the seam those generators plug into:

* :class:`AssetGenerator` — the interface one generator-per-``asset_type``
  implements (``asset_type`` class attribute + :meth:`~AssetGenerator.generate`).
* a registry keyed by ``asset_type`` (:func:`register`, :func:`get_generator`),
  and :func:`generate_asset`, which dispatches a ranked recommendation to its
  generator.
* :class:`AssetGeneratorNotImplemented` — the explicit ``next`` signal raised for
  an asset type that is recognised but not yet built. Only ``metric_view`` is
  implemented end-to-end in this stage; ``skill`` / ``tool`` / ``prompt_change``
  (and any other L3 :data:`~ail.l3.contract.AssetType`) raise this rather than
  half-emitting an asset.

Importing this module registers the metric-view generator and the ``next`` stubs
(see the bottom of the file), so :func:`generate_asset` works out of the box.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ail.l3.contract import RankedAsset
from ail.optimize.assets.asset_contract import GeneratedAsset

__all__ = [
    "AssetGenerator",
    "AssetGeneratorNotImplemented",
    "register",
    "get_generator",
    "registered_asset_types",
    "generate_asset",
]


class AssetGeneratorNotImplemented(NotImplementedError):
    """Raised when an asset type is recognised but its generator is not built yet.

    The clear ``next`` signal the orchestrator keys off: ``asset_type`` names the
    type that was requested and ``status`` is always ``"next"`` so a caller can
    branch on "this is a planned-but-unbuilt generator" without string-matching the
    message.
    """

    status = "next"

    def __init__(self, asset_type: str, detail: str = "") -> None:
        self.asset_type = asset_type
        msg = f"asset generator for {asset_type!r} is not implemented yet (next milestone)"
        if detail:
            msg = f"{msg}: {detail}"
        super().__init__(msg)


class AssetGenerator(ABC):
    """Generate a concrete asset from one ranked L3 recommendation.

    A generator handles exactly one :attr:`asset_type`. :meth:`generate` consumes
    the :class:`~ail.l3.contract.RankedAsset` (the aggregated, recurrence-ranked
    recommendation) and returns a typed :class:`~ail.optimize.assets.asset_contract.GeneratedAsset`.
    """

    #: The L3 ``asset_type`` this generator handles (registry key).
    asset_type: str

    @abstractmethod
    def generate(self, asset: RankedAsset, **options: Any) -> GeneratedAsset:
        """Build the asset for ``asset``; ``options`` are generator-specific."""
        raise NotImplementedError


class _NextAssetGenerator(AssetGenerator):
    """A registered placeholder for a recognised-but-unbuilt asset type.

    Registering these (rather than leaving the type unregistered) makes the
    roadmap explicit: dispatching ``skill`` / ``tool`` / ``prompt_change`` raises a
    clear :class:`AssetGeneratorNotImplemented` naming the type, never a generic
    ``KeyError`` and never a half-formed asset.
    """

    def __init__(self, asset_type: str, detail: str) -> None:
        self.asset_type = asset_type
        self._detail = detail

    def generate(self, asset: RankedAsset, **options: Any) -> GeneratedAsset:
        raise AssetGeneratorNotImplemented(self.asset_type, self._detail)


_REGISTRY: dict[str, AssetGenerator] = {}


def register(generator: AssetGenerator) -> AssetGenerator:
    """Register ``generator`` under its ``asset_type`` (replacing any prior one)."""
    if not getattr(generator, "asset_type", ""):
        raise ValueError("generator must set a non-empty asset_type")
    _REGISTRY[generator.asset_type] = generator
    return generator


def get_generator(asset_type: str) -> AssetGenerator:
    """The generator for ``asset_type``.

    Raises :class:`AssetGeneratorNotImplemented` for an unregistered type so an
    unknown/unbuilt asset type fails with the same explicit ``next`` signal as a
    placeholder stub.
    """
    generator = _REGISTRY.get(asset_type)
    if generator is None:
        raise AssetGeneratorNotImplemented(asset_type)
    return generator


def registered_asset_types() -> tuple[str, ...]:
    """The asset types that have a registered generator (placeholders included)."""
    return tuple(_REGISTRY)


def generate_asset(asset: RankedAsset, **options: Any) -> GeneratedAsset:
    """Dispatch ``asset`` to the generator for its ``asset_type`` and run it."""
    return get_generator(asset.asset_type).generate(asset, **options)


def _register_builtins() -> None:
    """Register the metric-view generator and the ``next`` stubs (idempotent)."""
    # Imported here (not at module top) to avoid an import cycle: metric_view
    # imports this module for AssetGenerator/register.
    from ail.optimize.assets.metric_view import MetricViewGenerator

    register(MetricViewGenerator())
    register(
        _NextAssetGenerator(
            "skill",
            "synthesising a SKILL.md behavioural lever from a recommendation is the "
            "next asset type",
        )
    )
    register(
        _NextAssetGenerator(
            "tool", "emitting an agent tool / UC function definition is the next asset type"
        )
    )
    register(
        _NextAssetGenerator(
            "prompt_change",
            "emitting a system-prompt / instruction diff is the next asset type",
        )
    )


_register_builtins()
