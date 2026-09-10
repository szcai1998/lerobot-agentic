"""Policy abstraction contract. [TR 7.1]

The recovery system must not need to know whether the base policy is ACT,
SmolVLA, MolmoAct2, or pi0.5 beyond the capabilities the adapter explicitly
declares.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from lerobot_reliability.data_types import (
    CanonicalActionView,
    CanonicalObservation,
    PolicyCapabilities,
    PolicyManifest,
    PolicyOutput,
)

__all__ = [
    "PolicyAdapter",
    "PolicyCapabilities",
    "PolicyManifest",
    "PolicyOutput",
]


@runtime_checkable
class PolicyAdapter(Protocol):
    """Uniform interface over a heterogeneous robot policy."""

    def reset(self) -> None:
        """Clear per-episode policy state (action queue, temporal ensemble, ...)."""
        ...

    def select_action(self, obs: CanonicalObservation) -> PolicyOutput:
        """Produce the next executable action (and chunk/features when available)."""
        ...

    def reset_or_truncate(self) -> None:
        """Invalidate any buffered action chunk so the next call re-infers.

        Only meaningful when ``capabilities().supports_queue_reset`` is True.
        """
        ...

    def native_action_to_canonical_view(self, action: object) -> CanonicalActionView | None:
        """Project a native action into the analysis view, or ``None`` if unavailable."""
        ...

    def capabilities(self) -> PolicyCapabilities:
        """Declare what this adapter supports. Callers must not assume beyond this."""
        ...

    def metadata(self) -> PolicyManifest:
        """Checkpoint/processor identity for integrity checks and provenance."""
        ...
