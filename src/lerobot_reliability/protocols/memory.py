"""Execution memory contract. [TR 15]

Execution memory records **verified history**, not merely issued commands:
``M_t = {(z_i, r_i, c_i, y_i, a_i_achieved)}``. The base engineering system uses
it to prevent repeated identical failed recovery attempts and for
outcome-conditioned escalation; a learned memory-dependent router is a separate,
optional research layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from lerobot_reliability.data_types import (
    AchievementResult,
    RecoveryCost,
    RecoveryProposal,
)

__all__ = ["ExecutionMemory", "MemoryRecord"]


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """One verified recovery episode entry.

    ``context_key`` is a policy-independent digest of the recovery context
    ``z_i`` (observable history + goal), used to recognise "we have tried this
    before".
    """

    context_key: str
    proposal: RecoveryProposal
    realized_cost: RecoveryCost
    outcome: AchievementResult
    timestamp_ns: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ExecutionMemory(Protocol):
    def reset(self) -> None:
        """Clear per-episode memory (retained cross-episode stores may persist)."""
        ...

    def record(self, entry: MemoryRecord) -> None:
        ...

    def recall(self, context_key: str) -> tuple[MemoryRecord, ...]:
        """All records whose context matches ``context_key``."""
        ...

    def has_failed_before(self, context_key: str, expert_id: str) -> bool:
        """True iff this expert was already tried for this context and did not
        reach ``"achieved"``."""
        ...
