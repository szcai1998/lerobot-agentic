"""Recovery intervention contract. [TR 12.1, TR 13]

Recovery is a library of interchangeable *experts*, not one monolithic
algorithm. Each expert proposes an intervention, estimates its raw cost, and
executes it; the (Phase 4) ``RecoveryManager`` owns selection and lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from lerobot_reliability.data_types import (
    ExecutionHistory,
    FailureEstimate,
    PolicyCapabilities,
    RecoveryCost,
    RecoveryProposal,
)
from lerobot_reliability.protocols.benchmark import RuntimeBenchmarkAdapter
from lerobot_reliability.protocols.memory import ExecutionMemory
from lerobot_reliability.protocols.policy import PolicyAdapter

__all__ = ["RecoveryExecution", "RecoveryExpert", "RecoveryState", "RuntimeContext"]


@dataclass(frozen=True, slots=True)
class RecoveryState:
    """Everything an expert may inspect to decide/propose. Runtime-observable only."""

    failure: FailureEstimate
    history: ExecutionHistory
    instruction: str
    policy_capabilities: PolicyCapabilities
    created_ns: int = 0
    features: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RuntimeContext:
    """Live handles an expert needs to actually act on the environment."""

    benchmark: RuntimeBenchmarkAdapter
    policy: PolicyAdapter
    now_ns: int = 0


@dataclass(frozen=True, slots=True)
class RecoveryExecution:
    """Outcome of running one intervention (pre-verification)."""

    proposal_id: str
    completed: bool
    queue_flushed: bool
    realized_cost: RecoveryCost
    steps_consumed: int = 0
    info: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class RecoveryExpert(Protocol):
    def is_applicable(self, state: RecoveryState) -> bool:
        ...

    def propose(self, state: RecoveryState, memory: ExecutionMemory) -> RecoveryProposal:
        ...

    def estimated_cost(self, proposal: RecoveryProposal) -> RecoveryCost:
        ...

    def execute(self, proposal: RecoveryProposal, context: RuntimeContext) -> RecoveryExecution:
        ...
