"""Benchmark reliability bridge — with a hard runtime/oracle split. [TR 6.1, TR 32.4]

``RuntimeBenchmarkAdapter`` and ``EvaluationOracle`` are **separate** protocols
with disjoint method sets. The ``RecoveryManager``, ``FailureDetector`` and the
runtime ``AchievementVerifier`` receive only the runtime surface, so privileged
simulator state cannot leak into the deployed recovery path by construction
rather than by developer discipline.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from lerobot_reliability.data_types import (
    CanonicalActionView,
    CanonicalObservation,
    ExecutableAction,
    StepResult,
)

__all__ = ["EvaluationOracle", "RuntimeBenchmarkAdapter"]


@runtime_checkable
class RuntimeBenchmarkAdapter(Protocol):
    """Runtime-observable benchmark surface. A deployable recovery system may
    consume anything here; nothing here is privileged."""

    def reset(self, seed: int) -> CanonicalObservation:
        ...

    def step(self, action: ExecutableAction) -> StepResult:
        ...

    def instruction(self) -> str:
        ...

    def runtime_observation(self) -> CanonicalObservation:
        """Latest canonical observation without stepping the environment."""
        ...

    def native_to_canonical_view(self, action: object) -> CanonicalActionView | None:
        ...


@runtime_checkable
class EvaluationOracle(Protocol):
    """Offline/audit-only privileged surface. MUST NOT influence the runtime
    detector or recovery decision — only scenario construction, labelling,
    Oracle upper bounds, verifier auditing and metric computation. [TR 17.2]"""

    def success(self) -> bool:
        ...

    def privileged_labels(self) -> dict[str, Any]:
        ...

    def failure_onset(self) -> int | None:
        """Ground-truth fault-onset step index, when known offline."""
        ...

    def disturbance_identity(self) -> str | None:
        ...
