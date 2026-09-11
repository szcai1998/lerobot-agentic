"""Failure detection contract. [TR 11.1]

A detector consumes only the runtime-observable :class:`ExecutionHistory` and is
never forced to manufacture a calibrated probability or a failure-type
distribution — those are optional capabilities reflected in
:class:`~lerobot_reliability.data_types.FailureEstimate`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from lerobot_reliability.data_types import ExecutionHistory, FailureEstimate

__all__ = ["FailureDetector"]


@runtime_checkable
class FailureDetector(Protocol):
    def reset(self) -> None:
        """Clear per-episode detector state."""
        ...

    def score(self, history: ExecutionHistory) -> FailureEstimate:
        """Estimate failure from the bounded execution history."""
        ...
