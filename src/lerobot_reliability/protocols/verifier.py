"""Achievement verification contract. [TR 14.2]

Closes recovery/step events by checking whether the intended physical outcome
actually occurred. ``"uncertain"`` is a first-class result; the verifier must not
turn missing evidence into a false success/failure claim. Simulator ground truth
may audit verifier quality but must not enter the primary runtime path.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from lerobot_reliability.data_types import (
    AchievementResult,
    AchievementSpec,
    ExecutionHistory,
)

__all__ = ["AchievementVerifier"]


@runtime_checkable
class AchievementVerifier(Protocol):
    def verify(
        self,
        history: ExecutionHistory,
        instruction: str,
        expected_achievement: AchievementSpec,
    ) -> AchievementResult:
        ...
