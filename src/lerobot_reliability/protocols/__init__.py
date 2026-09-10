"""Interface contracts for LeRobot Reliability Lab.

Structural ``typing.Protocol`` definitions only — no implementations. Every
concrete adapter/detector/expert/verifier/memory is validated against these in
``tests/test_contracts.py``. Dependency direction: ``protocols -> data_types``.
"""

from lerobot_reliability.protocols.benchmark import (
    EvaluationOracle,
    RuntimeBenchmarkAdapter,
)
from lerobot_reliability.protocols.detector import FailureDetector
from lerobot_reliability.protocols.memory import ExecutionMemory, MemoryRecord
from lerobot_reliability.protocols.policy import (
    PolicyAdapter,
    PolicyCapabilities,
    PolicyManifest,
    PolicyOutput,
)
from lerobot_reliability.protocols.recovery import (
    RecoveryExecution,
    RecoveryExpert,
    RecoveryState,
    RuntimeContext,
)
from lerobot_reliability.protocols.verifier import AchievementVerifier

__all__ = [
    "AchievementVerifier",
    "EvaluationOracle",
    "ExecutionMemory",
    "FailureDetector",
    "MemoryRecord",
    "PolicyAdapter",
    "PolicyCapabilities",
    "PolicyManifest",
    "PolicyOutput",
    "RecoveryExecution",
    "RecoveryExpert",
    "RecoveryState",
    "RuntimeBenchmarkAdapter",
    "RuntimeContext",
]
