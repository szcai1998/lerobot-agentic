"""Canonical runtime data model for LeRobot Reliability Lab.

Pure, dependency-light value types shared across every layer (policy adapters,
benchmark bridges, detectors, recovery experts, verifiers, telemetry, analysis).
This module MUST NOT import from ``lerobot_reliability.protocols`` — the
dependency direction is strictly ``protocols -> data_types``.

Design contracts (frozen Technical Report references in brackets):

* Canonical observations keep **named** views, never a flattened positional
  tensor. [TR 5.1]
* ``CanonicalActionView`` is an **analysis projection only**. It is not assumed
  invertible; a native executable action is never reconstructed from it unless a
  ``PolicyCapabilities`` flag explicitly allows it. [TR 5.2, TR 32.3]
* Attempted state is not achieved state: ``AchievementResult`` carries a
  first-class ``"uncertain"`` status. [TR 3.5, TR 14.2]
* Raw recovery cost dimensions are always retained; a scalar cost is never the
  only record. [TR 18]

Array policy: canonical arrays crossing this boundary are ``numpy.ndarray`` — no
autograd/device state, matching the ``vla-eval`` wire format and Parquet/JSONL
telemetry. Policy adapters convert to/from framework tensors at their own edge.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

Array = np.ndarray

__all__ = [
    "ACHIEVEMENT_STATUSES",
    "AchievementResult",
    "AchievementSpec",
    "AchievementStatus",
    "Array",
    "CanonicalActionView",
    "CanonicalObservation",
    "ExecutableAction",
    "ExecutionHistory",
    "FailureEstimate",
    "HistoryStep",
    "NativeReconstructionNotSupported",
    "PolicyCapabilities",
    "PolicyManifest",
    "PolicyOutput",
    "ReconstructFn",
    "RecoveryCost",
    "RecoveryProposal",
    "RuntimeEvent",
    "StepResult",
    "build_executable_action",
]


# --------------------------------------------------------------------------- #
# Runtime events & observations
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    """A single timestamped runtime event (``e`` in the execution history)."""

    timestamp_ns: int
    kind: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CanonicalObservation:
    """Benchmark-native observation normalised into a canonical runtime structure. [TR 5.1]

    ``images`` / ``depth`` preserve semantic view names (e.g. ``"agentview"``,
    ``"wrist"``); they are never silently concatenated into one tensor.
    """

    timestamp_ns: int
    instruction: str
    images: dict[str, Array] = field(default_factory=dict)
    depth: dict[str, Array] | None = None
    proprioception: Array | None = None
    gripper_state: Array | None = None
    benchmark_features: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Actions
# --------------------------------------------------------------------------- #
ControlMode = Literal[
    "joint_position",
    "joint_delta",
    "joint_velocity",
    "ee_pose",
    "ee_delta",
    "unknown",
]


@dataclass(frozen=True, slots=True)
class CanonicalActionView:
    """Analysis/comparison projection of an executed action. NOT executable. [TR 5.2]

    Any field may be ``None`` when the projection is unavailable for that policy /
    control mode. Never compare raw action vectors from heterogeneous policies as
    if their dimensions share semantics — compare through this view instead.
    """

    translation_delta_m: Array | None = None
    rotation_delta: Array | None = None
    joint_delta: Array | None = None
    gripper_command: float | None = None
    frame: str | None = None
    control_mode: ControlMode = "unknown"
    horizon_index: int | None = None


@dataclass(frozen=True, slots=True)
class ExecutableAction:
    """What is actually sent to the benchmark / robot controller. [TR 5.2]

    ``canonical_view`` is telemetry/analysis only and is explicitly *not* assumed
    to be a lossless inverse of ``native_action``.
    """

    native_action: Array
    native_space: str
    canonical_view: CanonicalActionView | None = None


class NativeReconstructionNotSupported(RuntimeError):
    """Raised when code tries to turn a ``CanonicalActionView`` back into an
    ``ExecutableAction`` without an adapter that declares & implements it. [TR 5.2]"""


#: Adapter-supplied callable that rebuilds a native action from a canonical view.
ReconstructFn = Callable[[CanonicalActionView], ExecutableAction]


def build_executable_action(
    view: CanonicalActionView,
    *,
    capabilities: PolicyCapabilities,
    reconstruct: ReconstructFn | None = None,
) -> ExecutableAction:
    """Reconstruct a native action from a canonical view — only when allowed.

    Fails closed: raises :class:`NativeReconstructionNotSupported` unless the
    owning adapter both sets
    ``capabilities.supports_canonical_to_native_recovery_actions`` and supplies a
    tested ``reconstruct`` callable.
    """

    if not capabilities.supports_canonical_to_native_recovery_actions or reconstruct is None:
        raise NativeReconstructionNotSupported(
            "canonical -> native action reconstruction requires "
            "PolicyCapabilities.supports_canonical_to_native_recovery_actions=True "
            "and an explicit reconstruct() callable from the policy adapter"
        )
    return reconstruct(view)


# --------------------------------------------------------------------------- #
# Execution history  (H_t = {o, a, Â, r, e})  [TR 5.3]
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class HistoryStep:
    """One aligned slice of execution history."""

    timestamp_ns: int
    observation: CanonicalObservation
    executed_action: ExecutableAction | None = None
    predicted_chunk: Array | None = None
    progress: float | None = None
    events: tuple[RuntimeEvent, ...] = ()


class ExecutionHistory:
    """Bounded rolling buffer of :class:`HistoryStep`, the sole input surface for
    failure detection and recovery-state estimation. [TR 5.3, TR 11.1]"""

    __slots__ = ("_steps", "maxlen")

    def __init__(self, maxlen: int = 64) -> None:
        if maxlen <= 0:
            raise ValueError("maxlen must be positive")
        self.maxlen = maxlen
        self._steps: deque[HistoryStep] = deque(maxlen=maxlen)

    def append(self, step: HistoryStep) -> None:
        self._steps.append(step)

    def extend(self, steps: list[HistoryStep]) -> None:
        self._steps.extend(steps)

    def clear(self) -> None:
        self._steps.clear()

    @property
    def steps(self) -> tuple[HistoryStep, ...]:
        return tuple(self._steps)

    def latest(self) -> HistoryStep | None:
        return self._steps[-1] if self._steps else None

    def window(self, n: int) -> tuple[HistoryStep, ...]:
        """The most recent ``n`` steps (fewer if the buffer is shorter)."""
        if n <= 0:
            return ()
        return tuple(self._steps)[-n:]

    def __len__(self) -> int:
        return len(self._steps)

    def __iter__(self) -> Iterator[HistoryStep]:
        return iter(self._steps)

    def __repr__(self) -> str:
        return f"ExecutionHistory(len={len(self._steps)}, maxlen={self.maxlen})"


# --------------------------------------------------------------------------- #
# Failure estimation  [TR 11.1]
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class FailureEstimate:
    """Detector output. Calibrated probability and typed classification are
    optional capabilities — a detector is never forced to invent them."""

    score: float
    trigger: bool
    probability: float | None = None
    type_distribution: dict[str, float] | None = None
    confidence: float | None = None
    diagnostic: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Achievement verification  [TR 14.2]
# --------------------------------------------------------------------------- #
AchievementStatus = Literal["achieved", "not_achieved", "uncertain"]
ACHIEVEMENT_STATUSES: tuple[AchievementStatus, ...] = ("achieved", "not_achieved", "uncertain")


@dataclass(frozen=True, slots=True)
class AchievementSpec:
    """Declares the physical outcome a recovery/step is expected to produce."""

    description: str
    check_kind: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AchievementResult:
    """Result of verifying whether an intended physical outcome actually occurred.

    ``"uncertain"`` is a first-class result: the verifier must not convert missing
    or ambiguous evidence into a false claim of success or failure. [TR 14.2]
    """

    status: AchievementStatus
    confidence: float | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    timestamp_ns: int = 0

    def __post_init__(self) -> None:
        if self.status not in ACHIEVEMENT_STATUSES:
            raise ValueError(
                f"status must be one of {ACHIEVEMENT_STATUSES}, got {self.status!r}"
            )


# --------------------------------------------------------------------------- #
# Recovery proposal & cost  [TR 12.3, TR 18]
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class RecoveryProposal:
    """A single proposed intervention from a :class:`RecoveryExpert`."""

    proposal_id: str
    expert_id: str
    recovery_type: str
    preconditions: dict[str, Any] = field(default_factory=dict)
    expected_achievement: AchievementSpec | None = None
    expected_latency_ms: float | None = None
    expected_compute_cost: float | None = None
    expected_api_usage: dict[str, Any] | None = None
    expected_api_cost_snapshot: float | None = None
    expected_disruption: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RecoveryCost:
    """Raw, multi-dimensional cost of an intervention. A scalar cost may be
    derived per-experiment but these raw dimensions are always retained. [TR 18]"""

    wall_clock_ms: float = 0.0
    gpu_ms: float | None = None
    api_calls: int = 0
    intervention_steps: int = 0
    queue_disruption: int = 0
    api_usage: dict[str, Any] | None = None
    api_cost_snapshot: float | None = None


# --------------------------------------------------------------------------- #
# Policy value types  [TR 7.1, TR 9.1]
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class PolicyCapabilities:
    """What a policy adapter can actually do. Recovery modules must not assume a
    capability the adapter does not declare. [TR 7.1]"""

    supports_chunking: bool = False
    supports_queue_reset: bool = False
    supports_candidate_sampling: bool = False
    supports_language_goal: bool = False
    exposes_policy_features: bool = False
    supports_async_inference: bool = False
    supports_canonical_to_native_recovery_actions: bool = False


@dataclass(frozen=True, slots=True)
class PolicyManifest:
    """Checkpoint/processor identity for fail-closed integrity & provenance. [TR 9.1]"""

    policy_class: str
    source_repo: str
    revision_sha: str
    config_hash: str = ""
    weight_hash: str = ""
    parameter_count: int = 0
    preprocessor_hash: str = ""
    postprocessor_hash: str = ""
    normalization_stats_hash: str = ""
    camera_mapping: dict[str, str] = field(default_factory=dict)
    action_mapping: dict[str, Any] = field(default_factory=dict)
    load_warnings: tuple[str, ...] = ()
    known_output_smoke_test: Literal["pass", "fail", "unknown"] = "unknown"


@dataclass(frozen=True, slots=True)
class PolicyOutput:
    """Return of ``PolicyAdapter.select_action``."""

    action: ExecutableAction
    action_chunk: Array | None = None
    policy_features: Array | None = None
    info: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Benchmark step result  [TR 6.1]
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class StepResult:
    """Return of ``RuntimeBenchmarkAdapter.step`` — runtime-observable only."""

    observation: CanonicalObservation
    reward: float | None = None
    terminated: bool = False
    truncated: bool = False
    info: dict[str, Any] = field(default_factory=dict)
