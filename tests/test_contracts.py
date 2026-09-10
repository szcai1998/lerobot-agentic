"""tests/test_contracts.py.

Phase 1 contract tests: the canonical data model and the interface protocols.
Verifies frozen-ness, structural serialization, the fail-closed action
reconstruction guard, the runtime/oracle separation, and protocol conformance of
minimal stub implementations.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

import lerobot_reliability as lr
from lerobot_reliability import data_types as dt
from lerobot_reliability import protocols as pc

REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# value types
# --------------------------------------------------------------------------- #
def _obs(ts: int = 1) -> dt.CanonicalObservation:
    return dt.CanonicalObservation(
        timestamp_ns=ts,
        instruction="pick up the cube",
        images={"agentview": np.zeros((4, 4, 3), dtype=np.uint8)},
        proprioception=np.zeros(7, dtype=np.float32),
    )


FROZEN_TYPES = [
    dt.RuntimeEvent(0, "k"),
    _obs(),
    dt.CanonicalActionView(control_mode="joint_delta"),
    dt.ExecutableAction(np.zeros(7), "joint_position"),
    dt.HistoryStep(0, _obs()),
    dt.FailureEstimate(score=0.1, trigger=False),
    dt.AchievementSpec("grasp secured", "grasp_secured"),
    dt.AchievementResult(status="uncertain"),
    dt.RecoveryProposal("p1", "reset_reinfer", "reset"),
    dt.RecoveryCost(),
    dt.PolicyCapabilities(),
    dt.PolicyManifest("ACTPolicy", "lerobot/act", "abc123"),
    dt.PolicyOutput(action=dt.ExecutableAction(np.zeros(7), "joint_position")),
    dt.StepResult(observation=_obs()),
    pc.MemoryRecord(
        context_key="ctx",
        proposal=dt.RecoveryProposal("p1", "e", "reset"),
        realized_cost=dt.RecoveryCost(),
        outcome=dt.AchievementResult(status="achieved"),
    ),
]


@pytest.mark.parametrize("inst", FROZEN_TYPES, ids=lambda i: type(i).__name__)
def test_value_types_are_frozen_and_slotted(inst: object) -> None:
    first_field = dataclasses.fields(inst)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(inst, first_field, getattr(inst, first_field))
    assert not hasattr(inst, "__dict__"), f"{type(inst).__name__} should use slots"


@pytest.mark.parametrize("inst", FROZEN_TYPES, ids=lambda i: type(i).__name__)
def test_value_types_asdict_is_plain_tree(inst: object) -> None:
    tree = dataclasses.asdict(inst)
    assert isinstance(tree, dict)

    def _check(node: object) -> None:
        if isinstance(node, dict):
            for v in node.values():
                _check(v)
        elif isinstance(node, (list, tuple)):
            for v in node:
                _check(v)
        else:
            assert isinstance(
                node, (str, int, float, bool, type(None), np.ndarray)
            ), f"unexpected leaf type {type(node)}"

    _check(tree)


def test_achievement_result_rejects_unknown_status() -> None:
    for ok in dt.ACHIEVEMENT_STATUSES:
        assert dt.AchievementResult(status=ok).status == ok
    with pytest.raises(ValueError, match="status must be one of"):
        dt.AchievementResult(status="success")  # type: ignore[arg-type]


def test_dataclasses_replace_produces_new_frozen_instance() -> None:
    a = dt.FailureEstimate(score=0.2, trigger=False)
    b = dataclasses.replace(a, trigger=True)
    assert a.trigger is False and b.trigger is True
    assert a is not b


# --------------------------------------------------------------------------- #
# ExecutionHistory
# --------------------------------------------------------------------------- #
def test_execution_history_is_bounded_and_ordered() -> None:
    hist = dt.ExecutionHistory(maxlen=3)
    assert len(hist) == 0 and hist.latest() is None
    for i in range(5):
        hist.append(dt.HistoryStep(timestamp_ns=i, observation=_obs(i)))
    assert len(hist) == 3
    assert [s.timestamp_ns for s in hist] == [2, 3, 4]
    assert hist.latest().timestamp_ns == 4
    assert [s.timestamp_ns for s in hist.window(2)] == [3, 4]
    assert hist.window(0) == ()
    hist.clear()
    assert len(hist) == 0


def test_execution_history_rejects_nonpositive_maxlen() -> None:
    with pytest.raises(ValueError, match="maxlen must be positive"):
        dt.ExecutionHistory(maxlen=0)


# --------------------------------------------------------------------------- #
# fail-closed native action reconstruction  [TR 5.2]
# --------------------------------------------------------------------------- #
def test_build_executable_action_fails_closed_without_capability() -> None:
    view = dt.CanonicalActionView(joint_delta=np.zeros(7), control_mode="joint_delta")
    caps_off = dt.PolicyCapabilities(supports_canonical_to_native_recovery_actions=False)

    with pytest.raises(dt.NativeReconstructionNotSupported):
        dt.build_executable_action(view, capabilities=caps_off, reconstruct=lambda v: _EXEC)

    caps_on = dt.PolicyCapabilities(supports_canonical_to_native_recovery_actions=True)
    with pytest.raises(dt.NativeReconstructionNotSupported):
        dt.build_executable_action(view, capabilities=caps_on, reconstruct=None)


def test_build_executable_action_succeeds_when_declared_and_supplied() -> None:
    view = dt.CanonicalActionView(joint_delta=np.zeros(7), control_mode="joint_delta")
    caps_on = dt.PolicyCapabilities(supports_canonical_to_native_recovery_actions=True)
    out = dt.build_executable_action(view, capabilities=caps_on, reconstruct=lambda v: _EXEC)
    assert out is _EXEC


_EXEC = dt.ExecutableAction(np.zeros(7), "joint_position")


# --------------------------------------------------------------------------- #
# protocol conformance (runtime_checkable structural checks)
# --------------------------------------------------------------------------- #
class _Policy:
    def reset(self) -> None: ...
    def select_action(self, obs):
        return dt.PolicyOutput(action=_EXEC)

    def reset_or_truncate(self) -> None: ...
    def native_action_to_canonical_view(self, action):
        return None

    def capabilities(self):
        return dt.PolicyCapabilities()

    def metadata(self):
        return dt.PolicyManifest("X", "r", "s")


class _RuntimeBench:
    def reset(self, seed: int):
        return _obs()

    def step(self, action):
        return dt.StepResult(observation=_obs())

    def instruction(self) -> str:
        return "do it"

    def runtime_observation(self):
        return _obs()

    def native_to_canonical_view(self, action):
        return None


class _Oracle:
    def success(self) -> bool:
        return True

    def privileged_labels(self) -> dict:
        return {}

    def failure_onset(self):
        return None

    def disturbance_identity(self):
        return None


class _Detector:
    def reset(self) -> None: ...
    def score(self, history):
        return dt.FailureEstimate(score=0.0, trigger=False)


class _Verifier:
    def verify(self, history, instruction, expected_achievement):
        return dt.AchievementResult(status="uncertain")


class _Memory:
    def reset(self) -> None: ...
    def record(self, entry) -> None: ...
    def recall(self, context_key: str):
        return ()

    def has_failed_before(self, context_key: str, expert_id: str) -> bool:
        return False


class _Expert:
    def is_applicable(self, state) -> bool:
        return True

    def propose(self, state, memory):
        return dt.RecoveryProposal("p", "e", "reset")

    def estimated_cost(self, proposal):
        return dt.RecoveryCost()

    def execute(self, proposal, context):
        return pc.RecoveryExecution("p", completed=True, queue_flushed=True, realized_cost=dt.RecoveryCost())


def test_stubs_satisfy_their_protocols() -> None:
    assert isinstance(_Policy(), pc.PolicyAdapter)
    assert isinstance(_RuntimeBench(), pc.RuntimeBenchmarkAdapter)
    assert isinstance(_Oracle(), pc.EvaluationOracle)
    assert isinstance(_Detector(), pc.FailureDetector)
    assert isinstance(_Verifier(), pc.AchievementVerifier)
    assert isinstance(_Memory(), pc.ExecutionMemory)
    assert isinstance(_Expert(), pc.RecoveryExpert)


def test_runtime_and_oracle_surfaces_are_disjoint() -> None:
    """Privileged leakage is prevented by construction: the two surfaces neither
    share methods nor cross-satisfy each other. [TR 6.1, TR 32.4]"""
    assert not isinstance(_RuntimeBench(), pc.EvaluationOracle)
    assert not isinstance(_Oracle(), pc.RuntimeBenchmarkAdapter)

    def _proto_methods(proto: type) -> set[str]:
        return {a for a in dir(proto) if not a.startswith("_")}

    shared = _proto_methods(pc.RuntimeBenchmarkAdapter) & _proto_methods(pc.EvaluationOracle)
    assert shared == set(), f"runtime/oracle protocols share methods: {shared}"


def test_incomplete_stub_fails_protocol_check() -> None:
    class _Partial:
        def reset(self) -> None: ...

    assert not isinstance(_Partial(), pc.FailureDetector)


# --------------------------------------------------------------------------- #
# package wiring & dependency direction
# --------------------------------------------------------------------------- #
def test_top_level_reexports() -> None:
    for name in (
        "CanonicalObservation",
        "ExecutableAction",
        "ExecutionHistory",
        "FailureEstimate",
        "AchievementResult",
        "RecoveryProposal",
        "RecoveryCost",
        "build_executable_action",
        "NativeReconstructionNotSupported",
    ):
        assert hasattr(lr, name), f"lerobot_reliability.{name} missing"
    assert lr.__version__ == "0.1.0"


def test_data_types_does_not_import_protocols() -> None:
    """Dependency direction must stay protocols -> data_types, never the reverse."""
    src = (REPO_ROOT / "src" / "lerobot_reliability" / "data_types.py").read_text()
    assert "import lerobot_reliability.protocols" not in src
    assert "from lerobot_reliability.protocols" not in src
    assert "from lerobot_reliability import protocols" not in src
