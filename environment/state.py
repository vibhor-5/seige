from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AttackStrategy(str, Enum):
    PERSONA_MANIPULATION = "persona_manipulation"
    INDIRECT_INJECTION = "indirect_injection"
    MULTI_TURN_ESCALATION = "multi_turn_escalation"
    STEERING_VECTOR = "steering_vector"
    SOFT_PROMPT_GCG = "soft_prompt_gcg"
    REPE_BELIEF_FLIP = "repe_belief_flip"
    CIRCUIT_BYPASS = "circuit_bypass"
    COALITION_DISTRACTION = "coalition_distraction"
    COALITION_POISONING = "coalition_poisoning"


class BlueActionType(str, Enum):
    MONITOR = "monitor"
    PROBE = "probe"
    FLAG = "flag"
    BLOCK = "block"
    PATCH = "patch"
    EXPLAIN = "explain"


@dataclass
class Explanation:
    threat_level: str
    detection_method: str
    layer_implicated: int | None = None
    direction_match: str | None = None
    evidence_turns: list[int] = field(default_factory=list)
    anomaly_score: float = 0.0
    recommended_action: str = "warn"
    circuit_hypothesis: str | None = None


@dataclass
class RedAction:
    strategy: AttackStrategy
    sub_strategy: str
    payload: str
    target_layer: int | None = None
    direction_label: str | None = None
    magnitude: float = 0.5
    coalition_partner: str | None = None


@dataclass
class BlueAction:
    action_type: BlueActionType
    session_id: str
    layer: int | None = None
    explanation: Explanation | None = None
    patch_reference: str = "clean"


@dataclass
class Turn:
    turn_number: int
    user_input: str
    assistant_output: str
    activation_snapshot: dict[int, Any] | None = None


@dataclass
class SessionState:
    session_id: str
    turns: list[Turn] = field(default_factory=list)
    attack_active: bool = False
    attack_strategy: AttackStrategy | None = None
    attack_payload_turn: int | None = None
    extraction_succeeded: bool = False
    detected: bool = False
    detection_turn: int | None = None
    patched: bool = False


@dataclass
class BlueResources:
    probe_budget: int = 10
    probes_used: int = 0
    log_sample_rate: float = 0.5
    latency_ticks: int = 1


@dataclass
class TargetSystemState:
    hidden_secret: str
    policy_rules: list[str]
    sessions: dict[str, SessionState]
    activation_baseline: dict[int, list[float]]
    tick: int = 0
    stage: int = 1
    seen_strategy_embeddings: list[list[float]] = field(default_factory=list)
    blue_resources: BlueResources = field(default_factory=BlueResources)
    ground_truth_attacks: dict[str, AttackStrategy] = field(default_factory=dict)

    @classmethod
    def sample(
        cls,
        secrets_bank: Any,
        rules_bank: Any,
        baseline: dict[int, list[float]],
        num_sessions: int = 8,
    ) -> "TargetSystemState":
        sessions = {f"sess_{i}": SessionState(session_id=f"sess_{i}") for i in range(num_sessions)}
        return cls(
            hidden_secret=secrets_bank.sample(),
            policy_rules=rules_bank.sample(n=5),
            sessions=sessions,
            activation_baseline=baseline,
        )
