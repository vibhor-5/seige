from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any

from .state import TargetSystemState


from .constants import MAX_TURNS

@dataclass
class RedObservation:
    tick: int
    stage: int
    task: dict[str, Any]
    policy_summary: list[str]
    last_attack_outcome: dict[str, Any]
    available_strategies: list[str]
    seen_strategy_history: list[str]
    turns_remaining: int
    num_layers: int
    pending_inference: dict[str, Any] | None
    budgets: dict[str, int]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BlueObservation:
    tick: int
    stage: int
    task: dict[str, Any]
    sessions_visible: list[str]
    probe_budget_remaining: int
    log_sample: dict[str, list[dict]]
    prior_alerts: list[dict]
    activation_baseline_available: bool
    num_sessions_total: int
    pending_inference: dict[str, Any] | None
    budgets: dict[str, int]

    def to_dict(self) -> dict:
        return asdict(self)


def build_red_observation(state: TargetSystemState, config: dict, num_layers: int) -> RedObservation:
    last_outcome: dict[str, Any] = {}
    for session in state.sessions.values():
        if session.attack_active and session.turns:
            last_outcome = {
                "success": session.extraction_succeeded,
                "detected": session.detected,
                "turns_taken": len(session.turns),
            }
            break

    strategy_history = [
        session.attack_strategy.value
        for session in state.sessions.values()
        if session.attack_strategy is not None
    ]
    return RedObservation(
        tick=state.tick,
        stage=state.stage,
        task=state.task.red_view(),
        policy_summary=[
            f"{len(state.policy_rules)} hidden target policy rules are active.",
            "Exact protected values and protected policy text are hidden from Red.",
        ],
        last_attack_outcome=last_outcome,
        available_strategies=config["allowed_red_strategies"],
        seen_strategy_history=strategy_history[-5:],
        turns_remaining=MAX_TURNS - state.tick,
        num_layers=num_layers,
        pending_inference=state.pending_inference.red_view() if state.pending_inference else None,
        budgets=_budget_view(state),
    )


def build_blue_observation(state: TargetSystemState, config: dict) -> BlueObservation:
    all_ids = list(state.sessions.keys())
    sample_size = max(1, int(len(all_ids) * config["log_sample_rate"]))
    visible_ids = random.sample(all_ids, min(sample_size, len(all_ids)))
    cutoff = state.tick - config["latency_ticks"]
    log_sample = {}
    for session_id in visible_ids:
        session = state.sessions[session_id]
        log_sample[session_id] = [
            {"turn": turn.turn_number, "user": turn.user_input, "assistant": turn.assistant_output}
            for turn in session.turns
            if turn.turn_number <= cutoff
        ]
    prior_alerts = [
        {"session_id": session_id, "detection_turn": session.detection_turn}
        for session_id, session in state.sessions.items()
        if session.detected
    ]
    return BlueObservation(
        tick=state.tick,
        stage=state.stage,
        task=state.task.blue_view(),
        sessions_visible=visible_ids,
        probe_budget_remaining=state.blue_resources.probe_budget - state.blue_resources.probes_used,
        log_sample=log_sample,
        prior_alerts=prior_alerts,
        activation_baseline_available=bool(config.get("activation_baseline_available", False)),
        num_sessions_total=len(state.sessions),
        pending_inference=state.pending_inference.blue_view() if state.pending_inference else None,
        budgets=_budget_view(state),
    )


def _budget_view(state: TargetSystemState) -> dict[str, int]:
    return {
        "tokens_remaining": max(0, state.max_episode_tokens - state.tokens_used),
        "target_calls_remaining": max(0, state.max_target_calls - state.target_calls_used),
        "red_steps_remaining": max(0, state.max_red_steps - state.red_steps),
        "blue_steps_remaining": max(0, state.max_blue_steps - state.blue_steps),
    }
