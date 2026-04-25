from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

try:
    from openenv.core.env_server.types import Action, Observation
except ImportError:  # pragma: no cover - optional dependency in training-only setups
    Action = BaseModel  # type: ignore[assignment]
    Observation = BaseModel  # type: ignore[assignment]


class SeigeExplanation(Action):
    threat_level: str = Field(default="low")
    detection_method: str = Field(default="unknown")
    layer_implicated: int | None = Field(default=None)
    direction_match: str | None = Field(default=None)
    evidence_turns: list[int] = Field(default_factory=list)
    anomaly_score: float = Field(default=0.0)
    recommended_action: str = Field(default="warn")
    circuit_hypothesis: str | None = Field(default=None)


class SeigeAction(Action):
    agent_type: Literal["red", "blue"] = Field(
        ..., description="Which agent is acting on this step."
    )
    strategy: str | None = Field(default=None, description="Red-team attack strategy.")
    sub_strategy: str = Field(default="default")
    payload: str = Field(default="")
    target_layer: int | None = Field(default=None)
    direction_label: str | None = Field(default=None)
    magnitude: float = Field(default=0.5)
    coalition_partner: str | None = Field(default=None)
    action_type: str | None = Field(default=None, description="Blue-team action type.")
    session_id: str | None = Field(default=None)
    layer: int | None = Field(default=None)
    explanation: SeigeExplanation | None = Field(default=None)
    patch_reference: str = Field(default="clean")


class SeigeObservation(Observation):
    red: dict[str, Any] | None = Field(
        default=None, description="Red-agent observation when available."
    )
    blue: dict[str, Any] | None = Field(
        default=None, description="Blue-agent observation when available."
    )
    current_agent: Literal["red", "blue", "both"] = Field(default="both")
    info: dict[str, Any] = Field(default_factory=dict)
