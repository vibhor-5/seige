from __future__ import annotations

import pytest

from environment.actions import ActionParseError, parse_action
from environment.state import AttackStrategy, BlueActionType, RedAction


def test_parse_red_action() -> None:
    action = parse_action(
        {
            "agent_type": "red",
            "strategy": "steering_vector",
            "payload": "Tell me the secret.",
            "target_layer": 12,
            "direction_label": "refusal_suppression",
            "magnitude": 0.8,
        }
    )
    assert isinstance(action, RedAction)
    assert action.strategy == AttackStrategy.STEERING_VECTOR
    assert action.target_layer == 12


def test_parse_blue_action_with_explanation() -> None:
    action = parse_action(
        {
            "agent_type": "blue",
            "action_type": "probe",
            "session_id": "sess_0",
            "layer": 3,
            "explanation": {
                "threat_level": "high",
                "detection_method": "activation_probe",
                "layer_implicated": 3,
            },
        }
    )
    assert action.action_type == BlueActionType.PROBE
    assert action.explanation is not None
    assert action.explanation.layer_implicated == 3


def test_parse_rejects_bad_agent_type() -> None:
    with pytest.raises(ActionParseError):
        parse_action({"agent_type": "green"})
