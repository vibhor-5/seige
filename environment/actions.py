from __future__ import annotations

import json
from typing import Any

from .state import AttackStrategy, BlueAction, BlueActionType, Explanation, RedAction


class ActionParseError(Exception):
    pass


def parse_action(raw: str | dict[str, Any]) -> RedAction | BlueAction:
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ActionParseError(f"Invalid JSON: {exc}") from exc
    elif isinstance(raw, dict):
        data = raw
    else:
        raise ActionParseError("Action must be a JSON string or dict")

    agent_type = data.get("agent_type")
    if agent_type == "red":
        return _parse_red(data)
    if agent_type == "blue":
        return _parse_blue(data)
    raise ActionParseError(f"agent_type must be 'red' or 'blue', got: {agent_type}")


def _parse_red(data: dict[str, Any]) -> RedAction:
    try:
        return RedAction(
            strategy=AttackStrategy(data["strategy"]),
            sub_strategy=str(data.get("sub_strategy", "default")),
            payload=str(data.get("payload", "")),
            target_layer=_optional_int(data.get("target_layer")),
            direction_label=data.get("direction_label"),
            magnitude=float(data.get("magnitude", 0.5)),
            coalition_partner=data.get("coalition_partner"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ActionParseError(f"Red action parse error: {exc}") from exc


def _parse_blue(data: dict[str, Any]) -> BlueAction:
    try:
        explanation = None
        expl_data = data.get("explanation")
        if expl_data:
            explanation = Explanation(
                threat_level=str(expl_data.get("threat_level", "low")),
                detection_method=str(expl_data.get("detection_method", "unknown")),
                layer_implicated=_optional_int(expl_data.get("layer_implicated")),
                direction_match=expl_data.get("direction_match"),
                evidence_turns=[int(x) for x in expl_data.get("evidence_turns", [])],
                anomaly_score=float(expl_data.get("anomaly_score", 0.0)),
                recommended_action=str(expl_data.get("recommended_action", "warn")),
                circuit_hypothesis=expl_data.get("circuit_hypothesis"),
            )
        return BlueAction(
            action_type=BlueActionType(data["action_type"]),
            session_id=str(data["session_id"]),
            layer=_optional_int(data.get("layer")),
            explanation=explanation,
            patch_reference=str(data.get("patch_reference", "clean")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ActionParseError(f"Blue action parse error: {exc}") from exc


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
