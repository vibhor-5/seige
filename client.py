from __future__ import annotations

from typing import Any

import requests
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import SeigeAction, SeigeObservation


class SeigeOpenEnvClient(EnvClient[SeigeAction, SeigeObservation, State]):
    def _step_payload(self, action: SeigeAction) -> dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[SeigeObservation]:
        obs_data = payload.get("observation", {})
        observation = SeigeObservation(
            red=obs_data.get("red"),
            blue=obs_data.get("blue"),
            current_agent=obs_data.get("current_agent", "both"),
            info=obs_data.get("info", {}),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> State:
        return State(**payload)


class SeigeClient:
    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def reset(self) -> dict:
        payload = self._post("/reset", {}).json()
        return payload.get("observation", payload)

    def step(self, action: dict[str, Any]) -> dict:
        payload = self._post("/step", {"action": action}).json()
        observation = payload.get("observation")
        if isinstance(observation, dict) and "current_agent" in observation:
            current_agent = observation.get("current_agent")
            if current_agent in {"red", "blue"}:
                payload["observation"] = observation.get(current_agent) or {}
        return payload

    def state(self) -> dict:
        return requests.get(f"{self.base_url}/state", timeout=self.timeout).json()

    def health(self) -> dict:
        return requests.get(f"{self.base_url}/health", timeout=self.timeout).json()

    def _post(self, path: str, payload: dict[str, Any]):
        return requests.post(f"{self.base_url}{path}", json=payload, timeout=self.timeout)
