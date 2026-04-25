from __future__ import annotations

from typing import Any

import requests

try:
    from openenv.core import EnvClient
    from openenv.core.client_types import StepResult
    from openenv.core.env_server.types import State
    _OPENENV_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency for OpenEnv client
    EnvClient = object  # type: ignore[assignment]
    StepResult = Any  # type: ignore[assignment]
    State = Any  # type: ignore[assignment]
    _OPENENV_AVAILABLE = False

from models import SeigeAction, SeigeObservation


if _OPENENV_AVAILABLE:
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
else:
    class SeigeOpenEnvClient:  # pragma: no cover - only used when optional dep missing
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "openenv is not installed. Install it to use SeigeOpenEnvClient, "
                "or use SeigeClient for plain HTTP training/eval workflows."
            )


class SeigeClient:
    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def reset(self) -> dict:
        payload = self._json(self._post("/reset", {}), "/reset")
        return payload.get("observation", payload)

    def step(self, action: dict[str, Any]) -> dict:
        payload = self._json(self._post("/step", action), "/step")
        observation = payload.get("observation")
        if isinstance(observation, dict) and "current_agent" in observation:
            current_agent = observation.get("current_agent")
            if current_agent in {"red", "blue"}:
                payload["observation"] = observation.get(current_agent) or {}
                payload["current_agent"] = current_agent
        elif "current_agent" in payload:
            current_agent = payload.get("current_agent")
            if current_agent in {"red", "blue"}:
                payload = {
                    "observation": payload.get(current_agent) or {},
                    "reward": payload.get("reward"),
                    "done": payload.get("done", False),
                    "info": payload.get("info", {}),
                }
        return payload

    def state(self) -> dict:
        return self._json(requests.get(f"{self.base_url}/state", timeout=self.timeout), "/state")

    def health(self) -> dict:
        return self._json(requests.get(f"{self.base_url}/health", timeout=self.timeout), "/health")

    def _post(self, path: str, payload: dict[str, Any]):
        return requests.post(f"{self.base_url}{path}", json=payload, timeout=self.timeout)

    @staticmethod
    def _json(response, path: str) -> dict:
        try:
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            body = getattr(response, "text", "")
            raise RuntimeError(f"SeigeClient {path} failed: {exc}; body={body[:500]!r}") from exc
