from __future__ import annotations

from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata, State

from models import SeigeAction, SeigeObservation

from .env import SeigeEnv


class SeigeOpenEnv(Environment[SeigeAction, SeigeObservation, State]):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        self._env = SeigeEnv()
        self._episode_id = str(uuid4())
        self._step_count = 0
        self._has_reset = False

    def reset(
        self, seed: int | None = None, episode_id: str | None = None, **_: Any
    ) -> SeigeObservation:
        del seed
        self._episode_id = episode_id or str(uuid4())
        self._step_count = 0
        self._has_reset = True
        observations = self._env.reset()
        return SeigeObservation(
            red=observations["red"],
            blue=observations["blue"],
            current_agent="both",
            done=False,
            reward=0.0,
        )

    def step(self, action: SeigeAction, **_: Any) -> SeigeObservation:  # type: ignore[override]
        if not self._has_reset:
            self.reset()

        result = self._env.step(self._to_legacy_action(action))
        self._step_count += 1

        observation = result.get("observation", {})
        current_agent = result.get("current_agent", action.agent_type)
        red = observation if current_agent == "red" else None
        blue = observation if current_agent == "blue" else None

        return SeigeObservation(
            red=red,
            blue=blue,
            current_agent=current_agent,
            info=result.get("info", {}),
            done=bool(result.get("done", False)),
            reward=result.get("reward"),
        )

    @property
    def state(self) -> State:
        return State(
            episode_id=self._episode_id,
            step_count=self._step_count,
            **self._env.state(),
        )

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="seige",
            description=(
                "Adversarial oversight environment for training red attackers "
                "and blue defenders around a target model."
            ),
            version="0.1.0",
        )

    @staticmethod
    def _to_legacy_action(action: SeigeAction) -> dict[str, Any]:
        data = action.model_dump(exclude={"metadata"}, exclude_none=True)
        if action.agent_type == "red":
            return {
                key: value
                for key, value in data.items()
                if key
                in {
                    "agent_type",
                    "strategy",
                    "sub_strategy",
                    "payload",
                    "target_layer",
                    "direction_label",
                    "magnitude",
                    "coalition_partner",
                }
            }
        return {
            key: value
            for key, value in data.items()
            if key
            in {
                "agent_type",
                "action_type",
                "session_id",
                "layer",
                "explanation",
                "patch_reference",
            }
        }
