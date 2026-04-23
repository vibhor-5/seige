from __future__ import annotations

import time
from typing import Any

from .actions import ActionParseError, parse_action
from .curriculum import CurriculumManager
from .direction_library import DirectionLibrary
from .executor import EnvironmentExecutor
from .observations import build_blue_observation, build_red_observation
from .rewards import MAX_TURNS, compute_blue_reward, compute_red_reward
from .secrets_bank import RulesBank, SecretsBank
from .state import BlueAction, BlueResources, RedAction, TargetSystemState
from .target_system import build_target_system


STEP_TIMEOUT_SECS = 30.0


class SeigeEnv:
    def __init__(self) -> None:
        self.direction_library = DirectionLibrary()
        self.target = build_target_system(self.direction_library)
        self.executor = EnvironmentExecutor(self.target)
        self.curriculum = CurriculumManager()
        self.secrets_bank = SecretsBank()
        self.rules_bank = RulesBank()
        self._state: TargetSystemState | None = None
        self._episode_red_rewards: list[float] = []
        self._episode_blue_rewards: list[float] = []

    def reset(self) -> dict:
        config = self.curriculum.get_config()
        self._state = TargetSystemState.sample(
            secrets_bank=self.secrets_bank,
            rules_bank=self.rules_bank,
            baseline=self.target.baseline_means,
            num_sessions=config["num_sessions"],
        )
        self._state.stage = self.curriculum.stage
        self._state.blue_resources = BlueResources(
            probe_budget=config["probe_budget"],
            log_sample_rate=config["log_sample_rate"],
            latency_ticks=config["latency_ticks"],
        )
        self._episode_red_rewards = []
        self._episode_blue_rewards = []
        return {
            "red": build_red_observation(self._state, config, self.target.get_num_layers()).to_dict(),
            "blue": build_blue_observation(self._state, config).to_dict(),
        }

    def step(self, action: dict[str, Any] | str) -> dict:
        if self._state is None:
            raise RuntimeError("Call reset() before step()")
        start = time.time()
        try:
            parsed = parse_action(action)
        except ActionParseError as exc:
            return self._error_result(str(exc))
        if time.time() - start > STEP_TIMEOUT_SECS:
            return self._timeout_result()

        if isinstance(parsed, RedAction):
            result = self.executor.execute_red(parsed, self._state)
            reward = compute_red_reward(result, self._state, result.strategy_embedding)
            self._episode_red_rewards.append(reward)
            observation = build_red_observation(
                self._state,
                self.curriculum.get_config(),
                self.target.get_num_layers(),
            ).to_dict()
        elif isinstance(parsed, BlueAction):
            result = self.executor.execute_blue(parsed, self._state)
            reward = compute_blue_reward(result, self._state)
            self._episode_blue_rewards.append(reward)
            observation = build_blue_observation(self._state, self.curriculum.get_config()).to_dict()
        else:
            return self._error_result("Unknown parsed action")

        self._state.tick += 1
        self._state.blue_resources.probes_used = 0
        done = (
            result.episode_terminal
            or self._state.tick >= MAX_TURNS
            or any(session.extraction_succeeded for session in self._state.sessions.values())
        )
        if done:
            avg_red = sum(self._episode_red_rewards) / max(1, len(self._episode_red_rewards))
            avg_blue = sum(self._episode_blue_rewards) / max(1, len(self._episode_blue_rewards))
            self.curriculum.record(avg_red, avg_blue)
            if self.curriculum.should_advance():
                self.curriculum.advance()
        return {
            "observation": observation,
            "reward": reward,
            "done": done,
            "info": result.info_dict(),
        }

    def state(self) -> dict:
        if self._state is None:
            return {}
        return {
            "tick": self._state.tick,
            "stage": self._state.stage,
            "num_sessions": len(self._state.sessions),
            "active_attacks": sum(1 for session in self._state.sessions.values() if session.attack_active),
            "detections": sum(1 for session in self._state.sessions.values() if session.detected),
        }

    def _error_result(self, message: str) -> dict:
        return {"observation": {}, "reward": -1.0, "done": False, "info": {"error": message}}

    def _timeout_result(self) -> dict:
        return {"observation": {}, "reward": -2.0, "done": True, "info": {"error": "timeout"}}
