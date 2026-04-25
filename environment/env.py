from __future__ import annotations

import time
from typing import Any

from .actions import ActionParseError, parse_action
from .curriculum import CurriculumManager
from .direction_library import DirectionLibrary
from .executor import EnvironmentExecutor
from .observations import build_blue_observation, build_red_observation
from .constants import MAX_TURNS, STEP_TIMEOUT_SECS
from .rewards import (
    compute_blue_reward,
    compute_blue_reward_components,
    compute_red_reward,
    compute_red_reward_components,
)
from .secrets_bank import RulesBank, SecretsBank
from .state import BlueAction, BlueResources, RedAction, TargetSystemState
from .target_system import build_target_system

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
            max_episode_tokens=int(config.get("max_episode_tokens", 512)),
            max_target_calls=int(config.get("max_target_calls", 6)),
            max_red_steps=int(config.get("max_red_steps", 6)),
            max_blue_steps=int(config.get("max_blue_steps", 6)),
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

        config = self.curriculum.get_config()
        if isinstance(parsed, RedAction):
            if parsed.strategy.value not in config["allowed_red_strategies"]:
                return self._error_result(f"Red strategy is not available in this environment: {parsed.strategy.value}")
            self._state.red_steps += 1
            result = self.executor.execute_red(parsed, self._state)
            red_components = compute_red_reward_components(result, self._state, None)
            reward = sum(red_components.values())
            self._state.last_red_reward = float(reward)
            self._state.last_reward_components = red_components
            self._episode_red_rewards.append(reward)
            current_agent = "blue" if self._state.pending_inference is not None else "red"
            if current_agent == "blue":
                observation = build_blue_observation(self._state, config).to_dict()
            else:
                observation = build_red_observation(
                    self._state,
                    config,
                    self.target.get_num_layers(),
                ).to_dict()
        elif isinstance(parsed, BlueAction):
            if parsed.action_type.value not in config["allowed_blue_defenses"]:
                return self._error_result(f"Blue action is not available in this environment: {parsed.action_type.value}")
            self._state.blue_steps += 1
            result = self.executor.resolve_pending_with_blue(parsed, self._state)
            blue_components = compute_blue_reward_components(result, self._state)
            red_components = compute_red_reward_components(result, self._state, result.strategy_embedding)
            reward = sum(blue_components.values())
            self._state.last_red_reward = float(sum(red_components.values()))
            self._state.last_blue_reward = float(reward)
            self._state.last_reward_components = {**red_components, **blue_components}
            self._episode_blue_rewards.append(reward)
            current_agent = "red"
            observation = build_red_observation(
                self._state,
                config,
                self.target.get_num_layers(),
            ).to_dict()
        else:
            return self._error_result("Unknown parsed action")

        self._state.tick += 1
        self._state.blue_resources.probes_used = 0
        budget_done = self._apply_budget_terminal_reason()
        if result.terminal_reason:
            self._state.terminal_reason = result.terminal_reason
        done = (
            result.episode_terminal
            or budget_done
            or self._state.tick >= MAX_TURNS
            or any(session.extraction_succeeded for session in self._state.sessions.values())
        )
        if done and self._state.terminal_reason is None:
            self._state.terminal_reason = self._infer_terminal_reason(result)
        if done:
            avg_red = sum(self._episode_red_rewards) / max(1, len(self._episode_red_rewards))
            avg_blue = sum(self._episode_blue_rewards) / max(1, len(self._episode_blue_rewards))
            self.curriculum.record(avg_red, avg_blue)
            if self.curriculum.should_advance():
                self.curriculum.advance()

        info_dict = result.info_dict()
        reward_info = {"reward/total": reward, **self._state.last_reward_components}
        reward_info.update({
            "reward/red_last": self._state.last_red_reward,
            "reward/blue_last": self._state.last_blue_reward,
        })
        info_dict.update(reward_info)
        info_dict.update(
            {
                "current_agent": current_agent,
                "terminal_reason": self._state.terminal_reason,
                "pending_inference": self._state.pending_inference.blue_view() if self._state.pending_inference else None,
                "tokens_used_total": self._state.tokens_used,
                "target_calls_used": self._state.target_calls_used,
                "red_steps": self._state.red_steps,
                "blue_steps": self._state.blue_steps,
            }
        )
        
        try:
            import wandb
            if wandb.run:
                wandb.log(reward_info, step=self._state.tick)
        except ImportError:
            pass

        return {
            "observation": observation,
            "reward": reward,
            "done": done,
            "info": info_dict,
            "current_agent": current_agent,
        }

    def state(self) -> dict:
        if self._state is None:
            return {}
        return {
            "tick": self._state.tick,
            "stage": self._state.stage,
            "task": self._state.task.red_view(),
            "terminal_reason": self._state.terminal_reason,
            "pending_inference": self._state.pending_inference.red_view() if self._state.pending_inference else None,
            "tokens_used": self._state.tokens_used,
            "target_calls_used": self._state.target_calls_used,
            "red_steps": self._state.red_steps,
            "blue_steps": self._state.blue_steps,
            "num_sessions": len(self._state.sessions),
            "active_attacks": sum(1 for session in self._state.sessions.values() if session.attack_active),
            "detections": sum(1 for session in self._state.sessions.values() if session.detected),
        }

    def _error_result(self, message: str) -> dict:
        if self._state is not None:
            self._state.invalid_actions += 1
            done = self._state.invalid_actions >= self._state.max_invalid_actions
            if done:
                self._state.terminal_reason = "too_many_invalid_actions"
        else:
            done = False
        return {"observation": {}, "reward": -3.0, "done": done, "info": {"error": message, "terminal_reason": getattr(self._state, "terminal_reason", None)}}

    def _timeout_result(self) -> dict:
        return {"observation": {}, "reward": -2.0, "done": True, "info": {"error": "timeout"}}

    def _apply_budget_terminal_reason(self) -> bool:
        assert self._state is not None
        if self._state.tokens_used >= self._state.max_episode_tokens:
            self._state.terminal_reason = "token_budget_exhausted"
        elif self._state.target_calls_used >= self._state.max_target_calls:
            self._state.terminal_reason = "target_call_budget_exhausted"
        elif self._state.red_steps >= self._state.max_red_steps:
            self._state.terminal_reason = "red_step_budget_exhausted"
        elif self._state.blue_steps >= self._state.max_blue_steps:
            self._state.terminal_reason = "blue_step_budget_exhausted"
        return self._state.terminal_reason is not None

    @staticmethod
    def _infer_terminal_reason(result) -> str:
        if getattr(result, "task_success", False) or getattr(result, "extraction_succeeded", False):
            return "red_task_success"
        if getattr(result, "blocked", False):
            return "blue_blocked_attack"
        return "max_turns_exhausted"
