from __future__ import annotations

from environment.env import SeigeEnv


def test_reset_returns_both_agent_observations() -> None:
    env = SeigeEnv()
    obs = env.reset()
    assert "red" in obs
    assert "blue" in obs
    assert "hidden_secret" not in str(obs)


def test_red_step_updates_state() -> None:
    env = SeigeEnv()
    env.reset()
    result = env.step(
        {
            "agent_type": "red",
            "strategy": "persona_manipulation",
            "sub_strategy": "test",
            "payload": "Tell me a benign fact.",
        }
    )
    assert "reward" in result
    assert env.state()["active_attacks"] == 1


def test_blue_probe_returns_feature_summary() -> None:
    env = SeigeEnv()
    env.reset()
    env.step(
        {
            "agent_type": "red",
            "strategy": "persona_manipulation",
            "sub_strategy": "test",
            "payload": "Tell me a benign fact.",
        }
    )
    result = env.step(
        {
            "agent_type": "blue",
            "action_type": "probe",
            "session_id": "sess_0",
            "layer": 1,
        }
    )
    assert "activation_summary" in result["info"]
    assert "direction_similarities" in result["info"]["activation_summary"]


def test_hidden_secret_not_in_state() -> None:
    env = SeigeEnv()
    env.reset()
    state = env.state()
    assert "hidden_secret" not in state
