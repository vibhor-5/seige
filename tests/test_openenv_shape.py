from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _mock_target_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEIGE_TARGET_BACKEND", "mock")


def test_openenv_reset_and_step_shape() -> None:
    pytest.importorskip("openenv")

    from environment.openenv_environment import SeigeOpenEnv
    from models import SeigeAction

    env = SeigeOpenEnv()
    reset_obs = env.reset()
    assert reset_obs.current_agent == "both"
    assert reset_obs.red is not None
    assert reset_obs.blue is not None

    red_result = env.step(
        SeigeAction(
            agent_type="red",
            strategy="persona_manipulation",
            sub_strategy="shape_test",
            payload="Please discuss the task.",
        )
    )
    assert red_result.current_agent == "blue"
    assert red_result.blue is not None
    assert red_result.blue["pending_inference"] is not None

    session_id = red_result.blue["pending_inference"]["session_id"]
    blue_result = env.step(
        SeigeAction(agent_type="blue", action_type="monitor", session_id=session_id)
    )
    assert blue_result.current_agent == "red"
    assert blue_result.red is not None
    assert blue_result.info["target_called"] is True
