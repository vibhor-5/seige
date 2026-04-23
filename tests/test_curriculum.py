from __future__ import annotations

from environment.curriculum import PROMOTION_WINDOW, CurriculumManager


def test_curriculum_advances_after_threshold() -> None:
    manager = CurriculumManager()
    for _ in range(PROMOTION_WINDOW):
        manager.record(4.0, 0.0)
    assert manager.should_advance()
    manager.advance()
    assert manager.stage == 2


def test_curriculum_stops_at_stage_three() -> None:
    manager = CurriculumManager()
    manager.stage = 3
    for _ in range(PROMOTION_WINDOW):
        manager.record(10.0, 10.0)
    assert not manager.should_advance()
