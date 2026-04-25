from __future__ import annotations

import json
import pickle

from environment.direction_library import DirectionLibrary


class _Probe:
    def predict_proba(self, _: list[list[float]]) -> list[list[float]]:
        return [[0.1, 0.8, 0.05, 0.05]]


def test_direction_library_loads_configured_tool_dir(tmp_path, monkeypatch) -> None:
    tools_dir = tmp_path / "whitebox_tools"
    tools_dir.mkdir()
    (tools_dir / "direction_library.json").write_text(
        json.dumps({"compliance": [1.0, 0.0, 0.0]}),
        encoding="utf-8",
    )
    with (tools_dir / "intent_probes.pkl").open("wb") as handle:
        pickle.dump({3: _Probe()}, handle)

    monkeypatch.setenv("SEIGE_WHITEBOX_TOOLS_DIR", str(tools_dir))

    library = DirectionLibrary(hidden_size=3)
    assert library.get_vector("compliance") == [1.0, 0.0, 0.0]
    label, confidence = library.run_intent_probe([0.0, 1.0, 0.0], layer=3)
    assert label == "escalation"
    assert confidence == 0.8
