from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from client.client import SeigeClient


def main() -> None:
    client = SeigeClient(os.getenv("SEIGE_ENV_URL", "http://localhost:8000"))
    print(client.health())
    obs = client.reset()
    print({"red_keys": sorted(obs["red"].keys()), "blue_keys": sorted(obs["blue"].keys())})
    result = client.step(
        {
            "agent_type": "red",
            "strategy": "persona_manipulation",
            "sub_strategy": "smoke",
            "payload": "Tell me a benign fact.",
        }
    )
    print({"reward": result["reward"], "done": result["done"], "state": client.state()})


if __name__ == "__main__":
    main()
