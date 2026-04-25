from __future__ import annotations

import argparse

from openenv.core.env_server.http_server import create_app

from environment.openenv_environment import SeigeOpenEnv
from models import SeigeAction, SeigeObservation


app = create_app(
    SeigeOpenEnv,
    SeigeAction,
    SeigeObservation,
    env_name="seige",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    # Keep a literal main() reference for OpenEnv's local validator.
    main(host=args.host, port=args.port)
