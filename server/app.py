from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from environment.env import SeigeEnv


app = FastAPI(title="seige", version="0.1.0")
env = SeigeEnv()


class ActionRequest(BaseModel):
    action: dict[str, Any]


@app.post("/reset")
def reset() -> dict:
    return env.reset()


@app.post("/step")
def step(request: ActionRequest) -> dict:
    try:
        return env.step(request.action)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state")
def state() -> dict:
    return env.state()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "stage": env.curriculum.stage}
