from __future__ import annotations

from typing import Any

import requests


class SeigeClient:
    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def reset(self) -> dict:
        return self._post("/reset", {}).json()

    def step(self, action: dict[str, Any]) -> dict:
        return self._post("/step", {"action": action}).json()

    def state(self) -> dict:
        return requests.get(f"{self.base_url}/state", timeout=self.timeout).json()

    def health(self) -> dict:
        return requests.get(f"{self.base_url}/health", timeout=self.timeout).json()

    def _post(self, path: str, payload: dict[str, Any]):
        return requests.post(f"{self.base_url}{path}", json=payload, timeout=self.timeout)
