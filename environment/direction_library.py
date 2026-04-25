from __future__ import annotations

import json
import math
import os
import pickle
import random
from typing import Any


DIRECTION_LABELS = [
    "authority",
    "refusal_suppression",
    "trust_escalation",
    "danger",
    "compliance",
    "distraction",
]

INTENT_CLASSES = ["benign", "escalation", "extraction", "distraction"]


class DirectionLibrary:
    def __init__(
        self,
        library_path: str | None = None,
        probe_path: str | None = None,
        hidden_size: int = 1024,
    ) -> None:
        tools_dir = os.getenv("SEIGE_WHITEBOX_TOOLS_DIR", "data/whitebox_tools")
        if library_path is None:
            candidate = os.path.join(tools_dir, "direction_library.json")
            library_path = candidate if os.path.exists(candidate) else "data/direction_library.json"
        if probe_path is None:
            candidate = os.path.join(tools_dir, "intent_probes.pkl")
            probe_path = candidate if os.path.exists(candidate) else "data/intent_probes.pkl"
        self.hidden_size = hidden_size
        self._vectors: dict[str, list[float]] = {}
        self._intent_probes: dict[int, Any] = {}
        if os.path.exists(library_path):
            self._load_vectors(library_path)
        else:
            self._init_random_vectors()
        if os.path.exists(probe_path):
            with open(probe_path, "rb") as handle:
                self._intent_probes = pickle.load(handle)

    def _load_vectors(self, path: str) -> None:
        with open(path) as handle:
            data = json.load(handle)
        self._vectors = {label: [float(x) for x in vec] for label, vec in data.items()}

    def _init_random_vectors(self) -> None:
        rng = random.Random(1337)
        for label in DIRECTION_LABELS:
            vec = [rng.gauss(0.0, 1.0) for _ in range(self.hidden_size)]
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            self._vectors[label] = [x / norm for x in vec]

    def get_vector(self, label: str) -> list[float] | None:
        return self._vectors.get(label)

    def all_vectors(self) -> dict[str, list[float]]:
        return dict(self._vectors)

    def run_intent_probe(self, activation: list[float], layer: int) -> tuple[str, float]:
        probe = self._intent_probes.get(layer)
        if probe is None:
            return ("benign", 0.5)
        probs = probe.predict_proba([activation])[0]
        idx = max(range(len(probs)), key=lambda i: probs[i])
        return (INTENT_CLASSES[idx], float(probs[idx]))

    def save(self, library_path: str, probe_path: str) -> None:
        os.makedirs(os.path.dirname(library_path), exist_ok=True)
        with open(library_path, "w") as handle:
            json.dump(self._vectors, handle)
        with open(probe_path, "wb") as handle:
            pickle.dump(self._intent_probes, handle)
