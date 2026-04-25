from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environment.direction_library import DirectionLibrary


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute seige direction vectors.")
    parser.add_argument("--library-path", default="data/direction_library.json")
    parser.add_argument("--probe-path", default="data/intent_probes.pkl")
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--mode", choices=["mock", "hf"], default="mock")
    parser.add_argument("--model-id", default="google/gemma-4-E2B")
    args = parser.parse_args()

    if args.mode == "mock":
        library = DirectionLibrary(
            library_path="",
            probe_path="",
            hidden_size=args.hidden_size,
        )
        library.save(args.library_path, args.probe_path)
        print(f"Saved random direction vectors (mock mode) to {args.library_path} and {args.probe_path}")
    else:
        _precompute_real_directions(args.model_id, args.library_path, args.probe_path)

def _precompute_real_directions(model_id: str, library_path: str, probe_path: str) -> None:
    print(f"Precomputing real directions for {model_id} (not fully implemented).")
    # Real contrastive extraction — implement from design doc
    # CONTRASTIVE_PAIRS, INTENT_EXAMPLES, get_layer_activations(), etc.
    pass


if __name__ == "__main__":
    main()
