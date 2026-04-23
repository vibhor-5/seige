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
    args = parser.parse_args()
    library = DirectionLibrary(
        library_path="",
        probe_path="",
        hidden_size=args.hidden_size,
    )
    library.save(args.library_path, args.probe_path)
    print(f"wrote {args.library_path} and {args.probe_path}")


if __name__ == "__main__":
    main()
