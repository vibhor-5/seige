#!/usr/bin/env python3
"""Upload a PEFT/LoRA adapter directory to a Hugging Face model repo (one subfolder per cycle)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create HF model repo if missing, then upload a folder to path_in_repo.",
    )
    parser.add_argument(
        "--local-dir",
        required=True,
        help="Local directory (e.g. archived final_adapter for one cycle).",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Model repo id, e.g. vibhor-5/seige-grpo-checkpoints",
    )
    parser.add_argument(
        "--path-in-repo",
        required=True,
        help="Subfolder in the repo, e.g. red_cycle_1 or blue_cycle_1",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="If set, create the repo as private (only applies on first create).",
    )
    args = parser.parse_args()

    local = Path(args.local_dir).resolve()
    if not local.is_dir():
        print(f"ERROR: {local} is not a directory", file=sys.stderr)
        return 1

    token = _token()
    if not token:
        print("ERROR: Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN", file=sys.stderr)
        return 1

    pin = args.path_in_repo.strip().strip("/")
    if not pin:
        print("ERROR: --path-in-repo must be non-empty", file=sys.stderr)
        return 1

    from huggingface_hub import HfApi, upload_folder

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        private=bool(args.private),
        exist_ok=True,
        repo_type="model",
    )
    upload_folder(
        folder_path=str(local),
        path_in_repo=pin,
        repo_id=args.repo_id,
        token=token,
    )
    print(f"Uploaded {local} -> {args.repo_id} / {pin}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
