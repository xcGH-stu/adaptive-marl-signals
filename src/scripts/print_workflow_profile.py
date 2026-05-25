from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiments.workflow_profiles import get_workflow_profile
from experiments.workflow_profiles import list_workflow_profiles


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow-profile", choices=list_workflow_profiles(), required=True)
    parser.add_argument("--format", choices=["json", "text"], default="text")
    return parser


def _print_text(profile_name: str, profile: dict) -> None:
    resource_policy = profile.get("resource_policy") or {}
    checkpointing_strategy = profile.get("checkpointing_strategy") or {}
    recovery_policy = profile.get("recovery_policy") or {}
    print(
        f"{profile_name}: env={profile.get('env_key')} algorithm={profile.get('train_config')} "
        f"t_max={profile.get('t_max')}"
    )
    print(
        "  "
        f"use_cuda={profile.get('use_cuda')} "
        f"stage3_use_cuda={profile.get('stage3_use_cuda')} "
        f"max_parallel_candidates={profile.get('max_parallel_candidates')}"
    )
    print(
        "  "
        f"field_rounds={profile.get('field_rounds')} "
        f"branch_budget_steps={profile.get('branch_budget_steps')}"
    )
    print(
        "  "
        f"llm_model={profile.get('model')} "
        f"native_original_pbrs_branching={profile.get('native_original_pbrs_branching')}"
    )
    print(
        "  "
        f"resource_policy={resource_policy} "
        f"checkpointing_strategy={checkpointing_strategy}"
    )
    print("  " f"recovery_policy={recovery_policy}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    profile = get_workflow_profile(args.workflow_profile)
    if args.format == "json":
        print(json.dumps(profile, indent=2, sort_keys=True, ensure_ascii=False))
        return
    _print_text(args.workflow_profile, profile)


if __name__ == "__main__":
    main()
