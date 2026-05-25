from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiments.workflow_profiles import get_workflow_profile
from workflows.single_llm_pbrs_v2_baseline import default_single_llm_baseline_profile
from workflows.single_llm_pbrs_v2_baseline import run_single_llm_pbrs_v2_baseline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow-id", required=True)
    parser.add_argument("--workflow-profile", "--profile", default="qmix_single_llm_pbrs_v2_reward_gen_seed1")
    parser.add_argument("--source-winner-workflow-dir", default=None)
    parser.add_argument("--results-root", default=None)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--candidate-budget-steps", type=int, default=None)
    parser.add_argument("--full-t-max", "--formal-t-max", dest="full_t_max", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--sparse-run-dir", default=None)
    parser.add_argument("--use-cuda", dest="use_cuda", action="store_true")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false")
    parser.add_argument("--use-real-llm", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tiny-smoke", action="store_true")
    parser.add_argument("--winner-only-formal", action="store_true")
    parser.add_argument("--consume-winner-plan", action="store_true")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--format", choices=["text", "json"], default="text")
    parser.set_defaults(use_cuda=None)
    return parser


def build_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    try:
        profile = deepcopy(get_workflow_profile(args.workflow_profile))
    except Exception:
        profile = deepcopy(default_single_llm_baseline_profile())
    merged = deepcopy(default_single_llm_baseline_profile())
    merged.update(profile)
    merged["workflow_profile"] = str(args.workflow_profile)
    merged["workflow_id"] = str(args.workflow_id)
    merged["python_executable"] = str(args.python_executable)
    if args.source_winner_workflow_dir is not None:
        merged["source_winner_workflow_dir"] = str(args.source_winner_workflow_dir)
    if args.results_root is not None:
        merged["results_root"] = str(args.results_root)
    if args.seed is not None:
        merged["seed"] = int(args.seed)
    if args.candidate_budget_steps is not None:
        merged["candidate_budget_steps"] = int(args.candidate_budget_steps)
    if args.full_t_max is not None:
        merged["full_training_t_max"] = int(args.full_t_max)
    if args.model is not None:
        merged["model"] = str(args.model)
    if args.api_key_env is not None:
        merged["api_key_env"] = str(args.api_key_env)
    if args.base_url is not None:
        merged["base_url"] = str(args.base_url)
    if args.sparse_run_dir is not None:
        merged["sparse_run_dir"] = str(args.sparse_run_dir)
    if args.use_cuda is not None:
        merged["use_cuda"] = bool(args.use_cuda)
    merged["use_real_llm"] = bool(args.use_real_llm)
    merged["execute"] = bool(args.execute)
    merged["dry_run"] = bool(args.dry_run)
    merged["tiny_smoke"] = bool(args.tiny_smoke)
    merged["winner_only_formal"] = bool(args.winner_only_formal or args.consume_winner_plan)
    if merged["tiny_smoke"]:
        merged["candidate_budget_steps"] = int(
            args.candidate_budget_steps or min(2000, int(merged.get("candidate_budget_steps") or 2000))
        )
        merged["full_training_t_max"] = int(
            args.full_t_max or min(4000, int(merged.get("full_training_t_max") or 4000))
        )
    if not merged["execute"] and not merged["dry_run"] and not merged["tiny_smoke"] and not merged["use_real_llm"]:
        merged["dry_run"] = True
    if merged["winner_only_formal"] and not merged["execute"]:
        merged["dry_run"] = True
    return merged


def _print_text(payload: Dict[str, Any]) -> None:
    manifest = payload.get("manifest") or {}
    config_summary = manifest.get("config_summary") or {}
    metrics = payload.get("metrics_summary") or {}
    print(
        f"{payload.get('workflow_id')}: status={manifest.get('status')} "
        f"profile={manifest.get('workflow_profile')} "
        f"env={config_summary.get('env_key')} algo={config_summary.get('train_config')}"
    )
    print(
        "  "
        f"candidate_budget_steps={config_summary.get('candidate_budget_steps')} "
        f"full_training_t_max={config_summary.get('full_training_t_max')} "
        f"eval_use_pbrs={config_summary.get('eval_use_pbrs')} "
        f"test_sparse_only={config_summary.get('test_sparse_only')}"
    )
    print(
        "  "
        f"winner_candidate_id={metrics.get('winner_candidate_id')} "
        f"rule_based={metrics.get('rule_based_winner_selection')} "
        f"workflow_dir={payload.get('workflow_dir')}"
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    config = build_config_from_args(args)
    payload = run_single_llm_pbrs_v2_baseline(config)
    if args.output_path:
        Path(args.output_path).write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        return
    _print_text(payload)


if __name__ == "__main__":
    main()
