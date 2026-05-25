from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from workflows.adaptive_checkpoint_replacement import (
    DEFAULT_RESULTS_ROOT,
    build_adaptive_checkpoint_replacement_manifest,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow-id", required=True)
    parser.add_argument("--stage-selection-result-json", required=True)
    parser.add_argument("--dense-reference-run-dir", required=True)
    parser.add_argument("--dense-reference-selection-json", required=True)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--branch-budget-steps", type=int, default=300000)
    parser.add_argument("--max-parallel-candidates", type=int, default=2)
    parser.add_argument("--cpu", dest="branch_use_cuda", action="store_false")
    parser.add_argument("--use-cuda", dest="branch_use_cuda", action="store_true")
    parser.set_defaults(branch_use_cuda=None)
    parser.add_argument("--mainline-use-cuda", dest="mainline_use_cuda", action="store_true")
    parser.add_argument("--mainline-cpu", dest="mainline_use_cuda", action="store_false")
    parser.set_defaults(mainline_use_cuda=None)
    parser.add_argument("--continuation-save-model-interval", type=int, default=300000)
    parser.add_argument("--t-max", type=int, default=2050000)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--loop-until-complete", action="store_true")
    parser.add_argument("--poll-interval-seconds", type=float, default=30.0)
    parser.add_argument("--use-real-llm", action="store_true")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--model", default="gpt-5.2")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--llm-timeout", type=float, default=60.0)
    parser.add_argument("--llm-retry-count", type=int, default=2)
    parser.add_argument("--llm-retry-backoff", type=float, default=5.0)
    parser.add_argument("--mock-llm-mode", action="store_true")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--format", choices=["text", "json"], default="text")
    return parser


def _run_once(args: argparse.Namespace) -> dict:
    stage_selection_result = json.loads(Path(args.stage_selection_result_json).read_text(encoding="utf-8"))
    dense_reference_selection_payload = json.loads(Path(args.dense_reference_selection_json).read_text(encoding="utf-8"))
    dense_reference_selection = deepcopy(
        dense_reference_selection_payload.get("dense_reference_selection") or dense_reference_selection_payload
    )
    adaptive_method = {
        "candidate_generation": {
            "beta_values": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
            "wc_values": [0.1, 0.3, 0.5, 0.7, 0.9],
            "beta_neighbor_radius": 1,
            "wc_neighbor_radius": 1,
            "include_diagonal_pairs": True,
            "include_no_change": True,
            "max_candidates_per_checkpoint": 9,
        },
        "use_llm_stage3_candidate_generation": True,
        "use_llm_stage3_result_diagnosis": True,
        "llm_stage3_max_candidates": 9,
        "llm_stage3_retry_count": args.llm_retry_count,
        "llm_stage3_use_cache": True,
        "enforce_recovery_candidate_when_underperforming": True,
        "llm_candidate_repair_retry_count": 1,
        "add_deterministic_recovery_fallback": True,
        "fallback_to_deterministic_on_llm_error": True,
        "launch_first_change_fixed_initial_baseline": False,
        "first_change_baseline_non_blocking": True,
        "mock_llm_mode": bool(args.mock_llm_mode),
        "decision_rule": {
            "primary_metric": "best_test_sparse_return_mean",
            "secondary_metric": "last_test_sparse_return_mean",
            "min_improvement": 0.03,
            "fallback_to_no_change": True,
        },
    }
    manifest = build_adaptive_checkpoint_replacement_manifest(
        workflow_id=args.workflow_id,
        stage_selection_result=stage_selection_result,
        dense_reference_run_dir=args.dense_reference_run_dir,
        dense_reference_selection=dense_reference_selection,
        python_executable=args.python_executable,
        execute=bool(args.execute),
        storage_root=args.results_root,
        branch_budget_steps=args.branch_budget_steps,
        max_parallel_candidates=args.max_parallel_candidates,
        branch_use_cuda=args.branch_use_cuda,
        mainline_use_cuda=args.mainline_use_cuda,
        continuation_save_model_interval=args.continuation_save_model_interval,
        t_max=args.t_max,
        adaptive_method=adaptive_method,
        use_real_llm=bool(args.use_real_llm),
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        llm_timeout=args.llm_timeout,
        llm_retry_count=args.llm_retry_count,
        llm_retry_backoff=args.llm_retry_backoff,
    )
    return {
        "workflow_id": args.workflow_id,
        "adaptive_checkpoint_replacement_manifest": manifest,
        "artifacts": {
            "manifest_json": str(Path(args.results_root) / args.workflow_id / "adaptive_checkpoint_replacement_manifest.json"),
            "final_spec_json": ((manifest.get("artifacts") or {}).get("adaptive_final_spec_json")),
            "final_mainline_run_json": ((manifest.get("artifacts") or {}).get("final_mainline_run_json")),
        },
    }


def _ready(payload: dict) -> bool:
    readiness = ((payload.get("adaptive_checkpoint_replacement_manifest") or {}).get("readiness") or {})
    return bool(readiness.get("all_decisions_completed")) and bool(readiness.get("final_mainline_completed"))


def _print_text(payload: dict) -> None:
    manifest = payload.get("adaptive_checkpoint_replacement_manifest") or {}
    decisions = list(manifest.get("decisions") or [])
    readiness = manifest.get("readiness") or {}
    print(
        f"{payload['workflow_id']}: decisions={len(decisions)} "
        f"completed={sum(1 for d in decisions if d.get('completed'))} "
        f"ready={readiness.get('all_decisions_completed')}"
    )
    for decision in decisions:
        continuation = decision.get("continuation") or {}
        print(
            "  "
            f"decision_{decision.get('decision_index')} stage={decision.get('stage_label')} "
            f"winner={((decision.get('winner_config') or {}).get('beta'), (decision.get('winner_config') or {}).get('wc'))} "
            f"continuation={continuation.get('status')} run_id={continuation.get('run_id')}"
        )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    payload = _run_once(args)
    if args.loop_until_complete and args.execute:
        while not _ready(payload):
            time.sleep(max(1.0, float(args.poll_interval_seconds)))
            payload = _run_once(args)
    if args.output_path:
        Path(args.output_path).write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        _print_text(payload)


if __name__ == "__main__":
    main()
