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

from workflows.baseline_run_support import inspect_baseline_checkpoint_readiness
from workflows.final_stage_conditioned_spec import synthesize_final_stage_conditioned_spec
from workflows.stage_conditioned_branching import (
    DEFAULT_RESULTS_ROOT,
    build_stage_conditioned_branching_manifest,
)
from workflows.stage_selection import build_stage_selection_result
from workflows.stage_selection import StageSelectionError


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow-id", required=True)
    parser.add_argument(
        "--stage-selection-result-json",
        default=None,
        help=(
            "Existing strict Stage 2 result JSON. When provided, Stage 2 is skipped and "
            "Stage 3 uses this selection directly."
        ),
    )
    parser.add_argument("--baseline-run-dir", default=None)
    parser.add_argument("--baseline-result-json", default=None)
    parser.add_argument("--branch-source-run-dir", default=None)
    parser.add_argument("--branch-source-result-json", default=None)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--reward-paradigm", choices=["pbrs"], default="pbrs")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--max-count", type=int, default=3)
    parser.add_argument("--min-step-gap", type=int, default=50000)
    parser.add_argument("--require-full-baseline", action="store_true", default=True)
    parser.add_argument("--expected-full-t-max", type=int, default=2050000)
    parser.add_argument("--max-points-per-metric", type=int, default=15)
    parser.add_argument(
        "--field-round",
        action="append",
        default=None,
        help=(
            "Field round(s) to execute, in order. Defaults to beta,wc. "
            "Repeat to pass multiple values, for example --field-round beta --field-round wc."
        ),
    )
    parser.add_argument(
        "--branch-budget-steps",
        type=int,
        default=None,
        help=(
            "Fixed continuation budget for each Stage 3 branch run. "
            "Defaults to one tenth of the full baseline t_max."
        ),
    )
    parser.add_argument(
        "--max-parallel-candidates",
        type=int,
        default=2,
        help=(
            "Maximum number of Stage 3 branch candidates to execute in parallel within the "
            "same field round."
        ),
    )
    parser.add_argument("--cpu", dest="branch_use_cuda", action="store_false")
    parser.add_argument("--use-cuda", dest="branch_use_cuda", action="store_true")
    parser.set_defaults(branch_use_cuda=None)
    parser.add_argument("--native-original-pbrs-branching", action="store_true")
    parser.add_argument("--eval-use-pbrs", action="store_true")
    parser.add_argument("--preferred-metric", action="append", default=None)
    parser.add_argument(
        "--checkpoint-optional-metric",
        action="append",
        default=None,
        help=(
            "Optional additional metrics to attach to each checkpoint candidate before sending "
            "them to the Stage 2 LLM. Sparse return metrics are always included."
        ),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--reuse-completed-candidates", action="store_true")
    parser.add_argument("--use-real-llm", action="store_true")
    parser.add_argument(
        "--use-real-llm-for-field-analysis",
        action="store_true",
        help=(
            "After each Stage 3 field round finishes, summarize the branch results and "
            "ask the real LLM to choose one candidate value per stage."
        ),
    )
    parser.add_argument("--api-key-env", default="IUSEAPI_API_KEY")
    parser.add_argument("--base-url", default="https://www.iuseapi.com/v1")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--llm-timeout", type=float, default=60.0)
    parser.add_argument("--llm-max-retries", type=int, default=3)
    parser.add_argument("--llm-retry-backoff", type=float, default=5.0)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--format", choices=["text", "json"], default="text")
    parser.add_argument(
        "--loop-until-complete",
        action="store_true",
        help=(
            "Keep supervising Stage 3 until the branching manifest reports "
            "ready_for_final_synthesis. Each loop runs one reconcile/dispatch tick, "
            "then sleeps before checking again."
        ),
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=15.0,
        help="Sleep interval between Stage 3 supervisor ticks when --loop-until-complete is enabled.",
    )
    return parser


def _fmt(value):
    return "-" if value is None else str(value)


def run_stage_conditioned_branching(args: argparse.Namespace) -> dict:
    if args.stage_selection_result_json is None and not args.use_real_llm:
        raise ValueError(
            "The new stage-conditioned workflow requires --use-real-llm for checkpoint stage selection. "
            "Heuristic fallback is disabled by design."
        )

    workflow_dir = Path(args.results_root) / args.workflow_id
    workflow_dir.mkdir(parents=True, exist_ok=True)
    if args.stage_selection_result_json:
        selection_result_path = Path(args.stage_selection_result_json)
        selection_result = json.loads(selection_result_path.read_text(encoding="utf-8"))
        readiness = inspect_baseline_checkpoint_readiness(
            baseline_run_dir=(
                args.branch_source_run_dir
                or str(Path(selection_result["baseline_run_config_json"]).parent)
            ),
            baseline_result_json=args.branch_source_result_json,
        )
        selection_result = _override_branch_source(
            selection_result=selection_result,
            readiness=readiness,
        )
    else:
        readiness = inspect_baseline_checkpoint_readiness(
            baseline_run_dir=args.baseline_run_dir,
            baseline_result_json=args.baseline_result_json,
        )
        baseline_t_max = int((readiness.get("config_summary") or {}).get("t_max") or 0)
        if args.require_full_baseline and baseline_t_max != int(args.expected_full_t_max):
            raise ValueError(
                "stage-conditioned workflow requires a full sparse baseline before Stage 2. "
                f"Expected t_max={int(args.expected_full_t_max)}, got {baseline_t_max}."
            )
        if not readiness.get("branching_ready"):
            raise ValueError(
                "baseline is not ready for stage-conditioned branching: "
                + ", ".join(readiness.get("blockers") or ["unknown_blocker"])
            )

        baseline_source = readiness["baseline_source"]
        try:
            selection_result = build_stage_selection_result(
                workflow_id=args.workflow_id,
                baseline_metrics_json=baseline_source["baseline_metrics_json"],
                baseline_run_config_json=baseline_source["baseline_run_config_json"],
                baseline_run_info_json=baseline_source["baseline_run_info_json"],
                reward_paradigm=args.reward_paradigm,
                max_rounds=args.max_rounds,
                min_count=args.min_count,
                max_count=args.max_count,
                min_step_gap=args.min_step_gap,
                max_points_per_metric=args.max_points_per_metric,
                preferred_metrics=args.preferred_metric,
                checkpoint_optional_metrics=args.checkpoint_optional_metric,
                use_real_llm=args.use_real_llm,
                api_key_env=args.api_key_env,
                base_url=args.base_url,
                model=args.model,
                temperature=args.temperature,
                llm_timeout=args.llm_timeout,
                llm_max_retries=args.llm_max_retries,
                llm_retry_backoff=args.llm_retry_backoff,
            )
        except StageSelectionError as exc:
            failure_payload = {
                "workflow_id": args.workflow_id,
                "baseline_readiness": readiness,
                "error": str(exc),
                "stage": "stage_selection",
                "details": exc.payload,
            }
            (workflow_dir / "stage_selection_failure.json").write_text(
                json.dumps(failure_payload, indent=2, sort_keys=True, ensure_ascii=False),
                encoding="utf-8",
            )
            raise
    # Save Stage 2 output immediately so longer Stage 3 execution can resume
    # from a completed checkpoint selection after interruptions.
    (workflow_dir / "stage_selection_result.json").write_text(
        json.dumps(selection_result, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    manifest = build_stage_conditioned_branching_manifest(
        workflow_id=args.workflow_id,
        stage_selection_result=selection_result,
        python_executable=args.python_executable,
        execute=args.execute,
        reuse_completed_candidates=args.reuse_completed_candidates,
        storage_root=args.results_root,
        field_rounds=args.field_round,
        branch_budget_steps=args.branch_budget_steps,
        max_parallel_candidates=args.max_parallel_candidates,
        branch_use_cuda=args.branch_use_cuda,
        native_original_pbrs_branching=args.native_original_pbrs_branching,
        eval_use_pbrs=args.eval_use_pbrs,
        use_real_llm_for_field_analysis=args.use_real_llm_for_field_analysis,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        llm_timeout=args.llm_timeout,
        llm_max_retries=args.llm_max_retries,
        llm_retry_backoff=args.llm_retry_backoff,
    )
    final_spec = None
    final_spec_error = None
    try:
        final_spec = synthesize_final_stage_conditioned_spec(manifest)
    except Exception as exc:  # keep Stage 3 execution artifacts even when final synthesis is deferred
        final_spec_error = str(exc)
    if final_spec is not None:
        (workflow_dir / "final_stage_conditioned_reward_spec.json").write_text(
            json.dumps(final_spec, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
    payload = {
        "workflow_id": args.workflow_id,
        "baseline_readiness": readiness,
        "stage_selection_result": selection_result,
        "stage_conditioned_branching_manifest": manifest,
        "final_stage_conditioned_reward_spec": final_spec,
        "final_stage_conditioned_reward_spec_error": final_spec_error,
        "artifacts": {
            "workflow_dir": str(workflow_dir),
            "stage_selection_result_json": str(workflow_dir / "stage_selection_result.json"),
            "branching_manifest_json": str(workflow_dir / "stage_conditioned_branching_manifest.json"),
            "final_stage_conditioned_reward_spec_json": (
                str(workflow_dir / "final_stage_conditioned_reward_spec.json")
                if final_spec is not None
                else None
            ),
        },
    }
    return payload


def _override_branch_source(
    *,
    selection_result: dict,
    readiness: dict,
) -> dict:
    overridden = deepcopy(selection_result)
    baseline_source = readiness.get("baseline_source") or {}
    checkpoint_summary = readiness.get("checkpoint_summary") or {}

    overridden["baseline_run_config_json"] = baseline_source.get(
        "baseline_run_config_json",
        overridden.get("baseline_run_config_json"),
    )
    overridden["baseline_run_info_json"] = baseline_source.get(
        "baseline_run_info_json",
        overridden.get("baseline_run_info_json"),
    )
    overridden["baseline_metrics_json"] = baseline_source.get(
        "baseline_metrics_json",
        overridden.get("baseline_metrics_json"),
    )
    overridden["baseline_checkpoint_root_dir"] = checkpoint_summary.get(
        "checkpoint_root_dir",
        overridden.get("baseline_checkpoint_root_dir"),
    )

    available_steps = list(
        checkpoint_summary.get("available_checkpoint_steps")
        or overridden.get("baseline_available_checkpoint_steps")
        or []
    )
    overridden["baseline_available_checkpoint_steps"] = available_steps

    selected = list(((overridden.get("selection") or {}).get("selected_checkpoints")) or [])
    if available_steps and selected:
        for item in selected:
            try:
                original_step = int(item.get("step"))
            except (TypeError, ValueError):
                continue
            aligned_step = min(available_steps, key=lambda value: abs(int(value) - original_step))
            item["step"] = int(aligned_step)
            item["name"] = f"ckpt_{int(aligned_step):07d}_{item.get('stage_label')}"
    return overridden


def print_text_payload(payload: dict) -> None:
    readiness = payload.get("baseline_readiness") or {}
    final_spec = payload.get("final_stage_conditioned_reward_spec") or {}
    manifest = payload.get("stage_conditioned_branching_manifest") or {}
    print(
        f"{payload['workflow_id']}: branching_ready={_fmt(readiness.get('branching_ready'))} "
        f"field_rounds={len(manifest.get('rounds', []))}"
    )
    for round_payload in manifest.get("rounds", []):
        recommendation = (round_payload.get("field_recommendation") or {}).get("recommended_values_by_stage") or {}
        field_round_summary = round_payload.get("field_round_summary") or {}
        synthesis_ready = (field_round_summary.get("readiness") or {}).get("ready_for_final_synthesis")
        print(
            "  "
            f"round_{round_payload.get('round_id'):02d} field={_fmt(round_payload.get('field_name'))} "
            f"recommended_values_by_stage={json.dumps(recommendation, ensure_ascii=False)} "
            f"ready_for_final_synthesis={_fmt(synthesis_ready)}"
        )
    print(
        "  "
        f"manifest_ready_for_final_synthesis={_fmt((manifest.get('readiness') or {}).get('ready_for_final_synthesis'))} "
        f"final_spec_path={_fmt((payload.get('artifacts') or {}).get('final_stage_conditioned_reward_spec_json'))}"
    )


def _stage3_ready(payload: dict) -> bool:
    manifest = payload.get("stage_conditioned_branching_manifest") or {}
    readiness = manifest.get("readiness") or {}
    return bool(readiness.get("ready_for_final_synthesis"))


def _run_stage3_supervisor(args: argparse.Namespace) -> dict:
    last_payload = {}
    while True:
        payload = run_stage_conditioned_branching(args)
        last_payload = payload
        if _stage3_ready(payload):
            return payload
        if args.format != "json":
            print(
                f"[stage3-supervisor] not ready yet; sleeping {args.poll_interval_seconds:.1f}s "
                "before next reconcile tick",
                flush=True,
            )
        time.sleep(max(1.0, float(args.poll_interval_seconds)))


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.loop_until_complete:
        payload = _run_stage3_supervisor(args)
    else:
        payload = run_stage_conditioned_branching(args)
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        return
    print_text_payload(payload)


if __name__ == "__main__":
    main()
