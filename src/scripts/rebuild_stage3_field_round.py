from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from workflows.stage_conditioned_branching import (
    REQUIRED_STAGE_LABELS,
    _analyze_field_round_with_llm,
    _build_field_round_summary,
    _build_result_record,
    _build_stage_comparison_table,
    _choose_ranking_metric,
)
from workflows.storage import WorkflowStorage


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow-id", required=True)
    parser.add_argument("--stage-selection-result-json", required=True)
    parser.add_argument("--field-name", required=True)
    parser.add_argument("--round-id", type=int, required=True)
    parser.add_argument("--results-root", default="results/stage_conditioned_workflows")
    parser.add_argument("--use-real-llm", action="store_true")
    parser.add_argument("--api-key-env", default="IUSEAPI_API_KEY")
    parser.add_argument("--base-url", default="https://www.iuseapi.com/v1")
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--llm-timeout", type=float, default=60.0)
    parser.add_argument("--llm-max-retries", type=int, default=3)
    parser.add_argument("--llm-retry-backoff", type=float, default=5.0)
    return parser


def _candidate_id(stage_label: str, field_name: str, value: Any) -> str:
    if isinstance(value, float):
        suffix = str(value).replace(".", "p")
    else:
        suffix = str(value)
    return f"{stage_label}_{field_name}_{suffix}"


def _extract_candidate_values(stage_dir: Path, field_name: str, stage_label: str) -> List[Any]:
    values = []
    prefix = f"{stage_label}_{field_name}_"
    for child in sorted(stage_dir.iterdir()):
        if not child.is_dir():
            continue
        if not child.name.startswith(prefix):
            continue
        raw = child.name[len(prefix):].replace("p", ".")
        try:
            values.append(float(raw))
        except ValueError:
            pass
    return values


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rebuild_field_round(args: argparse.Namespace) -> Dict[str, Any]:
    workflow_dir = ROOT / args.results_root / args.workflow_id
    if not workflow_dir.exists():
        raise FileNotFoundError(f"workflow directory does not exist: {workflow_dir}")

    stage_selection_result = _load_json(Path(args.stage_selection_result_json))
    selected_checkpoints = list((stage_selection_result.get("selection") or {}).get("selected_checkpoints", []))
    selected_by_label = {str(item["stage_label"]): item for item in selected_checkpoints}

    round_dir = workflow_dir / f"round_{args.round_id:02d}_{args.field_name}"
    stage_runs = []
    shared_candidate_values: List[Any] = []
    recommended_values_by_stage: Dict[str, Any] = {}

    for stage_label in REQUIRED_STAGE_LABELS:
        checkpoint = selected_by_label[stage_label]
        checkpoint_name = str(checkpoint["name"])
        stage_dir = round_dir / checkpoint_name
        if not stage_dir.exists():
            raise FileNotFoundError(f"stage directory does not exist: {stage_dir}")
        candidate_values = _extract_candidate_values(stage_dir, args.field_name, stage_label)
        if not shared_candidate_values:
            shared_candidate_values = list(candidate_values)
        records = []
        candidates = []
        for candidate_value in candidate_values:
            candidate_id = _candidate_id(stage_label, args.field_name, candidate_value)
            candidate_dir = stage_dir / candidate_id
            branch_result_path = candidate_dir / "branch_result.json"
            payload = {
                "candidate_id": candidate_id,
                "candidate_value": candidate_value,
                "reward_spec_path": str(candidate_dir / "reward_spec.json"),
                "reward_function_path": str(candidate_dir / "reward_function.py"),
            }
            if branch_result_path.exists():
                branch_result = _load_json(branch_result_path)
                payload["branch_result"] = branch_result
                payload["branch_result_status"] = "reused"
                records.append(
                    _build_result_record(
                        field_name=args.field_name,
                        stage_label=stage_label,
                        checkpoint=checkpoint,
                        candidate_id=candidate_id,
                        candidate_value=candidate_value,
                        branch_result=branch_result,
                    )
                )
            else:
                payload["branch_result_status"] = "missing"
            candidates.append(payload)

        ranking_metric = _choose_ranking_metric(records)
        best_candidate = None
        if records:
            if ranking_metric is None:
                best_candidate = records[0]
            else:
                best_candidate = max(
                    records,
                    key=lambda item: (
                        item.get(ranking_metric) is not None,
                        item.get(ranking_metric) or float("-inf"),
                    ),
                )
        if best_candidate is not None:
            recommended_values_by_stage[stage_label] = best_candidate.get("candidate_value")

        stage_runs.append(
            {
                "checkpoint": checkpoint,
                "field_name": args.field_name,
                "candidate_values": candidate_values,
                "candidate_count": len(candidate_values),
                "ranking_metric": ranking_metric,
                "best_candidate": best_candidate,
                "comparison_table": _build_stage_comparison_table(records),
                "records": records,
                "candidates": candidates,
            }
        )

    field_critic_result = None
    if args.use_real_llm:
        field_critic_result = _analyze_field_round_with_llm(
            round_id=args.round_id,
            field_name=args.field_name,
            shared_candidate_values=shared_candidate_values,
            stage_runs=stage_runs,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            model=args.model,
            temperature=args.temperature,
            llm_timeout=args.llm_timeout,
            llm_max_retries=args.llm_max_retries,
            llm_retry_backoff=args.llm_retry_backoff,
        )
        parsed = ((field_critic_result.get("response") or {}).get("parsed") or {})
        llm_recommended_values = parsed.get("recommended_values_by_stage") or {}
        recommended_values_by_stage = {
            stage_label: llm_recommended_values[stage_label]
            for stage_label in REQUIRED_STAGE_LABELS
        }

    field_round_summary = _build_field_round_summary(
        round_id=args.round_id,
        field_name=args.field_name,
        selected_checkpoints=selected_checkpoints,
        shared_candidate_values=shared_candidate_values,
        stage_runs=stage_runs,
        recommended_values_by_stage=recommended_values_by_stage,
    )

    round_payload = {
        "round_id": args.round_id,
        "field_name": args.field_name,
        "shared_candidate_values": shared_candidate_values,
        "stage_coverage": [str(checkpoint.get("stage_label")) for checkpoint in selected_checkpoints],
        "stage_runs": stage_runs,
        "candidate_count_by_stage": {
            str(stage_run.get("checkpoint", {}).get("stage_label")): stage_run.get("candidate_count")
            for stage_run in stage_runs
        },
        "field_recommendation": {
            "field_name": args.field_name,
            "recommended_values_by_stage": recommended_values_by_stage,
            "source": "real_llm" if field_critic_result is not None else "best_metric",
        },
        "field_round_summary": field_round_summary,
        "field_critic_result": field_critic_result,
    }

    storage = WorkflowStorage.load(args.workflow_id, root_dir=ROOT / args.results_root)
    storage.save_workflow_json(f"field_round_{args.round_id:02d}_{args.field_name}.json", round_payload)
    storage.save_workflow_json(
        f"field_round_{args.round_id:02d}_{args.field_name}_comparison.json",
        {
            "round_id": args.round_id,
            "field_name": args.field_name,
            "stage_comparisons": [
                {
                    "stage_label": stage_run.get("checkpoint", {}).get("stage_label"),
                    "checkpoint_name": stage_run.get("checkpoint", {}).get("name"),
                    "checkpoint_step": stage_run.get("checkpoint", {}).get("step"),
                    "ranking_metric": stage_run.get("ranking_metric"),
                    "best_candidate": stage_run.get("best_candidate"),
                    "comparison_table": stage_run.get("comparison_table"),
                }
                for stage_run in stage_runs
            ],
            "field_recommendation": round_payload.get("field_recommendation"),
        },
    )
    storage.save_workflow_json(
        f"field_round_{args.round_id:02d}_{args.field_name}_summary.json",
        field_round_summary,
    )
    if field_critic_result is not None:
        storage.save_workflow_json(
            f"field_round_{args.round_id:02d}_{args.field_name}_critic_response.json",
            field_critic_result,
        )
        storage.save_text(
            args.round_id,
            f"field_round_{args.round_id:02d}_{args.field_name}_critic_prompt.txt",
            field_critic_result["prompt"],
        )

    manifest_path = workflow_dir / "stage_conditioned_branching_manifest.json"
    if manifest_path.exists():
        manifest = _load_json(manifest_path)
    else:
        manifest = {"workflow_id": args.workflow_id, "rounds": []}
    rounds = [round_payload_existing for round_payload_existing in manifest.get("rounds", []) if not (
        int(round_payload_existing.get("round_id") or -1) == int(args.round_id)
        and str(round_payload_existing.get("field_name")) == str(args.field_name)
    )]
    rounds.append(round_payload)
    rounds = sorted(rounds, key=lambda item: int(item.get("round_id") or 0))
    manifest["rounds"] = rounds
    manifest["field_round_summaries"] = [item.get("field_round_summary") for item in rounds]
    manifest["readiness"] = {
        "field_round_count": len(rounds),
        "ready_for_final_synthesis": all(
            bool((summary or {}).get("readiness", {}).get("ready_for_final_synthesis"))
            for summary in manifest["field_round_summaries"]
        ),
    }
    storage.save_workflow_json("stage_conditioned_branching_manifest.json", manifest)

    print(json.dumps(
        {
            "workflow_id": args.workflow_id,
            "field_name": args.field_name,
            "round_id": args.round_id,
            "recommended_values_by_stage": recommended_values_by_stage,
            "ready_for_final_synthesis": (field_round_summary.get("readiness") or {}).get("ready_for_final_synthesis"),
        },
        ensure_ascii=False,
        indent=2,
    ))
    return round_payload


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    rebuild_field_round(args)


if __name__ == "__main__":
    main()
