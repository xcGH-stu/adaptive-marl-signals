from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts.critique_pbrs_branch_plan import build_branch_critique
from scripts.run_reward_workflow import load_existing_workflow_spec
from workflows.branch_critic_patch import (
    attach_branch_critic_payload_to_workflow,
    derive_workflow_patch,
    infer_target_next_round_id_from_workflow_manifest,
)
from workflows.storage import WorkflowStorage


DEFAULT_BRANCH_PLAN_ROOT = ROOT / "results" / "pbrs_branch_plans"
DEFAULT_WORKFLOW_ROOT = ROOT / "results" / "llm_reward_workflows"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-plan-id", required=True)
    parser.add_argument(
        "--branch-results-root",
        default=str(DEFAULT_BRANCH_PLAN_ROOT),
        help="Root directory containing pbrs_branch_plans.",
    )
    parser.add_argument(
        "--target-workflow-id",
        default=None,
        help="Optional PBRS workflow id whose manifest workflow_spec should be updated.",
    )
    parser.add_argument(
        "--workflow-results-root",
        default=str(DEFAULT_WORKFLOW_ROOT),
        help="Root directory containing llm_reward_workflows results.",
    )
    parser.add_argument("--use-real-llm", action="store_true")
    parser.add_argument("--api-key-env", default="IUSEAPI_API_KEY")
    parser.add_argument("--base-url", default="https://www.iuseapi.com/v1")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--llm-timeout", type=float, default=60.0)
    parser.add_argument("--llm-max-retries", type=int, default=3)
    parser.add_argument("--llm-retry-backoff", type=float, default=5.0)
    parser.add_argument(
        "--attach-to-workflow",
        action="store_true",
        help="Attach the derived patch and updated workflow_spec to the target workflow manifest.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional file path to save the patch payload.",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format.",
    )
    return parser


def build_patch_payload(args: argparse.Namespace) -> Dict[str, Any]:
    critique_args = argparse.Namespace(
        branch_plan_id=args.branch_plan_id,
        results_root=args.branch_results_root,
        use_real_llm=args.use_real_llm,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        llm_timeout=args.llm_timeout,
        llm_max_retries=args.llm_max_retries,
        llm_retry_backoff=args.llm_retry_backoff,
        output_path=None,
        format="json",
    )
    critique_payload = build_branch_critique(critique_args)
    base_workflow_spec = None
    target_next_round_id = None
    if args.target_workflow_id is not None:
        base_workflow_spec = load_existing_workflow_spec(
            args.target_workflow_id,
            storage_root=args.workflow_results_root,
        )
        try:
            workflow_storage = WorkflowStorage.load(
                args.target_workflow_id,
                root_dir=args.workflow_results_root,
            )
            target_next_round_id = infer_target_next_round_id_from_workflow_manifest(
                workflow_storage.load_manifest()
            )
        except FileNotFoundError:
            target_next_round_id = None
    derived = derive_workflow_patch(
        critique_payload=critique_payload,
        base_workflow_spec=base_workflow_spec,
        target_next_round_id=target_next_round_id,
    )
    payload = {
        "branch_plan_id": args.branch_plan_id,
        "target_workflow_id": args.target_workflow_id,
        "critique_payload": critique_payload,
        "workflow_patch": derived["patch"],
        "updated_workflow_spec": derived["updated_workflow_spec"],
    }

    if args.attach_to_workflow:
        if args.target_workflow_id is None:
            raise ValueError("--attach-to-workflow requires --target-workflow-id")
        attach_branch_critic_payload_to_workflow(
            workflow_id=args.target_workflow_id,
            storage_root=args.workflow_results_root,
            payload=payload,
            updated_workflow_spec=derived["updated_workflow_spec"],
        )

    return payload


def _fmt(value: Any) -> str:
    return "-" if value is None else str(value)


def print_text_payload(payload: Dict[str, Any]) -> None:
    patch = payload.get("workflow_patch") or {}
    print(
        f"{payload['branch_plan_id']}: target_workflow={_fmt(payload.get('target_workflow_id'))}"
    )
    print(
        "  "
        f"recommended_strategy={_fmt(patch.get('recommended_strategy'))} "
        f"recommended_next_field={_fmt(patch.get('recommended_next_field'))} "
        f"recommended_global_value={_fmt(patch.get('recommended_global_value'))}"
    )
    if patch.get("recommended_candidate_values") is not None:
        print(
            "  "
            f"recommended_candidate_values={json.dumps(patch.get('recommended_candidate_values'), ensure_ascii=False)}"
        )
    if patch.get("recommended_values_by_stage"):
        print(
            "  "
            f"recommended_values_by_stage={json.dumps(patch.get('recommended_values_by_stage'), ensure_ascii=False, sort_keys=True)}"
        )
    if patch.get("recommended_fields_by_stage"):
        print(
            "  "
            f"recommended_fields_by_stage={json.dumps(patch.get('recommended_fields_by_stage'), ensure_ascii=False, sort_keys=True)}"
        )
    design_transition = patch.get("recommended_design_transition") or {}
    if design_transition:
        print(
            "  "
            f"design_transition_action={_fmt(design_transition.get('transition_action'))} "
            f"scope={_fmt(design_transition.get('target_scope'))} "
            f"target_stage={_fmt(design_transition.get('target_stage'))}"
        )
        print(
            "  "
            f"allowed_structural_pbrs_fields={json.dumps(design_transition.get('allowed_structural_pbrs_fields') or [], ensure_ascii=False)}"
        )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    payload = build_patch_payload(args)
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
