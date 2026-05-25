from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts.summarize_pbrs_branch_plan import summarize_branch_plan
from workflows.clients.branch_critic_client import (
    StubPBRSBranchCriticBackend,
    TemplateBasedPBRSBranchCriticClient,
)
from workflows.clients.openai_backend import OpenAIChatBackend


DEFAULT_BRANCH_PLAN_ROOT = ROOT / "results" / "pbrs_branch_plans"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-plan-id", required=True)
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_BRANCH_PLAN_ROOT),
        help="Root directory containing pbrs_branch_plans.",
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
        "--output-path",
        default=None,
        help="Optional file path to save the critique payload.",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format.",
    )
    return parser


def _build_backend(args: argparse.Namespace):
    if args.use_real_llm:
        return OpenAIChatBackend(
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            model=args.model,
            temperature=args.temperature,
            timeout=args.llm_timeout,
            max_retries=args.llm_max_retries,
            retry_backoff_seconds=args.llm_retry_backoff,
        )
    return StubPBRSBranchCriticBackend()


def build_branch_critique(args: argparse.Namespace) -> dict:
    branch_plan_dir = Path(args.results_root) / args.branch_plan_id
    summary = summarize_branch_plan(branch_plan_dir)
    client = TemplateBasedPBRSBranchCriticClient(llm_backend=_build_backend(args))
    result = client.critique_branch_plan(branch_plan_summary=summary)
    payload = {
        "branch_plan_summary": summary,
        "branch_critic_result": result,
    }
    return payload


def print_text_payload(payload: dict) -> None:
    summary = payload["branch_plan_summary"]
    critique = payload["branch_critic_result"]
    parsed = (critique.get("response") or {}).get("parsed") or {}
    print(f"{summary['branch_plan_id']}: branch_critic")
    print(f"  analysis_summary={parsed.get('analysis_summary') or '-'}")
    final_recommendation = parsed.get("final_recommendation") or {}
    print(
        "  "
        f"recommended_strategy={final_recommendation.get('recommended_strategy') or '-'} "
        f"recommended_next_field={final_recommendation.get('recommended_next_field') or '-'} "
        f"recommended_global_value={final_recommendation.get('recommended_global_value')}"
    )
    if final_recommendation.get("recommended_candidate_values") is not None:
        print(
            "  "
            f"recommended_candidate_values={json.dumps(final_recommendation.get('recommended_candidate_values'), ensure_ascii=False)}"
        )
    if final_recommendation.get("recommended_fields_by_stage"):
        print(
            "  "
            f"recommended_fields_by_stage={json.dumps(final_recommendation.get('recommended_fields_by_stage'), ensure_ascii=False, sort_keys=True)}"
        )
    design_transition = final_recommendation.get("recommended_design_transition") or {}
    if design_transition:
        print(
            "  "
            f"design_transition_action={design_transition.get('transition_action') or '-'} "
            f"scope={design_transition.get('target_scope') or '-'} "
            f"target_stage={design_transition.get('target_stage') or '-'}"
        )
        print(
            "  "
            f"allowed_structural_pbrs_fields={json.dumps(design_transition.get('allowed_structural_pbrs_fields') or [], ensure_ascii=False)}"
        )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    payload = build_branch_critique(args)
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
