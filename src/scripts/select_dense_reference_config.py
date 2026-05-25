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
from workflows.phase1_method import select_dense_reference_with_llm


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow-id", required=True)
    parser.add_argument("--stage-selection-result-json", required=True)
    parser.add_argument("--workflow-profile", required=True)
    parser.add_argument("--api-key-env", default="IUSEAPI_API_KEY")
    parser.add_argument("--base-url", default="https://www.iuseapi.com/v1")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--llm-timeout", type=float, default=60.0)
    parser.add_argument("--llm-max-retries", type=int, default=3)
    parser.add_argument("--llm-retry-backoff", type=float, default=5.0)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--format", choices=["text", "json"], default="text")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    stage_selection_result = json.loads(
        Path(args.stage_selection_result_json).read_text(encoding="utf-8")
    )
    profile = get_workflow_profile(args.workflow_profile)
    dense_reference_config = (
        ((profile.get("phase1_method") or {}).get("dense_reference") or {})
    )

    payload = select_dense_reference_with_llm(
        workflow_id=args.workflow_id,
        sparse_stage_selection_result=stage_selection_result,
        dense_reference_config=dense_reference_config,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        llm_timeout=args.llm_timeout,
        llm_max_retries=args.llm_max_retries,
        llm_retry_backoff=args.llm_retry_backoff,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )

    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        return

    print(
        f"{args.workflow_id}: selected_beta={payload.get('selected_beta')} "
        f"selected_wc={payload.get('selected_wc')} selected_wp={payload.get('selected_wp')}"
    )
    print(f"  output_path={output_path}")


if __name__ == "__main__":
    main()
