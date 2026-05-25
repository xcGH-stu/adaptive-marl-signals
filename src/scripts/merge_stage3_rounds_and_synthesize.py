from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from workflows.final_stage_conditioned_spec import synthesize_final_stage_conditioned_spec


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-workflow-id", required=True)
    parser.add_argument(
        "--source-workflow-id",
        action="append",
        required=True,
        help="Stage 3 workflow ids to merge. Repeat for multiple sources.",
    )
    parser.add_argument("--results-root", default="results/stage_conditioned_workflows")
    return parser


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _choose_base_manifest(manifests: List[Dict[str, Any]]) -> Dict[str, Any]:
    best = None
    best_round_count = -1
    for manifest in manifests:
        rounds = manifest.get("rounds") or []
        if len(rounds) > best_round_count:
            best = manifest
            best_round_count = len(rounds)
    if best is None:
        raise ValueError("No manifests provided")
    return best


def _load_field_round_payloads(workflow_dir: Path) -> Dict[str, Dict[str, Any]]:
    payloads: Dict[str, Dict[str, Any]] = {}
    pattern = re.compile(r"field_round_(\d+)_(.+)\.json$")
    for path in workflow_dir.glob("field_round_*_*.json"):
        if path.name.endswith("_summary.json") or path.name.endswith("_comparison.json") or path.name.endswith("_critic_response.json"):
            continue
        match = pattern.match(path.name)
        if not match:
            continue
        payload = _load_json(path)
        field_name = payload.get("field_name")
        if isinstance(field_name, str):
            payloads[field_name] = payload
    return payloads


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    root = ROOT / args.results_root
    source_manifests = []
    for workflow_id in args.source_workflow_id:
        manifest_path = root / workflow_id / "stage_conditioned_branching_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")
        source_manifests.append(_load_json(manifest_path))

    base_manifest = _choose_base_manifest(source_manifests)
    merged_manifest = dict(base_manifest)
    merged_manifest["workflow_id"] = args.output_workflow_id

    rounds_by_field: Dict[str, Dict[str, Any]] = {}
    for workflow_id, manifest in zip(args.source_workflow_id, source_manifests):
        for round_payload in manifest.get("rounds") or []:
            field_name = round_payload.get("field_name")
            if isinstance(field_name, str):
                rounds_by_field[field_name] = round_payload
        workflow_dir = root / workflow_id
        rounds_by_field.update(_load_field_round_payloads(workflow_dir))

    required_order = ["beta", "wc", "wp"]
    merged_rounds = []
    for index, field_name in enumerate(required_order, start=1):
        round_payload = rounds_by_field.get(field_name)
        if round_payload is None:
            continue
        round_payload = dict(round_payload)
        round_payload["round_id"] = index
        merged_rounds.append(round_payload)

    merged_manifest["rounds"] = merged_rounds
    merged_manifest["field_round_summaries"] = [
        round_payload.get("field_round_summary") for round_payload in merged_rounds
    ]
    merged_manifest["readiness"] = {
        "field_round_count": len(merged_rounds),
        "ready_for_final_synthesis": (
            len(merged_rounds) == 3
            and all(
                bool((summary or {}).get("readiness", {}).get("ready_for_final_synthesis"))
                for summary in merged_manifest["field_round_summaries"]
            )
        ),
    }

    output_dir = root / args.output_workflow_id
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "stage_conditioned_branching_manifest.json"
    manifest_path.write_text(
        json.dumps(merged_manifest, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )

    final_spec = synthesize_final_stage_conditioned_spec(merged_manifest)
    final_spec_path = output_dir / "final_stage_conditioned_reward_spec.json"
    final_spec_path.write_text(
        json.dumps(final_spec, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output_workflow_id": args.output_workflow_id,
                "merged_fields": [round_payload.get("field_name") for round_payload in merged_rounds],
                "ready_for_final_synthesis": merged_manifest["readiness"]["ready_for_final_synthesis"],
                "manifest_path": str(manifest_path),
                "final_spec_path": str(final_spec_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
