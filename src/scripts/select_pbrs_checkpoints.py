from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts.run_reward_workflow import (  # noqa: E402
    build_arg_parser,
    build_workflow_spec_from_args,
)
from workflows.clients.checkpoint_selector_client import (  # noqa: E402
    StubCheckpointSelectorBackend,
    TemplateBasedCheckpointSelectorClient,
)
from workflows.clients.openai_backend import (  # noqa: E402
    OpenAIChatBackend,
)
from workflows.storage import WorkflowStorage  # noqa: E402


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sample_curve(metric_payload: Dict[str, Any], max_points: int = 15) -> Dict[str, Any]:
    values = metric_payload.get("values", [])
    steps = metric_payload.get("steps", [])
    if not values:
        return {
            "num_points": 0,
            "sampled_points": [],
        }
    point_count = len(values)
    if point_count <= max_points:
        indices = list(range(point_count))
    else:
        indices = sorted(
            {
                0,
                point_count - 1,
                *(int(round(i * (point_count - 1) / (max_points - 1))) for i in range(max_points)),
            }
        )
    sampled_points = [
        {
            "index": idx,
            "step": int(steps[idx]) if idx < len(steps) else None,
            "value": float(values[idx]),
        }
        for idx in indices
    ]
    return {
        "num_points": point_count,
        "first_step": int(steps[0]) if steps else None,
        "last_step": int(steps[-1]) if steps else None,
        "sampled_points": sampled_points,
    }


def build_baseline_summary(
    *,
    metrics_json_path: Path,
    run_config_path: Optional[Path],
    preferred_metrics: List[str],
    max_points_per_metric: int,
) -> Dict[str, Any]:
    metrics = _load_json(metrics_json_path)
    run_config = _load_json(run_config_path) if run_config_path is not None and run_config_path.exists() else {}
    metric_curves = {}
    for metric_name in preferred_metrics:
        payload = metrics.get(metric_name)
        if isinstance(payload, dict):
            metric_curves[metric_name] = _sample_curve(payload, max_points=max_points_per_metric)
    return {
        "run_metadata": {
            "algorithm": run_config.get("config"),
            "env_args": run_config.get("env_args"),
            "workflow_id": run_config.get("workflow_id"),
            "workflow_round": run_config.get("workflow_round"),
        },
        "metric_curves": metric_curves,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow-id", required=True)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--run-config-json", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument(
        "--attach-to-workflow",
        action="store_true",
        help="Save the selection under the workflow results directory and update workflow_spec.",
    )
    parser.add_argument(
        "--storage-root",
        default=None,
        help="Optional workflow storage root used when attaching the selection artifact.",
    )
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--max-count", type=int, default=4)
    parser.add_argument("--min-step-gap", type=int, default=50000)
    parser.add_argument("--max-points-per-metric", type=int, default=15)
    parser.add_argument(
        "--preferred-metric",
        action="append",
        default=None,
        help="Metric names to provide to the selector; can be passed multiple times.",
    )
    parser.add_argument("--use-real-llm", action="store_true")
    parser.add_argument("--api-key-env", default="IUSEAPI_API_KEY")
    parser.add_argument("--base-url", default="https://www.iuseapi.com/v1")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--llm-timeout", type=float, default=60.0)
    parser.add_argument("--llm-max-retries", type=int, default=3)
    parser.add_argument("--llm-retry-backoff", type=float, default=5.0)
    parser.add_argument("--reward-paradigm", choices=["pbrs", "heuristic"], default="pbrs")
    parser.add_argument("--max-rounds", type=int, default=3)
    args = parser.parse_args()

    if args.use_real_llm:
        llm_backend = OpenAIChatBackend(
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            model=args.model,
            temperature=args.temperature,
            timeout=args.llm_timeout,
            max_retries=args.llm_max_retries,
            retry_backoff_seconds=args.llm_retry_backoff,
        )
    else:
        llm_backend = StubCheckpointSelectorBackend()

    workflow_arg_parser = build_arg_parser()
    workflow_args = workflow_arg_parser.parse_args(
        [
            "--workflow-id",
            args.workflow_id,
            "--max-rounds",
            str(args.max_rounds),
            "--reward-paradigm",
            args.reward_paradigm,
        ]
    )
    workflow_spec = build_workflow_spec_from_args(workflow_args)
    preferred_metrics = args.preferred_metric or [
        "test_sparse_return_mean",
        "sparse_return_mean",
        "test_return_mean",
        "return_mean",
    ]
    baseline_summary = build_baseline_summary(
        metrics_json_path=Path(args.metrics_json),
        run_config_path=Path(args.run_config_json) if args.run_config_json else None,
        preferred_metrics=preferred_metrics,
        max_points_per_metric=args.max_points_per_metric,
    )
    selection_constraints = {
        "min_count": args.min_count,
        "max_count": args.max_count,
        "min_step_gap": args.min_step_gap,
    }
    client = TemplateBasedCheckpointSelectorClient(llm_backend=llm_backend)
    result = client.select_checkpoints(
        workflow_spec=workflow_spec,
        baseline_summary=baseline_summary,
        selection_constraints=selection_constraints,
    )
    output_payload = {
        "selection_constraints": selection_constraints,
        "baseline_summary": baseline_summary,
        "selection": result["selection"],
        "response": result["response"],
    }
    if args.attach_to_workflow:
        storage = WorkflowStorage.load(
            args.workflow_id,
            root_dir=args.storage_root,
        )
        storage.save_workflow_json(
            "baseline_checkpoint_selection.json",
            output_payload,
        )
        manifest = storage.load_manifest()
        workflow_spec = manifest.get("workflow_spec") or {}
        workflow_spec["pbrs_checkpoint_plan"] = result["selection"]
        manifest["workflow_spec"] = workflow_spec
        storage.save_manifest(manifest)
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(output_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    else:
        print(json.dumps(output_payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
