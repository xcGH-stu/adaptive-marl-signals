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

from workflows.baseline_run_support import inspect_baseline_checkpoint_readiness


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run-dir", default=None)
    parser.add_argument("--baseline-result-json", default=None)
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format.",
    )
    return parser


def _fmt(value: Any) -> str:
    return "-" if value is None else str(value)


def print_text_report(report: Dict[str, Any]) -> None:
    baseline_source = report.get("baseline_source") or {}
    config_summary = report.get("config_summary") or {}
    checkpoint_summary = report.get("checkpoint_summary") or {}
    metrics_summary = report.get("metrics_summary") or {}
    blockers = report.get("blockers") or []
    recommended_actions = report.get("recommended_actions") or []
    print(
        f"baseline_run_dir={_fmt(baseline_source.get('baseline_run_dir'))} "
        f"branching_ready={_fmt(report.get('branching_ready'))}"
    )
    print(
        "  "
        f"algorithm={_fmt(config_summary.get('algorithm'))} "
        f"env={_fmt(config_summary.get('env_key'))} "
        f"t_max={_fmt(config_summary.get('t_max'))}"
    )
    print(
        "  "
        f"checkpoint_root_exists={_fmt(checkpoint_summary.get('checkpoint_root_exists'))} "
        f"latest_checkpoint_step={_fmt(checkpoint_summary.get('latest_checkpoint_step'))}"
    )
    print(
        "  "
        f"available_checkpoint_steps={json.dumps(checkpoint_summary.get('available_checkpoint_steps', []), ensure_ascii=False)}"
    )
    print(
        "  "
        f"metric_count={_fmt(metrics_summary.get('metric_count'))}"
    )
    if blockers:
        print(f"  blockers={json.dumps(blockers, ensure_ascii=False)}")
    if recommended_actions:
        print(f"  recommended_actions={json.dumps(recommended_actions, ensure_ascii=False)}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    report = inspect_baseline_checkpoint_readiness(
        baseline_run_dir=args.baseline_run_dir,
        baseline_result_json=args.baseline_result_json,
    )
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
        return
    print_text_report(report)


if __name__ == "__main__":
    main()
