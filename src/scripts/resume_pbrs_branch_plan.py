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

from workflows.train_launcher import EPyMARLTrainLauncher


DEFAULT_BRANCH_PLAN_ROOT = ROOT / "results" / "pbrs_branch_plans"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-plan-id", required=True)
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_BRANCH_PLAN_ROOT),
        help="Root directory containing pbrs_branch_plans.",
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument(
        "--round-id",
        type=int,
        default=None,
        help="Optional round filter. If omitted, resume all rounds.",
    )
    parser.add_argument(
        "--candidate-id",
        default=None,
        help="Optional candidate filter within the selected round(s).",
    )
    parser.add_argument(
        "--reuse-completed-candidates",
        action="store_true",
        help="Reuse existing branch_result.json files instead of rerunning completed candidates.",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format.",
    )
    return parser


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)


def _load_existing_branch_result(candidate_dir: Path) -> Optional[Dict[str, Any]]:
    branch_result_path = candidate_dir / "branch_result.json"
    if not branch_result_path.exists():
        return None
    payload = _load_json(branch_result_path)
    if not isinstance(payload, dict):
        return None
    if not isinstance(payload.get("train_config"), dict):
        return None
    if not isinstance(payload.get("run_reference"), dict):
        return None
    return {
        "payload": payload,
        "path": str(branch_result_path),
    }


def _matches_filters(
    *,
    round_payload: Dict[str, Any],
    candidate_payload: Dict[str, Any],
    round_id: Optional[int],
    candidate_id: Optional[str],
) -> bool:
    if round_id is not None and int(round_payload.get("round_id", -1)) != int(round_id):
        return False
    if candidate_id is not None and str(candidate_payload.get("candidate_id")) != str(candidate_id):
        return False
    return True


def _compute_branch_execution_summary(rounds: List[Dict[str, Any]]) -> Dict[str, int]:
    executed = 0
    reused = 0
    planned_only = 0
    for round_payload in rounds:
        for candidate in round_payload.get("branch_candidates", []):
            status = candidate.get("branch_result_status")
            if status == "executed":
                executed += 1
            elif status == "reused":
                reused += 1
            elif status == "planned_only":
                planned_only += 1
    return {
        "executed_branch_candidate_count": executed,
        "reused_branch_candidate_count": reused,
        "planned_only_branch_candidate_count": planned_only,
    }


def resume_branch_plan(args: argparse.Namespace) -> Dict[str, Any]:
    branch_plan_dir = Path(args.results_root) / args.branch_plan_id
    manifest_path = branch_plan_dir / "manifest.json"
    manifest = _load_json(manifest_path)

    launcher = EPyMARLTrainLauncher(repo_root=ROOT, python_executable=args.python_executable)

    resumed_count = 0
    reused_count = 0
    skipped_count = 0

    for round_payload in manifest.get("rounds", []):
        for candidate in round_payload.get("branch_candidates", []):
            if not _matches_filters(
                round_payload=round_payload,
                candidate_payload=candidate,
                round_id=args.round_id,
                candidate_id=args.candidate_id,
            ):
                skipped_count += 1
                continue

            candidate_dir = Path((candidate.get("reward_artifacts") or {}).get("candidate_dir", ""))
            existing_result = None
            if args.reuse_completed_candidates and candidate_dir:
                existing_result = _load_existing_branch_result(candidate_dir)

            if existing_result is not None:
                candidate["branch_result"] = existing_result["payload"]
                candidate["branch_result_path"] = existing_result["path"]
                if candidate.get("branch_result_status") not in {"executed", "reused"}:
                    candidate["branch_result_status"] = "reused"
                reused_count += 1
                continue

            branch_plan = candidate.get("branch_plan") or {}
            resume_train_config = branch_plan.get("resume_train_config")
            if not isinstance(resume_train_config, dict):
                raise ValueError(
                    f"candidate {candidate.get('candidate_id')} does not contain a valid resume_train_config"
                )

            result = launcher.run_prebuilt_train_config(resume_train_config)
            candidate_result_payload = {
                "train_config": result["train_config"],
                "run_reference": result["run_reference"],
                "metrics_summary": result.get("metrics_summary"),
            }
            candidate_dir.mkdir(parents=True, exist_ok=True)
            branch_result_path = candidate_dir / "branch_result.json"
            _save_json(branch_result_path, candidate_result_payload)
            candidate["branch_result"] = candidate_result_payload
            candidate["branch_result_path"] = str(branch_result_path)
            candidate["branch_result_status"] = "executed"
            resumed_count += 1

    branch_execution_summary = manifest.get("branch_execution_summary") or {}
    branch_execution_summary.update(
        {
            "execute_requested": True,
            "reuse_completed_candidates": bool(args.reuse_completed_candidates),
        }
    )
    branch_execution_summary.update(_compute_branch_execution_summary(manifest.get("rounds", [])))
    manifest["branch_execution_summary"] = branch_execution_summary
    manifest["branch_resume_summary"] = {
        "round_id": args.round_id,
        "candidate_id": args.candidate_id,
        "reuse_completed_candidates": bool(args.reuse_completed_candidates),
        "resumed_candidate_count": resumed_count,
        "reused_candidate_count": reused_count,
        "skipped_candidate_count": skipped_count,
    }
    _save_json(manifest_path, manifest)

    return {
        "branch_plan_id": args.branch_plan_id,
        "branch_plan_dir": str(branch_plan_dir),
        "branch_execution_summary": manifest.get("branch_execution_summary"),
        "branch_resume_summary": manifest.get("branch_resume_summary"),
    }


def _fmt(value: Any) -> str:
    return "-" if value is None else str(value)


def print_text_payload(payload: Dict[str, Any]) -> None:
    execution_summary = payload.get("branch_execution_summary") or {}
    resume_summary = payload.get("branch_resume_summary") or {}
    print(f"{payload['branch_plan_id']}: resume_branch_plan")
    print(
        "  "
        f"resumed={_fmt(resume_summary.get('resumed_candidate_count'))} "
        f"reused={_fmt(resume_summary.get('reused_candidate_count'))} "
        f"skipped={_fmt(resume_summary.get('skipped_candidate_count'))}"
    )
    print(
        "  "
        f"executed_total={_fmt(execution_summary.get('executed_branch_candidate_count'))} "
        f"reused_total={_fmt(execution_summary.get('reused_branch_candidate_count'))} "
        f"planned_only_total={_fmt(execution_summary.get('planned_only_branch_candidate_count'))}"
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    payload = resume_branch_plan(args)
    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        return
    print_text_payload(payload)


if __name__ == "__main__":
    main()
