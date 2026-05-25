from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiments.workflow_profiles import get_workflow_profile
from experiments.workflow_profiles import is_deprecated_workflow_profile
from experiments.workflow_profiles import is_full_clean_runnable_workflow_profile
from experiments.workflow_profiles import is_summary_only_sidecar_profile
from experiments.workflow_profiles import list_launchable_workflow_profiles
from scripts.run_end_to_end_stage_conditioned_workflow import run_end_to_end_workflow
from scripts.run_end_to_end_stage_conditioned_workflow import _build_parser as _build_workflow_parser
from scripts.run_single_llm_pbrs_v2_baseline import build_config_from_args as build_single_llm_baseline_config
from workflows.single_llm_pbrs_v2_baseline import run_single_llm_pbrs_v2_baseline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow-id", required=True)
    parser.add_argument("--workflow-profile", "--profile", choices=list_launchable_workflow_profiles(), required=True)
    parser.add_argument("--results-root", default=str(ROOT / "results" / "end_to_end_workflows"))
    parser.add_argument("--python-executable", default="/home/epymarl/miniconda3/envs/epymarl_new/bin/python")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--llm-model-tier", default=None)
    parser.add_argument("--llm-validation-provider", default=None)
    parser.add_argument("--llm-validation-model", default=None)
    parser.add_argument("--llm-validation-api-key-env", default=None)
    parser.add_argument("--llm-validation-base-url", default=None)
    parser.add_argument("--llm-formal-provider", default=None)
    parser.add_argument("--llm-formal-model", default=None)
    parser.add_argument("--llm-formal-api-key-env", default=None)
    parser.add_argument("--llm-formal-base-url", default=None)
    parser.add_argument("--branch-budget-steps", type=int, default=None)
    parser.add_argument("--max-parallel-candidates", type=int, default=None)
    parser.add_argument("--field-round", action="append", default=None)
    parser.add_argument("--use-cuda", dest="use_cuda", action="store_true")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false")
    parser.add_argument("--stage1-use-cuda", dest="stage1_use_cuda", action="store_true")
    parser.add_argument("--stage1-cpu", dest="stage1_use_cuda", action="store_false")
    parser.add_argument("--stage3-use-cuda", dest="stage3_use_cuda", action="store_true")
    parser.add_argument("--stage3-cpu", dest="stage3_use_cuda", action="store_false")
    parser.add_argument("--stage5-use-cuda", dest="stage5_use_cuda", action="store_true")
    parser.add_argument("--stage5-cpu", dest="stage5_use_cuda", action="store_false")
    parser.add_argument("--baseline-id", default=None)
    parser.add_argument("--branching-workflow-id", default=None)
    parser.add_argument("--final-run-id", default=None)
    parser.add_argument("--existing-sparse-run-dir", default=None)
    parser.add_argument("--stage3-fork-spec-json", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-formal-execute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--loop-until-complete", action="store_true")
    parser.add_argument("--single-tick", action="store_true")
    parser.add_argument("--poll-interval-seconds", type=float, default=30.0)
    parser.add_argument("--max-supervisor-errors", type=int, default=10)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--format", choices=["json", "text"], default="text")
    parser.set_defaults(use_cuda=None, stage1_use_cuda=None, stage3_use_cuda=None, stage5_use_cuda=None)
    return parser


def _workflow_default_payload() -> Dict[str, Any]:
    defaults: Dict[str, Any] = {}
    for action in _build_workflow_parser()._actions:
        dest = getattr(action, "dest", None)
        if not dest or dest == "help":
            continue
        if action.default is argparse.SUPPRESS:
            continue
        defaults[dest] = deepcopy(action.default)
    return defaults


def _build_namespace(args: argparse.Namespace, profile: Dict[str, Any]) -> argparse.Namespace:
    payload = _workflow_default_payload()
    payload.update(deepcopy(profile))
    resource_policy = deepcopy(payload.get("resource_policy") or {})
    if args.seed is not None:
        payload["seed"] = args.seed
    if args.model is not None:
        payload["model"] = args.model
    if args.llm_model_tier is not None:
        payload["llm_model_tier_override"] = args.llm_model_tier
    if args.llm_validation_provider is not None:
        payload["llm_validation_provider"] = args.llm_validation_provider
    if args.llm_validation_model is not None:
        payload["llm_validation_model"] = args.llm_validation_model
    if args.llm_validation_api_key_env is not None:
        payload["llm_validation_api_key_env"] = args.llm_validation_api_key_env
    if args.llm_validation_base_url is not None:
        payload["llm_validation_base_url"] = args.llm_validation_base_url
    if args.llm_formal_provider is not None:
        payload["llm_formal_provider"] = args.llm_formal_provider
    if args.llm_formal_model is not None:
        payload["llm_formal_model"] = args.llm_formal_model
    if args.llm_formal_api_key_env is not None:
        payload["llm_formal_api_key_env"] = args.llm_formal_api_key_env
    if args.llm_formal_base_url is not None:
        payload["llm_formal_base_url"] = args.llm_formal_base_url
    if args.branch_budget_steps is not None:
        payload["branch_budget_steps"] = args.branch_budget_steps
    if args.max_parallel_candidates is not None:
        payload["max_parallel_candidates"] = args.max_parallel_candidates
    if args.field_round:
        payload["field_round"] = list(args.field_round)
    if args.use_cuda is not None:
        payload["use_cuda"] = bool(args.use_cuda)
    if args.stage1_use_cuda is not None:
        payload["stage1_use_cuda"] = bool(args.stage1_use_cuda)
    if args.stage3_use_cuda is not None:
        payload["stage3_use_cuda"] = bool(args.stage3_use_cuda)
    if args.stage5_use_cuda is not None:
        payload["stage5_use_cuda"] = bool(args.stage5_use_cuda)
    payload.setdefault("stage1_use_cuda", resource_policy.get("stage1_use_cuda", payload.get("use_cuda", True)))
    payload.setdefault("stage3_use_cuda", resource_policy.get("stage3_use_cuda", payload.get("use_cuda", True)))
    payload.setdefault("stage5_use_cuda", resource_policy.get("stage5_use_cuda", payload.get("use_cuda", True)))
    payload.update(
        {
            "workflow_id": args.workflow_id,
            "workflow_profile": args.workflow_profile,
            "results_root": args.results_root,
            "python_executable": args.python_executable,
            "output_path": args.output_path,
            "format": args.format,
            "execute": bool(args.execute) and not bool(args.dry_run),
            "allow_formal_execute": bool(args.allow_formal_execute),
            "loop_until_complete": bool(args.loop_until_complete),
            "poll_interval_seconds": float(args.poll_interval_seconds),
            "base_url": "https://www.iuseapi.com/v1",
            "api_key_env": "IUSEAPI_API_KEY",
            "preferred_metric": None,
            "checkpoint_optional_metric": None,
            "field_round": list(args.field_round) if args.field_round else payload.get("field_round"),
            "baseline_id": args.baseline_id,
            "branching_workflow_id": args.branching_workflow_id,
            "final_run_id": args.final_run_id,
            "existing_sparse_run_dir": args.existing_sparse_run_dir,
            "stage3_fork_spec_json": args.stage3_fork_spec_json,
        }
    )
    return argparse.Namespace(**payload)


def _print_text(payload: Dict[str, Any]) -> None:
    manifest = payload["manifest"]
    config = manifest.get("config_summary") or {}
    print(
        f"{payload['workflow_id']}: profile={config.get('workflow_profile')} "
        f"status={manifest.get('status')} env={config.get('env_key')} algo={config.get('train_config')}"
    )
    for stage_name, stage_payload in (manifest.get("stages") or {}).items():
        print(
            "  "
            f"{stage_name}: status={stage_payload.get('status')} artifact={stage_payload.get('artifact_json') or '-'}"
        )


def _should_supervise(args: argparse.Namespace) -> bool:
    if args.single_tick:
        return False
    return bool(args.execute)


def _validate_profile_launch_intent(workflow_profile: str) -> None:
    profile_name = str(workflow_profile or "")
    expects_full_clean = "full_clean" in profile_name.lower()
    if expects_full_clean and not is_full_clean_runnable_workflow_profile(profile_name):
        raise ValueError(
            f"workflow profile {profile_name!r} is not configured as a full clean final-continuation run"
        )
    if expects_full_clean and is_summary_only_sidecar_profile(profile_name):
        raise ValueError(
            f"workflow profile {profile_name!r} is summary-only/sidecar and cannot be launched as full clean"
        )


def _validate_small_pilot_guardrails(
    *,
    workflow_profile: str,
    namespace: argparse.Namespace,
) -> None:
    if workflow_profile == "qmix_dual_llm_policy_guided_single_seed_pilot":
        raise ValueError("formal policy-guided single-seed pilot is blocked for small pilot v2 repair")
    if str(getattr(namespace, "train_config", "") or "") != "qmix":
        raise ValueError("small pilot v2 guardrails only allow qmix")
    if "mappo" in str(getattr(namespace, "train_config", "") or "").lower():
        raise ValueError("small pilot v2 guardrails do not allow MAPPO")
    seed_value = getattr(namespace, "seed", None)
    if not isinstance(seed_value, int):
        raise ValueError("small pilot v2 guardrails require a single integer seed")
    if int(getattr(namespace, "t_max", 0) or 0) > 50000:
        raise ValueError("small pilot v2 guardrails require Stage 1 sparse t_max <= 50000")
    if int(getattr(namespace, "branch_budget_steps", 0) or 0) > 20000:
        raise ValueError("small pilot v2 guardrails require branch_budget_steps <= 20000")
    phase1_method = getattr(namespace, "phase1_method", None) or {}
    initial_dense_search = phase1_method.get("initial_dense_search") or {}
    candidate_budget_steps = int(
        getattr(namespace, "initial_dense_candidate_budget_steps", 0)
        or initial_dense_search.get("candidate_budget_steps")
        or 0
    )
    if candidate_budget_steps > 20000:
        raise ValueError("small pilot v2 guardrails require Stage 1b candidate budget <= 20000")
    if int(getattr(namespace, "t_max", 0) or 0) > 100000:
        raise ValueError("small pilot v2 guardrails require final t_max <= 100000")


def _validate_formal_pilot_guardrails(
    *,
    workflow_profile: str,
    namespace: argparse.Namespace,
    execute_requested: bool,
    allow_formal_execute: bool,
) -> None:
    is_formal_profile = (
        "formal_pilot" in workflow_profile
        or "formal_seed" in workflow_profile
        or bool(getattr(namespace, "formal_execute_guard_required", False))
        or "formal" in str(getattr(namespace, "budget_profile", "") or "")
    )
    if not is_formal_profile:
        return
    if execute_requested and not allow_formal_execute:
        raise ValueError(
            "formal pilot profiles are dry-run only by default; re-run with --allow-formal-execute "
            "only after explicit manual authorization"
        )
    seed_value = getattr(namespace, "seed", None)
    if not isinstance(seed_value, int):
        raise ValueError("formal pilot guardrails require a single integer seed")


def _load_manifest_if_present(results_root: str, workflow_id: str) -> Dict[str, Any] | None:
    manifest_path = Path(results_root) / workflow_id / "workflow_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _manifest_terminal(manifest: Dict[str, Any] | None) -> bool:
    if not manifest:
        return False
    return str(manifest.get("status") or "") in {"completed", "failed"}


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    profile = get_workflow_profile(args.workflow_profile)
    _validate_profile_launch_intent(args.workflow_profile)
    if str(profile.get("workflow_kind") or "") == "single_llm_pbrs_v2_baseline":
        baseline_args = argparse.Namespace(
            workflow_id=args.workflow_id,
            workflow_profile=args.workflow_profile,
            results_root=args.results_root,
            python_executable=args.python_executable,
            seed=args.seed,
            candidate_budget_steps=getattr(args, "branch_budget_steps", None),
            full_t_max=None,
            model=args.model,
            api_key_env=None,
            base_url=None,
            sparse_run_dir=args.existing_sparse_run_dir,
            use_cuda=args.use_cuda,
            use_real_llm=False,
            execute=bool(args.execute) and not bool(args.dry_run),
            dry_run=bool(args.dry_run),
            tiny_smoke=False,
            output_path=args.output_path,
            format=args.format,
        )
        payload = run_single_llm_pbrs_v2_baseline(build_single_llm_baseline_config(baseline_args))
        if args.output_path:
            Path(args.output_path).write_text(
                json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
                encoding="utf-8",
            )
        if args.format == "json":
            print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
            return
        _print_text(payload)
        return
    if is_deprecated_workflow_profile(args.workflow_profile):
        raise ValueError(
            f"workflow profile {args.workflow_profile!r} is deprecated for new end-to-end launches; "
            "use an *_adaptive profile instead."
        )
    namespace = _build_namespace(args, profile)
    if args.workflow_profile == "qmix_policy_guided_small_pilot_seed1":
        _validate_small_pilot_guardrails(
            workflow_profile=args.workflow_profile,
            namespace=namespace,
        )
    _validate_formal_pilot_guardrails(
        workflow_profile=args.workflow_profile,
        namespace=namespace,
        execute_requested=bool(args.execute) and not bool(args.dry_run),
        allow_formal_execute=bool(args.allow_formal_execute),
    )
    payload: Dict[str, Any]
    consecutive_errors = 0
    payload = run_end_to_end_workflow(namespace)
    supervise = _should_supervise(args) or (args.loop_until_complete and args.execute)
    if supervise:
        while True:
            manifest = payload.get("manifest") or {}
            if _manifest_terminal(manifest):
                break
            time.sleep(max(1.0, float(args.poll_interval_seconds)))
            try:
                payload = run_end_to_end_workflow(namespace)
                consecutive_errors = 0
            except Exception:
                consecutive_errors += 1
                manifest = _load_manifest_if_present(args.results_root, args.workflow_id)
                if _manifest_terminal(manifest):
                    payload = {"workflow_id": args.workflow_id, "manifest": manifest}
                    break
                if consecutive_errors >= max(1, int(args.max_supervisor_errors)):
                    raise
    if args.output_path:
        Path(args.output_path).write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        return
    _print_text(payload)


if __name__ == "__main__":
    main()
