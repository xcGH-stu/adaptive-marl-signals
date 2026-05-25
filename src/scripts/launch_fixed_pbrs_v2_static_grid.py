from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiments.fixed_pbrs_v2_static_grid import build_baseline_kwargs
from experiments.fixed_pbrs_v2_static_grid import build_expected_summary
from experiments.fixed_pbrs_v2_static_grid import build_workflow_id
from experiments.fixed_pbrs_v2_static_grid import default_env_key
from experiments.fixed_pbrs_v2_static_grid import default_t_max_for_algorithm
from experiments.fixed_pbrs_v2_static_grid import list_algorithm_ids
from experiments.fixed_pbrs_v2_static_grid import list_config_ids
from experiments.qmix_hparam_presets import list_qmix_hparam_presets
from scripts.run_fixed_reward_baseline import build_fixed_reward_baseline_plan


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch or dry-run the fixed PBRS-v2 static-grid baseline")
    parser.add_argument("--static-config-id", "--config-id", dest="config_id", choices=list_config_ids(), required=True)
    parser.add_argument("--algorithm", choices=list_algorithm_ids(), required=True)
    parser.add_argument("--env-key", default=default_env_key())
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--t-max", type=int, default=None)
    parser.add_argument(
        "--qmix-hparam-preset",
        choices=list_qmix_hparam_presets(),
        default="default",
    )
    parser.add_argument("--run-index", type=int, default=1)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output-path", default=None)
    execute_group = parser.add_mutually_exclusive_group()
    execute_group.add_argument("--execute", dest="execute", action="store_true")
    execute_group.add_argument("--dry-run", dest="execute", action="store_false")
    parser.add_argument("--use-cuda", dest="use_cuda", action="store_true")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false")
    parser.add_argument("--format", choices=["json", "text"], default="json")
    parser.set_defaults(execute=False)
    parser.set_defaults(use_cuda=True)
    return parser


def _namespace_from_kwargs(kwargs: Dict[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


def _enrich_plan(
    plan: Dict[str, Any],
    *,
    config_id: str,
    algorithm: str,
    env_key: str,
    seed: int,
    t_max: int,
    run_index: int,
) -> Dict[str, Any]:
    payload = dict(plan)
    workflow_id = build_workflow_id(
        config_id,
        seed,
        algorithm=algorithm,
        env_key=env_key,
        run_index=run_index,
    )
    payload["static_grid_context"] = {
        **build_expected_summary(),
        "workflow_id": workflow_id,
        "config_id": config_id,
        "algorithm": algorithm,
        "env_key": env_key,
        "seed": int(seed),
        "t_max": int(t_max),
        "run_index": int(run_index),
        "ready_to_launch_fixed_pbrs_v2_static_grid_seed1": (
            algorithm == "qmix"
            and env_key == default_env_key()
            and int(seed) == 1
        ),
    }
    payload["workflow_id"] = workflow_id
    return payload


def _print_text(payload: Dict[str, Any]) -> None:
    train_config = payload["train_config"]
    env_args = train_config.get("env_args") or {}
    context = payload.get("static_grid_context") or {}
    print(
        f"{payload['workflow_id']}: config_id={context.get('config_id')} seed={context.get('seed')} "
        f"algo={train_config.get('config')} env={env_args.get('key')} t_max={train_config.get('overrides', {}).get('t_max')}"
    )
    print(
        "  "
        f"pbrs_version={env_args.get('pbrs_version')} mode={env_args.get('pbrs_mode')} "
        f"beta={env_args.get('pbrs_beta')} eval_use_pbrs={env_args.get('eval_use_pbrs')}"
    )
    print("  command:")
    print("    " + " ".join(str(item) for item in payload["command"]))


def main() -> None:
    args = _build_parser().parse_args()
    resolved_t_max = int(args.t_max) if args.t_max is not None else default_t_max_for_algorithm(args.algorithm)
    kwargs = build_baseline_kwargs(
        config_id=args.config_id,
        algorithm=args.algorithm,
        env_key=args.env_key,
        seed=args.seed,
        t_max=resolved_t_max,
        run_index=args.run_index,
        python_executable=args.python_executable,
        use_cuda=bool(args.use_cuda),
        execute=bool(args.execute),
        qmix_hparam_preset=args.qmix_hparam_preset,
        output_path=args.output_path,
        result_root=args.results_root,
        format_name=args.format,
    )
    plan = build_fixed_reward_baseline_plan(_namespace_from_kwargs(kwargs))
    payload = _enrich_plan(
        plan,
        config_id=args.config_id,
        algorithm=args.algorithm,
        env_key=args.env_key,
        seed=args.seed,
        t_max=resolved_t_max,
        run_index=args.run_index,
    )
    output_path = kwargs.get("output_path")
    if output_path:
        path = Path(str(output_path))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        return
    _print_text(payload)


if __name__ == "__main__":
    main()
