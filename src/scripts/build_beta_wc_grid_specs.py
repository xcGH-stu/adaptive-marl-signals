from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-final-spec", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def _build_constant_grid_spec(
    final_spec: Dict[str, Any],
    *,
    beta: float,
    wc: float,
    wp: float,
) -> Dict[str, Any]:
    stage_reward_specs = deepcopy(final_spec.get("stage_reward_specs") or {})
    if not stage_reward_specs:
        raise ValueError("final spec does not contain stage_reward_specs")

    for reward_spec in stage_reward_specs.values():
        pbrs = reward_spec.setdefault("pbrs", {})
        pbrs["beta"] = beta
        pbrs["wc"] = wc
        pbrs["wp"] = wp

    payload = deepcopy(final_spec)
    payload["workflow_kind"] = "constant_beta_wc_grid_spec"
    payload["constant_grid_parameters"] = {
        "beta": beta,
        "wc": wc,
        "wp": wp,
    }
    payload["stage_reward_specs"] = stage_reward_specs
    return payload


def main() -> None:
    args = _build_parser().parse_args()
    final_spec = json.loads(Path(args.input_final_spec).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    values = [0.3, 0.5, 0.7]
    for beta in values:
        for wc in values:
            wp = round(1.0 - wc, 1)
            payload = _build_constant_grid_spec(final_spec, beta=beta, wc=wc, wp=wp)
            stem = f"beta_{beta:.1f}_wc_{wc:.1f}_wp_{wp:.1f}".replace(".", "p")
            file_name = f"{stem}.json"
            output_path = output_dir / file_name
            output_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
                encoding="utf-8",
            )
            print(output_path)


if __name__ == "__main__":
    main()
