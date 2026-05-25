from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiments.profiles import export_experiment_templates


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "configs" / "experiments"),
    )
    parser.add_argument("--seed", type=int, default=1)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    written = export_experiment_templates(output_dir=args.output_dir, seed=args.seed)
    print(json.dumps(written, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
