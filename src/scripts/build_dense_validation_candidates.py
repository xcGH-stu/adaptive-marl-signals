from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from workflows.phase1_method import build_dense_validation_candidates


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-spec-path", required=True)
    parser.add_argument("--branching-manifest-path", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output-path", required=True)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    final_spec = json.loads(Path(args.final_spec_path).read_text(encoding="utf-8"))
    branching_manifest = json.loads(Path(args.branching_manifest_path).read_text(encoding="utf-8"))

    candidates = build_dense_validation_candidates(
        final_spec=final_spec,
        field_round_summaries=list(branching_manifest.get("field_round_summaries") or []),
        top_k_schedules=int(args.top_k),
    )
    output = {
        "source_final_spec_path": str(Path(args.final_spec_path)),
        "source_branching_manifest_path": str(Path(args.branching_manifest_path)),
        "top_k": int(args.top_k),
        "dense_validation_candidates": candidates,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    print(str(output_path))


if __name__ == "__main__":
    main()
