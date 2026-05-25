from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-path", required=True, help="Write an SVG plot.")
    parser.add_argument(
        "--metric",
        default="mean_best_test_sparse_return_mean",
        help="Grouped metric field to plot.",
    )
    parser.add_argument(
        "--error-metric",
        default="std_best_test_sparse_return_mean",
        help="Grouped error metric field to display.",
    )
    parser.add_argument("--title", default="Experiment comparison")
    return parser


def _svg_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    grouped = payload.get("grouped_summary") or []
    width = 960
    height = 540
    margin_left = 80
    margin_right = 40
    margin_top = 60
    margin_bottom = 140
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom

    values = [float(row.get(args.metric) or 0.0) for row in grouped]
    errors = [float(row.get(args.error_metric) or 0.0) for row in grouped]
    max_value = max((v + e for v, e in zip(values, errors)), default=1.0)
    max_value = max(max_value, 1e-6)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-size="20" font-family="sans-serif">{_svg_escape(args.title)}</text>',
        f'<line x1="{margin_left}" y1="{margin_top + chart_height}" x2="{margin_left + chart_width}" y2="{margin_top + chart_height}" stroke="black"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + chart_height}" stroke="black"/>',
    ]

    n = max(len(grouped), 1)
    bar_slot = chart_width / n
    bar_width = bar_slot * 0.6
    palette = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948"]

    for idx, row in enumerate(grouped):
        value = values[idx]
        error = errors[idx]
        bar_height = (value / max_value) * chart_height if max_value > 0 else 0.0
        x = margin_left + idx * bar_slot + (bar_slot - bar_width) / 2
        y = margin_top + chart_height - bar_height
        color = palette[idx % len(palette)]
        label = f"{row.get('experiment_family')} | {row.get('budget_profile')} | n={row.get('num_seeds')}"
        parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="{color}"/>'
        )
        if error > 0:
            error_height = (error / max_value) * chart_height
            mid_x = x + bar_width / 2
            parts.append(
                f'<line x1="{mid_x:.2f}" y1="{y - error_height:.2f}" x2="{mid_x:.2f}" y2="{y + error_height:.2f}" stroke="black"/>'
            )
            parts.append(
                f'<line x1="{mid_x - 8:.2f}" y1="{y - error_height:.2f}" x2="{mid_x + 8:.2f}" y2="{y - error_height:.2f}" stroke="black"/>'
            )
            parts.append(
                f'<line x1="{mid_x - 8:.2f}" y1="{y + error_height:.2f}" x2="{mid_x + 8:.2f}" y2="{y + error_height:.2f}" stroke="black"/>'
            )
        parts.append(
            f'<text x="{x + bar_width/2:.2f}" y="{y - 8:.2f}" text-anchor="middle" font-size="12" font-family="sans-serif">{value:.4f}</text>'
        )
        parts.append(
            f'<text transform="translate({x + bar_width/2:.2f},{height - 20}) rotate(-35)" text-anchor="end" font-size="11" font-family="sans-serif">{_svg_escape(label)}</text>'
        )

    for tick_idx in range(5):
        tick_value = max_value * tick_idx / 4
        tick_y = margin_top + chart_height - (tick_value / max_value) * chart_height
        parts.append(
            f'<line x1="{margin_left - 5}" y1="{tick_y:.2f}" x2="{margin_left}" y2="{tick_y:.2f}" stroke="black"/>'
        )
        parts.append(
            f'<text x="{margin_left - 10}" y="{tick_y + 4:.2f}" text-anchor="end" font-size="11" font-family="sans-serif">{tick_value:.3f}</text>'
        )

    parts.append("</svg>")
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
