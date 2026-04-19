from __future__ import annotations

import argparse
import json
import os
from typing import Any

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


def _ensure_matplotlib() -> None:
    if plt is None:
        raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_no} in {path} is not a JSON object.")
            rows.append(record)
    if not rows:
        raise ValueError(f"No JSON records found in {path}.")
    return rows


def _default_save_path(jsonl_path: str, y_key: str) -> str:
    base, _ = os.path.splitext(jsonl_path)
    return f"{base}_{y_key}.png"


def build_series(rows: list[dict[str, Any]], x_key: str, y_key: str) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for idx, row in enumerate(rows, start=1):
        if x_key not in row:
            raise KeyError(f"Missing x-axis key '{x_key}' in record #{idx}.")
        if y_key not in row:
            raise KeyError(f"Missing y-axis key '{y_key}' in record #{idx}.")
        try:
            x_val = float(row[x_key])
            y_val = float(row[y_key])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Record #{idx} has non-numeric values for x='{x_key}' or y='{y_key}'."
            ) from exc
        xs.append(x_val)
        ys.append(y_val)
    return xs, ys


def plot_jsonl(
    jsonl_path: str,
    *,
    x_key: str = "timestep",
    y_key: str = "mean_score",
    save: bool = False,
    save_path: str | None = None,
) -> str | None:
    _ensure_matplotlib()
    rows = load_jsonl(jsonl_path)
    xs, ys = build_series(rows, x_key=x_key, y_key=y_key)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, ys, marker="o", linewidth=2, markersize=4)
    ax.set_title(f"{y_key} vs {x_key}")
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    actual_save_path: str | None = None
    if save:
        actual_save_path = save_path or _default_save_path(jsonl_path, y_key)
        _ensure_parent_dir(actual_save_path)
        fig.savefig(actual_save_path, dpi=150)

    plt.show()
    plt.close(fig)
    return actual_save_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a metric from a JSONL score/log curve file.")
    parser.add_argument("jsonl_path", type=str, help="Path to a JSONL file, e.g. eval/score_curve.jsonl")
    parser.add_argument("--x-key", type=str, default="timestep", help="Key to use for the x-axis")
    parser.add_argument("--y-key", type=str, default="mean_score", help="Key to use for the y-axis")
    parser.add_argument("--save", action="store_true", help="Save the figure to disk")
    parser.add_argument("--save-path", type=str, default=None, help="Optional output image path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    saved = plot_jsonl(
        args.jsonl_path,
        x_key=args.x_key,
        y_key=args.y_key,
        save=args.save,
        save_path=args.save_path,
    )
    if saved:
        print(saved)


if __name__ == "__main__":
    main()
