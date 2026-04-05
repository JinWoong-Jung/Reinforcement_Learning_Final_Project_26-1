from __future__ import annotations

import csv
import json
import os
from typing import Any

import yaml


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str, indent: int = 2) -> None:
    _ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_config(path: str) -> dict[str, Any]:
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as f:
        if ext in {".yaml", ".yml"}:
            data = yaml.safe_load(f)
        elif ext == ".json":
            data = json.load(f)
        else:
            raise ValueError("Config file must be .yaml/.yml or .json")
    if not isinstance(data, dict):
        raise ValueError("Config must be a dictionary object.")
    return data


def save_results_csv(rows: list[dict[str, Any]], path: str) -> None:
    _ensure_parent_dir(path)
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
