#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_suite(output_root: Path, suite: str, episodes_per_task: int) -> dict[str, object]:
    info_path = output_root / suite / "eval_info.json"
    if not info_path.exists():
        return {
            "suite": suite,
            "pc_success": "nan",
            "n_tasks": 0,
            "episodes_per_task": episodes_per_task,
            "total_episodes": 0,
            "eval_info": str(info_path),
            "status": "missing_eval_info",
        }

    data = json.loads(info_path.read_text())
    per_task = data.get("per_task", [])
    overall = data.get("overall", {})
    return {
        "suite": suite,
        "pc_success": float(overall.get("pc_success", float("nan"))),
        "n_tasks": len(per_task),
        "episodes_per_task": episodes_per_task,
        "total_episodes": int(overall.get("n_episodes", 0)),
        "eval_info": str(info_path),
        "status": "ok",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize LIBERO-UMI eval_info.json files.")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--suites", nargs="+", required=True)
    parser.add_argument("--episodes-per-task", required=True, type=int)
    parser.add_argument("--summary-path", required=True, type=Path)
    args = parser.parse_args()

    rows = [read_suite(args.output_root, suite, args.episodes_per_task) for suite in args.suites]
    valid = [float(row["pc_success"]) for row in rows if row["status"] == "ok"]
    avg = sum(valid) / len(valid) if valid else float("nan")
    rows.append(
        {
            "suite": "Avg.",
            "pc_success": avg,
            "n_tasks": sum(int(row["n_tasks"]) for row in rows),
            "episodes_per_task": args.episodes_per_task,
            "total_episodes": sum(int(row["total_episodes"]) for row in rows),
            "eval_info": "",
            "status": "ok" if valid else "no_valid_suites",
        }
    )

    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["suite", "pc_success", "n_tasks", "episodes_per_task", "total_episodes", "status", "eval_info"]
    with args.summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print("\t".join(str(row[name]) for name in fieldnames))


if __name__ == "__main__":
    main()
