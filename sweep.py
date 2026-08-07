#!/usr/bin/env python3
"""Последовательно запускает модели на категориях MVTec AD с resume."""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS = ["patchcore", "padim", "stfpm", "simplenet", "draem"]


def load_categories():
    with open(ROOT / "configs" / "tasks" / "anomaly_detection.yaml") as f:
        return yaml.safe_load(f)["categories"]


def completed_pairs(output_dir):
    """Возвращает пары model/category с непустым metrics.json."""
    completed = set()
    if not output_dir.exists():
        return completed
    for metrics_path in output_dir.glob("*__*__*/metrics.json"):
        parts = metrics_path.parent.name.rsplit("__", 2)
        if len(parts) != 3:
            continue
        try:
            metrics = json.loads(metrics_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if metrics:
            completed.add((parts[0], parts[1]))
    return completed


def write_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=DEFAULT_MODELS, default=DEFAULT_MODELS)
    parser.add_argument("--categories", nargs="+", default=load_categories())
    parser.add_argument("--paths", default="configs/paths/local.yaml")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", default="experiments")
    parser.add_argument("--action", choices=["train", "test", "all"], default="all")
    parser.add_argument("--resume", action="store_true",
                        help="пропустить пары, у которых уже есть непустые метрики")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="запускать следующие пары после ошибки")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    state_path = output_dir / "sweep_state.json"
    completed = completed_pairs(output_dir) if args.resume else set()
    state = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "models": args.models,
        "categories": args.categories,
        "runs": [],
    }

    total = len(args.models) * len(args.categories)
    index = 0
    for model in args.models:
        for category in args.categories:
            index += 1
            if (model, category) in completed:
                print(f"[sweep] {index}/{total} SKIP {model}/{category}: уже готово")
                state["runs"].append({"model": model, "category": category, "status": "skipped"})
                write_state(state_path, state)
                continue

            cmd = [
                sys.executable, str(ROOT / "run.py"),
                "--config", str(ROOT / "configs" / "models" / f"{model}.yaml"),
                "--paths", args.paths,
                "--category", category,
                "--action", args.action,
                "--device", args.device,
                "--output-dir", str(output_dir),
            ]
            if args.dry_run:
                cmd.append("--dry-run")
            print(f"[sweep] {index}/{total} RUN  {model}/{category}", flush=True)
            result = subprocess.run(cmd, cwd=ROOT)
            status = "dry-run" if args.dry_run and result.returncode == 0 else (
                "completed" if result.returncode == 0 else "failed"
            )
            state["runs"].append({
                "model": model,
                "category": category,
                "status": status,
                "returncode": result.returncode,
            })
            write_state(state_path, state)
            if result.returncode != 0 and not args.continue_on_error:
                raise SystemExit(result.returncode)

    state["finished_at"] = datetime.now().isoformat(timespec="seconds")
    write_state(state_path, state)
    print(f"[sweep] Готово. Состояние: {state_path}")


if __name__ == "__main__":
    main()
