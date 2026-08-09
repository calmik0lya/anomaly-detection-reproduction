#!/usr/bin/env python3
"""Собирает метрики experiment-папок в компактные CSV и Markdown-таблицы."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


RUN_NAME = re.compile(
    r"^(?P<model>[a-z0-9_]+)__(?P<category>[a-z0-9_]+)__(?P<timestamp>\d{8}_\d{6})$"
)
CORE_METRICS = ("image_auroc", "pixel_auroc", "pro_auroc")


def collect_latest_runs(experiments_dir: Path) -> list[dict]:
    """Возвращает последний успешный запуск для каждой пары model/category."""
    latest: dict[tuple[str, str], dict] = {}

    if not experiments_dir.exists():
        return []

    for run_dir in experiments_dir.iterdir():
        if not run_dir.is_dir():
            continue
        match = RUN_NAME.fullmatch(run_dir.name)
        if not match:
            continue

        metrics_path = run_dir / "metrics.json"
        if not metrics_path.is_file():
            continue
        try:
            metrics = json.loads(metrics_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(metrics, dict) or not metrics:
            continue

        row = {
            "model": match["model"],
            "category": match["category"],
            "timestamp": match["timestamp"],
            "run_dir": run_dir.name,
        }
        row.update({key: value for key, value in metrics.items() if isinstance(value, (int, float))})

        key = (row["model"], row["category"])
        if key not in latest or row["timestamp"] > latest[key]["timestamp"]:
            latest[key] = row

    return sorted(latest.values(), key=lambda row: (row["model"], row["category"]))


def metric_columns(rows: list[dict]) -> list[str]:
    discovered = {
        key
        for row in rows
        for key, value in row.items()
        if key not in {"model", "category", "timestamp", "run_dir"}
        and isinstance(value, (int, float))
    }
    ordered = [metric for metric in CORE_METRICS if metric in discovered]
    return ordered + sorted(discovered.difference(ordered))


def write_csv(rows: list[dict], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    columns = ["model", "category", "timestamp", *metric_columns(rows), "run_dir"]
    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def format_metric(value: object) -> str:
    return f"{value:.6f}" if isinstance(value, (int, float)) else "—"


def write_markdown(rows: list[dict], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    metrics = metric_columns(rows)
    models = sorted({row["model"] for row in rows})

    lines = [
        "# Сводка экспериментов",
        "",
        f"Успешных пар модель/категория: **{len(rows)}**.",
        "",
        "## Покрытие и средние метрики",
        "",
        "| Модель | Категорий | " + " | ".join(metrics) + " |",
        "|---|---:|" + "---:|" * len(metrics),
    ]
    for model in models:
        model_rows = [row for row in rows if row["model"] == model]
        means = []
        for metric in metrics:
            values = [row[metric] for row in model_rows if metric in row]
            means.append(format_metric(sum(values) / len(values)) if values else "—")
        lines.append(f"| {model} | {len(model_rows)} | " + " | ".join(means) + " |")

    lines.extend([
        "",
        "## Метрики по категориям",
        "",
        "| Модель | Категория | " + " | ".join(metrics) + " | Дата запуска |",
        "|---|---|" + "---:|" * len(metrics) + "---|",
    ])
    for row in rows:
        values = " | ".join(format_metric(row.get(metric)) for metric in metrics)
        lines.append(
            f"| {row['model']} | {row['category']} | {values} | {row['timestamp']} |"
        )

    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-dir", default="experiments")
    parser.add_argument("--csv", default="results.csv")
    parser.add_argument("--markdown", default="results.md")
    args = parser.parse_args()

    rows = collect_latest_runs(Path(args.experiments_dir))
    write_csv(rows, Path(args.csv))
    write_markdown(rows, Path(args.markdown))
    print(f"Собрано успешных пар: {len(rows)}")
    print(f"CSV: {Path(args.csv).resolve()}")
    print(f"Markdown: {Path(args.markdown).resolve()}")


if __name__ == "__main__":
    main()
