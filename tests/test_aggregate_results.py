import json

from aggregate_results import collect_latest_runs, write_csv, write_markdown


def make_run(root, name, metrics):
    run = root / name
    run.mkdir()
    (run / "metrics.json").write_text(json.dumps(metrics))
    return run


def test_collect_latest_runs_selects_latest_successful_pair(tmp_path):
    make_run(tmp_path, "stfpm__bottle__20260101_120000", {"image_auroc": 0.8})
    make_run(tmp_path, "stfpm__bottle__20260102_120000", {"image_auroc": 0.9})
    make_run(tmp_path, "stfpm__cable__20260101_120000", {})

    rows = collect_latest_runs(tmp_path)

    assert len(rows) == 1
    assert rows[0]["timestamp"] == "20260102_120000"
    assert rows[0]["image_auroc"] == 0.9


def test_writers_create_csv_and_markdown(tmp_path):
    rows = [{
        "model": "patchcore",
        "category": "bottle",
        "timestamp": "20260101_120000",
        "run_dir": "patchcore__bottle__20260101_120000",
        "image_auroc": 1.0,
        "pixel_auroc": 0.98,
    }]
    csv_path = tmp_path / "results.csv"
    markdown_path = tmp_path / "results.md"

    write_csv(rows, csv_path)
    write_markdown(rows, markdown_path)

    assert "patchcore,bottle,20260101_120000" in csv_path.read_text()
    assert "| patchcore | bottle | 1.000000 | 0.980000 |" in markdown_path.read_text()
