import json

import sweep


def test_completed_pairs_only_accepts_non_empty_metrics(tmp_path):
    complete = tmp_path / "patchcore__bottle__20260807_120000"
    incomplete = tmp_path / "draem__cable__20260807_130000"
    broken = tmp_path / "stfpm__grid__20260807_140000"
    complete.mkdir()
    incomplete.mkdir()
    broken.mkdir()
    (complete / "metrics.json").write_text(json.dumps({"image_auroc": 0.99}))
    (incomplete / "metrics.json").write_text("{}")
    (broken / "metrics.json").write_text("not json")

    assert sweep.completed_pairs(tmp_path) == {("patchcore", "bottle")}


def test_sweep_uses_all_15_categories():
    categories = sweep.load_categories()
    assert len(categories) == 15
    assert "bottle" in categories
    assert "transistor" in categories
