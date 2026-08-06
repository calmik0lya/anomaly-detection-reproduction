import os
from pathlib import Path

import run as run_module

ROOT = run_module.ROOT


def test_run_patchcore_substitutes_config_into_command(capsys):
    resolved = run_module.load_experiment_config(
        os.path.join(ROOT, "configs", "models", "patchcore.yaml")
    )
    config = run_module.flatten_experiment_config(resolved)
    config["category"] = "cable"  # проверяем, что подстановка реально идёт из конфига,
    config["patchsize"] = 5       # а не захардкожена

    run_module.run_patchcore(config, py="fake-python", mvtec_path="/fake/mvtec", dry_run=True)

    printed = capsys.readouterr().out
    assert "fake-python bin/run_patchcore.py" in printed
    assert "--log_group IM224_WR50_cable" in printed
    assert "-d cable mvtec /fake/mvtec" in printed
    assert "--patchsize 5" in printed
    assert "--faiss_num_workers 1" in printed
    assert "--pretrain_embed_dimension 1024" in printed


def test_next_iterated_patchcore_log_group(tmp_path):
    project_dir = tmp_path / "results" / "project"
    (project_dir / "group").mkdir(parents=True)
    (project_dir / "group_0").mkdir()

    actual = run_module.next_iterated_log_group(
        str(tmp_path), "results", "project", "group",
    )

    assert actual == "group_1"


def test_collect_weight_files_deduplicates_symlinks(tmp_path):
    source = tmp_path / "versions" / "v1" / "model.ckpt"
    source.parent.mkdir(parents=True)
    source.write_text("weights")
    latest = tmp_path / "latest"
    latest.symlink_to(source.parent, target_is_directory=True)

    files = run_module.collect_weight_files(
        str(tmp_path), ["versions/*/*.ckpt", "latest/*.ckpt"], {},
    )

    assert files == [str(source.resolve())]


def test_next_available_simplenet_run_name(tmp_path):
    group_dir = tmp_path / "results" / "project" / "group"
    (group_dir / "run").mkdir(parents=True)
    (group_dir / "run_0").mkdir()

    actual = run_module.next_available_run_name(
        str(tmp_path), "results", "project", "group", "run",
    )

    assert actual == "run_1"


def test_run_stfpm_action_test_only_runs_test_command(capsys):
    resolved = run_module.load_experiment_config(
        os.path.join(ROOT, "configs", "models", "stfpm.yaml")
    )
    config = run_module.flatten_experiment_config(resolved)
    config["category"] = "carpet"
    config["epochs"] = 42

    run_module.run_stfpm(config, py="fake-python", mvtec_path="/fake/mvtec",
                          action="test", dry_run=True)

    printed = capsys.readouterr().out
    assert "main.py train" not in printed
    assert "main.py test" in printed
    assert "--category carpet" in printed
    assert "--checkpoint snapshots/carpet/best.pth.tar" in printed


def test_run_stfpm_action_train_only_runs_train_command(capsys):
    resolved = run_module.load_experiment_config(
        os.path.join(ROOT, "configs", "models", "stfpm.yaml")
    )
    config = run_module.flatten_experiment_config(resolved)
    config["epochs"] = 7

    run_module.run_stfpm(config, py="fake-python", mvtec_path="/fake/mvtec",
                          action="train", dry_run=True)

    printed = capsys.readouterr().out
    assert "main.py train" in printed
    assert "--epochs 7" in printed
    assert "main.py test" not in printed


def test_run_padim_substitutes_dataset_and_model_config(capsys):
    resolved = run_module.load_experiment_config(
        os.path.join(ROOT, "configs", "models", "padim.yaml")
    )
    config = run_module.flatten_experiment_config(resolved)
    config["category"] = "carpet"
    config["train_batch_size"] = 4

    run_module.run_padim(
        config, py="fake-python", mvtec_path="/fake/mvtec", dry_run=True,
    )

    printed = capsys.readouterr().out
    assert "fake-python -c" in printed
    assert "root='/fake/mvtec'" in printed
    assert "category='carpet'" in printed
    assert "train_batch_size=4" in printed
    assert "backbone='resnet18'" in printed


def test_save_experiment_has_only_config_metrics_and_flat_weights(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    first_dir = model_dir / "results" / "run_a"
    second_dir = model_dir / "results" / "run_b"
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)
    (first_dir / "model.ckpt").write_text("first")
    (second_dir / "model.ckpt").write_text("second")

    experiments_dir = tmp_path / "experiments"
    monkeypatch.setattr(run_module, "EXPERIMENTS_DIR", str(experiments_dir))
    config = {
        "task": {"category": "bottle"},
        "model": {
            "name": "example",
            "weight_glob": ["results/*/*.ckpt"],
        },
        "metrics": {"source": {"type": "json"}},
        "runner": {"output_dir": "experiments"},
        "paths": {},
    }

    exp_dir = Path(run_module.save_experiment(
        "example", "bottle", config, {"image_auroc": 0.9},
        str(model_dir), config, "20260806_120000",
    ))

    assert {path.name for path in exp_dir.iterdir()} == {
        "config.yaml", "metrics.json", "weights",
    }
    saved_config = run_module.load_yaml(exp_dir / "config.yaml")
    assert set(saved_config) == {"task", "model", "metrics", "runner", "paths"}
    assert saved_config["model"]["name"] == "example"
    assert saved_config["metrics"]["source"]["type"] == "json"
    weight_paths = list((exp_dir / "weights").iterdir())
    assert len(weight_paths) == 2
    assert all(path.is_file() for path in weight_paths)
    assert not any(path.is_dir() for path in weight_paths)
