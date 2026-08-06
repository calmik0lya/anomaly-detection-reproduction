import os

import run as run_module

ROOT = run_module.ROOT


def test_run_patchcore_substitutes_config_into_command(capsys):
    config = run_module.load_yaml(os.path.join(ROOT, "configs", "patchcore.yaml"))
    config["category"] = "cable"  # проверяем, что подстановка реально идёт из конфига,
    config["patchsize"] = 5       # а не захардкожена

    run_module.run_patchcore(config, py="fake-python", mvtec_path="/fake/mvtec", dry_run=True)

    printed = capsys.readouterr().out
    assert "fake-python bin/run_patchcore.py" in printed
    assert "--log_group IM224_WR50_cable" in printed
    assert "-d cable mvtec /fake/mvtec" in printed
    assert "--patchsize 5" in printed
    assert "--pretrain_embed_dimension 1024" in printed


def test_run_stfpm_action_test_only_runs_test_command(capsys):
    config = run_module.load_yaml(os.path.join(ROOT, "configs", "stfpm.yaml"))
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
    config = run_module.load_yaml(os.path.join(ROOT, "configs", "stfpm.yaml"))
    config["epochs"] = 7

    run_module.run_stfpm(config, py="fake-python", mvtec_path="/fake/mvtec",
                          action="train", dry_run=True)

    printed = capsys.readouterr().out
    assert "main.py train" in printed
    assert "--epochs 7" in printed
    assert "main.py test" not in printed
