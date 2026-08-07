import glob
import os

import pytest
import yaml

import run as run_module

ROOT = run_module.ROOT
CONFIGS_DIR = run_module.CONFIGS_DIR
MODEL_CONFIG_FILES = sorted(glob.glob(os.path.join(CONFIGS_DIR, "models", "*.yaml")))
METRICS_CONFIG_FILES = sorted(glob.glob(os.path.join(CONFIGS_DIR, "metrics", "*.yaml")))

REQUIRED_MODEL_FIELDS = ["name", "python_env", "working_dir", "weight_glob"]
EXPECTED_MODELS = {"patchcore", "padim", "stfpm", "simplenet", "draem"}


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def test_expected_config_groups_exist():
    assert set(run_module.CONFIG_GROUP_DIRS) == {"task", "model", "metrics", "runner", "paths"}
    for directory in run_module.CONFIG_GROUP_DIRS.values():
        assert os.path.isdir(os.path.join(CONFIGS_DIR, directory))


def test_root_config_selects_every_group():
    root_config = load_yaml(os.path.join(CONFIGS_DIR, "config.yaml"))
    assert set(root_config["defaults"]) == set(run_module.CONFIG_GROUP_DIRS)


def test_model_and_metrics_configs_exist_for_every_model():
    model_names = {os.path.splitext(os.path.basename(path))[0] for path in MODEL_CONFIG_FILES}
    metrics_names = {os.path.splitext(os.path.basename(path))[0] for path in METRICS_CONFIG_FILES}
    assert model_names == EXPECTED_MODELS
    assert metrics_names == EXPECTED_MODELS


@pytest.mark.parametrize("config_path", MODEL_CONFIG_FILES + METRICS_CONFIG_FILES,
                         ids=lambda path: os.path.relpath(path, CONFIGS_DIR))
def test_config_is_valid_yaml_mapping(config_path):
    assert isinstance(load_yaml(config_path), dict)


@pytest.mark.parametrize("config_path", MODEL_CONFIG_FILES, ids=os.path.basename)
def test_model_config_has_required_fields(config_path):
    config = load_yaml(config_path)
    for field in REQUIRED_MODEL_FIELDS:
        assert field in config, f"{os.path.basename(config_path)}: нет поля '{field}'"
    assert config["name"] == os.path.splitext(os.path.basename(config_path))[0]
    assert isinstance(config["weight_glob"], list) and config["weight_glob"]


@pytest.mark.parametrize("config_path", MODEL_CONFIG_FILES, ids=os.path.basename)
def test_each_model_composes_to_complete_runtime_config(config_path):
    resolved = run_module.load_experiment_config(config_path)
    flat = run_module.flatten_experiment_config(resolved)

    assert set(resolved) == set(run_module.CONFIG_GROUP_DIRS)
    assert flat["model"] == os.path.splitext(os.path.basename(config_path))[0]
    assert flat["category"] == "bottle"
    assert flat["metrics_source"]["type"] in {"csv", "json", "regex_stdout"}
    assert flat["python_env"] in resolved["paths"]["python_envs"]


def test_default_config_composes_patchcore():
    resolved = run_module.load_experiment_config(os.path.join(CONFIGS_DIR, "config.yaml"))
    flat = run_module.flatten_experiment_config(resolved)
    assert flat["model"] == "patchcore"
    assert flat["category"] == "bottle"


def test_paths_config_has_required_fields():
    for name in ("local.yaml", "kaggle.yaml"):
        paths = load_yaml(os.path.join(CONFIGS_DIR, "paths", name))
        assert {"mvtec_path", "dtd_images_path", "python_envs"} <= set(paths)
        assert {"patchcore", "anomalib"} <= set(paths["python_envs"])


def test_task_lists_all_mvtec_categories():
    task = load_yaml(os.path.join(CONFIGS_DIR, "tasks", "anomaly_detection.yaml"))
    assert len(task["categories"]) == 15
    assert len(set(task["categories"])) == 15
    assert task["category"] in task["categories"]
