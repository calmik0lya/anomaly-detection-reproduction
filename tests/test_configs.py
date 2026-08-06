import glob
import os

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS_DIR = os.path.join(ROOT, "configs")

MODEL_CONFIG_FILES = sorted(
    f for f in glob.glob(os.path.join(CONFIGS_DIR, "*.yaml"))
    if os.path.basename(f) != "paths.yaml"
)

REQUIRED_MODEL_FIELDS = [
    "model", "python_env", "working_dir", "category", "weight_glob", "metrics_source",
]


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def paths_config():
    return load_yaml(os.path.join(CONFIGS_DIR, "paths.yaml"))


def test_model_configs_exist():
    names = {os.path.basename(f) for f in MODEL_CONFIG_FILES}
    assert names == {"patchcore.yaml", "padim.yaml", "stfpm.yaml", "simplenet.yaml", "draem.yaml"}


@pytest.mark.parametrize("config_path", MODEL_CONFIG_FILES, ids=os.path.basename)
def test_config_is_valid_yaml_mapping(config_path):
    config = load_yaml(config_path)
    assert isinstance(config, dict)


@pytest.mark.parametrize("config_path", MODEL_CONFIG_FILES, ids=os.path.basename)
def test_model_config_has_required_fields(config_path):
    config = load_yaml(config_path)
    for field in REQUIRED_MODEL_FIELDS:
        assert field in config, f"{os.path.basename(config_path)}: отсутствует обязательное поле '{field}'"


@pytest.mark.parametrize("config_path", MODEL_CONFIG_FILES, ids=os.path.basename)
def test_model_field_matches_filename(config_path):
    config = load_yaml(config_path)
    expected_model = os.path.splitext(os.path.basename(config_path))[0]
    assert config["model"] == expected_model


@pytest.mark.parametrize("config_path", MODEL_CONFIG_FILES, ids=os.path.basename)
def test_weight_glob_is_nonempty_list_of_strings(config_path):
    config = load_yaml(config_path)
    weight_glob = config["weight_glob"]
    assert isinstance(weight_glob, list) and weight_glob
    assert all(isinstance(pattern, str) and pattern for pattern in weight_glob)


@pytest.mark.parametrize("config_path", MODEL_CONFIG_FILES, ids=os.path.basename)
def test_metrics_source_has_known_type(config_path):
    config = load_yaml(config_path)
    assert config["metrics_source"]["type"] in {"csv", "json", "regex_stdout"}


@pytest.mark.parametrize("config_path", MODEL_CONFIG_FILES, ids=os.path.basename)
def test_python_env_is_defined_in_paths(config_path, paths_config):
    config = load_yaml(config_path)
    assert config["python_env"] in paths_config["python_envs"]


def test_paths_config_has_required_fields(paths_config):
    assert "mvtec_path" in paths_config
    assert "dtd_images_path" in paths_config
    assert "python_envs" in paths_config
    assert {"patchcore", "anomalib"} <= set(paths_config["python_envs"])
