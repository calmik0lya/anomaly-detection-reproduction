#!/usr/bin/env python3
"""
Простая обёртка над уже готовым и рабочим кодом пяти моделей.

Ничего не меняет в самих моделях (patchcore-inspection/, padim_run/, STFPM/,
SimpleNet/, DRAEM/) — просто вызывает те же самые команды, которые уже
использовались для получения результатов, и вытаскивает метрики из тех же
файлов/логов, куда модели сами их пишут.

Все параметры запуска (пути, гиперпараметры, epochs и т.д.) лежат в
YAML-конфигах в configs/ — по одному файлу на модель плюс общий
configs/paths.yaml с путями к датасету и питон-окружениям.

После каждого запуска run.py сохраняет использованный конфиг, метрики и
найденные файлы весов в experiments/<model>__<category>__<timestamp>/.

Использование:
    python run.py --config configs/patchcore.yaml
    python run.py --config configs/stfpm.yaml --epochs 100
    python run.py --config configs/draem.yaml --epochs 8 --action test
    python run.py --config configs/patchcore.yaml --dry-run   # только показать команду
"""
import argparse
import csv
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime

import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIGS_DIR = os.path.join(ROOT, "configs")
EXPERIMENTS_DIR = os.path.join(ROOT, "experiments")

BASE_ENV = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE")


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def run_cmd(cmd, cwd, env, dry_run):
    printable = " ".join(cmd)
    print(f"[run.py] cwd={cwd}")
    print(f"[run.py] {printable}")
    if dry_run:
        return None
    result = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    print(result.stdout[-4000:])
    if result.returncode != 0:
        print(result.stderr[-4000:], file=sys.stderr)
        raise RuntimeError(f"Команда завершилась с ошибкой (код {result.returncode})")
    return result.stdout


def read_auroc_csv(csv_path, category):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    row = next(r for r in rows if r["Row Names"] == f"mvtec_{category}")
    return {
        "image_auroc": float(row["instance_auroc"]),
        "pixel_auroc": float(row["full_pixel_auroc"]),
        "pro_auroc": float(row["anomaly_pixel_auroc"]),
    }


def parse_metrics(source, cwd, fmt, stdout):
    """Достаёт метрики из того места, куда модель сама их пишет.

    source — это поле metrics_source из конфига модели.
    """
    kind = source["type"]

    if kind == "csv":
        csv_path = os.path.join(cwd, source["path_template"].format(**fmt))
        return read_auroc_csv(csv_path, fmt["category"])

    if kind == "json":
        json_path = os.path.join(cwd, source["path"].format(**fmt))
        with open(json_path) as f:
            data = json.load(f)
        data = data[0] if isinstance(data, list) else data
        return {out_key: data.get(in_key) for out_key, in_key in source["fields"].items()}

    if kind == "regex_stdout":
        if stdout is None:
            raise RuntimeError("Нет stdout для парсинга метрик")
        if "pattern" in source:
            match = re.search(source["pattern"], stdout)
            if not match:
                raise RuntimeError(f"Не нашла метрики по паттерну: {source['pattern']}")
            return {field: float(v) for field, v in zip(source["fields"], match.groups())}
        result = {}
        for field, pattern in source["patterns"].items():
            match = re.search(pattern, stdout)
            if not match:
                raise RuntimeError(f"Не нашла метрику {field} по паттерну: {pattern}")
            result[field] = float(match.group(1))
        return result

    raise ValueError(f"Неизвестный тип metrics_source: {kind}")


def collect_weight_files(cwd, weight_glob, fmt):
    files = []
    for pattern in weight_glob:
        resolved = pattern.format(**fmt)
        files.extend(m for m in glob.glob(os.path.join(cwd, resolved)) if os.path.isfile(m))
    return sorted(set(files))


def save_experiment(model_name, category, config, metrics, cwd, fmt, timestamp):
    exp_dir = os.path.join(EXPERIMENTS_DIR, f"{model_name}__{category}__{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    with open(os.path.join(exp_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump(metrics or {}, f, indent=2, ensure_ascii=False)

    weight_files = collect_weight_files(cwd, config.get("weight_glob", []), fmt)
    if weight_files:
        weights_dir = os.path.join(exp_dir, "weights")
        for src in weight_files:
            dst = os.path.join(weights_dir, os.path.relpath(src, cwd))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
        print(f"[run.py] Скопировано файлов весов: {len(weight_files)}")
    else:
        print("[run.py] Файлы весов по weight_glob не найдены (возможно, dry-run "
              "или модель их не сохраняла)")

    return exp_dir


def run_patchcore(config, py, mvtec_path, dry_run):
    cwd = os.path.join(ROOT, config["working_dir"])
    category = config["category"]
    log_group = config["log_group_template"].format(category=category)
    results_dir = config["results_dir_template"].format(category=category)
    fmt = dict(config, category=category, log_group=log_group, results_dir=results_dir)

    cmd = [py, config["script"], "--seed", str(config["seed"])]
    if config.get("save_patchcore_model"):
        cmd.append("--save_patchcore_model")
    cmd += ["--log_group", log_group, "--log_project", config["log_project"], results_dir,
            "patch_core", "-b", config["backbone"]]
    for layer in config["layers_to_extract_from"]:
        cmd += ["-le", layer]
    cmd += ["--pretrain_embed_dimension", str(config["pretrain_embed_dimension"]),
            "--target_embed_dimension", str(config["target_embed_dimension"]),
            "--anomaly_scorer_num_nn", str(config["anomaly_scorer_num_nn"]),
            "--patchsize", str(config["patchsize"]),
            "sampler", "-p", str(config["sampler_percentage"]), config["sampler"],
            "dataset", "--num_workers", str(config["num_workers"]),
            "--resize", str(config["resize"]), "--imagesize", str(config["imagesize"]),
            "-d", category, "mvtec", mvtec_path]

    env = dict(BASE_ENV, **config.get("env", {}))
    run_cmd(cmd, cwd, env, dry_run)
    metrics = None if dry_run else parse_metrics(config["metrics_source"], cwd, fmt, None)
    return cwd, fmt, metrics


def run_padim(config, py, dry_run):
    cwd = os.path.join(ROOT, config["working_dir"])
    category = config["category"]
    if category != "bottle":
        raise ValueError(
            "run_padim.py сейчас поддерживает только category=bottle (категория и путь "
            "к датасету зашиты внутри скрипта, а не читаются из конфига). Чтобы прогнать "
            "другую категорию, нужно поправить эти строки в padim_run/run_padim.py вручную."
        )
    fmt = dict(config, category=category)
    env = dict(BASE_ENV, **config.get("env", {}))
    cmd = [py, config["script"]]
    run_cmd(cmd, cwd, env, dry_run)
    metrics = None if dry_run else parse_metrics(config["metrics_source"], cwd, fmt, None)
    return cwd, fmt, metrics


def run_stfpm(config, py, mvtec_path, action, dry_run):
    cwd = os.path.join(ROOT, config["working_dir"])
    category = config["category"]
    epochs = config["epochs"]
    model_save_path = config["model_save_path"]
    fmt = dict(config, category=category, epochs=epochs, model_save_path=model_save_path)
    env = dict(BASE_ENV, **config.get("env", {}))

    if action in ("train", "all"):
        train_cmd = [py, config["script"], "train",
                     "--mvtec-ad", mvtec_path, "--category", category,
                     "--epochs", str(epochs), "--model-save-path", model_save_path]
        run_cmd(train_cmd, cwd, env, dry_run)

    metrics = None
    if action in ("test", "all"):
        checkpoint = f"{model_save_path}/{category}/best.pth.tar"
        test_cmd = [py, config["script"], "test",
                    "--mvtec-ad", mvtec_path, "--category", category,
                    "--checkpoint", checkpoint]
        stdout = run_cmd(test_cmd, cwd, env, dry_run)
        if not dry_run:
            metrics = parse_metrics(config["metrics_source"], cwd, fmt, stdout)
    return cwd, fmt, metrics


def run_simplenet(config, py, mvtec_path, dry_run):
    cwd = os.path.join(ROOT, config["working_dir"])
    category = config["category"]
    log_group = config["log_group_template"].format(category=category)
    results_dir = config["results_dir_template"].format(category=category)
    fmt = dict(config, category=category, log_group=log_group, results_dir=results_dir)

    cmd = [py, config["script"], "--seed", str(config["seed"]),
           "--log_group", log_group, "--log_project", config["log_project"],
           "--results_path", results_dir, "--run_name", config["run_name"],
           "net", "-b", config["backbone"]]
    for layer in config["layers_to_extract_from"]:
        cmd += ["-le", layer]
    cmd += ["--pretrain_embed_dimension", str(config["pretrain_embed_dimension"]),
            "--target_embed_dimension", str(config["target_embed_dimension"]),
            "--patchsize", str(config["patchsize"]),
            "--meta_epochs", str(config["epochs"]),
            "--embedding_size", str(config["embedding_size"]),
            "--gan_epochs", str(config["gan_epochs"]),
            "--noise_std", str(config["noise_std"]),
            "--dsc_hidden", str(config["dsc_hidden"]),
            "--dsc_layers", str(config["dsc_layers"]),
            "--dsc_margin", str(config["dsc_margin"]),
            "--pre_proj", str(config["pre_proj"]),
            "dataset", "--num_workers", str(config["num_workers"]),
            "--batch_size", str(config["batch_size"]),
            "--resize", str(config["resize"]), "--imagesize", str(config["imagesize"]),
            "-d", category, "mvtec", mvtec_path]

    env = dict(BASE_ENV, **config.get("env", {}))
    run_cmd(cmd, cwd, env, dry_run)
    metrics = None if dry_run else parse_metrics(config["metrics_source"], cwd, fmt, None)
    return cwd, fmt, metrics


def run_draem(config, py, mvtec_path, dtd_path, action, dry_run):
    cwd = os.path.join(ROOT, config["working_dir"])
    category = config["category"]
    obj_list = config["obj_list"]
    if category not in obj_list:
        raise ValueError(f"Неизвестная категория для DRAEM: {category}")
    obj_id = obj_list.index(category)

    lr, bs, epochs = config["lr"], config["bs"], config["epochs"]
    checkpoint_path = config["checkpoint_path"]
    base_model_name = f"DRAEM_test_{lr}_{epochs}_bs{bs}"
    run_name = f"{base_model_name}_{category}_"
    fmt = dict(config, category=category, run_name=run_name, checkpoint_path=checkpoint_path)
    env = dict(BASE_ENV, **config.get("env", {}))

    if action in ("train", "all"):
        train_cmd = [py, config["train_script"], "--obj_id", str(obj_id), "--bs", str(bs),
                     "--lr", str(lr), "--epochs", str(epochs),
                     "--data_path", mvtec_path + "/", "--anomaly_source_path", dtd_path,
                     "--checkpoint_path", checkpoint_path, "--log_path", config["log_path"]]
        run_cmd(train_cmd, cwd, env, dry_run)

    if action not in ("test", "all"):
        return cwd, fmt, None
    if dry_run:
        print(f"[run.py] (test) в отдельном процессе Python вызвала бы "
              f"test(['{category}'], '{mvtec_path}/', '{checkpoint_path}', "
              f"'{base_model_name}')")
        return cwd, fmt, None

    # test_DRAEM.test() — обычная питоновская функция, не CLI, поэтому вызываем
    # её напрямую (в дочернем процессе через -c, чтобы не тянуть в run.py все
    # тяжёлые зависимости DRAEM/torch без необходимости).
    code = (
        "import sys, json, io, contextlib; sys.path.insert(0, '.'); "
        "from test_DRAEM import test; "
        "buf = io.StringIO()\n"
        "with contextlib.redirect_stdout(buf):\n"
        f"    test(['{category}'], '{mvtec_path}/', '{checkpoint_path}', "
        f"'{base_model_name}')\n"
        "print('===RUN_PY_CAPTURE_START===')\n"
        "print(buf.getvalue())\n"
        "print('===RUN_PY_CAPTURE_END===')\n"
    )
    test_cmd = [py, "-c", code]
    stdout = run_cmd(test_cmd, cwd, env, dry_run=False)
    metrics = parse_metrics(config["metrics_source"], cwd, fmt, stdout)
    return cwd, fmt, metrics


RUNNERS = {
    "patchcore": lambda config, paths, action, dry_run: run_patchcore(
        config, paths["python_envs"][config["python_env"]], paths["mvtec_path"], dry_run
    ),
    "padim": lambda config, paths, action, dry_run: run_padim(
        config, paths["python_envs"][config["python_env"]], dry_run
    ),
    "stfpm": lambda config, paths, action, dry_run: run_stfpm(
        config, paths["python_envs"][config["python_env"]], paths["mvtec_path"], action, dry_run
    ),
    "simplenet": lambda config, paths, action, dry_run: run_simplenet(
        config, paths["python_envs"][config["python_env"]], paths["mvtec_path"], dry_run
    ),
    "draem": lambda config, paths, action, dry_run: run_draem(
        config, paths["python_envs"][config["python_env"]], paths["mvtec_path"],
        paths["dtd_images_path"], action, dry_run
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True,
                         help="путь к YAML-конфигу модели, например configs/patchcore.yaml")
    parser.add_argument("--category", default=None, help="переопределить category из конфига")
    parser.add_argument("--epochs", type=int, default=None,
                         help="переопределить epochs из конфига (stfpm/simplenet/draem)")
    parser.add_argument("--action", default=None, choices=["train", "test", "all"],
                         help="переопределить action из конфига (для stfpm/draem — "
                              "можно прогнать только train или только test)")
    parser.add_argument("--dry-run", action="store_true",
                         help="только показать команду, которая будет вызвана, ничего не запускать")
    args = parser.parse_args()

    config = load_yaml(args.config)
    paths = load_yaml(os.path.join(CONFIGS_DIR, "paths.yaml"))
    paths = dict(paths, mvtec_path=os.path.expanduser(paths["mvtec_path"]),
                 dtd_images_path=os.path.expanduser(paths["dtd_images_path"]))

    if args.category:
        config["category"] = args.category
    if args.epochs is not None:
        config["epochs"] = args.epochs
    action = args.action or config.get("action", "all")

    model_name = config["model"]
    if model_name not in RUNNERS:
        raise ValueError(f"Неизвестная модель в конфиге: {model_name}")

    cwd, fmt, metrics = RUNNERS[model_name](config, paths, action, args.dry_run)

    if metrics:
        print("\n=== МЕТРИКИ ===")
        print(json.dumps(metrics, indent=2, ensure_ascii=False))

    if not args.dry_run:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = save_experiment(model_name, config["category"], config, metrics, cwd, fmt, timestamp)
        print(f"\n[run.py] Эксперимент сохранён в {exp_dir}")


if __name__ == "__main__":
    main()
