# Воспроизведение методов anomaly detection на MVTec AD

В репозитории воспроизведены пять методов промышленного поиска аномалий:
**PatchCore, PaDiM, STFPM, SimpleNet и DRAEM**. Эксперименты проведены на всех
15 категориях MVTec AD — всего 75 пар «модель/категория».

Исходная логика моделей не объединялась и не переписывалась. Каждая реализация
находится в `models/`, а общая обвязка отвечает за конфигурацию, запуск и
одинаковое сохранение результатов. Полные прогоны выполнялись на NVIDIA Tesla
P100 в Kaggle; локально проект можно использовать для проверки конфигов, тестов
и отдельных запусков.

## Что уже реализовано

- все параметры экспериментов вынесены в YAML;
- поддержан одиночный запуск модели и полный прогон по 15 категориям;
- завершённые эксперименты можно пропускать с помощью `--resume`;
- после каждого успешного запуска вместе сохраняются использованный конфиг,
  итоговые метрики и найденные веса;
- метрики всех запусков собираются в общую CSV- и Markdown-таблицу;
- для Kaggle подготовлен отдельный Notebook с проверкой GPU и восстановлением
  результатов предыдущей сессии.

## Структура проекта

```text
.
├── run.py                  # запуск одной модели по YAML-конфигу
├── sweep.py                # последовательный запуск моделей и категорий
├── aggregate_results.py    # сбор метрик из experiment-папок
├── configs/
│   ├── config.yaml         # состав эксперимента по умолчанию
│   ├── models/             # гиперпараметры пяти моделей и weight_glob
│   ├── tasks/              # задача и список 15 категорий MVTec AD
│   ├── metrics/            # правила извлечения метрик
│   ├── runner/             # общие параметры запуска
│   └── paths/              # локальные и Kaggle-пути
├── models/                 # исходные реализации моделей
├── kaggle/                 # Notebook для запуска на GPU
├── experiments/            # config.yaml, metrics.json и weights каждого запуска
├── outputs/                # итоговые Excel-таблицы
├── tests/                  # тесты конфигов, команд и агрегации
└── results.md              # сводка результатов
```

Тяжёлые веса и experiment-папки не хранятся в Git. После обучения они были
сохранены в приватных Kaggle Datasets, а в репозитории остаются код, конфиги и
инструменты для полного воспроизведения.

## Как устроены конфиги

Корневой `configs/config.yaml` собирает эксперимент из нескольких независимых
частей:

- `tasks/anomaly_detection.yaml` — категория, действие и полный список классов;
- `models/<model>.yaml` — архитектура, epochs, batch size и остальные параметры;
- `metrics/<model>.yaml` — откуда и как извлекать метрики;
- `runner/default.yaml` — папка результатов;
- `paths/local.yaml` или `paths/kaggle.yaml` — датасеты и Python-окружения.

В конфиге каждой модели есть `weight_glob`. По этим шаблонам `run.py` находит
реальные checkpoint-файлы исходной реализации и копирует их в папку
эксперимента.

## Запуск одной модели

```bash
python run.py --config configs/models/patchcore.yaml
python run.py --config configs/models/stfpm.yaml --category cable --epochs 50
python run.py --config configs/models/draem.yaml --category carpet --action test
```

Параметры `--category`, `--epochs` и `--action` переопределяют значения YAML.
Полезные дополнительные флаги:

- `--paths configs/paths/kaggle.yaml` — выбрать другой набор путей;
- `--device cuda` — запустить модель на GPU;
- `--output-dir experiments` — изменить папку экспериментов;
- `--dry-run` — показать итоговую команду без запуска.

## Запуск всех 15 категорий

```bash
python sweep.py \
  --paths configs/paths/kaggle.yaml \
  --device cuda \
  --output-dir /kaggle/working/experiments \
  --resume \
  --continue-on-error
```

По умолчанию `sweep.py` последовательно запускает пять моделей на всех категориях.
При необходимости можно выбрать только часть матрицы:

```bash
python sweep.py \
  --models patchcore stfpm \
  --categories bottle cable capsule \
  --paths configs/paths/kaggle.yaml \
  --device cuda \
  --resume
```

`--resume` проверяет непустой `metrics.json` и не повторяет уже завершённые пары.
Текущее состояние записывается в `experiments/sweep_state.json`.

## Результат каждого запуска

После успешного обучения и тестирования создаётся отдельная папка с моделью,
категорией и временем запуска:

```text
experiments/stfpm__bottle__20260807_152039/
├── config.yaml
├── metrics.json
└── weights/
    └── best.pth.tar
```

- `config.yaml` — фактически использованные task/model/metrics/runner/paths;
- `metrics.json` — итоговые image-AUROC, pixel-AUROC и другие доступные метрики;
- `weights/` — веса, найденные по `weight_glob`, без лишней структуры каталогов.

При ошибке модели папка успешного эксперимента не создаётся. `--dry-run` также
ничего не сохраняет.

## Сводная таблица метрик

После восстановления experiment-папок результаты собираются одной командой:

```bash
python aggregate_results.py \
  --experiments-dir experiments \
  --csv results.csv \
  --markdown results.md
```

Скрипт берёт последний успешный запуск каждой пары «модель/категория», пропускает
повреждённые и незавершённые папки, рассчитывает средние значения по категориям и
формирует подробную таблицу. Результаты этого проекта и значения из статей
сравниваются как средние по всем 15 категориям MVTec AD.

Готовые таблицы находятся в `outputs/`:

- [`Результаты_5_моделей_15_категорий.xlsx`](outputs/results-table/Результаты_5_моделей_15_категорий.xlsx) — сравнение средних метрик;
- [`Литературный_обзор_20_статей.xlsx`](outputs/literature-review/Литературный_обзор_20_статей.xlsx) — обзор связанных методов и исследований.

## Kaggle

Готовый Notebook находится в
[`kaggle/anomaly_detection_mvtec.ipynb`](kaggle/anomaly_detection_mvtec.ipynb).
Перед запуском нужно включить GPU и подключить MVTec AD. Датасет DTD требуется
только для DRAEM, где его текстуры используются для создания синтетических
дефектов.

Из-за ограничения длительности Kaggle-сессии модели удобно запускать группами.
После каждой группы результаты следует сохранять как приватный Kaggle Dataset. В
новой сессии Notebook восстанавливает experiment-папки, после чего та же команда
с `--resume` продолжает вычисления.

## Окружения и зависимости

Локально используются два окружения:

- `patchcore` — PatchCore, STFPM, SimpleNet и DRAEM;
- `anomalib` — PaDiM.

Пути к их интерпретаторам задаются в `configs/paths/local.yaml`. В Kaggle все
модели запускаются в одном GPU-окружении.

- `requirements.txt` — зависимости общей обвязки и тестов;
- `requirements-patchcore.txt` — зависимости первого локального окружения;
- `requirements-anomalib.txt` — стек PaDiM/anomalib;
- `requirements-kaggle.txt` — совместимое окружение для Kaggle Tesla P100.

## Тесты

```bash
pip install -r requirements.txt
pytest
```

Тесты не требуют GPU и MVTec AD. Они проверяют структуру YAML, сборку полного
конфига, подстановку параметров в команды, шаблоны весов и агрегацию метрик.

Открытые задачи перечислены в [`TODO.md`](TODO.md).
