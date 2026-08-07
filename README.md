Воспроизведение пяти методов anomaly detection на 15 категориях MVTec AD: PatchCore, PaDiM, STFPM, SimpleNet, DRAEM. Каждая модель — клон оригинального репозитория с точечными правками, логика моделей не переписана. Локально можно проверять код на CPU/macOS, полные эксперименты рассчитаны на NVIDIA GPU в Kaggle.

## Структура

```
.
├── run.py                    # сборка YAML-конфигов и запуск моделей
├── sweep.py                  # пакетный запуск 15 категорий с resume
├── kaggle/                   # готовый Kaggle Notebook
├── configs/
│   ├── config.yaml          # группы по умолчанию
│   ├── models/              # гиперпараметры и веса каждой модели
│   ├── tasks/               # постановка anomaly detection
│   ├── metrics/             # источники и парсинг метрик
│   ├── runner/              # параметры папки экспериментов
│   └── paths/               # датасеты и Python-окружения
├── experiments/               # результаты запусков: config/metrics/weights
├── tests/                     # pytest
├── requirements*.txt
├── results.md                 # метрики + сравнение со статьями
├── TODO.md
├── patchcore-inspection/       # PatchCore
├── padim_run/                  # PaDiM (anomalib)
├── STFPM/
├── SimpleNet/
└── DRAEM/
```

## Conda-окружения

- `patchcore` — PatchCore, STFPM, SimpleNet, DRAEM
- `anomalib` — PaDiM (свой тяжёлый стек: pytorch-lightning, kornia)

Пути к интерпретаторам — в `configs/paths/local.yaml`, `run.py` вызывает их сам.

## Запуск

```bash
python run.py --config configs/config.yaml
python run.py --config configs/models/patchcore.yaml
python run.py --config configs/models/stfpm.yaml --epochs 50
python run.py --config configs/models/draem.yaml --category carpet --action test
```

`configs/config.yaml` выбирает группы по умолчанию. Переданный файл из `configs/models/`
заменяет модель и подключает одноимённый конфиг из `configs/metrics/`.
`--category`, `--epochs`, `--action` переопределяют собранный конфиг, а `--dry-run`
печатает команду без запуска.

`--paths` выбирает другой набор путей, `--device cuda` включает GPU через разные
CLI-интерфейсы моделей, а `--output-dir` задаёт место сохранения экспериментов.

Параметры PaDiM (путь к MVTec AD, category, backbone, layers, batch size и epochs)
также берутся из YAML, несмотря на отдельное окружение `anomalib`.

## Результаты запусков

После успешного реального запуска создаётся
`experiments/<model>__<category>__<timestamp>/`:

```text
experiments/stfpm__bottle__20260806_194545/
├── config.yaml
├── metrics.json
└── weights/
    └── best.pth.tar
```

- `config.yaml` — полностью собранный конфиг (`task/model/metrics/runner/paths`)
- `metrics.json` — метрики из results.csv/лога модели
- `weights/` — только файлы весов, найденные по `weight_glob`, без лишних вложенных папок

При `--dry-run` или ошибке модели ложный успешный эксперимент не создаётся. Для
`--action train` без последующего тестирования `metrics.json` остаётся пустым до
получения итоговых метрик.

`experiments/` хранится локально и исключён из Git, как и тяжёлые outputs/checkpoints в типовых ML-проектах. Исходные конфиги из `configs/` остаются в репозитории.

Сводка по всем моделям — в `results.md`.

## Kaggle и запуск 15 категорий

Готовый Notebook: [`kaggle/anomaly_detection_mvtec.ipynb`](kaggle/anomaly_detection_mvtec.ipynb).
В Kaggle нужно включить GPU и подключить два Dataset: MVTec AD и DTD. Notebook
проверяет CUDA и пути, устанавливает зависимости и сначала запускает одну пару
PatchCore/bottle.

Полный последовательный запуск:

```bash
python sweep.py \
  --paths configs/paths/kaggle.yaml \
  --device cuda \
  --output-dir /kaggle/working/experiments \
  --resume --continue-on-error
```

`--resume` ищет непустой `metrics.json` и пропускает уже завершённые пары.
Текущее состояние записывается в `experiments/sweep_state.json`. Из-за лимита
сессии разумно запускать модели группами, например `--models patchcore padim`,
сохранять Kaggle Version вместе с output и продолжать той же командой. В новой
сессии предыдущий output нужно подключить как Dataset; Notebook содержит ячейку,
которая восстанавливает experiment-папки в `/kaggle/working/experiments`.

## Зависимости

- `requirements.txt` — для запуска `run.py` и тестов (PyYAML, pytest)
- `requirements-patchcore.txt` / `requirements-anomalib.txt` — под два conda-окружения

## Тесты

```bash
pip install -r requirements.txt
pytest
```

Проверяют валидность всех групп в `configs/`, композицию полного конфига для каждой
модели и корректность подстановки параметров в команду — без GPU и датасета.

Открытые задачи — в `TODO.md`.
