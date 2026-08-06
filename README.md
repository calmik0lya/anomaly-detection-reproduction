Воспроизведение пяти методов anomaly detection на MVTec AD (класс bottle): PatchCore, PaDiM, STFPM, SimpleNet, DRAEM. Каждая модель — клон оригинального репозитория с точечными правками под CPU/macOS, логика не переписана. Cреда — Mac без CUDA.

## Структура

```
.
├── run.py                    # python run.py --config configs/<model>.yaml
├── configs/                  # paths.yaml + конфиг на модель
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

Пути к интерпретаторам — в `configs/paths.yaml`, `run.py` вызывает их сам.

## Запуск

```bash
python run.py --config configs/patchcore.yaml
python run.py --config configs/stfpm.yaml --epochs 50
python run.py --config configs/draem.yaml --category carpet --action test
```

`--category`, `--epochs`, `--action` переопределяют конфиг. `--dry-run` печатает команду без запуска.

## Результаты запусков

`experiments/<model>__<category>__<timestamp>/`:
- `config.yaml` — использованный конфиг
- `metrics.json` — метрики из results.csv/лога модели
- `weights/` — только файлы весов, найденные по `weight_glob`, без лишних вложенных папок

`experiments/` хранится локально и исключён из Git, как и тяжёлые outputs/checkpoints в типовых ML-проектах. Исходные конфиги из `configs/` остаются в репозитории.

Сводка по всем моделям — в `results.md`.

## Зависимости

- `requirements.txt` — для запуска `run.py` и тестов (PyYAML, pytest)
- `requirements-patchcore.txt` / `requirements-anomalib.txt` — под два conda-окружения

## Тесты

```bash
pip install -r requirements.txt
pytest
```

Проверяют валидность `configs/*.yaml` и корректность подстановки параметров в команду — без GPU и датасета.

Открытые задачи — в `TODO.md`.
