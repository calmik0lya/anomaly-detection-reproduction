# Anomaly Detection — воспроизведение 5 моделей на MVTec AD

Воспроизведение пяти методов anomaly detection на датасете MVTec AD (класс `bottle`):
**PatchCore, PaDiM, STFPM, SimpleNet, DRAEM**. Модели используют разные подходы —
сравнение с эталонными признаками, статистическое моделирование, дистилляция
teacher-student, синтетические аномалии в пространстве признаков и на уровне пикселей.
Каждая модель запущена как есть, без переписывания архитектуры; цели — получить
реальные метрики и сравнить их с результатами из оригинальных статей. Среда —
Mac без CUDA (CPU-only).

## Структура репозитория

Каждая модель — клон оригинального репозитория автора с точечными правками под
CPU/macOS. Логику моделей мы не трогаем — только оборачиваем их запуск общим
раннером `run.py`, конфигурируемым через YAML.

```
.
├── run.py                       # общий раннер: python run.py --config configs/<model>.yaml
├── configs/                     # YAML-конфиги: paths.yaml + по одному файлу на модель
├── experiments/                 # результаты запусков run.py (config/metrics/weights)
├── tests/                       # pytest: валидность конфигов, сборка команд
├── requirements*.txt            # зависимости run.py и двух conda-окружений моделей
├── results.md                   # метрики всех 5 моделей + сравнение с оригинальными статьями
├── TODO.md                      # открытые задачи
├── patchcore-inspection/        # 1. PatchCore (amazon-science/patchcore-inspection)
├── padim_run/                   # 2. PaDiM (через библиотеку anomalib)
├── STFPM/                       # 3. STFPM (gdwang08/STFPM)
├── SimpleNet/                    # 4. SimpleNet (DonaldRR/SimpleNet)
└── DRAEM/                        # 5. DRAEM (VitjanZ/DRAEM)
```

## Conda-окружения

Вместо пяти изолированных окружений (диска было жалко) обошлись двумя общими:

- **`patchcore`** — для PatchCore, STFPM, SimpleNet и DRAEM
- **`anomalib`** — отдельно для PaDiM, потому что у библиотеки `anomalib` своё большое
  дерево зависимостей (pytorch-lightning, kornia и т.д.), которое проще не смешивать
  с остальным

Пути к обоим интерпретаторам заданы в `configs/paths.yaml` — `run.py` сам вызывает
нужный, активировать conda-окружение вручную не нужно.

## Запуск моделей

Все параметры запуска (путь к датасету, гиперпараметры моделей, epochs, batch size
и т.д.) вынесены из кода в YAML-конфиги в `configs/` — по одному файлу на модель
(`configs/patchcore.yaml`, `configs/padim.yaml`, `configs/stfpm.yaml`,
`configs/simplenet.yaml`, `configs/draem.yaml`) плюс общий `configs/paths.yaml` с
путём к датасету и путями к обоим conda-окружениям.

```bash
python run.py --config configs/patchcore.yaml
python run.py --config configs/stfpm.yaml --epochs 50
python run.py --config configs/draem.yaml --category carpet --action test
```

Необязательные флаги `--category`, `--epochs` и `--action` переопределяют
соответствующие поля конфига поверх YAML (для STFPM/DRAEM `--action train|test|all`
позволяет прогнать только обучение или только тест). Флаг `--dry-run` печатает
итоговую команду и ничего не запускает — удобно проверить, что подставилось.

## Результаты запусков (`experiments/`)

После каждого запуска (кроме `--dry-run`) `run.py` сохраняет в
`experiments/<model>__<category>__<timestamp>/`:

- **`config.yaml`** — реально использованный конфиг, с учётом CLI-переопределений;
- **`metrics.json`** — метрики, распарсенные из того же `results.csv`/лога/stdout,
  куда модель сама их пишет;
- **`weights/`** — скопированные файлы весов модели (найдены по `weight_glob` из
  конфига этой модели), если модель дообучилась и что-то сохранила. Папки не будет,
  если запускался только тест без обучения или файлы весов не нашлись.

Сводные метрики по классу `bottle` для всех пяти моделей — по-прежнему в
[results.md](results.md).

## Зависимости

- **`requirements.txt`** — зависимости самого `run.py` (PyYAML) и тестов (`pytest`).
  Ставится в окружение, из которого вы запускаете `run.py`; активировать
  conda-окружения моделей для этого не нужно.
- **`requirements-patchcore.txt`** — зависимости conda-окружения `patchcore`
  (PatchCore, STFPM, SimpleNet, DRAEM).
- **`requirements-anomalib.txt`** — зависимости conda-окружения `anomalib` (PaDiM).

## Тесты

```bash
pip install -r requirements.txt
pytest
```

Тесты в `tests/` не требуют GPU, датасета или установленных conda-окружений
моделей — они проверяют:

- что каждый `configs/*.yaml` — валидный YAML с обязательными полями (`model`,
  `python_env`, `category`, `weight_glob`, `metrics_source` и т.д.), а `python_env`
  каждого конфига действительно есть в `configs/paths.yaml`;
- что сборка команды в `run.py` реально подставляет значения из конфига
  (категорию, epochs, гиперпараметры), а не использует захардкоженные — на примере
  `run_patchcore`/`run_stfpm`.

## Результаты

Все метрики (image-AUROC, pixel-AUROC и сравнение с цифрами из оригинальных статей) —
в **[results.md](results.md)**.

Открытые задачи см. в [TODO.md](TODO.md).
