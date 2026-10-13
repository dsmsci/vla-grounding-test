# GR00T N1.7 × LIBERO: причинный MVP

Исследуем, можно ли вмешательствами во внутренние активации готовой GR00T предсказуемо менять **объект действия (T)** или **место назначения (G)** и сохранять правильное завершение задачи.

**RQ:** какие функциональные интерфейсы и блоки модели позволяют управлять этим выбором? H1 проверяем через прямой/обратный patch и сохранность остальных требований. H2 - через текстовые, визуальные и совместные изменения. H3 о композиции T+G в этом MVP пока не проверяется.

Методы: activation patching, отдельное additive steering, self/paraphrase/random-контроли, debiased CKA и выборочные attention-карты.

## Эксперимент

Готовый checkpoint - `nvidia/GR00T-N1.7-LIBERO`; Object и Goal используют свои suite-веса. `T_native`: суп/соус → корзина; `G_native`: миска → тарелка/плита. Конфигурация `color` добавляет одинаковые красные/синие банки и тарелки для проверки визуала.

Один seed **17**, 16 development и 30 отложенных test-сцен на семейство. Места заранее фиксированы: backbone, context и пары текстового/визуального cross-attention на ранней, средней и поздней глубине DiT. На development сравниваем все места и подбираем steering; на test проверяем выбранное по поведению место для каждого контраста без нового подбора.

Этапы: `smoke → baseline/diagnose → patch → steer → test`. Команда `develop` объединяет patch и steer, включая недостающие baseline и диагностику. Полный протокол, метрики и обоснование участков - [docs/protocol.md](docs/protocol.md).

## Структура

- `run.py` - Python Fire; запуск этапов.
- `conf/mvp.yaml`, `conf/color.yaml` - настройки Hydra и стандартные overrides.
- `src/` - среда, native policy, вмешательства, этапы, MLflow и статистика.
- `pyproject.toml` - Python и зависимости для uv.
- `docs/protocol.md`, `docs/NOTICE.md` - протокол и атрибуция.
- `results/`, `mlflow/`, `videos/` - внешние данные прогонов.

Описание каждого модуля находится в протоколе.

## Установка

Нужны Ubuntu/Linux x86_64, рабочий NVIDIA driver и GPU с BF16. По умолчанию PyTorch использует CUDA runtime 12.8 из wheels.

После установки uv и системных пакетов Ubuntu (список в протоколе), из корня репозитория:

```bash
export CC=gcc CXX=g++
uv sync
uv run python run.py setup
uv run python run.py doctor
uv run python run.py simcheck
```

`uv sync` автоматически скачает Python 3.12 по `requires-python` из TOML, если подходящего Python нет. Перед загрузкой весов передайте HF token с доступом к Cosmos-Reason2-2B:

```bash
read -rsp 'HF_TOKEN: ' HF_TOKEN
export HF_TOKEN
uv run python run.py download
```

Токен читается стандартной библиотекой Hugging Face и не входит в конфигурацию, метрики или образ.

## Базовый запуск

```bash
uv run python run.py smoke
uv run python run.py develop
uv run python run.py test
```

Команды выполняются последовательно. Для цветных семейств добавьте `--config=color`. Все остальные параметры берутся из Hydra. Пример изменения одного параметра: `uv run python run.py smoke save_videos=false`. Список этапов и бюджет доступны через `--help` и `plan`.

Повтор той же команды пропускает завершённые MLflow-эпизоды и повторяет оборванный из reset. Логи идут в терминал и `results/<run>/execution.log`; метрики обновляются в MLflow по вызовам модели. Изменение научных настроек/кода требует нового `output`.

Внешняя телеметрия MLflow отключена стандартной настройкой. Локальный интерфейс: `uv run python run.py ui` (127.0.0.1:5000). `report` создаёт CSV и сводку из завершённых MLflow runs. Видео записываются отдельно для каждого эпизода при включённой записи; демонстрационные примеры выбираете вы.

## Результаты для отчёта и notebook

По умолчанию выход - `results/mvp/`; цветной профиль использует `results/color/`. Таблицы обновляются после завершения этапа. Команда `uv run python run.py report` пересобирает их из завершённых MLflow-эпизодов без загрузки модели.

| Что сохраняется | Назначение и модуль |
|---|---|
| `episodes.csv`, `rates.csv`, `effects.csv`, `summary.json` | Исходы эпизодов, доли успеха, парные эффекты с CI и полнота test; `src/stats.py` |
| `config.yaml`, `manifest.json`, `runtime.json` | Разрешённый Hydra-конфиг, hashes кода/pins, Python/CUDA/GPU/драйвер; `src/tracking.py` копирует артефакты через MLflow API |
| `decisions/<family>/` | Допуск задач, результаты выбора места, зафиксированные настройки и steering-направление, если оно построено; `src/experiment.py` |
| `diagnostics/<family>/` | Debiased CKA, cosine, pooled NPZ, выборочные attention PNG/NPZ и маски; `src/experiment.py`, `src/policy.py`, `src/rollout.py` |
| `scenes/` | Начальные состояния и fingerprints для проверки парности; `src/rollout.py` |

Полные `episodes/.../attempt_*/` с `steps.jsonl`, `calls.jsonl`, исходами/ошибками, а также `execution.log` остаются на диске и исключены из git. MLflow хранит runs, метрики и копии артефактов в `mlflow/`; видео - в `videos/`.

## Docker

Образ использует тот же TOML/uv и содержит исходники, но не веса и токены. На хосте нужны Docker Compose с GPU и NVIDIA Container Toolkit.

```bash
docker compose build
docker compose run --rm experiment download
docker compose run --rm experiment smoke
```

Для следующих этапов замените `smoke` на `develop` или `test`. `HF_TOKEN` передаётся из окружения Compose. HF cache, checkpoint-каталог, MLflow, результаты и видео подключены с хоста. Для UI: `docker compose run --rm --service-ports experiment ui tracking.host=0.0.0.0`.

Открывайте MLflow UI в той среде, где созданы runs: локальные artifact paths записываются MLflow как абсолютные. Ссылки checkpoint-кэша тоже зависят от среды; не используйте один подготовленный checkpoint-каталог одновременно из host и Docker.