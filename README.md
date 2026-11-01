# Intelligent Urban Mobility Analytics (NYC TLC + NOAA)

End-to-end аналитика городской мобильности Нью-Йорка: зональный прогноз спроса на такси
с учётом погодных условий, ленивый ETL на Polars, хороплет-визуализация и REST API.

## Бизнес-задача
- Прогнозировать почасовой спрос на такси по 263 зонам NYC с горизонтом 24 часа.
- Оценить влияние погоды (температура, осадки, ветер) на спрос по зонам.
- Выявить зоны с аномальным спросом (события, погодные шоки).
- Дать API для интеграции с диспетчерскими системами такси.

## Данные
- **NYC TLC Yellow Taxi Trip Records** (Parquet, помесячно с 2019 г.) — pickup/dropoff
  datetime, PULocationID/DOLocationID, fare_amount, trip_distance, passenger_count [citation:3].
- **NYC Taxi Zones** (GeoJSON/shapefile) — 263 зоны с геометрией для хороплетов [citation:20].
- **NOAA Climate Data Online (GHCND)** — дневные сводки по станции Central Park
  (USW00094728): TMAX, TMIN, PRCP, SNOW, AWND [citation:4][citation:17].

## Стек
Python 3.11, Polars (lazy scan_parquet, predicate/projection pushdown) [citation:18],
GeoPandas, LightGBM, FastAPI, Pydantic v2, Docker.

## Быстрый старт
```bash
make setup
make download   # скачивает 1 год TLC + NOAA
make features
make train
make serve
curl "http://localhost:8000/demand?zone=161&horizon=24"
Метрики
WAPE, sMAPE, RMSE по зонам

Временная кросс-валидация (rolling origin, 7 дней)

Сравнение с baseline (историческое среднее по зоне/часу/дню недели)


---

## `pyproject.toml`

```toml
[project]
name = "urban-mobility-analytics"
version = "0.1.0"
description = "NYC TLC + NOAA urban mobility demand forecasting"
requires-python = ">=3.11"
dependencies = [
    "polars>=0.20.0",
    "duckdb>=0.10.0",
    "numpy>=1.26",
    "pandas>=2.2",
    "scikit-learn>=1.4",
    "lightgbm>=4.3",
    "geopandas>=0.14",
    "matplotlib>=3.8",
    "seaborn>=0.13",
    "fastapi>=0.110",
    "uvicorn[standard]>=0.29",
    "pydantic>=2.6",
    "pydantic-settings>=2.2",
    "pyyaml>=6.0",
    "python-dotenv>=1.0",
    "httpx>=0.27",
    "requests>=2.31",
    "joblib>=1.3",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.3",
    "mypy>=1.9",
]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["src*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[tool.ruff]
line-length = 100
target-version = "py311"
Как запустить (полная последовательность )


# 1. Установка
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Скачать данные
#    - TLC yellow taxi parquet (по месяцам, ~50-100 MB каждый)
#    - Taxi zones GeoJSON (~10 MB)
#    - NOAA GHCND daily (требуется бесплатный токен: https://www.ncdc.noaa.gov/cdo-web/token)
cp .env.example .env
# вписать NOAA_TOKEN=...
python scripts/download_data.py

# 3. Построить панель (lazy scan + aggregate + weather join)
python scripts/build_features.py

# 4. Обучение (rolling-origin CV + LightGBM + baseline)
python scripts/train.py

# 5. Тесты
pytest -q

# 6. Поднять API
./scripts/serve.sh
# → http://localhost:8000/docs

# 7. Проверка
curl "http://localhost:8000/health"
curl "http://localhost:8000/demand?zone=161&horizon=24"
curl "http://localhost:8000/zones/top?n=10"
curl "http://localhost:8000/weather-sensitivity"
curl "http://localhost:8000/model-info"

# 8. Docker
docker compose up --build
