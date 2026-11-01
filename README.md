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
