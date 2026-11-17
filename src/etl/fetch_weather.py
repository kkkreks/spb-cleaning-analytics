import pandas as pd
import requests
from pathlib import Path

OUT = Path(r"C:\Users\Admin\Desktop\УЧЁБА\Магистратура\Большие данные\ДАННЫЕ\weather_hourly.parquet")

r = requests.get(
    "https://archive-api.open-meteo.com/v1/archive",
    params={
        "latitude": 59.94, "longitude": 30.31,
        "start_date": "2026-10-17", "end_date": "2026-10-31",
        "hourly": "temperature_2m,precipitation,snowfall,wind_speed_10m",
        "timezone": "Europe/Moscow",
    },
    timeout=60,
).json()["hourly"]

w = pd.DataFrame({
    "dt_hour": pd.to_datetime(r["time"]).tz_localize("Europe/Moscow"),
    "temp_c": r["temperature_2m"],
    "precip_mm": r["precipitation"],
    "snow_cm": r["snowfall"],
    "wind_kmh": r["wind_speed_10m"],
})
w.to_parquet(OUT, index=False)
print(f"погода сохранена: {OUT} ({len(w)} часов)")