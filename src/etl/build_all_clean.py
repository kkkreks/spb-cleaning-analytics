import duckdb
import requests
import pandas as pd
from pathlib import Path

# ================= НАСТРОЙКИ =================
DATA_DIR = Path(r"C:\Users\Admin\Desktop\УЧЁБА\Магистратура\Большие данные\ДАННЫЕ")
OUT_PARQUET = DATA_DIR / "clean_all.parquet"
OUT_WEATHER = DATA_DIR / "weather_hourly.parquet"
DB_FILE = DATA_DIR / "clean.duckdb"
# ============================================

DATA_DIR.mkdir(parents=True, exist_ok=True)

con = duckdb.connect(str(DB_FILE))
con.execute("PRAGMA threads=4")
con.execute("PRAGMA memory_limit='4GB'")

print("Шаг 1/5: читаю *.jsonl ...")
con.execute(f"""
CREATE OR REPLACE TABLE raw_jsonl AS
SELECT
    CAST(json_extract_string(j, '$.id')   AS BIGINT)  AS id,
    CAST(json_extract_string(j, '$.ut')   AS BIGINT)  AS ut,
    CAST(json_extract(j, '$.loc[0]')      AS DOUBLE)  AS lat,
    CAST(json_extract(j, '$.loc[1]')      AS DOUBLE)  AS lon,
    CAST(json_extract_string(j, '$.info[0]') AS VARCHAR) AS op,
    CAST(json_extract_string(j, '$.info[1]') AS VARCHAR) AS info1,
    CAST(json_extract_string(j, '$.info[2]') AS VARCHAR) AS info2,
    CAST(json_extract_string(j, '$.info[3]') AS VARCHAR) AS info3,
    CAST(json_extract_string(j, '$.info[4]') AS VARCHAR) AS info4
FROM (
    SELECT json(line) AS j
    FROM read_csv('{DATA_DIR}/**/*.jsonl',
                  header=false, columns={{'line': 'VARCHAR'}},
                  ignore_errors=true, delim='\\x01')
)
WHERE j IS NOT NULL
""")
n1 = con.execute("SELECT COUNT(*) FROM raw_jsonl").fetchone()[0]
print(f"  строк: {n1}")

print("Шаг 2/5: читаю *.csv.gz.part (там строка вида \"{{\"\"v\"\":...}}\") ...")
con.execute(f"""
CREATE OR REPLACE TABLE raw_csv AS
SELECT
    CAST(json_extract_string(j, '$.id')   AS BIGINT)  AS id,
    CAST(json_extract_string(j, '$.ut')   AS BIGINT)  AS ut,
    CAST(json_extract(j, '$.loc[0]')      AS DOUBLE)  AS lat,
    CAST(json_extract(j, '$.loc[1]')      AS DOUBLE)  AS lon,
    CAST(json_extract_string(j, '$.info[0]') AS VARCHAR) AS op,
    CAST(json_extract_string(j, '$.info[1]') AS VARCHAR) AS info1,
    CAST(json_extract_string(j, '$.info[2]') AS VARCHAR) AS info2,
    CAST(json_extract_string(j, '$.info[3]') AS VARCHAR) AS info3,
    CAST(json_extract_string(j, '$.info[4]') AS VARCHAR) AS info4
FROM (
    SELECT replace(replace(line, '""', '"'), '\\r', '') AS j
    FROM read_csv('{DATA_DIR}/**/*.csv.gz.part',
                  header=false, columns={{'line': 'VARCHAR'}},
                  ignore_errors=true, delim='\\x01')
)
WHERE j IS NOT NULL AND j LIKE '{{%'
""")
n2 = con.execute("SELECT COUNT(*) FROM raw_csv").fetchone()[0]
print(f"  строк: {n2}")

print("Шаг 3/5: объединяю в raw ...")
con.execute("""
CREATE OR REPLACE TABLE raw AS
SELECT * FROM raw_jsonl
UNION ALL
SELECT * FROM raw_csv
""")
n = con.execute("SELECT COUNT(*) FROM raw").fetchone()[0]
print(f"  всего строк: {n}")

print("Шаг 4/5: enrich (dt, hour, dow, dist_km, speed, session_id) ...")
con.execute(f"""
COPY (
WITH t AS (
    SELECT *,
           to_timestamp(ut) AT TIME ZONE 'Europe/Moscow' AS dt
    FROM raw
    WHERE lat IS NOT NULL AND lon IS NOT NULL AND ut IS NOT NULL AND id IS NOT NULL
),
t2 AS (
    SELECT *,
           LAG(lat) OVER w AS lat_prev,
           LAG(lon) OVER w AS lon_prev,
           LAG(ut)  OVER w AS ut_prev
    FROM t
    WINDOW w AS (PARTITION BY id ORDER BY ut)
),
t3 AS (
    SELECT *,
           CASE WHEN ut_prev IS NOT NULL THEN
               2 * 6371.0 * asin(sqrt(
                   pow(sin(radians(lat - lat_prev)/2), 2) +
                   cos(radians(lat_prev)) * cos(radians(lat)) *
                   pow(sin(radians(lon - lon_prev)/2), 2)
               ))
           END AS dist_km,
           ut - ut_prev AS dt_gap_s
    FROM t2
),
t4 AS (
    SELECT *,
           CASE WHEN dt_gap_s > 0 AND dist_km IS NOT NULL
                THEN dist_km / (dt_gap_s / 3600.0)
           END AS speed_kmh,
           CASE WHEN ut_prev IS NULL OR (ut - ut_prev) > 900 THEN 1 ELSE 0 END
             AS new_session
    FROM t3
)
SELECT
    id, ut, lat, lon, op, info1, info2, info3, info4, dt,
    CAST(dt AS DATE) AS date,
    EXTRACT(hour FROM dt)::TINYINT AS hour,
    (EXTRACT(isodow FROM dt) - 1)::TINYINT AS dow,
    dist_km, dt_gap_s, speed_kmh,
    SUM(new_session) OVER (PARTITION BY id ORDER BY ut) AS session_id
FROM t4
) TO '{OUT_PARQUET}' (FORMAT PARQUET, COMPRESSION SNAPPY);
""")
print(f"  сохранено: {OUT_PARQUET}")

print("Шаг 5/5: финальная сводка ...")
s = con.execute(f"""
SELECT COUNT(*) rows, COUNT(DISTINCT id) machines,
       MIN(dt) min_dt, MAX(dt) max_dt,
       COUNT(DISTINCT date) days,
       COUNT(DISTINCT session_id) sessions
FROM read_parquet('{OUT_PARQUET}')
""").fetchone()
print(f"  rows={s[0]:,}  machines={s[1]:,}  days={s[4]}  sessions={s[5]:,}")
print(f"  период: {s[2]} → {s[3]}")

# --- погода ---
try:
    start = s[2].date().isoformat()
    end = s[3].date().isoformat()
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 59.94, "longitude": 30.31,
        "start_date": start, "end_date": end,
        "hourly": "temperature_2m,precipitation,snowfall,wind_speed_10m",
        "timezone": "Europe/Moscow",
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    h = r.json()["hourly"]
    w = pd.DataFrame({
        "dt_hour": pd.to_datetime(h["time"]).tz_localize("Europe/Moscow"),
        "temp_c": h["temperature_2m"],
        "precip_mm": h["precipitation"],
        "snow_cm": h["snowfall"],
        "wind_kmh": h["wind_speed_10m"],
    })
    w.to_parquet(OUT_WEATHER, index=False)
    print(f"  погода: {OUT_WEATHER} ({len(w)} часов)")
except Exception as e:
    print(f"  погода не скачалась: {e}")

con.close()
print("\nГотово.")