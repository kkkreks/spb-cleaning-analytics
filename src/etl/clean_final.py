import duckdb
from pathlib import Path

CLEAN = Path(r"C:\Users\Admin\Desktop\УЧЁБА\Магистратура\Большие данные\ДАННЫЕ\clean_all.parquet")
TMP = CLEAN.with_name("clean_all.filtered.parquet")

con = duckdb.connect()
con.execute("PRAGMA threads=4")

print("Читаю и фильтрую ...")
con.execute(f"""
    COPY (
        WITH filtered AS (
            SELECT
                id, ut, lat, lon, op, info1, info2, info3, info4,
                dt, date, hour, dow,
                dist_km, dt_gap_s, speed_kmh,
                CAST(session_id AS BIGINT) AS session_id
            FROM read_parquet('{CLEAN.as_posix()}')
            WHERE lat BETWEEN 59.70 AND 60.30
              AND lon BETWEEN 29.40 AND 30.80
              AND (speed_kmh IS NULL OR speed_kmh < 120)
        ),
        dedup AS (
            SELECT *,
                   ROW_NUMBER() OVER (PARTITION BY id, ut ORDER BY dt) AS rn
            FROM filtered
        )
        SELECT * EXCLUDE rn
        FROM dedup
        WHERE rn = 1
    ) TO '{TMP.as_posix()}' (FORMAT PARQUET, COMPRESSION SNAPPY)
""")
con.close()

print(f"Готово: {TMP}")
print(f"Было: {CLEAN.stat().st_size/1024/1024:.0f} MB")
print(f"Стало: {TMP.stat().st_size/1024/1024:.0f} MB")

TMP.replace(CLEAN)
print("clean_all.parquet заменён отфильтрованной версией")