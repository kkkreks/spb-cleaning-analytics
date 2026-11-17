import duckdb
from pathlib import Path

CLEAN = Path(r"C:\Users\Admin\Desktop\УЧЁБА\Магистратура\Большие данные\ДАННЫЕ\clean_all.parquet")
TMP = CLEAN.with_name("clean_all.fixed.parquet")

con = duckdb.connect()
con.execute(f"""
    COPY (
        SELECT
            id, ut, lat, lon, op, info1, info2, info3, info4,
            dt, date, hour, dow,
            dist_km, dt_gap_s, speed_kmh,
            CAST(session_id AS BIGINT) AS session_id
        FROM read_parquet('{CLEAN.as_posix()}')
    ) TO '{TMP.as_posix()}' (FORMAT PARQUET, COMPRESSION SNAPPY)
""")
con.close()

TMP.replace(CLEAN)
print("session_id переприведён к BIGINT, файл перезаписан")