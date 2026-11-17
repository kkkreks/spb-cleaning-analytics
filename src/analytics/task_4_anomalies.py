"""
Задача 4. Обнаружение аномалий в данных о работе техники:
1) резкие скачки координат (GPS-глюки);
2) длительная неподвижность (>2 ч в радиусе 100 м);
3) нетипичное время активности (для каждой операции).
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import duckdb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ================= НАСТРОЙКИ =================
CLEAN = Path(r"C:\Users\Admin\Desktop\УЧЁБА\Магистратура\Большие данные\ДАННЫЕ\clean_all.parquet")
FIG = Path(r"C:\Users\Admin\Desktop\УЧЁБА\Магистратура\Большие данные\figures\task4_anomalies")
FIG.mkdir(parents=True, exist_ok=True)

JUMP_KMH = 200          # скорость > 200 км/ч = скачок координат
STATIC_H = 2.0          # > 2 ч неподвижности в радиусе
STATIC_R_KM = 0.1       # 100 м
TIME_Q_LO, TIME_Q_HI = 0.05, 0.95   # 5–95 перцентили по часам
MIN_OBS_OP = 500        # операции с <500 точек не анализируем по времени
# ============================================

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 200

con = duckdb.connect()

# ============ 1. СКАЧКИ КООРДИНАТ ============
print("Шаг 1. Скачки координат (speed > 200 км/ч) ...")

jumps = con.execute(f"""
    SELECT id, ut, dt, lat, lon, speed_kmh, dist_km, dt_gap_s, session_id
    FROM read_parquet('{CLEAN.as_posix()}')
    WHERE speed_kmh > {JUMP_KMH}
    ORDER BY speed_kmh DESC
""").df()
print(f"  найдено скачков: {len(jumps):,}")
print(f"  машин затронуто: {jumps['id'].nunique() if len(jumps) else 0}")

if len(jumps):
    jumps.to_csv(FIG / "anomalies_jumps.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.hist(np.clip(jumps["speed_kmh"], 0, 5000), bins=60, color="#C44E52", edgecolor="white")
    ax.set_title(f"Распределение «скачков» координат (speed > {JUMP_KMH} км/ч, n={len(jumps):,})")
    ax.set_xlabel("Скорость между точками, км/ч (clip 5000)")
    ax.set_ylabel("Число точек")
    plt.tight_layout(); plt.savefig(FIG / "fig4_jumps_hist.png"); plt.close()

    top_jump_machines = (jumps.groupby("id").size().sort_values(ascending=False).head(10))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(top_jump_machines.index.astype(str), top_jump_machines.values, color="#8172B3")
    ax.set_title("Топ-10 машин по числу «скачков» координат")
    ax.set_xlabel("ID машины"); ax.set_ylabel("Скачков")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout(); plt.savefig(FIG / "fig4_jumps_top_machines.png"); plt.close()
else:
    print("  скачков не найдено (данные уже отфильтрованы)")

# ============ 2. ДЛИТЕЛЬНАЯ НЕПОДВИЖНОСТЬ ============
print("\nШаг 2. Длительная неподвижность (>{:.0f} ч в радиусе {:.0f} м) ...".format(STATIC_H, STATIC_R_KM*1000))

# Для каждой сессии считаем: сколько времени машина провела в радиусе 100 м от медианной точки
# Простая аппроксимация: сессии, где 95% точек в радиусе 100 м и длительность > 2 ч.
sess_stats = con.execute(f"""
    SELECT
        id,
        CAST(session_id AS BIGINT) AS sid,
        MIN(dt) AS t_start,
        MAX(dt) AS t_end,
        COUNT(*) AS n_points,
        AVG(lat) AS lat_mean,
        AVG(lon) AS lon_mean,
        STDDEV(lat) AS lat_std,
        STDDEV(lon) AS lon_std
    FROM read_parquet('{CLEAN.as_posix()}')
    GROUP BY id, CAST(session_id AS BIGINT)
""").df()

sess_stats["duration_h"] = (sess_stats["t_end"] - sess_stats["t_start"]).dt.total_seconds() / 3600.0
# 0.01° широты ≈ 1.11 км; 0.01° долготы на широте СПб ≈ 0.55 км
sess_stats["r_km"] = np.sqrt((sess_stats["lat_std"] * 111.0)**2 +
                             (sess_stats["lon_std"] * 55.0)**2)

static = sess_stats[
    (sess_stats["duration_h"] >= STATIC_H) &
    (sess_stats["r_km"] <= STATIC_R_KM) &
    (sess_stats["n_points"] >= 30)
].copy()
print(f"  сессий с длительной неподвижностью: {len(static):,}")
print(f"  машин затронуто: {static['id'].nunique()}")

static.sort_values("duration_h", ascending=False).head(500).to_csv(
    FIG / "anomalies_static.csv", index=False)

if len(static):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.hist(np.clip(static["duration_h"], 0, 24), bins=60, color="#DD8452", edgecolor="white")
    ax.set_title(f"Сессии с длительной неподвижностью (>{STATIC_H:.0f} ч в радиусе {STATIC_R_KM*1000:.0f} м)")
    ax.set_xlabel("Длительность, ч"); ax.set_ylabel("Число сессий")
    plt.tight_layout(); plt.savefig(FIG / "fig4_static_hist.png"); plt.close()

    top_static = static.groupby("id").size().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(top_static.index.astype(str), top_static.values, color="#4C72B0")
    ax.set_title("Топ-10 машин по числу длительных «стоянок» (>2 ч)")
    ax.set_xlabel("ID машины"); ax.set_ylabel("Число сессий")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout(); plt.savefig(FIG / "fig4_static_top_machines.png"); plt.close()

# ============ 3. НЕТИПИЧНОЕ ВРЕМЯ АКТИВНОСТИ ============
print("\nШаг 3. Нетипичное время активности по операциям ...")

ops = con.execute(f"""
    SELECT op, COUNT(*) n
    FROM read_parquet('{CLEAN.as_posix()}')
    GROUP BY op
    HAVING COUNT(*) >= {MIN_OBS_OP}
    ORDER BY n DESC
""").df()
print(f"  операций для анализа: {len(ops)}")

if len(ops):
    top_ops = ops.head(8)["op"].tolist()
    placeholders = ",".join(["?"] * len(top_ops))
    df_ops = con.execute(f"""
        SELECT op, hour, COUNT(*) n
        FROM read_parquet('{CLEAN.as_posix()}')
        WHERE op IN ({placeholders})
        GROUP BY op, hour
        ORDER BY op, hour
    """, top_ops).df()

    # 2×4 панель: гистограмма по часам для каждой операции + 5–95 перцентили
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True)
    axes = axes.flatten()
    summary_rows = []
    for ax, op in zip(axes, top_ops):
        sub = df_ops[df_ops["op"] == op].sort_values("hour")
        hours = sub["hour"].values
        counts = sub["n"].values
        ax.bar(hours, counts, color="#55A868")
        ax.set_title(f"op = {op}  (n={counts.sum():,})", fontsize=10)
        ax.set_xlabel("Час"); ax.set_ylabel("Точек")
        # 5–95 перцентили часов (по точкам, а не по уникальным часам)
        weighted = np.repeat(hours, counts)
        q_lo = np.quantile(weighted, TIME_Q_LO)
        q_hi = np.quantile(weighted, TIME_Q_HI)
        ax.axvline(q_lo, color="red", linestyle="--", linewidth=1)
        ax.axvline(q_hi, color="red", linestyle="--", linewidth=1)
        summary_rows.append({"op": op, "q05_hour": q_lo, "q95_hour": q_hi,
                             "n": int(counts.sum())})
    plt.suptitle(f"Активность по часам для топ-операций (красные — границы 5–95%)", y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG / "fig4_op_hours.png"); plt.close()

    pd.DataFrame(summary_rows).to_csv(FIG / "anomalies_op_time_window.csv", index=False)

    # Нетипичные точки: часы вне [q05, q95] для своей операции
    print("  ищу точки вне 5–95% по часам ...")
    windows = {r["op"]: (r["q05_hour"], r["q95_hour"]) for r in summary_rows}
    df_all = con.execute(f"""
        SELECT op, hour, COUNT(*) n
        FROM read_parquet('{CLEAN.as_posix()}')
        WHERE op IN ({placeholders})
        GROUP BY op, hour
        ORDER BY op, hour
    """, top_ops).df()

    atypical = []
    for _, row in df_all.iterrows():
        op = row["op"]; h = row["hour"]; n = row["n"]
        lo, hi = windows[op]
        if h < lo or h > hi:
            atypical.append({"op": op, "hour": h, "n": n, "outside": "before 5%" if h < lo else "after 95%"})
    atypical = pd.DataFrame(atypical).sort_values("n", ascending=False)
    atypical.to_csv(FIG / "anomalies_atypical_hours.csv", index=False)
    print(f"  нетипичных (op, hour) пар: {len(atypical)}")
    if len(atypical):
        print(atypical.head(10).to_string(index=False))

# ============ ИТОГ ============
print("\n=== ИТОГ ЗАДАЧИ 4 ===")
if len(jumps):
    print(f"  скачков координат:         {len(jumps):,} (машин: {jumps['id'].nunique()})")
print(f"  длительных «стоянок»:     {len(static):,} (машин: {static['id'].nunique() if len(static) else 0})")
print(f"  графиков и таблиц:         {FIG}")

print("\nГотово.")