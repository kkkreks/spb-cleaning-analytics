"""
Задача 2. Восстановление рабочих сессий и приближённых траекторий:
- длительности сессий
- пройденные расстояния
- доля движения/стоянки
- топ-10 машин по числу сессий
- число сессий по дням
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import duckdb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ================= НАСТРОЙКИ =================
CLEAN = Path(r"C:\Users\Admin\Desktop\УЧЁБА\Магистратура\Большие данные\ДАННЫЕ\clean_all.parquet")
FIG = Path(r"C:\Users\Admin\Desktop\УЧЁБА\Магистратура\Большие данные\figures\task2_sessions")
FIG.mkdir(parents=True, exist_ok=True)
MIN_SESS_H = 0.05   # отбрасываем сессии короче 3 минут
# ============================================

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 200

con = duckdb.connect()

print("Считаю сессии через DuckDB ...")
sess = con.execute(f"""
    SELECT
        id,
        CAST(session_id AS BIGINT) AS sid,
        MIN(dt) AS t_start,
        MAX(dt) AS t_end,
        COUNT(*) AS n_points,
        SUM(dist_km) AS dist_km,
        SUM(CASE WHEN speed_kmh IS NULL OR speed_kmh < 1 THEN 1 ELSE 0 END) AS n_stop,
        SUM(CASE WHEN speed_kmh >= 1 THEN 1 ELSE 0 END) AS n_move
    FROM read_parquet('{CLEAN.as_posix()}')
    GROUP BY id, CAST(session_id AS BIGINT)
""").df()

sess["duration_h"] = (sess["t_end"] - sess["t_start"]).dt.total_seconds() / 3600
sess = sess[sess["duration_h"] >= MIN_SESS_H].copy()
print(f"  сессий после фильтра ≥ {MIN_SESS_H} ч: {len(sess):,}")

print(f"\nМетрики сессий:")
print(f"  машин: {sess['id'].nunique():,}")
print(f"  средняя длительность: {sess['duration_h'].mean():.2f} ч")
print(f"  медианная:           {sess['duration_h'].median():.2f} ч")
print(f"  p95 длительности:    {sess['duration_h'].quantile(0.95):.2f} ч")
print(f"  средняя дистанция:   {sess['dist_km'].mean():.1f} км")
print(f"  медианная:           {sess['dist_km'].median():.1f} км")
print(f"  p95 дистанции:       {sess['dist_km'].quantile(0.95):.1f} км")
print(f"  сессий с дистанцией > 100 км: {(sess['dist_km'] > 100).sum():,}")
total_stop = int(sess["n_stop"].sum())
total_move = int(sess["n_move"].sum())
print(f"  точек в движении: {total_move:,} ({100*total_move/(total_move+total_stop):.1f}%)")
print(f"  точек в стоянке:  {total_stop:,} ({100*total_stop/(total_move+total_stop):.1f}%)")

# ---------- 2.1. Длительности ----------
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.hist(sess["duration_h"].clip(upper=12), bins=60, color="#4C72B0", edgecolor="white")
ax.set_title(f"Распределение длительности сессий (n={len(sess):,})")
ax.set_xlabel("Длительность, ч"); ax.set_ylabel("Число сессий")
ax.axvline(sess["duration_h"].median(), color="red", linestyle="--",
           label=f"медиана {sess['duration_h'].median():.2f} ч")
ax.legend()
plt.tight_layout(); plt.savefig(FIG / "fig2_durations.png"); plt.close()
print("\nfig2_durations.png сохранено")

# ---------- 2.2. Дистанции ----------
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.hist(sess["dist_km"].clip(upper=300), bins=60, color="#DD8452", edgecolor="white")
ax.set_title(f"Распределение пройденного расстояния за сессию (n={len(sess):,})")
ax.set_xlabel("Расстояние, км"); ax.set_ylabel("Число сессий")
ax.axvline(sess["dist_km"].median(), color="red", linestyle="--",
           label=f"медиана {sess['dist_km'].median():.1f} км")
ax.legend()
plt.tight_layout(); plt.savefig(FIG / "fig2_distances.png"); plt.close()
print("fig2_distances.png сохранено")

# ---------- 2.3. Движение/стоянка ----------
fig, ax = plt.subplots(figsize=(6, 6))
ax.pie([total_move, total_stop],
       labels=[f"в движении\n{total_move/1e6:.1f}M", f"стоянка\n{total_stop/1e6:.1f}M"],
       colors=["#55A868", "#C44E52"], autopct="%1.1f%%", startangle=90)
ax.set_title("Точки в движении vs на стоянке")
plt.tight_layout(); plt.savefig(FIG / "fig2_moving_pie.png"); plt.close()
print("fig2_moving_pie.png сохранено")

# ---------- 2.4. Топ-10 машин ----------
top_m = sess.groupby("id").size().sort_values(ascending=False).head(10)
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.bar(top_m.index.astype(str), top_m.values, color="#8172B3")
ax.set_title("Топ-10 машин по числу сессий")
ax.set_xlabel("ID машины"); ax.set_ylabel("Число сессий")
plt.xticks(rotation=45, ha="right")
plt.tight_layout(); plt.savefig(FIG / "fig2_top_machines.png"); plt.close()
print("fig2_top_machines.png сохранено")

# ---------- 2.5. Сессии по дням ----------
sess["date"] = sess["t_start"].dt.date
by_day = sess.groupby("date").size()
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.bar(by_day.index.astype(str), by_day.values, color="#4C72B0")
ax.set_title("Число сессий по дням")
ax.set_xlabel("Дата"); ax.set_ylabel("Число сессий")
plt.xticks(rotation=45, ha="right")
plt.tight_layout(); plt.savefig(FIG / "fig2_daily_sessions.png"); plt.close()
print("fig2_daily_sessions.png сохранено")

print(f"\nГотово. Файлы в {FIG}")