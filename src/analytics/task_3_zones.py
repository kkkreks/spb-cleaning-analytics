"""
Задача 3. Выявление участков с редкими наблюдениями уборочной техники
и длительными интервалами между повторными появлениями.

Логика:
- Для каждой H3-ячейки (res=7, ~1.2 км) считаем:
  * число визитов — сколько раз какая-либо машина побывала в ячейке (уникальные (id, session_id));
  * число уникальных машин;
  * медианный и p95 интервал между последовательными визитами в эту ячейку.
- «Редкие» = ячейки с числом визитов в нижних 10%.
- «Длинные интервалы» = ячейки, где медианный интервал между визитами > 6 ч.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import duckdb
import h3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

# ================= НАСТРОЙКИ =================
CLEAN = Path(r"C:\Users\Admin\Desktop\УЧЁБА\Магистратура\Большие данные\ДАННЫЕ\clean_all.parquet")
FIG = Path(r"C:\Users\Admin\Desktop\УЧЁБА\Магистратура\Большие данные\figures\task3_zones")
FIG.mkdir(parents=True, exist_ok=True)

H3_RES = 7                # ~1.2 км
MIN_POINTS_CELL = 100     # ячейки с <100 точек не рассматриваем
MIN_MACHINES_CELL = 3     # ячейки, где побывало <3 машин — не «рабочие»
RARE_QUANTILE = 0.10      # нижние 10% по визитам — «редкие»
LONG_INTERVAL_H = 6       # медианный интервал > 6 ч — «длинный»
# ============================================

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 200

con = duckdb.connect()

# ---------- 1. Считаем визиты и интервалы ----------
print("Шаг 1. Агрегирую визиты в H3-ячейки ...")

# Берём только движение (без стоянок/баз), по сессиям.
# Для каждой точки определяем h3, для каждой сессии — список ячеек,
# в которых она побывала; для каждой ячейки — timestamp визита = MIN(dt) за сессию.
df = con.execute(f"""
    SELECT
        id,
        CAST(session_id AS BIGINT) AS sid,
        dt,
        lat,
        lon
    FROM read_parquet('{CLEAN.as_posix()}')
    WHERE speed_kmh IS NULL OR speed_kmh >= 1
""").df()
print(f"  точек (в движении): {len(df):,}")

df["h3"] = [h3.latlng_to_cell(lat, lon, H3_RES)
            for lat, lon in zip(df["lat"].values, df["lon"].values)]
print(f"  H3-ячеек: {df['h3'].nunique():,}")

# Визит = (сессия, ячейка) с временем первого появления
visits = (df.groupby(["sid", "h3"], as_index=False)
            .agg(id=("id", "first"),
                 t_visit=("dt", "min"),
                 n_points=("dt", "size")))
print(f"  визитов: {len(visits):,}")

# ---------- 2. Агрегируем по ячейкам ----------
print("\nШаг 2. Агрегирую по ячейкам ...")

# Сортируем визиты внутри ячейки по времени — для расчёта интервалов
visits = visits.sort_values(["h3", "t_visit"]).reset_index(drop=True)
visits["prev_t"] = visits.groupby("h3")["t_visit"].shift()
visits["interval_h"] = (visits["t_visit"] - visits["prev_t"]).dt.total_seconds() / 3600.0

agg = (visits.groupby("h3", as_index=False)
              .agg(n_visits=("sid", "nunique"),
                   n_machines=("id", "nunique"),
                   n_points=("n_points", "sum"),
                   med_interval_h=("interval_h", lambda s: s.dropna().median()),
                   p95_interval_h=("interval_h", lambda s: s.dropna().quantile(0.95)),
                   mean_interval_h=("interval_h", "mean")))

# Координаты ячейки (центр)
agg["lat_c"], agg["lon_c"] = zip(*[h3.cell_to_latlng(h) for h in agg["h3"]])

# Фильтр: только «рабочие» ячейки, где реально бывает техника
agg = agg[(agg["n_points"] >= MIN_POINTS_CELL) &
          (agg["n_machines"] >= MIN_MACHINES_CELL)].copy()
print(f"  ячеек после фильтра (n_points>={MIN_POINTS_CELL}, n_machines>={MIN_MACHINES_CELL}): {len(agg):,}")

# ---------- 3. Классификация ----------
print("\nШаг 3. Классификация «редких» и «длинных интервалов» ...")

thr_visits = agg["n_visits"].quantile(RARE_QUANTILE)
agg["is_rare"] = agg["n_visits"] <= thr_visits
agg["is_long_interval"] = agg["med_interval_h"] > LONG_INTERVAL_H

print(f"  порог «редких» (10-й процентиль визитов): {thr_visits:.1f} визитов")
print(f"  «редких» ячеек: {agg['is_rare'].sum():,} ({100*agg['is_rare'].mean():.1f}%)")
print(f"  «длинный интервал» (>{LONG_INTERVAL_H} ч): {agg['is_long_interval'].sum():,} ({100*agg['is_long_interval'].mean():.1f}%)")

# ---------- 4. Сохраняем таблицы ----------
agg.to_csv(FIG / "h3_zones_full.csv", index=False)
print(f"  saved: {FIG / 'h3_zones_full.csv'}")

# Топ-20 самых редких (меньше всего визитов)
top_rare = (agg[agg["is_rare"]]
            .sort_values(["n_visits", "n_machines"])
            .head(20)[["h3", "lat_c", "lon_c", "n_visits", "n_machines",
                       "med_interval_h", "n_points"]])
top_rare.to_csv(FIG / "top20_rare.csv", index=False)
print(f"  saved: {FIG / 'top20_rare.csv'}")

# Топ-20 длинных интервалов
top_long = (agg[agg["is_long_interval"]]
            .sort_values("med_interval_h", ascending=False)
            .head(20)[["h3", "lat_c", "lon_c", "n_visits", "n_machines",
                       "med_interval_h", "p95_interval_h"]])
top_long.to_csv(FIG / "top20_long_interval.csv", index=False)
print(f"  saved: {FIG / 'top20_long_interval.csv'}")

print("\nТоп-10 «редких» ячеек:")
print(top_rare.head(10).to_string(index=False))

print("\nТоп-10 «длинных интервалов»:")
print(top_long.head(10).to_string(index=False))

# ---------- 5. Функция рисования карты ----------
def draw_cells(ax, df, value_col, cmap, norm=None, edge="none"):
    if norm is None:
        vmin = max(df[value_col].min(), 1)
        vmax = df[value_col].max()
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax) if vmin > 0 else None
    cmap_obj = plt.get_cmap(cmap)
    for _, row in df.iterrows():
        boundary = h3.cell_to_boundary(row["h3"])
        poly = [(lon, lat) for lat, lon in boundary]
        color = cmap_obj(norm(row[value_col])) if norm else cmap_obj(0.5)
        ax.fill(*zip(*poly), color=color, edgecolor=edge)
    return norm


# ---------- 6. Карта «число визитов» ----------
print("\nКарта 1. Число визитов по H3 ...")
fig, ax = plt.subplots(figsize=(11, 10))
norm = draw_cells(ax, agg, "n_visits", "YlOrRd")
ax.set_xlim(29.40, 30.80); ax.set_ylim(59.70, 60.30)
ax.set_aspect("equal"); ax.set_xlabel("Долгота"); ax.set_ylabel("Широта")
ax.set_title("Число визитов уборочной техники по H3-ячейкам (~1.2 км)")
sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=norm); sm.set_array([])
plt.colorbar(sm, ax=ax, shrink=0.7, label="визитов")
plt.tight_layout(); plt.savefig(FIG / "fig3_map_visits.png"); plt.close()
print("  fig3_map_visits.png сохранено")

# ---------- 7. Карта «редких» ----------
print("Карта 2. Редкие зоны (нижние 10% по визитам) ...")
fig, ax = plt.subplots(figsize=(11, 10))
# сначала все ячейки бледно-серым
for _, row in agg.iterrows():
    boundary = h3.cell_to_boundary(row["h3"])
    poly = [(lon, lat) for lat, lon in boundary]
    ax.fill(*zip(*poly), color="#EEEEEE", edgecolor="none")
# потом редкие — красным
rare = agg[agg["is_rare"]]
for _, row in rare.iterrows():
    boundary = h3.cell_to_boundary(row["h3"])
    poly = [(lon, lat) for lat, lon in boundary]
    ax.fill(*zip(*poly), color="#C44E52", edgecolor="none")
ax.set_xlim(29.40, 30.80); ax.set_ylim(59.70, 60.30)
ax.set_aspect("equal"); ax.set_xlabel("Долгота"); ax.set_ylabel("Широта")
ax.set_title(f"«Редкие» зоны: нижние {int(RARE_QUANTILE*100)}% по числу визитов (красные)")
plt.tight_layout(); plt.savefig(FIG / "fig3_map_rare.png"); plt.close()
print("  fig3_map_rare.png сохранено")

# ---------- 8. Карта «длинных интервалов» ----------
print("Карта 3. Длинные интервалы между визитами (медиана > 6 ч) ...")
fig, ax = plt.subplots(figsize=(11, 10))
for _, row in agg.iterrows():
    boundary = h3.cell_to_boundary(row["h3"])
    poly = [(lon, lat) for lat, lon in boundary]
    if row["is_long_interval"]:
        # цвет по медианному интервалу
        v = min(row["med_interval_h"], 72)
        color = plt.get_cmap("Reds")(0.3 + 0.7 * (v / 72))
    else:
        color = "#EEEEEE"
    ax.fill(*zip(*poly), color=color, edgecolor="none")
ax.set_xlim(29.40, 30.80); ax.set_ylim(59.70, 60.30)
ax.set_aspect("equal"); ax.set_xlabel("Долгота"); ax.set_ylabel("Широта")
ax.set_title(f"Зоны с длинными интервалами между визитами (медиана > {LONG_INTERVAL_H} ч)")
plt.tight_layout(); plt.savefig(FIG / "fig3_map_long_interval.png"); plt.close()
print("  fig3_map_long_interval.png сохранено")

# ---------- 9. Гистограмма интервалов ----------
print("Гистограмма интервалов между визитами ...")
ints = visits["interval_h"].dropna()
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.hist(ints.clip(upper=72), bins=72, color="#4C72B0", edgecolor="white")
ax.set_title(f"Распределение интервалов между визитами в одну H3-ячейку (n={len(ints):,})")
ax.set_xlabel("Интервал, ч"); ax.set_ylabel("Число пар визитов")
ax.axvline(ints.median(), color="red", linestyle="--", label=f"медиана {ints.median():.1f} ч")
ax.axvline(LONG_INTERVAL_H, color="orange", linestyle="--", label=f"порог «длинного» {LONG_INTERVAL_H} ч")
ax.legend()
plt.tight_layout(); plt.savefig(FIG / "fig3_hist_intervals.png"); plt.close()
print("  fig3_hist_intervals.png сохранено")

# ---------- 10. Топ-20 редких / длинных — barh ----------
print("Барчарты топ-20 ...")
fig, ax = plt.subplots(figsize=(10, 6))
tt = top_rare.sort_values("n_visits")
ax.barh(tt["h3"].str[:7] + "…" + tt["h3"].str[-4:], tt["n_visits"], color="#C44E52")
ax.set_title("Топ-20 «редких» H3-ячеек (меньше всего визитов)")
ax.set_xlabel("Число визитов")
plt.tight_layout(); plt.savefig(FIG / "fig3_top20_rare.png"); plt.close()
print("  fig3_top20_rare.png сохранено")

fig, ax = plt.subplots(figsize=(10, 6))
tt = top_long.sort_values("med_interval_h")
ax.barh(tt["h3"].str[:7] + "…" + tt["h3"].str[-4:], tt["med_interval_h"], color="#DD8452")
ax.set_title("Топ-20 зон с наибольшим медианным интервалом между визитами")
ax.set_xlabel("Медианный интервал, ч")
plt.tight_layout(); plt.savefig(FIG / "fig3_top20_long_interval.png"); plt.close()
print("  fig3_top20_long_interval.png сохранено")

print(f"\nГотово. Все файлы: {FIG}")
print(f"CSV: h3_zones_full.csv, top20_rare.csv, top20_long_interval.csv")
print(f"PNG: 6 карт и гистограмм")