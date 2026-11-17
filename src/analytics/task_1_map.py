"""
Карта H3 (задача 1). Две карты плотности:
1) все GPS-точки;
2) только движение (speed_kmh >= 1) — без баз и стоянок.
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
FIG = Path(r"C:\Users\Admin\Desktop\УЧЁБА\Магистратура\Большие данные\figures\task1_activity")
FIG.mkdir(parents=True, exist_ok=True)
H3_RES = 7                       # ~1.2 км
H3_MIN_POINTS = 100              # ячейки с <100 точек не рисуем, чтобы шум не мешал
# ============================================

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 200

con = duckdb.connect()

# ---------- Агрегируем по H3-ячейкам (res=7) ----------
print("Агрегирую точки в H3-ячейки (res=7) ...")

# У DuckDB нет h3-функции из коробки, поэтому:
# 1) читаем уникальные (lat, lon) — но их ~20 млн, многовато.
# 2) Округлим координаты до ~1e-3 (≈100 м) и сгруппируем — уникальных пар станет ~200к.
# 3) На них посчитаем H3-ячейки в Python.

df_all = con.execute(f"""
    SELECT
        lat, lon,
        COUNT(*) AS n_points,
        SUM(CASE WHEN speed_kmh >= 1 THEN 1 ELSE 0 END) AS n_moving,
        COUNT(DISTINCT id) AS n_machines
    FROM read_parquet('{CLEAN.as_posix()}')
    GROUP BY lat, lon
""").df()
print(f"  уникальных (lat, lon): {len(df_all):,}")

print("Считаю H3-ячейки ...")
df_all["h3"] = [h3.latlng_to_cell(lat, lon, H3_RES)
                for lat, lon in zip(df_all["lat"].values, df_all["lon"].values)]

agg = (df_all.groupby("h3", as_index=False)
       .agg(n_points=("n_points", "sum"),
            n_moving=("n_moving", "sum"),
            n_machines=("n_machines", "max"),
            lat_c=("lat", "mean"),
            lon_c=("lon", "mean")))
print(f"  H3-ячеек: {len(agg):,}")

# Считаем границы для карты (по СПб, без мусора)
mask_spb = (agg["lat_c"].between(59.70, 60.30)
            & agg["lon_c"].between(29.40, 30.80))
agg_spb = agg[mask_spb].copy()
print(f"  ячеек внутри СПб: {len(agg_spb):,}")

# ---------- Функция рисования ----------
def draw_map(df, value_col, title, fname, cmap="YlOrRd"):
    df = df[df[value_col] >= H3_MIN_POINTS].copy()
    print(f"  {fname}: {len(df):,} ячеек с {H3_MIN_POINTS}+ точек")
    if df.empty:
        print("    нет данных — пропускаю")
        return

    vmin, vmax = df[value_col].min(), df[value_col].max()
    norm = mcolors.LogNorm(vmin=max(vmin, 1), vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)

    fig, ax = plt.subplots(figsize=(11, 10))
    for _, row in df.iterrows():
        boundary = h3.cell_to_boundary(row["h3"])
        poly = [(lon, lat) for lat, lon in boundary]
        color = cmap_obj(norm(row[value_col]))
        ax.fill(*zip(*poly), color=color, edgecolor="none")

    # грубые границы СПб для ориентира
    ax.set_xlim(29.40, 30.80)
    ax.set_ylim(59.70, 60.30)
    ax.set_aspect("equal")
    ax.set_xlabel("Долгота")
    ax.set_ylabel("Широта")
    ax.set_title(title, fontsize=13)

    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7)
    cbar.set_label(value_col, fontsize=11)

    plt.tight_layout()
    plt.savefig(FIG / fname)
    plt.close()

# ---------- 1. Карта плотности всех точек ----------
print("\nКарта 1: все точки ...")
draw_map(agg_spb, "n_points",
         "Плотность GPS-точек по H3-ячейкам (~1.2 км)\n(все данные, включая базы и стоянки)",
         "fig1_map_all.png", cmap="YlOrRd")

# ---------- 2. Карта только движения ----------
print("\nКарта 2: только движение (speed_kmh >= 1) ...")
draw_map(agg_spb, "n_moving",
         "Плотность GPS-точек только в движении (speed ≥ 1 км/ч)\n(исключены базы, стоянки, депо)",
         "fig1_map_moving.png", cmap="YlGnBu")

# ---------- 3. Топ-20 «горячих» H3-ячеек по движению ----------
print("\nТоп-20 горячих ячеек (по движению) ...")
top = (agg_spb.sort_values("n_moving", ascending=False)
       .head(20)[["h3", "lat_c", "lon_c", "n_moving", "n_points", "n_machines"]])
print(top.to_string(index=False))

top.to_csv(FIG / "top20_h3_moving.csv", index=False)
print(f"  saved: {FIG / 'top20_h3_moving.csv'}")

# ---------- 4. Топ-20 «дыр» — ячейки с МАЛЫМ движением, но с техникой ----------
print("\nТоп-20 «дыр» (много машин, мало движения) ...")
# Берём ячейки с n_machines >= 10 (там реально бывает техника), но n_moving маленький
holes = (agg_spb[agg_spb["n_machines"] >= 10]
         .sort_values("n_moving", ascending=True)
         .head(20)[["h3", "lat_c", "lon_c", "n_moving", "n_points", "n_machines"]])
print(holes.to_string(index=False))
holes.to_csv(FIG / "top20_h3_holes.csv", index=False)
print(f"  saved: {FIG / 'top20_h3_holes.csv'}")

print(f"\nГотово. Карты и CSV в {FIG}")