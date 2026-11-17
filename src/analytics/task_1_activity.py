"""
Задача 1. Анализ активности уборочной техники СПб:
- по часам суток
- по дням недели
- heatmap «день × час»
- топ-15 операций (op)
- топ-20 H3-зон
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import duckdb
import h3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ================= НАСТРОЙКИ =================
CLEAN = Path(r"C:\Users\Admin\Desktop\УЧЁБА\Магистратура\Большие данные\ДАННЫЕ\clean_all.parquet")
FIG = Path(r"C:\Users\Admin\Desktop\УЧЁБА\Магистратура\Большие данные\figures\task1_activity")
FIG.mkdir(parents=True, exist_ok=True)
H3_RES = 7
# ============================================

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 200

con = duckdb.connect()

print("Читаю данные ...")
df = con.execute(f"""
    SELECT id, ut, lat, lon, op, date, hour, dow
    FROM read_parquet('{CLEAN.as_posix()}')
""").df()
print(f"  строк: {len(df):,}")

print("Добавляю H3-ячейки (res=7) ...")
df["h3_zone"] = [h3.latlng_to_cell(lat, lon, H3_RES)[:7]
                 for lat, lon in zip(df["lat"].values, df["lon"].values)]
print(f"  H3-зон: {df['h3_zone'].nunique()}")

# ---------- 1.1. По часам ----------
print("\n1.1. Активность по часам ...")
by_hour = df.groupby("hour", observed=True).size()
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.bar(by_hour.index, by_hour.values, color="#4C72B0")
ax.set_title("Активность уборочной техники по часам суток")
ax.set_xlabel("Час суток"); ax.set_ylabel("Число GPS-точек")
ax.set_xticks(range(24))
plt.tight_layout(); plt.savefig(FIG / "fig1_hours.png"); plt.close()
print(f"  пик {by_hour.idxmax()}:00 ({by_hour.max():,})")

# ---------- 1.2. По дням недели ----------
print("1.2. Активность по дням недели ...")
dow_names = ["Пн","Вт","Ср","Чт","Пт","Сб","Вс"]
by_dow = df.groupby("dow", observed=True).size()
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar([dow_names[i] for i in by_dow.index], by_dow.values, color="#DD8452")
ax.set_title("Активность по дням недели")
ax.set_ylabel("Число GPS-точек")
plt.tight_layout(); plt.savefig(FIG / "fig1_dow.png"); plt.close()

# ---------- 1.3. Heatmap день × час ----------
print("1.3. Heatmap день × час ...")
pivot = df.groupby(["date","hour"], observed=True).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(12, 5.5))
sns.heatmap(pivot, cmap="YlOrRd", ax=ax, linewidths=0.3,
            cbar_kws={"label": "GPS-точек"})
ax.set_title("Активность «день × час»")
ax.set_xlabel("Час суток"); ax.set_ylabel("Дата")
ax.set_xticks(range(0, 24, 2)); ax.set_xticklabels(range(0, 24, 2))
plt.tight_layout(); plt.savefig(FIG / "fig1_heatmap.png"); plt.close()

# ---------- 1.4. Топ-15 операций ----------
print("1.4. Топ-15 операций ...")
by_op = df.groupby("op", observed=True).size().sort_values(ascending=False).head(15)
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(by_op.index[::-1].astype(str), by_op.values[::-1], color="#55A868")
ax.set_title("Топ-15 операций (op)")
ax.set_xlabel("Число GPS-точек")
plt.tight_layout(); plt.savefig(FIG / "fig1_ops.png"); plt.close()
print("  топ-5:", by_op.head(5).to_dict())

# ---------- 1.5. Топ-20 H3-зон ----------
print("1.5. Топ-20 H3-зон ...")
by_h3 = df.groupby("h3_zone", observed=True).size().sort_values(ascending=False).head(20)
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(by_h3.index[::-1], by_h3.values[::-1], color="#C44E52")
ax.set_title("Топ-20 H3-зон по активности")
ax.set_xlabel("Число GPS-точек"); ax.set_ylabel("H3-зона")
plt.tight_layout(); plt.savefig(FIG / "fig1_h3_zones.png"); plt.close()

print(f"\nГотово. Файлы в {FIG}")