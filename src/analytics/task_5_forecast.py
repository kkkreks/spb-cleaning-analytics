"""
Задача 5. Прогнозирование активности уборочной техники.

Две задачи:
  A) Регрессия: сколько активных машин будет в H3-ячейке через 1 час.
  B) Бинарная: будет ли в H3-ячейке хотя бы 1 машина через 1 час.

Train: 18-20 октября (полные дни), Test: 21-23 октября (полные дни).
Обе задачи сравниваются с baseline.
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
import seaborn as sns

import lightgbm as lgb


def mae(a, b): return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))
def rmse(a, b): return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
def accuracy(a, b): return float(np.mean(np.asarray(a) == np.asarray(b)))
def f1_score_manual(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)


# ================= НАСТРОЙКИ =================
CLEAN = Path(r"C:\Users\Admin\Desktop\УЧЁБА\Магистратура\Большие данные\ДАННЫЕ\clean_all.parquet")
WEATHER = Path(r"C:\Users\Admin\Desktop\УЧЁБА\Магистратура\Большие данные\ДАННЫЕ\weather_hourly.parquet")
FIG = Path(r"C:\Users\Admin\Desktop\УЧЁБА\Магистратура\Большие данные\figures\task5_forecast")
FIG.mkdir(parents=True, exist_ok=True)

H3_RES = 7
MIN_HOUR_MACHINES = 1
TRAIN_END = "2026-10-20"     # train: 18-20
TEST_START = "2026-10-21"    # test:  21-23
TEST_END = "2026-10-23"
TARGET_SHIFT = 1
# ============================================

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 200

con = duckdb.connect()

# ---------- 1. Агрегация ----------
print("Шаг 1. Агрегирую (h3, час) -> число уникальных машин ...")
df = con.execute(f"""
    SELECT lat, lon,
           CAST(date_trunc('hour', dt) AS TIMESTAMP) AS dt_hour,
           id
    FROM read_parquet('{CLEAN.as_posix()}')
    WHERE speed_kmh IS NULL OR speed_kmh >= 1
""").df()
df["h3"] = [h3.latlng_to_cell(lat, lon, H3_RES)
            for lat, lon in zip(df["lat"].values, df["lon"].values)]

agg = (df.groupby(["h3", "dt_hour"], as_index=False)
         .agg(n_machines=("id", "nunique")))
agg = agg[agg["n_machines"] >= MIN_HOUR_MACHINES].copy()
print(f"  строк (h3, час): {len(agg):,}")
print(f"  период: {agg['dt_hour'].min()} → {agg['dt_hour'].max()}")

# полная сетка h3 × час
all_hours = pd.date_range(agg["dt_hour"].min(), agg["dt_hour"].max(),
                          freq="h", tz=agg["dt_hour"].dt.tz)
all_h3 = agg["h3"].unique()
grid = pd.MultiIndex.from_product([all_h3, all_hours], names=["h3", "dt_hour"]).to_frame(index=False)
data = grid.merge(agg, on=["h3", "dt_hour"], how="left")
data["n_machines"] = data["n_machines"].fillna(0).astype(int)
print(f"  полная сетка: {len(data):,}")

# ---------- 2. Погода ----------
print("Шаг 2. Погода ...")
w = pd.read_parquet(WEATHER)
w["dt_hour"] = w["dt_hour"].dt.tz_convert(data["dt_hour"].dt.tz)
data = data.merge(w, on="dt_hour", how="left")

# ---------- 3. Признаки ----------
print("Шаг 3. Признаки ...")
data = data.sort_values(["h3", "dt_hour"]).reset_index(drop=True)
data["hour"] = data["dt_hour"].dt.hour
data["dow"] = data["dt_hour"].dt.dayofweek
data["date"] = data["dt_hour"].dt.date

data["lag_1"] = data.groupby("h3")["n_machines"].shift(1)
data["lag_2"] = data.groupby("h3")["n_machines"].shift(2)
data["lag_3"] = data.groupby("h3")["n_machines"].shift(3)
data["lag_24"] = data.groupby("h3")["n_machines"].shift(24)
data["roll_3"] = (data.groupby("h3")["n_machines"]
                    .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean()))

# target: регрессия (сколько машин через 1 час) и бинарная (>= 1)
data["y_reg"] = data.groupby("h3")["n_machines"].shift(-TARGET_SHIFT)
data["y_bin"] = (data["y_reg"] >= 1).astype("int8")

data = data.dropna(subset=["y_reg", "lag_1"]).copy()
print(f"  строк для модели: {len(data):,}")

# ---------- 4. Train/Test только на полных днях ----------
print(f"Шаг 4. Train: до {TRAIN_END}, Test: {TEST_START}..{TEST_END}")
train = data[data["dt_hour"].dt.date <= pd.to_datetime(TRAIN_END).date()].copy()
test = data[(data["dt_hour"].dt.date >= pd.to_datetime(TEST_START).date()) &
            (data["dt_hour"].dt.date <= pd.to_datetime(TEST_END).date())].copy()
print(f"  train: {len(train):,}  ({train['dt_hour'].dt.date.min()} → {train['dt_hour'].dt.date.max()})")
print(f"  test:  {len(test):,}   ({test['dt_hour'].dt.date.min()} → {test['dt_hour'].dt.date.max()})")

FEATURES = ["hour", "dow",
            "lag_1", "lag_2", "lag_3", "lag_24", "roll_3",
            "temp_c", "precip_mm", "snow_cm", "wind_kmh"]
X_tr, X_te = train[FEATURES], test[FEATURES]

# ---------- 5. Baseline ----------
print("\nШаг 5. Baseline ...")
baseline = (train.groupby(["h3", "hour", "dow"])["n_machines"]
                 .mean().reset_index().rename(columns={"n_machines": "pred_base"}))
fallback = train.groupby(["hour", "dow"])["n_machines"].mean().reset_index()
test_b = test.merge(baseline, on=["h3", "hour", "dow"], how="left")
test_b = test_b.merge(fallback, on=["hour", "dow"], how="left", suffixes=("", "_fb"))
test_b["pred_base"] = test_b["pred_base"].fillna(test_b["n_machines_fb"]).fillna(0)
test_b["pred_base_bin"] = (test_b["pred_base"] >= 0.5).astype(int)

mae_b = mae(test["y_reg"], test_b["pred_base"])
rmse_b = rmse(test["y_reg"], test_b["pred_base"])
acc_b = accuracy(test["y_bin"], test_b["pred_base_bin"])
f1_b = f1_score_manual(test["y_bin"], test_b["pred_base_bin"])
print(f"  [REG] MAE={mae_b:.3f}  RMSE={rmse_b:.3f}")
print(f"  [BIN] Acc={acc_b:.3f}  F1={f1_b:.3f}")

# ---------- 6. LightGBM: регрессия ----------
print("\nШаг 6. LightGBM — регрессия ...")
lgb_reg = lgb.LGBMRegressor(
    n_estimators=800, learning_rate=0.03, num_leaves=63,
    min_child_samples=30, subsample=0.9, colsample_bytree=0.9,
    random_state=42, n_jobs=-1, verbose=-1,
)
lgb_reg.fit(
    X_tr, train["y_reg"],
    eval_set=[(X_te, test["y_reg"])],
    eval_metric="mae",
    callbacks=[lgb.early_stopping(50, verbose=False)],
)
pred_reg = np.clip(lgb_reg.predict(X_te), 0, None)
mae_l = mae(test["y_reg"], pred_reg)
rmse_l = rmse(test["y_reg"], pred_reg)
print(f"  [REG] MAE={mae_l:.3f}  RMSE={rmse_l:.3f}")

# ---------- 7. LightGBM: бинарная ----------
print("\nШаг 7. LightGBM — бинарная классификация ...")
lgb_bin = lgb.LGBMClassifier(
    n_estimators=800, learning_rate=0.03, num_leaves=63,
    min_child_samples=30, subsample=0.9, colsample_bytree=0.9,
    random_state=42, n_jobs=-1, verbose=-1,
)
lgb_bin.fit(
    X_tr, train["y_bin"],
    eval_set=[(X_te, test["y_bin"])],
    eval_metric="binary_logloss",
    callbacks=[lgb.early_stopping(50, verbose=False)],
)
pred_bin = lgb_bin.predict(X_te)
acc_l = accuracy(test["y_bin"], pred_bin)
f1_l = f1_score_manual(test["y_bin"], pred_bin)
print(f"  [BIN] Acc={acc_l:.3f}  F1={f1_l:.3f}")

# ---------- 8. Метрики по часам ----------
test["pred_reg"] = pred_reg
test["pred_base"] = test_b["pred_base"].values
test["pred_bin"] = pred_bin
test["pred_base_bin"] = test_b["pred_base_bin"].values

mae_by_hour_l = test.assign(err=(test["pred_reg"] - test["y_reg"]).abs()).groupby("hour")["err"].mean()
mae_by_hour_b = test.assign(err=(test["pred_base"] - test["y_reg"]).abs()).groupby("hour")["err"].mean()
acc_by_hour_l = test.assign(c=(test["pred_bin"] == test["y_bin"]).astype(int)).groupby("hour")["c"].mean()
acc_by_hour_b = test.assign(c=(test["pred_base_bin"] == test["y_bin"]).astype(int)).groupby("hour")["c"].mean()

# ---------- 9. Сохранение ----------
metrics_df = pd.DataFrame([
    {"task": "regression", "model": "baseline", "MAE": mae_b, "RMSE": rmse_b},
    {"task": "regression", "model": "LightGBM", "MAE": mae_l, "RMSE": rmse_l},
    {"task": "binary", "model": "baseline", "Accuracy": acc_b, "F1": f1_b},
    {"task": "binary", "model": "LightGBM", "Accuracy": acc_l, "F1": f1_l},
])
metrics_df.to_csv(FIG / "metrics.csv", index=False)
print("\nmetrics.csv:")
print(metrics_df.to_string(index=False))

# ---------- 10. Графики ----------
print("\nШаг 8. Графики ...")

# 10.1. Регрессия: факт vs прогноз (суммарно по всем ячейкам)
agg_t = (test.groupby("dt_hour", as_index=False)
             .agg(y=("y_reg", "sum"),
                  pred=("pred_reg", "sum"),
                  pred_b=("pred_base", "sum")))
fig, ax = plt.subplots(figsize=(14, 4.5))
ax.plot(agg_t["dt_hour"], agg_t["y"], label="факт", color="black", lw=1.5)
ax.plot(agg_t["dt_hour"], agg_t["pred"], label="LightGBM", color="#C44E52", lw=1.2)
ax.plot(agg_t["dt_hour"], agg_t["pred_b"], label="baseline", color="#4C72B0", lw=1.2, alpha=0.7)
ax.set_title(f"Регрессия: суммарные машины (test {TEST_START}..{TEST_END}) — MAE={mae_l:.2f}")
ax.set_xlabel("Время"); ax.set_ylabel("Машин")
ax.legend()
plt.tight_layout(); plt.savefig(FIG / "fig5_reg_series.png"); plt.close()

# 10.2. Регрессия: scatter
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(test["y_reg"], test["pred_reg"], s=4, alpha=0.25, color="#4C72B0")
lim = max(test["y_reg"].max(), test["pred_reg"].max())
ax.plot([0, lim], [0, lim], "r--", lw=1)
ax.set_xlabel("Факт"); ax.set_ylabel("Прогноз")
ax.set_title(f"Регрессия: LightGBM (MAE={mae_l:.2f}, RMSE={rmse_l:.2f})")
plt.tight_layout(); plt.savefig(FIG / "fig5_reg_scatter.png"); plt.close()

# 10.3. Регрессия: MAE по часам
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(mae_by_hour_l.index, mae_by_hour_l.values, marker="o", color="#C44E52", label="LightGBM")
ax.plot(mae_by_hour_b.index, mae_by_hour_b.values, marker="s", color="#4C72B0", label="baseline")
ax.set_title("Регрессия: MAE по часам")
ax.set_xlabel("Час"); ax.set_ylabel("MAE")
ax.set_xticks(range(24)); ax.legend()
plt.tight_layout(); plt.savefig(FIG / "fig5_reg_mae_by_hour.png"); plt.close()

# 10.4. Бинарная: accuracy по часам
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(acc_by_hour_l.index, acc_by_hour_l.values, marker="o", color="#C44E52", label="LightGBM")
ax.plot(acc_by_hour_b.index, acc_by_hour_b.values, marker="s", color="#4C72B0", label="baseline")
ax.set_title("Бинарная: Accuracy по часам")
ax.set_xlabel("Час"); ax.set_ylabel("Accuracy")
ax.set_xticks(range(24)); ax.set_ylim(0.5, 1.0); ax.legend()
plt.tight_layout(); plt.savefig(FIG / "fig5_bin_acc_by_hour.png"); plt.close()

# 10.5. Бинарная: confusion matrix
tp = int(((test["y_bin"] == 1) & (test["pred_bin"] == 1)).sum())
fp = int(((test["y_bin"] == 0) & (test["pred_bin"] == 1)).sum())
fn = int(((test["y_bin"] == 1) & (test["pred_bin"] == 0)).sum())
tn = int(((test["y_bin"] == 0) & (test["pred_bin"] == 0)).sum())
cm = np.array([[tn, fp], [fn, tp]])
fig, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["pred 0", "pred 1"], yticklabels=["true 0", "true 1"])
ax.set_title(f"Бинарная: Confusion Matrix (F1={f1_l:.3f})")
plt.tight_layout(); plt.savefig(FIG / "fig5_bin_cm.png"); plt.close()

# 10.6. Важность признаков
imp = pd.DataFrame({"feature": FEATURES, "gain": lgb_reg.feature_importances_}).sort_values("gain")
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(imp["feature"], imp["gain"], color="#55A868")
ax.set_title("Важность признаков (gain, регрессионная модель)")
ax.set_xlabel("Gain")
plt.tight_layout(); plt.savefig(FIG / "fig5_feature_importance.png"); plt.close()

# 10.7. Сравнение моделей (регрессия и бинарная)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].bar(["baseline", "LightGBM"], [mae_b, mae_l], color=["#4C72B0", "#C44E52"])
axes[0].set_title("Регрессия: MAE (меньше — лучше)")
for i, v in enumerate([mae_b, mae_l]):
    axes[0].text(i, v, f"{v:.2f}", ha="center", va="bottom")
axes[1].bar(["baseline", "LightGBM"], [f1_b, f1_l], color=["#4C72B0", "#C44E52"])
axes[1].set_title("Бинарная: F1 (больше — лучше)")
for i, v in enumerate([f1_b, f1_l]):
    axes[1].text(i, v, f"{v:.3f}", ha="center", va="bottom")
plt.tight_layout(); plt.savefig(FIG / "fig5_models_compare.png"); plt.close()

print(f"\n=== ИТОГ ===")
print(f"Регрессия:")
print(f"  baseline: MAE={mae_b:.3f}  RMSE={rmse_b:.3f}")
print(f"  LightGBM: MAE={mae_l:.3f}  RMSE={rmse_l:.3f}")
print(f"  улучшение MAE: {(mae_b-mae_l)/mae_b*100:.1f}%")
print(f"Бинарная:")
print(f"  baseline: Acc={acc_b:.3f}  F1={f1_b:.3f}")
print(f"  LightGBM: Acc={acc_l:.3f}  F1={f1_l:.3f}")
print(f"  улучшение F1: {(f1_l-f1_b)/max(f1_b,1e-9)*100:.1f}%")
print(f"\nГотово. Файлы: {FIG}")