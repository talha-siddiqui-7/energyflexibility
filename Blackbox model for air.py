# -*- coding: utf-8 -*-
"""
Seasonal (Winter/Summer/Year) averages & metrics with safe fallback:
- If a season has samples in TEST (last 20%), evaluate on TEST.
- Else, fall back to ALL-DATA (in-sample) and label clearly.

Targets (t+15 min): Extract T (°C), Supply ṁ (kg/s), Supply RH (%)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
DATA_PATH = r"M:\PhD\02 Data sets from simulations\blackbox datasets\Air\air_flex.csv"
WINTER_MONTHS = {12, 1, 2}
SUMMER_MONTHS = {6, 7, 8}
MIN_SAMPLES = 50  # threshold to avoid meaningless seasonal stats

# Schedules (your facility)
RH_DAY_PCT, RH_NIGHT_PCT = 55.0, 65.0   # 20:00–02:00 = 65%
TSA_SP_C = 30.0
ACT_DAY, ACT_NIGHT = 1.0, 0.5           # 09:00–20:00 = 1.0 else 0.5

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def find_col(frame, tokens, required=True):
    toks = [t.lower() for t in tokens]
    for c in frame.columns:
        name = str(c).lower()
        if all(t in name for t in toks):
            return c
    if required:
        raise KeyError(f"Column with tokens {tokens} not found. First cols: {list(frame.columns)[:12]}")
    return None

def add_lags(df, col, lags=(1, 4)):
    for L in lags:
        df[f"{col}__lag{L}"] = df[col].shift(L)
    return df

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mape(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    denom = np.maximum(np.abs(y_true), 1e-6)
    return float(np.mean(np.abs(y_pred - y_true) / denom) * 100.0)

def metrics_table(y_true, y_pred, y_pers):
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "MAPE_%": mape(y_true, y_pred),
        "R2":   float(r2_score(y_true, y_pred)),
        "Skill_vs_persistence": 1.0 - rmse(y_true, y_pred) / rmse(y_true, y_pers)
    }

def build_model():
    return HistGradientBoostingRegressor(
        max_depth=5, max_iter=200, learning_rate=0.08, min_samples_leaf=35,
        early_stopping=True, validation_fraction=0.1
    )

def average_daily_overlay(min_of_day, mask, y_true, y_pred, title, ylabel):
    dfp = pd.DataFrame({"mod": min_of_day[mask], "y_true": y_true[mask], "y_pred": y_pred[mask]})
    g = dfp.groupby("mod").mean().reset_index().sort_values("mod")
    x = g["mod"].values / 60.0
    plt.figure()
    plt.plot(x, g["y_true"].values, label="Actual", linewidth=1.6)
    plt.plot(x, g["y_pred"].values, label="Predicted", linewidth=1.2)
    plt.xlabel("Hour of day"); plt.ylabel(ylabel); plt.title(title)
    plt.grid(True); plt.xlim(0, 24); plt.legend(); plt.show()

def scatter_plot(y_true, y_pred, mask, title, xlabel):
    y1, y2 = y_true[mask], y_pred[mask]
    lo = float(min(np.min(y1), np.min(y2))); hi = float(max(np.max(y1), np.max(y2)))
    pad = 0.05*(hi-lo + 1e-9)
    plt.figure()
    plt.scatter(y1, y2, s=8, alpha=0.5)
    plt.plot([lo-pad, hi+pad], [lo-pad, hi+pad], "--", linewidth=1.0)
    plt.xlabel(f"Actual {xlabel}"); plt.ylabel(f"Predicted {xlabel}")
    plt.title(title); plt.grid(True); plt.show()

def residual_hist(y_true, y_pred, mask, title, unit):
    resid = y_pred[mask] - y_true[mask]
    plt.figure()
    plt.hist(resid, bins=40, alpha=0.9)
    plt.xlabel(f"Residual ({unit})"); plt.ylabel("Count")
    plt.title(title); plt.grid(True); plt.show()

# ---------------------------------------------------------------------
# Load + parse
# ---------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

# Time index (use fractional hours if numeric; else parse datetime)
time_col = find_col(df, ["time"], required=False) or find_col(df, ["hour"], required=False)
if time_col is None:
    raise ValueError("No 'Time'/'Hour' column detected.")
if np.issubdtype(df[time_col].dtype, np.number):
    t0 = pd.Timestamp("2024-01-01 00:00:00")
    df["ts"] = [t0 + pd.Timedelta(hours=float(h)) for h in df[time_col]]
else:
    df["ts"] = pd.to_datetime(df[time_col], errors="coerce")

# Key columns (tolerant to slight wording)
COL_T_OUT  = find_col(df, ["air", "temperature", "outside"])
COL_RH_OUT = find_col(df, ["rh", "outside"])
COL_T_EX   = find_col(df, ["air", "temperature", "extract"])
COL_T_SUP  = find_col(df, ["air", "temperature", "supply"])
COL_W_EX   = find_col(df, ["humidity", "ratio", "extract"])
COL_RH_SUP = find_col(df, ["supply", "rh"])
COL_M_SUP  = find_col(df, ["mass", "flow", "supply"])
COL_FA     = find_col(df, ["fresh", "air", "share"], required=False)
if COL_FA is None:
    df["Fresh air share (dummy)"] = 0.0
    COL_FA = "Fresh air share (dummy)"

# Schedules & encodings
df["hour"] = df["ts"].dt.hour
df["minute_of_day"] = df["hour"] * 60 + df["ts"].dt.minute
df["day_of_year"] = df["ts"].dt.dayofyear
df["month"] = df["ts"].dt.month

df["activity_factor"] = np.where((df["hour"] >= 9) & (df["hour"] < 20), ACT_DAY, ACT_NIGHT)
df["RH_sp_hall"] = RH_DAY_PCT
df.loc[(df["hour"] >= 20) | (df["hour"] < 2), "RH_sp_hall"] = RH_NIGHT_PCT
df["Tsa_sp"] = TSA_SP_C

df["sin_hour"] = np.sin(2.0 * pi * df["minute_of_day"] / (24 * 60))
df["cos_hour"] = np.cos(2.0 * pi * df["minute_of_day"] / (24 * 60))
df["sin_doy"]  = np.sin(2.0 * pi * (df["day_of_year"] - 1) / 365.0)
df["cos_doy"]  = np.cos(2.0 * pi * (df["day_of_year"] - 1) / 365.0)

for c in [COL_M_SUP, COL_RH_SUP, COL_T_EX, COL_W_EX]:
    df = add_lags(df, c, lags=(1, 4))

FEATURES = [
    COL_T_OUT, COL_RH_OUT, COL_T_EX, COL_W_EX, COL_T_SUP, COL_FA,
    "activity_factor", "RH_sp_hall", "Tsa_sp",
    "sin_hour", "cos_hour", "sin_doy", "cos_doy",
    f"{COL_M_SUP}__lag1", f"{COL_M_SUP}__lag4",
    f"{COL_RH_SUP}__lag1", f"{COL_RH_SUP}__lag4",
    f"{COL_T_EX}__lag1",  f"{COL_T_EX}__lag4",
    f"{COL_W_EX}__lag1",  f"{COL_W_EX}__lag4",
]

# Targets (t+1)
df["T_ex_next"]  = df[COL_T_EX].shift(-1)
df["M_sup_next"] = df[COL_M_SUP].shift(-1)
df["RH_sup_next"]= df[COL_RH_SUP].shift(-1)

data = df.dropna(subset=FEATURES + ["T_ex_next","M_sup_next","RH_sup_next"]).reset_index(drop=True)

# Chronological 80/20 split
n = len(data); split_idx = int(n*0.80)
train, test = data.iloc[:split_idx].copy(), data.iloc[split_idx:].copy()

X_tr, X_te = train[FEATURES].values, test[FEATURES].values
yT_tr, yT_te = train["T_ex_next"].values,  test["T_ex_next"].values
yM_tr, yM_te = train["M_sup_next"].values, test["M_sup_next"].values
yR_tr, yR_te = train["RH_sup_next"].values, test["RH_sup_next"].values

# Train
mT, mM, mR = build_model(), build_model(), build_model()
mT.fit(X_tr, yT_tr); mM.fit(X_tr, yM_tr); mR.fit(X_tr, yR_tr)

# Predictions
yT_pr_test, yM_pr_test, yR_pr_test = mT.predict(X_te), mM.predict(X_te), mR.predict(X_te)

# Baselines aligned to test (next = now)
yT_pers_test = train[COL_T_EX].values[-len(test):]
yM_pers_test = train[COL_M_SUP].values[-len(test):]
yR_pers_test = train[COL_RH_SUP].values[-len(test):]

# Also compute ALL-DATA predictions for fallback
X_all = data[FEATURES].values
yT_all_true = data["T_ex_next"].values
yM_all_true = data["M_sup_next"].values
yR_all_true = data["RH_sup_next"].values
yT_all_pred = mT.predict(X_all)
yM_all_pred = mM.predict(X_all)
yR_all_pred = mR.predict(X_all)
yT_pers_all = data[COL_T_EX].values
yM_pers_all = data[COL_M_SUP].values
yR_pers_all = data[COL_RH_SUP].values

# Time / masks
test_month = test["month"].values
test_mod   = test["minute_of_day"].values
all_month  = data["month"].values
all_mod    = data["minute_of_day"].values

targets = {
    "Extract air temperature (°C)": {"unit":"°C",
        "y_true_test": yT_te, "y_pred_test": yT_pr_test, "y_pers_test": yT_pers_test,
        "y_true_all": yT_all_true, "y_pred_all": yT_all_pred, "y_pers_all": yT_pers_all},
    "Supply airflow (kg/s)": {"unit":"kg/s",
        "y_true_test": yM_te, "y_pred_test": yM_pr_test, "y_pers_test": yM_pers_test,
        "y_true_all": yM_all_true, "y_pred_all": yM_all_pred, "y_pers_all": yM_pers_all},
    "Supply RH (%)": {"unit":"%",
        "y_true_test": yR_te, "y_pred_test": yR_pr_test, "y_pers_test": yR_pers_test,
        "y_true_all": yR_all_true, "y_pred_all": yR_all_pred, "y_pers_all": yR_pers_all},
}

seasons = [
    ("Winter (Dec–Feb)",  WINTER_MONTHS),
    ("Summer (Jun–Aug)",  SUMMER_MONTHS),
    ("Whole year",        set(range(1,13))),
]

print("\nSeasonal evaluation (prefers TEST; falls back to ALL-DATA if needed)\n")

for season_name, month_set in seasons:
    # masks
    mask_test = np.isin(test_month, list(month_set))
    mask_all  = np.isin(all_month,  list(month_set))

    print("="*72)
    print(f"{season_name}".center(72))
    print("="*72)

    use_test = (mask_test.sum() >= MIN_SAMPLES)
    source = "test" if use_test else "in-sample (all data)"
    print(f"Samples used: {mask_test.sum() if use_test else mask_all.sum()}  |  Source: {source}")

    for label, d in targets.items():
        unit = d["unit"]
        if use_test:
            y_true, y_pred, y_pers = d["y_true_test"], d["y_pred_test"], d["y_pers_test"]
            mod_arr = test_mod
            mask = mask_test
            tag = "(test)"
        else:
            y_true, y_pred, y_pers = d["y_true_all"], d["y_pred_all"], d["y_pers_all"]
            mod_arr = all_mod
            mask = mask_all
            tag = "(in-sample)"

        if mask.sum() < MIN_SAMPLES:
            print(f"\n{label} — not enough samples ({mask.sum()}) for reliable stats; skipping.\n")
            continue

        # Metrics
        m = metrics_table(y_true[mask], y_pred[mask], y_pers[mask])
        print(f"\n{label} {tag}")
        for k, v in m.items():
            print(f"  {k:>22}: {v:,.4f}")

        # Average daily overlay
        average_daily_overlay(
            mod_arr, mask, y_true, y_pred,
            title=f"{label} — average daily profile {tag} | {season_name}",
            ylabel=label
        )
        # Scatter & residuals
        scatter_plot(y_true, y_pred, mask,
                     title=f"Actual vs Predicted — {label} {tag} | {season_name}",
                     xlabel=label)
        residual_hist(y_true, y_pred, mask,
                      title=f"Residuals — {label} {tag} | {season_name}", unit=unit)

print("\nDone.")
