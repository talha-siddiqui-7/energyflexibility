# -*- coding: utf-8 -*-
"""
15-min ahead predictions for:
  - Extract (hall) air temperature [°C]
  - Supply mass flow [kg/s]
  - Supply RH [%]

Time-series safe split (chronological 80/20).
Shows pop-up plots for quick inspection.

Requires: pandas, numpy, scikit-learn, matplotlib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
DATA_PATH = r"M:\PhD\02 Data sets from simulations\blackbox datasets\Air\air_flex.csv"
# Schedules (from your description)
RH_DAY_PCT = 55.0
RH_NIGHT_PCT = 65.0          # 20:00–02:00
TSA_SP_C = 30.0              # constant
ACT_DAY = 1.0                # 09:00–20:00
ACT_NIGHT = 0.5

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def find_col(frame, tokens, required=True):
    """Loosely match a column by required substrings (case-insensitive)."""
    toks = [t.lower() for t in tokens]
    for c in frame.columns:
        name = str(c).lower()
        if all(t in name for t in toks):
            return c
    if required:
        raise KeyError(f"Column with tokens {tokens} not found. Columns: {list(frame.columns)[:10]} ...")
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

def build_model():
    return HistGradientBoostingRegressor(
        max_depth=5, max_iter=200, learning_rate=0.08, min_samples_leaf=35,
        early_stopping=True, validation_fraction=0.1
    )

# ---------------------------------------------------------------------
# Load + parse
# ---------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

# Time index (robust): try to parse "Time" as fractional hours; else parse a timestamp column
time_col = find_col(df, ["time"], required=False) or find_col(df, ["hour"], required=False)
if time_col:
    # If numeric → fractional hours from a fixed start
    if np.issubdtype(df[time_col].dtype, np.number):
        t0 = pd.Timestamp("2024-01-01 00:00:00")
        df["ts"] = [t0 + pd.Timedelta(hours=float(h)) for h in df[time_col]]
    else:
        df["ts"] = pd.to_datetime(df[time_col], errors="coerce")
else:
    raise ValueError("Could not detect a time or hour column; please add one or rename to 'Time'.")

# Column names (robust matching)
COL_T_OUT  = find_col(df, ["air", "temperature", "outside"])
COL_RH_OUT = find_col(df, ["rh", "outside"])
COL_T_EX   = find_col(df, ["air", "temperature", "extract"])
COL_T_SUP  = find_col(df, ["air", "temperature", "supply"])
COL_W_EX   = find_col(df, ["humidity", "ratio", "extract"])  # your CSV may say "ration" — the matcher is tolerant
COL_W_SUP  = find_col(df, ["humidity", "ratio", "supply"])
COL_M_SUP  = find_col(df, ["mass", "flow", "supply"])
COL_M_EX   = find_col(df, ["mass", "flow", "extract"], required=False)
COL_RH_SUP = find_col(df, ["supply", "rh"])
COL_FA     = find_col(df, ["fresh", "air", "share"], required=False)

# Fill optional columns if absent
if COL_FA is None:
    df["Fresh air share (dummy)"] = 0.0
    COL_FA = "Fresh air share (dummy)"

# ---------------------------------------------------------------------
# Build schedules and encodings
# ---------------------------------------------------------------------
df["hour"] = df["ts"].dt.hour
df["minute_of_day"] = df["hour"] * 60 + df["ts"].dt.minute
df["day_of_year"] = df["ts"].dt.dayofyear

# Activity schedule
df["activity_factor"] = np.where((df["hour"] >= 9) & (df["hour"] < 20), ACT_DAY, ACT_NIGHT)

# RH setpoint schedule: 65% from 20:00–02:00, else 55%
df["RH_sp_hall"] = RH_DAY_PCT
df.loc[(df["hour"] >= 20) | (df["hour"] < 2), "RH_sp_hall"] = RH_NIGHT_PCT

# Supply air temperature setpoint
df["Tsa_sp"] = TSA_SP_C

# Time features
df["sin_hour"] = np.sin(2.0 * pi * df["minute_of_day"] / (24 * 60))
df["cos_hour"] = np.cos(2.0 * pi * df["minute_of_day"] / (24 * 60))
df["sin_doy"]  = np.sin(2.0 * pi * (df["day_of_year"] - 1) / 365.0)
df["cos_doy"]  = np.cos(2.0 * pi * (df["day_of_year"] - 1) / 365.0)

# ---------------------------------------------------------------------
# Lags to capture closed-loop dynamics
# ---------------------------------------------------------------------
for c in [COL_M_SUP, COL_RH_SUP, COL_T_EX, COL_W_EX]:
    df = add_lags(df, c, lags=(1, 4))  # 15-min and 60-min

# Features at time t
FEATURES = [
    COL_T_OUT, COL_RH_OUT, COL_T_EX, COL_W_EX, COL_T_SUP, COL_FA,
    "activity_factor", "RH_sp_hall", "Tsa_sp",
    "sin_hour", "cos_hour", "sin_doy", "cos_doy",
    f"{COL_M_SUP}__lag1", f"{COL_M_SUP}__lag4",
    f"{COL_RH_SUP}__lag1", f"{COL_RH_SUP}__lag4",
    f"{COL_T_EX}__lag1",  f"{COL_T_EX}__lag4",
    f"{COL_W_EX}__lag1",  f"{COL_W_EX}__lag4",
]

# Targets at t+1 (next 15-min)
df["T_ex_next"]  = df[COL_T_EX].shift(-1)
df["M_sup_next"] = df[COL_M_SUP].shift(-1)
df["RH_sup_next"]= df[COL_RH_SUP].shift(-1)

data = df.dropna(subset=FEATURES + ["T_ex_next", "M_sup_next", "RH_sup_next"]).reset_index(drop=True)

# ---------------------------------------------------------------------
# Chronological split (time-series safe)
# ---------------------------------------------------------------------
n = len(data)
split_idx = int(n * 0.80)  # 80% train, 20% test; NO SHUFFLING
train = data.iloc[:split_idx].copy()
test  = data.iloc[split_idx:].copy()

X_tr = train[FEATURES].values
X_te = test[FEATURES].values
yT_tr, yT_te = train["T_ex_next"].values,  test["T_ex_next"].values
yM_tr, yM_te = train["M_sup_next"].values, test["M_sup_next"].values
yR_tr, yR_te = train["RH_sup_next"].values, test["RH_sup_next"].values

# ---------------------------------------------------------------------
# Train three separate models (often better than one multi-output)
# ---------------------------------------------------------------------
model_T = build_model().fit(X_tr, yT_tr)
model_M = build_model().fit(X_tr, yM_tr)
model_R = build_model().fit(X_tr, yR_tr)

yT_pr = model_T.predict(X_te)
yM_pr = model_M.predict(X_te)
yR_pr = model_R.predict(X_te)

# Persistence baselines (next = now, taken from last part of train)
yT_pers = train[COL_T_EX].values[-len(test):]
yM_pers = train[COL_M_SUP].values[-len(test):]
yR_pers = train[COL_RH_SUP].values[-len(test):]

# Metrics
def metrics_table(y_true, y_pred, y_persist):
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "MAPE_%": mape(y_true, y_pred),
        "R2":   float(r2_score(y_true, y_pred)),
        "Skill_vs_persistence": 1.0 - rmse(y_true, y_pred) / rmse(y_true, y_persist)
    }

metrics = {
    "Extract_T_C":  metrics_table(yT_te, yT_pr, yT_pers),
    "Supply_mdot_kg_s": metrics_table(yM_te, yM_pr, yM_pers),
    "Supply_RH_pct": metrics_table(yR_te, yR_pr, yR_pers),
}

print("\n=== Test Metrics (chronological 80/20 split) ===")
for k, v in metrics.items():
    print(f"\n{k}")
    for m, val in v.items():
        print(f"  {m:>22}: {val:,.4f}")

# ---------------------------------------------------------------------
# Plots — pop-up (no saving)
# ---------------------------------------------------------------------
test_t = pd.to_datetime(test["ts"])

def plot_last_7_days(y_true, y_pred, label):
    k = min(7*24*4, len(test_t))  # 7 days at 15-min
    t = test_t.iloc[-k:]
    yt = y_true[-k:]
    yp = y_pred[-k:]
    plt.figure()
    plt.plot(t, yt, label="Actual", linewidth=1.5)
    plt.plot(t, yp, label="Predicted", linewidth=1.2)
    plt.title(f"{label} — last 7 days (test)")
    plt.xlabel("Time"); plt.ylabel(label); plt.grid(True); plt.legend()
    plt.gcf().autofmt_xdate()
    plt.show()

def plot_scatter(y_true, y_pred, label):
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    pad = 0.05*(hi-lo + 1e-9)
    plt.figure()
    plt.scatter(y_true, y_pred, s=8, alpha=0.5)
    plt.plot([lo-pad, hi+pad], [lo-pad, hi+pad], "--", linewidth=1.0)
    plt.xlabel(f"Actual {label}"); plt.ylabel(f"Predicted {label}")
    plt.title(f"Actual vs Predicted — {label}"); plt.grid(True)
    plt.show()

def plot_residuals(y_true, y_pred, label):
    resid = y_pred - y_true
    plt.figure()
    plt.hist(resid, bins=40, alpha=0.9)
    plt.xlabel(f"Residual ({label})"); plt.ylabel("Count")
    plt.title(f"Residuals — {label}"); plt.grid(True)
    plt.show()

# Overlays
plot_last_7_days(yT_te, yT_pr, "Extract air temperature (°C)")
plot_last_7_days(yM_te, yM_pr, "Supply airflow (kg/s)")
plot_last_7_days(yR_te, yR_pr, "Supply RH (%)")

# Scatter
plot_scatter(yT_te, yT_pr, "T_extract (°C)")
plot_scatter(yM_te, yM_pr, "ṁ_sup (kg/s)")
plot_scatter(yR_te, yR_pr, "RH_sup (%)")

# Residual histograms
plot_residuals(yT_te, yT_pr, "°C")
plot_residuals(yM_te, yM_pr, "kg/s")
plot_residuals(yR_te, yR_pr, "%")

# Optional: quick feature importance (top 20)
def feature_importance(model, cols, title):
    imps = getattr(model, "feature_importances_", None)
    if imps is None:
        print(f"[{title}] Feature importances not available.")
        return
    order = np.argsort(imps)[::-1][:20]
    plt.figure()
    plt.barh(range(len(order))[::-1], imps[order][::-1])
    plt.yticks(range(len(order))[::-1], [cols[i] for i in order][::-1])
    plt.xlabel("Importance"); plt.title(title); plt.tight_layout(); plt.show()

feature_importance(model_T, FEATURES, "Feature importance — Extract T")
feature_importance(model_M, FEATURES, "Feature importance — Supply ṁ")
feature_importance(model_R, FEATURES, "Feature importance — Supply RH")
