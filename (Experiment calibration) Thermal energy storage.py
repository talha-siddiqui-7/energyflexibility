#!/usr/bin/env python3
"""
Water flexibility (pool sensible storage) — dynamic validation using 1-minute data + window schedule

Adds:
- Per-window outputs + model component summaries
- Global error metrics (overall)
- Error metrics separated by activity_factor (printed to console + stored in Excel)

NEW:
- Excludes a user-specified unreliable time range from ALL processing:
  04.10.2025 08:20:00  →  04.10.2025 13:00:00  (inclusive)

Meaning:
- Any window that overlaps this excluded period is skipped entirely (no model, no measured ΔT, no components)
- The excluded period is also removed from the main 1-minute dataset to prevent accidental inclusion

OUTPUT (per window) will contain ONLY these columns (in this order):
start_time
end_time
activity_factor
duration_hours
Average thermal power [kW]
Delta T measured (from 1-min) [°C]
Delta T model [°C]
Delta T error [°C]

Global metrics (repeated in every row):
MAE_all_windows_[°C]
RMSE_all_windows_[°C]
Bias_all_windows_[°C]
MedAE_all_windows_[°C]
P25_abs_error_[°C]
P75_abs_error_[°C]
IQR_abs_error_[°C]
MAE_duration_weighted_[°C]
Median_relative_error_[%]
P90_relative_error_[%]

Per-activity-factor metrics (repeated in every row; separate column set per activity):
For each unique activity_factor value af:
  MAE_af_<af>_[°C]
  RMSE_af_<af>_[°C]
  Bias_af_<af>_[°C]
  MedAE_af_<af>_[°C]
  IQR_af_<af>_[°C]
  MAE_weighted_af_<af>_[°C]
  MedianRel_af_<af>_[%]
  P90Rel_af_<af>_[%]

...then model component summaries (avg kW + total kWh):
Avg Q_evap [kW],  E_evap [kWh]
Avg Q_conv [kW],  E_conv [kWh]
Avg Q_cond [kW],  E_cond [kWh]
Avg P_in [kW],    E_in [kWh]
Avg Q_net [kW],   E_net [kWh]
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ============================================================
# USER SETTINGS / TOGGLES
# ============================================================
USE_CALIBRATED_DYNAMIC = False
PLOTS_SHOW = True

WINDOWS_FILE_WIN = r"M:\PhD\03 Experiments\Time_windows_with_DeltaT_AvgPower.xlsx"
WINDOWS_FILE_FALLBACK = "/mnt/data/Time_windows_with_DeltaT_AvgPower.xlsx"

DATA_FILE_WIN = r"M:\PhD\03 Experiments\Pool_thermal_energy_input_1min_complete.xlsx"
DATA_FILE_FALLBACK = "/mnt/data/Pool_thermal_energy_input_1min_complete.xlsx"

OUTPUT_XLSX_NAME = "Water flexibility model results.xlsx"
OUTPUT_PATH_WIN = r"M:\PhD\03 Experiments\Water flexibility model results.xlsx"
OUTPUT_PATH_FALLBACK = f"/mnt/data/{OUTPUT_XLSX_NAME}"

DROP_NAN_ROWS_IN_WINDOW = True
INTERPOLATE_IN_WINDOW_IF_NEEDED = False

# ============================================================
# EXCLUDE UNRELIABLE TIME RANGE (inclusive)
# ============================================================
EXCLUDE_START = pd.Timestamp("2025-10-04 08:20:00")
EXCLUDE_END   = pd.Timestamp("2025-10-04 13:00:00")

# ============================================================
# COLUMN NAMES
# ============================================================
WIN_START_COL = "start_time"
WIN_END_COL = "end_time"
WIN_ACT_COL = "activity_factor"

TIME_COL = "Time"
P_KW_COL = "Pool thermal power input [kW]"
TAIR_COL = "Hall air Temp"
RH_PCT_COL = "Hall air RH"
TPOOL_COL = "Pool water Temp"

# ============================================================
# POOL / MODEL CONSTANTS
# ============================================================
L_POOL, W_POOL, D_POOL = 8.0, 12.5, 2.0
U_SIDE, U_BOTTOM = 2.94, 0.50
ROOM_DELTA_K = 3.0
C_EVAP = 4.0e-8
HFG = 2.43e6
RHO_W = 1000.0
CP_W = 4186.0
H_FIXED_WM2K = 2.0

ALPHA_P = 0.90
SCALE_UA = 1.15
SCALE_EVAP = 1.10

# ============================================================
# HELPERS
# ============================================================
def psat_pa(TC: float) -> float:
    return 610.94 * math.exp(17.625 * TC / (TC + 243.04))

def geometry():
    A_pool = L_POOL * W_POOL
    A_side = 2.0 * (L_POOL + W_POOL) * D_POOL
    A_bottom = A_pool
    V_pool = A_pool * D_POOL
    m_pool = RHO_W * V_pool
    return A_pool, A_side, A_bottom, m_pool

def safe_path(win_path: str, fallback_path: str) -> str:
    return win_path if os.path.exists(win_path) else fallback_path

def safe_output_path(win_out: str, fallback_out: str) -> str:
    win_dir = os.path.dirname(win_out)
    if win_dir and os.path.isdir(win_dir):
        return win_out
    return fallback_out

def to_float_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)

    s2 = s.astype(str)
    s2 = s2.str.replace("\u00A0", "", regex=False)
    s2 = s2.str.replace(" ", "", regex=False)
    s2 = s2.str.replace("%", "", regex=False)
    s2 = s2.str.replace("°", "", regex=False)
    s2 = s2.str.replace(",", ".", regex=False)
    s2 = s2.str.replace(r"[^0-9\.\-]+", "", regex=True)
    return pd.to_numeric(s2, errors="coerce")

def overlaps_excluded_range(t0: pd.Timestamp, t1: pd.Timestamp) -> bool:
    """True if [t0,t1] overlaps [EXCLUDE_START, EXCLUDE_END] (inclusive)."""
    if pd.isna(t0) or pd.isna(t1):
        return False
    return (t0 <= EXCLUDE_END) and (t1 >= EXCLUDE_START)

def integrate_window_dynamic_1min_with_components(win: pd.DataFrame, activity_factor: float):
    A_pool, A_side, A_bottom, m_pool = geometry()
    UA = (U_SIDE * A_side + U_BOTTOM * A_bottom)
    if USE_CALIBRATED_DYNAMIC:
        UA *= SCALE_UA
    h = H_FIXED_WM2K

    T = float(win[TPOOL_COL].iloc[0])
    T0 = T

    T_air = win[TAIR_COL].to_numpy(dtype=float)
    RH_frac = win[RH_PCT_COL].to_numpy(dtype=float) / 100.0
    P_W = win[P_KW_COL].to_numpy(dtype=float) * 1000.0
    if USE_CALIBRATED_DYNAMIC:
        P_W = P_W * ALPHA_P

    dt = 60.0

    E_evap_J = 0.0
    E_conv_J = 0.0
    E_cond_J = 0.0
    E_in_J = 0.0
    E_net_J = 0.0

    sum_Q_evap = 0.0
    sum_Q_conv = 0.0
    sum_Q_cond = 0.0
    sum_P_in = 0.0
    sum_Q_net = 0.0

    n = len(win)
    for i in range(n):
        Pa = RH_frac[i] * psat_pa(float(T_air[i]))
        dP = psat_pa(T) - Pa
        if dP < 0:
            dP = 0.0

        Q_evap = C_EVAP * A_pool * float(activity_factor) * dP * HFG
        if USE_CALIBRATED_DYNAMIC:
            Q_evap *= SCALE_EVAP

        Q_conv = h * A_pool * (T - float(T_air[i]))

        T_room = T - ROOM_DELTA_K
        Q_cond = UA * (T - T_room)

        P_in = float(P_W[i])
        Q_net = P_in - (Q_evap + Q_conv + Q_cond)

        E_evap_J += Q_evap * dt
        E_conv_J += Q_conv * dt
        E_cond_J += Q_cond * dt
        E_in_J += P_in * dt
        E_net_J += Q_net * dt

        sum_Q_evap += Q_evap
        sum_Q_conv += Q_conv
        sum_Q_cond += Q_cond
        sum_P_in += P_in
        sum_Q_net += Q_net

        dTdt = Q_net / (m_pool * CP_W)
        T = T + dTdt * dt

        if not np.isfinite(T):
            nan_comp = {
                "Avg Q_evap [kW]": np.nan, "E_evap [kWh]": np.nan,
                "Avg Q_conv [kW]": np.nan, "E_conv [kWh]": np.nan,
                "Avg Q_cond [kW]": np.nan, "E_cond [kWh]": np.nan,
                "Avg P_in [kW]": np.nan,   "E_in [kWh]": np.nan,
                "Avg Q_net [kW]": np.nan,  "E_net [kWh]": np.nan,
            }
            return np.nan, nan_comp

    J_to_kWh = 1.0 / 3.6e6
    comp = {
        "Avg Q_evap [kW]": (sum_Q_evap / n) / 1000.0,
        "E_evap [kWh]": E_evap_J * J_to_kWh,
        "Avg Q_conv [kW]": (sum_Q_conv / n) / 1000.0,
        "E_conv [kWh]": E_conv_J * J_to_kWh,
        "Avg Q_cond [kW]": (sum_Q_cond / n) / 1000.0,
        "E_cond [kWh]": E_cond_J * J_to_kWh,
        "Avg P_in [kW]": (sum_P_in / n) / 1000.0,
        "E_in [kWh]": E_in_J * J_to_kWh,
        "Avg Q_net [kW]": (sum_Q_net / n) / 1000.0,
        "E_net [kWh]": E_net_J * J_to_kWh,
    }
    return float(T - T0), comp

def compute_metrics(err: np.ndarray, meas: np.ndarray, dur: np.ndarray):
    out = {
        "N": 0,
        "MAE": np.nan,
        "RMSE": np.nan,
        "Bias": np.nan,
        "MedAE": np.nan,
        "P25": np.nan,
        "P75": np.nan,
        "IQR": np.nan,
        "Weighted_MAE": np.nan,
        "MedRelPct": np.nan,
        "P90RelPct": np.nan,
    }

    m = np.isfinite(err)
    if not np.any(m):
        return out

    v = err[m]
    out["N"] = int(v.size)
    out["MAE"] = float(np.mean(np.abs(v)))
    out["RMSE"] = float(np.sqrt(np.mean(v ** 2)))
    out["Bias"] = float(np.mean(v))

    abs_v = np.abs(v)
    out["MedAE"] = float(np.median(abs_v))
    out["P25"] = float(np.percentile(abs_v, 25))
    out["P75"] = float(np.percentile(abs_v, 75))
    out["IQR"] = float(out["P75"] - out["P25"])

    mw = np.isfinite(err) & np.isfinite(dur) & (dur > 0)
    if np.any(mw):
        out["Weighted_MAE"] = float(np.sum(np.abs(err[mw]) * dur[mw]) / np.sum(dur[mw]))

    eps = 1e-6
    mr = np.isfinite(err) & np.isfinite(meas) & (np.abs(meas) > eps)
    if np.any(mr):
        rel = np.abs(err[mr]) / np.abs(meas[mr])
        out["MedRelPct"] = float(np.median(rel) * 100.0)
        out["P90RelPct"] = float(np.percentile(rel, 90) * 100.0)

    return out

def af_tag(val: float) -> str:
    if not np.isfinite(val):
        return "nan"
    s = f"{float(val):.3f}".rstrip("0").rstrip(".")
    return s.replace("-", "m").replace(".", "p")

# ============================================================
# LOAD DATA
# ============================================================
windows_path = safe_path(WINDOWS_FILE_WIN, WINDOWS_FILE_FALLBACK)
data_path = safe_path(DATA_FILE_WIN, DATA_FILE_FALLBACK)
out_path = safe_output_path(OUTPUT_PATH_WIN, OUTPUT_PATH_FALLBACK)

windows = pd.read_excel(windows_path)
data = pd.read_excel(data_path)

windows[WIN_START_COL] = pd.to_datetime(windows[WIN_START_COL], errors="coerce", dayfirst=True).dt.floor("min")
windows[WIN_END_COL] = pd.to_datetime(windows[WIN_END_COL], errors="coerce", dayfirst=True).dt.floor("min")
data[TIME_COL] = pd.to_datetime(data[TIME_COL], errors="coerce", dayfirst=True).dt.floor("min")

windows[WIN_ACT_COL] = to_float_series(windows[WIN_ACT_COL])

data = data.dropna(subset=[TIME_COL]).sort_values(TIME_COL).set_index(TIME_COL)
data = data[~data.index.duplicated(keep="first")]

# Remove the excluded range from the main 1-minute dataset (inclusive)
data = data.loc[~((data.index >= EXCLUDE_START) & (data.index <= EXCLUDE_END))].copy()

required_main_cols = [P_KW_COL, TAIR_COL, RH_PCT_COL, TPOOL_COL]
required_win_cols = [WIN_START_COL, WIN_END_COL, WIN_ACT_COL]

missing_main = [c for c in required_main_cols if c not in data.columns]
if missing_main:
    raise ValueError(f"Main dataset is missing required columns: {missing_main}\nSeen: {list(data.columns)}")

missing_win = [c for c in required_win_cols if c not in windows.columns]
if missing_win:
    raise ValueError(f"Windows file is missing required columns: {missing_win}\nSeen: {list(windows.columns)}")

# ============================================================
# RUN WINDOW-BY-WINDOW
# ============================================================
dT_model_list = []
dT_meas_list = []
dT_err_list = []
win_hours = []
avg_power_list = []

comp_cols = [
    "Avg Q_evap [kW]", "E_evap [kWh]",
    "Avg Q_conv [kW]", "E_conv [kWh]",
    "Avg Q_cond [kW]", "E_cond [kWh]",
    "Avg P_in [kW]",   "E_in [kWh]",
    "Avg Q_net [kW]",  "E_net [kWh]",
]
comp_lists = {c: [] for c in comp_cols}

skip_reasons = {
    "bad_time": 0,
    "missing_act": 0,
    "excluded_range_overlap": 0,   # NEW
    "too_short_before_clean": 0,
    "too_short_after_clean": 0,
    "nan_model": 0,
}

for _, row in windows.iterrows():
    t0 = row[WIN_START_COL]
    t1 = row[WIN_END_COL]
    act = row[WIN_ACT_COL]

    dur_h = np.nan
    if pd.notna(t0) and pd.notna(t1) and t1 >= t0:
        dur_h = (t1 - t0).total_seconds() / 3600.0
    win_hours.append(dur_h)

    # Skip if this window overlaps the excluded period
    if overlaps_excluded_range(t0, t1):
        skip_reasons["excluded_range_overlap"] += 1
        dT_model_list.append(np.nan)
        dT_meas_list.append(np.nan)
        dT_err_list.append(np.nan)
        avg_power_list.append(np.nan)
        for c in comp_cols:
            comp_lists[c].append(np.nan)
        continue

    if pd.isna(t0) or pd.isna(t1) or t1 < t0:
        skip_reasons["bad_time"] += 1
        dT_model_list.append(np.nan)
        dT_meas_list.append(np.nan)
        dT_err_list.append(np.nan)
        avg_power_list.append(np.nan)
        for c in comp_cols:
            comp_lists[c].append(np.nan)
        continue

    if pd.isna(act):
        skip_reasons["missing_act"] += 1
        dT_model_list.append(np.nan)
        dT_meas_list.append(np.nan)
        dT_err_list.append(np.nan)
        avg_power_list.append(np.nan)
        for c in comp_cols:
            comp_lists[c].append(np.nan)
        continue

    win = data.loc[t0:t1].copy()
    if len(win) < 2:
        skip_reasons["too_short_before_clean"] += 1
        dT_model_list.append(np.nan)
        dT_meas_list.append(np.nan)
        dT_err_list.append(np.nan)
        avg_power_list.append(np.nan)
        for c in comp_cols:
            comp_lists[c].append(np.nan)
        continue

    win[P_KW_COL] = to_float_series(win[P_KW_COL])
    win[TAIR_COL] = to_float_series(win[TAIR_COL])
    win[RH_PCT_COL] = to_float_series(win[RH_PCT_COL])
    win[TPOOL_COL] = to_float_series(win[TPOOL_COL])

    if DROP_NAN_ROWS_IN_WINDOW:
        win = win.dropna(subset=required_main_cols)
    else:
        if INTERPOLATE_IN_WINDOW_IF_NEEDED:
            win[required_main_cols] = win[required_main_cols].interpolate(limit_direction="both")

    if len(win) < 2:
        skip_reasons["too_short_after_clean"] += 1
        dT_model_list.append(np.nan)
        dT_meas_list.append(np.nan)
        dT_err_list.append(np.nan)
        avg_power_list.append(np.nan)
        for c in comp_cols:
            comp_lists[c].append(np.nan)
        continue

    avg_power_list.append(float(win[P_KW_COL].mean()))

    dT_model, comps = integrate_window_dynamic_1min_with_components(win, act)
    dT_meas = float(win[TPOOL_COL].iloc[-1] - win[TPOOL_COL].iloc[0])

    if not np.isfinite(dT_model):
        skip_reasons["nan_model"] += 1
        dT_model_list.append(np.nan)
        dT_meas_list.append(dT_meas)
        dT_err_list.append(np.nan)
        for c in comp_cols:
            comp_lists[c].append(np.nan)
        continue

    dT_model_list.append(dT_model)
    dT_meas_list.append(dT_meas)
    dT_err_list.append(dT_model - dT_meas)

    for c in comp_cols:
        comp_lists[c].append(float(comps.get(c, np.nan)))

windows["duration_hours"] = win_hours
windows["Average thermal power [kW]"] = avg_power_list
windows["Delta T measured (from 1-min) [°C]"] = dT_meas_list
windows["Delta T model [°C]"] = dT_model_list
windows["Delta T error [°C]"] = dT_err_list

for c in comp_cols:
    windows[c] = comp_lists[c]

# ============================================================
# METRICS: overall + per-activity_factor (printed + stored)
# ============================================================
err_all = windows["Delta T error [°C]"].to_numpy(dtype=float)
meas_all = windows["Delta T measured (from 1-min) [°C]"].to_numpy(dtype=float)
dur_all = windows["duration_hours"].to_numpy(dtype=float)

overall = compute_metrics(err_all, meas_all, dur_all)

print("\n================== SUMMARY (OVERALL) ==================")
print(f"Windows processed (valid): {overall['N']} / {len(windows)}")
print(f"MAE   ΔT error: {overall['MAE']:.4f} °C")
print(f"RMSE  ΔT error: {overall['RMSE']:.4f} °C")
print(f"Bias  ΔT error: {overall['Bias']:.4f} °C")
print(f"MedAE |ΔT error|: {overall['MedAE']:.4f} °C")
print(f"IQR   |ΔT error|: {overall['IQR']:.4f} °C  (P25={overall['P25']:.4f}, P75={overall['P75']:.4f})")
print(f"Duration-weighted MAE: {overall['Weighted_MAE']:.4f} °C")
print(f"Median relative error: {overall['MedRelPct']:.2f} %")
print(f"P90 relative error   : {overall['P90RelPct']:.2f} %")
print("Skip reasons:", skip_reasons)
print("=======================================================\n")

windows["MAE_all_windows_[°C]"] = overall["MAE"]
windows["RMSE_all_windows_[°C]"] = overall["RMSE"]
windows["Bias_all_windows_[°C]"] = overall["Bias"]
windows["MedAE_all_windows_[°C]"] = overall["MedAE"]
windows["P25_abs_error_[°C]"] = overall["P25"]
windows["P75_abs_error_[°C]"] = overall["P75"]
windows["IQR_abs_error_[°C]"] = overall["IQR"]
windows["MAE_duration_weighted_[°C]"] = overall["Weighted_MAE"]
windows["Median_relative_error_[%]"] = overall["MedRelPct"]
windows["P90_relative_error_[%]"] = overall["P90RelPct"]

af_vals = windows[WIN_ACT_COL].to_numpy(dtype=float)
unique_af = sorted([v for v in np.unique(af_vals[np.isfinite(af_vals)])])

print("================== SUMMARY (BY ACTIVITY FACTOR) ==================")
per_af_metrics = {}
for af in unique_af:
    m_af = np.isfinite(af_vals) & (af_vals == af)
    m_af = m_af & np.isfinite(err_all)

    err = err_all[m_af]
    meas = meas_all[m_af]
    dur = dur_all[m_af]

    stats = compute_metrics(err, meas, dur)
    per_af_metrics[af] = stats

    print(f"\nActivity factor = {af:g}  |  N = {stats['N']}")
    print(f"  MAE   : {stats['MAE']:.4f} °C")
    print(f"  RMSE  : {stats['RMSE']:.4f} °C")
    print(f"  Bias  : {stats['Bias']:.4f} °C")
    print(f"  MedAE : {stats['MedAE']:.4f} °C")
    print(f"  IQR   : {stats['IQR']:.4f} °C")
    print(f"  W-MAE : {stats['Weighted_MAE']:.4f} °C")
    print(f"  MedRel: {stats['MedRelPct']:.2f} %")
    print(f"  P90Rel: {stats['P90RelPct']:.2f} %")
print("\n==================================================================\n")

for af, stats in per_af_metrics.items():
    tag = af_tag(af)
    windows[f"MAE_af_{tag}_[°C]"] = stats["MAE"]
    windows[f"RMSE_af_{tag}_[°C]"] = stats["RMSE"]
    windows[f"Bias_af_{tag}_[°C]"] = stats["Bias"]
    windows[f"MedAE_af_{tag}_[°C]"] = stats["MedAE"]
    windows[f"IQR_af_{tag}_[°C]"] = stats["IQR"]
    windows[f"MAE_weighted_af_{tag}_[°C]"] = stats["Weighted_MAE"]
    windows[f"MedianRel_af_{tag}_[%]"] = stats["MedRelPct"]
    windows[f"P90Rel_af_{tag}_[%]"] = stats["P90RelPct"]

# ============================================================
# SELECT + ORDER OUTPUT COLUMNS
# ============================================================
base_cols = [
    WIN_START_COL,
    WIN_END_COL,
    WIN_ACT_COL,
    "duration_hours",
    "Average thermal power [kW]",
    "Delta T measured (from 1-min) [°C]",
    "Delta T model [°C]",
    "Delta T error [°C]",
    "MAE_all_windows_[°C]",
    "RMSE_all_windows_[°C]",
    "Bias_all_windows_[°C]",
    "MedAE_all_windows_[°C]",
    "P25_abs_error_[°C]",
    "P75_abs_error_[°C]",
    "IQR_abs_error_[°C]",
    "MAE_duration_weighted_[°C]",
    "Median_relative_error_[%]",
    "P90_relative_error_[%]",
]

per_af_cols = []
for af in unique_af:
    tag = af_tag(af)
    per_af_cols.extend([
        f"MAE_af_{tag}_[°C]",
        f"RMSE_af_{tag}_[°C]",
        f"Bias_af_{tag}_[°C]",
        f"MedAE_af_{tag}_[°C]",
        f"IQR_af_{tag}_[°C]",
        f"MAE_weighted_af_{tag}_[°C]",
        f"MedianRel_af_{tag}_[%]",
        f"P90Rel_af_{tag}_[%]",
    ])

out_cols = base_cols + per_af_cols + comp_cols
windows_out = windows.loc[:, [c for c in out_cols if c in windows.columns]].copy()

# ============================================================
# SAVE OUTPUT EXCEL
# ============================================================
windows_out.to_excel(out_path, index=False)
print(f"\nSaved results to: {out_path}")

# ============================================================
# PLOTS
# ============================================================
if PLOTS_SHOW:
    x = windows_out["Delta T measured (from 1-min) [°C]"].to_numpy(dtype=float)
    y = windows_out["Delta T model [°C]"].to_numpy(dtype=float)
    m = np.isfinite(x) & np.isfinite(y)

    plt.figure(figsize=(7, 6))
    plt.scatter(x[m], y[m])
    if np.any(m):
        mn = min(x[m].min(), y[m].min())
        mx = max(x[m].max(), y[m].max())
        plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel("Measured ΔT (from 1-min) [°C]")
    plt.ylabel("Model ΔT [°C]")
    plt.title("Measured vs Model ΔT (per window)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    err2 = windows_out["Delta T error [°C]"].to_numpy(dtype=float)
    err2 = err2[np.isfinite(err2)]
    plt.figure(figsize=(7, 4))
    plt.hist(err2, bins=20)
    plt.xlabel("ΔT error [°C] (Model - Measured)")
    plt.ylabel("Count")
    plt.title("ΔT Error distribution")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    dur2 = windows_out["duration_hours"].to_numpy(dtype=float)
    e2 = windows_out["Delta T error [°C]"].to_numpy(dtype=float)
    m2 = np.isfinite(dur2) & np.isfinite(e2)
    plt.figure(figsize=(7, 4))
    plt.scatter(dur2[m2], e2[m2])
    plt.axhline(0, linestyle="--")
    plt.xlabel("Window duration [h]")
    plt.ylabel("ΔT error [°C]")
    plt.title("ΔT error vs window duration")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    af = windows_out[WIN_ACT_COL].to_numpy(dtype=float)
    m3 = np.isfinite(af) & np.isfinite(e2)
    rng = np.random.default_rng(0)
    jitter = (rng.random(np.sum(m3)) - 0.5) * 0.04

    plt.figure(figsize=(7, 4))
    plt.scatter(af[m3] + jitter, e2[m3])
    plt.axhline(0, linestyle="--")
    plt.xlabel("Activity factor")
    plt.ylabel("ΔT error [°C]")
    plt.title("ΔT error by activity factor")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    t_start = pd.to_datetime(windows_out[WIN_START_COL], errors="coerce")
    m4 = t_start.notna() & windows_out["Delta T error [°C]"].notna()
    plt.figure(figsize=(10, 4))
    plt.plot(t_start[m4], windows_out.loc[m4, "Delta T error [°C]"], marker="o", linestyle="-")
    plt.axhline(0, linestyle="--")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d-%m\n%H:%M"))
    plt.xlabel("Window start time")
    plt.ylabel("ΔT error [°C]")
    plt.title("ΔT error over time (window start)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # EXTRA PLOT: Delta T error vs duration_hours, split by activity factor (0.3 and 1.0)
    dfp = windows_out.copy()
    dfp["duration_hours"] = pd.to_numeric(dfp["duration_hours"], errors="coerce")
    dfp["Delta T error [°C]"] = pd.to_numeric(dfp["Delta T error [°C]"], errors="coerce")
    dfp[WIN_ACT_COL] = pd.to_numeric(dfp[WIN_ACT_COL], errors="coerce")

    m = (
        np.isfinite(dfp["duration_hours"].to_numpy()) &
        np.isfinite(dfp["Delta T error [°C]"].to_numpy()) &
        np.isfinite(dfp[WIN_ACT_COL].to_numpy())
    )
    dfp = dfp.loc[m].copy()

    tol = 1e-9
    m03 = np.isclose(dfp[WIN_ACT_COL].to_numpy(dtype=float), 0.3, atol=tol, rtol=0)
    m10 = np.isclose(dfp[WIN_ACT_COL].to_numpy(dtype=float), 1.0, atol=tol, rtol=0)

    plt.figure(figsize=(8, 5))
    if np.any(m03):
        plt.scatter(dfp.loc[m03, "duration_hours"], dfp.loc[m03, "Delta T error [°C]"], label="Activity factor = 0.3")
    if np.any(m10):
        plt.scatter(dfp.loc[m10, "duration_hours"], dfp.loc[m10, "Delta T error [°C]"], label="Activity factor = 1.0")
    plt.axhline(0, linestyle="--")
    plt.xlabel("duration_hours [h]")
    plt.ylabel("Delta T error [°C] (Model - Measured)")
    plt.title("Delta T error vs duration (split by activity factor)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ============================================================
    # EXTRA PLOTS (2 figures):
    # For each activity factor (0.3 and 1.0):
    #   x-axis: duration_hours
    #   y-axis: ΔT
    #   Measured: blue line + dots
    #   Model: red line + dots
    # ============================================================
    dfp2 = windows_out.copy()

    # Ensure numeric
    dfp2["duration_hours"] = pd.to_numeric(dfp2["duration_hours"], errors="coerce")
    dfp2["Delta T measured (from 1-min) [°C]"] = pd.to_numeric(
        dfp2["Delta T measured (from 1-min) [°C]"], errors="coerce"
    )
    dfp2["Delta T model [°C]"] = pd.to_numeric(
        dfp2["Delta T model [°C]"], errors="coerce"
    )
    dfp2[WIN_ACT_COL] = pd.to_numeric(dfp2[WIN_ACT_COL], errors="coerce")

    # Keep only valid points
    m = (
            np.isfinite(dfp2["duration_hours"].to_numpy()) &
            np.isfinite(dfp2["Delta T measured (from 1-min) [°C]"].to_numpy()) &
            np.isfinite(dfp2["Delta T model [°C]"].to_numpy()) &
            np.isfinite(dfp2[WIN_ACT_COL].to_numpy())
    )
    dfp2 = dfp2.loc[m].copy()

    # Two activity-factor cases you want
    af_cases = [0.3,0.5, 1.0]
    tol = 1e-9

    for af_case in af_cases:
        m_af = np.isclose(dfp2[WIN_ACT_COL].to_numpy(dtype=float), af_case, atol=tol, rtol=0)
        df_af = dfp2.loc[m_af].copy()

        if df_af.empty:
            print(f"[PLOT] No valid rows found for activity_factor = {af_case}")
            continue

        # Sort by duration so the line connects logically
        df_af = df_af.sort_values("duration_hours")

        plt.figure(figsize=(8, 5))

        # Measured ΔT
        plt.plot(
            df_af["duration_hours"],
            df_af["Delta T measured (from 1-min) [°C]"],
            marker="o",
            linestyle="-",
            label="Measured ΔT",
            color="blue",
        )

        # Model ΔT
        plt.plot(
            df_af["duration_hours"],
            df_af["Delta T model [°C]"],
            marker="o",
            linestyle="-",
            label="Model ΔT",
            color="red",
        )

        plt.axhline(0, linestyle="--")
        plt.xlabel("duration_hours [h]")
        plt.ylabel("ΔT [°C]")
        plt.title(f"ΔT vs window duration (activity_factor = {af_case:g})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
