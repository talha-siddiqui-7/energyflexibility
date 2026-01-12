#!/usr/bin/env python3
"""
Daily setpoint = 60% analysis (manual schedule)
- Uses exact time windows from schedule CSV
  (columns: window_start, window_end)
- Excludes short event windows
- Runs model for each window twice: External & AHU modes
- Computes MAE/RMSE in:
    • kWh (latent energy)
    • g/kg (humidity ratio)
- Computes mean evaporation rate (kg/s) for:
    • Model trajectory
    • "Measured" trajectory (using same evap formula on W_meas)
- Saves per-window results to daily_setpoint_manual_results.csv

Infiltration humidity:
    w_inf(t) = 0.5 * w_dry + 0.5 * w_wet(t)
    where
        w_dry      = humidity ratio of DRY adjacent rooms
                     (constant RH, constant T, user-defined)
        w_wet(t)   = humidity ratio of MOIST rooms ~ showers
                     (same RH as hall extract; T = T_ext_room or user-defined)

Additional:
- Latent storage per buildup window (model + measured)
- Daily and total latent storage (per mode)
- Baseline latent COP using:
    • external sensor measurements (w_ext_room, w_sup_room, T_ext_room, m_ext_dry)
    • AHU Average W
    • baseline = all times inside experiment dataset but outside DR buildup windows
      and the short event windows
- Includes simple unit sanity checks/conversions for m_dot and AHU power
"""

import pandas as pd
import numpy as np
import math

# ============================================================
# --------------------- USER SETTINGS ------------------------
# ============================================================

DATA_CSV_PATH     = r"M:\PhD\03 Experiments\17-09-2025_16-10-2025_sensor_vs_AHU_data.csv"
# Schedule file: columns: window_start, window_end, base_setpoint_before_increase,
#                           raised_setpoint, delta_setpoint
SCHEDULE_CSV_PATH = r"C:\Users\mtasi\Downloads\RH_setpoint_increase_windows_daily_gt10.csv"

RESULTS_CSV               = r"M:\PhD\03 Experiments\daily_setpoint_manual_results.csv"
RESULTS_DAILY_LATENT_CSV  = r"M:\PhD\03 Experiments\daily_latent_storage.csv"
RESULTS_TOTAL_LATENT_CSV  = r"M:\PhD\03 Experiments\total_latent_storage.csv"
RESULTS_DAILY_COP_CSV     = r"M:\PhD\03 Experiments\daily_baseline_COP_external.csv"

INFILTRATION_ON     = True       # master toggle for infiltration in ODE
INFIL_SMOOTH_MIN    = 20         # only used for diagnostic smoothed series

USE_BALANCED_VENT   = False      # False = explicit supply/extract + imbalance term

# Infiltration assumptions for adjacent rooms
# Dry rooms (corridor, gym, guard room, etc.)
INFIL_DRY_RH   = 0.37    # DRY rooms RH as fraction (0–1), e.g. 0.45 = 45%
INFIL_DRY_T    = 28.0    # °C, DRY rooms temperature

# Moist rooms (showers/WC)
# If INFIL_T_WET is None → use actual hall extract temperature
# Else → recompute W from RH_ext_room at fixed INFIL_T_WET
INFIL_T_WET    = None

# Geometry / constants
VOLUME_M3    = 19.65 * 13.8 * 4.95
PRESSURE_PA  = 101325.0
POOL_AREA_M2 = 100.0
C_EVAP_BASE  = 4.0e-8   # kg/(s·m²·Pa)

# Activity factors
ACTIVITY_FACTOR_DEFAULTS = {"cover_on": 0.30, "cover_off": 0.5}
AF_GLOBAL_SCALE = 1.0

# Short event windows to exclude from analysis
EVENT_DEFINITIONS = [
    {"start_ts": "2025-09-17 22:31", "end_ts": "2025-09-17 23:01"},
    {"start_ts": "2025-09-18 00:20", "end_ts": "2025-09-18 00:41"},
    {"start_ts": "2025-09-18 00:47", "end_ts": "2025-09-18 01:08"},
    {"start_ts": "2025-10-16 12:45", "end_ts": "2025-10-16 13:21"},
    {"start_ts": "2025-10-16 13:40", "end_ts": "2025-10-16 14:04"},
    {"start_ts": "2025-09-17 23:59", "end_ts": "2025-09-18 00:13"},
]

# ============================================================
# ------------------- PSYCHROMETRICS -------------------------
# ============================================================

def psat_pa(T_C):
    return 610.94 * math.exp(17.625 * T_C / (T_C + 243.04))

def Pv_from_W(W, p=PRESSURE_PA):
    return p * (W / (0.622 + W))

def W_from_RH_T(RH, T, p=PRESSURE_PA):
    Ps = psat_pa(T)
    Pv = RH * Ps
    return 0.622 * Pv / (p - Pv)

def rho_da_from_wT(w, T, p=PRESSURE_PA):
    pv = p * (w / (0.622 + w))
    p_da = p - pv
    return p_da / (287.05 * (T + 273.15))

def h_fg_J_per_kg(T):
    return 2.501e6 - 2361.0 * T

# ============================================================
# ---------------- DATA LOADING ------------------------------
# ============================================================

def _parse_datetime(df):
    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"], errors="coerce", dayfirst=True)
    else:
        dt = pd.to_datetime(
            df["date"].astype(str).str.strip() + " " + df["time"].astype(str).str.strip(),
            errors="coerce",
            dayfirst=True,
        )
    return dt.dt.round("min")

def dec_col(df, col):
    s = df.get(col, None)
    if s is None:
        return pd.Series(np.nan, index=df.index)
    s = s.astype(str).str.replace("'", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")

def load_experiment_csv(path):
    df_raw = pd.read_csv(path)
    time = _parse_datetime(df_raw)

    df = pd.DataFrame({"time": time})
    # Primary signals (external & AHU)
    df["T_ext_room"]   = dec_col(df_raw, "Extract_air_Temp_sensor")
    df["RH_ext_room"]  = dec_col(df_raw, "Extract_air_RH_sensor")
    df["T_ext_AHU"]    = dec_col(df_raw, "Extract_air_Temp_AHU")
    df["RH_ext_AHU"]   = dec_col(df_raw, "Extract_air_RH_AHU")
    df["T_sup"]        = dec_col(df_raw, "Supply_air_temp_sensor")
    df["RH_sup_room"]  = dec_col(df_raw, "Supply_air_RH_sensor")
    df["RH_sup_AHU"]   = dec_col(df_raw, "Supply_air_RH_AHU")
    df["T_pool"]       = dec_col(df_raw, "Pool_OF")

    # Dry-air mass flows + infiltration dry-air (as in file)
    df["m_sup_dry"]    = dec_col(df_raw, "mdot_sup_dry")
    df["m_ext_dry"]    = dec_col(df_raw, "mdot_ext_dry")
    df["m_inf_dry"]    = dec_col(df_raw, "inf_dry_kg_s")

    # AHU electric power
    df["P_ahu_raw"]    = dec_col(df_raw, "AHU Average W")

    # ---- Unit sanity checks ----
    # 1) m_dot likely kg/s; if it looks like kg/h (very large), divide by 3600
    for col in ["m_sup_dry", "m_ext_dry", "m_inf_dry"]:
        if col in df.columns:
            med = df[col].median(skipna=True)
            if np.isfinite(med) and med > 50:     # >50 kg/s would be crazy for an AHU
                print(f"[Unit fix] {col} median {med:.2f} → assuming kg/h, converting to kg/s.")
                df[col] = df[col] / 3600.0
    # 2) AHU power: if median between ~1 and ~50, probably kW, convert to W
    P_med = df["P_ahu_raw"].median(skipna=True)
    if np.isfinite(P_med) and 1.0 <= P_med <= 50.0:
        print(f"[Unit fix] AHU Average W median {P_med:.2f} → assuming kW, converting to W.")
        df["P_ahu_W"] = df["P_ahu_raw"] * 1000.0
    else:
        df["P_ahu_W"] = df["P_ahu_raw"]

    # Replace NaNs in infiltration with 0
    df["m_inf_dry"] = df["m_inf_dry"].fillna(0.0)

    # Humidity ratios from RH/T (external sensors preferred)
    df["w_ext_room"] = [W_from_RH_T(r/100.0, T) for r, T in zip(df["RH_ext_room"], df["T_ext_room"])]
    df["w_ext_ahu"]  = [W_from_RH_T(r/100.0, T) for r, T in zip(df["RH_ext_AHU"],  df["T_ext_AHU"])]
    df["w_sup_room"] = [W_from_RH_T(r/100.0, T) for r, T in zip(df["RH_sup_room"], df["T_sup"])]
    df["w_sup_ahu"]  = [W_from_RH_T(r/100.0, T) for r, T in zip(df["RH_sup_AHU"],  df["T_sup"])]

    # NEW: infiltration humidity components for adjacent rooms

    # 1) DRY building node (corridor/gym/guard room) – constant in time
    w_dry_const = W_from_RH_T(INFIL_DRY_RH, INFIL_DRY_T)
    df["w_inf_dry"] = w_dry_const  # same value for all timesteps

    # 2) WET node (showers/WC) – uses hall extract RH
    if INFIL_T_WET is None:
        # Use existing hall extract humidity ratio as "wet" component
        df["w_inf_wet"] = df["w_ext_room"]
    else:
        # Recompute with a fixed wet-room temperature
        df["w_inf_wet"] = [
            W_from_RH_T(r/100.0, INFIL_T_WET) for r in df["RH_ext_room"]
        ]

    # (Smoothed extract kept only for diagnostics / plots if you still want it)
    df = df.set_index("time")
    roll = f"{INFIL_SMOOTH_MIN}min"
    df["w_ext_room_smooth"] = df["w_ext_room"].rolling(roll, min_periods=1).mean()
    df = df.reset_index()

    print("\n[Info] m_ext_dry stats (kg/s):")
    print(df["m_ext_dry"].describe())

    print("\n[Info] P_ahu_W stats (W):")
    print(df["P_ahu_W"].describe())

    return df

# ============================================================
# ---------------- SIMULATION CORE ---------------------------
# ============================================================

def _latent_Q_kWh(W_series, W0, m_air, T_series):
    T_mean = float(np.nanmean(T_series))
    hfg = h_fg_J_per_kg(T_mean)
    return hfg * m_air * (W_series - W0) / 3.6e6

def simulate_window(win, RH_target, activity_factor, mode):
    """
    Simulate humidity ratio W(t) over a window.
    Also returns window-mean evaporation rates (kg/s) for model and "measured".
    """
    tair   = win["T_ext_room"].values
    Twater = win["T_pool"].values

    # Infiltration humidity ratio time series (50% dry + 50% wet)
    if INFILTRATION_ON:
        w_inf_dry = win["w_inf_dry"].values
        w_inf_wet = win["w_inf_wet"].values
        w_inf = 0.5 * w_inf_dry + 0.5 * w_inf_wet
    else:
        w_inf = np.zeros(len(win))

    # Dry-air mass flows
    ms = win["m_sup_dry"].values.astype(float)
    me = win["m_ext_dry"].values.astype(float)
    mi = win["m_inf_dry"].values.astype(float) if INFILTRATION_ON else np.zeros(len(win))

    # Initial / supply / measured humidity ratios
    if mode == "external":
        W_init = win["w_ext_room"].values
        w_sup  = win["w_sup_room"].values
        W_meas = win["w_ext_room"].values
    else:
        W_init = win["w_ext_ahu"].values
        w_sup  = win["w_sup_ahu"].values
        W_meas = win["w_ext_ahu"].values

    if len(W_init) == 0 or not np.isfinite(W_init[0]) or not np.isfinite(tair[0]):
        return {"ok": False}

    W0   = float(W_init[0])
    m_air= rho_da_from_wT(W0, tair[0]) * VOLUME_M3

    # Time vector
    tsec = (win["time"] - win["time"].iloc[0]).dt.total_seconds().values
    dt   = np.diff(tsec, prepend=tsec[0])

    # Saturation vapour pressure over water → pool driving pressure
    Pw = np.array([psat_pa(Tw) for Tw in Twater])

    # Time stepping
    W = float(W0)
    W_series = [W0]

    for i in range(1, len(tsec)):
        dti = float(dt[i])
        if dti <= 0:
            W_series.append(W)
            continue

        # evaporation part
        m_evap = (C_EVAP_BASE * activity_factor) * POOL_AREA_M2 * (Pw[i] - Pv_from_W(W))

        # ventilation + infiltration + imbalance
        rhs = m_evap
        rhs += ms[i] * (w_sup[i] - W)
        rhs += mi[i] * (w_inf[i] - W)

        if not USE_BALANCED_VENT:
            rhs -= (me[i] - ms[i] - mi[i]) * W

        dWdt = rhs / m_air
        W += dWdt * min(dti, 60.0)  # cap substep to 60 s for stability

        # clamp to 0..W_at_setpoint
        W = max(0.0, min(W, W_from_RH_T(RH_target, tair[i])))

        W_series.append(W)

    W_series = np.array(W_series)
    W_meas   = np.array(W_meas)

    # Evaporation rate time series (kg/s)
    Pv_model = np.array([Pv_from_W(w) for w in W_series])
    Pv_meas  = np.array([Pv_from_W(w) for w in W_meas])

    m_evap_model = (C_EVAP_BASE * activity_factor) * POOL_AREA_M2 * (Pw - Pv_model)
    m_evap_meas  = (C_EVAP_BASE * activity_factor) * POOL_AREA_M2 * (Pw - Pv_meas)

    m_evap_model_mean = float(np.nanmean(m_evap_model))
    m_evap_meas_mean  = float(np.nanmean(m_evap_meas))

    return {
        "ok": True,
        "W_series": W_series,
        "W_meas":   W_meas,
        "m_air":    m_air,
        "W0":       W0,
        "tair":     tair,
        "m_evap_model_mean": m_evap_model_mean,  # kg/s
        "m_evap_meas_mean":  m_evap_meas_mean   # kg/s
    }

# ============================================================
# -------------------- MAIN PROCESS --------------------------
# ============================================================

def overlaps(a, b):
    return not (a[1] <= b[0] or a[0] >= b[1])

def main():
    df_all = load_experiment_csv(DATA_CSV_PATH)
    print(f"\nData range: {df_all['time'].min()} → {df_all['time'].max()}")
    print(f"Infiltration ON: {INFILTRATION_ON} (0.5 dry-node + 0.5 wet-node)")

    # --------------------------------------------------------
    # Load schedule (buildup windows)
    # --------------------------------------------------------
    sched = pd.read_csv(SCHEDULE_CSV_PATH)
    sched["window_start"] = pd.to_datetime(sched["window_start"], errors="coerce")
    sched["window_end"]   = pd.to_datetime(sched["window_end"],   errors="coerce")
    sched = sched.dropna(subset=["window_start", "window_end"]).sort_values("window_start")

    # Short event windows
    exp_wins = [(pd.to_datetime(e["start_ts"]), pd.to_datetime(e["end_ts"])) for e in EVENT_DEFINITIONS]

    windows = []
    for _, row in sched.iterrows():
        t0, t1 = row["window_start"], row["window_end"]
        if any(overlaps((t0, t1), ex) for ex in exp_wins):
            continue
        windows.append((t0, t1))

    print(f"Evaluating {len(windows)} setpoint-increase windows (after exclusion).")

    # --------------------------------------------------------
    # Per-window simulations and latent storage
    # --------------------------------------------------------
    rows = []
    for i, (t0, t1) in enumerate(windows, start=1):
        win = df_all[(df_all["time"] >= t0) & (df_all["time"] <= t1)].copy()
        if len(win) < 2:
            continue

        AF = ACTIVITY_FACTOR_DEFAULTS["cover_on"] * AF_GLOBAL_SCALE
        RH_target = 0.60  # 60%

        for mode in ("external", "ahu"):
            sim = simulate_window(win, RH_target, AF, mode)
            if not sim["ok"]:
                continue

            Wm = sim["W_meas"]
            Ws = sim["W_series"]

            # --- Humidity ratio error (g/kg) ---
            errW = (Ws - Wm) * 1000.0
            maeW = float(np.nanmean(np.abs(errW)))
            rmseW= float(np.sqrt(np.nanmean(errW**2)))
            dWf  = float(errW[-1])

            # --- Latent energy trajectories (kWh) ---
            Q_model = _latent_Q_kWh(Ws, sim["W0"], sim["m_air"], sim["tair"])
            Q_meas  = _latent_Q_kWh(Wm, sim["W0"], sim["m_air"], sim["tair"])

            # Latent storage at end of buildup window
            Q_model_final = float(Q_model[-1])
            Q_meas_final  = float(Q_meas[-1])

            # --- Latent energy error (kWh) ---
            errQ = Q_model - Q_meas
            maeQ = float(np.nanmean(np.abs(errQ)))
            rmseQ= float(np.sqrt(np.nanmean(errQ**2)))
            dQf  = float(errQ[-1])

            # --- Evaporation (kg/s, window-mean) ---
            m_evap_model_mean = sim["m_evap_model_mean"]
            m_evap_meas_mean  = sim["m_evap_meas_mean"]

            rows.append(dict(
                window_id=i,
                start_ts=str(t0),
                end_ts=str(t1),
                duration_h=round((t1 - t0).total_seconds()/3600.0, 3),
                mode=mode.upper(),
                MAE_W_gpkg=round(maeW, 3),
                RMSE_W_gpkg=round(rmseW, 3),
                dW_final_gpkg=round(dWf, 3),
                MAE_kWh=round(maeQ, 3),
                RMSE_kWh=round(rmseQ, 3),
                dQ_final_kWh=round(dQf, 3),
                Q_model_final_kWh=round(Q_model_final, 3),
                Q_meas_final_kWh=round(Q_meas_final, 3),
                m_evap_model_mean_kg_s=round(m_evap_model_mean, 6),
                m_evap_meas_mean_kg_s=round(m_evap_meas_mean, 6),
                infil_on=INFILTRATION_ON
            ))

    df_res = pd.DataFrame(rows)
    df_res.to_csv(RESULTS_CSV, index=False)
    print(f"\nResults saved → {RESULTS_CSV}")
    if not df_res.empty:
        print(df_res.head(10).to_string(index=False))
    else:
        print("No rows to export from buildup windows.")

    # --------------------------------------------------------
    # Daily and total latent storage (model & measured)
    # --------------------------------------------------------
    if not df_res.empty:
        df_res["start_ts"] = pd.to_datetime(df_res["start_ts"])
        df_res["day"] = df_res["start_ts"].dt.date

        daily_latent = (
            df_res
            .groupby(["day", "mode"], as_index=False)[["Q_model_final_kWh", "Q_meas_final_kWh"]]
            .sum()
            .rename(columns={
                "Q_model_final_kWh": "Q_model_daily_kWh",
                "Q_meas_final_kWh": "Q_meas_daily_kWh"
            })
        )

        total_latent = (
            df_res
            .groupby("mode", as_index=False)[["Q_model_final_kWh", "Q_meas_final_kWh"]]
            .sum()
            .rename(columns={
                "Q_model_final_kWh": "Q_model_total_kWh",
                "Q_meas_final_kWh": "Q_meas_total_kWh"
            })
        )

        daily_latent.to_csv(RESULTS_DAILY_LATENT_CSV, index=False)
        total_latent.to_csv(RESULTS_TOTAL_LATENT_CSV, index=False)

        print("\nDaily latent storage (kWh) per mode:")
        print(daily_latent.to_string(index=False))

        print("\nTotal latent storage over full period (kWh) per mode:")
        print(total_latent.to_string(index=False))

    # --------------------------------------------------------
    # Baseline latent COP using external measurements only
    # (unchanged – still uses w_ext_room, w_sup_room, m_ext_dry, P_ahu_W)
    # --------------------------------------------------------
    print("\nEstimating baseline latent COP using external sensors (experiment dataset only)...")

    baseline_mask = pd.Series(True, index=df_all.index)

    # Remove DR buildup windows
    for (t0, t1) in windows:
        baseline_mask &= ~((df_all["time"] >= t0) & (df_all["time"] <= t1))

    # Remove short event windows
    for (t0, t1) in exp_wins:
        baseline_mask &= ~((df_all["time"] >= t0) & (df_all["time"] <= t1))

    df_base = df_all[baseline_mask].copy()

    # Drop rows with missing key variables
    df_base = df_base[
        df_base[["T_ext_room", "m_ext_dry", "w_ext_room", "w_sup_room", "P_ahu_W"]].notna().all(axis=1)
    ]

    if df_base.empty:
        print("Baseline subset is empty after masking; cannot compute COP.")
    else:
        df_base = df_base.sort_values("time")
        dt_hours = df_base["time"].diff().dt.total_seconds() / 3600.0
        dt_hours = dt_hours.fillna(0.0)

        # Latent removal power (W) using external & supply room sensors
        T_air   = df_base["T_ext_room"].values
        hfg_vec = np.array([h_fg_J_per_kg(T) for T in T_air])

        m_ext_dry = df_base["m_ext_dry"].values
        w_ext     = df_base["w_ext_room"].values
        w_sup     = df_base["w_sup_room"].values

        Qdot_lat_W = m_ext_dry * (w_ext - w_sup) * hfg_vec
        Qdot_lat_W_pos = np.maximum(Qdot_lat_W, 0.0)

        # Integrate to kWh
        dq_lat_kWh = (Qdot_lat_W_pos / 1000.0) * dt_hours.values
        Q_lat_kWh  = float(dq_lat_kWh.sum())

        P_ahu_kW   = df_base["P_ahu_W"].values / 1000.0
        de_el_kWh  = P_ahu_kW * dt_hours.values
        E_ahu_kWh  = float(de_el_kWh.sum())

        # Per-day baseline COP → CSV
        df_base["day"] = df_base["time"].dt.date
        df_daily = pd.DataFrame({
            "day": df_base["day"].values,
            "dQ_lat_kWh": dq_lat_kWh,
            "dE_ahu_kWh": de_el_kWh
        })

        daily_cop = (
            df_daily
            .groupby("day", as_index=False)[["dQ_lat_kWh", "dE_ahu_kWh"]]
            .sum()
        )

        daily_cop["COP_baseline"] = daily_cop.apply(
            lambda r: r["dQ_lat_kWh"] / r["dE_ahu_kWh"] if r["dE_ahu_kWh"] > 0.1 else np.nan,
            axis=1
        )

        daily_cop.rename(columns={
            "dQ_lat_kWh": "Q_lat_baseline_kWh",
            "dE_ahu_kWh": "E_ahu_baseline_kWh"
        }, inplace=True)

        daily_cop.to_csv(RESULTS_DAILY_COP_CSV, index=False)
        print(f"\nDaily baseline COP saved → {RESULTS_DAILY_COP_CSV}")
        print(daily_cop.to_string(index=False))

        # Overall baseline COP
        if E_ahu_kWh > 0:
            COP_baseline_overall = Q_lat_kWh / E_ahu_kWh
            print("\n---- Baseline COP (external sensors, experiment dataset) ----")
            print(f"Total latent removal (baseline periods): {Q_lat_kWh:.2f} kWh_th")
            print(f"Total AHU electric energy (baseline):   {E_ahu_kWh:.2f} kWh_el")
            print(f"Overall baseline latent COP =           {COP_baseline_overall:.2f}")
        else:
            print("AHU baseline electric energy is zero or invalid; cannot compute COP.")

if __name__ == "__main__":
    main()
