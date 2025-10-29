#!/usr/bin/env python3
"""
Experimental validation of air flexibility model
------------------------------------------------

Workflow:
1. Load measured dataset from the pool hall.
2. For each experimental RH relaxation event
   (setpoint raised -> setpoint lowered), simulate RH rise using the model.
3. Compare model vs measured RH(t) and estimate latent kWh stored.
4. Report:
   - per-event summary
   - RH plot per event (Measured vs Model)
   - global MAE and RMSE for latent storage across all events.

Tuning knobs:
- ACTIVITY_FACTOR_DEFAULTS["cover_on"/"cover_off"]
- EVENT_DEFINITIONS (timestamps, RH targets)
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# ============================================================
# --------------------- USER SETTINGS ------------------------
# ============================================================

DATA_CSV_PATH = r"M:\PhD\03 Experiments\Complete_17-09-2025_16-10-2025_with_AHU_power.csv"
# If running locally, change to your local path.

# Geometry / physics
VOLUME_M3       = 19.65 * 13.8 * 4.95     # Pool hall volume [m³]
PRESSURE_PA     = 101325.0                # Ambient pressure [Pa]
POOL_AREA_M2    = 100.0                   # Pool water surface area [m²]

# Base evaporation constant [kg/(s·m²·Pa)]
C_EVAP_BASE = 4.0e-8

# Activity factors (surface disturbance / cover state)
ACTIVITY_FACTOR_DEFAULTS = {
    "cover_on": 0.5,
    "cover_off": 1.0,
}

# Experimental windows (from your experiment notes)
EVENT_DEFINITIONS = [
    # Night 17→18 Sept 2025
    {
        "start_ts": "2025-09-17 22:31",
        "end_ts":   "2025-09-17 23:24",
        "cover":    "cover_on",
        "RH_target_pct": 65.0,
    },
    {
        "start_ts": "2025-09-17 23:46",
        "end_ts":   "2025-09-18 00:13",
        "cover":    "cover_on",  # pool cover still on
        "RH_target_pct": 70.0,
    },
    {
        "start_ts": "2025-09-18 00:20",
        "end_ts":   "2025-09-18 00:41",
        "cover":    "cover_off",  # cover off
        "RH_target_pct": 70.0,
    },
    {
        "start_ts": "2025-09-18 00:47",
        "end_ts":   "2025-09-18 01:08",
        "cover":    "cover_off",
        "RH_target_pct": 70.0,
    },

    # Daytime 16 Oct 2025
    {
        "start_ts": "2025-10-16 12:45",
        "end_ts":   "2025-10-16 13:21",
        "cover":    "cover_on",
        "RH_target_pct": 65.0,
    },
    {
        "start_ts": "2025-10-16 13:40",
        "end_ts":   "2025-10-16 14:04",
        "cover":    "cover_on",
        "RH_target_pct": 60.0,
    },
]

PLOTS_SHOW   = True
MAX_SUBSTEP_S = 60.0
CLAMP_ENABLED = True

# ============================================================
# ----------- PSYCHROMETRIC / PHYSICAL HELPERS --------------
# ============================================================

def psat_pa(T_C: float) -> float:
    """Saturation vapor pressure [Pa] at air/water temp T_C [°C]."""
    T = float(T_C)
    return 610.94 * math.exp(17.625 * T / (T + 243.04))

def Pv_from_W(W: float, p_atm: float = PRESSURE_PA) -> float:
    """Water vapor partial pressure [Pa] from humidity ratio W [kg/kg]."""
    return p_atm * (W / (0.622 + W))

def W_from_RH_T(RH_frac: float, T_C: float, p_atm: float = PRESSURE_PA) -> float:
    """Convert RH (0-1) and T_C [°C] to humidity ratio W [kg/kg]."""
    Ps = psat_pa(T_C)
    Pv = RH_frac * Ps
    return 0.622 * Pv / (p_atm - Pv)

def RH_from_W_T(W: float, T_C: float, p_atm: float = PRESSURE_PA) -> float:
    """Convert humidity ratio W [kg/kg] and T_C [°C] to RH (0-1)."""
    return Pv_from_W(W, p_atm) / psat_pa(T_C)

def rho_da_from_wT(w: float, T_C: float, p: float = PRESSURE_PA) -> float:
    """
    Dry air density [kg/m³] for given humidity ratio w [kg/kg] and temp T_C [°C].
    Ideal gas split into dry air + vapor.
    """
    w = float(w)
    T_C = float(T_C)
    pv   = p * (w / (0.62198 + w))
    p_da = p - pv
    return p_da / (287.05 * (T_C + 273.15))

def h_fg_J_per_kg(T_C: float) -> float:
    """Latent heat of vaporization [J/kg] near T_C [°C]."""
    return 2.501e6 - 2361.0 * float(T_C)

def W_sat_from_T(T_C: float, p_atm: float = PRESSURE_PA) -> float:
    """Saturation humidity ratio [kg/kg] at T_C [°C]."""
    Ps = psat_pa(T_C)
    Ps = min(Ps, 0.99 * p_atm)  # safety to avoid division blowup
    return 0.622 * Ps / (p_atm - Ps)

def clamp_W(W: float, T_C: float) -> float:
    """Clamp to physical range: 0 ≤ W ≤ W_sat(T)."""
    if not CLAMP_ENABLED:
        return W
    return max(0.0, min(W, W_sat_from_T(T_C)))

# ============================================================
# --------------- DATA LOADER / PREPROCESS ------------------
# ============================================================

def load_experiment_csv(path: str) -> pd.DataFrame:
    """
    Loads the dataset and returns a cleaned dataframe with:
        time [datetime64]
        T_room_C [°C]       (hall/extract air temp)
        RH_room_pct [%]     (hall/extract air RH)
        T_sup_C [°C]        (supply air temp)
        RH_sup_pct [%]      (supply air RH)
        Pool_water_C [°C]   (pool water temp)
        m_sup [kg/s dry air]
        m_ret [kg/s dry air]

    Steps:
    - Parse datetime with dayfirst=True.
    - Fix comma decimals in flow columns ("10131,0" -> 10131.0).
    - Convert m³/h -> kg/s using density from local (T_room, w_ret).
    - Interpolate small gaps using time interpolation.
    """

    df_raw = pd.read_csv(path)

    # 1. Parse timestamp column (we assume 'datetime' exists)
    if "datetime" in df_raw.columns:
        time_col = "datetime"
    else:
        time_col = df_raw.columns[0]

    df = pd.DataFrame()
    df["time"] = pd.to_datetime(df_raw[time_col], errors="coerce", dayfirst=True)

    # 2. Clean numeric columns
    # direct numeric parse for temps / RH
    df["T_room_C"]    = pd.to_numeric(df_raw["Extract_air_Temp"], errors="coerce")
    df["RH_room_pct"] = pd.to_numeric(df_raw["Extract_air_RH"],   errors="coerce")
    df["T_sup_C"]     = pd.to_numeric(df_raw["Supply_air_temp"],  errors="coerce")
    df["RH_sup_pct"]  = pd.to_numeric(df_raw["Supply_air_RH"],    errors="coerce")

    # pool water temp (assumed °C)
    df["Pool_water_C"] = pd.to_numeric(df_raw["Pool_OF"], errors="coerce")

    # flow columns with comma decimals (m³/h)
    def parse_comma_decimal(series):
        return (
            series.astype(str)
                  .str.replace(",", ".", regex=False)
                  .astype(float)
        )

    sup_flow_m3ph = parse_comma_decimal(df_raw["Supply air flow rate (cb.m/hr)"])
    ret_flow_m3ph = parse_comma_decimal(df_raw["Extract air flow rate (cb.m/hr)"])

    # 3. Compute humidity ratio in hall air so we know density
    # (based on hall RH and hall T)
    w_ret_initial = np.array([
        W_from_RH_T(rh/100.0 if pd.notna(rh) else np.nan,
                    T if pd.notna(T) else np.nan)
        if (pd.notna(rh) and pd.notna(T)) else np.nan
        for rh, T in zip(df["RH_room_pct"], df["T_room_C"])
    ], dtype=float)

    rho_air = np.array([
        rho_da_from_wT(
            w if np.isfinite(w) else 0.01,
            T if np.isfinite(T) else 25.0,
            PRESSURE_PA
        )
        for w, T in zip(w_ret_initial, df["T_room_C"])
    ], dtype=float)

    # 4. Convert volumetric flow (m³/h) -> mass flow (kg/s)
    sup_flow_m3ps = sup_flow_m3ph / 3600.0
    ret_flow_m3ps = ret_flow_m3ph / 3600.0

    df["m_sup"] = sup_flow_m3ps * rho_air
    df["m_ret"] = ret_flow_m3ps * rho_air

    # 5. Sort by time, interpolate gaps on a time index
    df = df.sort_values("time").reset_index(drop=True)
    df = df.set_index("time")

    cols_to_interp = [
        "T_room_C", "RH_room_pct",
        "T_sup_C", "RH_sup_pct",
        "Pool_water_C",
        "m_sup", "m_ret"
    ]

    df[cols_to_interp] = (
        df[cols_to_interp]
        .interpolate(method="time", axis=0)
        .ffill()
        .bfill()
    )

    # bring time back
    df = df.reset_index()

    return df

# ============================================================
# ------------- CORE SIMULATION OF ONE WINDOW ----------------
# ============================================================

def simulate_window(win: pd.DataFrame,
                    RH_target_frac: float,
                    activity_factor: float,
                    max_substep_s: float = MAX_SUBSTEP_S):
    """
    Simulate hall humidity ratio W(t) in the event window.

    win columns required:
        time, T_room_C, RH_room_pct, T_sup_C, RH_sup_pct,
        Pool_water_C, m_sup, m_ret

    RH_target_frac: raised RH setpoint cap (0-1)
    activity_factor: multiplier for evaporation (cover_on/cover_off)
    """

    tair         = win["T_room_C"].values               # [°C]
    RH_meas_frac = (win["RH_room_pct"].values / 100.0).clip(0, 1)
    T_sup        = win["T_sup_C"].values                # [°C]
    RH_sup_frac  = (win["RH_sup_pct"].values / 100.0).clip(0, 1)
    Twater       = win["Pool_water_C"].values           # [°C]
    m_sup_arr    = win["m_sup"].values                  # [kg/s dry air]
    m_ret_arr    = win["m_ret"].values                  # [kg/s dry air]

    # Supply humidity ratio
    w_sup_arr = np.array([
        W_from_RH_T(rh, Ts)
        for rh, Ts in zip(RH_sup_frac, T_sup)
    ], dtype=float)

    # Effective ventilation mass flow [kg/s dry air]
    # match previous model convention: average of supply and return
    m_vent_arr = 0.5 * (m_sup_arr + m_ret_arr)

    # Initial humidity ratio in hall from first measurement
    W0 = W_from_RH_T(RH_meas_frac[0], tair[0])

    # Air mass in the room [kg dry air]
    rho_air0 = rho_da_from_wT(W0, tair[0], PRESSURE_PA)
    m_air = rho_air0 * VOLUME_M3

    # Time information
    tsec = (win["time"] - win["time"].iloc[0]).dt.total_seconds().values
    dt_iv = np.diff(tsec, prepend=tsec[0])

    # Pool water saturation vapor pressure at each timestep
    Pw_arr = np.array([psat_pa(Tw) for Tw in Twater], dtype=float)

    W_series = [float(W0)]
    W = float(W0)

    for i in range(1, len(tsec)):
        Δt = float(dt_iv[i])
        if Δt <= 0:
            W_series.append(W)
            continue

        # local terms
        dPa_dW = PRESSURE_PA * 0.622 / (0.622 + W)**2
        k_e_loc = (C_EVAP_BASE * activity_factor) * POOL_AREA_M2 * dPa_dW

        denom = (m_vent_arr[i] + k_e_loc)
        if denom <= 1e-12:
            tau_loc = np.inf
        else:
            tau_loc = m_air / denom

        # pick substep for numerical stability in nonlinear evaporation
        dt_target = min(
            max_substep_s,
            (tau_loc / 5.0) if np.isfinite(tau_loc) else max_substep_s,
            Δt
        )
        n_sub = max(1, int(math.ceil(Δt / dt_target)))
        dt_sub = Δt / n_sub

        for _ in range(n_sub):
            Pa    = Pv_from_W(W, PRESSURE_PA)
            Pw    = Pw_arr[i]
            # evaporation source (kg/s water vapor)
            m_ev  = (C_EVAP_BASE * activity_factor) * POOL_AREA_M2 * (Pw - Pa)

            # ventilation moisture exchange
            m_sink = m_vent_arr[i] * (w_sup_arr[i] - W)

            dWdt = (m_ev + m_sink) / m_air
            W    = W + dWdt * dt_sub

            # enforce RH target cap
            W_cap = W_from_RH_T(RH_target_frac, tair[i], PRESSURE_PA)
            if W > W_cap:
                W = W_cap

            # clamp physically
            if CLAMP_ENABLED:
                W = clamp_W(W, tair[i])

        W_series.append(W)

    return {
        "tsec": np.array(tsec),
        "W_series": np.array(W_series),
        "m_air": m_air,
        "W0": W0,
        "tair": tair,
        "RH_meas_frac": RH_meas_frac,
    }

# ============================================================
# ----------------- MAIN ANALYSIS / PLOTTING -----------------
# ============================================================

def main():
    df_all = load_experiment_csv(DATA_CSV_PATH)

    # we'll accumulate per-event latent storage numbers here
    results_for_stats = []

    for k, ev in enumerate(EVENT_DEFINITIONS, start=1):
        t0 = pd.to_datetime(ev["start_ts"])
        t1 = pd.to_datetime(ev["end_ts"])

        # slice the event window
        win = df_all[(df_all["time"] >= t0) & (df_all["time"] <= t1)].copy()
        if len(win) < 2:
            print(f"[Event {k}] WARNING: not enough data in window {t0} → {t1}")
            continue

        cover_key = ev.get("cover", "cover_on")
        activity_factor = ACTIVITY_FACTOR_DEFAULTS.get(cover_key, 1.0)
        RH_target_frac = ev["RH_target_pct"] / 100.0

        sim = simulate_window(
            win,
            RH_target_frac=RH_target_frac,
            activity_factor=activity_factor,
            max_substep_s=MAX_SUBSTEP_S
        )

        # RH curves (measured vs model) for plotting
        RH_meas_pct = sim["RH_meas_frac"] * 100.0
        RH_model_pct = [
            100.0 * RH_from_W_T(Wm, T)
            for (Wm, T) in zip(sim["W_series"], sim["tair"])
        ]

        # latent energy (kWh)
        W0 = sim["W0"]
        Wend_model = sim["W_series"][-1]
        Wend_meas  = W_from_RH_T(sim["RH_meas_frac"][-1], sim["tair"][-1])

        dW_model = Wend_model - W0
        dW_meas  = Wend_meas  - W0

        T_mean = np.mean(sim["tair"])
        Q_model_kWh = h_fg_J_per_kg(T_mean) * (sim["m_air"] * dW_model) / 3.6e6
        Q_meas_kWh  = h_fg_J_per_kg(T_mean) * (sim["m_air"] * dW_meas ) / 3.6e6

        err_Q = Q_model_kWh - Q_meas_kWh
        abs_err_Q = abs(err_Q)
        sq_err_Q = err_Q**2

        duration_h = (t1 - t0).total_seconds() / 3600.0

        print("\n======================================")
        print(f"Event {k}: {t0} → {t1}")
        print(f"cover = {cover_key} | RH_target = {ev['RH_target_pct']}%")
        print(f"Duration: {duration_h:.2f} h")
        print(f"Model latent storage (kWh):    {Q_model_kWh:.3f}")
        print(f"Measured latent storage (kWh): {Q_meas_kWh:.3f}")
        print(f"ΔW_model: {dW_model:.4e} kg/kg | ΔW_meas: {dW_meas:.4e} kg/kg")
        print(f"Latent storage error (model - meas): {err_Q:.3f} kWh")
        print("======================================")

        # save per-event stats for global MAE/RMSE later
        results_for_stats.append({
            "event_idx": k,
            "t0": t0,
            "t1": t1,
            "Q_model_kWh": Q_model_kWh,
            "Q_meas_kWh": Q_meas_kWh,
            "Q_err_kWh": err_Q,
            "Q_abs_err_kWh": abs_err_Q,
            "Q_sq_err_kWh": sq_err_Q,
            "cover": cover_key,
            "RH_target_pct": ev["RH_target_pct"],
            "duration_h": duration_h
        })

        # Plot for this event
        if PLOTS_SHOW:
            fig, ax = plt.subplots()
            ax.plot(win["time"].values,
                    RH_meas_pct,
                    lw=2,
                    label="Measured RH (%)")

            ax.plot(win["time"].values,
                    RH_model_pct,
                    "--",
                    lw=2,
                    label="Model RH (%)")

            ax.set_xlabel("Time")
            ax.set_ylabel("RH (%)")
            ax.set_title(
                f"Event {k}: {t0.strftime('%Y-%m-%d %H:%M')} → {t1.strftime('%H:%M')} "
                f"| cover={cover_key}, target={ev['RH_target_pct']}%"
            )
            ax.grid(True)
            ax.legend()
            plt.show()

    # ===== Global error stats across all events =====
    if len(results_for_stats) > 0:
        stats_df = pd.DataFrame(results_for_stats)

        # we'll only consider rows where both model and meas are finite numbers
        stats_df = stats_df[
            np.isfinite(stats_df["Q_model_kWh"]) &
            np.isfinite(stats_df["Q_meas_kWh"])
        ].copy()

        if not stats_df.empty:
            mae = stats_df["Q_abs_err_kWh"].mean()
            rmse = math.sqrt(stats_df["Q_sq_err_kWh"].mean())

            print("\n========== AGGREGATE ERROR METRICS ==========")
            print(f"Number of events: {len(stats_df)}")
            print(f"MAE  of latent storage (kWh):  {mae:.3f}")
            print(f"RMSE of latent storage (kWh):  {rmse:.3f}")
            print("=============================================\n")
        else:
            print("\n(No valid events for aggregate stats)\n")
    else:
        print("\n(No events processed at all)\n")


if __name__ == "__main__":
    main()
