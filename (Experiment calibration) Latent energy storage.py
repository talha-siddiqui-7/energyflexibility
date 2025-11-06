#!/usr/bin/env python3
"""
Experimental validation of air flexibility model (balanced vs explicit flows + optional infiltration)
---------------------------------------------------------------------------------------------------

Moisture ODE (dry-air basis for mixing):
Balanced (legacy):
  dW/dt = [ m_evap(W)
          + m_vent_dry*(w_sup - W)
          + (INFILTRATION_ON ? m_inf_dry*(w_inf - W) : 0) ] / m_air

Explicit (recommended when AHU is imbalanced):
  dW/dt = (1/m_air)*[
            m_evap
          + m_sup_dry*(w_sup - W)
          + (INFILTRATION_ON ? m_inf_dry*(w_inf - W) : 0)
          - (m_ext_dry - m_sup_dry - m_inf_dry)*W
        ]
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# ============================================================
# --------------------- USER SETTINGS ------------------------
# ============================================================

DATA_CSV_PATH = r"M:\PhD\03 Experiments\Complete_17-09-2025_16-10-2025_with_AHU_power_infiltration.csv"

# Toggles
INFILTRATION_ON   = True    # set False to disable infiltration term
USE_BALANCED_VENT = False   # True = legacy averaged flow; False = explicit supply/extract with imbalance term

# Geometry / constants
VOLUME_M3    = 19.65 * 13.8 * 4.95
PRESSURE_PA  = 101325.0
POOL_AREA_M2 = 100.0
C_EVAP_BASE  = 4.0e-8   # kg/(s·m²·Pa) before activity factor

# Activity factor knobs
ACTIVITY_FACTOR_DEFAULTS = {"cover_on": 0.15, "cover_off": 0.25}
AF_GLOBAL_SCALE = 1.0

# Events
EVENT_DEFINITIONS = [
    {"start_ts": "2025-09-17 22:31", "end_ts": "2025-09-17 23:01", "cover": "cover_on",  "RH_target_pct": 65.0},
    {"start_ts": "2025-09-18 00:20", "end_ts": "2025-09-18 00:41", "cover": "cover_off", "RH_target_pct": 70.0},
    {"start_ts": "2025-09-18 00:47", "end_ts": "2025-09-18 01:08", "cover": "cover_off", "RH_target_pct": 70.0},
    {"start_ts": "2025-10-16 12:45", "end_ts": "2025-10-16 13:21", "cover": "cover_on",  "RH_target_pct": 65.0},
    {"start_ts": "2025-10-16 13:40", "end_ts": "2025-10-16 14:04", "cover": "cover_on",  "RH_target_pct": 60.0},
    {"start_ts": "2025-09-17 23:59", "end_ts": "2025-09-18 00:13", "cover": "cover_off", "RH_target_pct": 70.0},
]

PLOTS_SHOW     = True
MAX_SUBSTEP_S  = 60.0
CLAMP_ENABLED  = True

# ============================================================
# ------------- PSYCHROMETRIC HELPERS ------------------------
# ============================================================

def psat_pa(T_C): return 610.94 * math.exp(17.625 * T_C / (T_C + 243.04))
def Pv_from_W(W, p=PRESSURE_PA): return p * (W / (0.622 + W))
def W_from_RH_T(RH, T, p=PRESSURE_PA):
    Ps = psat_pa(T); Pv = RH * Ps
    return 0.622 * Pv / (p - Pv)
def RH_from_W_T(W, T, p=PRESSURE_PA): return Pv_from_W(W, p) / psat_pa(T)
def rho_da_from_wT(w, T, p=PRESSURE_PA):
    pv = p * (w / (0.62198 + w)); p_da = p - pv
    return p_da / (287.05 * (T + 273.15))
def h_fg_J_per_kg(T): return 2.501e6 - 2361.0 * T
def W_sat_from_T(T, p=PRESSURE_PA):
    Ps = psat_pa(T); Ps = min(Ps, 0.99 * p)
    return 0.622 * Ps / (p - Ps)
def clamp_W(W, T): return max(0.0, min(W, W_sat_from_T(T))) if CLAMP_ENABLED else W

# ============================================================
# ---------------- DATA LOADING ------------------------------
# ============================================================

def _pick_time_col(df):
    cols = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    lowers = {c.lower(): c for c in cols}
    if "datetime" in lowers:
        return lowers["datetime"]
    if "date" in lowers and "time" in lowers:
        return ("__COMBINE_DATE_TIME__", lowers["date"], lowers["time"])
    best, best_frac = None, 0.0
    for c in cols:
        s = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
        frac = s.notna().mean()
        if frac > 0.7 and frac > best_frac:
            best, best_frac = c, frac
    if best is None:
        raise ValueError(f"Could not find a time-like column. Columns: {list(df.columns)}")
    return best

def _parse_datetime_series(series):
    dt = pd.to_datetime(series, format="%d-%m-%Y %H:%M:%S", errors="coerce", dayfirst=True)
    if dt.notna().any(): return dt
    dt = pd.to_datetime(series, format="%d-%m-%Y %H:%M", errors="coerce", dayfirst=True)
    if dt.notna().any(): return dt
    dt = pd.to_datetime(series, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    if dt.notna().any(): return dt
    return pd.to_datetime(series, errors="coerce", dayfirst=True)

def dec_col(df, col):
    s = df.get(col, None)
    if s is None:
        return pd.Series(np.nan, index=df.index)
    s = pd.Series(s).astype(str).str.replace("'", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")

def load_experiment_csv(path):
    df_raw = pd.read_csv(path)

    # build datetime (keep integer index until after assignments)
    sel = _pick_time_col(df_raw)
    if isinstance(sel, tuple) and sel[0] == "__COMBINE_DATE_TIME__":
        _, date_col, clock_col = sel
        time = _parse_datetime_series(
            df_raw[date_col].astype(str).str.strip() + " " + df_raw[clock_col].astype(str).str.strip()
        ).dt.round("min")
    else:
        time = _parse_datetime_series(df_raw[sel]).dt.round("min")
    if time.isna().all():
        raise ValueError("Datetime parsing failed.")

    df = pd.DataFrame({"time": time})

    # explicit mapping per your labels
    temp_col = "Extract_air_Temp"
    rh_col   = "Extract_air_RH"

    # assign by position (.values) to avoid index-alignment NaNs
    df["T_room_C"]     = dec_col(df_raw, temp_col).values
    df["RH_room_pct"]  = dec_col(df_raw, rh_col).values
    df["T_sup_C"]      = dec_col(df_raw, "Supply_air_temp").values
    df["RH_sup_pct"]   = dec_col(df_raw, "Supply_air_RH").values
    df["Pool_water_C"] = dec_col(df_raw, "Pool_OF").values

    # optional cb.m/hr (not used in ODE)
    sup_flow_m3ph = dec_col(df_raw, "Supply air flow rate (cb.m/hr)").values
    ret_flow_m3ph = dec_col(df_raw, "Extract air flow rate (cb.m/hr)").values

    # density for conversion (legacy; not used in ODE now)
    w_ret_initial = np.array([
        W_from_RH_T(rh/100, T) if np.isfinite(rh) and np.isfinite(T) else np.nan
        for rh, T in zip(df["RH_room_pct"], df["T_room_C"])
    ])
    rho_air = np.array([
        rho_da_from_wT(w if np.isfinite(w) else 0.01, T if np.isfinite(T) else 25)
        for w, T in zip(w_ret_initial, df["T_room_C"])
    ])
    df["m_sup"] = (sup_flow_m3ph / 3600.0) * rho_air
    df["m_ret"] = (ret_flow_m3ph / 3600.0) * rho_air

    # infiltration / dry-air mass flows (kg/s, already dry-air in your unified file)
    df["w_sup"]      = dec_col(df_raw, "w_sup").values
    df["w_ret_meas"] = dec_col(df_raw, "w_ext").values
    df["w_inf"]      = dec_col(df_raw, "w_out").values
    df["m_sup_dry"]  = dec_col(df_raw, "mdot_sup_dry").values
    df["m_ext_dry"]  = dec_col(df_raw, "mdot_ext_dry").values   # renamed for clarity
    df["m_inf_dry"]  = dec_col(df_raw, "inf_dry_kg_s").fillna(0.0).values

    # now set time index + interpolate
    df = df.sort_values("time").set_index("time")
    for c in ["T_room_C","RH_room_pct","T_sup_C","RH_sup_pct","Pool_water_C",
              "m_sup","m_ret","w_sup","w_ret_meas","w_inf","m_sup_dry","m_ext_dry","m_inf_dry"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].interpolate(method="time").ffill().bfill()

    print(f"[column map] T_room_C <- '{temp_col}', RH_room_pct <- '{rh_col}'")
    return df.reset_index()

# ============================================================
# ---------------- SIMULATION CORE ---------------------------
# ============================================================

def simulate_window(win, RH_target_frac, activity_factor, use_infiltration=True, use_balanced=True):
    """
    Stable time-stepping. If use_infiltration=False, infiltration term is disabled.
    If use_balanced=True, uses m_vent_dry = 0.5*(m_sup_dry + m_ext_dry).
    Else, uses explicit supply/extract and an imbalance term.
    """
    tair    = win["T_room_C"].values
    RH_meas = (win["RH_room_pct"].values / 100).clip(0, 1)
    T_sup   = win["T_sup_C"].values
    RH_sup  = (win["RH_sup_pct"].values / 100).clip(0, 1)
    Twater  = win["Pool_water_C"].values

    # humidity ratios
    if "w_ret_meas" in win.columns and np.isfinite(win["w_ret_meas"]).any():
        W_meas_series = win["w_ret_meas"].values.astype(float)
    else:
        W_meas_series = np.array([W_from_RH_T(rh, T) if np.isfinite(rh) and np.isfinite(T) else np.nan
                                  for rh, T in zip(RH_meas, tair)], dtype=float)

    if "w_sup" in win.columns and np.isfinite(win["w_sup"]).any():
        w_sup = win["w_sup"].values.astype(float)
    else:
        w_sup = np.array([W_from_RH_T(rh, Ts) if np.isfinite(rh) and np.isfinite(Ts) else np.nan
                          for rh, Ts in zip(RH_sup, T_sup)], dtype=float)

    w_inf = (win["w_inf"].values.astype(float)
             if "w_inf" in win.columns else np.copy(w_sup))

    # dry-air flows
    m_sup_dry = win["m_sup_dry"].values.astype(float) if "m_sup_dry" in win.columns else np.zeros_like(w_sup)
    m_ext_dry = win["m_ext_dry"].values.astype(float) if "m_ext_dry" in win.columns else np.zeros_like(w_sup)
    m_inf_dry = win["m_inf_dry"].values.astype(float) if "m_inf_dry" in win.columns else np.zeros_like(w_sup)
    if not use_infiltration:
        m_inf_dry = np.zeros_like(m_inf_dry)

    # guard
    if not (len(W_meas_series) and np.isfinite(W_meas_series[0]) and np.isfinite(tair[0])):
        return {"tsec": np.array([]), "W_series": np.array([]),
                "m_air": np.nan, "W0": np.nan, "tair": tair, "RH_meas": np.array([])}

    # initial state and air mass
    W0    = float(W_meas_series[0])
    m_air = rho_da_from_wT(W0, tair[0]) * VOLUME_M3

    tsec  = (win["time"] - win["time"].iloc[0]).dt.total_seconds().values
    dt    = np.diff(tsec, prepend=tsec[0])
    Pw    = np.array([psat_pa(Tw) for Tw in Twater])

    W = float(W0)
    W_series = [W0]

    for i in range(1, len(tsec)):
        dti = float(dt[i])
        if dti <= 0:
            W_series.append(W); continue

        dPa_dW = PRESSURE_PA * 0.622 / (0.622 + W)**2
        k_e = (C_EVAP_BASE * activity_factor) * POOL_AREA_M2 * dPa_dW

        if use_balanced:
            m_vent_dry = 0.5 * (m_sup_dry[i] + m_ext_dry[i]) if np.isfinite(m_sup_dry[i]) and np.isfinite(m_ext_dry[i]) else 0.0
            mi = m_inf_dry[i] if np.isfinite(m_inf_dry[i]) else 0.0
            # step-size denominator (positive contributions)
            denom = max(m_vent_dry, 0.0) + max(mi, 0.0) + k_e
        else:
            ms = m_sup_dry[i] if np.isfinite(m_sup_dry[i]) else 0.0
            me = m_ext_dry[i] if np.isfinite(m_ext_dry[i]) else 0.0
            mi = m_inf_dry[i] if np.isfinite(m_inf_dry[i]) else 0.0
            denom = max(ms, 0.0) + max(me, 0.0) + max(mi, 0.0) + k_e

        tau = m_air / denom if denom > 1e-12 else np.inf
        dt_target = min(MAX_SUBSTEP_S, (tau/5.0) if np.isfinite(tau) else MAX_SUBSTEP_S, dti)
        n_sub = max(1, int(math.ceil(dti / dt_target)))
        dt_sub = dti / n_sub

        for _ in range(n_sub):
            Pa   = Pv_from_W(W)
            m_ev = (C_EVAP_BASE * activity_factor) * POOL_AREA_M2 * (Pw[i] - Pa)

            ws = w_sup[i] if np.isfinite(w_sup[i]) else W
            wi = w_inf[i] if np.isfinite(w_inf[i]) else W

            if use_balanced:
                mv = 0.5 * (m_sup_dry[i] + m_ext_dry[i]) if np.isfinite(m_sup_dry[i]) and np.isfinite(m_ext_dry[i]) else 0.0
                mi = m_inf_dry[i] if np.isfinite(m_inf_dry[i]) else 0.0
                rhs = m_ev + mv*(ws - W) + mi*(wi - W)
            else:
                ms = m_sup_dry[i] if np.isfinite(m_sup_dry[i]) else 0.0
                me = m_ext_dry[i] if np.isfinite(m_ext_dry[i]) else 0.0
                mi = m_inf_dry[i] if np.isfinite(m_inf_dry[i]) else 0.0
                # explicit supply/extract with imbalance term
                rhs = (
                    m_ev
                    + ms*(ws - W)
                    + mi*(wi - W)
                    - (me - ms - mi)*W
                )

            dWdt = rhs / m_air
            W   += dWdt * dt_sub

            # clamp to target & saturation
            W_cap = W_from_RH_T(RH_target_frac, tair[i]) if np.isfinite(tair[i]) else W
            if W > W_cap: W = W_cap
            W = clamp_W(W, tair[i] if np.isfinite(tair[i]) else 25.0)

        W_series.append(W)

    RH_meas_used = np.array([RH_from_W_T(w, T) if np.isfinite(w) and np.isfinite(T) else np.nan
                             for w, T in zip(W_meas_series, tair)]).clip(0, 1)

    return {"tsec": np.array(tsec), "W_series": np.array(W_series),
            "m_air": m_air, "W0": W0, "tair": tair, "RH_meas": RH_meas_used}

# ============================================================
# ---------------- MAIN ANALYSIS -----------------------------
# ============================================================

def _check_events_cover_data(df, events):
    data_min, data_max = df["time"].min(), df["time"].max()
    inside, outside = [], []
    for i, ev in enumerate(events, start=1):
        t0, t1 = pd.to_datetime(ev["start_ts"]), pd.to_datetime(ev["end_ts"])
        if (t1 < data_min) or (t0 > data_max):
            outside.append((i, t0, t1))
        else:
            inside.append((i, t0, t1))
    return data_min, data_max, inside, outside

def main():
    df_all = load_experiment_csv(DATA_CSV_PATH)
    data_min, data_max, inside, outside = _check_events_cover_data(df_all, EVENT_DEFINITIONS)
    print(f"Data time range: {data_min} → {data_max}")
    print(f"Infiltration toggled: {'ON' if INFILTRATION_ON else 'OFF'} | Vent model: {'BALANCED' if USE_BALANCED_VENT else 'EXPLICIT'}")

    if not inside:
        raise RuntimeError(
            "All events are outside the data range.\n"
            f"- Data range: {data_min} → {data_max}\n"
            f"- Example event: {EVENT_DEFINITIONS[0]['start_ts']} → {EVENT_DEFINITIONS[0]['end_ts']}\n"
            "Fix: point DATA_CSV_PATH to the correct file, or update EVENT_DEFINITIONS."
        )
    if outside:
        print("Note: these events are outside the range and will be skipped:")
        for i, t0, t1 in outside:
            print(f"  - Event {i}: {t0} → {t1}")

    events_to_run = [EVENT_DEFINITIONS[i-1] for (i, _, _) in inside]

    for k, ev in enumerate(events_to_run, start=1):
        t0, t1 = pd.to_datetime(ev["start_ts"]), pd.to_datetime(ev["end_ts"])
        win = df_all[(df_all["time"] >= t0) & (df_all["time"] <= t1)].copy()
        if len(win) < 2:
            print(f"[skip] Event {k}: {t0}→{t1} has {len(win)} rows. Check timestamps.")
            continue

        needed = ["T_room_C","RH_room_pct","w_ret_meas","w_sup","w_inf","m_sup_dry","m_ext_dry","m_inf_dry"]
        coverage = {c: float(np.isfinite(win[c]).mean()) if c in win else 0.0 for c in needed}
        print(f"\n[Event {k} coverage] " + ", ".join(f"{c}={coverage[c]:.2f}" for c in needed))

        base_af = ev.get("AF", ACTIVITY_FACTOR_DEFAULTS[ev["cover"]])
        act = base_af * ev.get("AF_scale", 1.0) * AF_GLOBAL_SCALE

        RH_target = ev["RH_target_pct"] / 100.0
        sim = simulate_window(
            win, RH_target, act,
            use_infiltration=INFILTRATION_ON,
            use_balanced=USE_BALANCED_VENT
        )

        if sim["W_series"].size == 0 or not np.isfinite(sim["W0"]):
            print(f"[skip] Event {k}: inputs produced NaN initial state — check coverage line above.")
            continue

        RH_meas_pct  = sim["RH_meas"] * 100.0
        RH_model_pct = [100.0 * RH_from_W_T(Wm, T) for Wm, T in zip(sim["W_series"], sim["tair"])]

        # latent energy trajectories
        W0, m_air = sim["W0"], sim["m_air"]
        T_mean = float(np.mean(sim["tair"]))
        hfg = h_fg_J_per_kg(T_mean)

        Q_model = hfg * m_air * (sim["W_series"] - W0) / 3.6e6
        W_meas  = np.array([W_from_RH_T(rh, T) for rh, T in zip(sim["RH_meas"], sim["tair"])])
        Q_meas  = hfg * m_air * (W_meas - W0) / 3.6e6

        # metrics
        err  = Q_model - Q_meas
        mae  = float(np.nanmean(np.abs(err)))
        rmse = float(np.sqrt(np.nanmean(err**2)))

        Q_model_final = float(Q_model[-1])
        Q_meas_final  = float(Q_meas[-1])
        Q_err_final   = Q_model_final - Q_meas_final

        # quick imbalance diagnostic for explicit mode
        if not USE_BALANCED_VENT:
            imbalance_mean = float((win["m_ext_dry"] - win["m_sup_dry"] - (win["m_inf_dry"] if INFILTRATION_ON else 0.0)).mean())
            print(f"Imbalance mean (me - ms - mi) [kg/s]: {imbalance_mean:+.4f}")

        print("\n======================================")
        print(f"Event {k}: {t0} → {t1}")
        print(f"cover = {ev['cover']} | RH_target = {ev['RH_target_pct']}% | activity_factor = {act:.3f}")
        print(f"Infiltration: {'ON' if INFILTRATION_ON else 'OFF'} | Vent model: {'BALANCED' if USE_BALANCED_VENT else 'EXPLICIT'}")
        print(f"Duration: {(t1 - t0).total_seconds() / 3600.0:.2f} h")
        print(f"Model latent storage (kWh):    {Q_model_final:.3f}")
        print(f"Measured latent storage (kWh): {Q_meas_final:.3f}")
        print(f"MAE over time (kWh):  {mae:.3f}")
        print(f"RMSE over time (kWh): {rmse:.3f}")
        print(f"Latent storage error (model - meas): {Q_err_final:.3f} kWh")
        print("======================================")

        if PLOTS_SHOW:
            plt.figure()
            plt.plot(win["time"], RH_meas_pct, lw=2, label="Measured RH (%)")
            plt.plot(win["time"], RH_model_pct, "--", lw=2, label="Model RH (%)")
            plt.xlabel("Time"); plt.ylabel("RH (%)")
            plt.title(f"Event {k}: {ev['cover']}, {ev['RH_target_pct']}% | AF={act:.2f} | Infil={'ON' if INFILTRATION_ON else 'OFF'} | Vent={'BAL' if USE_BALANCED_VENT else 'EXP'}")
            plt.grid(True); plt.legend()
            plt.show()

if __name__ == "__main__":
    main()
