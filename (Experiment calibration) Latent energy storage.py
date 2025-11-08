#!/usr/bin/env python3
"""
Air flexibility validation: Pure External vs Pure AHU runs
- Balanced vs explicit ventilation (toggle)
- Optional infiltration (toggle)
- Robust datetime handling + safe interpolation
- Per-event metrics in kWh (MAE, RMSE, ΔQ_final)

Dataset columns expected:
(datetime, date, time, Air_exhaust, Extract_air_RH_sensor, Extract_air_Temp_sensor,
 Extract_air_Temp_AHU, Extract_air_RH_AHU, Outdoor_RH, Outdoor_Temp, Pool_OF,
 Supply_air_temp_sensor, Supply_air_RH_sensor, Supply_air_RH_AHU, Technical_area,
 AHU Peak W, AHU Average W, Supply air flow rate (cb.m/hr),
 Extract air flow rate (cb.m/hr), Setpoint Pool water temperature, Setpoint RH,
 Fresh air damper, Vdot_sup_m3s, Vdot_ext_m3s, rho_sup, rho_ext, w_sup, w_ext, w_out,
 mdot_sup_moist, mdot_ext_moist, mdot_sup_dry, mdot_ext_dry, inf_moist_kg_s,
 inf_dry_kg_s, inf_dry_from_water_kg_s)
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# ============================================================
# --------------------- USER SETTINGS ------------------------
# ============================================================

DATA_CSV_PATH = r"M:\PhD\03 Experiments\17-09-2025_16-10-2025_sensor_vs_AHU_data.csv"

# Which modes to run (order matters in output/plots)
RUN_MODES = ["external", "ahu"]   # choose any subset: ["external"], ["ahu"], or both

# Core physics toggles
INFILTRATION_ON   = True     # include infiltration mixing term
USE_BALANCED_VENT = False    # True: legacy average flow; False: explicit supply/extract with imbalance

# Geometry / constants
VOLUME_M3    = 19.65 * 13.8 * 4.95
PRESSURE_PA  = 101325.0
POOL_AREA_M2 = 100.0
C_EVAP_BASE  = 4.0e-8   # kg/(s·m²·Pa) base evaporation coefficient

# Activity factor knobs
ACTIVITY_FACTOR_DEFAULTS = {"cover_on": 0.15, "cover_off": 0.5}
AF_GLOBAL_SCALE = 1.0

# Events to simulate
EVENT_DEFINITIONS = [
    {"start_ts": "2025-09-17 22:31", "end_ts": "2025-09-17 23:01", "cover": "cover_on",  "RH_target_pct": 65.0},
    {"start_ts": "2025-09-18 00:20", "end_ts": "2025-09-18 00:41", "cover": "cover_off", "RH_target_pct": 70.0},
    {"start_ts": "2025-09-18 00:47", "end_ts": "2025-09-18 01:08", "cover": "cover_off", "RH_target_pct": 70.0},
    {"start_ts": "2025-10-16 12:45", "end_ts": "2025-10-16 13:21", "cover": "cover_on",  "RH_target_pct": 65.0},
    {"start_ts": "2025-10-16 13:40", "end_ts": "2025-10-16 14:04", "cover": "cover_on",  "RH_target_pct": 60.0},
    {"start_ts": "2025-09-17 23:59", "end_ts": "2025-09-18 00:13", "cover": "cover_off", "RH_target_pct": 70.0},
]

# Plot / solver
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
    # heuristic fallback
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

    # Build datetime, drop NaT and duplicates
    sel = _pick_time_col(df_raw)
    if isinstance(sel, tuple) and sel[0] == "__COMBINE_DATE_TIME__":
        _, date_col, clock_col = sel
        time = _parse_datetime_series(
            df_raw[date_col].astype(str).str.strip() + " " + df_raw[clock_col].astype(str).str.strip()
        )
    else:
        time = _parse_datetime_series(df_raw[sel])

    time = time.dt.round("min")
    ok = time.notna()
    if not ok.all():
        df_raw = df_raw.loc[ok].copy()
        time   = time.loc[ok]
    dup = time.duplicated()
    if dup.any():
        df_raw = df_raw.loc[~dup].copy()
        time   = time.loc[~dup]

    df = pd.DataFrame({"time": time})

    # --- Map required columns (exact names you provided) ---
    # Extract (room probe) and AHU extract
    df["T_ext_room"]   = dec_col(df_raw, "Extract_air_Temp_sensor").values
    df["RH_ext_room"]  = dec_col(df_raw, "Extract_air_RH_sensor").values
    df["T_ext_AHU"]    = dec_col(df_raw, "Extract_air_Temp_AHU").values
    df["RH_ext_AHU"]   = dec_col(df_raw, "Extract_air_RH_AHU").values

    # Supply (assume temp sensor is AHU duct temp)
    df["T_sup"]        = dec_col(df_raw, "Supply_air_temp_sensor").values
    df["RH_sup_room"]  = dec_col(df_raw, "Supply_air_RH_sensor").values
    df["RH_sup_AHU"]   = dec_col(df_raw, "Supply_air_RH_AHU").values

    # Water & outdoor
    df["T_pool"]       = dec_col(df_raw, "Pool_OF").values
    df["RH_outdoor"]   = dec_col(df_raw, "Outdoor_RH").values
    df["T_outdoor"]    = dec_col(df_raw, "Outdoor_Temp").values

    # Provided humidity ratios and (dry-air) mass flows
    df["w_sup_file"]   = dec_col(df_raw, "w_sup").values
    df["w_ext_file"]   = dec_col(df_raw, "w_ext").values
    df["w_out"]        = dec_col(df_raw, "w_out").values
    df["m_sup_dry"]    = dec_col(df_raw, "mdot_sup_dry").values
    df["m_ext_dry"]    = dec_col(df_raw, "mdot_ext_dry").values
    df["m_inf_dry"]    = dec_col(df_raw, "inf_dry_kg_s").fillna(0.0).values

    # Set tidy index + robust interpolation
    df = df.sort_values("time").set_index("time")

    def _safe_time_interpolate(s):
        s = pd.to_numeric(s, errors="coerce")
        try:
            return s.interpolate(method="time").ffill().bfill()
        except Exception:
            return s.interpolate(method="linear", limit_direction="both").ffill().bfill()

    for c in df.columns:
        df[c] = _safe_time_interpolate(df[c])

    # Build humidity ratios from RH+T pairs (pure sources)
    df["w_ext_room"] = [W_from_RH_T(rh/100.0, T) if np.isfinite(rh) and np.isfinite(T) else np.nan
                        for rh, T in zip(df["RH_ext_room"], df["T_ext_room"])]
    df["w_ext_ahu"]  = [W_from_RH_T(rh/100.0, T) if np.isfinite(rh) and np.isfinite(T) else np.nan
                        for rh, T in zip(df["RH_ext_AHU"],  df["T_ext_AHU"])]
    df["w_sup_room"] = [W_from_RH_T(rh/100.0, T) if np.isfinite(rh) and np.isfinite(T) else np.nan
                        for rh, T in zip(df["RH_sup_room"], df["T_sup"])]
    df["w_sup_ahu"]  = [W_from_RH_T(rh/100.0, T) if np.isfinite(rh) and np.isfinite(T) else np.nan
                        for rh, T in zip(df["RH_sup_AHU"],  df["T_sup"])]

    # Final fill for computed series
    for c in ["w_ext_room","w_ext_ahu","w_sup_room","w_sup_ahu"]:
        df[c] = _safe_time_interpolate(df[c])

    print("[OK] Columns mapped and humidities built (room/AHU).")
    return df.reset_index()

# ============================================================
# ---------------- SIMULATION CORE ---------------------------
# ============================================================

def _series_to_RH_pct(W_series, T_series):
    return np.array([100.0 * RH_from_W_T(w, T) if np.isfinite(w) and np.isfinite(T) else np.nan
                     for w, T in zip(W_series, T_series)])

def _latent_Q_kWh(W_series, W0, m_air, T_series):
    T_mean = float(np.nanmean(T_series))
    hfg = h_fg_J_per_kg(T_mean)
    return hfg * m_air * (W_series - W0) / 3.6e6

def pick_inputs_for_mode(win, mode):
    """Return dict of inputs for 'external' or 'ahu' run."""
    assert mode in ("external","ahu")
    if mode == "external":
        return dict(
            W_init = win["w_ext_room"].values.astype(float),
            w_sup  = (win["w_sup_file"].values.astype(float)
                      if np.isfinite(win["w_sup_file"]).any() else win["w_sup_room"].values.astype(float)),
            W_meas = win["w_ext_room"].values.astype(float),  # metrics reference
            tag    = "EXT"
        )
    else:
        return dict(
            W_init = win["w_ext_ahu"].values.astype(float),
            w_sup  = win["w_sup_ahu"].values.astype(float),
            W_meas = win["w_ext_ahu"].values.astype(float),   # metrics reference
            tag    = "AHU"
        )

def simulate_window(win, RH_target_frac, activity_factor, mode, use_infiltration=True, use_balanced=True):
    """Run one window in a given mode ('external' or 'ahu')."""
    tair    = win["T_ext_room"].values   # room air temperature path (state T)
    Twater  = win["T_pool"].values
    w_inf   = win["w_out"].values if "w_out" in win.columns else win["w_sup_room"].values

    # dry-air flows
    m_sup_dry = win["m_sup_dry"].values.astype(float) if "m_sup_dry" in win.columns else np.zeros_like(tair)
    m_ext_dry = win["m_ext_dry"].values.astype(float) if "m_ext_dry" in win.columns else np.zeros_like(tair)
    m_inf_dry = win["m_inf_dry"].values.astype(float) if "m_inf_dry" in win.columns else np.zeros_like(tair)
    if not use_infiltration:
        m_inf_dry = np.zeros_like(m_inf_dry)

    # pick pure inputs by mode
    sel = pick_inputs_for_mode(win, mode)
    W0_series = sel["W_init"]
    w_sup     = sel["w_sup"]
    W_meas    = sel["W_meas"]
    metric_tag= sel["tag"]

    # initial state
    if not len(W0_series) or not np.isfinite(W0_series[0]) or not np.isfinite(tair[0]):
        return {"ok": False}

    W0   = float(W0_series[0])
    m_air= rho_da_from_wT(W0, tair[0]) * VOLUME_M3

    tsec = (win["time"] - win["time"].iloc[0]).dt.total_seconds().values
    dt   = np.diff(tsec, prepend=tsec[0])
    Pw   = np.array([psat_pa(Tw) for Tw in Twater])

    W = float(W0); W_series = [W0]

    for i in range(1, len(tsec)):
        dti = float(dt[i])
        if dti <= 0:
            W_series.append(W); continue

        dPa_dW = PRESSURE_PA * 0.622 / (0.622 + W)**2
        k_e    = (C_EVAP_BASE * activity_factor) * POOL_AREA_M2 * dPa_dW

        if use_balanced:
            m_vent = 0.5 * (m_sup_dry[i] + m_ext_dry[i]) if np.isfinite(m_sup_dry[i]) and np.isfinite(m_ext_dry[i]) else 0.0
            mi     = m_inf_dry[i] if np.isfinite(m_inf_dry[i]) else 0.0
            denom  = max(m_vent,0.0) + max(mi,0.0) + k_e
        else:
            ms = m_sup_dry[i] if np.isfinite(m_sup_dry[i]) else 0.0
            me = m_ext_dry[i] if np.isfinite(m_ext_dry[i]) else 0.0
            mi = m_inf_dry[i] if np.isfinite(m_inf_dry[i]) else 0.0
            denom = max(ms,0.0) + max(me,0.0) + max(mi,0.0) + k_e

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
                rhs = m_ev + ms*(ws - W) + mi*(wi - W) - (me - ms - mi)*W

            dWdt = rhs / m_air
            W   += dWdt * dt_sub

            # clamp to target & saturation
            W_cap = W_from_RH_T(RH_target_frac, tair[i]) if np.isfinite(tair[i]) else W
            if W > W_cap: W = W_cap
            W = clamp_W(W, tair[i] if np.isfinite(tair[i]) else 25.0)

        W_series.append(W)

    return {
        "ok": True,
        "tsec": np.array(tsec),
        "W_series": np.array(W_series),
        "W_meas": W_meas,
        "m_air": m_air,
        "W0": W0,
        "tair": tair,
        "metric_tag": metric_tag,
    }

# ============================================================
# ---------------- MAIN ANALYSIS -----------------------------
# ============================================================

def _check_events_cover_data(df, events):
    data_min, data_max = df["time"].min(), df["time"].max()
    inside, outside = [], []
    for i, ev in enumerate(events, start=1):
        t0, t1 = pd.to_datetime(ev["start_ts"]), pd.to_datetime(ev["end_ts"])
        (outside if (t1 < data_min) or (t0 > data_max) else inside).append((i, t0, t1))
    return data_min, data_max, inside, outside

def main():
    df_all = load_experiment_csv(DATA_CSV_PATH)
    data_min, data_max, inside, outside = _check_events_cover_data(df_all, EVENT_DEFINITIONS)
    print(f"Data: {data_min} → {data_max}")
    print(f"Infiltration: {'ON' if INFILTRATION_ON else 'OFF'} | Vent: {'BALANCED' if USE_BALANCED_VENT else 'EXPLICIT'}")
    if not inside:
        raise RuntimeError("No events within data range.")
    if outside:
        print("Skipping events outside range:", [i for i,_,_ in outside])

    events_to_run = [EVENT_DEFINITIONS[i-1] for (i,_,_) in inside]

    for k, ev in enumerate(events_to_run, start=1):
        t0, t1 = pd.to_datetime(ev["start_ts"]), pd.to_datetime(ev["end_ts"])
        win = df_all[(df_all["time"] >= t0) & (df_all["time"] <= t1)].copy()
        if len(win) < 2:
            print(f"[skip] Event {k}: not enough rows"); continue

        base_af = ev.get("AF", ACTIVITY_FACTOR_DEFAULTS[ev["cover"]])
        act = base_af * ev.get("AF_scale", 1.0) * AF_GLOBAL_SCALE
        RH_target = ev["RH_target_pct"] / 100.0

        # For plotting context: build both measured RH series
        RH_ext_pct = _series_to_RH_pct(win["w_ext_room"].values, win["T_ext_room"].values)
        RH_ahu_pct = _series_to_RH_pct(win["w_ext_ahu"].values,  win["T_ext_room"].values)

        for mode in RUN_MODES:
            sim = simulate_window(
                win, RH_target, act, mode,
                use_infiltration=INFILTRATION_ON,
                use_balanced=USE_BALANCED_VENT
            )
            if not sim["ok"]:
                print(f"[skip] Event {k} ({mode}): bad initial state"); continue

            # RH curves
            RH_model_pct = _series_to_RH_pct(sim["W_series"], sim["tair"])

            # Latent energy metrics vs the mode's own measured series
            Q_model = _latent_Q_kWh(sim["W_series"], sim["W0"], sim["m_air"], sim["tair"])
            Q_meas  = _latent_Q_kWh(sim["W_meas"],  sim["W0"], sim["m_air"], sim["tair"])
            err  = Q_model - Q_meas
            mae  = float(np.nanmean(np.abs(err)))
            rmse = float(np.sqrt(np.nanmean(err**2)))
            dqf  = float(Q_model[-1] - Q_meas[-1])

            if not USE_BALANCED_VENT:
                imb = float((win["m_ext_dry"] - win["m_sup_dry"] - (win["m_inf_dry"] if INFILTRATION_ON else 0.0)).mean())
                print(f"[imbalance] Event {k} ({mode}): (me - ms - mi) = {imb:+.4f} kg/s")

            print("\n======================================")
            print(f"Event {k}: {t0} → {t1}")
            print(f"Mode={mode.upper()} | cover={ev['cover']} | RH_target={ev['RH_target_pct']}% | AF={act:.3f}")
            print(f"MAE={mae:.3f} kWh | RMSE={rmse:.3f} kWh | ΔQ_final={dqf:+.3f} kWh")
            print("======================================")

            if PLOTS_SHOW:
                plt.figure(figsize=(10,5))
                plt.plot(win["time"], RH_ext_pct,  lw=2, label="Measured RH ext (%)")
                plt.plot(win["time"], RH_ahu_pct,  lw=2, label="Measured RH AHU (%)")
                plt.plot(win["time"], RH_model_pct, "--", lw=2, label=f"Model RH (%) [{mode.upper()} run]")
                plt.xlabel("Time"); plt.ylabel("RH (%)")
                plt.title(
                    f"Event {k}: {ev['cover']}, {ev['RH_target_pct']}% | "
                    f"AF={act:.2f} | Infil={'ON' if INFILTRATION_ON else 'OFF'} | "
                    f"Vent={'BAL' if USE_BALANCED_VENT else 'EXP'} | Mode={mode.upper()}"
                )
                plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
