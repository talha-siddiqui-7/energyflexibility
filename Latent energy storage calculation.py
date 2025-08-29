#!/usr/bin/env python3
"""
Validation script (single-run, event windows)

Compares a lightweight RH-relaxation model vs. IDA ICE time series for many
short "events" (e.g., daily from START_HOUR for DURATION_H).

INPUT: a single CSV with columns like your screenshot
  - Hour (timestamp or parsable string)
  - Air temperature at extract, Deg-C
  - Air temperature at Supply, Deg-C
  - Humidity ratio at extract, kg/kg     (or RH at extract + T extract)
  - Humidity ratio at supply, kg/kg      (or RH at supply + T supply)
  - Mass flow at Supply, kg/s
  - Mass flow at extract, kg/s
  - Extract RH   (optional if W provided)
  - SUPPLY RH    (optional if W provided)

OUTPUTS:
  - per-event CSV with model and IDA ICE quantities
  - short console summary
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import math
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import math

# -------------------------
# HARD-CODED PATHS (KEEP)
# -------------------------
INPUT_CSV  = r"M:\PhD\02 Data sets from simulations\Air Flexibility\Winter_month.csv"        # single-run export
OUTPUT_CSV = r"M:\PhD\02 Data sets from simulations\Air Flexibility\true_storage_01to04.csv" # (reused) per-event output
OUTPUT_HOURLY    = r"M:\PhD\02 Data sets from simulations\Air Flexibility\true_storage_hourly.csv"  # (not used here)
OUTPUT_PNG       = r"M:\PhD\02 Data sets from simulations\Air Flexibility\true_storage_hourly.png"  # (not used here)
OUTPUT_PER_NIGHT = r"M:\PhD\02 Data sets from simulations\Air Flexibility\true_storage_01to04.csv"  # (same as OUTPUT_CSV)

# -------------------------
# SITE / MODEL SETTINGS
# -------------------------
VOLUME_M3     = 19.65 * 13.8 * 4.95     # hall volume
PRESSURE_PA   = 101325.0                # barometric pressure
RELAXED_RH_STAR = 0.65                  # relaxed RH target for cap (fraction, e.g. 0.65 = 65%)
POOL_WATER_T_C  = 28.0                  # assumed constant during window
# Evap. model constants (use what you calibrated; keep fixed across events)
C_EVAP = 4.0e-8                         # kg/(s·m²·Pa)
F_A    = 0.5                            # activity factor
POOL_AREA_M2 = 100.0                    # only affects k_e if you later calibrate via \dot m0; not essential for validation if using Direct

# Event definition:
START_HOUR  = 20      # daily event start hour (local hour in data)
DURATION_H  = 3       # event length (hours)
# Optional: explicit events list (timestamps). If non-empty, these override the daily pattern.
EXPLICIT_EVENTS = []  # e.g., ["2024-01-10 21:00", "2024-01-12 21:00"]

# -------------------------
# PSYCHRO HELPERS
# -------------------------
def psat_pa(T_C: float) -> float:
    """Saturation vapour pressure [Pa] (Tetens/Magnus form)."""
    T = float(T_C)
    return 610.94 * math.exp(17.625 * T / (T + 243.04))

def W_from_RH_T(RH: float, T_C: float, p_atm: float = PRESSURE_PA) -> float:
    """Humidity ratio from RH (0-1) and temperature [°C] at p_atm [Pa]."""
    Ps = psat_pa(T_C)
    Pv = RH * Ps
    return 0.622 * Pv / (p_atm - Pv)

def Pv_from_W(W: float, p_atm: float = PRESSURE_PA) -> float:
    """Vapour partial pressure [Pa] from humidity ratio W [kg/kg]."""
    return p_atm * (W / (0.622 + W))

def rho_da_from_wT(w: float, T_C: float, p: float = PRESSURE_PA) -> float:
    """Dry-air density [kg/m³] from humidity ratio w [kg/kg_da] and T_C [°C] at pressure p [Pa]."""
    w = float(w); T_C = float(T_C)
    pv = p * (w / (0.62198 + w))
    p_da = p - pv
    return p_da / (287.05 * (T_C + 273.15))

def h_fg_J_per_kg(T_C: float) -> float:
    """Latent heat [J/kg] at air temperature T_C [°C]."""
    return 2.501e6 - 2361.0 * float(T_C)

# -------------------------
# COLUMN-LOOKUP HELPERS (robust to naming)
# -------------------------
def find_col(df, keys):
    """Return first column whose name contains ALL tokens in 'keys' (case-insensitive)."""
    keys = [k.lower() for k in keys]
    for c in df.columns:
        name = c.lower()
        if all(k in name for k in keys):
            return c
    return None

def load_single_run(path):
    import pandas as pd
    import numpy as np

    df = pd.read_csv(path)

    # ---- 1) Flatten "two-row header" (Hour / Variables) if present
    def flatten_two_row_header(df0):
        try:
            row0 = df0.iloc[0].astype(str).str.strip().str.lower()
            row1 = df0.iloc[1].astype(str).str.strip()
        except Exception:
            return df0  # not enough rows to inspect

        looks_like_hour_row = row0.str.contains("hour").any()
        looks_like_variables = row0.str.contains("variables").any()
        if looks_like_hour_row and looks_like_variables:
            new_cols = [str(df0.iloc[0, 0]).strip() or "Hour"]
            for j in range(1, df0.shape[1]):
                name = str(df0.iloc[1, j]).strip()
                new_cols.append(name if name else f"col{j}")
            df1 = df0.iloc[2:].copy()
            df1.columns = new_cols
            return df1
        return df0

    df = flatten_two_row_header(df)

    # ---- 2) Helper to find a column by tokens (order-insensitive)
    def find_col(frame, tokens):
        toks = [t.lower() for t in tokens]
        for c in frame.columns:
            name = str(c).lower()
            if all(t in name for t in toks):
                return c
        return None

    # ---- 3) Find/parse time column robustly
    time_col = find_col(df, ["hour"]) or find_col(df, ["time"])
    parsed_time = None
    if time_col is not None:
        parsed_time = pd.to_datetime(df[time_col], errors="coerce", dayfirst=True)

    if parsed_time is None or parsed_time.notna().sum() == 0:
        # Fallback: scan every column; pick any with >70% parsable datetimes
        best = None
        best_series = None
        best_frac = 0.0
        for c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
            frac = s.notna().mean()
            if frac > 0.7 and frac > best_frac:
                best, best_series, best_frac = c, s, frac
        if best is None:
            raise ValueError(f"No time-like column found. Columns: {list(df.columns)}")
        time_col, parsed_time = best, best_series

    # ---- 4) Build normalized frame
    out = pd.DataFrame({
        "time": parsed_time,
        "date": parsed_time.dt.date,
        "hour": parsed_time.dt.hour
    })

    # Temperatures
    T_room_col = find_col(df, ["air", "extract"]) or find_col(df, ["room"])
    T_sup_col  = find_col(df, ["air temperature at supply"]) or find_col(df, ["supply", "deg"])
    out["T_room_C"] = pd.to_numeric(df[T_room_col], errors="coerce") if T_room_col else np.nan
    out["T_sup_C"]  = pd.to_numeric(df[T_sup_col],  errors="coerce") if T_sup_col  else np.nan

    # Humidity ratio at extract (room)
    w_ret_col = find_col(df, ["humidity", "extract"])
    RH_ret_col = find_col(df, ["extract", "rh"])
    if w_ret_col is not None:
        out["w_ret"] = pd.to_numeric(df[w_ret_col], errors="coerce")
    elif RH_ret_col is not None and T_room_col is not None:
        RH = pd.to_numeric(df[RH_ret_col], errors="coerce")/100.0
        out["w_ret"] = [W_from_RH_T(r, t) if pd.notna(r) and pd.notna(t) else np.nan
                        for r, t in zip(RH, out["T_room_C"])]
    else:
        raise ValueError("Need 'Humidity ratio at extract' or 'Extract RH' + 'Air temperature at extract'.")

    # Humidity ratio at supply
    w_sup_col = find_col(df, ["humidity", "supply"])
    RH_sup_col = find_col(df, ["supply", "rh"])
    if w_sup_col is not None:
        out["w_sup"] = pd.to_numeric(df[w_sup_col], errors="coerce")
    elif RH_sup_col is not None and T_sup_col is not None:
        RHs = pd.to_numeric(df[RH_sup_col], errors="coerce")/100.0
        out["w_sup"] = [W_from_RH_T(r, t) if pd.notna(r) and pd.notna(t) else np.nan
                        for r, t in zip(RHs, out["T_sup_C"])]
    else:
        raise ValueError("Need 'Humidity ratio at supply' or 'SUPPLY RH' + 'Air temperature at Supply'.")

    # Flows
    m_sup_col = find_col(df, ["mass flow", "supply"])
    m_ret_col = find_col(df, ["mass flow", "extract"])
    out["m_sup"] = pd.to_numeric(df[m_sup_col], errors="coerce") if m_sup_col else np.nan
    out["m_ret"] = pd.to_numeric(df[m_ret_col], errors="coerce") if m_ret_col else np.nan

    # Final cleaning
    out = out.dropna(subset=["time", "w_ret", "w_sup", "m_sup"]).reset_index(drop=True)
    return out

# -------------------------
# LIGHTWEIGHT MODEL (Direct option)
# -------------------------
def ke_from_W0(W0, p_atm=PRESSURE_PA):
    """Slope (kg/s per unit W) multiplying (W - W0) via linearization with C_EVAP*A*F_A."""
    dPa_dW = p_atm * 0.622 / (0.622 + W0)**2
    return C_EVAP * POOL_AREA_M2 * F_A * dPa_dW

def model_event(W0, Wsup_bar, m_vent_bar, Tair_bar, Tw_bar, dt_hours, RH_star=None, p_atm=PRESSURE_PA):
    """Return dict with tau, Winf, Wend, dW_ach, Qlatent for one event (Direct inputs)."""
    k_e = ke_from_W0(W0, p_atm)
    m_air = rho_da_from_wT(W0, Tair_bar, p_atm) * VOLUME_M3
    # estimate m_dot0 at start from pressure gap (optional; cancels out in k_e? kept for completeness)
    Pw = psat_pa(Tw_bar)
    Pa0 = Pv_from_W(W0, p_atm)
    m_dot0 = C_EVAP * POOL_AREA_M2 * (Pw - Pa0) * F_A

    tau = m_air / (m_vent_bar + k_e) if (m_vent_bar + k_e) > 0 else np.inf
    Winf = (m_dot0 + k_e*W0 + m_vent_bar*Wsup_bar) / (m_vent_bar + k_e) if (m_vent_bar + k_e) > 0 else W0
    # response during window
    Wt = Winf + (W0 - Winf) * math.exp(-(dt_hours*3600.0) / tau) if np.isfinite(tau) else W0

    # cap by target if provided (relaxing upward assumed)
    if RH_star is not None:
        Wstar = W_from_RH_T(RH_star, Tair_bar, p_atm)
        Wend = min(Wt, Wstar)
    else:
        Wend = Wt

    dW_ach = Wend - W0
    Qlatent = h_fg_J_per_kg(Tair_bar) * m_air * dW_ach / 3.6e6  # kWh
    return dict(k_e=k_e, m_air=m_air, tau=tau, Winf=Winf, Wend=Wend, dW_ach=dW_ach, Qlatent_kWh=Qlatent)

# -------------------------
# EVENT GENERATION
# -------------------------
def build_events(df):
    if EXPLICIT_EVENTS:
        starts = pd.to_datetime(EXPLICIT_EVENTS)
    else:
        # one event per date at START_HOUR
        starts = []
        for d, grp in df.groupby("date"):
            row = grp[grp["hour"] == START_HOUR]
            if not row.empty:
                starts.append(row.iloc[0]["time"])
    events = []
    for t0 in starts:
        t1 = t0 + pd.Timedelta(hours=DURATION_H)
        events.append((t0, t1))
    return events

# -------------------------
# MAIN
# -------------------------
# ---- plotting toggle ----
SHOW_PLOTS = True        # set False if you don’t want pop-ups

def main():
    df = load_single_run(INPUT_CSV)
    events = build_events(df)
    if not events:
        raise SystemExit("No events found (check START_HOUR / EXPLICIT_EVENTS and the timestamps).")

    rows = []
    for (t0, t1) in events:
        win = df[(df["time"] >= t0) & (df["time"] <= t1)]
        if len(win) < 2:
            continue

        # window averages (Direct option)
        W0        = df.loc[df["time"]==t0, "w_ret"].iloc[0]
        Tair_bar  = float(win["T_room_C"].mean())
        Tw_bar    = POOL_WATER_T_C
        Wsup_bar  = float(win["w_sup"].mean())
        mvent_bar = float(win["m_sup"].mean())  # treat as dry-air flow

        # model prediction
        mod = model_event(W0, Wsup_bar, mvent_bar, Tair_bar, Tw_bar,
                          (t1-t0).total_seconds()/3600.0,
                          RH_star=RELAXED_RH_STAR, p_atm=PRESSURE_PA)

        # "truth" from IDA ICE over the same window (apply same cap)
        W_end_ice = float(win["w_ret"].iloc[-1])
        if RELAXED_RH_STAR is not None:
            W_star = W_from_RH_T(RELAXED_RH_STAR, Tair_bar, PRESSURE_PA)
            W_end_ice = min(W_end_ice, W_star)
        dW_ice = W_end_ice - W0
        m_air  = mod["m_air"]  # use same air mass
        Q_ice  = h_fg_J_per_kg(Tair_bar) * m_air * dW_ice / 3.6e6

        rows.append({
            "t0": t0, "t1": t1, "dt_h": (t1-t0).total_seconds()/3600.0,
            "W0": W0, "Tair_bar_C": Tair_bar, "Wsup_bar": Wsup_bar, "mvent_bar_kgps": mvent_bar,
            "tau_model_min": mod["tau"]/60.0 if np.isfinite(mod["tau"]) else np.nan,
            "Winf_model": mod["Winf"],
            "Wend_model": mod["Wend"],
            "dW_model": mod["dW_ach"],
            "Q_latent_model_kWh": mod["Qlatent_kWh"],
            "Wend_IDA": W_end_ice,
            "dW_IDA": dW_ice,
            "Q_latent_IDA_kWh": Q_ice
        })

    out = pd.DataFrame(rows)
    if out.empty:
        print("No valid events produced. Check timestamps and column parsing.")
        return

    # Errors for summary/plots
    out["dW_error"]    = out["dW_model"] - out["dW_IDA"]
    out["Q_error_kWh"] = out["Q_latent_model_kWh"] - out["Q_latent_IDA_kWh"]
    out["Q_rel_err_%"] = 100.0 * out["Q_error_kWh"] / (out["Q_latent_IDA_kWh"].abs() + 1e-9)

    # Keep your CSV outputs (unchanged file names)
    out.to_csv(OUTPUT_CSV, index=False)
    out.to_csv(OUTPUT_PER_NIGHT, index=False)

    print(f"Events processed: {len(out)}")
    print(f"Mean |ΔW error|: {out['dW_error'].abs().mean():.4e}")
    print(f"Mean |Q error| (kWh): {out['Q_error_kWh'].abs().mean():.3f}")

    # ---------- Helpers to convert W <-> RH ----------
    def RH_from_W_T(W, T_C, p_atm=PRESSURE_PA):
        Pv = Pv_from_W(W, p_atm)
        return Pv / psat_pa(T_C)

    # ---------- Event RH series & monthly average ----------
    BIN_MIN = 10  # averaging bin width (minutes)

    # Collect all events’ RH time series (IDA & model) as (rel_time_min, RH) pairs
    rh_records = []  # list of dicts: {month, rel_min, RH_ida, RH_model}

    for idx, r in out.iterrows():
        t0 = pd.to_datetime(r["t0"]);
        t1 = pd.to_datetime(r["t1"])
        win = df[(df["time"] >= t0) & (df["time"] <= t1)].copy()
        if win.empty:
            continue

        # relative time in minutes
        rel_min = (win["time"] - t0).dt.total_seconds().values / 60.0
        # IDA RH from room W and T at each timestamp
        RH_ida = [RH_from_W_T(w, T) for w, T in zip(win["w_ret"].values, win["T_room_C"].values)]

        # Model W(t) at these timestamps
        tau_s = float(r["tau_model_min"]) * 60.0 if np.isfinite(r["tau_model_min"]) else 1e12
        Winf = float(r["Winf_model"])
        W0 = float(r["W0"])
        W_model = Winf + (W0 - Winf) * np.exp(- (rel_min * 60.0) / tau_s)

        # Cap by relaxed target (optional)
        if RELAXED_RH_STAR is not None:
            Wstar = [W_from_RH_T(RELAXED_RH_STAR, T, PRESSURE_PA) for T in win["T_room_C"].values]
            W_model = np.minimum(W_model, np.array(Wstar))

        # Convert model W(t) to RH using the instantaneous room T
        RH_model = [RH_from_W_T(w, T) for w, T in zip(W_model, win["T_room_C"].values)]

        # Store per-point with month tag
        month_tag = pd.Timestamp(t0).strftime("%Y-%m")
        for rm, ri, tt in zip(RH_model, RH_ida, rel_min):
            rh_records.append({"month": month_tag,
                               "rel_min": tt,
                               "RH_model": rm,
                               "RH_ida": ri})

    rh_df = pd.DataFrame(rh_records)
    if not rh_df.empty and SHOW_PLOTS:
        # Bin by relative minutes
        rh_df["bin"] = (rh_df["rel_min"] // BIN_MIN) * BIN_MIN
        for mon, grp in rh_df.groupby("month"):
            g = grp.groupby("bin").agg(
                RH_model_mean=("RH_model", "mean"),
                RH_ida_mean=("RH_ida", "mean"),
                n=("RH_model", "count")
            ).reset_index()

            # Plot monthly averaged RH trajectory
            fig, ax = plt.subplots()
            ax.plot(g["bin"], 100.0 * g["RH_ida_mean"], label=f"IDA ICE (avg of {g['n'].sum()} pts)", lw=2)
            ax.plot(g["bin"], 100.0 * g["RH_model_mean"], "--", label="Model (avg)", lw=2)
            ax.set_xlabel(f"Minutes from event start (bin={BIN_MIN} min)")
            ax.set_ylabel("Relative Humidity (%)")
            ax.set_title(f"Monthly average RH trajectory — {mon}")
            ax.grid(True);
            ax.legend()

    # ---------- Change the three representative time-series to RH ----------
    if SHOW_PLOTS:
        idx_sorted = out["Q_error_kWh"].abs().sort_values().index.to_list()
        picks = []
        if len(idx_sorted) >= 1: picks.append(idx_sorted[-1])  # worst
        if len(idx_sorted) >= 2: picks.append(idx_sorted[len(idx_sorted) // 2])  # median
        if len(idx_sorted) >= 3: picks.append(idx_sorted[0])  # best

        for idx in picks:
            r = out.loc[idx]
            t0, t1 = pd.to_datetime(r["t0"]), pd.to_datetime(r["t1"])
            win = df[(df["time"] >= t0) & (df["time"] <= t1)].copy()
            if win.empty:
                continue

            rel_min = (win["time"] - t0).dt.total_seconds().values / 60.0
            RH_ida = [RH_from_W_T(w, T) for w, T in zip(win["w_ret"].values, win["T_room_C"].values)]

            tau_s = float(r["tau_model_min"]) * 60.0 if np.isfinite(r["tau_model_min"]) else 1e12
            Winf = float(r["Winf_model"]);
            W0 = float(r["W0"])
            W_model = Winf + (W0 - Winf) * np.exp(- (rel_min * 60.0) / tau_s)
            if RELAXED_RH_STAR is not None:
                Wstar = [W_from_RH_T(RELAXED_RH_STAR, T, PRESSURE_PA) for T in win["T_room_C"].values]
                W_model = np.minimum(W_model, np.array(Wstar))
            RH_model = [RH_from_W_T(w, T) for w, T in zip(W_model, win["T_room_C"].values)]

            fig, ax = plt.subplots()
            ax.plot(win["time"], 100.0 * np.array(RH_ida), label="IDA ICE (RH)", lw=2)
            ax.plot(win["time"], 100.0 * np.array(RH_model), "--", label="Model RH(t)", lw=2)
            ax.set_xlabel("Time");
            ax.set_ylabel("RH (%)")
            title = f"Event {t0:%Y-%m-%d %H:%M} → {t1:%H:%M} | τ={r['tau_model_min']:.1f} min,  W∞={r['Winf_model']:.5f}"
            ax.set_title(title);
            ax.grid(True);
            ax.legend()

    if SHOW_PLOTS:
        # ---------- 1) Scatter: ΔW ----------
        fig, ax = plt.subplots()
        ax.scatter(out["dW_IDA"], out["dW_model"], alpha=0.8)
        xy = np.concatenate([out["dW_IDA"].values, out["dW_model"].values])
        lo, hi = np.nanmin(xy), np.nanmax(xy)
        pad = 0.05 * (hi - lo + 1e-12)
        ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], "k--", lw=1)
        ax.set_xlabel("ΔW from IDA ICE (kg/kg)")
        ax.set_ylabel("ΔW from model (kg/kg)")
        ax.set_title("Event ΔW: model vs IDA ICE (1:1 dashed)")
        ax.grid(True)

        # ---------- 2) Scatter: Q_latent ----------
        fig, ax = plt.subplots()
        ax.scatter(out["Q_latent_IDA_kWh"], out["Q_latent_model_kWh"], alpha=0.8)
        xy = np.concatenate([out["Q_latent_IDA_kWh"].values, out["Q_latent_model_kWh"].values])
        lo, hi = np.nanmin(xy), np.nanmax(xy)
        pad = 0.05 * (hi - lo + 1e-12)
        ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], "k--", lw=1)
        rmse = np.sqrt(np.nanmean((out["Q_error_kWh"])**2))
        ax.set_xlabel("Q_latent from IDA ICE (kWh)")
        ax.set_ylabel("Q_latent from model (kWh)")
        ax.set_title(f"Event Q_latent: model vs IDA ICE (1:1 dashed) — RMSE={rmse:.3f} kWh")
        ax.grid(True)

        # ---------- 3) Histogram: relative error (Q) ----------
        fig, ax = plt.subplots()
        clean_rel = out["Q_rel_err_%"].replace([np.inf, -np.inf], np.nan).dropna()
        ax.hist(clean_rel, bins=20, alpha=0.85)
        ax.set_xlabel("Relative error in Q_latent (%) = 100*(model−IDA)/|IDA|")
        ax.set_ylabel("Count")
        ax.set_title("Relative error distribution (Q_latent)")
        ax.grid(True)

        # ---------- 4) Time-series for 2–3 representative events ----------
        # pick worst, median, best by |Q_error|
        idx_sorted = out["Q_error_kWh"].abs().sort_values().index.to_list()
        picks = []
        if len(idx_sorted) >= 1: picks.append(idx_sorted[-1])            # worst
        if len(idx_sorted) >= 2: picks.append(idx_sorted[len(idx_sorted)//2])  # median
        if len(idx_sorted) >= 3: picks.append(idx_sorted[0])            # best

        for idx in picks:
            r = out.loc[idx]
            t0, t1 = pd.to_datetime(r["t0"]), pd.to_datetime(r["t1"])
            win = df[(df["time"] >= t0) & (df["time"] <= t1)].copy()
            if win.empty:
                continue

            # model curve across the window
            tau_s = float(r["tau_model_min"]) * 60.0 if not math.isnan(r["tau_model_min"]) else np.inf
            Winf  = float(r["Winf_model"])
            W0    = float(r["W0"])
            tsec  = (win["time"] - t0).dt.total_seconds().values
            Wmodel = Winf + (W0 - Winf) * np.exp(-tsec / (tau_s if np.isfinite(tau_s) else 1e12))

            # apply target cap (relaxing upward)
            if RELAXED_RH_STAR is not None:
                Tair_bar = float(win["T_room_C"].mean())
                Wstar = W_from_RH_T(RELAXED_RH_STAR, Tair_bar, PRESSURE_PA)
                Wmodel = np.minimum(Wmodel, Wstar)

            fig, ax = plt.subplots()
            ax.plot(win["time"], win["w_ret"], label="IDA ICE (room W)", lw=2)
            ax.plot(win["time"], Wmodel, label="Model W(t)", lw=2, linestyle="--")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %H:%M"))
            ax.set_xlabel("Time")
            ax.set_ylabel("Humidity ratio W (kg/kg)")
            title = f"Event {t0.strftime('%Y-%m-%d %H:%M')} → {t1.strftime('%H:%M')} | τ={r['tau_model_min']:.1f} min,  W∞={Winf:.5f}"
            ax.set_title(title)
            ax.grid(True)
            ax.legend()

        plt.show()

if __name__ == "__main__":
    main()