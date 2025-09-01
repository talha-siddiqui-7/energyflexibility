#!/usr/bin/env python3
"""
Validation script (single-run, event windows)
Adds:
- Nonlinear evaporation + time-varying W_sup(t), m_vent(t) toggles
- Per-event fit of tau_fit, Winf_fit and ke_fit from IDA ICE trace
- PRINTS average total latent energy storage per day (IDA vs Model)
- STABLE time-varying simulation:
    * exact interval update for linearized mode
    * adaptive sub-stepping + physical clamps for nonlinear mode
"""

import pandas as pd, numpy as np, math, os
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ------------------------- PATHS (yours kept) -------------------------
INPUT_CSV  = r"M:\PhD\02 Data sets from simulations\Air Flexibility\Winter_month.csv"
OUTPUT_CSV = r"M:\PhD\02 Data sets from simulations\Air Flexibility\true_storage_01to04.csv"
OUTPUT_HOURLY    = r"M:\PhD\02 Data sets from simulations\Air Flexibility\true_storage_hourly.csv"
OUTPUT_PNG       = r"M:\PhD\02 Data sets from simulations\Air Flexibility\true_storage_hourly.png"
OUTPUT_PER_NIGHT = r"M:\PhD\02 Data sets from simulations\Air Flexibility\true_storage_01to04.csv"

# ------------------------- SITE / MODEL SETTINGS -------------------------
VOLUME_M3       = 19.65 * 13.8 * 4.95
PRESSURE_PA     = 101325.0
RELAXED_RH_STAR = 0.58
POOL_WATER_T_C  = 28.0
C_EVAP = 4.0e-8          # kg/(s·m²·Pa)  (calibrate once on a clean event)
F_A    = 0.5
POOL_AREA_M2 = 100.0

# Event definition
START_HOUR  = 20
DURATION_H  = 6
EXPLICIT_EVENTS = []

# ------------------------- TOGGLES -------------------------
USE_TIME_VARYING = True    # use W_sup(t), m_vent(t) within window
USE_NONLINEAR    = True    # exact evaporation law (else linearized)
USE_AVG_FLOW     = False   # m_vent = 0.5*(m_sup+m_ext) else m_sup
PLOTS_SHOW       = True

# ---- Stability controls for nonlinear path ----
MAX_SUBSTEP_S    = 60.0    # cap substep to 60 s
CLAMP_ENABLED    = True    # clamp 0 <= W <= W_sat(T) each substep

# ------------------------- Psychro helpers -------------------------
def psat_pa(T_C: float) -> float:
    T = float(T_C)
    return 610.94 * math.exp(17.625 * T / (T + 243.04))

def W_from_RH_T(RH: float, T_C: float, p_atm: float = PRESSURE_PA) -> float:
    Ps = psat_pa(T_C)
    Pv = RH * Ps
    return 0.622 * Pv / (p_atm - Pv)

def Pv_from_W(W: float, p_atm: float = PRESSURE_PA) -> float:
    return p_atm * (W / (0.622 + W))

def rho_da_from_wT(w: float, T_C: float, p: float = PRESSURE_PA) -> float:
    w = float(w); T_C = float(T_C)
    pv = p * (w / (0.62198 + w))
    p_da = p - pv
    return p_da / (287.05 * (T_C + 273.15))

def h_fg_J_per_kg(T_C: float) -> float:
    return 2.501e6 - 2361.0 * float(T_C)

def RH_from_W_T(W, T_C, p_atm=PRESSURE_PA):
    return Pv_from_W(W, p_atm) / psat_pa(T_C)

def W_sat_from_T(T_C: float, p_atm: float = PRESSURE_PA) -> float:
    Ps = psat_pa(T_C)
    Ps = min(Ps, 0.99*p_atm)  # safety
    return 0.622 * Ps / (p_atm - Ps)

def clamp_W(W: float, T_C: float) -> float:
    if not CLAMP_ENABLED:
        return W
    return max(0.0, min(W, W_sat_from_T(T_C)))

# ------------------------- CSV loader (robust to 2-row header) -------------------------
def load_single_run(path):
    df = pd.read_csv(path)

    def flatten_two_row_header(df0):
        try:
            row0 = df0.iloc[0].astype(str).str.strip().str.lower()
        except Exception:
            return df0
        if row0.str.contains("hour").any() and row0.str.contains("variables").any():
            new_cols = [str(df0.iloc[0, 0]).strip() or "Hour"]
            for j in range(1, df0.shape[1]):
                name = str(df0.iloc[1, j]).strip()
                new_cols.append(name if name else f"col{j}")
            df1 = df0.iloc[2:].copy()
            df1.columns = new_cols
            return df1
        return df0

    df = flatten_two_row_header(df)

    def find_col(frame, tokens):
        toks = [t.lower() for t in tokens]
        for c in frame.columns:
            if all(t in str(c).lower() for t in toks):
                return c
        return None

    # time
    time_col = find_col(df, ["hour"]) or find_col(df, ["time"])
    parsed_time = None
    if time_col is not None:
        parsed_time = pd.to_datetime(df[time_col], errors="coerce", dayfirst=True)
    if parsed_time is None or parsed_time.notna().sum() == 0:
        best = None; best_series = None; best_frac = 0.0
        for c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
            frac = s.notna().mean()
            if frac > 0.7 and frac > best_frac:
                best, best_series, best_frac = c, s, frac
        if best is None:
            raise ValueError(f"No time-like column found. Columns: {list(df.columns)}")
        parsed_time = best_series

    out = pd.DataFrame({
        "time": parsed_time,
        "date": parsed_time.dt.date,
        "hour": parsed_time.dt.hour
    })

    # temps
    T_room_col = find_col(df, ["air", "extract"]) or find_col(df, ["room"])
    T_sup_col  = find_col(df, ["air temperature at supply"]) or find_col(df, ["supply", "deg"])
    out["T_room_C"] = pd.to_numeric(df[T_room_col], errors="coerce") if T_room_col else np.nan
    out["T_sup_C"]  = pd.to_numeric(df[T_sup_col],  errors="coerce") if T_sup_col  else np.nan

    # W room
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

    # W supply
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

    # flows
    m_sup_col = find_col(df, ["mass flow", "supply"])
    m_ret_col = find_col(df, ["mass flow", "extract"])
    out["m_sup"] = pd.to_numeric(df[m_sup_col], errors="coerce") if m_sup_col else np.nan
    out["m_ret"] = pd.to_numeric(df[m_ret_col], errors="coerce") if m_ret_col else np.nan

    return out.dropna(subset=["time", "w_ret", "w_sup", "m_sup"]).reset_index(drop=True)

# ------------------------- Linearized (for reference / constants) -------------------------
def ke_from_W0(W0, p_atm=PRESSURE_PA):
    dPa_dW = p_atm * 0.622 / (0.622 + W0)**2
    return C_EVAP * POOL_AREA_M2 * F_A * dPa_dW

def model_event_linear_avg(W0, Wsup_bar, m_vent_bar, Tair_bar, Tw_bar, dt_h, RH_star=None, p_atm=PRESSURE_PA):
    k_e = ke_from_W0(W0, p_atm)
    m_air = rho_da_from_wT(W0, Tair_bar, p_atm) * VOLUME_M3
    Pw = psat_pa(Tw_bar); Pa0 = Pv_from_W(W0, p_atm)
    m_dot0 = C_EVAP * POOL_AREA_M2 * (Pw - Pa0) * F_A
    tau = m_air / (m_vent_bar + k_e) if (m_vent_bar + k_e) > 0 else np.inf
    Winf = (m_dot0 + k_e*W0 + m_vent_bar*Wsup_bar) / (m_vent_bar + k_e) if (m_vent_bar + k_e) > 0 else W0
    Wt = Winf + (W0 - Winf) * math.exp(-(dt_h*3600.0) / tau) if np.isfinite(tau) else W0
    Wstar = W_from_RH_T(RH_star, Tair_bar, p_atm) if RH_star is not None else np.inf
    Wend = min(Wt, Wstar)
    dW = Wend - W0
    Q = h_fg_J_per_kg(Tair_bar) * m_air * dW / 3.6e6
    return {"k_e":k_e,"m_air":m_air,"tau":tau,"Winf":Winf,"Wend":Wend,"dW_ach":dW,"Qlatent_kWh":Q}

# ------------------------- STABLE time-varying simulator -------------------------
def simulate_event_timevary(win: pd.DataFrame, W0: float, Tair_bar: float, Tw_bar: float,
                            RH_star: float|None, use_nonlinear=True, use_avg_flow=False,
                            p_atm: float = PRESSURE_PA):
    """
    Stable integration of:
        dW/dt = [ m_evap(W) + m_vent(t)*(W_sup(t) - W) ] / m_air
    If use_nonlinear=False: exact per-interval update for linearized source.
    If use_nonlinear=True : adaptive sub-stepping + clamps.
    """
    # air mass constant over window
    m_air = rho_da_from_wT(W0, Tair_bar, p_atm) * VOLUME_M3

    # choose flow timeline
    if use_avg_flow and "m_ret" in win:
        mvent = 0.5*(win["m_sup"].values + win["m_ret"].values)
    else:
        mvent = win["m_sup"].values
    wsup  = win["w_sup"].values
    tair  = win["T_room_C"].values
    tsec  = (win["time"] - win["time"].iloc[0]).dt.total_seconds().values
    dt_iv = np.diff(tsec, prepend=tsec[0])

    Pw  = psat_pa(Tw_bar)
    Pa0 = Pv_from_W(W0, p_atm)
    m0  = C_EVAP * POOL_AREA_M2 * (Pw - Pa0) * F_A
    k_e0 = ke_from_W0(W0, p_atm)

    W = float(W0)
    W_series = [W0]

    for i in range(1, len(tsec)):
        Δt = float(dt_iv[i])
        if Δt <= 0:
            W_series.append(W); continue

        if not use_nonlinear:
            # ----- exact interval update for linearized source -----
            denom = (mvent[i] + k_e0)
            if denom <= 0:
                W_series.append(W); continue
            τk   = m_air / denom
            Winf = (m0 + k_e0*W0 + mvent[i]*wsup[i]) / denom
            Wnew = Winf + (W - Winf) * math.exp(-Δt/τk)
            # cap to relaxed target (relax-up)
            if RH_star is not None:
                Wstar = W_from_RH_T(RH_star, tair[i], p_atm)
                Wnew  = min(Wnew, Wstar)
            Wnew = clamp_W(Wnew, tair[i])
            W = Wnew
            W_series.append(W)
            continue

        # ----- nonlinear evaporation: adaptive sub-step + clamps -----
        # local linear slope for tau estimate
        dPa_dW = p_atm * 0.622 / (0.622 + W)**2
        k_e_loc = C_EVAP * POOL_AREA_M2 * F_A * dPa_dW
        denom = (mvent[i] + k_e_loc)
        τloc = m_air/denom if denom > 1e-12 else np.inf
        dt_target = min(MAX_SUBSTEP_S, (τloc/5.0) if np.isfinite(τloc) else MAX_SUBSTEP_S, Δt)
        n_sub = max(1, int(math.ceil(Δt / dt_target)))
        dt_sub = Δt / n_sub

        for _ in range(n_sub):
            Pa    = Pv_from_W(W, p_atm)
            m_ev  = C_EVAP * POOL_AREA_M2 * (Pw - Pa) * F_A
            m_sink= mvent[i] * (wsup[i] - W)
            dWdt  = (m_ev + m_sink) / m_air
            W     = W + dWdt * dt_sub
            # cap upward & clamp physical range
            if RH_star is not None:
                Wstar = W_from_RH_T(RH_star, tair[i], p_atm)
                W = min(W, Wstar)
            W = clamp_W(W, tair[i])

        W_series.append(W)

    return np.array(tsec), np.array(W_series), m_air

# ------------------------- Events -------------------------
def build_events(df):
    if EXPLICIT_EVENTS:
        starts = pd.to_datetime(EXPLICIT_EVENTS)
    else:
        starts = []
        for _, grp in df.groupby("date"):
            row = grp[grp["hour"] == START_HOUR]
            if not row.empty:
                starts.append(row.iloc[0]["time"])
    return [(t0, t0 + pd.Timedelta(hours=DURATION_H)) for t0 in starts]

# ------------------------- Fit tau, Winf, ke from IDA trace -------------------------
def fit_tau_Winf_from_trace(times: pd.Series, W: pd.Series):
    """
    Simple log-linear fit assuming W(t) ~ Winf + (W0-Winf) exp(-t/tau).
    We set Winf_guess = average of last third of samples, then linearize.
    Returns tau_fit [s] and Winf_fit; NaN if insufficient variation.
    """
    t = (times - times.iloc[0]).dt.total_seconds().values
    Wv = W.values.astype(float)
    if len(t) < 4:
        return np.nan, np.nan
    Winf_guess = np.nanmean(Wv[max(1,len(Wv)*2//3):])
    num = (Wv - Winf_guess)*(Wv[0]-Winf_guess)
    valid = (num > 0) & np.isfinite(t) & np.isfinite(Wv)
    if valid.sum() < 3:
        return np.nan, np.nan
    y = np.log( (Wv[valid] - Winf_guess) / (Wv[0] - Winf_guess) )
    x = t[valid]
    A = np.vstack([x, np.ones_like(x)]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    tau_fit = -1.0/slope if slope < 0 else np.nan
    return tau_fit, Winf_guess

# ------------------------- MAIN -------------------------
def main():
    df = load_single_run(INPUT_CSV)
    events = build_events(df)
    if not events:
        raise SystemExit("No events found (check START_HOUR / EXPLICIT_EVENTS and timestamps).")

    rows = []
    for (t0, t1) in events:
        win = df[(df["time"] >= t0) & (df["time"] <= t1)].copy()
        if len(win) < 2:
            continue

        W0        = float(df.loc[df["time"]==t0, "w_ret"].iloc[0])
        Tair_bar  = float(win["T_room_C"].mean())
        Tw_bar    = POOL_WATER_T_C

        if USE_AVG_FLOW and "m_ret" in win:
            mvent_bar = float(0.5*(win["m_sup"]+win["m_ret"]).mean())
        else:
            mvent_bar = float(win["m_sup"].mean())
        Wsup_bar  = float(win["w_sup"].mean())
        dt_h = (t1-t0).total_seconds()/3600.0

        # --- Model prediction (respect toggles) ---
        if USE_TIME_VARYING:
            tsec, W_series, m_air = simulate_event_timevary(
                win, W0, Tair_bar, Tw_bar, RELAXED_RH_STAR,
                use_nonlinear=USE_NONLINEAR, use_avg_flow=USE_AVG_FLOW, p_atm=PRESSURE_PA
            )
            Wend_model = float(W_series[-1])
            dW_model   = Wend_model - W0
            Q_model    = h_fg_J_per_kg(Tair_bar) * m_air * dW_model / 3.6e6
            k_e = ke_from_W0(W0)
            tau_eff = m_air / (mvent_bar + k_e) if (mvent_bar + k_e) > 0 else np.inf
            Winf_model = np.nan
        else:
            res = model_event_linear_avg(W0, Wsup_bar, mvent_bar, Tair_bar, Tw_bar, dt_h,
                                         RH_star=RELAXED_RH_STAR, p_atm=PRESSURE_PA)
            k_e = res["k_e"]; m_air = res["m_air"]
            tau_eff = res["tau"]; Winf_model = res["Winf"]
            Wend_model = res["Wend"]; dW_model = res["dW_ach"]; Q_model = res["Qlatent_kWh"]

        # --- "Truth" from IDA over the same window (with same cap) ---
        W_end_ice = float(win["w_ret"].iloc[-1])
        if RELAXED_RH_STAR is not None:
            W_star = W_from_RH_T(RELAXED_RH_STAR, Tair_bar, PRESSURE_PA)
            W_end_ice = min(W_end_ice, W_star)
        dW_ice = W_end_ice - W0
        Q_ice  = h_fg_J_per_kg(Tair_bar) * m_air * dW_ice / 3.6e6

        tau_fit_s, Winf_fit = fit_tau_Winf_from_trace(win["time"], win["w_ret"])
        ke_fit = (m_air / tau_fit_s) - mvent_bar if np.isfinite(tau_fit_s) else np.nan

        rows.append({
            "t0": t0, "t1": t1, "dt_h": dt_h,
            "USE_TIME_VARYING": USE_TIME_VARYING, "USE_NONLINEAR": USE_NONLINEAR,
            "W0": W0, "Tair_bar_C": Tair_bar, "Wsup_bar": Wsup_bar,
            "mvent_bar_kgps": mvent_bar, "m_air_kg": m_air,
            "k_e_model": k_e, "tau_model_min": (tau_eff/60.0) if np.isfinite(tau_eff) else np.nan,
            "Winf_model": Winf_model, "Wend_model": Wend_model,
            "dW_model": dW_model, "Q_latent_model_kWh": Q_model,
            "Wend_IDA": W_end_ice, "dW_IDA": dW_ice, "Q_latent_IDA_kWh": Q_ice,
            "tau_fit_min": (tau_fit_s/60.0) if np.isfinite(tau_fit_s) else np.nan,
            "Winf_fit": Winf_fit, "k_e_fit": ke_fit
        })

    out = pd.DataFrame(rows)
    if out.empty:
        print("No valid events produced. Check timestamps and column parsing.")
        return

    # errors
    out["dW_error"]     = out["dW_model"] - out["dW_IDA"]
    out["Q_error_kWh"]  = out["Q_latent_model_kWh"] - out["Q_latent_IDA_kWh"]
    out["Q_rel_err_%"]  = 100.0 * out["Q_error_kWh"] / (out["Q_latent_IDA_kWh"].abs() + 1e-9)

    out.to_csv(OUTPUT_CSV, index=False)
    out.to_csv(OUTPUT_PER_NIGHT, index=False)

    print(f"Events processed: {len(out)}")
    print(f"Mean |ΔW error|: {out['dW_error'].abs().mean():.4e}")
    print(f"Mean |Q error| (kWh): {out['Q_error_kWh'].abs().mean():.3f}")
    print(f"Median tau_fit (min): {np.nanmedian(out['tau_fit_min']):.2f} | Median k_e_fit: {np.nanmedian(out['k_e_fit']):.3e}")

    # ---- Average total latent storage per day (IDA vs Model) ----
    out["day"] = pd.to_datetime(out["t0"]).dt.date
    daily = out.groupby("day").agg(
        Q_IDA_day_kWh=("Q_latent_IDA_kWh", "sum"),
        Q_model_day_kWh=("Q_latent_model_kWh", "sum"),
        n_events=("t0", "count")
    ).reset_index()
    avg_ida_day   = float(daily["Q_IDA_day_kWh"].mean()) if not daily.empty else float("nan")
    avg_model_day = float(daily["Q_model_day_kWh"].mean()) if not daily.empty else float("nan")
    print(f"Days covered: {len(daily)} (avg events/day = {daily['n_events'].mean():.2f})")
    print(f"Average total latent storage per day — IDA ICE: {avg_ida_day:.3f} kWh/day")
    print(f"Average total latent storage per day — Model  : {avg_model_day:.3f} kWh/day")

    if not PLOTS_SHOW:
        return

    # ---- Average RH across the event window (pooled) ----
    BIN_MIN = 10
    avg_records = []
    for _, r in out.iterrows():
        t0, t1 = pd.to_datetime(r["t0"]), pd.to_datetime(r["t1"])
        win = df[(df["time"] >= t0) & (df["time"] <= t1)].copy()
        if win.empty:
            continue
        rel_min = (win["time"] - t0).dt.total_seconds().values / 60.0

        RH_ida = [100.0 * RH_from_W_T(w, T) for w, T in zip(win["w_ret"].values, win["T_room_C"].values)]
        if USE_TIME_VARYING:
            tsec, W_series, _ = simulate_event_timevary(
                win, r["W0"], r["Tair_bar_C"], POOL_WATER_T_C, RELAXED_RH_STAR,
                use_nonlinear=USE_NONLINEAR, use_avg_flow=USE_AVG_FLOW, p_atm=PRESSURE_PA
            )
            RH_model = [100.0 * RH_from_W_T(w, T) for w, T in zip(W_series, win["T_room_C"].values)]
        else:
            tau_s = float(r["tau_model_min"]) * 60.0 if np.isfinite(r["tau_model_min"]) else 1e12
            Winf  = float(r["Winf_model"]); W0 = float(r["W0"])
            W_model = Winf + (W0 - Winf) * np.exp(- (rel_min * 60.0) / tau_s)
            if RELAXED_RH_STAR is not None:
                Wstar = [W_from_RH_T(RELAXED_RH_STAR, T, PRESSURE_PA) for T in win["T_room_C"].values]
                W_model = np.minimum(W_model, np.array(Wstar))
            RH_model = [100.0 * RH_from_W_T(w, T) for w, T in zip(W_model, win["T_room_C"].values)]

        bins = (rel_min // BIN_MIN).astype(int) * BIN_MIN
        for b, rhi, rhm in zip(bins, RH_ida, RH_model):
            avg_records.append({"bin": int(b), "RH_ida": float(rhi), "RH_model": float(rhm)})

    if avg_records:
        avg_df = pd.DataFrame(avg_records)
        g = avg_df.groupby("bin").agg(
            RH_ida_mean=("RH_ida", "mean"),
            RH_model_mean=("RH_model", "mean"),
            RH_ida_p25=("RH_ida", lambda s: np.percentile(s, 25)),
            RH_ida_p75=("RH_ida", lambda s: np.percentile(s, 75)),
            RH_model_p25=("RH_model", lambda s: np.percentile(s, 25)),
            RH_model_p75=("RH_model", lambda s: np.percentile(s, 75)),
            n=("RH_ida", "count")
        ).reset_index()

        fig, ax = plt.subplots()
        ax.plot(g["bin"], g["RH_ida_mean"], lw=2, label=f"IDA ICE (mean, n={int(g['n'].sum())})")
        ax.fill_between(g["bin"], g["RH_ida_p25"], g["RH_ida_p75"], alpha=0.15, label="IDA IQR (25–75%)")
        ax.plot(g["bin"], g["RH_model_mean"], "--", lw=2, label="Model (mean)")
        ax.fill_between(g["bin"], g["RH_model_p25"], g["RH_model_p75"], alpha=0.15, label="Model IQR (25–75%)")
        ax.set_xlabel(f"Minutes from event start (bin = {BIN_MIN} min)")
        ax.set_ylabel("Relative Humidity (%)")
        ax.set_title("Average RH across the event window — IDA vs Model")
        ax.grid(True); ax.legend()
        plt.show()
    else:
        print("Average RH plot: no records were collected (check events/windows).")

    # ---- scatter ΔW ----
    fig, ax = plt.subplots()
    ax.scatter(out["dW_IDA"], out["dW_model"], alpha=0.85)
    xy = np.concatenate([out["dW_IDA"].values, out["dW_model"].values])
    lo, hi = np.nanmin(xy), np.nanmax(xy); pad = 0.05*(hi-lo+1e-12)
    ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], "k--", lw=1)
    ax.set_xlabel("ΔW from IDA ICE (kg/kg)"); ax.set_ylabel("ΔW from model (kg/kg)")
    ax.set_title("Event ΔW: model vs IDA ICE (1:1 dashed)"); ax.grid(True)

    # ---- scatter Q ----
    fig, ax = plt.subplots()
    ax.scatter(out["Q_latent_IDA_kWh"], out["Q_latent_model_kWh"], alpha=0.85)
    xy = np.concatenate([out["Q_latent_IDA_kWh"].values, out["Q_latent_model_kWh"].values])
    lo, hi = np.nanmin(xy), np.nanmax(xy); pad = 0.05*(hi-lo+1e-12)
    ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], "k--", lw=1)
    rmse = np.sqrt(np.nanmean((out["Q_error_kWh"])**2))
    ax.set_xlabel("Q_latent from IDA ICE (kWh)"); ax.set_ylabel("Q_latent from model (kWh)")
    ax.set_title(f"Event Q_latent: model vs IDA ICE — RMSE={rmse:.3f} kWh"); ax.grid(True)

    # ---- histogram rel error ----
    fig, ax = plt.subplots()
    clean_rel = out["Q_rel_err_%"].replace([np.inf, -np.inf], np.nan).dropna()
    ax.hist(clean_rel, bins=20, alpha=0.85)
    ax.set_xlabel("Relative error in Q_latent (%) = 100*(model−IDA)/|IDA|")
    ax.set_ylabel("Count"); ax.set_title("Relative error distribution (Q_latent)"); ax.grid(True)

    # ---- 3 representative time-series in RH ----
    idx_sorted = out["Q_error_kWh"].abs().sort_values().index.to_list()
    picks = []
    if len(idx_sorted) >= 1: picks.append(idx_sorted[-1])
    if len(idx_sorted) >= 2: picks.append(idx_sorted[len(idx_sorted)//2])
    if len(idx_sorted) >= 3: picks.append(idx_sorted[0])

    for idx in picks:
        r = out.loc[idx]
        t0, t1 = pd.to_datetime(r["t0"]), pd.to_datetime(r["t1"])
        win = df[(df["time"] >= t0) & (df["time"] <= t1)].copy()
        if win.empty:
            continue

        if USE_TIME_VARYING:
            tsec, W_series, _ = simulate_event_timevary(
                win, r["W0"], r["Tair_bar_C"], POOL_WATER_T_C, RELAXED_RH_STAR,
                use_nonlinear=USE_NONLINEAR, use_avg_flow=USE_AVG_FLOW
            )
            Wm = W_series
        else:
            tau_s = (r["tau_model_min"]*60.0) if np.isfinite(r["tau_model_min"]) else 1e12
            Winf = r["Winf_model"]; W0 = r["W0"]
            tsec = (win["time"] - win["time"].iloc[0]).dt.total_seconds().values
            Wm = Winf + (W0 - Winf)*np.exp(-tsec/tau_s)

        RH_ida = [100*RH_from_W_T(w, T) for w,T in zip(win["w_ret"].values, win["T_room_C"].values)]
        RH_mod = [100*RH_from_W_T(w, T) for w,T in zip(Wm, win["T_room_C"].values)]

        fig, ax = plt.subplots()
        ax.plot(win["time"], RH_ida, label="IDA ICE (RH)", lw=2)
        ax.plot(win["time"], RH_mod, "--", label="Model RH(t)", lw=2)
        ax.set_xlabel("Time"); ax.set_ylabel("RH (%)"); ax.grid(True)
        ax.set_title(f"Event {t0:%Y-%m-%d %H:%M} → {t1:%H:%M} | τ_fit={r['tau_fit_min']:.1f} min, k_e_fit={r['k_e_fit']:.2e}")
        ax.legend()

    plt.show()

if __name__ == "__main__":
    main()
