#!/usr/bin/env python3
"""
Water Flexibility Validation (Pool Sensible Storage) — model vs simulation

- Loads your IDA ICE export (two-row header or normal header)
- Simulates pool-water temperature during "events" and compares to simulation
- Prints energy/temperature metrics and plots representative events + monthly averages

Additions:
- Heun (trapezoidal) integrator for the dynamic path
- Optional "calibrated" dynamic path with three physical scaling knobs:
    ALPHA_P:   scale on "Pool heater supply, W" to approximate net heat to pool water
    SCALE_UA:  scale on side/bottom conductive UA losses
    SCALE_EVAP: scale on evaporation heat loss
- Toggle USE_CALIBRATED_DYNAMIC to use the improved dynamic path
"""

import pandas as pd, numpy as np, math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ------------------------- PATHS -------------------------
INPUT_CSV = r"M:\PhD\02 Data sets from simulations\WATER FLEXIBILITY\Winter month.csv"   # <--- your original file path

# ------------------------- GEOMETRY & FABRIC (edit as needed) -------------------------
L_POOL = 8.0         # m
W_POOL = 12.5        # m
D_POOL = 2.0         # m

U_SIDE = 2.94        # W/m2K (side walls)
U_BOTTOM = 0.50      # W/m2K (bottom)
AIR_VELOCITY = 0.10  # m/s over water surface
ROOM_DELTA_K = 3.0   # Technical room air ~ T_pool - 1 K

# ------------------------- CONSTANTS -------------------------
C_EVAP = 4.0e-8      # kg/(s·m2·Pa)
F_ACTIVITY = 1.0    # activity factor
P_ATM = 101325.0     # Pa
HFG = 2.43e6         # J/kg
RHO_W = 1000.0       # kg/m3
CP_W = 4186.0        # J/kg/K
NU_AIR = 15.89e-6    # m2/s
K_AIR = 0.025        # W/m/K
PR_AIR = 0.7         # --
H_FIXED_WM2K = 2     # if using fixed convection (see toggle)

# ------------------------- GREY HEAT RECOVERY & MAKEUP -------------------------
GREY_HEAT_ON = False
GREY_HEAT_EFF = 0.60
TAP_WATER_C = 8.0
HYGIENE_L_PER_DAY = 0.0
INCLUDE_MAKEUP = False

# ------------------------- EVENT DEFINITION -------------------------
START_HOUR = 9        # default events start at 03:00 (change to 20 for your 20:00–02:00 case)
DURATION_H = 6        # default 4 hours (change to 6 for your 6-hour events)
EXPLICIT_EVENTS = []  # e.g. ["2025-01-10 20:00", "2025-01-28 20:00"]
PLOTS_SHOW = True

# ------------------------- MODEL TOGGLES -------------------------
USE_TIME_VARYING = True     # dynamic step-by-step vs constant-window closed form
USE_EVAP_EXACT   = False    # exact evap each step; else linearized around T0 (constant path uses mean)
USE_ROOM_TRACK   = True     # room air follows T_pool - 1 K inside window (approx)
USE_CONV_FROM_VEL= False     # if False and H_FIXED_WM2K is set, use the fixed h; else velocity correlation
CAP_TMAX = None             # e.g. 32.0 to cap pool temp

# New toggle: use improved dynamic integrator with calibration
USE_CALIBRATED_DYNAMIC = True

# Calibrated scaling knobs (used only if USE_CALIBRATED_DYNAMIC = True)
ALPHA_P   = 0.90   # scale on heater power (net to pool)
SCALE_UA  = 1.15   # scale on conductive losses (side+bottom)
SCALE_EVAP= 1.10   # scale on evaporation losses

# ========================= PSYCHROMETRICS =========================
def psat_pa(TC: float) -> float:
    return 610.94 * math.exp(17.625 * TC / (TC + 243.04))

def dpsat_dT(TC: float) -> float:
    Ps = psat_pa(TC)
    return Ps * (17.625 * 243.04) / ((TC + 243.04) ** 2)

def Pv_from_RH_Tair(RH_frac: float, T_air_C: float) -> float:
    return RH_frac * psat_pa(T_air_C)

# ------------------------- CSV LOADER (robust for 2-row header) -------------------------
def load_pool_run(csv_path: str) -> pd.DataFrame:
    """
    Robust CSV loader for the pool dataset.
    Handles both:
      A) two-row header like:
           Row0: 'Hour', 'Variables', '', '', ...
           Row1: '', 'Pool heater supply, W', 'Pool water return temp, Deg-C', ...
      B) already flattened header with proper column names.
    """
    # Read raw (no header) so we can inspect the first two rows safely
    raw0 = pd.read_csv(csv_path, header=None, dtype=str, encoding="utf-8-sig")
    r0 = raw0.iloc[0].astype(str).str.strip()
    r1 = raw0.iloc[1].astype(str).str.strip()

    first_cell = r0.iloc[0].lower() if len(r0) else ""
    second_cell = (r0.iloc[1].lower() if len(r0) > 1 else "")
    row0_join = " ".join(r0.astype(str).str.lower().tolist())

    two_row = (
        ("hour" in first_cell) and
        ("variable" in second_cell or "variable" in row0_join)
    )

    if two_row:
        # Flatten: first col stays the time label, others come from row1
        new_cols = [r0.iloc[0] if r0.iloc[0] else "Hour"]
        for j in range(1, raw0.shape[1]):
            name = r1.iloc[j] if j < len(r1) else ""
            name = str(name).strip()
            new_cols.append(name if name and name.lower() != "nan" else f"col{j}")
        df = raw0.iloc[2:].copy()
        df.columns = new_cols
        print("Detected two-row header and flattened it.")
    else:
        # Fall back: file already has a single real header row
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        print("Did not detect two-row header; using file header as-is.")

    print("Headers after loader:", list(df.columns)[:8])

    # ---- helpers to find a column by tokens (order-insensitive) ----
    def find_contains_all(tokens):
        toks = [t.lower() for t in tokens]
        for c in df.columns:
            name = str(c).lower()
            if all(t in name for t in toks):
                return c
        return None

    def find_any(patterns):
        for toks in patterns:
            col = find_contains_all(toks)
            if col is not None:
                return col
        return None

    # Map required columns (built for your labels)
    t_col  = find_any([["hour"], ["time"]])

    # "Pool heater supply, W"
    p_col  = find_any([
        ["pool", "heater", "supply", "w"],
        ["heater", "supply", "w"],
        ["heater", "supply"],
        ["heater", "power"],
        ["supply", "w"]
    ])

    # "Pool water return temp, Deg-C"
    tp_col = find_any([
        ["pool", "water", "return", "temp"],
        ["pool", "return", "temp"],
        ["pool", "water", "temp"]
    ])

    # "Air temperature at extract"
    ta_col = find_any([
        ["air", "temperature", "extract"],
        ["air", "extract"],
        ["air", "temp"]
    ])

    # "RH at extract"
    rh_col = find_any([
        ["rh", "extract"],
        ["relative", "humidity", "extract"],
        ["rh"]
    ])

    if not all([t_col, p_col, tp_col, ta_col, rh_col]):
        missing = [("P_heater", p_col), ("T_pool", tp_col), ("T_air", ta_col), ("RH", rh_col)]
        have = [str(c) for c in df.columns]
        raise ValueError(
            "Missing columns: " + ", ".join(k for k, v in missing if v is None) +
            f"\nSeen headers: {have[:8]} ..."
        )

    # Build normalized frame
    out = pd.DataFrame({
        "time":       pd.to_datetime(df[t_col], errors="coerce", dayfirst=True),
        "P_heater_W": pd.to_numeric(df[p_col], errors="coerce"),
        "T_pool_C":   pd.to_numeric(df[tp_col], errors="coerce"),
        "T_air_C":    pd.to_numeric(df[ta_col], errors="coerce"),
        "RH_frac":    pd.to_numeric(df[rh_col], errors="coerce")/100.0,
    }).dropna(subset=["time"]).reset_index(drop=True)

    print("Matched columns:",
          f"time='{t_col}'  P_heater='{p_col}'  T_pool='{tp_col}'  T_air='{ta_col}'  RH='{rh_col}'")

    out["date"] = out["time"].dt.date
    out["hour"] = out["time"].dt.hour
    return out

# ========================= GEOMETRY HELPERS =========================
def geometry():
    A_pool = L_POOL * W_POOL
    A_side = 2.0 * (L_POOL + W_POOL) * D_POOL
    A_bottom = A_pool
    V_pool = A_pool * D_POOL
    m_pool = RHO_W * V_pool
    return A_pool, A_side, A_bottom, V_pool, m_pool

def h_conv_Wm2K():
    """Returns convective h. If USE_CONV_FROM_VEL=False and H_FIXED_WM2K is set, uses the fixed value."""
    if not USE_CONV_FROM_VEL and H_FIXED_WM2K is not None:
        return float(H_FIXED_WM2K)
    Re = AIR_VELOCITY * L_POOL / NU_AIR
    Nu = 0.0296 * (Re ** 0.8) * (PR_AIR ** (1.0 / 3.0))
    return (Nu * K_AIR) / L_POOL

# ========================= MODEL CORE =========================
def evap_exact_Q(Tw_C, Tair_C, RH_frac, A=None):
    """Exact evaporation heat loss at instantaneous Tw, Tair, RH."""
    if A is None:
        A = geometry()[0]   # pool surface area
    Pa = Pv_from_RH_Tair(RH_frac, Tair_C)
    m_evap = C_EVAP * A * F_ACTIVITY * max(0.0, psat_pa(Tw_C) - Pa)  # kg/s
    return HFG * m_evap, m_evap

def linearize_evap_at_T0(T0_C, Tair0_C, RH0_frac, A=None):
    """Return (Qev0, kev_W_per_K, m0) at start conditions for linearized evap."""
    if A is None:
        A = geometry()[0]
    Pa0 = Pv_from_RH_Tair(RH0_frac, Tair0_C)
    Ps0 = psat_pa(T0_C)
    m0 = C_EVAP * A * F_ACTIVITY * max(0.0, Ps0 - Pa0)
    km = C_EVAP * A * F_ACTIVITY * dpsat_dT(T0_C)             # kg/s/K
    return HFG * m0, HFG * km, m0

# ---------- Original dynamic integrator (kept, but upgraded to Heun/trapezoid) ----------
def integrate_timevary(win: pd.DataFrame, cap_Tmax=None):
    """
    Time-varying step integration over window 'win' using simulation timestamps.
    Uses Heun's method (predictor-corrector) for better accuracy.
    """
    A_pool, A_side, A_bottom, V_pool, m_pool = geometry()
    h = h_conv_Wm2K()
    UA_solid = U_SIDE * A_side + U_BOTTOM * A_bottom

    tsec = (win["time"] - win["time"].iloc[0]).dt.total_seconds().values

    # Initial values
    T = float(win["T_pool_C"].iloc[0]); T0 = T
    T_room0 = (T0 - ROOM_DELTA_K) if USE_ROOM_TRACK else float(win["T_air_C"].iloc[0]) - 1.0
    Qev0, kev, m0 = linearize_evap_at_T0(T0, float(win["T_air_C"].iloc[0]), float(win["RH_frac"].iloc[0]), A_pool)

    def rhs_at(T_state, idx):
        T_air = float(win["T_air_C"].iloc[idx])
        RH    = float(win["RH_frac"].iloc[idx])
        P_heater = float(win["P_heater_W"].iloc[idx])
        T_room = (T_state - ROOM_DELTA_K) if USE_ROOM_TRACK else T_room0
        Q_conv = h * A_pool * (T_state - T_air)
        Q_cond = UA_solid * (T_state - T_room)
        if USE_EVAP_EXACT:
            Q_evap, m_evap = evap_exact_Q(T_state, T_air, RH, A_pool)
        else:
            Q_evap = Qev0 + kev * (T_state - T0)
        Q_rec = 0.0
        Q_make = 0.0
        if INCLUDE_MAKEUP:
            m_hyg = (HYGIENE_L_PER_DAY / 86400.0)
            m_make = (m0 if not USE_EVAP_EXACT else m_evap) + m_hyg
            Q_make = m_make * CP_W * max(0.0, T_state - TAP_WATER_C)
        return P_heater + Q_rec - Q_make - (Q_conv + Q_cond + Q_evap)

    # Heun integration
    T_series = [T0]
    rhs_prev = rhs_at(T, 0)
    for i in range(1, len(tsec)):
        dt = tsec[i] - tsec[i-1]
        T_pred = T + rhs_prev * dt / (m_pool * CP_W)
        rhs_curr = rhs_at(T_pred, i)
        T = T + 0.5*(rhs_prev + rhs_curr) * dt / (m_pool * CP_W)
        if cap_Tmax is not None:
            T = min(T, cap_Tmax)
        T_series.append(T)
        rhs_prev = rhs_curr

    return np.array(tsec), np.array(T_series), T0, m_pool

# ---------- Calibrated dynamic integrator (new) ----------
def integrate_timevary_calibrated(win: pd.DataFrame, cap_Tmax=None):
    """
    Dynamic integration with:
      - Heun (trapezoid) ODE step
      - ALPHA_P scaling on heater power (net to pool)
      - SCALE_UA scaling on conductive losses
      - SCALE_EVAP scaling on evaporation losses
    """
    A_pool, A_side, A_bottom, V_pool, m_pool = geometry()
    h = h_conv_Wm2K()
    UA_solid = (U_SIDE * A_side + U_BOTTOM * A_bottom) * SCALE_UA

    tsec = (win["time"] - win["time"].iloc[0]).dt.total_seconds().values

    P_vec = win["P_heater_W"].astype(float).values * ALPHA_P
    Tair_vec = win["T_air_C"].astype(float).values
    RH_vec = win["RH_frac"].astype(float).values

    T = float(win["T_pool_C"].iloc[0]); T0 = T
    T_room0 = (T0 - ROOM_DELTA_K) if USE_ROOM_TRACK else float(Tair_vec[0]) - 1.0

    def rhs_at(T_state, idx):
        T_air = float(Tair_vec[idx])
        RH    = float(RH_vec[idx])
        P_heater = float(P_vec[idx])
        T_room = (T_state - ROOM_DELTA_K) if USE_ROOM_TRACK else T_room0
        Q_conv = h * A_pool * (T_state - T_air)
        Q_cond = UA_solid * (T_state - T_room)
        if USE_EVAP_EXACT:
            Q_evap, m_evap = evap_exact_Q(T_state, T_air, RH, A_pool)
        else:
            # local linearization around T_state
            Qev0, kev, _ = linearize_evap_at_T0(T_state, T_air, RH, A_pool)
            Q_evap = Qev0
        Q_evap *= SCALE_EVAP
        Q_rec = 0.0
        Q_make = 0.0
        if INCLUDE_MAKEUP:
            m_hyg = (HYGIENE_L_PER_DAY / 86400.0)
            m_make = (m_evap if USE_EVAP_EXACT else 0.0) + m_hyg
            Q_make = m_make * CP_W * max(0.0, T_state - TAP_WATER_C)
        return P_heater + Q_rec - Q_make - (Q_conv + Q_cond + Q_evap)

    T_series = [T0]
    rhs_prev = rhs_at(T, 0)
    for i in range(1, len(tsec)):
        dt = tsec[i] - tsec[i-1]
        T_pred = T + rhs_prev * dt / (m_pool * CP_W)
        rhs_curr = rhs_at(T_pred, i)
        T = T + 0.5*(rhs_prev + rhs_curr) * dt / (m_pool * CP_W)
        if cap_Tmax is not None:
            T = min(T, cap_Tmax)
        T_series.append(T)
        rhs_prev = rhs_curr

    return np.array(tsec), np.array(T_series), T0, m_pool

def closed_form_constant(win: pd.DataFrame, cap_Tmax=None):
    """
    Constant-inputs (window-average) + linearized evaporation at T0 (closed form).
    Uses ALPHA_P to keep heater power definition consistent with calibrated dynamic.
    """
    A_pool, A_side, A_bottom, V_pool, m_pool = geometry()
    h = h_conv_Wm2K()
    UA_solid = U_SIDE * A_side + U_BOTTOM * A_bottom

    # Averages within window
    T0 = float(win["T_pool_C"].iloc[0])
    T_air_bar = float(win["T_air_C"].mean())
    RH_bar = float(win["RH_frac"].mean())
    P_heater_bar = float(win["P_heater_W"].mean()) * (ALPHA_P if USE_CALIBRATED_DYNAMIC else 1.0)

    T_room_bar = (T0 - ROOM_DELTA_K) if USE_ROOM_TRACK else (T_air_bar - 1.0)
    if USE_EVAP_EXACT:
        Qev0, _ = evap_exact_Q(T0, T_air_bar, RH_bar, A_pool)
        kev = C_EVAP * A_pool * F_ACTIVITY * dpsat_dT(T0) * HFG
    else:
        Qev0, kev, _ = linearize_evap_at_T0(T0, T_air_bar, RH_bar, A_pool)

    # Total linearized conductance
    K = (h * A_pool + UA_solid) * (SCALE_UA if USE_CALIBRATED_DYNAMIC else 1.0) + \
        kev * (SCALE_EVAP if USE_CALIBRATED_DYNAMIC else 1.0)

    # Constant term
    C = (P_heater_bar
         + (h * A_pool) * T_air_bar * (SCALE_UA if USE_CALIBRATED_DYNAMIC else 1.0)
         + (UA_solid)  * T_room_bar * (SCALE_UA if USE_CALIBRATED_DYNAMIC else 1.0)
         + kev * T0    * (SCALE_EVAP if USE_CALIBRATED_DYNAMIC else 1.0)
         - Qev0        * (SCALE_EVAP if USE_CALIBRATED_DYNAMIC else 1.0))

    tau_s = (m_pool * CP_W) / K
    T_inf = C / K

    dt_h = (win["time"].iloc[-1] - win["time"].iloc[0]).total_seconds() / 3600.0
    T_end = T_inf + (T0 - T_inf) * math.exp(-dt_h * 3600.0 / tau_s)
    if cap_Tmax is not None:
        T_end = min(T_end, cap_Tmax)

    # synthetic trajectory at timestamps (for plotting)
    tsec = (win["time"] - win["time"].iloc[0]).dt.total_seconds().values
    T_series = T_inf + (T0 - T_inf) * np.exp(-tsec / tau_s)

    return tsec, T_series, T0, m_pool, tau_s, T_inf

# ========================= EVENTS =========================
def build_events(df: pd.DataFrame):
    """Create list of (t0, t1) event windows."""
    if EXPLICIT_EVENTS:
        starts = pd.to_datetime(EXPLICIT_EVENTS)
    else:
        starts = []
        for d, grp in df.groupby("date"):
            row = grp[grp["hour"] == START_HOUR]
            if not row.empty:
                starts.append(row.iloc[0]["time"])
    return [(t0, t0 + pd.Timedelta(hours=DURATION_H)) for t0 in starts]

# ========================= MAIN =========================
def main():
    df = load_pool_run(INPUT_CSV)
    events = build_events(df)
    if not events:
        raise SystemExit("No events found—check START_HOUR / EXPLICIT_EVENTS and timestamps.")

    rows = []
    for (t0, t1) in events:
        win = df[(df["time"] >= t0) & (df["time"] <= t1)].copy()
        if len(win) < 2:
            continue

        # --- model path: time-varying vs constant ---
        if USE_TIME_VARYING:
            if USE_CALIBRATED_DYNAMIC:
                tsec, T_mod, T0, m_pool = integrate_timevary_calibrated(win, cap_Tmax=CAP_TMAX)
                T_end_model = float(T_mod[-1])
                tau_eff, T_inf = np.nan, np.nan
            else:
                tsec, T_mod, T0, m_pool = integrate_timevary(win, cap_Tmax=CAP_TMAX)
                T_end_model = float(T_mod[-1])
                tau_eff, T_inf = np.nan, np.nan
        else:
            tsec, T_mod, T0, m_pool, tau_s, T_inf = closed_form_constant(win, cap_Tmax=CAP_TMAX)
            T_end_model = float(T_mod[-1])
            tau_eff = tau_s

        # --- “truth” from simulation over same window ---
        T_end_sim = float(win["T_pool_C"].iloc[-1])
        if CAP_TMAX is not None:
            T_end_sim = min(T_end_sim, CAP_TMAX)

        # --- sensible energy stored (kWh) ---
        Q_model_kWh = (m_pool * CP_W * (T_end_model - T0)) / 3.6e6
        Q_sim_kWh   = (m_pool * CP_W * (T_end_sim   - T0)) / 3.6e6

        rows.append({
            "date": str(pd.Timestamp(t0).date()),
            "t0": t0, "t1": t1,
            "T0_C": T0,
            "T_end_sim_C": T_end_sim,
            "T_end_model_C": T_end_model,
            "dT_sim_C": T_end_sim - T0,
            "dT_model_C": T_end_model - T0,
            "Q_sim_kWh": Q_sim_kWh,
            "Q_model_kWh": Q_model_kWh,
            "tau_model_min": (tau_eff/60.0) if isinstance(tau_eff,(int,float)) and np.isfinite(tau_eff) else np.nan,
            "T_inf_model_C": T_inf
        })

        print(f"{t0:%Y-%m-%d} | T0={T0:5.2f}°C | T_end(sim)={T_end_sim:5.2f}°C | "
              f"T_end(model)={T_end_model:5.2f}°C | ΔT_sim={T_end_sim-T0:+5.2f} K | "
              f"ΔT_model={T_end_model-T0:+5.2f} K | Q_sim={Q_sim_kWh:+6.3f} kWh | Q_mod={Q_model_kWh:+6.3f} kWh")

    out = pd.DataFrame(rows)
    if out.empty:
        print("No valid events produced. Check timestamps and column parsing.")
        return

    # ------- summary -------
    T_err = out["T_end_model_C"] - out["T_end_sim_C"]
    Q_err = out["Q_model_kWh"]   - out["Q_sim_kWh"]
    print("\nSUMMARY")
    print(f"Events processed: {len(out)}")
    print(f"Mean abs T_end error: {T_err.abs().mean():.3f} K | RMSE: {np.sqrt((T_err**2).mean()):.3f} K")
    print(f"Mean abs Q error:     {Q_err.abs().mean():.3f} kWh | RMSE: {np.sqrt((Q_err**2).mean()):.3f} kWh")
    print(f"Total Q_sim:  {out['Q_sim_kWh'].sum():.1f} kWh | Total Q_model: {out['Q_model_kWh'].sum():.1f} kWh")

    if not PLOTS_SHOW:
        return

    # ------- representative events (worst/median/best by |T_end error|) -------
    idx_sorted = T_err.abs().sort_values().index.to_list()
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
            if USE_CALIBRATED_DYNAMIC:
                tsec, T_mod, _, _ = integrate_timevary_calibrated(win, cap_Tmax=CAP_TMAX)
            else:
                tsec, T_mod, _, _ = integrate_timevary(win, cap_Tmax=CAP_TMAX)
        else:
            tsec, T_mod, _, _, _, _ = closed_form_constant(win, cap_Tmax=CAP_TMAX)

        fig, ax = plt.subplots()
        ax.plot(win["time"], win["T_pool_C"], label="Simulation T_pool", lw=2)
        ax.plot(win["time"], T_mod, "--", label="Model T_pool", lw=2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %H:%M"))
        ax.set_xlabel("Time"); ax.set_ylabel("Pool water temperature (°C)")
        ax.set_title(f"Event {t0:%Y-%m-%d %H:%M} → {t1:%H:%M} | "
                     f"T_end(sim)={r['T_end_sim_C']:.2f}°C, model={r['T_end_model_C']:.2f}°C")
        ax.grid(True); ax.legend()

    # ------- monthly average trajectory (10-min bins) -------
    BIN_MIN = 10
    recs = []
    for _, r in out.iterrows():
        t0 = pd.to_datetime(r["t0"])
        win = df[(df["time"] >= t0) & (df["time"] <= r["t1"])].copy()
        if win.empty:
            continue
        rel_min = (win["time"] - t0).dt.total_seconds().values / 60.0

        if USE_TIME_VARYING:
            if USE_CALIBRATED_DYNAMIC:
                tsec, T_mod, _, _ = integrate_timevary_calibrated(win, cap_Tmax=CAP_TMAX)
            else:
                tsec, T_mod, _, _ = integrate_timevary(win, cap_Tmax=CAP_TMAX)
        else:
            tsec, T_mod, _, _, _, _ = closed_form_constant(win, cap_Tmax=CAP_TMAX)

        mon = t0.strftime("%Y-%m")
        for tt, Ts, Tm in zip(rel_min, win["T_pool_C"].values, T_mod):
            recs.append({"month": mon, "bin": (tt // BIN_MIN) * BIN_MIN,
                         "T_sim": Ts, "T_mod": Tm})
    if recs:
        mdf = pd.DataFrame(recs).groupby(["month","bin"]).agg(
            T_sim=("T_sim","mean"), T_mod=("T_mod","mean"), n=("T_sim","count")
        ).reset_index()
        for mon, grp in mdf.groupby("month"):
            fig, ax = plt.subplots()
            ax.plot(grp["bin"], grp["T_sim"], lw=2, label=f"Sim (avg of {int(grp['n'].sum())} pts)")
            ax.plot(grp["bin"], grp["T_mod"], "--", lw=2, label="Model (avg)")
            ax.set_xlabel(f"Minutes from event start (bin={BIN_MIN} min)")
            ax.set_ylabel("Pool water temperature (°C)")
            ax.set_title(f"Monthly average pool T trajectory — {mon}")
            ax.grid(True); ax.legend()

    plt.show()

# ========================= RUN =========================
if __name__ == "__main__":
    main()
