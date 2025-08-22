#!/usr/bin/env python3
"""
Room water mass balance and latent-energy balance from side-by-side CSV.
Displays the 24-hour average plot instead of saving it.

CSV format (as in Winter_month.csv):
  Row 0:  "RH 53-55", ..., "RH 56-65", ...
  Row 1:  variable names
  Row 2+: data
Left block (tight):    cols [0..6]  -> time | T_room_C | T_sup_C | w_ret | w_sup | m_sup | m_ret
Right block (relaxed): cols [8..14] -> time | T_room_C | T_sup_C | w_ret | w_sup | m_sup | m_ret
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# HARD-CODED PATHS (EDIT)
# -------------------------
INPUT_CSV   = r"M:\PhD\02 Data sets from simulations\Air Flexibility\Winter_month.csv"         # <- your CSV
RUN         = "tight"   # "tight" or "relaxed"
OUT_PERHOUR = r"M:\PhD\02 Data sets from simulations\Air Flexibility\mass_energy_balance_perhour.csv"
OUT_24AVG   = r"M:\PhD\02 Data sets from simulations\Air Flexibility\mass_energy_balance_24h_avg.csv"

# -------------------------
# SITE CONSTANTS
# -------------------------
VOLUME_M3   = 19.65 * 13.8 * 4.95     # m³
PRESSURE_PA = 101325.0
DT_H        = 1.0                     # dataset is hourly
DT_S        = DT_H * 3600.0

# Psychrometric helpers
def h_fg(T_C: float) -> float:
    return 2.501e6 - 2361.0 * float(T_C)   # J/kg

def rho_da_from_wT(w: float, T_C: float, p: float = PRESSURE_PA) -> float:
    w = float(w); T_C = float(T_C)
    pv  = p * (w / (0.62198 + w))          # Pa
    pda = p - pv
    return pda / (287.05 * (T_C + 273.15)) # kg/m³

# -------------------------
# LOAD CSV & PICK RUN
# -------------------------
df = pd.read_csv(INPUT_CSV)

tight = pd.DataFrame({
    "time":     pd.to_datetime(df.iloc[2:, 0], errors="coerce", dayfirst=True),
    "T_room_C": pd.to_numeric(df.iloc[2:, 1], errors="coerce"),
    "T_sup_C":  pd.to_numeric(df.iloc[2:, 2], errors="coerce"),
    "w_ret":    pd.to_numeric(df.iloc[2:, 3], errors="coerce"),
    "w_sup":    pd.to_numeric(df.iloc[2:, 4], errors="coerce"),
    "m_sup":    pd.to_numeric(df.iloc[2:, 5], errors="coerce"),
    "m_ret":    pd.to_numeric(df.iloc[2:, 6], errors="coerce"),
}).dropna(subset=["time"]).reset_index(drop=True)

relaxed = pd.DataFrame({
    "time":     pd.to_datetime(df.iloc[2:, 8], errors="coerce", dayfirst=True),
    "T_room_C": pd.to_numeric(df.iloc[2:, 9], errors="coerce"),
    "T_sup_C":  pd.to_numeric(df.iloc[2:,10], errors="coerce"),
    "w_ret":    pd.to_numeric(df.iloc[2:,11], errors="coerce"),
    "w_sup":    pd.to_numeric(df.iloc[2:,12], errors="coerce"),
    "m_sup":    pd.to_numeric(df.iloc[2:,13], errors="coerce"),
    "m_ret":    pd.to_numeric(df.iloc[2:,14], errors="coerce"),
}).dropna(subset=["time"]).reset_index(drop=True)

data = tight if RUN.lower()=="tight" else relaxed
data = data.sort_values("time").reset_index(drop=True)
data["hour"] = data["time"].dt.hour
data["date"] = data["time"].dt.date

# -------------------------
# HOURLY MASS BALANCE
# -------------------------
rows = []
for i in range(len(data)-1):
    t0, t1 = data.loc[i], data.loc[i+1]
    if (t1["time"] - t0["time"]).total_seconds() < 0.5*DT_S:
        continue

    rho0 = rho_da_from_wT(t0["w_ret"], t0["T_room_C"], PRESSURE_PA)
    rho1 = rho_da_from_wT(t1["w_ret"], t1["T_room_C"], PRESSURE_PA)
    m_da0 = rho0 * VOLUME_M3
    m_da1 = rho1 * VOLUME_M3

    M_w0 = m_da0 * t0["w_ret"]
    M_w1 = m_da1 * t1["w_ret"]
    dM_w = M_w1 - M_w0                     # kg over the hour

    m_w_in  = t0["m_sup"] * t0["w_sup"] / (1.0 + t0["w_sup"])
    m_w_out = t0["m_ret"] * t0["w_ret"] / (1.0 + t0["w_ret"])

    m_evap = (dM_w / DT_S) - (m_w_in - m_w_out)  # kg/s

    Tavg = 0.5*(float(t0["T_room_C"])+float(t1["T_room_C"]))
    L = h_fg(Tavg)                          # J/kg

    P_store_kW = (dM_w * L / DT_S) / 1000.0
    P_vent_kW  = ((m_w_in - m_w_out) * L) / 1000.0
    P_evap_kW  = (m_evap * L) / 1000.0
    E_store_kWh = P_store_kW * DT_H

    rows.append({
        "time_start": t0["time"], "time_end": t1["time"],
        "hour_start": t0["hour"],
        "M_w_room_start_kg": M_w0, "M_w_room_end_kg": M_w1, "dM_w_kg": dM_w,
        "m_w_in_kg_s": m_w_in, "m_w_out_kg_s": m_w_out, "m_evap_kg_s": m_evap,
        "P_store_kW": P_store_kW, "P_vent_kW": P_vent_kW, "P_evap_kW": P_evap_kW,
        "E_store_kWh": E_store_kWh
    })

perhour = pd.DataFrame(rows)
perhour.to_csv(OUT_PERHOUR, index=False)
print(f"Saved per-hour mass/energy balance to: {OUT_PERHOUR}")

# -------------------------
# 24-HOUR AVERAGES (BY HOUR)
# -------------------------
avg24 = perhour.groupby("hour_start")[["P_store_kW","P_vent_kW","P_evap_kW"]].mean().reset_index()
avg24.to_csv(OUT_24AVG, index=False)
print(f"Saved 24-hour average (by hour) to: {OUT_24AVG}")

# -------------------------
# DISPLAY PLOT (no saving)
# -------------------------
plt.figure()
plt.plot(avg24["hour_start"], avg24["P_evap_kW"], marker="o", label="Inferred evaporation (kW)")
plt.plot(avg24["hour_start"], avg24["P_vent_kW"], marker="o", label="Ventilation transport (kW)")
plt.plot(avg24["hour_start"], avg24["P_store_kW"], marker="o", label="Storage rate (kW)")
plt.xlabel("Hour of day (start of 1-h window)")
plt.ylabel("Latent power equivalent (kW)")
plt.title(f"Water mass/latent energy balance — {RUN} run")
plt.grid(True); plt.xticks(range(0,24,1)); plt.legend()
plt.tight_layout()
plt.show()  # <-- display instead of saving
