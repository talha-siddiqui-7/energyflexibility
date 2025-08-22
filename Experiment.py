#!/usr/bin/env python3
"""
Compute TRUE latent storage (relaxed − tight) from a side-by-side CSV export
and plot the average TRUE storage by hour (24 points).

CSV format (as in Winter_month.csv):
  Row 0:  "RH 53-55", ..., "RH 56-65", ...
  Row 1:  variable names for each block
  Row 2+: data rows

Left block (tight):    cols [0..6]  -> time | T_room_C | T_sup_C | w_ret | w_sup | m_sup | m_ret
Right block (relaxed): cols [8..14] -> time | T_room_C | T_sup_C | w_ret | w_sup | m_sup | m_ret
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt


# -------------------------
# HARD-CODED PATHS (EDIT)
# -------------------------
INPUT_CSV  = r"M:\PhD\02 Data sets from simulations\Air Flexibility\Winter_month.csv"        # <-- change to your CSV
OUTPUT_CSV = r"M:\PhD\02 Data sets from simulations\Air Flexibility\true_storage_01to04.csv" # <-- where to save per-night result CSV
OUTPUT_HOURLY    = r"M:\PhD\02 Data sets from simulations\Air Flexibility\true_storage_hourly.csv"       # 24-point table
OUTPUT_PNG       = r"M:\PhD\02 Data sets from simulations\Air Flexibility\true_storage_hourly.png"       # plot image
OUTPUT_PER_NIGHT = r"M:\PhD\02 Data sets from simulations\Air Flexibility\true_storage_01to04.csv"       # per-night results

# -------------------------
# SITE / WINDOW SETTINGS
# -------------------------
VOLUME_M3   = 19.65 * 13.8 * 4.95   # hall volume
PRESSURE_PA = 101325.0              # barometric pressure
COP         = 4.0                   # dehumidifier COP
START_HOUR  = 21                     # night window start (for per-night calc)
END_HOUR    = 0                     # night window end   (per-night calc)

# -------------------------
# PSYCHRO HELPERS
# -------------------------
def h_fg(T_C: float) -> float:
    """Latent heat [J/kg] at air temperature T_C [°C]."""
    return 2.501e6 - 2361.0 * float(T_C)

def rho_da_from_wT(w: float, T_C: float, p: float = PRESSURE_PA) -> float:
    """Dry-air density [kg/m³] from humidity ratio w [kg/kg_da] and T_C [°C] at pressure p [Pa]."""
    w = float(w); T_C = float(T_C)
    pv = p * (w / (0.62198 + w))   # water vapour partial pressure
    p_da = p - pv                  # dry-air partial pressure
    return p_da / (287.05 * (T_C + 273.15))

# -------------------------
# LOAD & PARSE CSV (side-by-side blocks)
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

for d in (tight, relaxed):
    d["date"] = d["time"].dt.date
    d["hour"] = d["time"].dt.hour

# -------------------------
# STORAGE CALC PER RUN (generic start→end)
# -------------------------
def storage_window(d: pd.DataFrame, start_hour: int, end_hour: int, V: float, p: float):
    """Compute E_store (kWh) for a single date window start→end. Returns dict {date: E_kWh}."""
    out = {}
    for date, grp in d.groupby("date"):
        s = grp[grp["hour"] == start_hour]
        if s.empty:
            continue
        s = s.iloc[0]
        # end can be same date or next day if crossing midnight
        if end_hour >= start_hour:
            end_date = date
        else:
            end_date = (pd.to_datetime(date) + pd.Timedelta(days=1)).date()
        e = d[(d["date"] == end_date) & (d["hour"] == end_hour)]
        if e.empty:
            continue
        e = e.iloc[0]

        rho = rho_da_from_wT(s["w_ret"], s["T_room_C"], p)
        m_da = rho * V
        T_avg = 0.5 * (float(s["T_room_C"]) + float(e["T_room_C"]))
        dW = float(e["w_ret"] - s["w_ret"])
        E_kWh = m_da * dW * h_fg(T_avg) / 3.6e6
        out[date] = E_kWh
    return out

# -------------------------
# PART 1: PER-NIGHT TRUE STORAGE (for chosen window START_HOUR→END_HOUR)
# -------------------------
stor_tight_map = storage_window(tight, START_HOUR, END_HOUR, VOLUME_M3, PRESSURE_PA)
stor_relax_map = storage_window(relaxed, START_HOUR, END_HOUR, VOLUME_M3, PRESSURE_PA)

common_dates = sorted(set(stor_tight_map).intersection(stor_relax_map))
per_night_rows = []
for dt in common_dates:
    E_t = stor_tight_map[dt]
    E_r = stor_relax_map[dt]
    per_night_rows.append({
        "date": dt,
        "E_store_tight_kWh": E_t,
        "E_store_relaxed_kWh": E_r,
        "TRUE_storage_kWh (relaxed - tight)": E_r - E_t,
        f"Shifted_electric_kWh (COP={COP})": (E_r - E_t) / (COP - 1.0)
    })
per_night_df = pd.DataFrame(per_night_rows)
per_night_df.to_csv(OUTPUT_PER_NIGHT, index=False)

# Quick console summary
avg_true = float(per_night_df["TRUE_storage_kWh (relaxed - tight)"].mean()) if len(per_night_df) else float("nan")
nights = len(per_night_df)
shifted = avg_true / (COP - 1.0) if COP and COP > 1 else float("nan")
print("Per-night summary:")
print(f"  Nights processed: {nights}")
print(f"  Avg TRUE storage (kWh/night): {avg_true:.6f}")
print(f"  Avg shifted electric (kWh/night, COP={COP}): {shifted:.6f}")
print(f"  Saved per-night CSV: {OUTPUT_PER_NIGHT}")

# -------------------------
# PART 2: AVERAGE TRUE STORAGE BY HOUR (24 hourly windows h→h+1)
# -------------------------
def hourly_true_storage_avg(tight_df: pd.DataFrame, relaxed_df: pd.DataFrame, V: float, p: float):
    """
    For each hour h in 0..23, compute TRUE storage (relaxed - tight) for the window h→h+1, per date,
    then average across all dates where both runs have the required timestamps.
    Returns DataFrame with columns: hour, avg_true_kWh, n_days_used
    """
    rows = []
    all_dates = sorted(set(tight_df["date"]).intersection(relaxed_df["date"]))
    # Also ensure next-day exists for the 23→0 window
    for h in range(24):
        h_next = (h + 1) % 24
        diffs = []
        used = 0
        for dt in all_dates:
            # For 23→0, end is next calendar date
            end_date = dt if h_next > h else (pd.to_datetime(dt) + pd.Timedelta(days=1)).date()

            # Tight start/end
            s_t = tight_df[(tight_df["date"] == dt) & (tight_df["hour"] == h)]
            e_t = tight_df[(tight_df["date"] == end_date) & (tight_df["hour"] == h_next)]
            # Relax start/end
            s_r = relaxed_df[(relaxed_df["date"] == dt) & (relaxed_df["hour"] == h)]
            e_r = relaxed_df[(relaxed_df["date"] == end_date) & (relaxed_df["hour"] == h_next)]
            if len(s_t) == 1 and len(e_t) == 1 and len(s_r) == 1 and len(e_r) == 1:
                s_t, e_t = s_t.iloc[0], e_t.iloc[0]
                s_r, e_r = s_r.iloc[0], e_r.iloc[0]
                # Per-run storage for this hour and date
                rho_t = rho_da_from_wT(s_t["w_ret"], s_t["T_room_C"], p)
                m_da = rho_t * V  # use start density
                Tavg_t = 0.5 * (float(s_t["T_room_C"]) + float(e_t["T_room_C"]))
                Tavg_r = 0.5 * (float(s_r["T_room_C"]) + float(e_r["T_room_C"]))
                E_t = m_da * (float(e_t["w_ret"]) - float(s_t["w_ret"])) * h_fg(Tavg_t) / 3.6e6
                E_r = m_da * (float(e_r["w_ret"]) - float(s_r["w_ret"])) * h_fg(Tavg_r) / 3.6e6
                diffs.append(E_r - E_t)
                used += 1
        avg = float(np.mean(diffs)) if diffs else np.nan
        rows.append({"hour": h, "avg_TRUE_storage_kWh": avg, "n_days_used": used})
    return pd.DataFrame(rows)

hourly_df = hourly_true_storage_avg(tight, relaxed, VOLUME_M3, PRESSURE_PA)
hourly_df.to_csv(OUTPUT_HOURLY, index=False)
print(f"Saved hourly averages CSV: {OUTPUT_HOURLY}")

# -------------------------
# PLOT: Average TRUE storage by hour (kWh)
# -------------------------
plt.figure()
plt.plot(hourly_df["hour"], hourly_df["avg_TRUE_storage_kWh"], marker="o")
plt.xlabel("Hour of day (window h → h+1)")
plt.ylabel("Average TRUE storage (kWh)")
plt.title("Average TRUE latent storage by hour")
plt.grid(True)
plt.xticks(range(0,24,1))
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150)
print(f"Saved plot: {OUTPUT_PNG}")