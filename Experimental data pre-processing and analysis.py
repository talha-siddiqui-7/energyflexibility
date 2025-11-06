import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import datetime as dt

# ===========================================================
# FILE PATHS & COLUMNS
# ===========================================================
FILE_PATH   = r"M:\PhD\03 Experiments\Complete_17-09-2025_16-10-2025_with_AHU_power.csv"
COL_POOL    = 'Pool_OF'
COL_POOL_SP = 'Setpoint Pool water temperature'
COL_RH      = 'Extract_air_RH'
COL_RH_SP   = 'Setpoint RH'
COL_DAMPER  = 'Fresh air damper'

# ===========================================================
# TOGGLE OPTION FOR SETPOINTS
# ===========================================================
ASK_FOR_TOGGLE = True
SHOW_SETPOINTS = True
if ASK_FOR_TOGGLE:
    ans = input("Overlay setpoint lines on weekly plots? [y/N]: ").strip().lower()
    SHOW_SETPOINTS = (ans == '' or ans.startswith('y'))

# ===========================================================
# LOAD DATA & PREPROCESS
# ===========================================================
df = pd.read_csv(FILE_PATH)
df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True, errors='coerce')
df = df.sort_values('datetime').set_index('datetime')
df = df.loc[df.index.notna()]

# Coerce numeric
non_numeric_cols = {'date', 'time'}
for c in df.columns:
    if c not in non_numeric_cols and df[c].dtype == 'object':
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False), errors='coerce')

# Interpolate numeric columns
dfi = df.copy()
num_cols = dfi.select_dtypes(include='number').columns
dfi[num_cols] = dfi[num_cols].interpolate(method='linear', limit_direction='both')

# Pool smoothing only (30 min)
dfi['Pool_OF_smooth_30min'] = dfi[COL_POOL].rolling(window=30, min_periods=1, center=True).mean()

# ===========================================================
# HELPER FUNCTIONS
# ===========================================================
def style_time_axis(ax):
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b\n%Y'))
    ax.tick_params(axis='x', labelsize=9)

def auto_ylim(ax, series_list, pad_fraction=0.05, hard_min=None, hard_max=None):
    vals = pd.concat(series_list).dropna()
    vals = pd.to_numeric(vals, errors='coerce').dropna()
    if vals.empty:
        return
    vmin, vmax = float(vals.min()), float(vals.max())
    pad = (vmax - vmin) * pad_fraction
    ymin, ymax = vmin - pad, vmax + pad
    if hard_min is not None: ymin = max(ymin, hard_min)
    if hard_max is not None: ymax = min(ymax, hard_max)
    ax.set_ylim(ymin, ymax)

# ===========================================================
# OVERALL PLOTS (NO SETPOINTS)
# ===========================================================
# 1️⃣ Pool overall
plt.figure(figsize=(13,5))
plt.plot(dfi.index, dfi[COL_POOL], lw=0.6, alpha=0.35, label='Pool_OF (raw 1-min)')
plt.plot(dfi.index, dfi['Pool_OF_smooth_30min'], lw=2.0, label='Pool_OF (30-min smooth)')
plt.title('Pool Water Temperature Over Time')
plt.xlabel('Time'); plt.ylabel('Pool_OF [°C]')
style_time_axis(plt.gca())
plt.legend(); plt.tight_layout(); plt.show(block=True)

# 2️⃣ RH overall
plt.figure(figsize=(13,5))
plt.plot(dfi.index, dfi[COL_RH], lw=1.0, label='Extract_air_RH (raw 1-min)')
plt.title('Extract Air Relative Humidity Over Time')
plt.xlabel('Time'); plt.ylabel('Extract_air_RH [%RH]')
style_time_axis(plt.gca())
plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(6))
plt.ylim(20, 90)
plt.legend(); plt.tight_layout(); plt.show(block=True)

# ===========================================================
# WEEKLY ANALYSIS (SETPOINTS OPTIONAL)
# ===========================================================
weekly_schedule = {
    0: [("08:00", "09:00"), ("13:00", "15:00"), ("17:00", "21:00")],
    1: [("09:00", "11:00"), ("17:00", "21:00")],
    2: [("09:00", "11:00"), ("16:00", "20:00")],
    3: [("09:00", "11:00"), ("17:00", "21:00")],
    4: [("09:00", "14:00"), ("17:00", "19:00")],
    5: [("13:00", "15:00")],
    6: [("13:00", "17:00"), ("19:00", "21:00")],
}
def is_active(ts):
    wd = ts.weekday()
    if wd not in weekly_schedule:
        return False
    for s, e in weekly_schedule[wd]:
        s_t = dt.datetime.combine(ts.date(), dt.time.fromisoformat(s))
        e_t = dt.datetime.combine(ts.date(), dt.time.fromisoformat(e))
        if s_t <= ts <= e_t:
            return True
    return False

dfi['pool_active'] = dfi.index.to_series().apply(is_active)
iso = dfi.index.isocalendar()
week_ids = iso['year'].astype(str) + '-W' + iso['week'].astype(str).str.zfill(2)
unique_weeks = week_ids.unique()

def weekly_plot(df_week, y_col, sp_col, title_prefix, y_label, rh_mode=False, show_sp=True):
    ax = plt.gca()
    t = df_week.index
    y = df_week[y_col]
    act = df_week['pool_active']

    # colored segments by occupancy
    if len(df_week) > 1:
        seg_start = 0
        for i in range(1, len(df_week)):
            if act.iloc[i] != act.iloc[i-1]:
                ax.plot(t[seg_start:i], y.iloc[seg_start:i],
                        lw=1.8, color=('red' if act.iloc[i-1] else 'blue'),
                        label=('Active / occupied' if act.iloc[i-1] else 'Inactive / empty'))
                seg_start = i
        ax.plot(t[seg_start:], y.iloc[seg_start:], lw=1.8,
                color=('red' if act.iloc[-1] else 'blue'),
                label=('Active / occupied' if act.iloc[-1] else 'Inactive / empty'))

    # shaded active
    active_start = None
    for i in range(len(df_week)):
        ts = df_week.index[i]
        now = act.iloc[i]
        if now and active_start is None:
            active_start = ts
        elif (not now or i == len(df_week)-1) and active_start is not None:
            ts_end = ts if not now else df_week.index[i]
            ax.axvspan(active_start, ts_end, color='red', alpha=0.08)
            active_start = None

    # setpoints (optional)
    if show_sp and sp_col in df_week.columns:
        ax.plot(df_week.index, df_week[sp_col], ls='--', lw=1.0, alpha=0.9,
                label=f'{sp_col} (from data)')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b\n%H:%M'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    series_for_ylim = [df_week[y_col]]
    if show_sp and sp_col in df_week.columns:
        series_for_ylim.append(df_week[sp_col])
    if rh_mode:
        auto_ylim(ax, series_for_ylim, pad_fraction=0.05, hard_min=20, hard_max=90)
    else:
        auto_ylim(ax, series_for_ylim, pad_fraction=0.03)

    wk = df_week.index[0]
    wk_str = f"{wk.isocalendar().year}-W{wk.isocalendar().week:02d}"
    ax.set_title(f"{title_prefix} — Week {wk_str}")
    ax.set_xlabel("Time"); ax.set_ylabel(y_label)
    h, l = ax.get_legend_handles_labels()
    uniq = dict(zip(l, h))
    if uniq:
        ax.legend(uniq.values(), uniq.keys(), loc='best', fontsize=9)
    plt.tight_layout()

# generate weekly plots
for wk in unique_weeks:
    dfw = dfi.loc[week_ids == wk].copy()
    if dfw.empty: continue
    plt.figure(figsize=(13,5))
    weekly_plot(dfw, 'Pool_OF_smooth_30min', COL_POOL_SP,
                'Pool Water Temperature (30-min smooth)', 'Pool_OF [°C]',
                rh_mode=False, show_sp=SHOW_SETPOINTS)
    plt.show(block=True)
    plt.figure(figsize=(13,5))
    weekly_plot(dfw, COL_RH, COL_RH_SP,
                'Extract Air Relative Humidity (raw 1-min)', 'Extract_air_RH [%RH]',
                rh_mode=True, show_sp=SHOW_SETPOINTS)
    plt.show(block=True)

# ===========================================================
# FRESH AIR DAMPER — show vs RH setpoint (no shading)
# ===========================================================
COL_DAMPER  = 'Fresh air damper'   # expected column name
PAD_MIN     = 15                   # context minutes around experiment window

if COL_DAMPER in dfi.columns and COL_RH_SP in dfi.columns:
    # Ensure numeric
    dfi[COL_DAMPER] = pd.to_numeric(dfi[COL_DAMPER], errors='coerce')
    dfi[COL_RH_SP]  = pd.to_numeric(dfi[COL_RH_SP],  errors='coerce')

    # Your two experiment windows
    exp_windows = [
        (pd.Timestamp('2025-09-17 22:31'), pd.Timestamp('2025-09-18 01:08')),
        (pd.Timestamp('2025-10-16 12:45'), pd.Timestamp('2025-10-16 14:04')),
    ]

    def plot_damper_vs_setpoint(df_src, t_start, t_end, pad_minutes=PAD_MIN):
        pad = pd.Timedelta(minutes=pad_minutes)
        t0, t1 = t_start - pad, t_end + pad

        # Slice data for the plot window (with padding)
        dfw = df_src.loc[t0:t1, [COL_DAMPER, COL_RH_SP]].dropna(how='all')
        if dfw.empty:
            return

        fig, ax = plt.subplots(figsize=(12,5))

        # Plot damper status (left axis)
        ax.plot(dfw.index, dfw[COL_DAMPER], lw=1.8, color='tab:blue', label=COL_DAMPER)
        ax.set_xlabel('Time')
        ax.set_ylabel('Fresh air damper', color='tab:blue')

        # Reasonable y-limits for damper
        y = dfw[COL_DAMPER].dropna()
        if not y.empty:
            ymin, ymax = float(y.min()), float(y.max())
            if 0 <= ymin and ymax <= 1.0:
                lo = max(0.0, ymin - 0.05); hi = min(1.0, ymax + 0.05)
                if hi - lo < 0.2: hi = lo + 0.2
                ax.set_ylim(lo, hi)
            elif 0 <= ymin and ymax <= 100:
                lo = max(0.0, ymin - 2.0); hi = min(100.0, ymax + 2.0)
                if hi - lo < 5.0: hi = lo + 5.0
                ax.set_ylim(lo, hi)

        # RH setpoint (right axis)
        ax2 = ax.twinx()
        ax2.plot(dfw.index, dfw[COL_RH_SP], ls='--', lw=1.2, alpha=0.9, color='gray', label='Setpoint RH')
        ax2.set_ylabel('Setpoint RH [%]', color='gray')

        # Time axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b\n%H:%M'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())

        # Titles & legend
        ax.set_title(
            f"Fresh air damper vs RH setpoint\n"
            f"{t_start:%d-%b %Y %H:%M} → {t_end:%d-%b %Y %H:%M}"
        )

        # Combined legend from both axes
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        uniq = dict(zip(l1 + l2, h1 + h2))
        ax.legend(uniq.values(), uniq.keys(), loc='best')

        plt.tight_layout()
        plt.show(block=True)

    # Generate plots sequentially
    for (t_start, t_end) in exp_windows:
        plot_damper_vs_setpoint(dfi, t_start, t_end)

else:
    print(f"⚠️ Missing columns: need '{COL_DAMPER}' and '{COL_RH_SP}'.")


import pandas as pd
import numpy as np

FILE_PATH = r"M:\PhD\03 Experiments\Complete_17-09-2025_16-10-2025_with_AHU_power.csv"

# Column names in your file
COL_DT   = "datetime"
COL_VS   = "Supply air flow rate (cb.m/hr)"   # m³/h, comma decimals
COL_VE   = "Extract air flow rate (cb.m/hr)"  # m³/h, comma decimals
COL_TS   = "Supply_air_temp"
COL_RHS  = "Supply_air_RH"
COL_TE   = "Extract_air_Temp"
COL_RHE  = "Extract_air_RH"
COL_TO   = "Outdoor_Temp"
COL_RHO  = "Outdoor_RH"

P_ATM = 101325.0  # Pa

def psat_pa(Tc):
    return 611.21 * np.exp((18.678 - Tc/234.5) * (Tc/(257.14 + Tc)))

def humidity_ratio(Tc, RH_frac, p_atm=P_ATM):
    p_ws = psat_pa(Tc); p_w = np.clip(RH_frac, 0, 1) * p_ws
    return 0.621945 * p_w / (p_atm - p_w)

def moist_air_density(Tc, RH_frac, p_atm=P_ATM):
    T = Tc + 273.15
    p_ws = psat_pa(Tc); p_w = np.clip(RH_frac, 0, 1) * p_ws
    p_da = p_atm - p_w
    R_da, R_v = 287.058, 461.495
    return p_da/(R_da*T) + p_w/(R_v*T)

df = pd.read_csv(FILE_PATH)
df[COL_DT] = pd.to_datetime(df[COL_DT], dayfirst=True, errors="coerce")
df = df.sort_values(COL_DT).set_index(COL_DT).loc[lambda x: x.index.notna()]

# --- KEY FIX: convert comma-decimal strings to floats ---
for c in [COL_VS, COL_VE]:
    df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")

# Other numerics
for c in [COL_TS, COL_RHS, COL_TE, COL_RHE, COL_TO, COL_RHO]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Volumetric flows (m³/s)
df["Vdot_sup_m3s"] = df[COL_VS] / 3600.0
df["Vdot_ext_m3s"] = df[COL_VE] / 3600.0

# RH fraction
rhs = (df[COL_RHS]/100).clip(0,1)
rhe = (df[COL_RHE]/100).clip(0,1)
rho = (df[COL_RHO]/100).clip(0,1)

# Psychrometrics
df["w_sup"]  = humidity_ratio(df[COL_TS], rhs, P_ATM)
df["w_ext"]  = humidity_ratio(df[COL_TE], rhe, P_ATM)
df["w_out"]  = humidity_ratio(df[COL_TO], rho, P_ATM)
df["rho_sup"] = moist_air_density(df[COL_TS], rhs, P_ATM)
df["rho_ext"] = moist_air_density(df[COL_TE], rhe, P_ATM)

# Mass flows
df["mdot_sup_moist"] = df["rho_sup"] * df["Vdot_sup_m3s"]
df["mdot_ext_moist"] = df["rho_ext"] * df["Vdot_ext_m3s"]
df["mdot_sup_dry"]   = df["mdot_sup_moist"] / (1.0 + df["w_sup"])
df["mdot_ext_dry"]   = df["mdot_ext_moist"] / (1.0 + df["w_ext"])

# Infiltration
df["inf_moist_kg_s"] = df["mdot_ext_moist"] - df["mdot_sup_moist"]
df["inf_dry_kg_s"]   = df["mdot_ext_dry"]   - df["mdot_sup_dry"]

# Optional moisture-balance estimate (use only in quiet periods)
df["mdot_water_ext"] = df["mdot_ext_dry"] * df["w_ext"]
df["mdot_water_sup"] = df["mdot_sup_dry"] * df["w_sup"]
df["inf_dry_from_water_kg_s"] = (df["mdot_water_ext"] - df["mdot_water_sup"]) / df["w_out"].replace(0, np.nan)

out_path = FILE_PATH.replace(".csv", "_with_infiltration.csv")
keep = [
    "Vdot_sup_m3s","Vdot_ext_m3s",
    "rho_sup","rho_ext","w_sup","w_ext","w_out",
    "mdot_sup_moist","mdot_ext_moist","mdot_sup_dry","mdot_ext_dry",
    "inf_moist_kg_s","inf_dry_kg_s","inf_dry_from_water_kg_s"
]
df[keep].to_csv(out_path)
print(f"Wrote: {out_path}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PATH = r"M:\PhD\03 Experiments\Complete_17-09-2025_16-10-2025_with_AHU_power_with_infiltration.csv"

def parse_time_col(df):
    # pick likely time col
    for c in ["time","datetime","timestamp","Dato","date"]:
        if c in df.columns:
            t = pd.to_datetime(df[c], errors="coerce")
            if t.notna().any(): return t
    # fallback: try every col
    best = None; frac = 0
    for c in df.columns:
        t = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
        f = t.notna().mean()
        if f > frac:
            best, frac = t, f
    return best

df = pd.read_csv(PATH)
t = parse_time_col(df)
df = df.assign(time=t).dropna(subset=["time"]).sort_values("time")
df = df.set_index("time")

# Prefer dry-air mass flows if present; fallback to volume flows
if {"mdot_sup_dry","mdot_ext_dry"}.issubset(df.columns):
    ms = pd.to_numeric(df["mdot_sup_dry"], errors="coerce")
    me = pd.to_numeric(df["mdot_ext_dry"], errors="coerce")
    units = "kg/s (dry air)"
else:
    # volume flows in cb.m/hr -> m³/s
    vs = pd.to_numeric(df.get("Supply air flow rate (cb.m/hr)"), errors="coerce")/3600.0
    ve = pd.to_numeric(df.get("Extract air flow rate (cb.m/hr)"), errors="coerce")/3600.0
    ms, me = vs, ve
    units = "m³/s"

Δm = (me - ms).rename("imbalance")
Δm_rolling = Δm.rolling("30min", min_periods=1).mean()

pos_share = (Δm > 0).mean()*100.0
print(f"Samples with Extract > Supply: {pos_share:.1f}%")

plt.figure(figsize=(12,4))
plt.plot(Δm.index, Δm, lw=1, label="Instantaneous imbalance (Extract - Supply)")
plt.plot(Δm_rolling.index, Δm_rolling, lw=2, ls="--", label="30-min rolling mean")
plt.axhline(0, lw=1, color="k")
plt.title(f"Supply vs Extract imbalance — {units}")
plt.ylabel(units); plt.xlabel("Time")
plt.legend(); plt.grid(True)
plt.tight_layout()

plt.figure(figsize=(6,4))
Δm.dropna().hist(bins=40)
plt.title("Imbalance distribution (Extract − Supply)")
plt.xlabel(units); plt.ylabel("Count"); plt.grid(True)
plt.tight_layout()
plt.show()
