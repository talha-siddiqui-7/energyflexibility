#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import datetime as dt
import numpy as np
from matplotlib.dates import DateFormatter, HourLocator

# ===========================================================
# FILE PATHS & COLUMNS
# ===========================================================
FILE_PATH       = r"M:\PhD\03 Experiments\17-09-2025_16-10-2025_sensor_vs_AHU_data.csv"
# FILE_PATH    = r"/mnt/data/17-09-2025_16-10-2025_sensor_vs_AHU_data.csv"  # for local testing

# Pool temperatures
COL_POOL            = 'Pool_OF'                    # external sensor
COL_POOL_AHU        = 'Pool_water_temp_AHU'        # AHU-side
COL_POOL_SP         = 'Setpoint Pool water temperature'

# RH columns (both variants)
COL_RH_SENSOR       = 'Extract_air_RH_sensor'
COL_RH_AHU          = 'Extract_air_RH_AHU'
COL_RH_SP           = 'Setpoint RH'

# Keep legacy name for existing overall plot; default to the external sensor
COL_RH              = COL_RH_SENSOR

# Damper
COL_DAMPER          = 'Fresh air damper'

# ===========================================================
# TOGGLES
# ===========================================================
ASK_FOR_TOGGLE = True

# 1) Setpoint overlay toggle
SHOW_SETPOINTS = True
if ASK_FOR_TOGGLE:
    try:
        ans = input("Overlay setpoint lines on weekly plots? [y/N]: ").strip().lower()
        SHOW_SETPOINTS = (ans == '' or ans.startswith('y'))
    except Exception:
        SHOW_SETPOINTS = True  # safe default in non-interactive runs

# 2) Pool source toggle for weekly plots
#    'ext' => external only, 'ahu' => AHU only, 'both' => both lines together
POOL_SOURCE = 'both'
if ASK_FOR_TOGGLE:
    try:
        sel = input("Pool temp source for weekly plots? [B]oth / [E]xternal / [A]HU (default B): ").strip().lower()
        if sel.startswith('e'): POOL_SOURCE = 'ext'
        elif sel.startswith('a'): POOL_SOURCE = 'ahu'
        else: POOL_SOURCE = 'both'
    except Exception:
        POOL_SOURCE = 'both'

# 3) Representative-day plots: include external sensor?
INCLUDE_EXT_REP_DAYS = True
if ASK_FOR_TOGGLE:
    try:
        sel2 = input("Include EXTERNAL sensor on representative-day plots? [Y/n]: ").strip().lower()
        INCLUDE_EXT_REP_DAYS = (sel2 == '' or sel2.startswith('y'))
    except Exception:
        INCLUDE_EXT_REP_DAYS = True

# ===========================================================
# LOAD DATA & PREPROCESS
# ===========================================================
df = pd.read_csv(FILE_PATH)

# Parse datetime
df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True, errors='coerce')
df = df.sort_values('datetime').set_index('datetime')
df = df.loc[df.index.notna()]

# Coerce numeric (handles comma decimals too)
non_numeric_cols = {'date', 'time'}
for c in df.columns:
    if c not in non_numeric_cols and df[c].dtype == 'object':
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False), errors='coerce')

# Interpolate numeric columns
dfi = df.copy()
num_cols = dfi.select_dtypes(include='number').columns
dfi[num_cols] = dfi[num_cols].interpolate(method='linear', limit_direction='both')

# Pool smoothing (30 min) for both sources
if COL_POOL in dfi.columns:
    dfi['Pool_OF_smooth_30min'] = dfi[COL_POOL].rolling(window=30, min_periods=1, center=True).mean()
if COL_POOL_AHU in dfi.columns:
    dfi['Pool_AHU_smooth_30min'] = dfi[COL_POOL_AHU].rolling(window=30, min_periods=1, center=True).mean()

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

    # shaded active regions
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

def weekly_pool_plot(df_week, sp_col, show_sp=True):
    """Weekly pool plot with toggle: External / AHU / Both."""
    ax = plt.gca()
    t = df_week.index
    act = df_week['pool_active']

    lines = []
    if POOL_SOURCE in ('ext', 'both') and 'Pool_OF_smooth_30min' in df_week.columns:
        lines.append((df_week['Pool_OF_smooth_30min'], 'Pool water (External, 30-min)', dict(lw=2.0)))
    if POOL_SOURCE in ('ahu', 'both') and 'Pool_AHU_smooth_30min' in df_week.columns:
        lines.append((df_week['Pool_AHU_smooth_30min'], 'Pool water (AHU, 30-min)', dict(lw=1.8, ls='--')))

    # Plot the lines
    for series, label, style in lines:
        ax.plot(t, series, label=label, **style)

    # shaded active regions
    active_start = None
    for i in range(len(df_week)):
        now = act.iloc[i]
        ts = df_week.index[i]
        if now and active_start is None:
            active_start = ts
        elif (not now or i == len(df_week)-1) and active_start is not None:
            ts_end = ts if not now else df_week.index[i]
            ax.axvspan(active_start, ts_end, color='red', alpha=0.08)
            active_start = None

    # setpoint overlay
    if show_sp and sp_col in df_week.columns:
        ax.plot(df_week.index, df_week[sp_col], ls=':', lw=1.1, alpha=0.9, label=f'{sp_col}')

    # axes, limits, legend
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b\n%H:%M'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    series_for_ylim = [s for (s, _, _) in lines]
    if show_sp and sp_col in df_week.columns:
        series_for_ylim.append(df_week[sp_col])
    auto_ylim(ax, series_for_ylim, pad_fraction=0.03)

    wk = df_week.index[0]
    wk_str = f"{wk.isocalendar().year}-W{wk.isocalendar().week:02d}"
    src_lab = {'ext':'External', 'ahu':'AHU', 'both':'External + AHU'}[POOL_SOURCE]
    ax.set_title(f"Pool Water Temperature ({src_lab}) — Week {wk_str}")
    ax.set_xlabel("Time"); ax.set_ylabel("Pool water [°C]")

    h, l = ax.get_legend_handles_labels()
    uniq = dict(zip(l, h))
    if uniq:
        ax.legend(uniq.values(), uniq.keys(), loc='best', fontsize=9)
    plt.tight_layout()

# ===========================================================
# OVERALL PLOTS (NO SETPOINTS)
# ===========================================================
# 1) Pool overall (show both raw lines if available)
plt.figure(figsize=(13,5))
if COL_POOL in dfi.columns:
    plt.plot(dfi.index, dfi[COL_POOL], lw=0.6, alpha=0.35, label='Pool_OF (raw 1-min)')
if COL_POOL_AHU in dfi.columns:
    plt.plot(dfi.index, dfi[COL_POOL_AHU], lw=0.6, alpha=0.35, label='Pool_water_temp_AHU (raw 1-min)')
if 'Pool_OF_smooth_30min' in dfi.columns:
    plt.plot(dfi.index, dfi['Pool_OF_smooth_30min'], lw=2.0, label='Pool_OF (30-min smooth)')
if 'Pool_AHU_smooth_30min' in dfi.columns:
    plt.plot(dfi.index, dfi['Pool_AHU_smooth_30min'], lw=2.0, ls='--', label='Pool_AHU (30-min smooth)')
plt.title('Pool Water Temperature Over Time')
plt.xlabel('Time'); plt.ylabel('Pool water [°C]')
style_time_axis(plt.gca())
plt.legend(); plt.tight_layout(); plt.show(block=True)

# 2) RH overall (external sensor by default, preserving legacy COL_RH usage)
plt.figure(figsize=(13,5))
plt.plot(dfi.index, dfi[COL_RH], lw=1.0, label=f'{COL_RH} (raw 1-min)')
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

# ===================== Weekly plots =====================
for wk in unique_weeks:
    dfw = dfi.loc[week_ids == wk].copy()
    if dfw.empty:
        continue

    # ---- Pool water weekly (with source toggle) ----
    plt.figure(figsize=(13,5))
    weekly_pool_plot(dfw, sp_col=COL_POOL_SP, show_sp=SHOW_SETPOINTS)
    plt.show(block=True)

    # ---- Legacy weekly RH (external by default via COL_RH) ----
    plt.figure(figsize=(13,5))
    weekly_plot(
        dfw, COL_RH, COL_RH_SP,
        'Extract Air Relative Humidity (raw 1-min)', 'Extract_air_RH [%RH]',
        rh_mode=True, show_sp=SHOW_SETPOINTS
    )
    plt.show(block=True)

# ===========================================================
# FRESH AIR DAMPER — show vs RH setpoint (no shading)
# ===========================================================
PAD_MIN = 15  # context minutes around experiment window

if COL_DAMPER in dfi.columns and COL_RH_SP in dfi.columns:
    dfi[COL_DAMPER] = pd.to_numeric(dfi[COL_DAMPER], errors='coerce')
    dfi[COL_RH_SP]  = pd.to_numeric(dfi[COL_RH_SP],  errors='coerce')

    exp_windows = [
        (pd.Timestamp('2025-09-17 22:31'), pd.Timestamp('2025-09-18 01:08')),
        (pd.Timestamp('2025-10-16 12:45'), pd.Timestamp('2025-10-16 14:04')),
    ]

    def plot_damper_vs_setpoint(df_src, t_start, t_end, pad_minutes=PAD_MIN):
        pad = pd.Timedelta(minutes=pad_minutes)
        t0, t1 = t_start - pad, t_end + pad
        dfw = df_src.loc[t0:t1, [COL_DAMPER, COL_RH_SP]].dropna(how='all')
        if dfw.empty:
            return

        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(dfw.index, dfw[COL_DAMPER], lw=1.8, label=COL_DAMPER)
        ax.set_xlabel('Time'); ax.set_ylabel('Fresh air damper')

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

        ax2 = ax.twinx()
        ax2.plot(dfw.index, dfw[COL_RH_SP], ls='--', lw=1.2, alpha=0.9, label='Setpoint RH')
        ax2.set_ylabel('Setpoint RH [%]')

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b\n%H:%M'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())

        ax.set_title(f"Fresh air damper vs RH setpoint\n{t_start:%d-%b %Y %H:%M} → {t_end:%d-%b %Y %H:%M}")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        uniq = dict(zip(l1 + l2, h1 + h2))
        ax.legend(uniq.values(), uniq.keys(), loc='best')
        plt.tight_layout(); plt.show(block=True)

    for (t_start, t_end) in exp_windows:
        plot_damper_vs_setpoint(dfi, t_start, t_end)
else:
    print(f"⚠️ Missing columns: need '{COL_DAMPER}' and '{COL_RH_SP}'.")

# ===========================================================
# REPRESENTATIVE AVERAGE DAY (PER WEEKDAY) — exclude experiments
# Plots mean profiles of RH (AHU + optional External), Setpoint RH, and Fresh Air Damper
# ===========================================================
exp_windows = [
    (pd.Timestamp('2025-09-17 22:31'), pd.Timestamp('2025-09-18 01:08')),
    (pd.Timestamp('2025-10-16 12:45'), pd.Timestamp('2025-10-16 14:04')),
]
mask_exp = pd.Series(False, index=dfi.index)
for t0, t1 in exp_windows:
    mask_exp |= (dfi.index >= t0) & (dfi.index <= t1)

exclude_days = {pd.Timestamp('2025-09-17').date(),
                pd.Timestamp('2025-09-18').date(),
                pd.Timestamp('2025-10-16').date()}
mask_days_excl = np.isin(dfi.index.date.astype('O'), list(exclude_days))
mask_drop = mask_exp | mask_days_excl
dff = dfi.loc[~mask_drop].copy()

# required columns (AHU + setpoint + damper; external optional)
base_cols = [COL_RH_AHU, COL_RH_SP, COL_DAMPER]
opt_cols  = [COL_RH_SENSOR] if INCLUDE_EXT_REP_DAYS else []
need_cols = base_cols + opt_cols

missing = [c for c in need_cols if c not in dff.columns]
if missing:
    print(f"⚠️ Missing columns for representative weekday plots: {missing}")
else:
    dff['min_of_day'] = dff.index.hour * 60 + dff.index.minute
    dff['weekday']    = dff.index.weekday  # 0=Mon..Sun

    grp        = dff.groupby(['weekday', 'min_of_day'])
    prof_mean  = grp[need_cols].mean()

    weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    for wd in range(7):
        sub = prof_mean.loc[wd]                  # DataFrame indexed by 0..1439 minutes
        x = pd.to_datetime(sub.index, unit='m', origin='2000-01-01')

        fig, ax = plt.subplots(figsize=(12,4.8))

        # RH lines
        ax.plot(x, sub[COL_RH_AHU], lw=1.8, label='Extract RH (AHU)')
        if INCLUDE_EXT_REP_DAYS and COL_RH_SENSOR in sub.columns:
            ax.plot(x, sub[COL_RH_SENSOR], lw=1.4, label='Extract RH (External)')
        ax.plot(x, sub[COL_RH_SP], lw=1.2, ls='--', label='Setpoint RH')

        ax.set_title(f"Representative Average Day — {weekday_names[wd]} (excl. experiment days)")
        ax.set_ylabel("RH [%]")
        ax.set_xlabel("Time of day")

        # Right axis for damper
        if COL_DAMPER in sub.columns:
            ax2 = ax.twinx()
            ax2.plot(x, sub[COL_DAMPER], lw=1.2, alpha=0.8, label='Fresh air damper', color='tab:gray')
            # Auto-scale for 0..1 or 0..100
            dvals = sub[COL_DAMPER].dropna()
            if not dvals.empty:
                dmin, dmax = float(dvals.min()), float(dvals.max())
                if 0 <= dmin and dmax <= 1.0:
                    ax2.set_ylim(-0.05, 1.05)
                    ax2.set_ylabel("Damper [0–1]")
                else:
                    lo = max(0.0, dmin - 2.0); hi = min(100.0, dmax + 2.0)
                    if hi - lo < 5: hi = lo + 5
                    ax2.set_ylim(lo, hi)
                    ax2.set_ylabel("Damper [%]")

            # Build combined legend
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            uniq = dict(zip(l1 + l2, h1 + h2))
            ax.legend(uniq.values(), uniq.keys(), loc='best', fontsize=9)
        else:
            ax.legend(loc='best', fontsize=9)

        # X/Y formatting
        ax.xaxis.set_major_locator(HourLocator(interval=2))
        ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
        ax.grid(True, which="both", axis="both", alpha=0.25)

        # RH y-limits clamped to [0,100] with small pad
        ycols = [COL_RH_AHU, COL_RH_SP] + ([COL_RH_SENSOR] if (INCLUDE_EXT_REP_DAYS and COL_RH_SENSOR in sub.columns) else [])
        yvals = sub[ycols].to_numpy().ravel()
        yvals = yvals[~np.isnan(yvals)]
        if yvals.size > 0:
            lo, hi = float(np.nanmin(yvals)), float(np.nanmax(yvals))
            pad = max(2.0, 0.05*(hi - lo))
            ax.set_ylim(max(0, lo - pad), min(100, hi + pad))

        plt.tight_layout()
        plt.show(block=True)

# ===========================================================
# REPRESENTATIVE AVERAGE DAY (PER WEEKDAY)
# Extract/Supply RH from External sensors + AHU — one plot per weekday
# ===========================================================
need_cols2 = [
    'Extract_air_RH_sensor',
    'Supply_air_RH_sensor',
    'Extract_air_RH_AHU',
    'Supply_air_RH_AHU'
]
missing2 = [c for c in need_cols2 if c not in dff.columns]
if missing2:
    print(f"⚠️ Missing columns for Extract/Supply representative plots: {missing2}")
else:
    dff['min_of_day'] = dff.index.hour * 60 + dff.index.minute
    dff['weekday']    = dff.index.weekday
    grp2 = dff.groupby(['weekday', 'min_of_day'])
    prof_mean2 = grp2[need_cols2].mean()
    weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    for wd in range(7):
        sub = prof_mean2.loc[wd]
        x = pd.to_datetime(sub.index, unit='m', origin='2000-01-01')

        fig, ax = plt.subplots(figsize=(12,4.8))
        ax.plot(x, sub['Extract_air_RH_sensor'], lw=1.6, label='Extract RH (External)')
        ax.plot(x, sub['Supply_air_RH_sensor'],  lw=1.6, label='Supply RH (External)')
        ax.plot(x, sub['Extract_air_RH_AHU'],    lw=1.2, ls='--', label='Extract RH (AHU)')
        ax.plot(x, sub['Supply_air_RH_AHU'],     lw=1.2, ls='--', label='Supply RH (AHU)')

        ax.set_title(f"Representative Average Day — {weekday_names[wd]}\n(excluding experiment days)")
        ax.set_ylabel("RH [%]")
        ax.set_xlabel("Time of day")
        ax.xaxis.set_major_locator(HourLocator(interval=2))
        ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
        ax.grid(True, which="both", axis="both", alpha=0.25)
        yvals = sub[need_cols2].to_numpy().ravel()
        yvals = yvals[~np.isnan(yvals)]
        if yvals.size > 0:
            lo, hi = float(np.nanmin(yvals)), float(np.nanmax(yvals))
            pad = max(2.0, 0.05*(hi - lo))
            ax.set_ylim(max(0, lo - pad), min(100, hi + pad))
        ax.legend(loc="best", fontsize=9)
        plt.tight_layout()
        plt.show(block=True)
