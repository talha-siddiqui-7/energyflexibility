#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection

# ---------- File path ----------
file_path_win = r"M:\PhD\03 Experiments\Pool_Thermal_energydata_experiment_period.xlsx"
file_path_fallback = "/mnt/data/Pool_Thermal_energydata_experiment_period.xlsx"
file_path = file_path_win if os.path.exists(file_path_win) else file_path_fallback

# ---------- Columns ----------
TIME_COL   = "Time"
TEMP_COL   = "Pool water Temperatur [°C]"
LEVEL_COL  = "Holding tank water level [cm]"
POOL_Q_COL = "Pool thermal energy input [kW]"

# ---------- Load ----------
df = pd.read_excel(file_path)

# Fix date parsing warning
df[TIME_COL] = pd.to_datetime(df[TIME_COL], dayfirst=True, errors="coerce")
df = df.dropna(subset=[TIME_COL]).sort_values(TIME_COL)

# Keep only required columns (ADDED POOL_Q_COL)
df = df[[TIME_COL, TEMP_COL, LEVEL_COL, POOL_Q_COL]].copy()
df = df.dropna(subset=[TEMP_COL, LEVEL_COL])  # keep NaNs in POOL_Q_COL if any

# ---------- Create Activity factor ----------
df["Activity factor"] = df[LEVEL_COL].apply(lambda x: 1 if x > 75 else 0.5)
df["is_active"] = df["Activity factor"] == 1

# ---------- Weekly key (ISO weeks) ----------
iso = df[TIME_COL].dt.isocalendar()
df["week_key"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)

# ---------- Plot helper (WEEK) ----------
def plot_week(week_df, week_label):
    # Temperature colored line
    x = mdates.date2num(week_df[TIME_COL].to_numpy())
    y_temp = week_df[TEMP_COL].to_numpy()

    points = list(zip(x, y_temp))
    segments = [[points[i], points[i + 1]] for i in range(len(points) - 1)]
    colors = ["red" if week_df["is_active"].iloc[i] else "blue"
              for i in range(len(points) - 1)]
    lc = LineCollection(segments, colors=colors, linewidths=2)

    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.add_collection(lc)
    ax1.set_xlim(x.min(), x.max())
    ax1.set_ylim(y_temp.min() - 0.1, y_temp.max() + 0.1)

    ax1.set_title(f"Weekly Pool Water Temperature + Thermal Input — {week_label}")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Pool water Temperatur [°C]")

    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m %H:%M"))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

    # Secondary axis: pool thermal energy input
    ax2 = ax1.twinx()
    ax2.plot(week_df[TIME_COL], week_df[POOL_Q_COL], linewidth=1.5)
    ax2.set_ylabel("Pool thermal energy input [kW]")

    # Legend (dummy lines for colored temperature + actual Q line)
    ax1.plot([], [], color="blue", label="Temp (Activity factor = 0.5)")
    ax1.plot([], [], color="red",  label="Temp (Activity factor = 1)")
    ax2.plot([], [], label="Pool thermal energy input [kW]")
    ax1.legend(loc="upper left")

    ax1.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# ---------- Plot helper (OVERALL) ----------
def plot_overall(all_df):
    # Temperature colored line
    x = mdates.date2num(all_df[TIME_COL].to_numpy())
    y_temp = all_df[TEMP_COL].to_numpy()

    points = list(zip(x, y_temp))
    segments = [[points[i], points[i + 1]] for i in range(len(points) - 1)]
    colors = ["red" if all_df["is_active"].iloc[i] else "blue"
              for i in range(len(points) - 1)]
    lc = LineCollection(segments, colors=colors, linewidths=1.8)

    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.add_collection(lc)
    ax1.set_xlim(x.min(), x.max())
    ax1.set_ylim(y_temp.min() - 0.1, y_temp.max() + 0.1)

    ax1.set_title("Overall Pool Water Temperature + Thermal Input (17 Sep – 16 Oct)")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Pool water Temperatur [°C]")

    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

    # Secondary axis: pool thermal energy input
    ax2 = ax1.twinx()
    ax2.plot(all_df[TIME_COL], all_df[POOL_Q_COL], linewidth=1.2)
    ax2.set_ylabel("Pool thermal energy input [kW]")

    # Legend
    ax1.plot([], [], color="blue", label="Temp (Activity factor = 0.5)")
    ax1.plot([], [], color="red",  label="Temp (Activity factor = 1)")
    ax2.plot([], [], label="Pool thermal energy input [kW]")
    ax1.legend(loc="upper left")

    ax1.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# ---------- OVERALL plot ----------
if len(df) > 2:
    plot_overall(df)

# ---------- Weekly plots ----------
for week, g in df.groupby("week_key", sort=True):
    if len(g) > 2:
        plot_week(g, week)
