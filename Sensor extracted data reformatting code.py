#!/usr/bin/env python3
"""
Reformat swimming pool sensor exports and aggregate to 1-minute intervals.
Author: Talha Siddiqui (PhD: Energy flexibility in swimming pool facilities)
"""

import pandas as pd
from pathlib import Path

# -------------------------------------------------------------------------
# 1️⃣  USER SETTINGS
# -------------------------------------------------------------------------
input_file = Path(r"M:\PhD\03 Experiments\EXPORT_16-09-2025_13-10-2025.csv")
output_prefix = input_file.with_name(input_file.stem + "_formatted")

# -------------------------------------------------------------------------
# 2️⃣  LOAD AND PARSE
# -------------------------------------------------------------------------
df = pd.read_csv(input_file)

# Robust timestamp parsing
df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time"], errors="coerce")
df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

# Base time index (union of all sensor times)
base = pd.DataFrame({"timestamp": pd.Index(sorted(df["timestamp"].unique()))}).set_index("timestamp")


def series_from_filter(name, subset):
    """Return a Series of 'sample' values indexed by timestamp."""
    s = subset.set_index("timestamp")["sample"].sort_index()
    s.name = name
    return s


# -------------------------------------------------------------------------
# 3️⃣  VARIABLE MAPPING  (your fixed structure)
# -------------------------------------------------------------------------
wide = base.copy()

# Air_exhaust
wide = wide.join(series_from_filter("Air_exhaust", df[df["friendly name"].eq("Air_exhaust")]), how="left")

# Extract_air_AHU
mask = df["friendly name"].eq("Extract_air_AHU")
wide = wide.join(series_from_filter("Extract_air_RH", df[mask & df["sensor address"].eq("60-3-18")]), how="left")
wide = wide.join(series_from_filter("Extract_air_Temp", df[mask & df["sensor address"].eq("60-1-7")]), how="left")

# Outdoor
mask = df["friendly name"].eq("Outdoor")
wide = wide.join(series_from_filter("Outdoor_RH", df[mask & df["channel index"].eq(0)]), how="left")
wide = wide.join(series_from_filter("Outdoor_Temp", df[mask & df["channel index"].eq(1)]), how="left")

# Pool_OF
wide = wide.join(series_from_filter("Pool_OF", df[df["friendly name"].eq("Pool_OF")]), how="left")

# Supply_air_AHU → temperature
wide = wide.join(series_from_filter("Supply_air_temp", df[df["friendly name"].eq("Supply_air_AHU")]), how="left")

# Supply_RH_AHU → RH (channel 0 only)
mask = df["friendly name"].eq("Supply_RH_AHU")
wide = wide.join(series_from_filter("Supply_air_RH", df[mask & df["channel index"].eq(0)]), how="left")

# Technical_area
wide = wide.join(series_from_filter("Technical_area", df[df["friendly name"].eq("Technical_area")]), how="left")

# -------------------------------------------------------------------------
# 4️⃣  AGGREGATION: 1-minute grid
# -------------------------------------------------------------------------
wide = wide.sort_index()
mean_1min = wide.resample("1min").mean()
ffill_1min = mean_1min.ffill(limit=5)  # ≤5-minute hold

# -------------------------------------------------------------------------
# 5️⃣  ADD DATE & TIME COLUMNS + SAVE
# -------------------------------------------------------------------------
def add_meta(df_in):
    df_out = df_in.copy()
    df_out["date"] = df_out.index.date.astype(str)
    df_out["time"] = df_out.index.time.astype(str)
    ordered = ["date", "time"] + [c for c in df_in.columns]
    return df_out[ordered]


mean_1min_out = add_meta(mean_1min)
ffill_1min_out = add_meta(ffill_1min)

out_mean = output_prefix.with_name(output_prefix.stem + "_1min_mean_only.csv")
out_fill = output_prefix.with_name(output_prefix.stem + "_1min_ffill5min.csv")

mean_1min_out.to_csv(out_mean, index_label="datetime")
ffill_1min_out.to_csv(out_fill, index_label="datetime")

print(f"✅ Saved:\n - {out_mean}\n - {out_fill}")
