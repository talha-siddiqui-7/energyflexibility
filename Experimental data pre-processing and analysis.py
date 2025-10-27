import pandas as pd
import matplotlib.pyplot as plt

# --- Load and interpolate (same as before) ---
file_path = r"M:\PhD\03 Experiments\Complete_17-09-2025_16-10-2025_formatted_1min_mean_only.csv"
df = pd.read_csv(file_path)

# datetime handling
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').set_index('datetime')

# linear interpolation for missing values
df_interpolated = df.interpolate(method='linear', limit_direction='both')

# --- Create 30-minute smoothed version of Pool_OF for visualization ---
# rolling window of 30 minutes = 30 samples (because data is 1 point per minute)
df_interpolated['Pool_OF_smooth_30min'] = (
    df_interpolated['Pool_OF']
    .rolling(window=30, min_periods=1, center=True)
    .mean()
)

# --- Make the plot ---
plt.figure(figsize=(12,5))

# raw signal (1-min)
plt.plot(
    df_interpolated.index,
    df_interpolated['Pool_OF'],
    linewidth=0.5,
    alpha=0.4,
    label='Pool_OF (raw 1-min)'
)

# smoothed signal (30-min rolling mean)
plt.plot(
    df_interpolated.index,
    df_interpolated['Pool_OF_smooth_30min'],
    linewidth=2,
    label='Pool_OF (30-min smooth)'
)

plt.title("Pool Water Temperature Over Time")
plt.xlabel("Time")
plt.ylabel("Pool_OF [°C]")
plt.legend()
plt.tight_layout()
plt.show()

# ===============================
# EXTRA CODE (append after your current plt.show())
# ===============================

import datetime as dt
import matplotlib.dates as mdates

# 1. DEFINE WEEKLY OCCUPANCY SCHEDULE
# Fill this with the ORANGE BOX periods from your calendar.
# Keys are weekday names (0 = Monday ... 6 = Sunday).
# Each value is a list of (start_time, end_time) tuples in 24h "HH:MM" format.
#
# EXAMPLE BELOW. You MUST adjust these to match your real orange blocks.
weekly_schedule = {
    0: [("08:00", "09:00"),
        ("13:00", "15:00"),
        ("17:00", "20:00")],      # Monday
    1: [("09:00", "11:00"),
        ("17:00", "21:00")],      # Tuesday
    2: [("09:00", "11:00"),
        ("17:00", "20:00")],      # Wednesday
    3: [("09:00", "11:00"),
        ("17:00", "21:00")],      # Thursday
    4: [("08:00", "14:00"),
        ("17:00", "19:00")],      # Friday
    5: [("13:00", "15:00")],      # Saturday
    6: [("13:00", "17:00")]       # Sunday
}

# Helper to check if a timestamp falls into any active block that day
def is_active(ts):
    weekday = ts.weekday()  # 0=Mon ... 6=Sun
    if weekday not in weekly_schedule:
        return False
    # For each (start,end) interval that day:
    for start_str, end_str in weekly_schedule[weekday]:
        start_t = dt.datetime.combine(ts.date(), dt.time.fromisoformat(start_str))
        end_t   = dt.datetime.combine(ts.date(), dt.time.fromisoformat(end_str))
        if start_t <= ts <= end_t:
            return True
    return False

# 2. CREATE activity column on the interpolated dataframe
df_interpolated['pool_active'] = df_interpolated.index.to_series().apply(is_active)

# 3. We'll plot per calendar week.
# Create a "week id" using ISO week (year-week) so weeks don't mix across month boundary
week_ids = df_interpolated.index.to_series().apply(lambda t: f"{t.isocalendar().year}-W{t.isocalendar().week:02d}")
unique_weeks = week_ids.unique()

# 4. Loop through each week and make a plot
for wk in unique_weeks:
    week_mask = (week_ids == wk)
    df_week = df_interpolated.loc[week_mask].copy()

    if df_week.empty:
        continue

    # We'll plot in segments where activity state is constant.
    # For readability we use the smoothed signal.
    t_values = df_week.index
    y_values = df_week['Pool_OF_smooth_30min']
    active_values = df_week['pool_active']

    # Prepare figure
    plt.figure(figsize=(12,5))

    # Walk through contiguous segments of same active/inactive state
    seg_start_idx = 0
    for i in range(1, len(df_week)):
        # if activity state changes, we draw the previous segment
        if active_values.iloc[i] != active_values.iloc[i-1]:
            seg_t = t_values[seg_start_idx:i]
            seg_y = y_values.iloc[seg_start_idx:i]

            if active_values.iloc[i-1]:
                seg_color = 'red'    # active (people in pool)
                seg_label = 'Active / occupied'
            else:
                seg_color = 'blue'   # inactive (empty)
                seg_label = 'Inactive / empty'

            plt.plot(seg_t, seg_y, linewidth=2, color=seg_color, label=seg_label)

            seg_start_idx = i

    # draw the last segment
    seg_t = t_values[seg_start_idx:len(df_week)]
    seg_y = y_values.iloc[seg_start_idx:len(df_week)]
    if active_values.iloc[-1]:
        seg_color = 'red'
        seg_label = 'Active / occupied'
    else:
        seg_color = 'blue'
        seg_label = 'Inactive / empty'
    plt.plot(seg_t, seg_y, linewidth=2, color=seg_color, label=seg_label)

    # cosmetics
    plt.title(f"Pool Water Temperature (30-min smooth) — Week {wk}")
    plt.xlabel("Time")
    plt.ylabel("Pool_OF [°C]")

    # Make x-axis nicer (day + hour ticks)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%b\n%H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())

    # Build a clean legend with unique labels only
    handles, labels = plt.gca().get_legend_handles_labels()
    seen = {}
    uniq_handles = []
    uniq_labels = []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = True
            uniq_handles.append(h)
            uniq_labels.append(l)
    plt.legend(uniq_handles, uniq_labels, loc='best')

    plt.tight_layout()
    plt.show()
