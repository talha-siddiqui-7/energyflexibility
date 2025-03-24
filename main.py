import numpy as np
import matplotlib.pyplot as plt

# Given values
T_min = 26.0  # °C (Heating ON threshold)
T_max = 26.5  # °C (Heating OFF threshold)
T_air_facility = T_min + 2  # Facility air temperature

# Pool properties
rho_water = 1000  # kg/m³
V_pool = 25 * 12 * 2  # m³
m_water = rho_water * V_pool  # kg
C_p_water = 4184  # J/kg·K

# Surface areas
A_pool = 25 * 12  # m² (Pool surface area for convection & evaporation)
A_s = 2 * (25 * 2 + 12 * 2) * 2  # m² (Side walls)
A_b = 25 * 12  # m² (Bottom area)

# User inputs
rel_humidity = float(input("Enter relative humidity (e.g., 60 for 60%): ")) / 100
F_a = float(input("Enter activity factor (F_a): "))

# Heating power calculation
delta_T = 0.5
t_heating_seconds = 3600
Q_heating_W = (m_water * C_p_water * delta_T) / t_heating_seconds

# Air properties
k_air = 0.025
nu_air = 15.89e-6
Pr_air = 0.7
V_air = 0.2
L_pool = 25

# Convective heat transfer
Re = (V_air * L_pool) / nu_air
Nu = 0.0296 * (Re**0.8) * (Pr_air**(1/3))
h_conv = (Nu * k_air) / L_pool
Q_conv = 2 * h_conv * A_pool

# Conductive heat transfer coefficients
U_s = 2.94
U_b = 0.5
T_rs = float(input("Enter T_rs (space temperature) in °C: "))
T_g = float(input("Enter T_g (ground temperature) in °C: "))

Q_cond = (U_s * A_s * (T_max - T_rs)) + (U_b * A_b * (T_max - T_g))

# **Evaporative Loss Calculation**
def P_sat(T):
    return 610.7 * np.exp((17.27 * T) / (T + 237.3))

def dew_point(T_air, RH):
    return (237.3 * np.log(RH) + (17.27 * T_air)) / (17.27 - np.log(RH))

Pw = P_sat(T_max)
T_dew = dew_point(T_air_facility, rel_humidity)
Pa = P_sat(T_dew)

Pw_inHg = Pw * 0.0002953
Pa_inHg = Pa * 0.0002953

M_evap_lb_hr = 0.1 * A_pool * (Pw_inHg - Pa_inHg) * F_a
M_evap_kg_hr = M_evap_lb_hr * 0.453592
M_evap_kg_s = M_evap_kg_hr / 3600

L = 2.43e6  # Latent heat of vaporization
Q_evap = M_evap_kg_s * L  # Evaporative heat loss in W

# **ON/OFF Time Calculations**
Q_net_cooling_conv = Q_conv  # Case 1: Only convective gain
Q_net_cooling_conv_cond = Q_conv - Q_cond  # Case 2: Adding conductive loss
Q_net_cooling_full = Q_conv - (Q_cond + Q_evap)  # Case 3: Adding evaporative loss

# **Heating ON Time**
t_ON_conv = abs((m_water * C_p_water * (T_max - T_min)) / (Q_heating_W + Q_conv))
t_ON_conv_cond = abs((m_water * C_p_water * (T_max - T_min)) / (Q_heating_W + Q_conv - Q_cond))
t_ON_full = abs((m_water * C_p_water * (T_max - T_min)) / (Q_heating_W + Q_conv - Q_cond - Q_evap))

# **Heating OFF Time**
t_OFF_conv = abs((m_water * C_p_water * (T_max - T_min)) / Q_net_cooling_conv)
t_OFF_conv_cond = abs((m_water * C_p_water * (T_max - T_min)) / Q_net_cooling_conv_cond)
t_OFF_full = abs((m_water * C_p_water * (T_max - T_min)) / Q_net_cooling_full)

# Convert OFF Time to Days
t_OFF_days_conv = t_OFF_conv / 86400
t_OFF_days_conv_cond = t_OFF_conv_cond / 86400
t_OFF_days_full = t_OFF_full / 86400

# **Total Simulation Time for 2 Full Cycles**
total_simulation_hours = 2 * (t_ON_full + t_OFF_full) / 3600

# **Convert Energy to kWh**
total_heating_kWh = 2*(Q_heating_W * t_ON_full) / 3600 / 1000
total_conv_kWh = (Q_conv * total_simulation_hours * 3600) / 3600 / 1000
total_cond_kWh = (Q_cond * total_simulation_hours * 3600) / 3600 / 1000
total_evap_kWh = (Q_evap * total_simulation_hours * 3600) / 3600 / 1000

# **Print Energy Balance Results**
print("\n===== Total Energy Balance (kWh) =====")
print(f"Total Heating Input: {total_heating_kWh:.2f} kWh")
print(f"Total Convective Gain: {total_conv_kWh:.2f} kWh")
print(f"Total Conductive Loss: {total_cond_kWh:.2f} kWh")
print(f"Total Evaporative Loss: {total_evap_kWh:.2f} kWh")

print("\n===== Heating ON Durations (in hours) =====")
print(f"Heating ON (Only Convective Gain): {t_ON_conv / 3600:.2f} hours")
print(f"Heating ON (Convective Gain - Conductive Loss): {t_ON_conv_cond / 3600:.2f} hours")
print(f"Heating ON (Convective Gain - Conductive Loss - Evaporative Loss): {t_ON_full / 3600:.2f} hours")

print("\n===== Heating OFF Durations (in hours and days) =====")
print(f"Heating OFF (Only Convective Gain): {t_OFF_conv / 3600:.2f} hours ({t_OFF_days_conv:.2f} days)")
print(f"Heating OFF (Convective Gain - Conductive Loss): {t_OFF_conv_cond / 3600:.2f} hours ({t_OFF_days_conv_cond:.2f} days)")
print(f"Heating OFF (Convective Gain - Conductive Loss - Evaporative Loss): {t_OFF_full / 3600:.2f} hours ({t_OFF_days_full:.2f} days)")

# **Simulation for Temperature Evolution**
time_step = 60
time = np.arange(0, total_simulation_hours * 3600, time_step)
temperature_conv = np.zeros_like(time, dtype=float)
temperature_conv_cond = np.zeros_like(time, dtype=float)
temperature_full = np.zeros_like(time, dtype=float)

# **Run Simulation for Each Case**
def simulate_temperature(t_ON, t_OFF, Q_gain, Q_loss, temp_array):
    T_current = T_min
    heating = True
    for i, t in enumerate(time):
        if heating:
            dT_dt = Q_gain / (m_water * C_p_water)
            T_current += dT_dt * time_step
            if T_current >= T_max:
                heating = False
        else:
            dT_dt = Q_loss / (m_water * C_p_water)
            T_current += dT_dt * time_step
            if T_current <= T_min:
                heating = True
        temp_array[i] = T_current

simulate_temperature(t_ON_full, t_OFF_full, Q_heating_W + Q_conv, Q_net_cooling_conv, temperature_conv)
simulate_temperature(t_ON_full, t_OFF_full, Q_heating_W + Q_conv - Q_cond, Q_net_cooling_conv_cond, temperature_conv_cond)
simulate_temperature(t_ON_full, t_OFF_full, Q_heating_W + Q_conv - Q_cond - Q_evap, Q_net_cooling_full, temperature_full)

plt.figure(figsize=(10, 5))
plt.plot(time / 3600, temperature_conv, label="Only Convective Gain", color='b')
plt.plot(time / 3600, temperature_conv_cond, label="Convective + Conductive Loss", color='orange')
plt.plot(time / 3600, temperature_full, label="Convective + Conductive + Evaporative Loss", color='r')
plt.axhline(T_max, linestyle="--", color="k")
plt.axhline(T_min, linestyle="--", color="g")
plt.xlabel("Time (hours)")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid()
plt.show()

# === Corrected Plot: Two Full Cycles for Convective & Convective + Conductive ===

# Define correct total time (2 full cycles) based on the two scenarios
t_total_2cycles_conv = 2 * (t_ON_conv + t_OFF_conv)
t_total_2cycles_cond = 2 * (t_ON_conv_cond + t_OFF_conv_cond)
t_max_sim = max(t_total_2cycles_conv, t_total_2cycles_cond)

time_2cycles = np.arange(0, t_max_sim, time_step)
temperature_conv_2 = np.zeros_like(time_2cycles, dtype=float)
temperature_conv_cond_2 = np.zeros_like(time_2cycles, dtype=float)

# Simulate two full cycles for each case
def simulate_cycles(t_ON, t_OFF, Q_gain, Q_loss, temp_array):
    T_current = T_min
    heating = True
    cycle_time = t_ON + t_OFF
    for i, t in enumerate(time_2cycles):
        cycle_progress = t % cycle_time
        if cycle_progress < t_ON:
            dT_dt = Q_gain / (m_water * C_p_water)
        else:
            dT_dt = Q_loss / (m_water * C_p_water)
        T_current += dT_dt * time_step
        T_current = min(max(T_current, T_min), T_max)  # clamp
        temp_array[i] = T_current

simulate_cycles(t_ON_conv, t_OFF_conv, Q_heating_W + Q_conv, Q_net_cooling_conv, temperature_conv_2)
simulate_cycles(t_ON_conv_cond, t_OFF_conv_cond, Q_heating_W + Q_conv - Q_cond, Q_net_cooling_conv_cond, temperature_conv_cond_2)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(time_2cycles / 3600, temperature_conv_2, label="Only Convective Gain", color='b')
plt.plot(time_2cycles / 3600, temperature_conv_cond_2, label="Convective + Conductive Loss", color='orange')
plt.axhline(T_max, linestyle="--", color="k", label="Heating OFF Threshold (26.5°C)")
plt.axhline(T_min, linestyle="--", color="g", label="Heating ON Threshold (26°C)")
plt.xlabel("Time (hours)")
plt.ylabel("Temperature (°C)")
plt.title("Convective vs Convective + Conductive Loss")
plt.legend()
plt.grid()
plt.show()

import matplotlib.ticker as ticker

# === Bar Chart: Convective + Conductive Loss Case ===
plt.figure(figsize=(6, 4))
plt.bar(['Heating ON', 'Heating OFF'],
        [t_ON_conv_cond / 3600, t_OFF_conv_cond / 3600],
        color=['orange', 'gray'])
plt.ylabel("Time (hours)")
plt.title("Heating ON/OFF Duration\n(Convective + Conductive Loss)")
plt.grid(axis='y', linestyle='--')
plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.show()

# === Bar Chart: Convective + Conductive + Evaporative Loss Case ===
plt.figure(figsize=(6, 4))
plt.bar(['Heating ON', 'Heating OFF'],
        [t_ON_full / 3600, t_OFF_full / 3600],
        color=['red', 'gray'])
plt.ylabel("Time (hours)")
plt.title("Heating ON/OFF Duration\n(Convective + Conductive + Evaporative Loss)")
plt.grid(axis='y', linestyle='--')
plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.show()

