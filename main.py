import math

# --- Universal Constants (SI units) ---
LATENT_HEAT_VAPORIZATION = 2.43e6     # J/kg
SPECIFIC_HEAT_WATER = 4186            # J/(kg·°C)
KINEMATIC_VISCOSITY_AIR = 15.89e-6    # m²/s
AIR_THERMAL_CONDUCTIVITY = 0.025      # W/(m·K)
PRANDTL_NUMBER_AIR = 0.7              # dimensionless
DENSITY_WATER = 1000                   # kg/m³
SECONDS_PER_HOUR = 3600
Q_HEATER_DURATION_HR = 72             # Guideline-based duration
U_s = 2.94                            # W/m²·K (side walls)
U_b = 0.5                             # W/m²·K (bottom)
AIR_VELOCITY = 0.2                    # m/s (assumed constant)
HEAT_RECOVERY_EFFICIENCY = 0.6        # Assumed 60% efficiency

# --- Input collection ---
print("Enter pool and hall dimensions:")

pool_length = float(input("Pool length (m): "))
pool_width = float(input("Pool width (m): "))
pool_depth = float(input("Average pool depth (m): "))

hall_length = float(input("Pool hall length (m): "))
hall_width = float(input("Pool hall width (m): "))
hall_height = float(input("Pool hall height (m): "))

tap_water_temp = float(input("Tap water temperature (°C): "))
pool_setpoint_temp = float(input("Desired pool water temperature setpoint (°C): "))
technical_room_temp = float(input("Technical room temperature (°C): "))

heating_duration_hr = float(input("Heating duration for analysis (hours): "))
max_relative_humidity = float(input("Maximum acceptable relative humidity (%): "))
current_relative_humidity = float(input("Current indoor relative humidity (%): "))
activity_factor = float(input("Activity factor for evaporation calculation (e.g. 1.0 for normal use): "))
number_of_persons_per_day = int(input("Number of pool users per day: "))

# --- Derived quantities ---
pool_surface_area = pool_length * pool_width
pool_volume = pool_surface_area * pool_depth
hall_area = hall_length * hall_width
hall_volume = hall_area * hall_height

# --- Q_heater Calculation ---
mass_pool_water = pool_volume * DENSITY_WATER
delta_T_heating = pool_setpoint_temp - tap_water_temp
Q_heater_J = mass_pool_water * SPECIFIC_HEAT_WATER * delta_T_heating
Q_heater_kW = Q_heater_J / (Q_HEATER_DURATION_HR * SECONDS_PER_HOUR * 1000)

# --- Q_cond Calculation ---
A_s = 2 * (pool_length + pool_width) * pool_depth
A_b = pool_length * pool_width
delta_T_cond = pool_setpoint_temp - technical_room_temp
Q_side = U_s * A_s * delta_T_cond
Q_bottom = U_b * A_b * delta_T_cond
Q_cond = Q_side + Q_bottom

# --- Q_evap Calculation ---
air_temp = pool_setpoint_temp + 2
RH_decimal = current_relative_humidity / 100

def saturation_pressure(T):
    return 610.7 * math.exp((17.27 * T) / (T + 237.3))

P_w = saturation_pressure(pool_setpoint_temp)
ln_RH = math.log(RH_decimal)
T_dew = (237.3 * ln_RH) / (17.27 - ln_RH) + air_temp
P_a = saturation_pressure(T_dew)

P_w_inHg = P_w * 0.0002953
P_a_inHg = P_a * 0.0002953

# Convert area from m² to ft²
pool_surface_area_ft2 = pool_surface_area * 10.7639

# Then use this in the evaporation formula
evaporation_lb_per_hr = 0.1 * pool_surface_area_ft2 * (P_w_inHg - P_a_inHg) * activity_factor

evaporation_kg_per_hr = evaporation_lb_per_hr * 0.453592
Q_evap = evaporation_kg_per_hr * LATENT_HEAT_VAPORIZATION / 3600

# --- Q_conv Calculation ---
Re = (AIR_VELOCITY * pool_length) / KINEMATIC_VISCOSITY_AIR
Nu = 0.0296 * (Re ** 0.8) * (PRANDTL_NUMBER_AIR ** (1 / 3))
h_conv = (Nu * AIR_THERMAL_CONDUCTIVITY) / pool_length
Q_conv = 2 * h_conv * pool_surface_area

# --- Q_makeup Calculation ---
hygiene_water_L_per_day = 30 * number_of_persons_per_day
evaporation_L_per_day = evaporation_kg_per_hr * 24
total_makeup_water_L = hygiene_water_L_per_day + evaporation_L_per_day
mass_makeup_water = total_makeup_water_L
delta_T_makeup = pool_setpoint_temp - tap_water_temp
Q_makeup_kJ_per_day = mass_makeup_water * SPECIFIC_HEAT_WATER * delta_T_makeup / 1000
Q_makeup_kW = Q_makeup_kJ_per_day / 3600 / 24

# --- Q_recovery Calculation ---
T_drain = pool_setpoint_temp
T_inlet = tap_water_temp
Q_recovery_kJ_per_day = HEAT_RECOVERY_EFFICIENCY * hygiene_water_L_per_day * SPECIFIC_HEAT_WATER * (T_drain - T_inlet) / 1000
Q_recovery_kW = Q_recovery_kJ_per_day / 3600 / 24

# --- Final ΔT Calculation (Temperature Rise in Pool Water) ---
# Ensure all Q terms are in Watts
Q_heater_W = Q_heater_kW * 1000
Q_recovery_W = Q_recovery_kW * 1000
Q_makeup_W = Q_makeup_kW * 1000

# Duration in seconds
t_seconds = heating_duration_hr * SECONDS_PER_HOUR

# Net energy input to water (Joules)
Q_net = t_seconds * (Q_heater_W + Q_conv + Q_recovery_W - Q_cond - Q_evap - Q_makeup_W)

# ΔT calculation
delta_T_pool_water = Q_net / (mass_pool_water * SPECIFIC_HEAT_WATER)




# --- Output Section ---
print("\n--- Derived Parameters ---")
print(f"Pool surface area: {pool_surface_area:.2f} m²")
print(f"Pool volume: {pool_volume:.2f} m³")
print(f"Pool hall volume: {hall_volume:.2f} m³")

print("\n--- Q_heater Calculation ---")
print(f"Q_heater (avg. power over 72h): {Q_heater_kW:.2f} kW")

print("\n--- Q_cond Calculation ---")
print(f"Q_cond (total): {Q_cond:.2f} W ≈ {Q_cond / 1000:.2f} kW")

print("\n--- Q_evap Calculation ---")
print(f"Evaporation rate: {evaporation_kg_per_hr:.2f} kg/hr")
print(f"Q_evap: {Q_evap:.2f} W ≈ {Q_evap / 1000:.2f} kW")

print("\n--- Q_conv Calculation ---")
print(f"Q_conv: {Q_conv:.2f} W ≈ {Q_conv / 1000:.2f} kW")

print("\n--- Q_makeup Calculation ---")
print(f"Total makeup water: {total_makeup_water_L:.2f} L/day")
print(f"Q_makeup: {Q_makeup_kJ_per_day:.2f} kJ/day ≈ {Q_makeup_kW:.2f} kW")

print("\n--- Q_recovery Calculation ---")
print(f"Q_recovery: {Q_recovery_kJ_per_day:.2f} kJ/day ≈ {Q_recovery_kW:.2f} kW")

# --- Print Result ---
print("\n--- Final Result ---")
print(f"Temperature increase in pool water (ΔT): {delta_T_pool_water:.2f} °C")
