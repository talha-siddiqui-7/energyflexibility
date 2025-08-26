# Latent storage with realistic caps: ventilation + supply humidity options
import math

def psat_pa(Tc):  # Pa
    return 610.94 * math.exp(17.625 * Tc / (Tc + 243.04))

def W_from_RH_T(RH_frac, Tc, p_atm):
    Pv = RH_frac * psat_pa(Tc)
    return 0.622 * Pv / (p_atm - Pv)

def Pv_from_W(W, p_atm):
    return p_atm * W / (0.622 + W)

def RH_from_W_T(W, Tc, p_atm):
    Pv = Pv_from_W(W, p_atm)
    return max(0.0, min(1.0, Pv / psat_pa(Tc)))

def h_fg_J_per_kg(Tc):  # J/kg
    return (2500.8 - 2.36 * Tc) * 1000.0

def m_evap(A, Fa, C_evap, T_water, W_zone, p_atm):  # kg/s
    Pw = psat_pa(T_water)
    Pa = Pv_from_W(W_zone, p_atm)
    return max(0.0, C_evap * A * max(0.0, Pw - Pa) * Fa)

def dPa_dW(W, p_atm):
    return p_atm * 0.622 / (0.622 + W)**2

def ask_float(prompt, default=None):
    s = input(f"{prompt}" + (f" [default {default}]: " if default is not None else ": "))
    if s.strip() == "":
        if default is None:
            raise ValueError("This value is required.")
        return float(default)
    return float(s)

print("\n=== Latent Energy Storage (RH relaxation) — Moisture-Balance Calculator ===\n")

# Core geometry/conditions
A = ask_float("Pool surface area A (m^2)")
V_hall = ask_float("Hall air volume V_hall (m^3)")
T_air = ask_float("Indoor air dry-bulb temperature (°C)")
T_water = ask_float("Pool water temperature (°C)")
RH0_pct = ask_float("Initial RH (%)")
RHt_pct = ask_float("Relaxed target RH (%)")
p_atm = ask_float("Atmospheric pressure (Pa)", 101325)
Fa = ask_float("Activity factor Fa (0.3–1.0)", 1.0)
dt_h = ask_float("DR window Δt (hours)")

# Psychro start
RH0 = RH0_pct/100.0
RHt = RHt_pct/100.0
W0 = W_from_RH_T(RH0, T_air, p_atm)
Wtarg = W_from_RH_T(RHt, T_air, p_atm)
dW_set = Wtarg - W0

# Moist air density (ideal gas mixture)
R_d, R_v = 287.058, 461.495
Tk = T_air + 273.15
Pv0 = Pv_from_W(W0, p_atm)
rho_air = (p_atm - Pv0)/(R_d*Tk) + Pv0/(R_v*Tk)
m_air = rho_air * V_hall

# Evaporation coefficient (calibrate if you can)
C_evap_default = 4.0e-8  # kg/(s·m^2·Pa) ~ mapped from the classic 0.1*A*(ΔP_inHg) lb/h
C_evap = ask_float("Evaporation coefficient C_evap (kg/(s·m^2·Pa))", C_evap_default)

# -------- Ventilation entry (don’t leave at 0 if you want a cap) --------
print("\n--- Ventilation / Dehumidification ---")
mode_v = input("Provide (1) mass flow kg/s, (2) ACH, or (3) supply airflow m3/h? [1/2/3, default 2]: ").strip() or "2"
m_dot_vent = 0.0
if mode_v == "1":
    m_dot_vent = ask_float("m_dot_vent (kg/s)", 0.0)
elif mode_v == "2":
    ach = ask_float("Air changes per hour (ACH)", 2.0)
    m_dot_vent = rho_air * (ach * V_hall / 3600.0)  # kg/s
elif mode_v == "3":
    Vdot_m3h = ask_float("Supply airflow (m^3/h)")
    m_dot_vent = rho_air * (Vdot_m3h / 3600.0)
else:
    print("Unknown option, assuming 0 kg/s.")

# -------- Supply humidity ratio W_sup options --------
print("\n--- Supply humidity under RELAXED mode (choose one) ---")
print("1) Coil dewpoint (°C)  → W_sup = W_sat(T_dew)")
print("2) Recirculation fraction + outdoor (coil OFF): W_sup = r*W0 + (1-r)*W_out")
print("3) Direct supply RH (%) and T (°C)")
print("Enter to skip → assumes W_sup = W0 (no removal).")

W_sup = W0  # conservative default
choice = input("Choice [1/2/3 or Enter]: ").strip()

if choice == "1":
    T_dew = ask_float("Coil leaving dewpoint (°C)")
    W_sup = W_from_RH_T(1.0, T_dew, p_atm)  # saturated at coil dewpoint
elif choice == "2":
    r = ask_float("Recirculation fraction r (0–1)", 0.8)
    T_out = ask_float("Outdoor air temperature (°C)")
    RH_out = ask_float("Outdoor RH (%)")
    W_out = W_from_RH_T(RH_out/100.0, T_out, p_atm)
    W_sup = r * W0 + (1.0 - r) * W_out
elif choice == "3":
    RHs = ask_float("Supply RH (%)")
    Ts = ask_float("Supply temperature (°C)")
    W_sup = W_from_RH_T(RHs/100.0, Ts, p_atm)

# -------- Linearized first-order solution --------
m_evap0 = m_evap(A, Fa, C_evap, T_water, W0, p_atm)
k_e = C_evap * A * Fa * dPa_dW(W0, p_atm)
den = max(1e-9, m_dot_vent + k_e)
C0 = m_evap0 + k_e*W0 + m_dot_vent*W_sup
W_inf = C0 / den
tau = m_air / den                  # s
dt_s = dt_h * 3600.0
W_t = W_inf + (W0 - W_inf) * math.exp(-dt_s / tau)

# Hit the target or not?
if dW_set >= 0:
    W_end = min(W_t, Wtarg)
    dW_ach = max(0.0, W_end - W0)
else:
    W_end = max(W_t, Wtarg)
    dW_ach = min(0.0, W_end - W0)

hfg = h_fg_J_per_kg(T_air)
Q_latent_J = hfg * m_air * dW_ach
Q_latent_kWh = Q_latent_J / 3.6e6
Q_latent_MJ = Q_latent_J / 1e6
RH_end = RH_from_W_T(W_end, T_air, p_atm) * 100.0

# Time to reach the target (if reachable)
t_hit = None
if (W_inf - W0) * (Wtarg - W0) > 0:  # moving toward target
    num = W_inf - W0
    den_hit = max(1e-12, W_inf - Wtarg)
    ratio = num / den_hit
    if ratio > 1.0:
        t_hit = -tau * math.log(1.0 / ratio)

print("\n=== RESULTS ===")
print(f"Air mass in hall (kg):        {m_air:,.1f}")
print(f"Initial humidity ratio W0:     {W0:.6f} kg/kg")
print(f"Target  humidity ratio W*:     {Wtarg:.6f} kg/kg")
print(f"Equilibrium W_inf (relaxed):   {W_inf:.6f} kg/kg")
print(f"Time constant τ (minutes):     {tau/60.0:,.1f}")
print(f"Evaporation at start (kg/h):   {m_evap0*3600.0:,.2f}")
if t_hit is not None:
    print(f"Time to reach target (minutes): {t_hit/60.0:,.1f}")
print(f"Achieved ΔW in window:         {dW_ach:.6f} kg/kg")
print(f"End-of-window RH (%):          {RH_end:.2f}")
print(f"\nLatent energy stored (MJ):     {Q_latent_MJ:,.2f}")
print(f"Latent energy stored (kWh):    {Q_latent_kWh:,.2f}\n")

if W_inf < Wtarg and dW_set > 0:
    print("Cap in effect: equilibrium under relaxed mode is below target RH.")
elif t_hit is not None and t_hit > dt_s and dW_set > 0:
    print("Cap in effect: window too short to reach the relaxed RH.")
elif abs(dW_ach - dW_set) < 1e-6:
    print("Note: No cap triggered — with your inputs the space reaches the target within the window.")
