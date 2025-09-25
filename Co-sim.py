import ctypes, time

# --- Load DLL (same folder) ---
mydll = ctypes.CDLL(r'.\idalibN.dll')

# --- Prototypes ---
mydll.CreateStorage.restype  = ctypes.c_longlong
mydll.CreateStorage.argtypes = [ctypes.c_char_p, ctypes.c_longlong, ctypes.c_longlong,
                                ctypes.c_longlong, ctypes.c_void_p, ctypes.c_longlong]
mydll.RetrieveStoredData.restype  = ctypes.c_longlong
mydll.RetrieveStoredData.argtypes = [ctypes.c_longlong, ctypes.c_double, ctypes.c_void_p]
mydll.StoreData.restype  = ctypes.c_longlong
mydll.StoreData.argtypes = [ctypes.c_longlong, ctypes.c_double, ctypes.c_void_p, ctypes.c_longlong]
mydll.CloseStorage.restype  = ctypes.c_longlong
mydll.CloseStorage.argtypes = [ctypes.c_longlong]

# --------- CHANNEL NAMES (must match 'channel' in the blocks) ----------
MEAS_NAME = b'ICE_AHU'     # Export: measured RH (%)
CTRL_NAME = b'ICE_RHCtrl'  # Import: RH setpoint (%)

# Open storages
nVar = ctypes.c_longlong(1)
nSync = ctypes.c_longlong(1)
VarInterp = ctypes.c_double(0.0)
opc = ctypes.c_longlong(0)

# Read (no extrapolation)
h_meas = mydll.CreateStorage(ctypes.c_char_p(MEAS_NAME),
                             nVar, nSync, ctypes.c_longlong(0),
                             ctypes.byref(VarInterp), opc)
# Write (75 s extrap limit like demo)
h_ctrl = mydll.CreateStorage(ctypes.c_char_p(CTRL_NAME),
                             nVar, nSync, ctypes.c_longlong(75),
                             ctypes.byref(VarInterp), opc)

if h_meas <= 0 or h_ctrl <= 0:
    raise RuntimeError(f"CreateStorage failed: h_meas={h_meas}, h_ctrl={h_ctrl}.")

# --------- RH setpoint logic (SECONDS) ----------
dt_s   = 60.0
end_s  = 24*3600.0
steps  = int(end_s / dt_s)

RH_LOW, RH_HIGH = 48.0, 52.0
SP_LOW, SP_HIGH = 40.0, 45.0
SP_MIN, SP_MAX  = 30.0, 70.0

sp = 52.0
meas = ctypes.c_double(0.0)
out  = ctypes.c_double(0.0)
extrap = ctypes.c_longlong(0)

# Prime t = 0 s
out.value = max(SP_MIN, min(SP_MAX, sp))
mydll.StoreData(h_ctrl, ctypes.c_double(0.0), ctypes.byref(out), extrap)
print("Primed setpoint at t = 0 s")

t_s = 0.0
for k in range(steps):
    t_s += dt_s
    tt = ctypes.c_double(t_s)

    # Write setpoint first so ICE always has data
    out.value = max(SP_MIN, min(SP_MAX, sp))
    mydll.StoreData(h_ctrl, tt, ctypes.byref(out), extrap)

    # Try to read RH for info/hysteresis; if not available yet, keep sp
    rc = mydll.RetrieveStoredData(h_meas, tt, ctypes.byref(meas))
    rh = meas.value
    if 0.0 <= rh <= 100.0:
        if rh < RH_LOW:
            sp = SP_HIGH
        elif rh > RH_HIGH:
            sp = SP_LOW

    if ((k+1) % int(600.0/dt_s)) == 0:  # log every 10 min of sim time
        print(f"T: {t_s:6.0f} s  ({t_s/60:5.1f} min)  RH_meas: {rh:6.2f}%  ->  RH_setpoint: {sp:5.1f}%")

print(f"Finished writing setpoints to t = {t_s:.0f} s ({t_s/60:.1f} min)")
print("Keep this window open while ICE runs. Press Enter here AFTER ICE finishes.")

# --- Keep channels open until ICE is done ---
try:
    input()
finally:
    mydll.CloseStorage(h_meas)
    mydll.CloseStorage(h_ctrl)
