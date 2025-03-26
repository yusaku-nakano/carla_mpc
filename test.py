from ctypes import CDLL
dll_path = r"C:\Users\yusak\carla_sims\MPC\c_generated_code\libacados_ocp_solver_KinematicBicycleFrenet.dll"
try:
    dll = CDLL(dll_path)
    print("DLL loaded successfully!")
except OSError as e:
    print(f"Failed to load DLL: {e}")