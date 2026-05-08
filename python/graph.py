import time
from scipy.io import savemat, loadmat
import last_ssm as ssm
import numpy as np
import matplotlib.pyplot as plt

SPDE_solver = ssm.SPDE_solver
tmax=300

Ca=0.07076144289241278
Cer=11.938623880388873
h=0.8103527888661052
s=0.0
w=0.9775241714687221
x=0.0#0.14634
Na=16.31192778501403
K=114.2749962998676
volt=-71.37667670139595
eta_u=0.0

D1=0.
D2=0.
D3=0
D4=0
C1=1
C2=0
C3=0
C4=0
Q1=0.
Q2=0#0.000149#0.
Q3=0#9.06487*10**(-6)#0.
Q4=0#1.1157*10**(-5)#0.
C0a=1
C1a=0
C2a=0
O=0
D1a=0
D2a=0

C0n=1
C1n=0
C2n=0
On=0
D2n=0


F = 96485.3321
Vosteo=6.5
aa=time.time()
ICs = np.array([Ca, Cer, h, s, w, x, Na , K, eta_u, D1, D2, D3, D4, C1, C2, C3, C4, Q1, Q2, Q3, Q4, C0a, C1a, C2a, O, D1a, D2a, C0n, C1n, C2n, On, D2n,volt])

dt = 0.01 
dx = 1
# temp = SPDE_solver(ICs, dt=dt, dx=dx, ATP = 0.1*10**(-3))
#####################
# ============================================================
# Fig 5 C reproduction of different ATP concentrations;
# ============================================================
def normalize_01(x):
    x = np.asarray(x)
    return (x - x.min()) / (x.max() - x.min() + 1e-12)
def normalize_dff(x, baseline_frames=100):
    """Normalize using ΔF/F0, where F0 is the baseline (mean of initial frames)"""
    x = np.asarray(x)
    F0 = np.mean(x[:baseline_frames])  # baseline is mean of first N frames
    return (x - F0) / (F0)
def run_ensemble_stepATP(ICs, dx, dt, n_iter, seed0=9, tmax=tmax,
                         ATP_M=0.043*10**(-3), ATP_start=60, ATP_time=180, idx=4):
    traces = []
    # np.random.seed(seed0)
    _ = SPDE_solver(ICs, dx = dx, dt = dt, tmax=1, ATP=ATP_M, ATP_start=ATP_start, ATP_time=ATP_time)

    for i in range(n_iter):
        seed = seed0 + i
        np.random.seed(seed)

        temp = SPDE_solver(ICs, dx = dx, dt = dt, tmax=tmax,
                           ATP=ATP_M, ATP_start=ATP_start, ATP_time=ATP_time)

        ca = temp[0].T[idx] # midpoint Ca
        ca = normalize_01(ca)
        # ca = normalize_dff(ca) # ΔF/F0 normalization
        traces.append(ca)

    traces = np.asarray(traces)
    t = np.linspace(0, tmax, traces.shape[1])
    return t, traces

def plot_many(t, traces, dx, dt, ATP_start=60, ATP_end=240, max_to_plot=None):
    plt.figure(figsize=(10, 5))

    if max_to_plot is None or max_to_plot >= traces.shape[0]:
        idxs = range(traces.shape[0])
    else:
        idxs = range(max_to_plot)

    for k in idxs:
        plt.plot(t, traces[k], lw=0.8, alpha=0.15)  # low alpha is key

    plt.axvline(ATP_start, ls="--")
    plt.axvline(ATP_end, ls="--")

    plt.xlabel("Time (s)")
    plt.ylabel("ΔF/F0")
    plt.tight_layout()
    plt.tight_layout()
    plt.show()

# ---- run + plot (this should look much closer to Fig 5A style) ----
# ATP: 0.1, 0.055, 0.043 mM
# t, traces = run_ensemble_stepATP(ICs, dx = dx, dt = dt, n_iter=5, seed0=9,
#                                  ATP_M=0.1*10**(-3), ATP_start=60, ATP_time=180, idx=4)
# plot_many(t, traces, dx, dt, ATP_start=60, ATP_end=180, max_to_plot=50)

# t, traces = run_ensemble_stepATP(ICs, dx = dx, dt = dt, n_iter=5, seed0=9,
#                                  ATP_M=0.043*10**(-3), ATP_start=60, ATP_time=180, idx=4)
# plot_many(t, traces, dx, dt, ATP_start=60, ATP_end=180, max_to_plot=50)

# ============================================================
# Fig 6 C reproduction of different glut concentrations;
# ============================================================
#
# def normalize_01(x):
#     x = np.asarray(x)
#     return (x - x.min()) / (x.max() - x.min() + 1e-12)
#
#
# def normalize_dff(x, baseline_frames=60):
#     """Normalize using ΔF/F0, where F0 is the baseline (mean of initial frames)"""
#     x = np.asarray(x)
#     F0 = np.mean(x[:baseline_frames])  # baseline is mean of first N frames
#     return (x - F0) / (F0)
#
# def run_ensemble_stepGlut(ICs, dx, dt, n_iter=3, seed0=9, tmax=tmax,
#                          Glut_M=0.05*10**(-3), Glut_start=60, Glut_time=120, idx=4):
#     traces = []
#     # np.random.seed(seed0)
#     _ = SPDE_solver(ICs, tmax=1.0, dx = dx, dt=dt, G=Glut_M, Glut_start=Glut_start, Glut_time=Glut_time)
#
#     for i in range(n_iter):
#         seed = seed0 + i
#         np.random.seed(seed)
#
#         temp = SPDE_solver(ICs, tmax=tmax, dx = dx, dt=dt,
#                            G=Glut_M, Glut_start=Glut_start, Glut_time=Glut_time)
#
#         ca = temp[0].T[idx] # midpoint Ca
#         ca = normalize_01(ca)
#         # ca = normalize_dff(ca) # ΔF/F0 normalization
#         traces.append(ca)
#
#     traces = np.asarray(traces)
#     t = np.linspace(0, tmax, traces.shape[1])
#     return t, traces
#
# def plot_many(t, traces, Glut_start=60, Glut_end=240, max_to_plot=None):
#     plt.figure(figsize=(10, 5))
#
#     if max_to_plot is None or max_to_plot >= traces.shape[0]:
#         idxs = range(traces.shape[0])
#     else:
#         idxs = range(max_to_plot)
#
#     for k in idxs:
#         plt.plot(t, traces[k], lw=0.8, alpha=0.15)  # low alpha is key
#
#     plt.axvline(Glut_start, ls="--")
#     plt.axvline(Glut_end, ls="--")
#
#     plt.xlabel("Time (s)")
#     plt.ylabel("ΔF/F0")
#     plt.tight_layout()
#     plt.tight_layout()
#     plt.show()
#
# # ---- run + plot (this should look much closer to Fig 5A style) ----
# t, traces = run_ensemble_stepGlut(ICs, dx, dt, n_iter=3, seed0=9, Glut_M=0.05*10**(-3), idx=4)
# plot_many(t, traces, Glut_start=60, Glut_end=180, max_to_plot=50)



####### test ########
# def run_ensemble_stepGlut_debug(ICs, dx, dt, seed0=9, tmax=tmax,
#                                 Glut_M=0.05 * 10 ** (-3), Glut_start=60, Glut_time=120, idx=4):
#     """Same as run_ensemble_stepGlut but returns IP3 and time info for debugging"""
#
#     np.random.seed(seed0)
#     temp = SPDE_solver(ICs, tmax=tmax, dx=dx, dt=dt,
#                        G=Glut_M, Glut_start=Glut_start, Glut_time=Glut_time)
#
#     ca = temp[0].T[idx]  # calcium at location idx
#     ip3 = temp[20]  # IP3 signal (last element in return)
#
#     # Create CORRECT time axis: actual number of timesteps
#     n_steps = len(ca)
#     t = np.arange(n_steps) * dt  # This is the CORRECT time axis!
#
#     return t, ca, ip3
#
#
# # Test it
# t, ca, ip3 = run_ensemble_stepGlut_debug(ICs, dx, dt, Glut_M=0.05 * 10 ** (-3), Glut_start=60, Glut_time=120)
#
# # Plot both calcium and IP3 to see stimulus response
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
#
# ax1.plot(t, ca, lw=1)
# ax1.axvline(60, ls="--", color='red', label='Glut start')
# ax1.axvline(180, ls="--", color='red', label='Glut end')
# ax1.set_ylabel('Ca (μM)')
# ax1.set_title('Calcium Response')
# ax1.legend()
# ax1.grid(True, alpha=0.3)
#
# ax2.plot(t, ip3, lw=1, color='orange')
# ax2.axvline(60, ls="--", color='red')
# ax2.axvline(180, ls="--", color='red')
# ax2.set_ylabel('IP3 (mM)')
# ax2.set_xlabel('Time (s)')
# ax2.set_title('IP3 Signal (verifies glutamate is being applied)')
# ax2.grid(True, alpha=0.3)
#
# plt.tight_layout()
# plt.show()
#
#
# def run_ensemble_stepGlut_full_debug(ICs, dx, dt, seed0=9, tmax=tmax,
#                                      Glut_M=0.05 * 10 ** (-3),
#                                      Glut_start=60, Glut_time=120, idx=4):
#     """Return all variables needed to diagnose the delay"""
#
#     np.random.seed(seed0)
#     temp = SPDE_solver(ICs, tmax=tmax, dx=dx, dt=dt,
#                        G=Glut_M, Glut_start=Glut_start, Glut_time=Glut_time)
#
#     n_steps = len(temp[0])
#     t = np.arange(n_steps) * dt
#
#     ca = temp[0].T[idx]  # Calcium
#     cer = temp[3].T[idx]  # ER calcium
#     h_var = temp[5].T[idx]  # IP3 receptor inactivation
#     ip3 = temp[20]  # IP3
#     jip3 = temp[13]  # IP3-mediated flux (Jip3)
#     jryr = temp[14]  # Ryanodine flux (Jryr)
#     jserca = temp[-1]  # SERCA (placeholder - check actual index)
#
#     return t, ca, cer, h_var, ip3, jip3, jryr
#
#
# # Run debug
# t, ca, cer, h_var, ip3, jip3, jryr = run_ensemble_stepGlut_full_debug(
#     ICs, dx, dt, Glut_M=0.05 * 10 ** (-3), Glut_start=60, Glut_time=120)
#
# # Plot everything to identify the bottleneck
# fig, axs = plt.subplots(5, 1, figsize=(12, 10))
#
# # IP3
# axs[0].plot(t, ip3, lw=1.5, color='purple')
# axs[0].axvline(60, ls="--", color='red', alpha=0.5)
# axs[0].axvline(180, ls="--", color='red', alpha=0.5)
# axs[0].set_ylabel('IP3 (mM)', fontsize=10)
# axs[0].set_title('IP3 from Glutamate')
# axs[0].grid(True, alpha=0.3)
#
# # h (inactivation variable) - should DECREASE to allow Ca release
# axs[1].plot(t, h_var, lw=1.5, color='brown')
# axs[1].axvline(60, ls="--", color='red', alpha=0.5)
# axs[1].axvline(180, ls="--", color='red', alpha=0.5)
# axs[1].set_ylabel('h (IP3R inactivation)', fontsize=10)
# axs[1].set_title('h variable: should DECREASE when Ca rises')
# axs[1].grid(True, alpha=0.3)
#
# # IP3-mediated flux
# axs[2].plot(t, jip3, lw=1.5, color='blue')
# axs[2].axvline(60, ls="--", color='red', alpha=0.5)
# axs[2].axvline(180, ls="--", color='red', alpha=0.5)
# axs[2].axhline(0, ls="-", color='k', alpha=0.2)
# axs[2].set_ylabel('Jip3 (μM/s)', fontsize=10)
# axs[2].set_title('IP3-mediated Ca release flux')
# axs[2].grid(True, alpha=0.3)
#
# # ER calcium
# axs[3].plot(t, cer, lw=1.5, color='green')
# axs[3].axvline(60, ls="--", color='red', alpha=0.5)
# axs[3].axvline(180, ls="--", color='red', alpha=0.5)
# axs[3].set_ylabel('Cer (μM)', fontsize=10)
# axs[3].set_title('ER Calcium')
# axs[3].grid(True, alpha=0.3)
#
# # Cytosolic calcium
# axs[4].plot(t, ca, lw=1.5, color='orange')
# axs[4].axvline(60, ls="--", color='red', alpha=0.5, label='Glut applied')
# axs[4].axvline(180, ls="--", color='red', alpha=0.5, label='Glut removed')
# axs[4].set_ylabel('Ca (μM)', fontsize=10)
# axs[4].set_xlabel('Time (s)', fontsize=10)
# axs[4].set_title('Cytosolic Calcium')
# axs[4].legend()
# axs[4].grid(True, alpha=0.3)
#
# plt.tight_layout()
# plt.show()
#
# print(f"IP3 rise time: {np.where(ip3 > 0.01)[0][0] * dt:.1f}s")
# print(f"h decrease: from {h_var[6000]:.4f} to {h_var[15000]:.4f}")
# print(f"Jip3 onset: {np.argmax(np.abs(jip3[5000:15000])) * dt + 50:.1f}s")
# print(f"Ca rise time: {np.where(ca > 0.2)[0][0] * dt if np.any(ca > 0.2) else 'No rise'}s")
#
