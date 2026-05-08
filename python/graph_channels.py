import os
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

# ============================================================
# Fig 9 reproduction, in vivo condition when apply ATP, Glut = 0;
# ============================================================
# np.random.seed(2)
# temp = SPDE_solver(ICs, dx = dx, dt = dt, rdmATP=True, tmax = 300)
# Jsoc = temp[9]
# Jip3 = temp[13]
# Jryr = temp[14]
# Jl = temp[15]
# Jt = temp[16]
# Jp2x7 = temp[17]
# Jampa = temp[18]
# Jnmda = temp[19]
# ip3 = temp[20]
# 
# time = np.arange(len(Jsoc))  # create time array using your dt timestep
# 
# fig, axs = plt.subplots(2, 3)
# axs[0,0].plot(time, Jsoc, linewidth=2)
# axs[0,0].set_ylabel(r'$\mathrm{J_{SOCE}}$ ($\mu M \cdot s^{-1}$)', fontsize=11)
# axs[0,0].text(-0.15, 1.1, 'A', transform=axs[0,0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
# axs[0,1].plot(time, Jip3+Jryr, linewidth=2)
# axs[0,1].set_ylabel(r'$\mathrm{J_{CICR}}$ ($\mu M \cdot s^{-1}$)', fontsize=11)
# axs[0,1].text(-0.15, 1.1, 'B', transform=axs[0,1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
# axs[0,2].plot(time[2000:29000], Jl[2000:29000]+Jt[2000:29000], linewidth=2)
# axs[0,2].set_ylabel(r'$\mathrm{J_{VGCC}}$ ($\mu M \cdot s^{-1}$)', fontsize=11)
# axs[0,2].text(-0.15, 1.1, 'C', transform=axs[0,2].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
# axs[1,0].plot(time, Jp2x7, linewidth=2)
# axs[1,0].set_ylabel(r'$\mathrm{J_{P2X7}}$ ($\mu M \cdot s^{-1}$)', fontsize=11)
# axs[1,0].text(-0.15, 1.1, 'D', transform=axs[1,0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
# axs[1,1].plot(time, Jampa+Jnmda, linewidth=2)
# axs[1,1].set_ylabel(r'$\mathrm{J_{Syn}}$ ($\mu M \cdot s^{-1}$)', fontsize=11)
# axs[1,1].text(-0.15, 1.1, 'E', transform=axs[1,1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
# axs[1,2].plot(time, ip3, linewidth=2)
# axs[1,2].set_ylabel(r'$\mathrm{[IP_3](mM)}$', fontsize=11)
# axs[1,2].text(-0.15, 1.1, 'F', transform=axs[1,2].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
# 
# # Remove boxes and configure axes
# for i, ax in enumerate(axs.flat):
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     # Format y-axis with scientific notation
#     ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#     ax.set_xticks(np.arange(0, 40000, 10000))
#     # Set x-axis label only on bottom row
#     if i >= 3:
#         ax.set_xlabel('Time (s)', fontsize=12)
#         ax.set_xticklabels(['0', '100', '200', '300'])
#     else:
#         ax.set_xticklabels([])
# plt.subplots_adjust(wspace=0.3)
# os.makedirs('figures', exist_ok=True)
# plt.savefig('figures/vivo_atp.pdf')
# plt.show()

# ============================================================
# Fig S3 reproduction,
# ============================================================
np.random.seed(2)
temp = SPDE_solver(ICs, dt = dt, dx = dx, ATP = 0, G=0)  # no ATP, no Glut
Jsoc = temp[9]
Jip3 = temp[13]
Jryr = temp[14]
Jl = temp[15]
Jt = temp[16]
Jp2x7 = temp[17]
Jampa = temp[18]
Jnmda = temp[19]
ip3 = temp[20]
time = np.arange(len(Jsoc))  # create time array using your dt timestep

fig, axs = plt.subplots(2, 3)
axs[0,0].plot(time, Jsoc, linewidth=2)
axs[0,0].set_ylabel(r'$\mathrm{J_{SOCE}}$ ($\mu M \cdot s^{-1}$)', fontsize=11)
axs[0,0].text(-0.15, 1.1, 'A', transform=axs[0,0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axs[0,1].plot(time, Jip3+Jryr, linewidth=2)
axs[0,1].set_ylabel(r'$\mathrm{J_{CICR}}$ ($\mu M \cdot s^{-1}$)', fontsize=11)
axs[0,1].text(-0.15, 1.1, 'B', transform=axs[0,1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axs[0,2].plot(time[2000:29000], Jl[2000:29000]+Jt[2000:29000], linewidth=2)
axs[0,2].set_ylabel(r'$\mathrm{J_{VGCC}}$ ($\mu M \cdot s^{-1}$)', fontsize=11)
axs[0,2].text(-0.15, 1.1, 'C', transform=axs[0,2].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axs[1,0].plot(time, Jp2x7, linewidth=2)
axs[1,0].set_ylabel(r'$\mathrm{J_{P2X7}}$ ($\mu M \cdot s^{-1}$)', fontsize=11)
axs[1,0].text(-0.15, 1.1, 'D', transform=axs[1,0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axs[1,1].plot(time, Jampa+Jnmda, linewidth=2)
axs[1,1].set_ylabel(r'$\mathrm{J_{Syn}}$ ($\mu M \cdot s^{-1}$)', fontsize=11)
axs[1,1].text(-0.15, 1.1, 'E', transform=axs[1,1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axs[1,2].plot(time, ip3, linewidth=2)
axs[1,2].set_ylabel(r'$\mathrm{[IP_3](\mu M)}$', fontsize=11)
axs[1,2].text(-0.15, 1.1, 'F', transform=axs[1,2].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

# Remove boxes and configure axes
for i, ax in enumerate(axs.flat):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Format y-axis with scientific notation
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.set_xticks(np.arange(0, 40000, 10000))
    # Set x-axis label only on bottom row
    if i >= 3:
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_xticklabels(['0', '100', '200', '300'])
    else:
        ax.set_xticklabels([])
plt.subplots_adjust(wspace=0.5)
plt.savefig('figures/vivo_no_atp_no_glut.pdf')
plt.show()