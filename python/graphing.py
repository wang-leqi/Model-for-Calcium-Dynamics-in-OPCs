"""
Graphs for data visualization. Recorded different atp/glut levels and their effects on calcium concentration.
Data imported from xpp time series, imported as .dat file. 
Concentration: 10^(-10), 10^(-4.5), 10^(-4)mM for both atp and glut.
--> Modifications: 1. 10^(-5)mM, delete last row
--> Modifications: 2. 10^(-4.8)mM
"""
import numpy as np
import matplotlib.pyplot as plt

# read data
atp_0 = np.loadtxt('dat/atp0.dat') # 10^(-10) mM
atp_48 = np.loadtxt('dat/atp48.dat') # 10^(-4.8) mM
# atp_4 = np.loadtxt('dat/atp4.dat') # 10^(-4) mM
# print(atp_0.shape) # returns(80001, 23) (rows, cols)

glut_0 = np.loadtxt('dat/glut0.dat') # 10^(-10) mM
glut_5 = np.loadtxt('dat/glut5.dat') # 10^(-5) mM
# glut_4 = np.loadtxt('dat/glut4.dat') # 10^(-4) mM

# plot
fig, axs = plt.subplots(2, 2)

axs[0,0].plot(atp_0[:,0], atp_0[:,1])
axs[0,0].set_ylim(0,2.3)
axs[0,0].set_yticks(np.arange(0,3,1))
axs[0,0].set_ylabel(r'$\mathrm{[Ca^{2+}](\mu M)}$', fontsize=12)
axs[0,0].text(-0.2, 1, 'A', transform=axs[0,0].transAxes, fontsize=14, fontweight='bold', va='top')

axs[0,1].plot(glut_0[:,0], glut_0[:,1])
axs[0,1].set_ylim(0,2.3)
axs[0,1].set_yticks(np.arange(0,3,1), labels=[])
axs[0,1].text(-0.2, 1, 'B', transform=axs[0,1].transAxes, fontsize=14, fontweight='bold', va='top')

axs[1,0].plot(atp_48[:,0], atp_48[:,1])
axs[1,0].set_ylim(0,2.3)
axs[1,0].set_yticks(np.arange(0,3,1))
axs[1,0].set_xlabel('Time (s)', fontsize=12)
axs[1,0].set_ylabel(r'$\mathrm{[Ca^{2+}](\mu M)}$', fontsize=12)

axs[1,1].plot(glut_5[8000:,0], glut_5[8000:,1])
axs[1,1].set_ylim(0,2.3)
axs[1,1].set_yticks(np.arange(0,3,1), labels=[])
axs[1,1].set_xlabel('Time (s)', fontsize=12)

# axs[2,0].plot(atp_4[:,0], atp_4[:,1])
# axs[2,0].set_ylim(0,60)
# axs[2,0].set_yticks(np.arange(0,60,10))
# axs[2,0].set_xlabel('Time (s)')
# axs[2,0].set_ylabel('Ca(mM)')

# axs[2,1].set_title('Glut = 10^(-4) mM', fontsize = 12, fontweight='bold')
# axs[2,1].plot(glut_4[:,0], glut_4[:,1])
# axs[2,1].set_ylim(0,60)
# axs[2,1].set_yticks(np.arange(0,60,10))
# axs[2,1].set_xlabel('Time (s)')
# axs[2,1].set_ylabel('Ca(mM)')

# Remove box, keep axes for all subplots
for ax in axs.flat:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.subplots_adjust(wspace=0.2)
# plt.tight_layout()
plt.savefig('figures/fig3.pdf')
plt.show()