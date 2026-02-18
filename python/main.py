from last_ssm import SPDE_solver
import numpy as np
import matplotlib.pyplot as plt

# Full initial conditions as in last_ssm.py
ICs = np.array([
	0.07076144289241278,  # Ca
	11.938623880388873,   # Cer
	0.8103527888661052,   # h
	0.0,                  # s
	0.9775241714687221,   # w
	0.0,                  # x
	16.31192778501403,    # Na
	114.2749962998676,    # K
	0.0,                  # eta_u
	0.0,                  # D1
	0.0,                  # D2
	0,                    # D3
	0,                    # D4
	1,                    # C1
	0,                    # C2
	0,                    # C3
	0,                    # C4
	0.0,                  # Q1
	0,                    # Q2
	0,                    # Q3
	0,                    # Q4
	1,                    # C0a
	0,                    # C1a
	0,                    # C2a
	0,                    # O
	0,                    # D1a
	0,                    # D2a
	1,                    # C0n
	0,                    # C1n
	0,                    # C2n
	0,                    # On
	0                     # D2n
], dtype=np.float64)

ca_all = []
for i in range(50):
	temp = SPDE_solver(ICs, seed=i)
	# Extract Ca from midpoint (column 4)
	ca_all.append(temp[0])

# Concatenate all Ca data
ca_all = np.concatenate(ca_all)

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(ca_all, bins=200, edgecolor='black', alpha=0.7)
plt.xlabel('Ca Concentration (μM)')
plt.ylabel('Frequency')
plt.title('Histogram of Ca Concentration (50 runs)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('Ca_histogram_50runs.svg', format='svg')
plt.show()