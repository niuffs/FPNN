import numpy as np
import matplotlib.pyplot as plt

# Model loss
FPNN = np.load('FPNN_MLP_4D_Ring_results.npy', allow_pickle=True).item()
steps = FPNN['Steps_loss']
loss_score_fpnn = FPNN['Score_loss']
loss_pde_fpnn = FPNN['Plain_pde_loss']
TFFN = np.load('TFFN_4D_Ring_results.npy', allow_pickle=True).item()
s = np.array(TFFN['Steps_loss'])
steps_ = s[s <= 10000]
loss_pde_tffn = np.array(TFFN['Plain_pde_loss'])[s <= 10000]

# Plot
plt.figure(figsize=(6, 5))

plt.plot(steps, loss_score_fpnn, color='#1f77b4', linestyle='-.', label='FPNN $\mathcal{J}_{score}$')
plt.plot(steps, loss_pde_fpnn, color='#ff7f0e', linestyle='-', label='FPNN $\mathcal{J}_{plain}$')
plt.plot(steps_, loss_pde_tffn, color='#2ca02c', linestyle='-', label='TFFN $\mathcal{J}_{plain}$')

plt.tight_layout(pad=3.5)
plt.xlabel('Steps', fontsize=18, fontname='Arial')
plt.ylabel('PDE Loss', fontsize=18, fontname='Arial')
plt.yscale('log')
plt.legend(loc='upper right')
plt.savefig("PDE loss.png", dpi=300)
plt.close()