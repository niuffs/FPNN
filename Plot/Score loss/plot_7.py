import numpy as np
import matplotlib.pyplot as plt

# TNN
path_TNN = ['FPNN_TNN_4D_Ring_results.npy', 'FPNN_TNN_6D_Multi-modal_results.npy', 'FPNN_TNN_Gaussian_mixture_results.npy', 'FPNN_TNN_Gaussian_results.npy']

# MLP
path_MLP = ['FPNN_MLP_4D_Ring_results.npy', 'FPNN_MLP_6D_Multi-modal_results.npy', 'FPNN_MLP_Gaussian_mixture_results.npy']

# Define labels and colors
labels = ['4D Ring', '6D Multi-modal', '10D Gaussian mixture', '20D Gaussian']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Plot
plt.figure(figsize=(6, 4))

for i in range(4):
    TNN = np.load(path_TNN[i], allow_pickle=True).item()
    s = np.array(TNN['Steps_loss'])
    steps = s[s <= 5000]
    loss_TNN = np.array(TNN['Score_loss'])[s <= 5000]
    plt.plot(steps, loss_TNN, color=colors[i], linestyle='-', label=f'{labels[i]} TNN')

    if i<3:
        MLP = np.load(path_MLP[i], allow_pickle=True).item()
        loss_MLP = np.array(MLP['Score_loss'])[s<=5000]
        plt.plot(steps, loss_MLP, color=colors[i], linestyle='-.', label=f'{labels[i]} MLP')

plt.tight_layout(pad=3)
plt.xlabel('Steps', fontsize=18, fontname='Arial')
plt.ylabel('Score Loss', fontsize=18, fontname='Arial')
plt.yscale('log')
plt.legend()
plt.savefig("Score PDE loss.png", dpi=300)
plt.close()