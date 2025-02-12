import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# TNN
path1 = './FPNN_TNN_10D_Multi-modal_plot.npy'
model1 = np.load(path1, allow_pickle=True).item()
X = model1['x']
p_true = model1['true']
p_pred1 = model1['pred'][-1]
mae1 = model1['mae'][-1]

# MLP
path2 = './FPNN_MLP_10D_Multi-modal_plot.npy'
model2 = np.load(path2, allow_pickle=True).item()
p_pred2 = model2['pred'][-1]
mae2 = model2['mae'][-1]

fig, axes = plt.subplots(3, 4, figsize=(15, 9), subplot_kw={'projection': '3d'})

axes[0, 0].set_title('$(0, \ldots, 0, x_9, x_{10})$', fontsize=16)
axes[0, 1].set_title('$(0, \ldots, 0, x_8, 0.5, x_{10})$', fontsize=16)
axes[0, 2].set_title('$(0, \ldots, 0, x_8, x_9, 0)$', fontsize=16)
axes[0, 3].set_title('$(0, \ldots, 0, x_7, x_8, 0.5, 0)$', fontsize=16)

N = 50

for i in range(4):
    if i == 0:
        x = X[i][:, 8].reshape(N, N)
        y = X[i][:, 9].reshape(N, N)
    elif i == 1:
        x = X[i][:, 7].reshape(N, N)
        y = X[i][:, 9].reshape(N, N)
    elif i == 2:
        x = X[i][:, 7].reshape(N, N)
        y = X[i][:, 8].reshape(N, N)
    elif i == 3:
        x = X[i][:, 6].reshape(N, N)
        y = X[i][:, 7].reshape(N, N)

    axes[0, i].plot_surface(x, y, p_true[i].reshape(N, N), cmap='rainbow')
    axes[1, i].plot_surface(x, y, p_pred1[i].reshape(N, N), cmap='rainbow')
    axes[2, i].plot_surface(x, y, p_pred2[i].reshape(N, N), cmap='rainbow')

for i in range(3):
    axes[i, 0].set_xlabel('$x_9$', fontsize=10)
    axes[i, 0].set_ylabel('$x_{10}$', fontsize=10)
    axes[i, 1].set_xlabel('$x_8$', fontsize=10)
    axes[i, 1].set_ylabel('$x_{10}$', fontsize=10)
    axes[i, 2].set_xlabel('$x_8$', fontsize=10)
    axes[i, 2].set_ylabel('$x_9$', fontsize=10)
    axes[i, 3].set_xlabel('$x_7$', fontsize=10)
    axes[i, 3].set_ylabel('$x_8$', fontsize=10)

for ax in axes.flat:
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.zaxis.set_major_locator(MaxNLocator(4))

titles = ['Exact Solution', 'FPNN (TNN)', 'FPNN (MLP)']
for i, title in enumerate(titles):
    fig.text(0.02, 0.84 - i * 0.33, title, va='center', rotation='vertical', fontsize=16, fontname='Arial')

plt.tight_layout(pad=3)
plt.savefig("10D Multi-modal Models.png", dpi=300)
plt.close()