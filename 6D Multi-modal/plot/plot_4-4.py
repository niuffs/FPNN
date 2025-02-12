import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# TNN
path1 = './FPNN_TNN_6D_Multi-modal_plot.npy'
model1 = np.load(path1, allow_pickle=True).item()
X = model1['x']
p_true = model1['true']
p_pred1 = model1['pred'][-1]
mae1 = model1['mae'][-1]

# MLP
path2 = './FPNN_MLP_6D_Multi-modal_plot.npy'
model2 = np.load(path2, allow_pickle=True).item()
p_pred2 = model2['pred'][-1]
mae2 = model2['mae'][-1]

# TFFN
path3 = './TFFN_6D_Multi-modal_plot.npy'
model3 = np.load(path3, allow_pickle=True).item()
p_pred3 = model3['pred'][-1]
mae3 = model3['mae'][-1]

fig, axes = plt.subplots(4, 4, figsize=(16, 12), subplot_kw={'projection': '3d'})

axes[0, 0].set_title('$(x_1, x_2, 0, 0, 0, 0)$', fontsize=16)
axes[0, 1].set_title('$(x_1, 0, x_3, 0, 0, 0)$', fontsize=16)
axes[0, 2].set_title('$(1, x_2, x_3, 0, 0, 0)$', fontsize=16)
axes[0, 3].set_title('$(0, x_2, 0, x_4, 0, 0)$', fontsize=16)

N = 50

for i in range(4):
    if i == 0:
        x = X[i][:, 0].reshape(N, N)
        y = X[i][:, 1].reshape(N, N)
    elif i == 1:
        x = X[i][:, 0].reshape(N, N)
        y = X[i][:, 2].reshape(N, N)
    elif i == 2:
        x = X[i][:, 1].reshape(N, N)
        y = X[i][:, 2].reshape(N, N)
    elif i == 3:
        x = X[i][:, 1].reshape(N, N)
        y = X[i][:, 3].reshape(N, N)

    axes[0, i].plot_surface(x, y, p_true[i].reshape(N, N), cmap='rainbow')
    axes[1, i].plot_surface(x, y, p_pred1[i].reshape(N, N), cmap='rainbow')
    axes[2, i].plot_surface(x, y, p_pred2[i].reshape(N, N), cmap='rainbow')
    axes[3, i].plot_surface(x, y, p_pred3[i].reshape(N, N), cmap='rainbow')

    axes[i, 0].set_xlabel('$x_1$', fontsize=12)
    axes[i, 0].set_ylabel('$x_2$', fontsize=12)
    axes[i, 1].set_xlabel('$x_1$', fontsize=12)
    axes[i, 1].set_ylabel('$x_3$', fontsize=12)
    axes[i, 2].set_xlabel('$x_2$', fontsize=12)
    axes[i, 2].set_ylabel('$x_3$', fontsize=12)
    axes[i, 3].set_xlabel('$x_2$', fontsize=12)
    axes[i, 3].set_ylabel('$x_4$', fontsize=12)

for ax in axes.flat:
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.zaxis.set_major_locator(MaxNLocator(4))

titles = ['Exact Solution', 'FPNN (TNN)', 'FPNN (MLP)', 'TFFN']
for i, title in enumerate(titles):
    fig.text(0.02, 0.88 - i * 0.25, title, va='center', rotation='vertical', fontsize=16, fontname='Arial')

plt.tight_layout(pad=2)
plt.savefig("6D Multi-modal Models.png", dpi=300)
plt.close()