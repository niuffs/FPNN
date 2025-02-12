import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# TNN
path1 = './FPNN_TNN_4D_Ring_plot.npy'
model1 = np.load(path1, allow_pickle=True).item()
X = model1['x']
p_true = model1['true']
p_pred1 = model1['pred'][-1]
mae1 = model1['mae'][-1]

# MLP
path2 = './FPNN_MLP_4D_Ring_plot.npy'
model2 = np.load(path2, allow_pickle=True).item()
p_pred2 = model2['pred'][-1]
mae2 = model2['mae'][-1]

# TFFN
path3 = './TFFN_4D_Ring_plot.npy'
model3 = np.load(path3, allow_pickle=True).item()
p_pred3 = model3['pred'][-1]
mae3 = model3['mae'][-1]

fig, axes = plt.subplots(4, 4, figsize=(16, 12), subplot_kw={'projection': '3d'})

axes[0, 0].set_title('$(x_1, x_2, 0, 0)$', fontsize=16)
axes[0, 1].set_title('$(x_1, x_2, 0.5, 0.5)$', fontsize=16)
axes[0, 2].set_title('$(x_1, x_2, 1, 0)$', fontsize=16)
axes[0, 3].set_title('$(x_1, x_2, 1, 1)$', fontsize=16)

N = 50

vmin_list = [d.min() for d in p_true]
vmax_list = [d.max() for d in p_true]

for i in range(4):
    x = X[i][:, 0].reshape(N, N)
    y = X[i][:, 1].reshape(N, N)

    axes[0, i].plot_surface(x, y, p_true[i].reshape(N, N), cmap='rainbow', vmin=vmin_list[i], vmax=vmax_list[i])
    axes[1, i].plot_surface(x, y, p_pred1[i].reshape(N, N), cmap='rainbow', vmin=vmin_list[i], vmax=vmax_list[i])
    axes[2, i].plot_surface(x, y, p_pred2[i].reshape(N, N), cmap='rainbow', vmin=vmin_list[i], vmax=vmax_list[i])
    axes[3, i].plot_surface(x, y, p_pred3[i].reshape(N, N), cmap='rainbow', vmin=vmin_list[i], vmax=vmax_list[i])

for ax in axes.flat:
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)

    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.zaxis.set_major_locator(MaxNLocator(5))

titles = ['Exact Solution', 'FPNN (TNN)', 'FPNN (MLP)', 'TFFN']
for i, title in enumerate(titles):
    fig.text(0.02, 0.88 - i * 0.25, title, va='center', rotation='vertical', fontsize=16, fontname='Arial')

plt.tight_layout(pad=2)
plt.savefig("4D Ring Models.png", dpi=300)
plt.close()