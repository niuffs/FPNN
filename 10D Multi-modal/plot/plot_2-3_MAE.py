import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator

# TNN
path1 = './FPNN_TNN_10D_Multi-modal_plot.npy'
model1 = np.load(path1, allow_pickle=True).item()
X = model1['x']
p = model1['true']
p_pred1 = model1['pred'][-1]
mae1 = model1['mae'][-1]

# MLP
path2 = './FPNN_MLP_10D_Multi-modal_plot.npy'
model2 = np.load(path2, allow_pickle=True).item()
p_pred2 = model2['pred'][-1]
mae2 = model2['mae'][-1]

N = 50

fig, axes = plt.subplots(2, 3, figsize=(11, 6))

axes[0, 0].axis('off')
axes[0, 0] = fig.add_subplot(2, 3, 1, projection='3d')
axes[0, 0].set_title('$(0, \ldots, 0, x_9, x_{10})$', fontsize=16)
x = X[0][:, 8].reshape(N, N)
y = X[0][:, 9].reshape(N, N)
axes[0, 0].plot_surface(x, y, p[0].reshape(N, N), cmap='rainbow')

axes[1, 0].axis('off')
axes[1, 0] = fig.add_subplot(2, 3, 4, projection='3d')
axes[1, 0].set_title('$(0, \ldots, 0, x_8, 0.5, x_{10})$', fontsize=16)
x = X[1][:, 7].reshape(N, N)
y = X[1][:, 9].reshape(N, N)
axes[1, 0].plot_surface(x, y, p[1].reshape(N, N), cmap='rainbow')


def plot(x, y, mae, ax):
    vmin = mae.min()
    vmax = mae.max()
    contour = ax.contourf(x, y, mae.reshape(N, N), levels=200, vmin=vmin, vmax=vmax, cmap='jet')
    cbar = fig.colorbar(contour, ax=ax, shrink=1)

    locator = ticker.MaxNLocator(nbins=10, integer=True)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))

    cbar.ax.yaxis.set_major_locator(locator)
    cbar.ax.yaxis.set_major_formatter(formatter)


for i in range(2):
    if i == 0:
        x = X[i][:, 8].reshape(N, N)
        y = X[i][:, 9].reshape(N, N)
    elif i == 1:
        x = X[i][:, 7].reshape(N, N)
        y = X[i][:, 9].reshape(N, N)

    plot(x, y, mae1[i], axes[i, 1])
    plot(x, y, mae2[i], axes[i, 2])

    for j in range(3):
        if i == 0:
            axes[i, j].set_xlabel('$x_9$', fontsize=12)
            axes[i, j].set_ylabel('$x_{10}$', fontsize=12)
        elif i == 1:
            axes[i, j].set_xlabel('$x_8$', fontsize=12)
            axes[i, j].set_ylabel('$x_{10}$', fontsize=12)

for ax in axes.flat:
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(3))

titles = ['FPNN (TNN)', 'FPNN (MLP)']
for i, title in enumerate(titles):
    fig.text(0.42 + i * 0.34, 0.96, title, va='center', fontsize=16, fontname='Arial')

plt.tight_layout(pad=2)
plt.savefig("10D Multi-modal MAE.png", dpi=300)
plt.close()