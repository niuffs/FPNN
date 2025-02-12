import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator

# TNN
path1 = './FPNN_TNN_4D_Ring_plot.npy'
model1 = np.load(path1, allow_pickle=True).item()
X = model1['x']
p = model1['true']
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

N = 50

fig, axes = plt.subplots(2, 4, figsize=(14, 6))

axes[0, 0].axis('off')
axes[0, 0] = fig.add_subplot(2, 4, 1, projection='3d')
axes[0, 0].set_title('$(x_1, x_2, 0, 0)$', fontsize=16)
x = X[0][:, 0].reshape(N, N)
y = X[0][:, 1].reshape(N, N)
axes[0, 0].plot_surface(x, y, p[0].reshape(N, N), cmap='rainbow')

axes[1, 0].axis('off')
axes[1, 0] = fig.add_subplot(2, 4, 5, projection='3d')
axes[1, 0].set_title('$(x_1, x_2, 0.5, 0.5)$', fontsize=16)
x = X[1][:, 0].reshape(N, N)
y = X[1][:, 1].reshape(N, N)
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
    x = X[i][:, 0].reshape(N, N)
    y = X[i][:, 1].reshape(N, N)

    plot(x, y, mae1[i], axes[i, 1])
    plot(x, y, mae2[i], axes[i, 2])
    plot(x, y, mae3[i], axes[i, 3])

for ax in axes.flat:
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)

    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(3))

titles = ['FPNN (TNN)', 'FPNN (MLP)', 'TFFN']
locations = [0.31, 0.565, 0.845]
for i, title in enumerate(titles):
    fig.text(locations[i], 0.96, title, va='center', fontsize=16, fontname='Arial')

plt.tight_layout(pad=2)
plt.savefig("4D Ring MAE.png", dpi=300)
plt.close()