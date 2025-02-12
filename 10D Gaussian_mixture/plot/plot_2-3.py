import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# TNN
path1 = './FPNN_TNN_Gaussian_mixture_plot.npy'
model1 = np.load(path1, allow_pickle=True).item()
X = model1['x']
p_true = model1['true']
p_pred1 = model1['pred'][-1]
mae1 = model1['mae'][-1]

# MLP
path2 = './FPNN_MLP_Gaussian_mixture_plot.npy'
model2 = np.load(path2, allow_pickle=True).item()
p_pred2 = model2['pred'][-1]
mae2 = model2['mae'][-1]

fig, axes = plt.subplots(2, 3, figsize=(12, 6), subplot_kw={'projection': '3d'})

axes[0, 0].set_title('Exact Solution', fontsize=16, fontname='Arial')
axes[0, 1].set_title('FPNN (TNN)', fontsize=16, fontname='Arial')
axes[0, 2].set_title('FPNN (MLP)', fontsize=16, fontname='Arial')

N = 50

for i in range(2):
    if i == 0:
        x = X[i][:, 0].reshape(N, N)
        y = X[i][:, 1].reshape(N, N)
    elif i == 1:
        x = X[i][:, 0].reshape(N, N)
        y = X[i][:, 2].reshape(N, N)

    axes[i, 0].plot_surface(x, y, p_true[i].reshape(N, N), cmap='rainbow')
    axes[i, 1].plot_surface(x, y, p_pred1[i].reshape(N, N), cmap='rainbow')
    axes[i, 2].plot_surface(x, y, p_pred2[i].reshape(N, N), cmap='rainbow')

for i in range(3):
    axes[0, i].set_xlabel('$x_1$', fontsize=10)
    axes[0, i].set_ylabel('$x_2$', fontsize=10)
    axes[1, i].set_xlabel('$x_1$', fontsize=10)
    axes[1, i].set_ylabel('$x_3$', fontsize=10)

for ax in axes.flat:
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.zaxis.set_major_locator(MaxNLocator(4))

titles = ['$(x_1,x_2,0,\ldots,0)$', '$(x_1,0,x_3,0,\ldots,0)$']
for i, title in enumerate(titles):
    fig.text(0.03, 0.75 - i * 0.50, title, va='center', rotation='vertical', fontsize=16, fontname='Arial')

plt.tight_layout(pad=3)
plt.savefig("Gaussian mixture Models.png", dpi=300)
plt.close()