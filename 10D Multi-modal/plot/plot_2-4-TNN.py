import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

path = './FPNN_TNN_10D_Multi-modal_plot.npy'

model = np.load(path, allow_pickle=True).item()
steps = model['it']
print('FPNN steps: ', steps)
p_pred = model['pred']
X = model['x']
p_true = model['true']

fig, axes = plt.subplots(2, 4, figsize=(16, 6))

N = 50

x1 = X[0][:, 8].reshape(N, N)
y1 = X[0][:, 9].reshape(N, N)

x2 = X[2][:, 7].reshape(N, N)
y2 = X[2][:, 8].reshape(N, N)

axes[0, 0].set_title('Exact solution', fontsize=22, fontname='Arial')
axes[0, 0].axis('off')
axes[0, 0] = fig.add_subplot(2, 4, 1, projection='3d')
axes[0, 0].plot_surface(x1, y1, p_true[0].reshape(N, N), cmap='rainbow')
axes[0, 0].set_xlabel('$x_9$', fontsize=12)
axes[0, 0].set_ylabel('$x_{10}$', fontsize=12)

axes[1, 0].axis('off')
axes[1, 0] = fig.add_subplot(2, 4, 5, projection='3d')
axes[1, 0].plot_surface(x2, y2, p_true[2].reshape(N, N), cmap='rainbow')
axes[1, 0].set_xlabel('$x_8$', fontsize=12)
axes[1, 0].set_ylabel('$x_9$', fontsize=12)

steps_list = ['0.1k', '0.2k', '0.3k']
idx_list = [1, 3, 5]

for i in range(1, 4):
    id = idx_list[i - 1]
    axes[0, i].set_title(steps_list[i - 1] + ' steps', fontsize=22, fontname='Arial')
    axes[0, i].axis('off')
    axes[0, i] = fig.add_subplot(2, 4, i + 1, projection='3d')
    axes[0, i].plot_surface(x1, y1, p_pred[id][0].reshape(N, N), cmap='rainbow')
    axes[0, i].set_xlabel('$x_9$', fontsize=12)
    axes[0, i].set_ylabel('$x_{10}$', fontsize=12)

    axes[1, i].axis('off')
    axes[1, i] = fig.add_subplot(2, 4, i + 5, projection='3d')
    axes[1, i].plot_surface(x2, y2, p_pred[id][2].reshape(N, N), cmap='rainbow')
    axes[1, i].set_xlabel('$x_8$', fontsize=12)
    axes[1, i].set_ylabel('$x_9$', fontsize=12)

for ax in axes.flat:
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.zaxis.set_major_locator(MaxNLocator(4))

plt.tight_layout(pad=3)
fig.text(0.44, 0.95, 'FPNN (TNN)', va='center', fontsize=26, fontname='Arial', weight='semibold')
plt.savefig("10D Multi-modal TNN Steps.png", dpi=300)
