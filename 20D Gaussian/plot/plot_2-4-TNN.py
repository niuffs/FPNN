import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

path = './FPNN_TNN_Gaussian_plot.npy'

model = np.load(path, allow_pickle=True).item()
steps = model['it']
print('FPNN steps: ', steps)
p_pred = model['pred']
X = model['x']
p_true = model['true']

fig, axes = plt.subplots(2, 4, figsize=(16, 6))

N = 50

x1 = X[0][:, 0].reshape(N, N)
y1 = X[0][:, 1].reshape(N, N)

x2 = X[1][:, 2].reshape(N, N)
y2 = X[1][:, 3].reshape(N, N)

axes[0, 0].set_title('Exact solution', fontsize=22, fontname='Arial')
axes[0, 0].axis('off')
axes[0, 0] = fig.add_subplot(2, 4, 1, projection='3d')
axes[0, 0].plot_surface(x1, y1, p_true[0].reshape(N, N), cmap='rainbow')
axes[0, 0].set_xlabel('$x_1$', fontsize=12)
axes[0, 0].set_ylabel('$x_2$', fontsize=12)

axes[1, 0].axis('off')
axes[1, 0] = fig.add_subplot(2, 4, 5, projection='3d')
axes[1, 0].plot_surface(x2, y2, p_true[1].reshape(N, N), cmap='rainbow')
axes[1, 0].set_xlabel('$x_3$', fontsize=12)
axes[1, 0].set_ylabel('$x_4$', fontsize=12)

steps_list = ['0.5k', '1k', '1.5k']
index = [0, 1, 2]

for i in range(1, 4):
    axes[0, i].set_title(steps_list[i - 1] + ' steps', fontsize=22, fontname='Arial')
    axes[0, i].axis('off')
    axes[0, i] = fig.add_subplot(2, 4, i + 1, projection='3d')
    axes[0, i].plot_surface(x1, y1, p_pred[index[i - 1]][0].reshape(N, N), cmap='rainbow')
    axes[0, i].set_xlabel('$x_1$', fontsize=12)
    axes[0, i].set_ylabel('$x_2$', fontsize=12)

    axes[1, i].axis('off')
    axes[1, i] = fig.add_subplot(2, 4, i + 5, projection='3d')
    axes[1, i].plot_surface(x2, y2, p_pred[index[i - 1]][1].reshape(N, N), cmap='rainbow')
    axes[1, i].set_xlabel('$x_3$', fontsize=12)
    axes[1, i].set_ylabel('$x_4$', fontsize=12)

for ax in axes.flat:
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.zaxis.set_major_locator(MaxNLocator(4))

plt.tight_layout(pad=3)
fig.text(0.45, 0.95, 'FPNN (TNN)', va='center', fontsize=26, fontname='Arial', weight='semibold')
plt.savefig("Gaussian TNN Steps.png", dpi=300)
plt.close()
