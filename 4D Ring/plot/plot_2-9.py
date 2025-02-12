import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors

# TFFN
path1 = './TFFN_4D_Ring_plot.npy'
model1 = np.load(path1, allow_pickle=True).item()
steps1 = model1['it']
print('TFFN steps: ', steps1)  #[1000, 5000, 10000, 20000]
p_pred1 = model1['pred']
mae1 = model1['mae']

X = model1['x']
p_true = model1['true']

# MLP
path2 = './FPNN_MLP_4D_Ring_plot.npy'
model2 = np.load(path2, allow_pickle=True).item()
steps2 = model2['it']
print('FPNN steps: ', steps2)  #[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 5k, 10k]
p_pred2 = model2['pred']
mae2 = model2['mae']


fig, axes = plt.subplots(2, 9, figsize=(26, 5.5))

N = 50

x1 = X[0][:, 0].reshape(N, N)
y1 = X[0][:, 1].reshape(N, N)

x2 = X[2][:, 0].reshape(N, N)
y2 = X[2][:, 1].reshape(N, N)

axes[0, 0].set_title('$(x_1, x_2, 0, 0)$', fontsize=16)
axes[0, 0].axis('off')
axes[0, 0] = fig.add_subplot(2, 9, 1, projection='3d')
axes[0, 0].plot_surface(x1, y1, p_true[0].reshape(N, N), cmap='rainbow')
axes[0, 0].set_xlabel('$x_1$', fontsize=10)
axes[0, 0].set_ylabel('$x_2$', fontsize=10)

axes[1, 0].set_title('$(x_1, x_2, 0.5, 0.5)$', fontsize=16)
axes[1, 0].axis('off')
axes[1, 0] = fig.add_subplot(2, 9, 10, projection='3d')
axes[1, 0].plot_surface(x2, y2, p_true[2].reshape(N, N), cmap='rainbow')
axes[1, 0].set_xlabel('$x_1$', fontsize=10)
axes[1, 0].set_ylabel('$x_2$', fontsize=10)

p_pred_TFFN_1k = p_pred1[0]
mae_TFFN_1k = mae1[0]

p_pred_TFFN_10k = p_pred1[2]
mae_TFFN_10k = mae1[2]

p_pred_FPNN_800 = p_pred2[7]
mae_FPNN_800 = mae2[7]

p_pred_FPNN_1k = p_pred2[9]
mae_FPNN_1k = mae2[9]

vmin = min([d.min() for d in mae_TFFN_1k + mae_TFFN_10k + mae_FPNN_800 + mae_FPNN_1k])
vmax = max([d.max() for d in mae_TFFN_1k + mae_TFFN_10k + mae_FPNN_800 + mae_FPNN_1k])


def plot_2(fig_id, x, y, p, mae):
    axes[0, fig_id].axis('off')
    axes[0, fig_id] = fig.add_subplot(2, 9, fig_id + 1, projection='3d')
    axes[0, fig_id].plot_surface(x, y, p.reshape(N, N), cmap='rainbow')

    axes[1, fig_id].axis('off')
    axin = axes[1, fig_id].inset_axes([0.16, 0.07, 0.75, 0.86])
    axin.contourf(x, y, mae.reshape(N, N), levels=200, vmin=vmin, vmax=vmax, cmap='jet')

    axes[0, fig_id].set_xlabel('$x_1$', fontsize=10)
    axes[0, fig_id].set_ylabel('$x_2$', fontsize=10)
    axin.set_xlabel('$x_1$', fontsize=10, labelpad=-1)
    axin.set_ylabel('$x_2$', fontsize=10, labelpad=-5)

    return


plot_2(1, x1, y1, p_pred_TFFN_1k[0], mae_TFFN_1k[0])
plot_2(2, x2, y2, p_pred_TFFN_1k[3], mae_TFFN_1k[3])
plot_2(3, x1, y1, p_pred_TFFN_10k[0], mae_TFFN_10k[0])
plot_2(4, x2, y2, p_pred_TFFN_10k[3], mae_TFFN_10k[3])
plot_2(5, x1, y1, p_pred_FPNN_800[0], mae_FPNN_800[0])
plot_2(6, x2, y2, p_pred_FPNN_800[3], mae_FPNN_800[3])
plot_2(7, x1, y1, p_pred_FPNN_1k[0], mae_FPNN_1k[0])
plot_2(8, x2, y2, p_pred_FPNN_1k[3], mae_FPNN_1k[3])


plt.tight_layout(pad=3)
plt.subplots_adjust(wspace=0.15, hspace=0.35)

norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
cmap = plt.get_cmap('jet')
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar_ax = fig.add_axes([0.96, 0.14, 0.02, 0.28])
cbar_ax.axis('off')
cbar = fig.colorbar(sm, ax=cbar_ax)
cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

fig.text(0.23, 0.90, 'TFFN, 1k steps', ha='center', fontsize=18.5, fontname='Arial', weight='light')
fig.text(0.45, 0.90, 'TFFN, 10k steps', ha='center', fontsize=18.5, fontname='Arial', weight='light')
fig.text(0.67, 0.90, 'FPNN (MLP), 0.8k steps', ha='center', fontsize=18.5, fontname='Arial', weight='semibold')
fig.text(0.89, 0.90, 'FPNN (MLP), 1k steps', ha='center', fontsize=18.5, fontname='Arial', weight='semibold')

plt.savefig("4D Ring Steps.png", dpi=300)
plt.close()