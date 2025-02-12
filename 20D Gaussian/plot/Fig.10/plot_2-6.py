import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FuncFormatter
from FPNN_TNN_Gaussian import MLP, TNN, FPNN, DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_Xd(n, xL, xR, device):
    dim = 20
    d = 2
    x = torch.linspace(xL, xR, n, device=device).unsqueeze(1)

    X = x.repeat(n, 1)
    Y = x.repeat(1, n).reshape(n ** d, 1)
    Xd = []
    for i in range(10):
        Z1 = torch.zeros(size=(n ** d, 2*i), device=device)
        Z2 = torch.zeros(size=(n ** d, dim-2*(i+1)), device=device)
        xi = torch.cat([Z1, X, Y, Z2], dim=1)
        Xd.append(xi)

    return Xd


N = 50
xL = -2
xR = 2
Xd = create_Xd(N, xL, xR, device)

# TNN
FPNN = torch.load('FPNN_TNN_Gaussian.pth', map_location=device)
FPNN.model.device = device
p_pred = [(FPNN.model.predict(x)).cpu().detach().numpy() for x in Xd]
x_plot = FPNN.plot['x'][0]
x = x_plot[:, 0].reshape(N, N)
y = x_plot[:, 1].reshape(N, N)
p_plot = FPNN.plot['true'][0]
X = [x.cpu().detach().numpy() for x in Xd]

fig, axes = plt.subplots(2, 6, figsize=(18, 5), subplot_kw={'projection': '3d'})

axes[0, 0].set_title('Exact Solution', fontsize=16, fontname='Arial')
axes[0, 0].plot_surface(x, y, p_plot.reshape(N, N), cmap='rainbow')

for i in range(10):
    x = X[i][:, 2*i].reshape(N, N)
    y = X[i][:, 2*i+1].reshape(N, N)
    axes[i//5, i%5+1].plot_surface(x, y, p_pred[i].reshape(N, N), cmap='rainbow')
    axes[i//5, i%5+1].set_xlabel(f'$x_{{{2*i+1}}}$', fontsize=10)
    axes[i//5, i%5+1].set_ylabel(f'$x_{{{2*i+2}}}$', fontsize=10)

for ax in axes.flat:
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.zaxis.set_major_locator(MaxNLocator(4))

axes[1, 0].remove()
axes[1, 0] = fig.add_subplot(2, 6, 7)
axin = axes[1, 0].inset_axes([0.25, 0.075, 0.75, 0.85])
axes[1, 0].axis('off')
axes[1, 0] = axin
axes[1, 0].set_xlabel('Steps', fontname='Arial')
axes[1, 0].set_ylabel('MAPE', fontname='Arial')
axes[1, 0].yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
axin.plot(FPNN.results['Steps_error'], FPNN.results['MAPE'], 'o-', zorder=3)

plt.tight_layout(pad=3)
plt.savefig("Gaussian prediction.png", dpi=300)
plt.close()