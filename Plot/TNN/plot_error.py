import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from FPNN_TNN_Gaussian_mixture import FPNN, TNN, MLP, DataLoader

torch.manual_seed(111)
torch.cuda.manual_seed(111)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

FPNN = torch.load('FPNN_TNN_Gaussian_mixture.pth')
x = FPNN.x_error
p_true = FPNN.p_test.cpu().detach().numpy()
p_ = FPNN.model(x).cpu().detach().numpy()

xR = FPNN.xR
xL = FPNN.xL
dim = FPNN.dim

Z_list = []
mape_list = []

x_data = [i*1000 for i in range(5, 100, 5)]
x_labels = [str(i)+'k' for i in range(5, 100, 5)] + [r'$Z_{GL}$']

for N in x_data:
    x_norm = (xR - xL) * torch.rand((N, dim), dtype=torch.float, device=device, requires_grad=False) + xL
    Z = ((xR - xL) ** dim * FPNN.model(x_norm).mean()).cpu().detach().numpy()
    p_pred = p_/Z
    mape = np.abs((p_pred - p_true) / p_true).mean()
    Z_list.append(Z)
    mape_list.append(mape)

Z_int = FPNN.model.Z.item()
p_pred = p_/Z_int
mape = np.abs((p_pred - p_true) / p_true).mean()
Z_list.append(Z_int)
mape_list.append(mape)

# Data
x = np.arange(20)
y1 = np.array(Z_list)
y2 = np.array(mape_list)

fig, ax1 = plt.subplots(figsize=(5, 4))

ax1.plot(x, y1, 's-.', ms=3, color='#4C72B0', alpha=1, label='$Z$')
ax1.set_xlabel('$|\mathcal{D}_Z|$', fontsize=18, fontname='Arial')
ax1.set_xticks(x[1::2])
ax1.set_xticklabels(x_labels[1::2])
ax1.set_ylabel('Partition Function', fontsize=14, fontname='Arial')
ax1.tick_params(axis='y')
ax1.set_ylim(0, y1.max()*1.15)
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.plot(x, y2, 'o-', ms=4, color='#DD8452', alpha=1, label='$MAPE$')
ax2.set_ylabel('$MAPE$', fontsize=14, fontname='Arial')
ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
ax2.tick_params(axis='y')
ax2.set_ylim(0, y2.max()*1.15)
ax2.legend(loc='upper right')

fig.tight_layout()
plt.savefig("TNN norm.png", dpi=300)
plt.close()
