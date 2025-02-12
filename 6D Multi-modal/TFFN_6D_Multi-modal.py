import os
import time
import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(111)


def create_Xd(n, xL, xR, device):
    d = 2
    x = torch.linspace(xL, xR, n, device=device).unsqueeze(1)

    X = x.repeat(n, 1)
    Y = x.repeat(1, n).reshape(n ** d, 1)
    ones = torch.ones(size=(n ** d, 1), device=device)
    zeros = torch.zeros(size=(n ** d, 1), device=device)

    x1 = torch.cat([X, Y] + [zeros for _ in range(4)], dim=1)
    x2 = torch.cat([X, zeros, Y] + [zeros for _ in range(3)], dim=1)
    x3 = torch.cat([ones, X, Y] + [zeros for _ in range(3)], dim=1)
    x4 = torch.cat([zeros, X, zeros, Y, zeros, zeros], dim=1)

    return [x1, x2, x3, x4]


def trace_df_dz(f, z):
    sum_diag = 0.
    for i in range(f.shape[1]):
        sum_diag += torch.autograd.grad(f[:, i].sum(), z, create_graph=True)[0].contiguous()[:, i].contiguous()

    return sum_diag.contiguous()


class MLP(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.net = self.create_net(layers)
        self.act = torch.nn.Softplus()

    def create_net(self, layers):
        linears = torch.nn.ModuleList([])
        for i in range(len(layers) - 1):
            f = torch.nn.Linear(layers[i], layers[i + 1], bias=True)
            torch.nn.init.normal_(f.weight, 0, 0.01)
            linears.append(f)

        return linears

    def forward(self, x):
        for linear in self.net[:-1]:
            x = torch.tanh(linear(x))
            
        out = self.act(self.net[-1](x))

        return out


class TNN(torch.nn.Module):
    def __init__(self, layers, m, r, xL, xR, dim, device):
        super().__init__()
        self.m = m
        self.r = r
        self.xL = xL
        self.xR = xR
        self.dim = dim
        self.device = device
        self.net = self.create_model(layers)

    def create_model(self, layers):
        layers[0:0] = [self.m]
        layers.append(self.r)
        net = torch.nn.ModuleList([MLP(layers).to(self.device) for i in range(self.dim)])

        return net

    def integrate(self, a, b, n=16):
        # The roots xi and weights wi of Legendre polynomial
        nodes, w = torch.tensor(np.array(np.polynomial.legendre.leggauss(n)), dtype=torch.float, device=self.device)

        # Transform xi and wi to [a, b]
        t = (0.5 * (nodes + 1) * (b - a) + a).reshape(-1, 1)
        w = (0.5 * (b - a) * w).reshape(1, -1)

        X = [t ** i for i in range(1, self.m + 1)]
        X = torch.cat(X, dim=1)

        out = [torch.mm(w, self.net[i](X)) for i in range(self.dim)]

        return torch.cat(out, dim=0)

    def predict(self, x, num_intervals=10):
        xs = torch.linspace(self.xL, self.xR, num_intervals + 1)
        int_a_b = 0
        for i in range(num_intervals):
            int_a_b += self.integrate(xs[i], xs[i + 1])

        out = 1
        for i in range(self.dim):
            out *= int_a_b[i]

        self.Z = out.sum()

        p = self.forward(x) / self.Z

        return p

    def forward(self, x):
        x = x.unsqueeze(2)
        X = [x ** i for i in range(1, self.m + 1)]
        X = torch.cat(X, dim=2)
        out = self.net[0](X[:, 0, :])
        for i in range(1, self.dim):
            out = out * self.net[i](X[:, i, :])

        return out.sum(dim=1)


class TFFN(torch.nn.Module):
    def __init__(self, config):
        super(TFFN, self).__init__()
        self.dim = config['dim']
        self.N = config['N']
        self.xL = config['xL']
        self.xR = config['xR']
        self.m = config['m']
        self.r = config['r']
        self.layers = config['layers']
        self.N_in = config['N_in']
        self.lr = config['lr']
        self.num_steps = config['num_steps']
        self.log_freq = config['log_freq']
        self.test_freq = config['test_freq']
        self.plot_freq = config['plot_freq']
        self.device = config['device']
        self.path = config['path']
        self.model_path = config['model_path']

        self.p_test = None
        self.model = TNN(self.layers, self.m, self.r, self.xL, self.xR, self.dim, self.device)
        self.opt_Adam = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.results = {'Steps_error': [it for it in range(int(self.num_steps / self.test_freq) + 1)], 'MAE': [], 'MAPE': [], 'Steps_loss':[], 'Train_loss':[]}
        self.plot = {'it': [], 'pred': [], 'mae': []}

    def p_true(self, X):
        H = 2 * (X[:, 0]**2 + X[:, 1]**2 + X[:, 2]**2 + 0.5 * (X[:, 0] * X[:, 1] + X[:, 0] * X[:, 2] + X[:, 1] * X[:, 2])) - torch.log(X[:, 0]**2 + 0.02) - torch.log(X[:, 1]**2 + 0.02) + 0.5 * (X[:, 3]**2 + X[:, 4]**2 + X[:, 5]**2 + 0.2 * (X[:, 3] * X[:, 4] + X[:, 3] * X[:, 5] + X[:, 4] * X[:, 5]))
        p = torch.exp(-H) / 3.24729309964116
        return p

    def compute_mu(self, x):
        A = torch.tensor([[4., 1, 1, 0, 0, 0],
                          [1., 4, 1, 0, 0, 0],
                          [1, 1, 4, 0, 0, 0],
                          [0, 0, 0, 1, 0.1, 0.1],
                          [0, 0, 0, 0.1, 1, 0.1],
                          [0, 0, 0, 0.1, 0.1, 1]], device=self.device)
        mask = torch.tensor([[1., 1, 0, 0, 0, 0]], device=self.device)

        dH = torch.mm(x, A) - 2 * x / (x ** 2 + 0.02) * mask

        return - dH
        
    def plain_pde_loss(self, x):
        x.requires_grad = True
        p = self.model.predict(x)
        p_x = torch.autograd.grad(p.sum(), x, retain_graph=True, create_graph=True)[0]
        mu = self.compute_mu(x)

        residual = trace_df_dz(p.reshape(-1, 1) * mu - p_x, x)
        loss = torch.pow(residual, 2).mean()

        return loss

    def train_one_step(self):
        x_pde = (xR - xL) * torch.rand((self.N_in, self.dim), dtype=torch.float, device=self.device) + xL
        x_pde.requires_grad = True

        self.opt_Adam.zero_grad()
        loss = self.plain_pde_loss(x_pde)
        loss.backward()
        self.opt_Adam.step()

        return loss.item()

    def train_TFFN(self):
        print("Start training!")
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total number of trainable parameters: {total_params}")

        start = time.time()
        print('It: 0', end='  ')
        self.test()
        for it in range(1, self.num_steps + 1):
            # Train
            train_start = time.time()
            train_loss = self.train_one_step()
            train_iteration_time = time.time() - train_start

            # Print
            if it % self.log_freq == 0:
                self.results['Steps_loss'].append(it)
                self.results['Train_loss'].append(train_loss)
                print('It: %d, Time: %.2f, pde loss: %.2e' % (it, train_iteration_time * self.log_freq, train_loss))

            # Test
            if it % self.test_freq == 0:
                self.test()

            # Plot
            if it % self.plot_freq == 0 or it in [1000, 5000, 10000, 20000]:
                self.plot_fig(it)

        elapsed = time.time() - start
        print('Training complete! Total time: %.2f h' % (elapsed / 3600))

    def test(self):
        if self.p_test == None:
            self.x_plot = create_Xd(n=self.N, xL=self.xL, xR=self.xR, device=self.device)
            self.p_plot = [self.p_true(x) for x in self.x_plot]

            x_error = np.load(self.path + 'x_error.npy')
            self.x_error = (torch.from_numpy(x_error).float()).to(self.device)
            self.p_test = self.p_true(self.x_error)
            self.plot['x'] = [x.cpu().detach().numpy() for x in self.x_plot]
            self.plot['true'] = [p.cpu().detach().numpy() for p in self.p_plot]

        p_pred = self.model.predict(self.x_error)
        mae = torch.abs(p_pred - self.p_test).mean().item()
        mape = torch.abs((p_pred - self.p_test) / self.p_test).mean().item()
        self.results['MAE'].append(mae)
        self.results['MAPE'].append(mape)
        print('Partition function Z: %.2e' % (self.model.Z.item()))
        print('Predict by FPNN: MAE: %.3e, MAPE: %.3e' % (mae, mape))

    def plot_fig(self, it):
        fig, axes = plt.subplots(3, 4, figsize=(18, 10),  subplot_kw={'projection': '3d'})

        X = [x.cpu().detach().numpy() for x in self.x_plot]
        p_true = [p.cpu().detach().numpy().reshape(self.N, self.N) for p in self.p_plot]
        p_pred = [(self.model.predict(x).cpu().detach().numpy()).reshape(self.N, self.N) for x in self.x_plot]
        mae = [np.abs(p_pred[i] - p_true[i]) for i in range(4)]

        vmin = min([d.min() for d in mae])
        vmax = max([d.max() for d in mae])

        axes[0, 0].set_title('$(x_1, x_2, 0, 0, 0, 0)$', fontsize=16)
        axes[0, 1].set_title('$(x_1, 0, x_3, 0, 0, 0)$', fontsize=16)
        axes[0, 2].set_title('$(1, x_2, x_3, 0, 0, 0)$', fontsize=16)
        axes[0, 3].set_title('$(0, x_2, 0, x_4, 0, 0)$', fontsize=16)

        ax_third_row = []

        for i in range(4):
            if i == 0:
                x = X[i][:, 0].reshape(self.N, self.N)
                y = X[i][:, 1].reshape(self.N, self.N)
            elif i == 1:
                x = X[i][:, 0].reshape(self.N, self.N)
                y = X[i][:, 2].reshape(self.N, self.N)
            elif i == 2:
                x = X[i][:, 1].reshape(self.N, self.N)
                y = X[i][:, 2].reshape(self.N, self.N)
            elif i == 3:
                x = X[i][:, 1].reshape(self.N, self.N)
                y = X[i][:, 3].reshape(self.N, self.N)

            axes[0, i].plot_surface(x, y, p_true[i], cmap='rainbow')
            axes[1, i].plot_surface(x, y, p_pred[i], cmap='rainbow')

            axes[2, i].remove()
            axes[2, i] = fig.add_subplot(3, 4, 9 + i)
            ax_third_row.append(axes[2, i])
            axin = axes[2, i].inset_axes([0.25, 0.075, 0.75, 0.85])
            axes[2, i].axis('off')
            axes[2, i] = axin
            axin.contourf(x, y, mae[i], levels=200, vmin=vmin, vmax=vmax, cmap='viridis')

        for i in range(3):
            axes[i, 0].set_xlabel('$x_1$', fontsize=12)
            axes[i, 0].set_ylabel('$x_2$', fontsize=12)
            axes[i, 1].set_xlabel('$x_1$', fontsize=12)
            axes[i, 1].set_ylabel('$x_3$', fontsize=12)
            axes[i, 2].set_xlabel('$x_2$', fontsize=12)
            axes[i, 2].set_ylabel('$x_3$', fontsize=12)
            axes[i, 3].set_xlabel('$x_2$', fontsize=12)
            axes[i, 3].set_ylabel('$x_4$', fontsize=12)

        for ax in axes.flat:
            ax.set_xticks(np.linspace(self.xL, self.xR, 5))
            ax.set_yticks(np.linspace(self.xL, self.xR, 5))

        titles = ['Exact Solution', 'TFFN', 'MAE']
        for i, title in enumerate(titles):
            fig.text(0.01, 0.85 - i * 0.33, title, va='center', rotation='vertical', fontsize=16)

        plt.tight_layout()

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap('viridis')
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, ax=ax_third_row, fraction=0.05, pad=0.01, shrink=0.85)
        cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        plt.savefig(self.model_path + "/Steps_" + str(it) + ".png", dpi=300)
        plt.close()

        if it in [1000, 5000, 10000, 20000]:
            self.plot['it'].append(it)
            self.plot['pred'].append(p_pred)
            self.plot['mae'].append(mae)

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # PDE parameters
    dim = 6
    N = 50
    xL = -2
    xR = 2
    
    # Hyperparameters
    m = 1
    r = 64
    hidden_layers = [64, 64]
    N_in = 2000
    lr = 1e-2
    num_steps = 20000
    log_freq = 100
    test_freq = 500
    plot_freq = 500
    path = './data/'

    model_path = path + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '-TFFN-' + str([m] + hidden_layers + [r])
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    config = {
        'dim': dim,
        'N': N,
        'xL': xL,
        'xR': xR,
        'm': m,
        'r': r,
        'layers': hidden_layers,
        'N_in': N_in,
        'lr': lr,
        'num_steps': num_steps,
        'log_freq': log_freq,
        'test_freq': test_freq,
        'plot_freq': plot_freq,
        'device': device,
        'path': path,
        'model_path': model_path
    }

    model = TFFN(config).to(device)
    model.train_TFFN()

    # Save
    torch.save(model, model_path + '/TFFN_6D_Multi-modal.pth')
    np.save(model_path + '/TFFN_6D_Multi-modal_results.npy', model.results, allow_pickle=True)
    np.save(model_path + '/TFFN_6D_Multi-modal_plot.npy', model.plot, allow_pickle=True)

    # Loss
    plt.figure(figsize=(8, 6))
    plt.title('Plain pde loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.plot(model.results['Steps_loss'], model.results['Train_loss'], zorder=5, label='TFFN')
    plt.legend()
    plt.savefig(model_path + '/pde_loss.png')
    plt.close()