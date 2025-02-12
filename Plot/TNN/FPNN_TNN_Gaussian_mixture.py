import os
import time
import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from Gaussian_mixture_training_data import Gaussian_mixture


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
    zeros = torch.zeros(size=(n ** d, 1), device=device)

    x1 = torch.cat([X, Y] + [zeros for i in range(8)], dim=1)
    x2 = torch.cat([X, zeros, Y] + [zeros for i in range(7)], dim=1)

    return [x1, x2]


class DataLoader(object):
    def __init__(self, N_in, path, shuffle=True):
        self.x = np.load(path + '/data.npy')
        self.N_x = self.x.shape[0]
        self.d = self.x.shape[1]
        self.batch_size = N_in
        self.num_batch = self.N_x // self.batch_size
        if shuffle:
            permutation = np.random.permutation(self.N_x)
            self.x = self.x[permutation, :]

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.N_x, self.batch_size * (self.current_ind + 1))
                x_i = self.x[start_ind: end_ind, :]
                yield x_i
                self.current_ind += 1

        return _wrapper()


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
        self.x_norm = (xR - xL) * torch.rand((100000, self.dim), dtype=torch.float, device=self.device, requires_grad=False) + xL
        self.Z_mc = torch.ones(1, requires_grad=False, device=self.device)

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
        self.Z_mc.data = ((self.xR - self.xL) ** self.dim * self.forward(self.x_norm).mean()).detach()

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


class FPNN(torch.nn.Module):
    def __init__(self, config):
        super(FPNN, self).__init__()
        self.dim = config['dim']
        self.N = config['N']
        self.xL = config['xL']
        self.xR = config['xR']
        self.m = config['m']
        self.r = config['r']
        self.layers = config['layers']
        self.N_in = config['N_in']
        self.lr = config['lr']
        self.num_epoch = config['num_epoch']
        self.log_freq = config['log_freq']
        self.test_freq = config['test_freq']
        self.plot_freq = config['plot_freq']
        self.device = config['device']
        self.path = config['path']
        self.model_path = config['model_path']
        self.data = DataLoader(self.N_in, self.path)

        self.GM = Gaussian_mixture(device=self.device)
        self.p_test = None
        self.model = TNN(self.layers, self.m, self.r, self.xL, self.xR, self.dim, self.device)
        self.opt_Adam = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.results = {'Steps_error': [i * self.test_freq * self.data.num_batch for i in range(int(self.num_epoch / self.test_freq) + 1)], 'MAE': [], 'MAPE': [], 'Steps_loss':[], 'Score_loss':[]}
        self.Loss = []
        self.plot = {'it': [], 'pred': [], 'mae': []}

    def p_true(self, x):
        return self.GM.compute_p(x)

    def score_pde_loss(self, x):
        logp = torch.log(self.model(x))
        logp_x = torch.autograd.grad(logp.sum(), x, retain_graph=True, create_graph=True)[0]
        mu_ = self.GM.compute_mu(x) - logp_x

        residual = (logp_x * mu_).sum(dim=1) + trace_df_dz(mu_, x)
        loss = torch.abs(residual).mean()

        return loss

    def train_one_epoch(self):
        train_iterator = self.data.get_iterator()
        loss_list = []
        for idx, x in enumerate(train_iterator):
            x_pde = (torch.from_numpy(x).float()).to(self.device)
            x_pde.requires_grad = True

            self.opt_Adam.zero_grad()
            loss = self.score_pde_loss(x_pde)
            loss.backward()
            self.opt_Adam.step()

            loss_list.append(loss.item())

        return np.mean(loss_list)

    def train_FPNN(self):
        print("Start training!")
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total number of trainable parameters: {total_params}")

        start = time.time()
        print('It: 0', end='  ')
        self.test()
        for epoch in range(1, self.num_epoch + 1):
            it = epoch * self.data.num_batch

            # Train
            train_start = time.time()
            train_loss = self.train_one_epoch()
            self.Loss.append([train_loss])
            self.results['Steps_loss'].append(it)
            self.results['Score_loss'].append(train_loss)
            train_iteration_time = time.time() - train_start

            # Print
            if epoch % self.log_freq == 0:
                print('It: %d, Time: %.2f, score pde loss: %.2e' % (it, train_iteration_time * self.log_freq, train_loss))

            # Test
            if epoch % self.test_freq == 0:
                self.test()

            # Plot
            if epoch % self.plot_freq == 0 or it in [100, 200, 400, 600, 800, 1000, 5000]:
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
        print('Partition function Z: %.2e' % (self.model.Z.item()), end='  ')
        print('Monte Carlo estimation Z_mc: %.2e' % (self.model.Z_mc.item()))
        print('Predict by FPNN: MAE: %.3e, MAPE: %.3e' % (mae, mape))

    def plot_fig(self, it):
        fig, axes = plt.subplots(1, 6, figsize=(20, 3), subplot_kw={'projection': '3d'})

        X = [x.cpu().detach().numpy() for x in self.x_plot]
        p_true = [p.cpu().detach().numpy().reshape(self.N, self.N) for p in self.p_plot]
        p_pred = [(self.model.predict(x).cpu().detach().numpy()).reshape(self.N, self.N) for x in self.x_plot]
        mae = [np.abs(p_pred[i] - p_true[i]) for i in range(2)]

        vmin = min([d.min() for d in mae])
        vmax = max([d.max() for d in mae])
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap('viridis')
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

        axes[0].set_title('Exact solution\n$(x_1,x_2,0,\ldots,0)$')
        axes[1].set_title('$FPNN$')
        axes[2].remove()
        axes[2] = fig.add_subplot(1, 6, 3)
        axin1 = axes[2].inset_axes([0.2, 0.1, 0.8, 0.8])
        axin1.set_title('$MAE$')
        axes[2].axis('off')

        x1 = X[0][:, 0].reshape(self.N, self.N)
        y1 = X[0][:, 1].reshape(self.N, self.N)
        axes[0].plot_surface(x1, y1, p_true[0], cmap='rainbow')
        axes[1].plot_surface(x1, y1, p_pred[0], cmap='rainbow')
        axin1.contourf(x1, y1, mae[0], levels=200, vmin=vmin, vmax=vmax, cmap='viridis')
        cbar1 = fig.colorbar(sm, ax=axes[2], fraction=0.05, pad=0.05, shrink=0.8)
        cbar1.ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        axes[3].set_title('Exact solution\n$(x_1,0,x_3,0,\ldots,0)$')
        axes[4].set_title('$FPNN$')
        axes[5].remove()
        axes[5] = fig.add_subplot(1, 6, 6)
        axin2 = axes[5].inset_axes([0.2, 0.1, 0.8, 0.8])
        axin2.set_title('$MAE$')
        axes[5].axis('off')

        x2 = X[1][:, 0].reshape(self.N, self.N)
        y2 = X[1][:, 2].reshape(self.N, self.N)
        axes[3].plot_surface(x2, y2, p_true[1], cmap='rainbow')
        axes[4].plot_surface(x2, y2, p_pred[1], cmap='rainbow')
        axin2.contourf(x2, y2, mae[1], levels=200, vmin=vmin, vmax=vmax, cmap='viridis')
        cbar2 = fig.colorbar(sm, ax=axes[5], fraction=0.05, pad=0.05, shrink=0.8)
        cbar2.ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        for i in range(6):
            if i < 3:
                axes[i].set_xlabel('$x_1$')
                axes[i].set_ylabel('$x_2$')
            else:
                axes[i].set_xlabel('$x_1$')
                axes[i].set_ylabel('$x_3$')

            axes[i].set_xticks(np.linspace(self.xL, self.xR, 5))
            axes[i].set_yticks(np.linspace(self.xL, self.xR, 5))

        plt.tight_layout(pad=2)

        plt.savefig(self.model_path + "/Steps_" + str(it) + ".png", dpi=300)
        plt.close()

        if it in [100, 200, 400, 600, 800, 1000, 5000]:
            self.plot['it'].append(it)
            self.plot['pred'].append(p_pred)
            self.plot['mae'].append(mae)


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # PDE parameters
    dim = 10
    N = 50
    xL = -5
    xR = 5

    # Hyperparameters
    m = 1
    r = 64
    hidden_layers = [64, 64, 64]
    N_in = 2000
    lr = 1e-2
    num_epoch = 500
    log_freq = 10
    test_freq = 50
    plot_freq = 50
    path = './data/'

    model_path = path + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '-TNN-' + str(m) + '-' + str(hidden_layers) + '-' + str(r)
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
        'num_epoch': num_epoch,
        'log_freq': log_freq,
        'test_freq': test_freq,
        'plot_freq': plot_freq,
        'device': device,
        'path': path,
        'model_path': model_path
    }

    model = FPNN(config).to(device)
    model.train_FPNN()

    # Save
    torch.save(model, model_path + '/FPNN_TNN_Gaussian_mixture.pth')
    np.save(model_path + '/FPNN_TNN_Gaussian_mixture_results.npy', model.results, allow_pickle=True)
    np.save(model_path + '/FPNN_TNN_Gaussian_mixture_plot.npy', model.plot, allow_pickle=True)

    # Loss
    plt.figure(figsize=(8, 6))
    plt.title('Score pde loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(np.arange(1, num_epoch + 1), model.Loss, zorder=5, label='FPNN')
    plt.legend()
    plt.savefig(model_path + '/score_pde_loss.png')
    plt.close()