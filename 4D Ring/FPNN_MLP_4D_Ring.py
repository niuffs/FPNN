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

    x1 = torch.cat([X, Y, zeros, zeros], dim=1)
    x2 = torch.cat([X, Y, 0.5 * ones, 0.5 * ones], dim=1)
    x3 = torch.cat([X, Y, ones, zeros], dim=1)
    x4 = torch.cat([X, Y, ones, ones], dim=1)

    return [x1, x2, x3, x4]


class DataLoader(object):
    def __init__(self, N_in, path, shuffle=True):
        self.x = np.load(path + 'data.npy')
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
    def __init__(self, layers, xL, xR, dim, device):
        super().__init__()
        self.net = self.create_net(layers)
        self.act = torch.nn.Softplus()
        self.xL = xL
        self.xR = xR
        self.dim = dim
        self.device = device
        self.x_norm = (xR - xL) * torch.rand((20000, dim), dtype=torch.float, device=device, requires_grad=False) + xL
        self.Z_mc = torch.ones(1, requires_grad=False, device=device)

    def create_net(self, layers):
        linears = torch.nn.ModuleList([])
        for i in range(len(layers) - 1):
            f = torch.nn.Linear(layers[i], layers[i + 1], bias=True)
            torch.nn.init.normal_(f.weight, 0, 0.01)
            linears.append(f)

        return linears

    def predict(self, x):
        self.Z_mc.data = ((self.xR - self.xL) ** self.dim * self.forward(self.x_norm).mean()).detach()
        p = self.forward(x) / self.Z_mc

        return p

    def forward(self, x):
        for linear in self.net[:-1]:
            x = torch.tanh(linear(x))
            
        out = self.act(self.net[-1](x))

        return out.sum(dim=1)


class FPNN(torch.nn.Module):
    def __init__(self, config):
        super(FPNN, self).__init__()
        self.dim = config['dim']
        self.N = config['N']
        self.xL = config['xL']
        self.xR = config['xR']
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

        self.p_test = None
        self.model = MLP(self.layers, self.xL, self.xR, self.dim, self.device).to(self.device)
        self.opt_Adam = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.results = {'Steps_error': [i * self.test_freq * self.data.num_batch for i in range(int(self.num_epoch / self.test_freq) + 1)], 'MAE': [], 'MAPE': [], 'Steps_loss':[], 'Score_loss':[], 'Plain_pde_loss':[]}
        self.plot = {'it': [], 'pred': [], 'mae': []}

    def p_true(self, X):
            V = (X.norm(dim=1)**2 - 1) ** 2
            p = torch.exp(-2 * V) / 12.4222285101916
            return p

    def compute_mu(self, x):
        A = torch.tensor([[0, -1, 0, 0],
                          [1., 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]], device=self.device)

        mu = - 4 * x * (x.norm(dim=1, keepdim=True)**2 - 1).repeat(1, 4) + torch.mm(x, A)

        return mu

    def score_pde_loss(self, x):
        logp = torch.log(self.model(x))
        logp_x = torch.autograd.grad(logp.sum(), x, retain_graph=True, create_graph=True)[0]
        mu_ = self.compute_mu(x) - 0.5 * logp_x

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
            self.results['Steps_loss'].append(it)
            self.results['Score_loss'].append(train_loss)
            pde_loss = self.plain_pde_loss(self.x_pde_loss)
            self.results['Plain_pde_loss'].append(pde_loss)
            train_iteration_time = time.time() - train_start

            # Print
            if epoch % self.log_freq == 0:
                print('It: %d, Time: %.2f, score pde loss: %.2e' % (it, train_iteration_time * self.log_freq, train_loss))

            # Test
            if epoch % self.test_freq == 0:
                self.test()

            # Plot
            if epoch % self.plot_freq == 0 or it in [100*(k+1) for k in range(10)] + [5000, 10000]:
                self.plot_fig(it)

        elapsed = time.time() - start
        print('Training complete! Total time: %.2f h' % (elapsed / 3600))

    def plain_pde_loss(self, x):
        x.requires_grad = True
        p = self.model.predict(x)
        p_x = torch.autograd.grad(p.sum(), x, retain_graph=True, create_graph=True)[0]
        mu = self.compute_mu(x)

        residual = trace_df_dz(p.reshape(-1, 1) * mu - 0.5 * p_x, x)
        loss = torch.pow(residual, 2).mean()

        return loss.item()
    
    def test(self):
        if self.p_test == None:
            self.x_plot = create_Xd(n=self.N, xL=self.xL, xR=self.xR, device=self.device)
            self.p_plot = [self.p_true(x) for x in self.x_plot]

            x_pde_loss = np.load(self.path + 'x_pde_loss.npy')
            self.x_pde_loss = (torch.from_numpy(x_pde_loss).float()).to(self.device)
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
        print('Monte Carlo estimation Z_mc: %.2e' % (self.model.Z_mc.item()))
        print('Predict by FPNN: MAE: %.3e, MAPE: %.3e' % (mae, mape))

    def plot_fig(self, it):
        fig, axes = plt.subplots(3, 4, figsize=(18, 10),  subplot_kw={'projection': '3d'})

        X = [x.cpu().detach().numpy() for x in self.x_plot]
        p_true = [p.cpu().detach().numpy().reshape(self.N, self.N) for p in self.p_plot]
        p_pred = [(self.model.predict(x).cpu().detach().numpy()).reshape(self.N, self.N) for x in self.x_plot]
        mae = [np.abs(p_pred[i] - p_true[i]) for i in range(4)]

        vmin = min([d.min() for d in mae])
        vmax = max([d.max() for d in mae])

        axes[0, 0].set_title('$(x_1, x_2, 0, 0)$', fontsize=16)
        axes[0, 1].set_title('$(x_1, x_2, 0.5, 0.5)$', fontsize=16)
        axes[0, 2].set_title('$(x_1, x_2, 1, 0)$', fontsize=16)
        axes[0, 3].set_title('$(x_1, x_2, 1, 1)$', fontsize=16)

        ax_third_row = []

        for i in range(4):
            x = X[i][:, 0].reshape(self.N, self.N)
            y = X[i][:, 1].reshape(self.N, self.N)

            axes[0, i].plot_surface(x, y, p_true[i], cmap='rainbow')
            axes[1, i].plot_surface(x, y, p_pred[i], cmap='rainbow')

            axes[2, i].remove()
            axes[2, i] = fig.add_subplot(3, 4, 9 + i)
            ax_third_row.append(axes[2, i])
            axin = axes[2, i].inset_axes([0.25, 0.075, 0.75, 0.85])
            axes[2, i].axis('off')
            axes[2, i] = axin
            axin.contourf(x, y, mae[i], levels=200, vmin=vmin, vmax=vmax, cmap='viridis')

            for j in range(3):
                axes[j, i].set_xlabel('$x_1$', fontsize=12)
                axes[j, i].set_ylabel('$x_2$', fontsize=12)

        for ax in axes.flat:
            ax.set_xticks(np.linspace(self.xL, self.xR, 5))
            ax.set_yticks(np.linspace(self.xL, self.xR, 5))

        titles = ['Exact Solution', 'FPNN', 'MAE']
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

        if it in [100*(k+1) for k in range(10)] + [5000, 10000]:
            self.plot['it'].append(it)
            self.plot['pred'].append(p_pred)
            self.plot['mae'].append(mae)


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # PDE parameters
    dim = 4
    N = 50
    xL = -1.8
    xR = 1.8
    
    # Hyperparameters
    layers = [dim, 20, 20, 20, 20]
    N_in = 2000
    lr = 1e-2
    num_epoch = 1000
    log_freq = 10
    test_freq = 50
    plot_freq = 50
    path = './data/'

    model_path = path + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '-MLP-' + str(layers)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    config = {
        'dim': dim,
        'N': N,
        'xL': xL,
        'xR': xR,
        'layers': layers,
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
    torch.save(model, model_path + '/FPNN_MLP_4D_Ring.pth')
    np.save(model_path + '/FPNN_MLP_4D_Ring_results.npy', model.results, allow_pickle=True)
    np.save(model_path + '/FPNN_MLP_4D_Ring_plot.npy', model.plot, allow_pickle=True)

    # Loss
    plt.figure(figsize=(8, 6))
    plt.title('Score pde loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(model.results['Steps_loss'], model.results['Score_loss'], zorder=5, label='FPNN')
    plt.legend()
    plt.savefig(model_path + '/score_pde_loss.png')
    plt.close()