import os
import torch
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(111)

class Gaussian_mixture:
    def __init__(self, dim=10, device='cpu'):
        self.dim = dim
        self.device = device
        self.create_model()

    def create_model(self):
        sigma1 = torch.zeros(self.dim, self.dim).to(self.device)
        sigma1[0:3, 0:3] = torch.tensor([[2.2, 0, 0],
                                         [0, 1.2, 0],
                                         [0, 0, 2]])
        sigma1[3:6, 3:6] = torch.tensor([[1.5, 0.2, 0.4],
                                         [-0.1, 1.2, 0.4],
                                         [-0.2, -0.2, 0.8]])
        sigma1[6:8, 6:8] = torch.tensor([[0.4, 0.3],
                                         [0.3, 0.9]])
        sigma1[8:10, 8:10] = torch.eye(2)
        p1 = torch.distributions.MultivariateNormal(
            loc=torch.tensor([-1.5, -0.8, 1.3, 0.2, -0.1, 0, 0, 0, 0, 0]).to(self.device),
            covariance_matrix=sigma1)

        sigma2 = torch.zeros(self.dim, self.dim).to(self.device)
        sigma2[0:3, 0:3] = torch.tensor([[2.2, 0, 0],
                                         [0, 1, 0],
                                         [0, 0, 1.5]])
        sigma2[3:6, 3:6] = torch.tensor([[1.2, 0.4, -0.2],
                                         [-0.4, 1.2, -0.3],
                                         [0.2, -0.1, 1.2]])
        sigma2[6:8, 6:8] = torch.tensor([[0.8, 0],
                                         [0, 0.3]])
        sigma2[8:10, 8:10] = torch.eye(2)
        p2 = torch.distributions.MultivariateNormal(
            loc=torch.tensor([1.2, 1, -1.5, 0, 0, 0.1, 0, 0, 0, 0]).to(self.device),
            covariance_matrix=sigma2)

        self.probability = [p1, p2]
        self.weight = [0.55, 0.45]

        return

    def compute_p(self, x):
        if sum(self.weight) == 1:
            pi = [self.weight[i] * self.probability[i].log_prob(x).exp() for i in range(2)]
            return sum(pi)
        else:
            print("Check the weights of Gaussian mixture model!")

    def compute_mu(self, x):
        x.requires_grad = True
        logp = torch.log(self.compute_p(x))
        logp_x = torch.autograd.grad(logp.sum(), x, retain_graph=True, create_graph=True)[0]
        return logp_x

def srk_sde_solver(x0, T, N):
    bs, dim = x0.shape
    dt = T / N
    t = torch.linspace(0, T, N + 1)
    X = torch.zeros(N + 1, bs, dim)
    X[0] = x0
    GM = Gaussian_mixture()

    for i in range(1, N + 1):
        W = torch.randn(bs, dim)*dt**0.5
        mu_0 = GM.compute_mu(X[i - 1])
        sigma = 2**0.5

        X_mid = (X[i - 1] + 0.5 * mu_0 * (3 * dt - W ** 2) + sigma * W).detach()

        mu_1 = GM.compute_mu(X_mid)

        X[i] = (X[i - 1] + 0.5 * (mu_0 + mu_1) * dt + sigma * W).detach()

        print("t=%.3f," % (t[i].item()), end='  ')
        print("X_0.01: %.4f," % (torch.quantile(X[i], 0.01).item()), end='  ')
        print("X_0.99: %.4f" % (torch.quantile(X[i], 0.99).item()), end='\n')

    return t, X

def plot_data(xt, data_path):
    # xt:(N_t, N_x, d)
    data = xt.cpu().numpy()

    # Plot data (x1, x2)
    plt.figure(figsize=(3, 3))
    plt.scatter(data[:, 0], data[:, 1], s=6)
    plt.title('Gaussian mixture: $(x_1, x_2)$')

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.xticks(np.linspace(-6, 6, 5))
    plt.yticks(np.linspace(-6, 6, 5))
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)

    plt.savefig(data_path + 'Gaussian_mixture_1.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot data (x3, x4)
    plt.figure(figsize=(3, 3))
    plt.scatter(data[:, 0], data[:, 2], s=6)
    plt.title('Gaussian mixture: $(x_1, x_3)$')

    plt.xlabel("$x_1$")
    plt.ylabel("$x_3$")
    plt.xticks(np.linspace(-6, 6, 5))
    plt.yticks(np.linspace(-6, 6, 5))
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)

    plt.savefig(data_path + 'Gaussian_mixture_2.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":

    T = 5
    dim = 10
    N_in = 20000
    N_t = 500

    p_x0 = torch.distributions.MultivariateNormal(
            loc=torch.zeros(dim),
            covariance_matrix=torch.eye(dim))

    x0 = p_x0.sample([N_in])
    print("t=0 ", end=' ')
    print("max:%.4f" % (torch.max(x0).item()), end=' ')
    print("min:%.4f" % (torch.min(x0).item()), end='\n')

    t, xt = srk_sde_solver(x0, T, N_t)

    # xt:(N_t, N_x, dim)
    xT = xt[-1]
    nan_count = torch.sum(torch.isnan(xT))
    print("The number of ”nan“ element:", nan_count.item())

    path = './dim={}/'.format(dim)
    if not os.path.exists(path):
        os.makedirs(path)

    # Plot
    plot_data(xT, path)
    # Data
    np.save(path + 'data.npy', xT.cpu().numpy())