import os
import torch
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(111)

def compute_mu(x):
    A = torch.tensor([[4., 1, 1, 0, 0, 0],
                      [1., 4, 1, 0, 0, 0],
                      [1, 1, 4, 0, 0, 0],
                      [0, 0, 0, 1, 0.1, 0.1],
                      [0, 0, 0, 0.1, 1, 0.1],
                      [0, 0, 0, 0.1, 0.1, 1]])
    mask = torch.tensor([[1., 1, 0, 0, 0, 0]])

    dH = torch.mm(x, A) - 2 * x / (x ** 2 + 0.02) * mask

    return - dH

def srk_sde_solver(x0, T, N):
    bs, dim = x0.shape
    dt = T / N
    t = torch.linspace(0, T, N + 1)
    X = torch.zeros(N + 1, bs, dim)
    X[0] = x0

    for i in range(1, N + 1):
        W = torch.randn(bs, dim)*dt**0.5
        mu_0 = compute_mu(X[i - 1])
        sigma = 2**0.5

        X_mid = X[i - 1] + 0.5 * mu_0 * (3 * dt - W ** 2) + sigma * W
        mu_1 = compute_mu(X_mid)

        X[i] = X[i - 1] + 0.5 * (mu_0 + mu_1) * dt + sigma * W

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
    plt.title('6D Multi-modal: $(x_1, x_2)$')

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.xticks(np.linspace(-2, 2, 5))
    plt.yticks(np.linspace(-2, 2, 5))
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)

    plt.savefig(data_path + '6D_Multi-modal_1.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot data (x1, x3)
    plt.figure(figsize=(3, 3))
    plt.scatter(data[:, 0], data[:, 2], s=6)
    plt.title('6D Multi-modal: $(x_1, x_3)$')

    plt.xlabel("$x_1$")
    plt.ylabel("$x_3$")
    plt.xticks(np.linspace(-2, 2, 5))
    plt.yticks(np.linspace(-2, 2, 5))
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)

    plt.savefig(data_path + '6D_Multi-modal_2.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":

    T = 1
    dim = 6
    N_in = 20000
    N_t = 200

    p_x0 = torch.distributions.MultivariateNormal(
            loc=torch.zeros(dim),
            covariance_matrix=0.1*torch.eye(dim))

    x0 = p_x0.sample([N_in])
    print("t=0 ", end=' ')
    print("max:%.4f" % (torch.max(x0).item()), end=' ')
    print("min:%.4f" % (torch.min(x0).item()), end='\n')

    t, xt = srk_sde_solver(x0, T, N_t)

    # xt:(N_t, N_x, dim)
    xT = xt[-1]
    nan_count = torch.sum(torch.isnan(xT))
    print("The number of 'nan' element:", nan_count.item())

    path = './data/'
    if not os.path.exists(path):
        os.makedirs(path)

    # Plot
    plot_data(xT, path)
    # Data
    np.save(path + 'data.npy', xT.cpu().numpy())