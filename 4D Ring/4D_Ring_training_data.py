import os
import torch
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(111)

def compute_mu(x):
    A = torch.tensor([[0, -1, 0, 0],
                      [1., 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]])

    mu = - 4 * x * (x.norm(dim=1, keepdim=True) ** 2 - 1).repeat(1, 4) + torch.mm(x, A)

    return mu

def srk_sde_solver(x0, T, N):
    bs, dim = x0.shape
    dt = T / N
    t = torch.linspace(0, T, N + 1)
    X = torch.zeros(N + 1, bs, dim)
    X[0] = x0

    for i in range(1, N + 1):
        W = torch.randn(bs, dim)*dt**0.5
        mu_0 = compute_mu(X[i - 1])
        sigma = 1

        X_mid = X[i - 1] + 0.5 * mu_0 * (3 * dt - W ** 2) + sigma * W
        mu_1 = compute_mu(X_mid)

        X[i] = X[i - 1] + 0.5 * (mu_0 + mu_1) * dt + sigma * W

        print("t=%.3f" % (t[i].item()), end=' ')
        print("max:%.4f" % (torch.max(X[i]).item()), end=' ')
        print("min:%.4f" % (torch.min(X[i]).item()), end='\n')

    return t, X

def plot_data(xt, data_path):
    # xt:(N_t, N_x, d)
    data = xt.cpu().numpy()

    # Plot data
    plt.figure(figsize=(3, 3))
    x3, x4, dx = 0, 0, 0.2
    points = data[(data[:, 2] > x3 - dx) & (data[:, 2] < x3 + dx) & (data[:, 3] > x4 - dx) & (data[:, 3] < x4 + dx)]
    plt.scatter(points[:, 0], points[:, 1], s=6)
    plt.title('4D Ring: $(x_1, x_2, 0, 0)$')

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.xticks(np.linspace(-2, 2, 5))
    plt.yticks(np.linspace(-2, 2, 5))
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)

    plt.savefig(data_path + '4D_Ring.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":

    T = 1
    dim = 4
    #N_in = 100000
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
    print("The number of 'nan' element:", nan_count.item())

    path = './data/'
    if not os.path.exists(path):
        os.makedirs(path)

    # Plot
    plot_data(xT, path)
    # Data
    np.save(path + 'data.npy', xT.cpu().numpy())