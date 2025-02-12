import os
import torch
import autograd.numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(111)

dim = 20
xL = -2
xR = 2
a = 3
sigma = 1.5
path = './data/'
if not os.path.exists(path):
    os.makedirs(path)

Gaussian = torch.distributions.MultivariateNormal(
        loc=torch.zeros(dim),
        covariance_matrix=0.5*sigma*sigma/a * torch.eye(dim))

# Data for mae, mape
x_error = Gaussian.sample([10000])
np.save(path + 'x_error.npy', x_error)

print("max:%.4f" % (torch.max(x_error).item()), end=' ')
print("min:%.4f" % (torch.min(x_error).item()), end='\n')

# Plot x_error data
plt.figure(figsize=(3, 3))
plt.scatter(x_error[:, 0], x_error[:, 1], s=6)
plt.title('Gaussian mixture: $(x_1, x_2)$')

plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.xticks(np.linspace(-4, 4, 5))
plt.yticks(np.linspace(-4, 4, 5))
plt.xlim(-4, 4)
plt.ylim(-4, 4)

plt.show()