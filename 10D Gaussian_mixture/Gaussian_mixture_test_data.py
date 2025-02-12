import os
import torch
import autograd.numpy as np
import matplotlib.pyplot as plt
from Gaussian_mixture_training_data import Gaussian_mixture

np.random.seed(111)

def sample(self, num_samples):
    if sum(self.weight) == 1:
        xi = [self.probability[i].sample([int(self.weight[i] * num_samples)]) for i in range(2)]
        return torch.cat(xi, dim=0)
    else:
        print("Check the weights of Gaussian mixture model!")

dim = 10
path = './data/'
if not os.path.exists(path):
    os.makedirs(path)

Gaussian_mixture.sample = sample
GM = Gaussian_mixture()

# Data for mae, mape
x_error = GM.sample(10000)
np.save(path + 'x_error.npy', x_error)

# Plot x_error data
plt.figure(figsize=(3, 3))
plt.scatter(x_error[:, 0], x_error[:, 1], s=6)
plt.title('Gaussian mixture: $(x_1, x_2)$')

plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.xticks(np.linspace(-6, 6, 5))
plt.yticks(np.linspace(-6, 6, 5))
plt.xlim(-6, 6)
plt.ylim(-6, 6)

plt.show()