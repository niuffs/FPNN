import os
import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import elementwise_grad

np.random.seed(111)

def H(X):
    H = 2.5 * (X[:, 0] ** 2 + X[:, 1] ** 2 + X[:, 2] ** 2 + 0.1 * (X[:, 0] * X[:, 1] + X[:, 0] * X[:, 2] + X[:, 1] * X[:, 2])) + 2.0 * (X[:, 3] ** 2 + X[:, 4] ** 2 + X[:, 5] ** 2 + 0.2 * (X[:, 3] * X[:, 4] + X[:, 3] * X[:, 5] + X[:, 4] * X[:, 5])) + 3.0 * (X[:, 6] ** 2 + X[:, 7] ** 2 - 0.01 * (X[:, 6] * X[:, 7])) + 3.0 * (X[:, 8] ** 2 + X[:, 9] ** 2 - 0.01 * (X[:, 8] * X[:, 9])) - np.log(2 * X[:, 8] ** 2 + 0.02)
    return H

def p_true(x):
    p = np.exp(-H(x)) / 1.09396764651324
    return p

dim = 10
xL = -1.2
xR = 1.2
path = './data/'
if not os.path.exists(path):
    os.makedirs(path)

# Data for mae, mape
g = elementwise_grad(H)

# Initialize x
x = (xR - xL) * np.random.rand(10000, dim) + xL
lr = 1e-3
threshold = 1e-8
max_iterations = 1000

for it in range(max_iterations):
    g_x = g(x)
    x -= lr * g_x
    p = p_true(x)
    print("It: {}, min p: {:.2e}".format(it, min(p)))

    if np.all(p > threshold):
        print("Minimum p-value for data: {:.2e}".format(min(p)))
        np.save(path + 'x_error.npy', x)
        break

# Plot x_error data
plt.figure(figsize=(3, 3))
plt.scatter(x[:, 8], x[:, 9], s=6)
plt.title('10D Multi-modal: $(x_9, x_{10})$')

plt.xlabel("$x_9$")
plt.ylabel("$x_{10}$")
plt.xticks(np.linspace(-2, 2, 5))
plt.yticks(np.linspace(-2, 2, 5))
plt.xlim(-2, 2)
plt.ylim(-2, 2)

plt.show()