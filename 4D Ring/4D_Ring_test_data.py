import os
import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import elementwise_grad

np.random.seed(111)

def V(x):
    V = (np.linalg.norm(x, axis=1) ** 2 - 1) ** 2
    return V

def p_true(x):
    p = np.exp(-2 * V(x)) / 12.4222285101916
    return p

dim = 4
xL = -1.8
xR = 1.8
path = './data/'
if not os.path.exists(path):
    os.makedirs(path)

# Data for plain pde loss
x_pde_loss = (xR - xL) * np.random.rand(10000, dim) + xL
np.save(path + 'x_pde_loss.npy', x_pde_loss)

# Data for mae, mape
g = elementwise_grad(V)

# Initialize x
x = (xR - xL) * np.random.rand(10000, dim) + xL
lr = 1e-3
threshold = 1e-3
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
plt.scatter(x[:, 0], x[:, 1], s=6)
plt.title('4D Ring: $(x_1, x_2)$')

plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.xticks(np.linspace(-2, 2, 5))
plt.yticks(np.linspace(-2, 2, 5))
plt.xlim(-2, 2)
plt.ylim(-2, 2)

plt.show()