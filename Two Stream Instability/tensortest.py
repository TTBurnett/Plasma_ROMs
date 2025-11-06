import numpy as np
import matplotlib.pyplot as plt
from opt_einsum import contract
import sparse

def g(x):
    n = x.shape[0]
    coords = []
    for i in range(n):
        coords.append([i, i, i, i])
    H = sparse.COO(np.array(coords).T, np.ones(n), shape=(n, n, n, n))
    return contract('ijkl,j,k,l', H, x, x, x).reshape(-1, 1)

def dg(x):
    n = x.shape[0]
    H = np.zeros((n, n ,n, n))
    for i in range(n):
        H[i, i, i, i] = 1
    return np.einsum('ijk,jlk', H, (np.tensordot(x.squeeze(), np.identity(n), axes=0) + np.tensordot(np.identity(n), x.squeeze(), axes=0)))

H = np.zeros((3, 3, 3))
for i in range(3):
    H[i, i, i] = 1

x = np.array([1, 2, 3], dtype=float).T
num = g(x)
print(num)

den = dg(x)
print(den)

for i in range(10):
    x -= np.linalg.inv(dg(x)) @ g(x)
    print(x)