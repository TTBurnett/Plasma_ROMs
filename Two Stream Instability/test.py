import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(formatter={'all': lambda x: f'{x:.1f}'})
n_nodes = 5
laplacian = np.zeros((n_nodes, n_nodes))
for i in range(n_nodes):
    laplacian[i, i-1] = 1
    laplacian[i, i] = -2
    if i+1 < n_nodes:
        laplacian[i, i+1] = 1
    else:
        laplacian[i, 0] = 1
inv_laplacian = np.linalg.pinv(laplacian)
B = np.zeros((n_nodes, n_nodes))
for i in range(n_nodes-1):
    B[i, i+1] = 1
    B[i, i-1] = -1
B[-1, 0] = 1
B[-1, -2] = -1

print(0.5*B @ inv_laplacian)
print(np.linalg.pinv(0.5*B))