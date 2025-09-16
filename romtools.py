import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd
from typing import Literal

def proj_error(psi, data):
    return np.linalg.norm((np.identity(data.shape[0]) - psi @ psi.T) @ data) / np.linalg.norm(data)

def rel_error(y1, y2, axis=None):
    return np.linalg.norm(y1 - y2, axis=axis) / np.linalg.norm(y1, axis=axis)

def print_matrix(matrix):
    string = '['
    for i in range(matrix.shape[0]):
        string += '['
        for j in range(matrix.shape[1]):
            string += f'{matrix[i, j]:.3g} '
        string = string[:-1] + ']\n'
    string = string[:-1] + ']'
    print(string)

def print_where_not_zero(matrix):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] > 1e-14:
                print(f'{i}, {j}: {matrix[i, j]:.3g}')

def get_basis_for_all(n_modes, *args, show_singular_values=False, show_projection_errors=False):
    unit_args = [a / np.linalg.norm(a) for a in args]
    u, s, _ = np.linalg.svd(np.hstack(unit_args), full_matrices=False, compute_uv=True)
    psi = u[:, :n_modes]
    if show_singular_values:
        plt.scatter(range(1, s.shape[0]+1), np.cumsum(s)/np.sum(s) * 100)
        plt.ylabel('Cumulative Energy (%)')
        plt.xlabel('# of Modes')
        plt.title('Singular Value Decay')
        plt.show()
    if show_projection_errors:
        projection_errors = [proj_error(psi, a) for a in args]
        labels = [f'Arg {i}' for i in range(1, len(args)+1)]
        bars = plt.bar(labels, projection_errors)
        plt.bar_label(bars, fmt='%.3g', padding=2)
        plt.ylabel('Projection Error')
        plt.title('Normalized Projection Errors')
        plt.show()
    return psi

def get_pod_basis(data, n_modes=None, random=False, plot_svs=False, n_modes_to_plot=0, plot_projection_errors=False, error_step=5):
    if random:
        u, s, _ = randomized_svd(data, n_modes)
    else:
        u, s, _ = np.linalg.svd(data, full_matrices=False, compute_uv=True)

    if plot_svs:
        plt.scatter(range(1, 1+s.shape[0]), np.cumsum(s) / np.sum(s) * 100)
        plt.title('Singular Value Energies')
        plt.ylabel('Cumulative Energy (%)')
        plt.xlabel('# of Modes')
        plt.show()
        if n_modes_to_plot > 0:
            plt.plot((u @ np.diag(s))[:, :n_modes_to_plot])
            plt.show()
    if plot_projection_errors:
        r_idx = range(error_step, u.shape[1], error_step)
        errors = [proj_error(u[:, :i], data) for i in r_idx]
        plt.scatter(r_idx, errors)
        plt.title('Projection Errors')
        plt.ylabel('Normalized Error')
        plt.xlabel('# of Modes')
        plt.yscale('log')
        plt.show()
    
    return u[:, :n_modes]

def construct_measurments(max_idx, percent_hyperreduction, hyperreduction_algorithm: Literal['Gappy', 'DEIM'], u):
    n_hyperreduction_points = round(1e-2*percent_hyperreduction*max_idx)

    match hyperreduction_algorithm:
        case 'Gappy':
            measurement_idx = np.random.choice(np.arange(max_idx), size=n_hyperreduction_points, replace=False)
        case 'DEIM':
            n_basis = u.shape[1]
            measurement_idx = np.empty(n_hyperreduction_points, dtype=int)
            measurement_idx[0] = np.argmax(u[:, 0])
            for i in range(1, n_hyperreduction_points):
                print(f'DEIM: {i}/{n_hyperreduction_points} ({i/n_hyperreduction_points:.2%})')
                c = np.linalg.pinv(u[measurement_idx[:i], :i]) @ u[measurement_idx[:i], i]
                residual = u[:, i] - u[:, :i] @ c
                measurement_idx[i] = np.argmax(residual)
                if i == n_basis-1:
                    all_idx = np.arange(max_idx)
                    remaining_idx = np.delete(all_idx, measurement_idx[:n_basis])
                    measurement_idx[n_basis:] = np.random.choice(remaining_idx, size=n_hyperreduction_points-n_basis, replace=False)
                    break
        case _:
            raise ValueError(f'Invalid hyperreduction algorithm "{hyperreduction_algorithm}".')

    measurement_idx.sort()
    return measurement_idx