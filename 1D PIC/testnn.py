import numpy as np
import torch.nn as nn
import torch
from scipy.linalg import schur
import constants
import romtools
from nntraining import get_interpolation_matrix

class NN(nn.Module):
    def __init__(self, input_size, output_size: tuple, hidden_layers, activation=nn.SiLU(),
                 input_shift=0, input_scale=1, output_shift=0, output_scale=1, use_layer_norm=True):
        super().__init__()

        layers = []
        layers.append(nn.Linear(input_size, hidden_layers[0]))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_layers[0]))
        layers.append(activation)
        n = len(hidden_layers)
        for i in range(n-1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_layers[i+1]))
            layers.append(activation)
        layers.append(nn.Linear(hidden_layers[-1], output_size[0]*output_size[1]))
        self.net = nn.Sequential(*layers)

        self.output_size = output_size
        self.shift_x = input_shift
        self.scale_x = input_scale
        self.shift_y = output_shift
        self.scale_y = output_scale

    def forward(self, x):
        nx = (x - self.shift_x) / self.scale_x
        return self.net(nx).reshape(self.output_size[0], self.output_size[1]).detach().numpy()*self.scale_y + self.shift_y

if __name__ == "__main__":
    n_particle_modes = 400
    n_node_modes = 60
    n_cells = 60
    n_particles_per_cell = 100

    filename = f'two_stream_{n_cells}c{n_particles_per_cell}ppc_circle'
    particle_snapshots = np.loadtxt(f'{filename}_particles.csv', dtype=float, delimiter=',')[:, 1:]
    node_snapshots = np.loadtxt(f'{filename}_nodes.csv', dtype=float, delimiter=',')[:, 1:]
    node_positions = np.loadtxt(f'{filename}_node_positions.csv', dtype=float, delimiter=',')

    n0 = 1e15
    electron_plasma_frequency = np.sqrt(n0*constants.q_electron**2 / (constants.epsilon0 * constants.m_electron))
    inv_w_pe = 1/electron_plasma_frequency
    mean_v = 0.2*constants.c
    resonance = 2*np.pi*mean_v*inv_w_pe
    max_x = resonance * 1.25
    weight_factor = 2*max_x*n0 / (n_particles_per_cell*n_cells)
    n0 = 1e15
    bg_charge_density = -constants.q_electron * n0
    dt = 0.005*inv_w_pe*15
    end_time = dt*1e3
    times = np.arange(dt, end_time+0.1*dt, dt)

    n_particles = particle_snapshots.shape[0] // 3
    n_cells = node_positions.shape[0]
    dx = node_positions[1] - node_positions[0]
    L = n_cells*dx
    px = particle_snapshots[:n_particles]
    pv = particle_snapshots[n_particles:2*n_particles]
    pe = particle_snapshots[2*n_particles:]
    nq = node_snapshots[:n_cells]
    ne = node_snapshots[n_cells:]
    particle_ones = np.ones((pv.shape[0], 1))
    node_ones = np.ones((nq.shape[0], 1))

    psi_x = romtools.get_basis_for_all(n_particle_modes, px)
    psi_v = romtools.get_basis_for_all(n_particle_modes, pv, pe, particle_ones)
    psi_qe = romtools.get_basis_for_all(n_node_modes, nq-bg_charge_density, ne, node_ones)

    input_size = n_particle_modes
    output_size = (n_particle_modes, n_node_modes)
    hidden_layers = [20, 20]
    input_shift, input_scale, output_shift, output_scale = np.loadtxt(f'{filename}_interp_net_scaling_{n_particle_modes}pm{n_node_modes}nm', delimiter=',')
    model = NN(input_size, output_size, hidden_layers, nn.SiLU(), input_shift, input_scale, output_shift, output_scale, use_layer_norm=False).to(torch.double)
    model.load_state_dict(torch.load(f'{filename}_interp_net_{n_particle_modes}pm{n_node_modes}nm', weights_only=True))
    model.eval()

    rx = (psi_x.T @ px).T

    A = np.zeros((n_cells, n_cells))
    for i in range(n_cells):
        A[i, i-1] = 1
        A[i, i] = -2
        if i+1 < n_cells:
            A[i, i+1] = 1
        else:
            A[i, 0] = 1
    B = np.zeros((n_cells, n_cells))
    for i in range(n_cells-1):
        B[i, i+1] = 1
        B[i, i-1] = -1
    B[-1, 0] = 1
    B[-1, -2] = -1
    Q = 0.5 * B @ np.linalg.pinv(A) / constants.epsilon0
    rom_Q = psi_qe.T @ Q @ psi_qe

    bg_charge_density = -constants.q_electron * n0
    q_rom = psi_v.T @ particle_ones * constants.q_electron * weight_factor
    bg_vect = psi_qe.T @ node_ones*bg_charge_density*dx

    error = np.zeros(rx.shape[0])
    nn_error = np.zeros(rx.shape[0])
    with torch.no_grad():
        for i, t in enumerate(times):
            print(f'time: {t:.3g}s')
            interp = get_interpolation_matrix(psi_x, psi_v, psi_qe, n_particle_modes, n_node_modes, rx[i].reshape(1, -1), node_positions)[0]
            error[i] = romtools.rel_error(pe[:, i].reshape(-1, 1), psi_v @ interp @ rom_Q @ (interp.T @ q_rom + bg_vect))
            print(f'Interpolation error: {error[i]:.2%}')
            interp_nn = model(torch.from_numpy(rx[i]))
            nn_error[i] = romtools.rel_error(pe[:, i].reshape(-1, 1), psi_v @ interp_nn @ rom_Q @ (interp_nn.T @ q_rom + bg_vect))
            print(f'NN interpolation error: {nn_error[i]:.2%}')
            print(f'Relative error: {romtools.rel_error(interp, interp_nn):.2%}')

    print('\n-------------------Error Summary-------------------')
    print(f'Mean Error: {error.mean():.2%}')
    print(f'Min Error: {error.min():.2%}')
    print(f'Max Error: {error.max():.2%}')
    print(f'\nNN Mean Error: {nn_error.mean():.2%}')
    print(f'NN Min Error: {nn_error.min():.2%}')
    print(f'NN Max Error: {nn_error.max():.2%}')