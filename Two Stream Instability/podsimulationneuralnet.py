from typing import Literal
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import constants as const
import torch
import torch.nn as nn
import romtools
from nntraining import get_interpolation_matrix
import time

class DistillingNN(nn.Module):
    def __init__(self, node_positions, psi_particles, psi_nodes):
        super().__init__()
        self.n_nodes = node_positions.shape[0]
        self.n_particles = psi_particles.shape[0]
        self.n_particle_modes = psi_particles.shape[1]
        self.n_node_modes = psi_nodes.shape[1]
        self.dx = node_positions[1] - node_positions[0]
        self.min_x = node_positions[0] - 0.5*self.dx
        self.L = self.dx*self.n_nodes

        nx = np.repeat(node_positions, self.n_particles)
        self.nx = torch.from_numpy(nx)
        psi_p = np.tile(psi_particles, (self.n_nodes, 1))
        self.psi_p = torch.from_numpy(psi_p)
        reduce_matrix = np.kron(psi_nodes, psi_particles).T
        self.reduce_matrix = torch.from_numpy(reduce_matrix)

    def get_distance(self, x_modulus):
        return 0.5*self.L - torch.abs(torch.abs(x_modulus - self.nx) - 0.5*self.L)
    
    def b_spline(self, x):
        y = torch.zeros_like(x)
        mask1 = x <= 0.5
        y[mask1] = 0.75-x[mask1]**2
        mask2 = (x > 0.5) & (x < 1.5)
        y[mask2] = 0.5*(1.5-x[mask2])**2
        return y

    def activation(self, x):
        x_modulus = torch.remainder(x - self.min_x, self.L) + self.min_x
        x_ref = self.get_distance(x_modulus) / self.dx
        return self.b_spline(x_ref)

    def forward(self, x):
        x_lifted = self.psi_p @ x
        h1 = self.activation(x_lifted)
        y = self.reduce_matrix @ h1
        return y.numpy().reshape(self.n_node_modes, self.n_particle_modes)
    
    def drop_hidden_neurons(self, fraction):
        n_keep = round((1-fraction)*self.reduce_matrix.shape[1])
        idx = torch.randint(0, self.reduce_matrix.shape[1], size=(n_keep,))
        self.psi_p = nn.Parameter(self.psi_p.data[idx, :])
        self.nx = nn.Parameter(self.nx.data[idx])
        self.reduce_matrix = nn.Parameter(self.reduce_matrix.data[:, idx])
        print(f'Reduced to {n_keep} hidden neurons')

# class NN(nn.Module):
#     def __init__(self, input_size, output_size: tuple, hidden_layers, activation=nn.ReLU(),
#                  input_shift=0, input_scale=1, output_shift=0, output_scale=1, use_layer_norm=True):
#         super().__init__()

#         layers = []
#         layers.append(nn.Linear(input_size, hidden_layers[0]))
#         if use_layer_norm:
#             layers.append(nn.LayerNorm(hidden_layers[0]))
#         layers.append(activation)
#         n = len(hidden_layers)
#         for i in range(n-1):
#             layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
#             if use_layer_norm:
#                 layers.append(nn.LayerNorm(hidden_layers[i+1]))
#             layers.append(activation)
#         layers.append(nn.Linear(hidden_layers[-1], output_size))
#         self.net = nn.Sequential(*layers)

#         self.shift_x = input_shift
#         self.scale_x = input_scale
#         self.shift_y = output_shift
#         self.scale_y = output_scale

#     def forward(self, x):
#         nx = (x - self.shift_x) / self.scale_x
#         return self.net(nx).detach().numpy()*self.scale_y + self.shift_y

class Snapshot:
    def __init__(self, time, x, v):
        self.time = time
        self.x = x
        self.v = v

def get_moments(px, pv, node_positions, weight_factor):
    dx = node_positions[1] - node_positions[0]
    half_dx = 0.5*dx
    moments = np.zeros((3, node_positions.shape[0]))
    for i, nx in enumerate(node_positions):
        x_diff = np.abs(px-nx)
        idx = np.argwhere(x_diff < half_dx)
        moments[0, i] = np.sum(x_diff < half_dx)*const.m_electron*weight_factor/dx
        moments[1, i] = np.sum(pv[idx])*const.m_electron*weight_factor/dx
        moments[2, i] = np.sum((pv**2)[idx])*const.m_electron*weight_factor/dx
    return moments

class PodSimulation:
    def __init__(self, model_file,
                 n_particle_modes, n_node_modes,
                 particle_snapshots,
                 node_snapshots,
                 node_positions,
                 dt, end_time,
                 x_domain, v_domain,
                 background_charge_density,
                 particle_weight_factor,
                 use_neural_net = True,
                 snapshot_interval = 10,
                 color_rule = lambda x, v: None):
        # Setup simulation
        self.model_file = model_file
        self.use_neural_net = use_neural_net
        self.n_particle_modes = n_particle_modes
        self.n_node_modes = n_node_modes
        self.n_particles = particle_snapshots.shape[0] // 3
        self.n_cells = node_positions.shape[0]
        self.node_positions = node_positions
        self.nq_snapshots = node_snapshots[:self.n_cells]
        self.ne_snapshots = node_snapshots[self.n_cells:]
        self.px_snapshots = particle_snapshots[:self.n_particles]
        self.pv_snapshots = particle_snapshots[self.n_particles:2*self.n_particles]
        self.pe_snapshots = particle_snapshots[2*self.n_particles:]
        self.time = 0
        self.end_time = end_time
        self.dt = dt
        self.x_domain = x_domain
        self.L = x_domain[1] - x_domain[0]
        self.v_domain = v_domain
        self.dx = node_positions[1] - node_positions[0]
        self.colors = color_rule(self.px_snapshots[:, 0], self.pv_snapshots[:, 0])
        self.weight_factor = particle_weight_factor
        self.background_charge_density = background_charge_density
        self.snapshot_interval = snapshot_interval
        self.setup_calculations()
        self.snapshots = []
        self.save_snapshot()

    def do_particle_update(self):
        self.px += self.pv*self.dt
        with torch.no_grad():
            if self.use_neural_net:
                interp = self.model(self.px)
                e_field_particles = interp.T @ self.rom_e_field_matrix @ (interp @ self.rom_particle_charge_vector + self.bg_charge_vector)
            else:
                interp = get_interpolation_matrix(self.psi_p, self.psi_p, self.psi_qe, self.n_particle_modes, self.n_node_modes, self.px.reshape(1, -1), self.node_positions)[0]
                e_field_particles = interp @ self.rom_e_field_matrix @ (interp.T @ self.rom_particle_charge_vector + self.bg_charge_vector)
        self.pv += const.q_over_m * self.dt * e_field_particles.ravel()

    def update(self):
        self.do_particle_update()
        self.time += self.dt

    def setup_calculations(self):
        # Set up psi matrices
        particle_ones = np.ones((self.n_particles, 1))
        node_ones = np.ones((self.n_cells, 1))
        self.psi_p = romtools.get_basis_for_all(self.n_particle_modes, self.px_snapshots, self.pv_snapshots, particle_ones)
        self.psi_qe = romtools.get_basis_for_all(self.n_node_modes, self.nq_snapshots-self.background_charge_density,
                                            self.ne_snapshots, node_ones)
        self.px = self.psi_p.T @ self.px_snapshots[:, 0]
        self.pv = self.psi_p.T @ self.pv_snapshots[:, 0]

        # Set up node matrix
        A = np.zeros((self.n_cells, self.n_cells))
        for i in range(self.n_cells):
            A[i, i-1] = 1
            A[i, i] = -2
            if i+1 < self.n_cells:
                A[i, i+1] = 1
            else:
                A[i, 0] = 1
        B = np.zeros((self.n_cells, self.n_cells))
        for i in range(self.n_cells-1):
            B[i, i+1] = 1
            B[i, i-1] = -1
        B[-1, 0] = 1
        B[-1, -2] = -1
        e_field_matrix = 0.5 * B @ np.linalg.pinv(A) / const.epsilon0
        self.rom_e_field_matrix = self.psi_qe.T @ e_field_matrix @ self.psi_qe

        # Set up vectors
        self.bg_charge_vector = self.psi_qe.T @ node_ones * self.background_charge_density * self.dx
        self.rom_particle_charge_vector = self.psi_p.T @ particle_ones * const.q_electron * self.weight_factor

        # Setup model
        if self.use_neural_net:
            self.model = DistillingNN(self.node_positions, self.psi_p, self.psi_qe).to(torch.double)
            for i in range(100):
                try:
                    self.model.load_state_dict(torch.load(self.model_file, weights_only=True))
                    self.model.numpy()
                    return
                except:
                    self.model.drop_hidden_neurons(0.05)

    def run(self, save_snapshots=True):
        start = time.perf_counter()
        while self.time < self.end_time:
            self.update()
            if round(self.time / self.dt) % self.snapshot_interval == 0:
                print(f't = {self.time:.3g}/{self.end_time:.3g} ({self.time/self.end_time:.2%})')
                if save_snapshots:
                    self.save_snapshot()
        print(f'Elapsed time: {time.perf_counter() - start:.4f} seconds')

    def save_snapshot(self):
        self.snapshots.append(Snapshot(self.time, (self.psi_p @ self.px - self.x_domain[0]) % self.L + self.x_domain[0], self.psi_p @ self.pv))

    def show_snapshots(self, fps=10, save_animation=False, filename='PIC_simulation', repeat=True, show_moments=True, show_cells=False):
        print('Generating animation...')
        if show_moments:
            n_plots = 4
        else:
            n_plots = 1
        fig, ax = plt.subplots(n_plots, 1, figsize=(10, n_plots*5), sharex=True)
        colors = np.unique(self.colors)
        if show_moments:
            moment_array = np.array([get_moments(s.x, s.v, self.node_positions, self.weight_factor) for s in self.snapshots])

        def show(ts):
            t_idx, s = ts

            if not show_moments:
                ax.clear()
                for color in colors:
                    x, v = np.array([[x, v] for c, x, v in zip(self.colors, s.x, s.v) if c == color]).T.reshape(2, -1)
                    ax.scatter(x, v, alpha=0.5, color=color)
                ax.set_title(f'Time = {s.time: .3g}')
                ax.set_ylabel('$v_x$ (m/s)')
                ax.set_ylim(self.v_domain)
                ax.set_xlim(self.x_domain)
                ax.set_xlabel('$x$ (m)')
            else:
                ax[0].clear()
                for color in colors:
                    x, v = np.array([[x, v] for c, x, v in zip(self.colors, s.x, s.v) if c == color]).T.reshape(2, -1)
                    ax[0].scatter(x, v, alpha=0.5, color=color)
                ax[0].set_title(f'Time = {s.time: .3g}')
                ax[0].set_ylabel('$v_x$ (m/s)')
                ax[0].set_ylim(self.v_domain)

                moments = moment_array[t_idx, :, :]
                x = self.node_positions

                for i in range(3):
                    ax[i+1].clear()
                    ax[i+1].plot(x, moments[i])
                    ax[i+1].set_ylabel(f'Moment {i}')
                    ax[i+1].set_ylim([moment_array[:, i, :].min(), moment_array[:, i, :].max()])

                ax[3].set_xlim(self.x_domain)
                ax[3].set_xlabel('$x$ (m)')

            if show_cells:
                for n in self.nodes:
                    for a in ax:
                        a.vlines(n.x-0.5*self.dx, *self.v_domain, linestyles='--', color='k')

        show((0, self.snapshots[0]))
        ani = animation.FuncAnimation(fig=fig, func=show, frames=enumerate(self.snapshots), interval=1e3/fps, repeat=repeat)
        if save_animation:
            print('Saving...')
            writer = animation.PillowWriter(fps=fps)
            ani.save(f'{filename}.gif', writer=writer)
        print('Displaying...')
        plt.show()
        print('Done!')

    def show_singular_values(self, type: Literal['POD Stacked', 'POD', 'PSD'] = 'POD', n_modes=10):
        match(type):
            case 'POD Stacked':
                U, particle_svs, _ = np.linalg.svd(np.vstack((self.px_snapshots, self.pv_snapshots)), compute_uv=True, full_matrices=False)
            case 'POD':
                Ux, sv_x, _ = np.linalg.svd(self.px_snapshots, compute_uv=True, full_matrices=False)
                Uv, sv_v, _ = np.linalg.svd(self.pv_snapshots, compute_uv=True, full_matrices=False)
                particle_svs = np.hstack((sv_x, sv_v))
                U = np.hstack((Ux, Uv))
            case 'PSD':
                u = np.hstack((self.px_snapshots / np.linalg.norm(self.px_snapshots), self.pv_snapshots / np.linalg.norm(self.pv_snapshots)))
                U, particle_svs, _ = np.linalg.svd(u, compute_uv=True, full_matrices=False)

        particle_mags = np.cumsum(particle_svs) / np.sum(particle_svs) * 100
        plt.figure()
        plt.scatter(range(len(particle_mags)), particle_mags)
        plt.grid()
        plt.title('Particle Singular Values')
        plt.xlabel('Singular Value')
        plt.ylabel('Cumulative Energy (%)')
        plt.ylim([0, 105])
        plt.show()

        plt.figure()
        for i in range(n_modes):
            plt.plot(range(U.shape[0]), U[:, i], '-o', label=f'Mode #{i+1}')
        plt.grid()
        plt.title('Mode Shapes')
        plt.xlabel('Index')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.show()

    def show_projection_error_function(self, start=0, max=np.inf, step=1, type: Literal['POD', 'PSD']='POD'):
        errors = []
        modes = []
        for i in range(start, min(max+1, self.n_particles*2), step):
            print(i)

            match(type):
                case 'POD':
                    Ux = np.linalg.svd(self.px_snapshots, compute_uv=True, full_matrices=False).U
                    psi_x = Ux[:, :i]
                    Uv = np.linalg.svd(self.pv_snapshots, compute_uv=True, full_matrices=False).U
                    psi_v = Uv[:, :i]

                    psi = np.block([[psi_x, np.zeros_like(psi_x)],
                                        [np.zeros_like(psi_v), psi_v]])
                    psi_t = np.block([[psi_x.T, np.zeros_like(psi_x.T)],
                                        [np.zeros_like(psi_v.T), psi_v.T]])
                    
                case 'PSD':
                    u = np.hstack((self.px_snapshots / np.linalg.norm(self.px_snapshots), self.pv_snapshots / np.linalg.norm(self.pv_snapshots)))
                    U = np.linalg.svd(u, compute_uv=True, full_matrices=False).U
                    psi_u = U[:, :i]
                    psi = np.block([[psi_u, np.zeros_like(psi_u)],
                                        [np.zeros_like(psi_u), psi_u]])
                    psi_t = np.block([[psi_u.T, np.zeros_like(psi_u.T)],
                                        [np.zeros_like(psi_u.T), psi_u.T]])

            u = np.vstack((self.px, self.pv))
            error = (np.identity(2*self.n_particles) - psi @ psi_t) @ u
            norm = np.linalg.norm(u)
            errors.append(np.linalg.norm(error) / norm)
            modes.append(i)
        plt.plot(modes, errors)
        plt.yscale('log')
        plt.show()

    def show_statistics(self, title=None, write_statistics=False):
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 10))

        if title != None:
            ax[0].set_title(title)

        errors = []
        hamiltonian = []
        t = []
        for i, s in enumerate(self.snapshots):
            t.append(s.time)
            u = np.vstack((self.px_snapshots[:, i].reshape(-1, 1), self.pv_snapshots[:, i].reshape(-1, 1)))
            u_rom = np.zeros_like(u)
            e_field = np.zeros(self.n_particles)
            for i, particle in enumerate(s.particles):
                u_rom[i, 0] = particle.x
                u_rom[i+self.n_particles, 0] = particle.v
                e_field[i] = particle.electric_field
            error = u_rom - u
            norm = np.linalg.norm(u)
            errors.append(np.linalg.norm(error) / norm)

            h = 0.5*np.sum(u_rom[self.n_particles:]**2+e_field*self.dx)
            hamiltonian.append(h)
        ax[0].plot(t, errors)
        ax[0].set_ylabel('Relative Error')
            
        ax[1].plot(t, hamiltonian)
        ax[1].set_ylabel('$(H-H_{ref})/H_{ref}$')
        ax[1].set_xlabel('Time (s)')

        # Save stats to text file
        np.set_printoptions(formatter={'float': lambda x: format(x, '.3g')})
        if write_statistics:
            with open('ROM_results.txt', mode='a') as f:
                f.write(f'-------------------------{title}-------------------------\n')
                f.write(f'Mean Relative Error: {np.mean(errors):.3g}\n')
                f.write(f'Relative Error Curve Fit Coefficients: {np.polyfit(t, errors, 1)}\n')
                f.write(f'Mean Hamiltonian: {np.mean(hamiltonian):.3g}\n')
                f.write(f'Hamiltionian Curve Fit Coefficients: {np.polyfit(t, hamiltonian, 1)}\n\n')
            plt.savefig(f'{title}.png')
        plt.show()