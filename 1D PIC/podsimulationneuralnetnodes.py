from particle import Particle
from node import Node
from typing import Literal
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import constants as const
import torch
import torch.nn as nn
import nntools

class NN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, input_mean=0, input_std=1, output_mean=0, output_std=1,
                 activation=nn.ReLU(), use_batch_norm=False):
        super().__init__()

        layers = []
        layers.append(nn.Linear(input_size, hidden_layers[0]))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_layers[0]))
        layers.append(activation)
        n = len(hidden_layers)
        for i in range(n-1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_layers[i+1]))
            layers.append(activation)
        layers.append(nn.Linear(hidden_layers[-1], output_size))
        self.net = nn.Sequential(*layers)

        self.input_mean = input_mean
        self.input_std = input_std
        self.output_mean = output_mean
        self.output_std = output_std

    def forward(self, x):
        nx = (x - self.input_mean) / self.input_std
        return nntools.unnormalize(self.net(nx), self.output_mean, self.output_std).squeeze()

class Snapshot:
    def __init__(self, time, x, v, romq):
        self.time = time
        self.x = x
        self.v = v
        self.romq = romq

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
    def __init__(self,
                 e_field_model_file, romq_model_file,
                 e_field_scaling_file, romq_scaling_file,
                 hidden_layers,
                 n_node_modes,
                 node_snapshots,
                 particle_snapshots,
                 node_positions,
                 dt, end_time,
                 x_domain, v_domain,
                 background_charge_density,
                 particle_weight_factor,
                 snapshot_interval = 10,
                 color_rule = lambda x, v: None):
        # Setup simulation
        self.n_node_modes = n_node_modes
        self.node_snapshots = node_snapshots
        self.n_particles = particle_snapshots.shape[0] // 2
        self.n_nodes = node_positions.shape[0]
        self.node_positions = node_positions
        self.particle_snapshots = particle_snapshots
        self.time = 0
        self.end_time = end_time
        self.dt = dt
        self.x_domain = x_domain
        self.L = x_domain[1] - x_domain[0]
        self.v_domain = v_domain
        self.dx = node_positions[1] - node_positions[0]
        self.u0 = particle_snapshots[:, 0].reshape(-1, 1)
        x = self.u0[:self.n_particles, 0]
        v = self.u0[self.n_particles:, 0]
        self.px = x.copy()
        self.pv = v.copy()
        self.colors = color_rule(x, v)
        self.weight_factor = particle_weight_factor
        self.background_charge_density = background_charge_density
        self.snapshot_interval = snapshot_interval
        self.setup_calculations()
        self.snapshots = []
        self.save_snapshot()

        # Setup electric field model
        input_size = n_node_modes
        output_size = self.n_particles
        input_mean, input_std, output_mean, output_std = np.loadtxt(e_field_scaling_file, delimiter=',')
        self.e_field_nn = NN(input_size, output_size, hidden_layers, input_mean, input_std, output_mean, output_std,
                             activation=nn.SiLU(), use_batch_norm=True).to(torch.double)
        self.e_field_nn.load_state_dict(torch.load(e_field_model_file, weights_only=True))
        self.e_field_nn.eval()

        # Setup charge deposition model
        input_size = self.n_particles
        output_size = n_node_modes
        input_mean, input_std, output_mean, output_std = np.loadtxt(romq_scaling_file, delimiter=',')
        self.romq_nn = NN(input_size, output_size, hidden_layers, input_mean, input_std, output_mean, output_std,
                          activation=nn.SiLU(), use_batch_norm=True).to(torch.double)
        self.romq_nn.load_state_dict(torch.load(romq_model_file, weights_only=True))
        self.romq_nn.eval()

    def do_particle_acceleration(self):
        with torch.no_grad():
            e_field_particles = self.e_field_nn(self.romq.reshape(1, -1)).numpy()
            self.pv += const.q_over_m * self.dt * e_field_particles

    def do_particle_motion(self):
        self.px = (self.px + self.pv*self.dt + self.x_domain[0]) % self.L + self.x_domain[0]

    def do_charge_update(self):
        with torch.no_grad():
            self.romq = self.romq_nn(torch.from_numpy(self.px).reshape(1, -1))

    def update(self):
        self.do_particle_acceleration()
        self.do_particle_motion()
        self.do_charge_update()
        self.time += self.dt

    def setup_calculations(self):
        # Set up psi matrices
        self.psi_q = np.linalg.svd(self.node_snapshots, compute_uv=True, full_matrices=False).U[:, :self.n_node_modes]
        self.psi_qt = torch.from_numpy(self.psi_q.T)

        self.romq = (self. psi_qt @ torch.from_numpy(self.node_snapshots[:, 0].reshape(-1, 1))).ravel()

    def run(self, save_snapshots=True):
        while self.time < self.end_time:
            self.update()
            if round(self.time / self.dt) % self.snapshot_interval == 0:
                print(f't = {self.time:.3g}/{self.end_time:.3g} ({self.time/self.end_time:.2%})')
                if save_snapshots:
                    self.save_snapshot()

    def save_snapshot(self):
        self.snapshots.append(Snapshot(self.time, self.px.copy(), self.pv.copy(), self.romq.numpy().copy()))

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

    def show_singular_values(self, n_mode_shapes_to_show=10):
        U, svs, _ = np.linalg.svd(self.node_snapshots, compute_uv=True, full_matrices=False)
        energies = np.cumsum(svs) / np.sum(svs) * 100

        plt.figure()
        plt.scatter(range(1, len(energies)+1), energies)
        plt.grid()
        plt.title('Singular Values')
        plt.xlabel('# of Modes')
        plt.ylabel('Cumulative Energy (%)')
        plt.ylim([0, 105])
        plt.show()

        plt.figure()
        for i in range(n_mode_shapes_to_show):
            plt.plot(range(U.shape[0]), U[:, i], '-o', label=f'Mode #{i+1}')
        plt.grid()
        plt.title('Mode Shapes')
        plt.xlabel('Index')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.show()

    def show_projection_error_function(self, start=0, max=np.inf, step=1):
        errors = []
        modes = []
        U = np.linalg.svd(self.node_snapshots, compute_uv=True, full_matrices=False).U
        for i in range(start, min(max+1, self.n_nodes), step):
            print(i)
            psi = U[:, :i]
            error = (np.identity(self.n_nodes) - psi @ psi.T) @ self.node_snapshots
            norm = np.linalg.norm(self.node_snapshots)
            errors.append(np.linalg.norm(error) / norm)
            modes.append(i+1)
        plt.plot(modes, errors)
        plt.yscale('log')
        plt.show()

    def show_statistics(self):
        errors = []
        t = []
        for i, s in enumerate(self.snapshots):
            t.append(s.time)
            u = self.node_snapshots[:, i].reshape(-1, 1)
            u_rom = self.psi_q @ s.romq.reshape(-1, 1)
            errors.append(np.linalg.norm(u_rom - u) / np.linalg.norm(u))

        plt.plot(t, errors)
        plt.title('Rom Error')
        plt.xlabel('Time (s)')
        plt.ylabel('Charge Relative Error')
        plt.show()