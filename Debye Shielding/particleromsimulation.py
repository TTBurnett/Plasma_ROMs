import sys
sys.path.append('..')
from typing import Literal
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import constants as const
import romtools
import time
from scipy import sparse

class Snapshot:
    def __init__(self, time, px, pv, nq):
        self.time = time
        self.x = px
        self.v = pv
        self.nq = nq

def spline(x_ref, order: int = 1):
    match order:
        case 0:
            return np.where(x_ref < 1, 1 - x_ref, 0)
        case 1:
            conditions = [x_ref <= 0.5, x_ref < 1.5]
            values = [0.75-x_ref**2, 0.5*(1.5-x_ref)**2]
            return np.select(conditions, values, default=0)
        case _:
            raise ValueError(f'Invalid spline order {order}. [0, 1] are supported.')

class RomSimulation:
    def __init__(self,
                 particle_snapshots,
                 node_positions,
                 dt, end_time,
                 x_domain, v_domain,
                 background_charge_density,
                 particle_order = 1,
                 particle_weight_factor=1.0,
                 snapshot_interval = 10,
                 color_rule = None):
        self.node_positions = node_positions
        self.n_particles = particle_snapshots.shape[0] // 2
        self.px_snapshots = particle_snapshots[:self.n_particles]
        self.pv_snapshots = particle_snapshots[self.n_particles:]
        self.n_nodes = node_positions.shape[0]
        self.time = 0
        self.end_time = end_time
        self.dt = dt
        self.x_domain = x_domain
        self.L = x_domain[1] - x_domain[0]
        self.v_domain = v_domain
        self.dx = node_positions[1] - node_positions[0]
        x0 = self.px_snapshots[:, 0]
        v0 = self.pv_snapshots[:, 0]
        if color_rule != None:
            self.colors = color_rule(x0, v0)
        else:
            self.colors = np.array(['c' for x in x0])
        self.particle_order = particle_order
        self.weight_factor = particle_weight_factor
        self.background_charge_density = background_charge_density(self.node_positions)
        self.snapshot_interval = snapshot_interval
        self.snapshots = []

    def get_moments(self, px, pv):
        moments = np.zeros((3, self.node_positions.shape[0]))
        x_ref = self.get_distance(px, self.node_positions.reshape(-1, 1))/self.dx
        interpolation = spline(x_ref, self.particle_order)
        moments[0, :] = np.sum(interpolation, axis=1)*const.m_electron*self.weight_factor/self.dx
        moments[1, :] = (interpolation @ pv)*const.m_electron*self.weight_factor/self.dx
        moments[2, :] = (interpolation @ pv**2)*const.m_electron*self.weight_factor/self.dx
        return moments
    
    def get_interpolation_matrix(self, x):
        n_idx = np.repeat(((x - self.x_domain[0]) % self.L) // self.dx, 3).astype(int) + self.n_idx_tiling
        n_idx %= self.n_nodes
        interp_vals = spline(self.get_distance(x[self.p_idx], self.node_positions[n_idx])/self.dx)
        return sparse.csr_array((interp_vals, (n_idx, self.p_idx)), shape=(self.n_nodes, self.n_particles))

    def shift_x_to_domain(self, x):
        return (x - self.x_domain[0]) % self.L + self.x_domain[0]

    def get_distance(self, px, nx):
        px_domain = self.shift_x_to_domain(px)
        return 0.5*self.L - np.abs(np.abs(px_domain - nx) - 0.5*self.L)

    def push_particles(self):
        self.last_acceleration = const.q_over_m*self.pe_field
        self.px += self.v_to_x @ (self.pv*self.dt + 0.5*self.last_acceleration*self.dt**2)

    def interpolate_particles_to_field(self):
        self.interpolation = self.get_interpolation_matrix(self.psi_px @ self.px)
        self.nq = const.q_electron*self.weight_factor*self.interpolation.sum(axis=1) + self.background_charge_density*self.dx

    def update_electric_field(self):
        self.ne_field = self.electric_field_matrix @ self.nq

    def interpolate_field_to_particles(self):
        self.pe_field = self.psi_pv.T @ self.interpolation.T @ self.ne_field

    def accelerate_particles(self):
        self.pv += 0.5*(const.q_over_m*self.pe_field+self.last_acceleration)*self.dt

    def update(self):
        self.push_particles()
        self.interpolate_particles_to_field()
        self.update_electric_field()
        self.interpolate_field_to_particles()
        self.accelerate_particles()
        self.time += self.dt

    def run(self, n_particle_modes, pod_type: Literal['POD', 'PSD', 'Full'] = 'POD', save_snapshots=True):
        match(pod_type):
            case 'POD':
                self.psi_px = romtools.get_pod_basis(self.px_snapshots, n_modes=n_particle_modes)
                self.psi_pv = romtools.get_pod_basis(self.pv_snapshots, n_modes=n_particle_modes)
                self.v_to_x = self.psi_px.T @ self.psi_pv
            case 'PSD':
                self.psi_px = romtools.get_basis_for_all(n_particle_modes, self.px_snapshots, self.pv_snapshots)
                self.psi_pv = self.psi_px
                self.v_to_x = np.identity(self.psi_px.shape[1])
            case 'Full':
                self.psi_px = np.identity(self.n_particles)
                self.psi_pv = np.identity(self.n_particles)
                self.v_to_x = np.identity(self.psi_px.shape[1])
            case _:
                raise ValueError(f'Invalid type "{pod_type}" for modal decomposition.')
            
        # Set up vectors
        self.px = self.psi_px.T @ self.px_snapshots[:, 0]
        self.pv = self.psi_pv.T @ self.pv_snapshots[:, 0]
        self.n_idx_tiling = np.tile([-1, 0, 1], self.n_particles)
        self.p_idx = np.repeat(range(self.n_particles), 3)

        # Set up operators
        laplacian = np.zeros((self.n_nodes, self.n_nodes))
        for i in range(self.n_nodes):
            laplacian[i, i-1] = 1
            laplacian[i, i] = -2
            if i+1 < self.n_nodes:
                laplacian[i, i+1] = 1
            else:
                laplacian[i, 0] = 1
        self.inv_laplacian = np.linalg.pinv(laplacian)
        B = np.zeros((self.n_nodes, self.n_nodes))
        for i in range(self.n_nodes-1):
            B[i, i+1] = 1
            B[i, i-1] = -1
        B[-1, 0] = 1
        B[-1, -2] = -1
        self.electric_field_matrix = 0.5 * B @ self.inv_laplacian / const.epsilon0

        # Take initial condition snapshot
        self.interpolate_particles_to_field()
        self.update_electric_field()
        self.interpolate_field_to_particles()
        self.save_snapshot()

        start = time.perf_counter()
        while self.time < self.end_time:
            self.update()
            if round(self.time / self.dt) % self.snapshot_interval == 0:
                print(f't = {self.time:.3g}/{self.end_time:.3g} ({self.time/self.end_time:.2%})')
                if save_snapshots:
                    self.save_snapshot()
        print(f'Elapsed time: {time.perf_counter() - start:.4f} seconds')

    def save_snapshot(self):
        self.snapshots.append(Snapshot(self.time, self.shift_x_to_domain(self.psi_px @ self.px), self.psi_pv @ self.pv, self.nq))

    def show_snapshots(self, fps=10, save_animation=False, filename='PIC_simulation', repeat=True, show_moments=True, show_cells=False):
        print('Generating animation...')
        if show_moments:
            n_plots = 4
        else:
            n_plots = 1
        fig, ax = plt.subplots(n_plots, 1, figsize=(10, n_plots*5), sharex=True)
        colors = np.unique(self.colors)
        if show_moments:
            moment_array = np.array([self.get_moments(s.x, s.v) for s in self.snapshots])

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

    def show_singular_values(self, n_modes=0):
        U, svs, _ = np.linalg.svd(self.px_snapshots, compute_uv=True, full_matrices=False)

        sv_mags = np.cumsum(svs) / np.sum(svs) * 100
        plt.figure()
        plt.scatter(range(len(sv_mags)), sv_mags)
        plt.grid()
        plt.title('Position Singular Values')
        plt.xlabel('Singular Value')
        plt.ylabel('Cumulative Energy (%)')
        plt.ylim([0, 105])
        plt.show()

        if n_modes > 0:
            plt.figure()
            plt.plot(range(U.shape[0]), (U @ np.diag(svs))[:, :n_modes], '-o')
            plt.grid()
            plt.title('Position Mode Shapes')
            plt.xlabel('Index')
            plt.ylabel('Amplitude')
            plt.legend([f'Mode #{i}' for i in range(1, n_modes+1)])
            plt.show()
        
        U, svs, _ = np.linalg.svd(self.pv_snapshots, compute_uv=True, full_matrices=False)

        sv_mags = np.cumsum(svs) / np.sum(svs) * 100
        plt.figure()
        plt.scatter(range(len(sv_mags)), sv_mags)
        plt.grid()
        plt.title('Velocity Singular Values')
        plt.xlabel('Singular Value')
        plt.ylabel('Cumulative Energy (%)')
        plt.ylim([0, 105])
        plt.show()

        if n_modes > 0:
            plt.figure()
            plt.plot(range(U.shape[0]), (U @ np.diag(svs))[:, :n_modes], '-o')
            plt.grid()
            plt.title('Velocity Mode Shapes')
            plt.xlabel('Index')
            plt.ylabel('Amplitude')
            plt.legend([f'Mode #{i}' for i in range(1, n_modes+1)])
            plt.show()

    def show_projection_error_function(self, start=0, max=np.inf, step=1):
        x_errors = []
        v_errors = []
        modes = []
        U = romtools.get_basis_for_all(2*self.n_particles, self.px_snapshots, self.pv_snapshots)
        for i in range(start, min(max+1, U.shape[1]), step):
            print(i)
            psi = U[:, :i]
            x_error = (np.identity(self.n_particles) - psi @ psi.T) @ self.px_snapshots
            x_norm = np.linalg.norm(self.px_snapshots)
            v_error = (np.identity(self.n_particles) - psi @ psi.T) @ self.pv_snapshots
            v_norm = np.linalg.norm(self.pv_snapshots)
            x_errors.append(np.linalg.norm(x_error) / x_norm)
            v_errors.append(np.linalg.norm(v_error) / v_norm)
            modes.append(i+1)
        plt.plot(modes, x_errors, label='x')
        plt.plot(modes, v_errors, label='v')
        plt.legend()
        plt.ylabel('Relative Error')
        plt.xlabel('# of Modes')
        plt.title('Projection Error')
        plt.yscale('log')
        plt.show()

    def show_statistics(self):
        n = len(self.snapshots)
        kinetic_energy = np.zeros(n)
        electric_potential_energy = np.zeros(n)
        total_energy = np.zeros(n)
        t = np.zeros(n)
        for i, s in enumerate(self.snapshots):
            electric_potential = self.inv_laplacian @ -s.nq / const.epsilon0
            electric_potential_energy[i] = 0.5*np.sum(electric_potential*s.nq)*self.dx
            kinetic_energy[i] = 0.5*const.m_electron*self.weight_factor*np.sum(s.v**2)
            total_energy[i] = electric_potential_energy[i] + kinetic_energy[i]
            t[i] = s.time

        plt.plot(t, total_energy, label='Total Energy')
        plt.plot(t, electric_potential_energy, label='Electric Potential Energy', linestyle='--')
        plt.plot(t, kinetic_energy, label='Kinetic Energy', linestyle='--')
        plt.legend()
        plt.title('Energy vs. Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Energy (J)')
        plt.show()