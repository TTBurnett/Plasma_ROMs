from particle import Particle
from node import Node
from typing import Literal
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import copy
import constants as const

class Snapshot:
    def __init__(self, time, particles, rom_charge, rom_e_field):
        self.time = time
        self.particles = particles
        self.rom_charge = rom_charge
        self.rom_e_field = rom_e_field

def get_moments(particles: list[Particle], node_positions, dx):
    half_dx = 0.5*dx
    moments = np.zeros((3, node_positions.shape[0]))
    for i, x in enumerate(node_positions):
        moments[0, i] = np.sum(p.weight_factor for p in particles if abs(p.x-x) < half_dx)*const.m_electron/dx
        moments[1, i] = np.sum(p.weight_factor*p.v for p in particles if abs(p.x-x) < half_dx)*const.m_electron/dx
        moments[2, i] = np.sum(p.weight_factor*p.v**2 for p in particles if abs(p.x-x) < half_dx)*const.m_electron/dx
    return moments

def spline(x_ref, order: int):
    match order:
        case 0:
            return np.where(x_ref < 1, 1 - x_ref, 0)
        case 1:
            conditions = [x_ref <= 0.5, x_ref < 1.5]
            values = [0.75-x_ref**2, 0.5*(1.5-x_ref)**2]
            return np.select(conditions, values, default=0)
        case _:
            raise ValueError(f'Invalid spline order {order}. [0, 1] are supported.')

class PodSimulation:
    def __init__(self,
                 particle_snapshots, node_snapshots,
                 node_positions,
                 dt, end_time,
                 x_domain, v_domain,
                 background_charge_density,
                 particle_order = 1,
                 particle_weight_factor=1.0,
                 snapshot_interval = 10,
                 color_rule = lambda x, v: None):
        n_particles = particle_snapshots.shape[0] // 2
        self.n_cells = node_positions.shape[0]
        self.node_snapshots = node_snapshots
        self.time = 0
        self.end_time = end_time
        self.dt = dt
        self.x_domain = x_domain
        self.L = x_domain[1] - x_domain[0]
        self.v_domain = v_domain
        self.dx = node_positions[1] - node_positions[0]
        u0 = particle_snapshots[:, 0].reshape(-1, 1)
        x = u0[:n_particles, 0]
        v = u0[n_particles:, 0]
        self.particles = [Particle(pos, vel, spline_order=particle_order, weight_factor=particle_weight_factor, color=color_rule(pos, vel)) for pos, vel in zip(x, v)]
        self.particle_order = particle_order
        self.weight_factor = particle_weight_factor
        self.background_charge_density = background_charge_density
        self.snapshot_interval = snapshot_interval
        self.node_positions = node_positions
        self.snapshots = []

    def get_distance(self, x1, x2):
        return 0.5*self.L - np.abs(np.abs(x1 - x2) - 0.5*self.L)

    def do_electric_field_update(self):
        self.rom_e_field = self.electric_field_matrix @ self.rom_charge

    def do_field_gather(self):
        electric_field = self.rom_e_field
        for p in self.particles:
            x_ref = self.get_distance(self.node_positions, p.x)/self.dx
            area = spline(x_ref, self.particle_order)
            p.electric_field = np.sum(a*e for a, e in zip(area, electric_field[:, 0]))

    def do_particle_update(self):
        self.area_sum = np.zeros((self.n_cells, 1))
        for p in self.particles:
            p.v += const.q_over_m*p.electric_field*self.dt
            p.x = (p.x + p.v*self.dt + self.x_domain[0]) % self.L + self.x_domain[0]
            x_ref = self.get_distance(self.node_positions, p.x)/self.dx
            area = spline(x_ref, self.particle_order)
            self.area_sum[:, 0] += area

    def do_particle_scatter(self):
        self.rom_charge = self.psi_t @ (const.q_electron*self.weight_factor*self.area_sum + self.background_charge_density*self.dx)

    def update(self):
        self.do_electric_field_update()
        self.do_field_gather()
        self.do_particle_update()
        self.do_particle_scatter()
        self.time += self.dt

    def run(self, n_node_modes, type: Literal['POD', 'Full'] = 'POD', save_snapshots=True):
        match(type):
            case 'POD':
                U = np.linalg.svd(self.node_snapshots, compute_uv=True, full_matrices=False).U
                self.psi = U[:, :n_node_modes]
                self.psi_t = self.psi.T
            case 'Full':
                self.psi = np.identity(self.n_cells)
                self.psi_t = np.identity(self.n_cells)
            case _:
                raise ValueError('Invalid Type for Modal Decomposition.')

        # Set up operators
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
        electric_field_matrix = 0.5 * B @ np.linalg.pinv(A) / const.epsilon0
        self.electric_field_matrix = electric_field_matrix @ self.psi

        # Take initial condition snapshot
        self.rom_charge = self.psi_t @ np.zeros((self.n_cells , 1))
        self.do_electric_field_update()
        self.save_snapshot()

        while self.time < self.end_time:
            self.update()
            if round(self.time / self.dt) % self.snapshot_interval == 0:
                print(f't = {self.time:.3g}/{self.end_time:.3g} ({self.time/self.end_time:.2%})')
                if save_snapshots:
                    self.save_snapshot()

    def save_snapshot(self):
        self.snapshots.append(Snapshot(self.time, copy.deepcopy(self.particles), self.rom_charge.copy(), self.rom_e_field.copy()))

    def show_snapshots(self, fps=10, save_animation=False, filename='PIC_simulation', repeat=True, show_moments=True, show_cells=False):
        print('Generating animation...')
        if show_moments:
            n_plots = 4
        else:
            n_plots = 1
        fig, ax = plt.subplots(n_plots, 1, figsize=(10, n_plots*5), sharex=True)
        colors = np.unique([p.color for p in self.particles])
        if show_moments:
            moment_array = np.array([get_moments(s.particles, self.node_positions, self.dx) for s in self.snapshots])

        def show(tuple):
            t_idx, s = tuple
            if not show_moments:
                ax.clear()
                for color in colors:
                    x, v = np.array([[p.x, p.v] for p in s.particles if p.color == color]).T
                    ax.scatter(x, v, alpha=0.5, color=color)
                ax.set_title(f'Time = {s.time: .3g}')
                ax.set_ylabel('$v_x$ (m/s)')
                ax.set_ylim(self.v_domain)
                ax.set_xlim(self.x_domain)
                ax.set_xlabel('$x$ (m)')
            else:
                ax[0].clear()
                for color in colors:
                    x, v = np.array([[p.x, p.v] for p in s.particles if p.color == color]).T
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
                for a in ax:
                    a.vlines(self.node_positions-0.5*self.dx, *self.v_domain, linestyles='--', color='k')

        show((0, self.snapshots[0]))
        ani = animation.FuncAnimation(fig=fig, func=show, frames=enumerate(self.snapshots), interval=1e3/fps, repeat=repeat)
        if save_animation:
            print('Saving...')
            writer = animation.PillowWriter(fps=fps)
            ani.save(f'{filename}.gif', writer=writer)
        print('Displaying...')
        plt.show()
        print('Done!')

    def show_singular_values(self, n_modes=10):
        U, svs, _ = np.linalg.svd(self.node_snapshots, compute_uv=True, full_matrices=False)

        sv_mags = np.cumsum(svs) / np.sum(svs) * 100
        plt.figure()
        plt.scatter(range(len(sv_mags)), sv_mags)
        plt.grid()
        plt.title('Node Singular Values')
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
        for i in range(start, min(max+1, self.n_cells), step):
            print(i)

            U = np.linalg.svd(self.node_snapshots, compute_uv=True, full_matrices=False).U
            psi = U[:, :i]

            error = (np.identity(self.n_cells) - psi @ psi.T) @ self.node_snapshots
            norm = np.linalg.norm(self.node_snapshots)
            errors.append(np.linalg.norm(error) / norm)
            modes.append(i+1)
        plt.plot(modes, errors)
        plt.yscale('log')
        plt.ylabel('Relative Error')
        plt.xlabel('Number of Modes')
        plt.title('Projection Error')
        plt.show()

    def show_statistics(self):
        plt.figure(figsize=(10, 10))

        errors = []
        t = []
        for i, s in enumerate(self.snapshots):
            u = (self.node_snapshots[:, i].reshape(-1, 1) - self.background_charge_density)*self.dx
            u_rom = self.psi @ s.rom_charge - self.background_charge_density*self.dx
            norm = np.linalg.norm(u)
            if norm > 0:
                t.append(s.time)
                errors.append(np.linalg.norm(u_rom - u) / norm)

        plt.plot(t, errors)
        plt.title('Node ROM Error')
        plt.xlabel('Time (s)')
        plt.ylabel('Relative Error')
        plt.show()

    def save_snapshots_to_csv(self, filename):
        print(f'Saving to {filename}...')
        particle_list = []
        node_list = []
        for s in self.snapshots:
            particle_list.append(np.array([[p.x, p.electric_field] for p in s.particles]).flatten('F'))
            node_list.append(s.rom_e_field)
        particle_array = np.vstack(particle_list).T
        node_array = np.hstack(node_list)
        np.savetxt(f'{filename}_particles.csv', particle_array, delimiter=',')
        np.savetxt(f'{filename}_nodes.csv', node_array, delimiter=',')
        np.savetxt(f'{filename}_node_positions.csv', self.node_positions, delimiter=',')
        print('Done!')