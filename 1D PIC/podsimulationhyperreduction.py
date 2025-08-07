from typing import Literal
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import constants as const

class Snapshot:
    def __init__(self, time, px, pv, rom_charge, ne_field):
        self.time = time
        self.x = px
        self.v = pv
        self.rom_charge = rom_charge
        self.ne_field = ne_field

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
        
def construct_measurments(max_idx, percent_hyperreduction, hyperreduction_algorithm: Literal['Gappy', 'DEIM'], psi):
    n_hyperreduction_points = round(1e-2*percent_hyperreduction*max_idx)

    match hyperreduction_algorithm:
        case 'Gappy':
            measurement_idx = np.random.choice(range(max_idx), size=n_hyperreduction_points, replace=False)
        case 'DEIM':
            measurement_idx = np.empty(n_hyperreduction_points, dtype=int)
            measurement_idx[0] = np.argmax(psi[:, 0])
            for i in range(1, n_hyperreduction_points):
                print(f'DEIM: {i}/{n_hyperreduction_points} ({i/n_hyperreduction_points:.2%})')
                c = np.linalg.pinv(psi[measurement_idx[:i], :i]) @ psi[measurement_idx[:i], i]
                residual = psi[:, i] - psi[:, :i] @ c
                measurement_idx[i] = np.argmax(residual)
        case _:
            raise ValueError(f'Invalid hyperreduction algorithm "{hyperreduction_algorithm}".')

    measurement_idx.sort()
    return measurement_idx

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
        self.n_particles = particle_snapshots.shape[0] // 3
        self.particle_e_field_snapshots = particle_snapshots[2*self.n_particles:, 1:]
        self.n_nodes = node_positions.shape[0]
        self.node_charge_snapshots = node_snapshots[:self.n_nodes, 1:]
        self.time = 0
        self.end_time = end_time
        self.dt = dt
        self.x_domain = x_domain
        self.L = x_domain[1] - x_domain[0]
        self.v_domain = v_domain
        self.dx = node_positions[1] - node_positions[0]
        u0 = particle_snapshots[:, 0].reshape(-1, 1)
        x = u0[:self.n_particles, 0]
        v = u0[self.n_particles:2*self.n_particles, 0]
        self.px = x.copy()
        self.pv = v.copy()
        self.colors = color_rule(x, v)
        self.particle_order = particle_order
        self.weight_factor = particle_weight_factor
        self.background_charge_density = background_charge_density
        self.snapshot_interval = snapshot_interval
        self.node_positions = node_positions
        self.snapshots = []

    def get_distance(self, x1, x2):
        return 0.5*self.L - np.abs(np.abs(x1 - x2) - 0.5*self.L)

    def do_electric_field_update(self):
        self.ne_field = self.electric_field_matrix @ self.rom_charge

    def do_field_gather(self):
        x_ref = self.get_distance(self.px[self.particle_measurement_idx].reshape(-1, 1), self.node_positions)/self.dx
        reduced_interpolation = spline(x_ref, self.particle_order)
        reduced_pe_field = reduced_interpolation @ self.ne_field
        self.pe_field = (self.unhyperreduce_pe_field @ reduced_pe_field).ravel()

    def do_particle_update(self):
        self.pv += const.q_over_m*self.pe_field*self.dt
        self.px = (self.px + self.pv*self.dt + self.x_domain[0]) % self.L + self.x_domain[0]

    def do_particle_scatter(self):
        x_ref = self.get_distance(self.px, self.node_positions[self.node_measurement_idx].reshape(-1, 1))/self.dx
        reduced_interpolation = spline(x_ref, self.particle_order)
        reduced_node_charge = const.q_electron*self.weight_factor*np.sum(reduced_interpolation, axis=1) + self.background_charge_density*self.dx
        self.rom_charge = self.unhyperreduce_node_charge @ reduced_node_charge

    def update(self):
        self.do_electric_field_update()
        self.do_field_gather()
        self.do_particle_update()
        self.do_particle_scatter()
        self.time += self.dt

    def run(self, n_node_modes, percent_hyperreduction_points: float=100, percent_hyperreduction_modes: float=100,
            hyperreduction_algorithm: Literal['Gappy', 'DEIM'] = 'Gappy', type: Literal['POD', 'Full'] = 'POD', save_snapshots=True):
        match(type):
            case 'POD':
                self.psi_nq = np.linalg.svd(self.node_charge_snapshots, compute_uv=True, full_matrices=False).U[:, :n_node_modes]
                self.psi_qelectrons = np.linalg.svd(self.node_charge_snapshots-self.background_charge_density, compute_uv=True, full_matrices=False).U[:, :n_node_modes]

                n_particle_hyperreduction_modes = round(1e-2*percent_hyperreduction_modes*self.n_particles)
                psi_pe = np.linalg.svd(self.particle_e_field_snapshots, compute_uv=True, full_matrices=False).U[:, :n_particle_hyperreduction_modes]
                if n_particle_hyperreduction_modes > psi_pe.shape[1]:
                    psi_pe = np.identity(self.n_particles)

                self.particle_measurement_idx = construct_measurments(self.n_particles, percent_hyperreduction_points, hyperreduction_algorithm, psi_pe)
                self.node_measurement_idx = construct_measurments(self.n_nodes, percent_hyperreduction_points, hyperreduction_algorithm, self.psi_qelectrons)
                self.unhyperreduce_pe_field = psi_pe @ np.linalg.pinv(psi_pe[self.particle_measurement_idx])
                self.unhyperreduce_node_charge = self.psi_nq.T @ self.psi_qelectrons @ np.linalg.pinv(self.psi_qelectrons[self.node_measurement_idx])

            case 'Full':
                self.psi_nq = np.identity(self.n_nodes)
                self.unhyperreduce_node_charge = np.identity(self.n_nodes)
                self.unhyperreduce_pe_field = np.identity(self.n_particles)
            case _:
                raise ValueError(f'Invalid type "{type}" for modal decomposition.')

        # Set up operators
        A = np.zeros((self.n_nodes, self.n_nodes))
        for i in range(self.n_nodes):
            A[i, i-1] = 1
            A[i, i] = -2
            if i+1 < self.n_nodes:
                A[i, i+1] = 1
            else:
                A[i, 0] = 1
        B = np.zeros((self.n_nodes, self.n_nodes))
        for i in range(self.n_nodes-1):
            B[i, i+1] = 1
            B[i, i-1] = -1
        B[-1, 0] = 1
        B[-1, -2] = -1
        electric_field_matrix = 0.5 * B @ np.linalg.pinv(A) / const.epsilon0
        self.electric_field_matrix = electric_field_matrix @ self.psi_nq

        # Take initial condition snapshot
        self.do_particle_scatter()
        self.do_electric_field_update()
        self.save_snapshot()

        while self.time < self.end_time:
            self.update()
            if round(self.time / self.dt) % self.snapshot_interval == 0:
                print(f't = {self.time:.3g}/{self.end_time:.3g} ({self.time/self.end_time:.2%})')
                if save_snapshots:
                    self.save_snapshot()

    def save_snapshot(self):
        self.snapshots.append(Snapshot(self.time, self.px.copy(), self.pv.copy(), self.rom_charge.copy(), self.ne_field.copy()))

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

    def show_singular_values(self, n_modes=10):
        U, svs, _ = np.linalg.svd(self.node_charge_snapshots, compute_uv=True, full_matrices=False)

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

    def show_singular_values(self, n_mode_shapes_to_show=10):
        U, svs, _ = np.linalg.svd(self.node_charge_snapshots, compute_uv=True, full_matrices=False)
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
        U = np.linalg.svd(self.node_charge_snapshots, compute_uv=True, full_matrices=False).U
        for i in range(start, min(max+1, self.n_nodes), step):
            print(i)
            psi = U[:, :i]
            error = (np.identity(self.n_nodes) - psi @ psi.T) @ self.node_charge_snapshots
            norm = np.linalg.norm(self.node_charge_snapshots)
            errors.append(np.linalg.norm(error) / norm)
            modes.append(i+1)
        plt.plot(modes, errors)
        plt.yscale('log')
        plt.show()

    def show_statistics(self):
        errors = []
        t = []
        for i, s in enumerate(self.snapshots):
            u = (self.node_charge_snapshots[:, i].reshape(-1, 1) - self.background_charge_density)*self.dx
            u_rom = self.psi_nq @ s.rom_charge - self.background_charge_density*self.dx
            norm = np.linalg.norm(u)
            if norm > 0:
                t.append(s.time)
                errors.append(np.linalg.norm(u_rom - u) / norm)

        plt.plot(t, errors)
        plt.title('Rom Error')
        plt.xlabel('Time (s)')
        plt.ylabel('Charge Relative Error')
        plt.show()