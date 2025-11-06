from simulation import Simulation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pdfsampler import PdfSampler
import constants as const

class Snapshot:
    def __init__(self, time, pa):
        self.time = time
        self.pa = pa

def get_bspline_area_function(order: int):
    match order:
        case 1:
            def f(x):
                x_ref = np.abs(x)
                conditions = [x_ref <= 0.5, x_ref < 1.5]
                values = [0.75-x_ref**2, 0.5*(1.5-x_ref)**2]
                return np.select(conditions, values, default=0)
        case _:
            def f(x):
                x_ref = np.abs(x)
                return np.where(x_ref < 1, 1 - x_ref, 0)
    return f

class PodSimulation(Simulation):
    def __init__(self,
                 particle_snapshots, node_snapshots,
                 node_positions,
                 dt, end_time,
                 x_domain, v_domain,
                 background_charge_density,
                 particle_order: int = 1,
                 particle_weight_factor=1.0,
                 snapshot_interval = 10,
                 color_rule = lambda x, v: None):
        self.n_particles = particle_snapshots.shape[0] // 2
        self.n_cells = node_positions.shape[0]
        self.x_domain = x_domain
        self.v_domain = v_domain
        self.L = x_domain[1] - x_domain[0]
        self.dt = dt
        self.end_time = end_time
        self.background_charge_density = background_charge_density
        # x, v = PdfSampler.sample(self.f0, [self.x_domain, self.v_domain], self.n_particles)
        # self.particle_array = np.zeros((self.n_particles*2, 1))
        # self.particle_array[:self.n_particles, 0] = x
        # self.particle_array[self.n_particles:, 0] = v
        self.particle_array = particle_snapshots[:, 0].reshape(-1, 1)
        self.colors = color_rule(self.particle_array[:self.n_particles], self.particle_array[self.n_particles:])
        self.area_function = get_bspline_area_function(particle_order)
        self.particle_weight_factor = particle_weight_factor
        self.dx = node_positions[1] - node_positions[0]
        self.snapshot_interval = snapshot_interval
        color_rule = color_rule
        self.time = 0
        self.snapshots = []

        # Snapshot data
        self.particle_snapshots = particle_snapshots
        self.node_snapshots = node_snapshots[:, 1:]
        self.node_positions = node_positions.reshape(-1, 1)
        
    def show_singular_values(self):
        particle_svs = np.linalg.svd(self.particle_snapshots, compute_uv=False)
        particle_mags = np.cumsum(particle_svs) / np.sum(particle_svs) * 100
        plt.figure()
        plt.scatter(range(len(particle_mags)), particle_mags)
        plt.grid()
        plt.title('Particle Singular Values')
        plt.xlabel('Singular Value')
        plt.ylabel('Cumulative Energy (%)')
        plt.show()

        node_svs = np.linalg.svd(self.node_snapshots, compute_uv=False)
        node_mags = np.cumsum(node_svs) / np.sum(node_svs) * 100
        plt.figure()
        plt.scatter(range(len(node_mags)), node_mags)
        plt.grid()
        plt.title('Node Singular Values')
        plt.xlabel('Singular Value')
        plt.ylabel('Cumulative Energy (%)')
        plt.show()

    def show_projection_error_function(self, start=1, max=np.inf, step=1):
        U = np.linalg.svd(self.particle_snapshots, compute_uv=True, full_matrices=False).U
        norm = np.linalg.norm(self.particle_snapshots)
        identity = np.identity(2*self.n_particles)
        stop = min(max, U.shape[1])
        errors = []
        modes = []
        for i in range(start, stop, step):
            print(f'{i}/{stop} ({i/stop:.2%})')
            psi = U[:, :i]
            error = (identity - psi @ psi.T) @ self.particle_snapshots
            errors.append(np.linalg.norm(error) / norm)
            modes.append(i)
        plt.plot(modes, errors)
        plt.yscale('log')
        plt.show()

    def update(self):
        x_diff = (self.difference_operator @ self.pa).reshape((self.n_particles, self.n_cells), order='F')
        A = self.area_function(x_diff)
        force = A @ self.e_field_force_extractor @ A.T.reshape((-1, 1), order='F')
        u = self.particle_pusher @ self.pa[:-1] + self.force_conversion @ force
        x = u[:self.n_particles]
        conditions = [x > self.x_domain[1], x < self.x_domain[0]]
        values = [-self.L, self.L]
        shift = np.select(condlist=conditions, choicelist=values, default=0)
        u[:self.n_particles] += shift
        self.pa[:-1] = self.psi_pt @ u
        self.time += self.dt
        
    def run(self, n_particle_modes=None, n_node_modes=None):
        ### Initialize
        # Particles
        if n_particle_modes == None:
            self.psi_p = np.identity(2*self.n_particles)
            self.psi_pt = np.identity(2*self.n_particles)
            n_particle_modes = 2*self.n_particles
        else:
            U_p = np.linalg.svd(self.particle_snapshots, compute_uv=True, full_matrices=False).U
            row_idx = np.random.choice(2*self.n_particles, size=n_particle_modes)
            self.psi_p = U_p[:, :n_particle_modes]
            self.psi_pt = self.psi_p.T
        # Nodes
        if n_node_modes == None:
            self.psi_n = np.identity(self.n_cells)
            self.psi_nt = np.identity(self.n_cells)
            n_node_modes = self.n_cells
        else:
            U_n = np.linalg.svd(self.node_snapshots, compute_uv=True, full_matrices=False).U
            self.psi_n = U_n[:, :n_node_modes]
            self.psi_nt = self.psi_n.T

        # Set up initial conditions for particle modes
        self.pa = np.ones((n_particle_modes+1, 1))
        self.pa[:-1] = self.psi_pt @ self.particle_array
        self.save_snapshot()
        # Note: The last value in this array is always one to allow affine operations to be performed as matrix products

        ### Define linear operators
        # Get new x-positions of particles for interpolation onto grid
        particle_pusher = np.block([[np.identity(self.n_particles), np.identity(self.n_particles)*self.dt],
                                    [np.zeros((self.n_particles, self.n_particles)), np.identity(self.n_particles)]])
        self.particle_pusher = particle_pusher @ self.psi_p
        node_position_matrix = np.ones((self.n_particles, 1)) @ self.node_positions.T
        node_ones = np.ones_like(self.node_positions)
        linear_operator1 = np.kron(node_ones, particle_pusher[:self.n_particles, :] @ self.psi_p)
        self.difference_operator = np.block([linear_operator1, -node_position_matrix.reshape((-1, 1), order='F')]) / self.dx
        # Get electric field at nodes
        phi_matrix = np.zeros((self.n_cells, self.n_cells))
        for i in range(self.n_cells):
            phi_matrix[i, i-1] = 1
            phi_matrix[i, i] = -2
            if i+1 < self.n_cells:
                phi_matrix[i, i+1] = 1
            else:
                phi_matrix[i, 0] = 1
        particle_ones = np.ones((1, self.n_particles))
        charge_ones = -const.q_electron*self.particle_weight_factor*self.dx/const.epsilon0 * particle_ones
        linear_operator2 = np.kron(charge_ones, np.linalg.pinv(phi_matrix))
        e_matrix = np.zeros((self.n_cells, self.n_cells))
        for i in range(self.n_cells):
            e_matrix[i, i-1] = 1
            if i+1 < self.n_cells:
                e_matrix[i, i+1] = -1
            else:
                e_matrix[i, 0] = -1
        self.e_field_force_extractor = const.q_electron * self.dt * e_matrix @ linear_operator2 / (2 * self.dx * const.m_electron)
        # Update velocities and positions
        self.force_conversion = np.block([[np.zeros((self.n_particles, self.n_particles))], [np.identity(self.n_particles)]])

        # Run
        super().run()

    def save_snapshot(self):
        self.snapshots.append(Snapshot(self.time, self.pa.copy()))

    def show_snapshots(self, fps=10, save_animation=False, filename='PIC_simulation', repeat=True, show_cells=False):
        print('Generating animation...')
        fig, ax = plt.subplots(figsize=(10, 10))
        colors = np.unique(self.colors)

        def show(s):
            ax.clear()
            u = self.psi_p @ s.pa[:-1]
            x = u[:self.n_particles]
            v = u[self.n_particles:]
            for color in colors:
                idx = np.where(self.colors == color)
                plt.scatter(x[idx], v[idx], alpha=0.5, color=color)
            plt.title(f'Time = {s.time: .3g}')
            plt.xlabel('$x$ (m)')
            plt.xlim(self.x_domain)
            plt.ylabel('$v_x$ (m/s)')
            plt.ylim(self.v_domain)

            if show_cells:
                for n in self.nodes:
                    plt.vlines(n.x-0.5*self.dx, *self.v_domain, linestyles='--', color='k')

        show(self.snapshots[0])
        ani = animation.FuncAnimation(fig=fig, func=show, frames=self.snapshots, interval=1e3/fps, repeat=repeat)
        if save_animation:
            print('Saving...')
            writer = animation.PillowWriter(fps=fps)
            ani.save(f'{filename}.gif', writer=writer)
        print('Displaying...')
        plt.show()
        print('Done!')