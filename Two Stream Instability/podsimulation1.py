from pdfsampler import PdfSampler
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import constants as const
from typing import Literal
from numba import njit

class Snapshot:
    def __init__(self, time, pa):
        self.time = time
        self.pa = pa

@njit
def get_area_fraction(order, x_ref):
    match order:
        case 0:
            if x_ref < 1:
                return 1 - x_ref
            return 0.0
        case 1:
            if x_ref <= 0.5:
                return 0.75-x_ref**2
            if x_ref <= 1.5:
                return 0.5*(1.5-x_ref)**2
            return 0.0
        case _:
            raise ValueError(f'Invalid spline order {order}. [0, 1] are supported.')
            
class PodSimulation():
    def __init__(self,
                 particle_snapshots, node_snapshots,
                 node_positions,
                 dt, end_time,
                 f0, x_domain, v_domain,
                 background_charge_density,
                 particle_order: int = 1,
                 particle_weight_factor=1.0,
                 snapshot_interval = 10,
                 color_rule = lambda x, v: None):
        self.n_particles = particle_snapshots.shape[0] // 2
        self.n_cells = node_positions.shape[0]
        self.f0 = f0
        self.x_domain = x_domain
        self.v_domain = v_domain
        self.L = x_domain[1] - x_domain[0]
        self.dt = dt
        self.end_time = end_time
        self.background_charge_density = background_charge_density
        self.particle_array = particle_snapshots[:, 0].reshape(-1, 1)
        self.x = self.particle_array[:self.n_particles]
        self.v = self.particle_array[self.n_particles:]
        self.colors = color_rule(self.particle_array[:self.n_particles], self.particle_array[self.n_particles:])
        self.particle_order = particle_order
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
        self.node_particles = []
        for n in self.node_positions.ravel():
            self.node_particles.append([])
        
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

    def show_projection_error_function(self, start=0, max=np.inf, step=1, type: Literal['POD', 'PSD']='POD'):
        errors = []
        modes = []
        for i in range(start, min(max+1, self.n_particles*2), step):
            print(i)

            match(type):
                case 'POD':
                    Ux = np.linalg.svd(self.particle_snapshots[:self.n_particles, :], compute_uv=True, full_matrices=False).U
                    psi_x = Ux[:, :i]
                    Uv = np.linalg.svd(self.particle_snapshots[self.n_particles:, :], compute_uv=True, full_matrices=False).U
                    psi_v = Uv[:, :i]

                    psi = np.block([[psi_x, np.zeros_like(psi_x)],
                                        [np.zeros_like(psi_v), psi_v]])
                    psi_t = np.block([[psi_x.T, np.zeros_like(psi_x.T)],
                                        [np.zeros_like(psi_v.T), psi_v.T]])
                    
                case 'PSD':
                    x = self.particle_snapshots[:self.n_particles, :]
                    v = self.particle_snapshots[self.n_particles:, :]
                    u = np.hstack((x, v))
                    U = np.linalg.svd(u, compute_uv=True, full_matrices=False).U
                    psi_u = U[:, :i]
                    psi = np.block([[psi_u, np.zeros_like(psi_u)],
                                        [np.zeros_like(psi_u), psi_u]])
                    psi_t = np.block([[psi_u.T, np.zeros_like(psi_u.T)],
                                        [np.zeros_like(psi_u.T), psi_u.T]])

            error = (np.identity(2*self.n_particles) - psi @ psi_t) @ self.particle_snapshots
            norm = np.linalg.norm(self.particle_snapshots)
            errors.append(np.linalg.norm(error) / norm)
            modes.append(i)
        plt.plot(modes, errors)
        plt.yscale('log')
        plt.show()

    def do_particle_motion(self):
        # Calculate area contributions
        for i in range(self.n_particles):
            self.x[i, 0] += self.v[i, 0] * self.dt
            if self.x[i, 0] > self.x_domain[1]:
                self.x[i, 0] -= self.L
            elif self.x[i, 0] < self.x_domain[0]:
                self.x[i, 0] += self.L
            for nx, particles in zip(self.node_positions.ravel(), self.node_particles):
                x_ref = abs(self.x[i, 0] - nx) / self.dx
                A = get_area_fraction(self.particle_order, x_ref)
                if A > 0:
                    particles.append((i, A))
                elif i == 0:
                    A = get_area_fraction(self.particle_order, x_ref)
                    if A > 0:
                        particles.append((i, A))
                elif i == self.n_cells - 1:
                    A = get_area_fraction(self.particle_order, x_ref)
                    if A > 0:
                        particles.append((i, A))

    def do_particle_acceleration(self):
        charge_density = np.zeros_like(self.node_positions)
        for i, particles in enumerate(self.node_particles):
            charge_density[i, 0] = const.q_electron / self.dx * self.particle_weight_factor * sum(A for p, A in particles)
            charge_density[i, 0] += self.background_charge_density
        b = -self.dx**2*charge_density/const.epsilon0
        phi = np.linalg.pinv(self.phi_matrix) @ b

        #node_e_field = self.e_field_extractor @ charge_density

        electric_field = np.zeros_like(self.node_positions)
        for i in range(self.n_cells-1):
            electric_field[i, 0] = -0.5*(phi[i+1] - phi[i-1])/self.dx
        electric_field[-1, 0] = -0.5*(phi[0] - phi[i-1])/self.dx

        # Update velocities
        for e_field, particles in zip(electric_field.ravel(), self.node_particles):
            for i, A in particles:
                self.v[i, 0] += e_field*A*const.q_electron*self.dt/const.m_electron
            particles.clear()

    def update(self):
        u = self.psi @ self.pa
        self.x = u[:self.n_particles]
        self.v = u[self.n_particles:]
        self.do_particle_motion()
        self.do_particle_acceleration()
        self.pa = self.psi_t @ u
        self.time += self.dt

    def run(self, n_particle_modes, type: Literal['POD Stacked', 'POD', 'PSD', 'Full'] = 'POD'):
        ### Initialize
        # Particles
        match(type):
            case 'POD Stacked':
                U = np.linalg.svd(self.particle_snapshots, compute_uv=True, full_matrices=False).U
                self.psi = U[:, :n_particle_modes]
                self.psi_t = self.psi.T
            case 'POD':
                Ux = np.linalg.svd(self.particle_snapshots[:self.n_particles, :], compute_uv=True, full_matrices=False).U
                psi_x = Ux[:, :n_particle_modes]
                Uv = np.linalg.svd(self.particle_snapshots[self.n_particles:, :], compute_uv=True, full_matrices=False).U
                psi_v = Uv[:, :n_particle_modes]
                self.psi = np.block([[psi_x, np.zeros_like(psi_x)],
                                     [np.zeros_like(psi_v), psi_v]])
                self.psi_t = np.block([[psi_x.T, np.zeros_like(psi_x.T)],
                                       [np.zeros_like(psi_v.T), psi_v.T]])
            case 'PSD':
                x = self.particle_snapshots[:self.n_particles, :]
                v = self.particle_snapshots[self.n_particles:, :]
                u = np.hstack((x, v))
                U = np.linalg.svd(u, compute_uv=True, full_matrices=False).U
                psi = U[:, :n_particle_modes]
                self.psi = np.block([[psi, np.zeros_like(psi)],
                                     [np.zeros_like(psi), psi]])
                self.psi_t = np.block([[psi.T, np.zeros_like(psi.T)],
                                       [np.zeros_like(psi.T), psi.T]])
            case 'Full':
                self.psi = np.identity(2*self.n_particles)
                self.psi_t = np.identity(2*self.n_particles)
            case _:
                raise ValueError('Invalid Type for Modal Decomposition.')

        # Set up initial conditions for particle modes
        self.pa = self.psi_t @ self.particle_array
        self.save_snapshot()

        # Setup operators
        self.phi_matrix = np.zeros((self.n_cells, self.n_cells))
        for i in range(self.n_cells):
            self.phi_matrix[i, i-1] = 1
            self.phi_matrix[i, i] = -2
            if i+1 < self.n_cells:
                self.phi_matrix[i, i+1] = 1
            else:
                self.phi_matrix[i, 0] = 1
        inv_phi = np.linalg.pinv(self.phi_matrix)
        e_matrix = np.zeros((self.n_cells, self.n_cells))
        for i in range(self.n_cells):
            e_matrix[i, i-1] = 1
            if i+1 < self.n_cells:
                e_matrix[i, i+1] = -1
            else:
                e_matrix[i, 0] = -1
        self.e_field_extractor = -self.dx * e_matrix @ inv_phi / (2 * const.epsilon0)

        while self.time < self.end_time:
            self.update()
            if round(self.time / self.dt) % self.snapshot_interval == 0:
                print(f't = {self.time:.3g}/{self.end_time:.3g} ({self.time/self.end_time:.2%})')
                self.save_snapshot()

    def save_snapshot(self):
        self.snapshots.append(Snapshot(self.time, self.pa.copy()))

    def show_snapshots(self, fps=10, save_animation=False, filename='PIC_simulation', repeat=True, show_cells=False):
        print('Generating animation...')
        fig, ax = plt.subplots(figsize=(10, 10))
        colors = np.unique(self.colors)

        def show(s):
            ax.clear()
            u = self.psi @ s.pa
            x = u[:self.n_particles]
            v = u[self.n_particles:]
            for color in colors:
                idx = np.where(self.colors == color)[0]
                plt.scatter(x[idx], v[idx], alpha=0.5, color=color)
            plt.title(f'Time = {s.time: .3g}')
            plt.xlabel('$x$ (m)')
            plt.xlim(self.x_domain)
            plt.ylabel('$v_x$ (m/s)')
            plt.ylim(self.v_domain)

            if show_cells:
                for nx in self.node_positions.ravel():
                    plt.vlines(nx-0.5*self.dx, *self.v_domain, linestyles='--', color='k')

        show(self.snapshots[0])
        ani = animation.FuncAnimation(fig=fig, func=show, frames=self.snapshots, interval=1e3/fps, repeat=repeat)
        if save_animation:
            print('Saving...')
            writer = animation.PillowWriter(fps=fps)
            ani.save(f'{filename}.gif', writer=writer)
        print('Displaying...')
        plt.show()
        print('Done!')

    def show_statistics(self, t, h_ref, title=None, write_statistics=False):
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 10))

        if title != None:
            ax[0].set_title(title)

        errors = []
        for i, s in enumerate(self.snapshots):
            u = self.true_trajectories[:, i].reshape(-1, 1).astype(np.float64)
            error = (self.psi @ s.pa - u).astype(np.float64)
            norm = np.linalg.norm(u)
            errors.append(np.linalg.norm(error) / norm)
        ax[0].plot(t[:len(errors)], errors)
        ax[0].set_ylabel('Relative Error')
        if np.max(error) > 1:
            ax[0].set_ylim([0, 1])

        hamiltonian = []
        for s in self.snapshots:
            u = (self.psi @ s.pa).astype(np.float64)
            h = 0.5*np.sum(u[self.n_particles:]**2+self.beta**2*u[:self.n_particles]**2)
            hamiltonian.append((h - h_ref) / h_ref)
        ax[1].plot(t[:len(hamiltonian)], hamiltonian)
        ax[1].set_ylabel('$(H-H_{ref})/H_{ref}$')
        ax[1].set_xlabel('Time (s)')
        if np.max(hamiltonian) > 1:
            ax[1].set_ylim([-0.6, 1])

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