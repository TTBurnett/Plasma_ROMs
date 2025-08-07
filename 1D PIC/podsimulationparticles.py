from particle import Particle
from node import Node
from typing import Literal
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import copy
import constants as const
import romtools

class Snapshot:
    def __init__(self, time, particles, nodes):
        self.time = time
        self.particles = particles
        self.nodes = nodes

def get_moments(particles: list[Particle], nodes: list[Node], dx):
    half_dx = 0.5*dx
    moments = np.zeros((3, len(nodes)))
    for i, n in enumerate(nodes):
        moments[0, i] = np.sum(p.weight_factor for p in particles if abs(p.x-n.x) < half_dx)*const.m_electron/dx
        moments[1, i] = np.sum(p.weight_factor*p.v for p in particles if abs(p.x-n.x) < half_dx)*const.m_electron/dx
        moments[2, i] = np.sum(p.weight_factor*p.v**2 for p in particles if abs(p.x-n.x) < half_dx)*const.m_electron/dx
    return moments

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
        self.n_particles = particle_snapshots.shape[0] // 2
        self.n_cells = node_positions.shape[0]
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
        self.particles = [Particle(pos, vel, x_domain, spline_order=particle_order, weight_factor=particle_weight_factor, color=color_rule(pos, vel)) for pos, vel in zip(x, v)]
        self.nodes = [Node(x) for x in np.arange(x_domain[0]+0.5*self.dx, x_domain[1], self.dx)]
        self.background_charge_density = background_charge_density
        self.snapshot_interval = snapshot_interval
        self.snapshots = []
        self.save_snapshot()

    def do_particle_motion(self):
        for p in self.particles:
            p.x_true += p.v * self.dt
            if p.x > self.x_domain[1]:
                p.x -= self.L
            if p.x < self.x_domain[0]:
                p.x += self.L
            for n in self.nodes:
                A = p.get_area_fraction(n.x, self.dx)
                if A > 0:
                    n.particles.append((p, A))
                elif n == self.nodes[0]:
                    A = p.get_area_fraction(n.x+self.L, self.dx)
                    if A > 0:
                        n.particles.append((p, A))
                elif n == self.nodes[-1]:
                    A = p.get_area_fraction(n.x-self.L, self.dx)
                    if A > 0:
                        n.particles.append((p, A))

    def do_charge_assignment(self):
        for n in self.nodes:
            n.charge_density = const.q_electron/self.dx * sum(p.weight_factor*A for p, A in n.particles)
            n.charge_density += self.background_charge_density

    def do_electrostatic_potential(self):
        A = np.zeros((self.n_cells, self.n_cells))
        for i in range(self.n_cells):
            A[i, i-1] = 1
            A[i, i] = -2
            if i+1 < self.n_cells:
                A[i, i+1] = 1
            else:
                A[i, 0] = 1
                
        b = np.fromiter((-self.dx**2*n.charge_density/const.epsilon0 for n in self.nodes), dtype=float)
        phi = np.linalg.pinv(A) @ b
        for i, n in enumerate(self.nodes):
            n.electrostatic_potential = phi[i]

    def do_electric_field_evolution(self):
        for i in range(self.n_cells-1):
            self.nodes[i].electric_field = -0.5*(self.nodes[i+1].electrostatic_potential - self.nodes[i-1].electrostatic_potential)/self.dx
        self.nodes[-1].electric_field = -0.5*(self.nodes[0].electrostatic_potential - self.nodes[-2].electrostatic_potential)/self.dx

    def do_force_assignment(self):
        for p in self.particles:
            p.electric_field = 0
        for n in self.nodes:
            for p, A in n.particles:
                p.electric_field += n.electric_field*A
            n.particles.clear()

    def do_particle_acceleration(self):
        for p in self.particles:
            p.v += const.q_electron/const.m_electron * p.electric_field*self.dt

    def update(self):
        u = self.psi @ self.pa
        for i, p in enumerate(self.particles):
            p.x_true = u[i, 0]
            p.v = u[i+self.n_particles, 0]
        self.do_particle_motion()
        self.do_charge_assignment()
        self.do_electrostatic_potential()
        self.do_electric_field_evolution()
        self.do_force_assignment()
        self.do_particle_acceleration()
        for i, p in enumerate(self.particles):
            u[i, 0] = p.x_true
            u[i+self.n_particles, 0] = p.v
        self.pa = self.psi_t @ u
        self.time += self.dt

    def run(self, n_particle_modes, type: Literal['POD Stacked', 'POD', 'PSD', 'Full'] = 'POD'):
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
                psi = romtools.get_basis_for_all(n_particle_modes, x, v, show_singular_values=True)
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
        self.pa = self.psi_t @ self.u0

        while self.time < self.end_time:
            self.update()
            if round(self.time / self.dt) % self.snapshot_interval == 0:
                print(f't = {self.time:.3g}/{self.end_time:.3g} ({self.time/self.end_time:.2%})')
                self.save_snapshot()

    def save_snapshot(self):
        self.snapshots.append(Snapshot(self.time, copy.deepcopy(self.particles), copy.deepcopy(self.nodes)))

    def show_snapshots(self, fps=10, save_animation=False, filename='PIC_simulation', repeat=True, show_moments=True, show_cells=False):
        print('Generating animation...')
        if show_moments:
            n_plots = 4
        else:
            n_plots = 1
        fig, ax = plt.subplots(n_plots, 1, figsize=(10, n_plots*5), sharex=True)
        colors = np.unique([p.color for p in self.particles])
        if show_moments:
            moment_array = np.array([get_moments(s.particles, s.nodes, self.dx) for s in self.snapshots])

        def show(s):
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

                moments = get_moments(s.particles, s.nodes, self.dx)
                x = np.array([n.x for n in s.nodes])

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

        show(self.snapshots[0])
        ani = animation.FuncAnimation(fig=fig, func=show, frames=self.snapshots, interval=1e3/fps, repeat=repeat)
        if save_animation:
            print('Saving...')
            writer = animation.PillowWriter(fps=fps)
            ani.save(f'{filename}.gif', writer=writer)
        print('Displaying...')
        plt.show()
        print('Done!')

    def save_snapshots_to_csv(self, filename):
        print(f'Saving to {filename}...')
        particle_list = []
        node_list = []
        for s in self.snapshots:
            particle_list.append(np.array([[p.x, p.v] for p in s.particles]).flatten('F'))
            node_list.append(np.array([n.electric_field for n in s.nodes]))
        particle_array = np.vstack(particle_list).T
        node_array = np.vstack(node_list).T
        node_positions = np.fromiter((n.x for n in self.nodes), dtype=float)
        np.savetxt(f'{filename}_particles.csv', particle_array, delimiter=',')
        np.savetxt(f'{filename}_nodes.csv', node_array, delimiter=',')
        np.savetxt(f'{filename}_node_positions.csv', node_positions, delimiter=',')
        print('Done!')

    def show_singular_values(self, type: Literal['POD Stacked', 'POD', 'PSD'] = 'POD', n_modes=10):
        match(type):
            case 'POD Stacked':
                U, particle_svs, _ = np.linalg.svd(self.particle_snapshots, compute_uv=True, full_matrices=False)
            case 'POD':
                Ux, sv_x, _ = np.linalg.svd(self.particle_snapshots[:self.n_particles, :], compute_uv=True, full_matrices=False)
                Uv, sv_v, _ = np.linalg.svd(self.particle_snapshots[self.n_particles:, :], compute_uv=True, full_matrices=False)
                particle_svs = np.hstack((sv_x, sv_v))
                U = np.hstack((Ux, Uv))
            case 'PSD':
                x = self.particle_snapshots[:self.n_particles, :]
                v = self.particle_snapshots[self.n_particles:, :]
                u = np.hstack((x / np.linalg.norm(x), v / np.linalg.norm(v)))
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
        x_errors = []
        v_errors = []
        modes = []
        match(type):
            case 'POD':
                Ux = np.linalg.svd(self.particle_snapshots[:self.n_particles, :], compute_uv=True, full_matrices=False).U
                Uv = np.linalg.svd(self.particle_snapshots[self.n_particles:, :], compute_uv=True, full_matrices=False).U
            case 'PSD':
                x = self.particle_snapshots[:self.n_particles, :]
                v = self.particle_snapshots[self.n_particles:, :]
                U = romtools.get_basis_for_all(self.n_particles, x, v)

        for i in range(start, min(max+1, self.n_particles*2, self.particle_snapshots.shape[1]), step):
            print(i)

            match(type):
                case 'POD':
                    psi_x = Ux[:, :i]
                    psi_v = Uv[:, :i]
                    psi = np.block([[psi_x, np.zeros_like(psi_x)],
                                        [np.zeros_like(psi_v), psi_v]])
                    psi_t = np.block([[psi_x.T, np.zeros_like(psi_x.T)],
                                        [np.zeros_like(psi_v.T), psi_v.T]])
                case 'PSD':
                    psi_u = U[:, :i]
                    psi = np.block([[psi_u, np.zeros_like(psi_u)],
                                        [np.zeros_like(psi_u), psi_u]])
                    psi_t = np.block([[psi_u.T, np.zeros_like(psi_u.T)],
                                        [np.zeros_like(psi_u.T), psi_u.T]])

            error = (np.identity(2*self.n_particles) - psi @ psi_t) @ self.particle_snapshots
            x_error = error[:self.n_particles]
            v_error = error[self.n_particles:]
            x_norm = np.linalg.norm(self.particle_snapshots[:self.n_particles])
            v_norm = np.linalg.norm(self.particle_snapshots[self.n_particles:])
            x_errors.append(np.linalg.norm(x_error) / x_norm)
            v_errors.append(np.linalg.norm(v_error) / v_norm)
            modes.append(i)
        plt.plot(modes, x_errors, label='x')
        plt.plot(modes, v_errors, label='v')
        plt.legend()
        plt.ylabel('Relative Error')
        plt.xlabel('# of Modes')
        plt.title('Projection Errors')
        plt.yscale('log')
        plt.grid(True)
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
            u = self.particle_snapshots[:, i].reshape(-1, 1)
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