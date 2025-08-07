from particle import Particle
from node import Node
from pdfsampler import PdfSampler
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import copy
import constants as const

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

class Simulation:
    def __init__(self,
                 n_cells, n_particles_per_cell,
                 dt, end_time,
                 f0, x_domain, v_domain,
                 background_charge_density,
                 particle_order = 1,
                 particle_weight_factor=1.0,
                 snapshot_interval = 10,
                 color_rule = lambda x, v: None):
        self.n_cells = n_cells
        self.n_particles_per_cell = n_particles_per_cell
        self.time = 0
        self.end_time = end_time
        self.dt = dt
        self.x_domain = x_domain
        self.L = x_domain[1] - x_domain[0]
        self.v_domain = v_domain
        self.dx = (x_domain[1] - x_domain[0]) / n_cells
        x, v = PdfSampler.sample(f0, [x_domain, v_domain], n_particles_per_cell*n_cells)
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
        self.do_particle_motion()
        self.do_charge_assignment()
        self.do_electrostatic_potential()
        self.do_electric_field_evolution()
        self.do_force_assignment()
        self.do_particle_acceleration()
        self.time += self.dt

    def run(self, save_snapshots=True):
        while self.time < self.end_time:
            self.update()
            if round(self.time / self.dt) % self.snapshot_interval == 0:
                print(f't = {self.time:.3g}/{self.end_time:.3g} ({self.time/self.end_time:.2%})')
                if save_snapshots:
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
            particle_list.append(np.array([[p.x_true, p.v, p.electric_field] for p in s.particles]).flatten('F'))
            node_list.append(np.array([[n.charge_density, n.electric_field] for n in s.nodes]).flatten('F'))
        particle_array = np.vstack(particle_list).T
        node_array = np.vstack(node_list).T
        node_positions = np.fromiter((n.x for n in self.nodes), dtype=float)
        np.savetxt(f'{filename}_particles.csv', particle_array, delimiter=',')
        np.savetxt(f'{filename}_nodes.csv', node_array, delimiter=',')
        np.savetxt(f'{filename}_node_positions.csv', node_positions, delimiter=',')
        print('Done!')