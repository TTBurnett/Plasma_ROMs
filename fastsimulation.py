import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import constants as const
from pdfsampler import PdfSampler
import time
from scipy import sparse
from scipy.special import erf
import scipy.fft as fft

class Snapshot:
    def __init__(self, time, px, pv, phi, e_field):
        self.time = time
        self.x = px
        self.v = pv
        self.phi = phi
        self.e_field = e_field

class Simulation:
    def __init__(self,
                 n_nodes, n_particles_per_cell,
                 dt, end_time,
                 f0, x_domain, v_domain,
                 background_charge_density,
                 particle_weight_factor,
                 particle_shape_function,
                 snapshot_interval = 10,
                 color_rule = None):
        self.n_nodes = n_nodes
        self.n_particles = n_particles_per_cell*n_nodes
        self.time = 0
        self.end_time = end_time
        self.dt = dt
        self.x_domain = x_domain
        self.L = x_domain[1] - x_domain[0]
        self.v_domain = v_domain
        self.dx = (x_domain[1] - x_domain[0]) / n_nodes
        self.px, self.pv = PdfSampler.sample(f0, [x_domain, v_domain], self.n_particles)
        if color_rule != None:
            self.colors = color_rule(self.px, self.pv)
        else:
            self.colors = np.array(['c' for x in self.px])
        self.weight_factor = particle_weight_factor
        self.shape_function = particle_shape_function
        self.node_positions = np.arange(x_domain[0]+0.5*self.dx, x_domain[1], self.dx)
        self.bg_charge_density = background_charge_density(self.node_positions)
        self.snapshot_interval = snapshot_interval
        self.snapshots = []

    def get_moments(self, px, pv):
        px = self.shift_x_to_domain(px)
        moments = np.zeros((3, self.node_positions.shape[0]))
        interpolation = self.get_interpolation_matrix(px)
        moments[0, :] = np.sum(interpolation, axis=1).ravel()*const.m_electron*self.weight_factor/self.dx
        moments[1, :] = (interpolation @ pv)*const.m_electron*self.weight_factor/self.dx
        moments[2, :] = (interpolation @ pv**2)*0.5*const.m_electron*self.weight_factor/self.dx
        return moments

    def get_interpolation_matrix(self, x):
        n_idx = np.repeat(((x - self.x_domain[0]) % self.L) // self.dx, self.shape_function_width).astype(int) + self.n_idx_tiling
        n_idx %= self.n_nodes
        interp_vals = self.shape_function(self.get_distance(x[self.p_idx], self.node_positions[n_idx])/self.dx)
        return sparse.csr_array((interp_vals, (n_idx, self.p_idx)), shape=(self.n_nodes, self.n_particles))

    def shift_x_to_domain(self, x):
        return (x - self.x_domain[0]) % self.L + self.x_domain[0]

    def get_distance(self, px, nx):
        px_domain = self.shift_x_to_domain(px)
        return 0.5*self.L - np.abs(np.abs(px_domain - nx) - 0.5*self.L)

    def push_particles(self):
        self.px += self.pv*self.dt
        #self.px = self.shift_x_to_domain(self.px)

    def interpolate_particles_to_field(self):
        self.interpolation = self.get_interpolation_matrix(self.px)
        self.nrho = -const.q_electron*self.weight_factor*self.interpolation.sum(axis=1)/self.dx**3 + self.bg_charge_density

    def update_electric_field(self):
        self.ne_field = self.electric_field_matrix @ self.nrho

    def interpolate_field_to_particles(self):
        self.pe_field = self.interpolation.T @ self.ne_field

    def accelerate_particles(self):
        self.pv += -const.q_over_m*self.dt*self.pe_field

    def update(self):
        self.push_particles()
        self.interpolate_particles_to_field()
        self.update_electric_field()
        self.interpolate_field_to_particles()
        self.accelerate_particles()
        self.time += self.dt

    def run(self, save_snapshots=True):
        # Set up vectors
        y1 = self.shape_function(np.arange(-100, 101))
        y2 = self.shape_function(np.arange(-100.5, 101))
        width = max(np.sum(np.abs(y1) > 0), np.sum(np.abs(y2) > 0))
        relative_n_idx = np.arange(-(width//2), width//2 + 1, dtype=int)
        self.shape_function_width = relative_n_idx.shape[0]
        self.n_idx_tiling = np.tile(relative_n_idx, self.n_particles)
        self.p_idx = np.repeat(range(self.n_particles), self.shape_function_width)

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
        self.phi_matrix = -self.dx**2 * np.linalg.pinv(A) / const.epsilon0
        self.electric_field_matrix = -(0.5 / self.dx) * B @ self.phi_matrix

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
        self.snapshots.append(Snapshot(self.time, self.px.copy(), self.pv.copy(), self.phi_matrix @ self.nrho, self.ne_field.copy()))

    def show_potential(self, n0, debye_length, fps=10, save_animation=False, filename='Electric_potential', repeat=True):
        print('Generating animation...')
        fig = plt.figure(figsize=(10, 5))
        prefactor = -debye_length*const.q_electron*n0/(2*const.epsilon0)
        x = -0.5*np.abs(self.node_positions-0.5*self.L)/debye_length
        a = 0.5*(0.1*self.L/debye_length)**2
        theory = prefactor*(4*debye_length+self.L*np.exp(a+2*x)*(erf((x+a)/np.sqrt(a))-1-np.exp(-4*x)*(erf((x-a)/np.sqrt(a))+1)))
        
        max_phi = np.array([s.phi for s in self.snapshots]).max()
        min_phi = np.array([s.phi for s in self.snapshots]).min()
        phi_domain = [min_phi, max_phi]

        def show(ts):
            t_idx, s = ts

            fig.clear()
            plt.plot(self.node_positions, s.phi, label='Simulated')
            plt.title(f'Time = {s.time: .3g}')
            plt.ylabel(r'Electric Potential ($V$)')
            plt.xlabel('x (m)')
            plt.ylim(*phi_domain)
            plt.xlim(self.x_domain)
            
            # Theory
            plt.plot(self.node_positions, theory, linestyle='--', color='k', label='Theory')
            plt.legend()

        show((0, self.snapshots[0]))
        ani = animation.FuncAnimation(fig=fig, func=show, frames=enumerate(self.snapshots), interval=1e3/fps, repeat=repeat)
        if save_animation:
            print('Saving...')
            writer = animation.PillowWriter(fps=fps)
            ani.save(f'{filename}.gif', writer=writer)
        print('Displaying...')
        plt.show()
        print('Done!')
        
    def show_electric_field(self, fps=10, save_animation=False, filename='Electric_field', repeat=True):
        print('Generating plot...')
        fig = plt.figure(figsize=(10, 5))
        e_field = np.abs(np.array([fft.fft(s.e_field)[1] for s in self.snapshots]))
        time = np.array([s.time for s in self.snapshots])
        plt.plot(time, e_field)
        plt.ylabel('Electric Field (N/C)')
        plt.xlabel('Time (s)')
        plt.yscale('log')
        
        # Determine maximum points
        idx = np.array([i if e_field[i-1] < e_field[i] and e_field[i+1] < e_field[i] else 0 for i in range(1, e_field.shape[0]-1)])
        idx = idx[idx > 0]
        # Determine decay region
        decay_idx = [idx[0]]
        for i in range(1, idx.shape[0]):
            if e_field[idx[i]] >= e_field[idx[i-1]]:
                break
            decay_idx.append(idx[i])
        decay_idx = np.array(decay_idx)
        # Determine growth region
        growth_idx = [decay_idx[-1]]
        for i in range(decay_idx.shape[0] , idx.shape[0]):
            if e_field[idx[i]] <= e_field[idx[i-1]]:
                break
            growth_idx.append(idx[i])
        # Fit curves
        log_decay = np.log(e_field[decay_idx])
        gamma_decay, a = np.polyfit(time[decay_idx], log_decay, 1)
        decay_line = np.exp(gamma_decay * time[decay_idx] + a)
        log_growth = np.log(e_field[growth_idx])
        gamma_growth, a = np.polyfit(time[growth_idx], log_growth, 1)
        growth_line = np.exp(gamma_growth * time[growth_idx] + a)
        plt.plot(time[decay_idx], decay_line, color='r', linestyle='--', label=f'$\gamma$ = {gamma_decay:.3g}')
        plt.plot(time[growth_idx], growth_line, color='g', linestyle='--', label=f'$\gamma$ = {gamma_growth:.3g}')
        plt.legend()
        plt.show()
        
        print('Generating animation...')
        fig = plt.figure(figsize=(10, 5))
        
        max_e_field = np.array([s.e_field for s in self.snapshots]).max()
        min_e_field = np.array([s.e_field for s in self.snapshots]).min()
        e_field_domain = [min_e_field, max_e_field]

        def show(ts):
            t_idx, s = ts

            fig.clear()
            plt.plot(self.node_positions, s.e_field)
            plt.title(f'Time = {s.time: .3g}')
            plt.ylabel('Electric Field (N/C)')
            plt.xlabel('x (m)')
            plt.ylim(*e_field_domain)
            plt.xlim(self.x_domain)

        show((0, self.snapshots[0]))
        ani = animation.FuncAnimation(fig=fig, func=show, frames=enumerate(self.snapshots), interval=1e3/fps, repeat=repeat)
        if save_animation:
            print('Saving...')
            writer = animation.PillowWriter(fps=fps)
            ani.save(f'{filename}.gif', writer=writer)
        print('Displaying...')
        plt.show()
        print('Done!')

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
                    x, v = np.array([[self.shift_x_to_domain(x), v] for c, x, v in zip(self.colors, s.x, s.v) if c == color]).T.reshape(2, -1)
                    ax.scatter(x, v, alpha=0.5, color=color)
                ax.set_title(f'Time = {s.time: .3g}')
                ax.set_ylabel('$v_x$ (m/s)')
                ax.set_ylim(self.v_domain)
                ax.set_xlim(self.x_domain)
                ax.set_xlabel('$x$ (m)')
            else:
                ax[0].clear()
                for color in colors:
                    x, v = np.array([[self.shift_x_to_domain(x), v] for c, x, v in zip(self.colors, s.x, s.v) if c == color]).T.reshape(2, -1)
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

    def save_snapshots_to_csv(self, filename):
        print(f'Saving to {filename}...')
        particle_list = []
        node_list = []
        for s in self.snapshots:
            particle_list.append(np.concat((s.x, s.v)))
            node_list.append(s.e_field)
        particle_array = np.column_stack(particle_list)
        node_array = np.column_stack(node_list)
        np.savetxt(f'{filename}_particles.csv', particle_array, delimiter=',')
        np.savetxt(f'{filename}_node_positions.csv', self.node_positions, delimiter=',')
        np.savetxt(f'{filename}_nodes.csv', node_array, delimiter=',')
        print('Done!')

    def show_integrated_moments(self, save=False, filename='Integrated Moments'):
        moment_array = np.array([self.get_moments(s.x, s.v) for s in self.snapshots])
        integrated_moments = np.trapezoid(moment_array, [s.time for s in self.snapshots], axis=0)
        moment_labels = ['Density (kg/m)', 'Momentum Density (kg/s)', 'Energy Density (J/m)']
        fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        ax[0].set_title('Integrated Moments')
        for a, m, label in zip(ax, integrated_moments, moment_labels):
            a.plot(self.node_positions, m)
            a.set_ylabel(label)
        plt.xlabel('x (m)')
        if save:
            plt.savefig(filename)
        plt.show()
        
    def plot_fastest_particle_trajectory(self):
        v_avg = np.array([s.v for s in self.snapshots]).sum(axis=0)
        idx = np.argmax(v_avg)
        x = [s.x[idx] for s in self.snapshots]
        t = [s.time for s in self.snapshots]
        plt.plot(t, x)
        plt.xlabel('t (s)')
        plt.ylabel('x (m)')
        plt.show()