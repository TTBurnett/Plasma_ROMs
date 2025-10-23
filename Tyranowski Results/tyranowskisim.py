from pdfsampler import PdfSampler
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import constants as const
from typing import Literal
from scipy.integrate import solve_ivp
import sparse
from opt_einsum import contract

class Snapshot:
    def __init__(self, time, pa):
        self.time = time
        self.pa = pa

class TyranowskiSimulation:
    def __init__(self, particle_snapshots,
                 dt, end_time,
                 x_domain, v_domain,
                 beta,
                 snapshot_frames_to_use=None,
                 snapshot_interval = 10):
        self.particle_snapshots = particle_snapshots
        self.snapshot_frames_to_use = snapshot_frames_to_use
        self.time = 0
        self.end_time = end_time
        self.dt = dt
        self.x_domain = x_domain
        self.v_domain = v_domain
        self.n_particles = particle_snapshots.shape[0] // 2
        self.u0 = particle_snapshots[:, 0].reshape(-1, 1)
        self.particle_snapshots = particle_snapshots - self.u0
        self.snapshot_interval = snapshot_interval
        self.snapshots = []
        self.beta = beta

    def show_projection_error_function(self, start=0, max=np.inf, step=1, type: Literal['POD', 'PSD']='POD', end_projection_at_last_training_frame=False):
        errors = []
        modes = []
        match(type):
            case 'POD':
                Ux = np.linalg.svd(self.particle_snapshots[:self.n_particles, :self.snapshot_frames_to_use], compute_uv=True, full_matrices=False).U
                Uv = np.linalg.svd(self.particle_snapshots[self.n_particles:, :self.snapshot_frames_to_use], compute_uv=True, full_matrices=False).U
            case 'PSD':
                x = self.particle_snapshots[:self.n_particles, :self.snapshot_frames_to_use]
                v = self.particle_snapshots[self.n_particles:, :self.snapshot_frames_to_use]
                u = np.hstack((x, v))
                U = np.linalg.svd(u, compute_uv=True, full_matrices=False).U

        max_frame = self.snapshot_frames_to_use if self.snapshot_frames_to_use != None else self.particle_snapshots.shape[1]
        for i in range(start, min(max+1, 2*self.n_particles, max_frame), step):
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
                    
            end_frame = self.snapshot_frames_to_use if end_projection_at_last_training_frame else None
            error = (np.identity(2*self.n_particles) - psi @ psi_t) @ self.particle_snapshots[:, :end_frame]
            norm = np.linalg.norm(self.particle_snapshots[:, :end_frame])
            errors.append(np.linalg.norm(error) / norm)
            modes.append(i)
        plt.plot(modes, errors)
        plt.ylabel('Projection Error')
        plt.xlabel('# of Modes')
        plt.yscale('log')
        plt.show()

    def dpa(self, a):
        linear = self.linear_operator @ a
        # if self.type == 'Full':
        a_cubed = np.power(self.psi @ a, 3)
        # else:
        #     a_cubed = sparse.einsum('ijkl,j,k,l', self.cubing_tensor, a.ravel(), a.ravel(), a.ravel())
        cubic = self.cubic_operator @ a_cubed
        return linear + cubic.reshape(-1, 1)

    def update(self):
        dpa = self.dpa(self.pa)
        self.pa += 0.5*(dpa + self.dpa(self.pa + dpa*self.dt))*self.dt
        self.time += self.dt

    def run(self, n_modes, type: Literal['POD', 'POD Stacked', 'PSD', 'Full']='POD'):
        self.n_modes = n_modes
        self.type = type

        match(type):
            case 'POD Stacked':
                U = np.linalg.svd(self.particle_snapshots[:, :self.snapshot_frames_to_use], compute_uv=True, full_matrices=False).U
                self.psi = U[:, :n_modes]
                self.psi_t = self.psi.T
            case 'POD':
                Ux = np.linalg.svd(self.particle_snapshots[:self.n_particles, :self.snapshot_frames_to_use], compute_uv=True, full_matrices=False).U
                psi_x = Ux[:, :n_modes]
                Uv = np.linalg.svd(self.particle_snapshots[self.n_particles:, :self.snapshot_frames_to_use], compute_uv=True, full_matrices=False).U
                psi_v = Uv[:, :n_modes]
                self.psi = np.block([[psi_x, np.zeros_like(psi_x)],
                                     [np.zeros_like(psi_v), psi_v]])
                self.psi_t = np.block([[psi_x.T, np.zeros_like(psi_x.T)],
                                       [np.zeros_like(psi_v.T), psi_v.T]])
                self.n_modes *= 2
            case 'PSD':
                x = self.particle_snapshots[:self.n_particles, :self.snapshot_frames_to_use]
                v = self.particle_snapshots[self.n_particles:, :self.snapshot_frames_to_use]
                u = np.hstack((x, v))
                U = np.linalg.svd(u, compute_uv=True, full_matrices=False).U
                psi = U[:, :n_modes]
                self.psi = np.block([[psi, np.zeros_like(psi)],
                                     [np.zeros_like(psi), psi]])
                self.psi_t = np.block([[psi.T, np.zeros_like(psi.T)],
                                       [np.zeros_like(psi.T), psi.T]])
                self.n_modes *= 2
            case 'Full':
                self.psi = np.identity(2*self.n_particles)
                self.psi_t = np.identity(2*self.n_particles)
                self.n_modes = 2*self.n_particles
            case _:
                raise ValueError('Invalid Type for Modal Decomposition.')

        self.pa = self.psi_t @ self.u0
        self.save_snapshot()

        # Linear operator
        identity = np.identity(self.n_particles)
        zeros = np.zeros((self.n_particles, self.n_particles))
        self.linear_operator = self.psi_t @ np.block([[zeros, identity], [zeros, zeros]]) @ self.psi
        # if type != 'Full':
        #     coords = []
        #     for i in range(self.n_particles):
        #         coords.append([i, i, i, i])
        #     H = sparse.COO(np.array(coords).T, np.ones(self.n_particles), shape=(2*self.n_particles, 2*self.n_particles, 2*self.n_particles, 2*self.n_particles))
        #     self.cubing_tensor = sparse.einsum('ijkl,jm,kn,lo', H, self.psi, self.psi, self.psi)
        self.cubic_operator = self.psi_t @ np.block([[zeros, zeros], [-self.beta**2*identity, zeros]])

        while self.time < self.end_time:
            self.update()
            if round(self.time / self.dt) % self.snapshot_interval == 0:
                print(f't = {self.time:.3g}/{self.end_time:.3g} ({self.time/self.end_time:.2%})')
                self.save_snapshot()

    def save_snapshot(self):
        self.snapshots.append(Snapshot(self.time, self.pa.copy()))

    def show_statistics(self, t, h_ref, hfunc, title=None, write_statistics=False):
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 10))

        if title != None:
            ax[0].set_title(title)

        errors = []
        for i, s in enumerate(self.snapshots):
            u = self.particle_snapshots[:, i].reshape(-1, 1)
            error = (self.psi @ s.pa - u)
            norm = np.linalg.norm(u)
            errors.append(np.linalg.norm(error) / norm)
        ax[0].plot(t[:len(errors)], errors)
        ax[0].set_ylabel('Relative Error')
        if np.max(error) > 1:
            ax[0].set_ylim([0, 1])

        hamiltonian = []
        for s in self.snapshots:
            u = self.psi @ s.pa
            x = u[:self.n_particles, 0]
            v = u[self.n_particles:, 0]
            h = hfunc(x, v)
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

    def show_snapshots(self, fps=10, save_animation=False, filename='PIC_simulation', repeat=True, compare=True):
        print('Generating animation...')
        fig, ax = plt.subplots(figsize=(10, 10))

        def show(i_s):
            i, s = i_s
            ax.clear()
            u = self.psi @ s.pa
            x = u[:self.n_particles]
            v = u[self.n_particles:]
            plt.scatter(x, v, alpha=0.5, color='c', label='ROM')
            if compare:
                xp = self.particle_snapshots[:self.n_particles, i]
                vp = self.particle_snapshots[self.n_particles:, i]
                plt.scatter(xp, vp, alpha=0.5, color='m', label='FOM')
                plt.legend()
            plt.title(f'Time = {s.time: .3g}')
            plt.xlabel('$x$ (m)')
            plt.xlim(self.x_domain)
            plt.ylabel('$v_x$ (m/s)')
            plt.ylim(self.v_domain)

        show((0, self.snapshots[0]))
        ani = animation.FuncAnimation(fig=fig, func=show, frames=enumerate(self.snapshots), interval=1e3/fps, repeat=repeat)
        if save_animation:
            print('Saving...')
            writer = animation.PillowWriter(fps=fps)
            ani.save(f'{filename}.gif', writer=writer)
        print('Displaying...')
        plt.show()
        print('Done!')

    def get_integrated_moments(self, t, u, n_bins):
        dx = (self.x_domain[1] - self.x_domain[0]) / n_bins
        bins = np.arange(self.x_domain[0]-0.5*dx, self.x_domain[1]+0.51*dx, dx)
        px = u[:self.n_particles]
        pv = u[self.n_particles:]
        moments = np.zeros((3, n_bins+1, t.shape[0]))
        for i in range(n_bins+1):
            condition = np.logical_and(px >= bins[i], px < bins[i+1])
            v = np.where(condition, pv, 0)
            moments[0, i, :] = np.sum(condition, axis=0)/dx
            moments[1, i, :] = np.sum(v, axis=0)/dx
            moments[2, i, :] = np.sum(v**2, axis=0)/dx
        integrated_moments = np.trapezoid(moments, t, axis=2)
        x = 0.5*(bins[1:] + bins[:-1])
        return x, integrated_moments

    def show_moments(self, n_bins, compare=False):
        t = np.fromiter((s.time for s in self.snapshots), dtype=float)
        u = np.zeros((2*self.n_particles, t.shape[0]))
        for i, s in enumerate(self.snapshots):
            u[:, i] = (self.psi @ s.pa).ravel()
        x, integrated_moments = self.get_integrated_moments(t, u, n_bins)
        moment_names = ['Density', 'Momentum Density', 'Energy Density']
        colors = ['c','m','g']
        for i, m in enumerate(integrated_moments):
            plt.plot(x, m, color=colors[i], label=f'ROM {moment_names[i]}')
        if compare:
            _, snapshot_moments = self.get_integrated_moments(t, self.particle_snapshots[:, :t.shape[0]], n_bins)
            for i, m in enumerate(snapshot_moments):
                plt.plot(x, m, color=colors[i], label=f'FOM {moment_names[i]}', linestyle='--')
        plt.legend()
        plt.xlabel('$x$ (m)')
        plt.show()

    def save_snapshots_to_csv(self, filename):
        print(f'Saving to {filename}...')
        u = np.zeros((2*self.n_particles, len(self.snapshots)))
        for i, s in enumerate(self.snapshots):
            u[:, i] = (self.psi @ s.pa).ravel()
        np.savetxt(f'{filename}.csv', u, delimiter=',')
        print('Done!')