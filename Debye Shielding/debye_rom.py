import numpy as np
import sys
sys.path.append('..')
from particleromsimulation import RomSimulation
import constants

if __name__ == "__main__":
    n_cells = 64
    n_particles_per_cell = 200

    n0 = 1e23
    Te = 1e8
    electron_plasma_frequency = np.sqrt(n0*constants.q_electron**2 / (constants.epsilon0 * constants.m_electron))
    inv_w_pe = 1/electron_plasma_frequency
    debye_length = np.sqrt((constants.epsilon0*constants.boltzmann*Te)/(n0*constants.q_electron**2))
    v_T = np.sqrt(constants.boltzmann*Te/constants.m_electron)
    L = 20*debye_length
    wq = 0.1*L
    xc = L/2
    weight_factor = (L/n_cells)**3*n0 / n_particles_per_cell
    background_charge_density = lambda x: 0.5*constants.q_electron*n0*(1 + L*np.exp(-(x - xc)**2/(2*wq**2))/(wq*np.sqrt(2*np.pi)))

    def f(x, v):
        return np.exp(-v**2/(2*v_T**2))/(np.sqrt(2*np.pi)*v_T*L)

    dt = 0.005*inv_w_pe
    end_time = dt*1e4

    print('Loading snapshots...')
    filename = f'debye_shielding_{n_cells}c{n_particles_per_cell}ppc'
    particle_snapshots = np.loadtxt(f'{filename}_particles.csv', dtype=float, delimiter=',')
    node_positions = np.loadtxt(f'{filename}_node_positions.csv', dtype=float, delimiter=',')

    print('Setting up simulation...')
    sim = RomSimulation(
        particle_snapshots=particle_snapshots,
        node_positions=node_positions,
        dt=dt, end_time=end_time,
        x_domain=(0, L),
        v_domain=(-10*v_T, 10*v_T),
        background_charge_density=background_charge_density,
        particle_weight_factor=weight_factor,
        particle_order=1,
        snapshot_interval=10,
        interpolation_snapshot_file=f'{filename}_interpolations.npz'
    )
    
    n_particle_modes = 50
    n_hyperreduction_points = 100
    n_hyperreduction_modes = n_hyperreduction_points*n_cells
    pod_type = 'PSD'
    sim.run(n_particle_modes, n_hyperreduction_points, n_hyperreduction_modes, pod_type=pod_type)
    sim.show_energy(filename=f'{filename}_energy')
    sim.show_snapshots(fps=15, save_animation=True,
                       filename=f'{filename}_particle_rom_{pod_type}_{n_particle_modes}pm_{n_hyperreduction_points}hp', show_moments=False)