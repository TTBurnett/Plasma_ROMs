import numpy as np
from particleromsimulation import RomSimulation
import sys
sys.path.append('..')
import constants

if __name__ == "__main__":
    n_cells = 200
    n_particles_per_cell = 128

    n0 = 1e23
    Te = 1e8
    std_v = np.sqrt(constants.boltzmann*Te/constants.m_electron)
    electron_plasma_frequency = np.sqrt(n0*constants.q_electron**2 / (constants.epsilon0 * constants.m_electron))
    inv_w_pe = 1/electron_plasma_frequency
    debye_length = np.sqrt((constants.epsilon0*constants.boltzmann*Te)/(n0*constants.q_electron**2))
    L = 20*debye_length
    wq = 0.1*L
    xc = L/2
    weight_factor = (L/n_cells)**3*n0 / n_particles_per_cell
    background_charge_density = lambda x: constants.q_electron*n0

    dt = 0.01*inv_w_pe
    end_time = dt*5e3

    print('Loading snapshots...')
    filename = f'landau_damping_{n_cells}c{n_particles_per_cell}ppc'
    particle_snapshots = np.loadtxt(f'{filename}_particles.csv', dtype=float, delimiter=',')
    node_positions = np.loadtxt(f'{filename}_node_positions.csv', dtype=float, delimiter=',')
    node_snapshots = np.loadtxt(f'{filename}_nodes.csv', dtype=float, delimiter=',')

    print('Setting up simulation...')
    sim = RomSimulation(
        particle_snapshots=particle_snapshots,
        node_positions=node_positions,
        dt=dt, end_time=end_time,
        x_domain=(0, L),
        v_domain=(-5*std_v, 5*std_v),
        background_charge_density=background_charge_density,
        particle_weight_factor=weight_factor,
        particle_order=1,
        snapshot_interval=10
    )
    
    n_particle_modes = 50
    pod_type = 'PSD'
    sim.show_singular_values()
    sim.run(n_particle_modes, pod_type=pod_type)
    sim.compare_electric_field(node_snapshots)
    sim.show_snapshots(fps=15, save_animation=False,
                       filename=f'{filename}_particle_rom_{pod_type}_{n_particle_modes}pm', show_moments=False)
    sim.show_statistics()