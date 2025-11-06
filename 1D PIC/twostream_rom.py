import numpy as np
import sys
sys.path.append('..')
from particleromsimulation import RomSimulation
import constants

if __name__ == "__main__":
    n_cells = 64
    n_particles_per_cell = 200

    n0 = 1e23
    electron_plasma_frequency = np.sqrt(n0*constants.q_electron**2 / (constants.epsilon0 * constants.m_electron))
    inv_w_pe = 1/electron_plasma_frequency
    plasma_skin_depth = constants.c*inv_w_pe
    mean_v = 0.2*constants.c
    std_v = 0.01*constants.c
    max_v = 2.2*mean_v
    debye_length = np.sqrt((0.5*constants.m_electron*constants.epsilon0*mean_v**2)/(n0*constants.q_electron**2))
    resonance = 2*np.pi*mean_v*inv_w_pe
    max_x = resonance
    weight_factor = (2*max_x/n_cells)**3*n0 / n_particles_per_cell

    def f(x, v):
        return np.exp(-0.5 * ((abs(v) - mean_v)/std_v)**2)
    
    def color_rule(x, v):
        return np.where(v > 0, 'c', 'm')

    dt = 0.005*inv_w_pe
    end_time = dt*1e4
    background_charge_density = lambda x: constants.q_electron * n0

    print('Loading snapshots...')
    filename = f'two_stream_{n_cells}c{n_particles_per_cell}ppc'
    particle_snapshots = np.loadtxt(f'{filename}_particles.csv', dtype=float, delimiter=',')
    node_positions = np.loadtxt(f'{filename}_node_positions.csv', dtype=float, delimiter=',')

    print('Setting up simulation...')
    sim = RomSimulation(
        particle_snapshots=particle_snapshots,
        node_positions=node_positions,
        dt=dt, end_time=end_time,
        x_domain=(-max_x, max_x),
        v_domain=(-max_v, max_v),
        background_charge_density=background_charge_density,
        particle_weight_factor=weight_factor,
        particle_order=1,
        snapshot_interval=15,
        interpolation_snapshot_file=f'{filename}_interpolations.npz'
    )
    
    n_particle_modes = 50
    n_hyperreduction_points = 100
    n_interpolation_modes = int(0.9*n_hyperreduction_points*n_cells)
    n_pe_field_modes = int(0.9*n_hyperreduction_points)
    pod_type = 'PSD'
    sim.run(n_particle_modes, n_hyperreduction_points, n_interpolation_modes, n_pe_field_modes, pod_type=pod_type)
    sim.show_energy(filename=f'{filename}_energy')
    sim.show_snapshots(fps=15, save_animation=True,
                       filename=f'{filename}_particle_rom_{pod_type}_{n_particle_modes}pm{n_hyperreduction_points}hp{n_interpolation_modes}im{n_pe_field_modes}pem',
                       show_moments=False)