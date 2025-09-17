import numpy as np
from particleromsimulation import RomSimulation
import constants

if __name__ == "__main__":
    n_cells = 64
    n_particles_per_cell = 100

    n0 = 1e15
    Te = 10000000
    electron_plasma_frequency = np.sqrt(n0*constants.q_electron**2 / (constants.epsilon0 * constants.m_electron))
    inv_w_pe = 1/electron_plasma_frequency
    plasma_skin_depth = constants.c*inv_w_pe
    debye_length = np.sqrt((constants.epsilon0*constants.boltzmann*Te)/(n0*constants.q_electron**2))
    v0 = electron_plasma_frequency*debye_length
    v_T = np.sqrt(constants.boltzmann*Te/constants.m_electron)
    L = 20*debye_length
    wq = 0.1*L
    xc = L/2
    background_charge_density = lambda x: -constants.q_electron*n0*np.exp(-(x - xc)**2/(2*wq**2))/(np.sqrt(2*np.pi)*wq)
    weight_factor = L*n0 / (n_particles_per_cell*n_cells)

    def f(x, v):
        return np.exp(-v**2/(2*v_T**2))/(np.sqrt(2*np.pi)*v_T*L)

    dt = 0.05*inv_w_pe
    end_time = dt*2e2

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
        v_domain=(-20*v0, 20*v0),
        background_charge_density=background_charge_density,
        particle_weight_factor=weight_factor,
        particle_order=1,
        snapshot_interval=3
    )
    
    n_particle_modes = 400
    pod_type = 'PSD'
    # sim.show_singular_values()
    # sim.show_projection_error_function(max=500, step=20)
    sim.run(n_particle_modes, pod_type=pod_type)
    sim.show_snapshots(fps=10, save_animation=True,
                       filename=f'{filename}_particle_rom_{pod_type}_{n_particle_modes}pm', show_moments=False)
    sim.show_statistics()