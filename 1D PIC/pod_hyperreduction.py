import numpy as np
from hyperreducedsimulation import PodSimulation
import constants

if __name__ == "__main__":
    n_cells = 50
    n_particles_per_cell = 50

    n0 = 1e15
    electron_plasma_frequency = np.sqrt(n0*constants.q_electron**2 / (constants.epsilon0 * constants.m_electron))
    inv_w_pe = 1/electron_plasma_frequency
    plasma_skin_depth = constants.c*inv_w_pe
    mean_v = 0.2*constants.c
    std_v = 0.01*constants.c
    max_v = 2.2*mean_v
    resonance = 2*np.pi*mean_v*inv_w_pe
    max_x = resonance * 1.25
    weight_factor = 2*max_x*n0 / (n_particles_per_cell*n_cells)

    def f(x, v):
        return np.exp(-0.5 * ((abs(v) - mean_v)/std_v)**2)
    
    def color_rule(x, v):
        return np.where(v > 0, 'c', 'm')

    dt = 0.05*inv_w_pe
    end_time = dt*20e2
    background_charge_density = -constants.q_electron * n0

    print('Loading snapshots...')
    filename = f'two_stream_{n_cells}c{n_particles_per_cell}ppc'
    node_snapshots = np.loadtxt(f'{filename}_nodes.csv', dtype=float, delimiter=',')
    particle_snapshots = np.loadtxt(f'{filename}_particles.csv', dtype=float, delimiter=',')
    node_positions = np.loadtxt(f'{filename}_node_positions.csv', dtype=float, delimiter=',')
    interpolation_snapshots = np.loadtxt(f'{filename}_interpolations.csv', dtype=float, delimiter=',')

    print('Setting up simulation...')
    sim = PodSimulation(
        particle_snapshots=particle_snapshots,
        node_snapshots=node_snapshots,
        node_positions=node_positions,
        dt=dt, end_time=end_time,
        x_domain=(-max_x, max_x),
        v_domain=(-max_v, max_v),
        background_charge_density=background_charge_density,
        particle_weight_factor=weight_factor,
        particle_order=1,
        snapshot_interval=2,
        color_rule=color_rule
    )

    n_particle_modes = 200
    percent_hyperreduction_points = 5 #%
    sim.run(n_particle_modes, percent_hyperreduction_points=percent_hyperreduction_points,
             hyperreduction_algorithm='DEIM')
    sim.show_snapshots(fps=20, save_animation=False,
                       filename=f'hyperreduction_{filename}_{n_particle_modes}pm_{percent_hyperreduction_points:.3g}hr')