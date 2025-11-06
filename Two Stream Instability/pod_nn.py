import numpy as np
from podsimulationneuralnet import PodSimulation
import constants

if __name__ == "__main__":
    # Select data
    n_particle_modes = 200
    n_node_modes = 20
    n_cells = 50
    n_particles_per_cell = 60

    # Setup simulation
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

    filename = f'two_stream_{n_cells}c{n_particles_per_cell}ppc'
    node_snapshots = np.loadtxt(f'{filename}_nodes.csv', dtype=float, delimiter=',')[:, :900]
    particle_snapshots = np.loadtxt(f'{filename}_particles.csv', dtype=float, delimiter=',')[:, :900]
    node_positions = np.loadtxt(f'{filename}_node_positions.csv', dtype=float, delimiter=',')

    sim = PodSimulation(
        model_file=f'{filename}_interp_net_{n_particle_modes}pm{n_node_modes}nm',
        use_neural_net=True,
        n_particle_modes=n_particle_modes,
        n_node_modes=n_node_modes,
        particle_snapshots=particle_snapshots,
        node_snapshots=node_snapshots,
        node_positions=node_positions,
        dt=dt, end_time=end_time,
        x_domain=(-max_x, max_x),
        v_domain=(-max_v, max_v),
        background_charge_density=background_charge_density,
        particle_weight_factor=weight_factor,
        snapshot_interval=2,
        color_rule=color_rule
    )

    sim.run()
    sim.show_snapshots(fps=20, save_animation=False, filename=f'nn_rom_{filename}_{n_particle_modes}pm{n_node_modes}nm')
    # sim.show_statistics()