import numpy as np
from podsimulationneuralnetnodes import PodSimulation
import constants

if __name__ == "__main__":
    # Select data
    n_cells = 80
    n_particles_per_cell = 200
    n_node_modes = 40

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

    dt = 0.005*inv_w_pe
    end_time = dt*15e3
    background_charge_density = -constants.q_electron * n0

    filename = f'two_stream_{n_cells}c{n_particles_per_cell}ppc'
    node_snapshots = np.loadtxt(f'{filename}_nodes.csv', dtype=float, delimiter=',')[:n_cells]
    particle_snapshots = np.loadtxt(f'{filename}_particles.csv', dtype=float, delimiter=',')[:2*n_particles_per_cell*n_cells]
    node_positions = np.loadtxt(f'{filename}_node_positions.csv', dtype=float, delimiter=',')
    hidden_layers = [1000 for i in range(2)]

    sim = PodSimulation(
        e_field_model_file=f'{filename}_romq_to_pe_net',
        romq_model_file=f'{filename}_px_to_romq_net',
        e_field_scaling_file=f'{filename}_romq_to_pe_scaling',
        romq_scaling_file=f'{filename}_px_to_romq_scaling',
        hidden_layers=hidden_layers,
        n_node_modes=n_node_modes,
        node_snapshots=node_snapshots,
        particle_snapshots=particle_snapshots,
        node_positions=node_positions,
        dt=dt, end_time=end_time,
        x_domain=(-max_x, max_x),
        v_domain=(-max_v, max_v),
        background_charge_density=background_charge_density,
        particle_weight_factor=weight_factor,
        snapshot_interval=15,
        color_rule=color_rule
    )

    sim.run()
    sim.show_statistics()
    sim.show_snapshots(fps=20, save_animation=True, filename=f'nn_node_rom_{filename}_{n_node_modes}nm')