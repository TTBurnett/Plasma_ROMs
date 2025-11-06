import numpy as np
from podsimulationparticles import PodSimulation
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

    dt = 0.005*inv_w_pe
    end_time = dt*15e3
    background_charge_density = -constants.q_electron * n0

    filename = f'two_stream_{n_cells}c{n_particles_per_cell}ppc'
    node_snapshots = np.loadtxt(f'{filename}_nodes.csv', dtype=float, delimiter=',')[:n_cells]
    particle_snapshots = np.loadtxt(f'{filename}_particles.csv', dtype=float, delimiter=',')[:2*n_particles_per_cell*n_cells]
    node_positions = np.loadtxt(f'{filename}_node_positions.csv', dtype=float, delimiter=',')
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
        snapshot_interval=15,
        color_rule=color_rule
    )

    type = 'PSD'
    sim.show_singular_values(type)

    n_modes=125
    sim.run(n_modes, type)
    sim.show_snapshots(fps=20, save_animation=True, filename=f'particle_{type}_rom_{filename}_{n_modes}m')
    sim.show_statistics()