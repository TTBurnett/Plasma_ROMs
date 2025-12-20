import numpy as np
import sys
sys.path.append('..')
from src.picsimulation import Simulation
import src.constants as constants
import src.filters as filters

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
    external_electric_field = lambda x: 0
    shape = filters.SplineFilter(3)
    
    sim = Simulation(
        n_nodes=n_cells,
        n_particles_per_cell=n_particles_per_cell,
        particle_weight_factor=weight_factor,
        particle_shape_function=shape,
        dt=dt, end_time=end_time,
        f0=f,
        x_domain=(-max_x, max_x),
        v_domain=(-max_v, max_v),
        external_electric_field=external_electric_field,
        snapshot_interval=15,
        color_rule=color_rule
    )
    
    filename = f'two_stream_{n_cells}c{n_particles_per_cell}ppc'

    sim.run()
    sim.save_snapshots_to_csv(filename=filename)
    sim.show_snapshots(fps=15, save_animation=True, filename=filename, show_moments=False)
