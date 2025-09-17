import numpy as np
from fastsimulation import Simulation
import constants

if __name__ == "__main__":
    n_cells = 64
    n_particles_per_cell = 100

    n0 = 1e15
    electron_plasma_frequency = np.sqrt(n0*constants.q_electron**2 / (constants.epsilon0 * constants.m_electron))
    inv_w_pe = 1/electron_plasma_frequency
    plasma_skin_depth = constants.c*inv_w_pe
    mean_v = 0.2*constants.c
    std_v = 0.01*constants.c
    max_v = 2.2*mean_v
    resonance = 2*np.pi*mean_v*inv_w_pe
    max_x = resonance * 1
    weight_factor = 2*max_x*n0 / (n_particles_per_cell*n_cells)

    def f(x, v):
        return np.exp(-0.5 * ((abs(v) - mean_v)/std_v)**2)
    
    def color_rule(x, v):
        return np.where(v > 0, 'c', 'm')

    dt = 0.05*inv_w_pe
    end_time = dt*20e2
    background_charge_density = -constants.q_electron * n0
    
    sim = Simulation(
        n_nodes=n_cells,
        n_particles_per_cell=n_particles_per_cell,
        particle_weight_factor=weight_factor,
        particle_order=1,
        dt=dt, end_time=end_time,
        f0=f,
        x_domain=(-max_x, max_x),
        v_domain=(-max_v, max_v),
        background_charge_density=background_charge_density,
        snapshot_interval=2,
        color_rule=color_rule
    )
    
    filename = f'two_stream_{n_cells}c{n_particles_per_cell}ppc'

    sim.run()
    sim.show_snapshots(fps=20, save_animation=False, filename=filename, show_moments=False)
