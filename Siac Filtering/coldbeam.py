import numpy as np
import sys
sys.path.append('..')
from fastsimulation import Simulation
import constants
import filters

if __name__ == "__main__":
    n_cells = 64
    n_particles_per_cell = 200

    n0 = 1e23
    Te = 1e8
    electron_plasma_frequency = np.sqrt(n0*constants.q_electron**2 / (constants.epsilon0 * constants.m_electron))
    inv_w_pe = 1/electron_plasma_frequency
    debye_length = np.sqrt((constants.epsilon0*constants.boltzmann*Te)/(n0*constants.q_electron**2))
    vT = np.sqrt(constants.boltzmann*Te/constants.m_electron)
    vc = 10*vT
    L = debye_length*200
    wq = 0.1*L
    xc = L/2
    weight_factor = (L/n_cells)**3*n0 / n_particles_per_cell
    background_charge_density = lambda x: constants.q_electron*n0
 
    def f(x, v):
        return np.exp(-0.5*((v-vc)/vT)**2)/(np.sqrt(2*np.pi)*vT*L)

    dt = 0.05*inv_w_pe
    end_time = dt*1e3
    shape = filters.SiacFilter(6, 5)
    
    sim = Simulation(
        n_nodes=n_cells,
        n_particles_per_cell=n_particles_per_cell,
        particle_weight_factor=weight_factor,
        particle_shape_function=shape,
        dt=dt, end_time=end_time,
        f0=f,
        x_domain=(0, L),
        v_domain=(0, 20*vT),
        background_charge_density=background_charge_density,
        snapshot_interval=5
    )
    
    filename = f'debye_shielding_{n_cells}c{n_particles_per_cell}ppc'

    sim.run()
    sim.show_snapshots(fps=15, save_animation=False, filename=filename, show_moments=False)
