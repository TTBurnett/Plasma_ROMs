import numpy as np
import sys
sys.path.append('..')
from fastsimulation import Simulation
import constants
import filters

if __name__ == "__main__":
    n_cells = 10
    n_particles_per_cell = 10

    n0 = 1e23
    Te = 1e8
    electron_plasma_frequency = np.sqrt(n0*constants.q_electron**2 / (constants.epsilon0 * constants.m_electron))
    inv_w_pe = 1/electron_plasma_frequency
    debye_length = np.sqrt((constants.epsilon0*constants.boltzmann*Te)/(n0*constants.q_electron**2))
    v_T = np.sqrt(constants.boltzmann*Te/constants.m_electron)
    L = 20*debye_length
    wq = 0.1*L
    xc = L/2
    weight_factor = (L/n_cells)**3*n0 / n_particles_per_cell
    background_charge_density = lambda x: 0.5*constants.q_electron*n0*(1 + L*np.exp(-(x - xc)**2/(2*wq**2))/(wq*np.sqrt(2*np.pi)))
 
    def f(x, v):
        return np.exp(-v**2/(2*v_T**2))/(np.sqrt(2*np.pi)*v_T*L)

    dt = 0.005*inv_w_pe
    end_time = dt*1e4
    shape = filters.SplineFilter(3)
    
    sim = Simulation(
        n_nodes=n_cells,
        n_particles_per_cell=n_particles_per_cell,
        particle_weight_factor=weight_factor,
        particle_shape_function=shape,
        dt=dt, end_time=end_time,
        f0=f,
        x_domain=(0, L),
        v_domain=(-10*v_T, 10*v_T),
        background_charge_density=background_charge_density,
        snapshot_interval=10
    )
    
    filename = f'debye_shielding_{n_cells}c{n_particles_per_cell}ppc'

    sim.run()
    sim.save_snapshots_to_csv(filename)
    #sim.show_potential(fps=15, n0=n0, debye_length=debye_length, save_animation=False)
    sim.show_snapshots(fps=15, save_animation=True, filename=filename, show_moments=False)
    # sim.show_integrated_moments(save=True, filename=f'{filename}_moments')
