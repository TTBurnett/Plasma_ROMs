import numpy as np
import sys
sys.path.append('..')
from src.picsimulation import Simulation
import src.constants as constants
import src.filters as filters

if __name__ == "__main__":
    n_cells = 200
    n_particles_per_cell = 128

    n0 = 1e23
    Te = 1e8
    std_v = np.sqrt(constants.boltzmann*Te/constants.m_electron)
    electron_plasma_frequency = np.sqrt(n0*constants.q_electron**2 / (constants.epsilon0 * constants.m_electron))
    inv_w_pe = 1/electron_plasma_frequency
    debye_length = np.sqrt((constants.epsilon0*constants.boltzmann*Te)/(n0*constants.q_electron**2))
    L = 20*debye_length
    wq = 0.1*L
    xc = L/2
    weight_factor = (L/n_cells)**3*n0 / n_particles_per_cell
    background_charge_density = lambda x: constants.q_electron*n0
 
    alpha = 0.2
    k = 2*np.pi/L
    def f(x, v):
        return (1 + alpha*np.cos(k*x))*np.exp(-0.5*(v/std_v)**2)/(std_v*np.sqrt(2*np.pi))

    dt = 0.01*inv_w_pe
    end_time = dt*5e3
    shape = filters.SplineFilter(3)
    
    khat = k*std_v*inv_w_pe
    gamma_predicted = -np.sqrt(np.pi/8)*np.exp(-1/(2*khat**2))/khat**3
    print(gamma_predicted)
    
    sim = Simulation(
        n_nodes=n_cells,
        n_particles_per_cell=n_particles_per_cell,
        particle_weight_factor=weight_factor,
        particle_shape_function=shape,
        dt=dt, end_time=end_time,
        f0=f,
        x_domain=(0, L),
        v_domain=(-5*std_v, 5*std_v),
        background_charge_density=background_charge_density,
        snapshot_interval=10
    )
    
    filename = f'landau_damping_{n_cells}c{n_particles_per_cell}ppc'

    sim.run()
    sim.save_snapshots_to_csv(filename)
    sim.show_electric_field(fps=15, save_animation=False, filename=f'{filename}_efield')
    sim.show_snapshots(fps=15, save_animation=False, filename=filename, show_moments=False)
