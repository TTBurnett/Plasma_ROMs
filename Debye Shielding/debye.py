import numpy as np
from fastsimulation import Simulation
import constants

if __name__ == "__main__":
    n_cells = 40
    n_particles_per_cell = 20

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
    end_time = dt*20e2
    
    sim = Simulation(
        n_nodes=n_cells,
        n_particles_per_cell=n_particles_per_cell,
        particle_weight_factor=weight_factor,
        dt=dt, end_time=end_time,
        f0=f,
        x_domain=(0, L),
        v_domain=(-20*v0, 20*v0),
        background_charge_density=background_charge_density,
        snapshot_interval=3
    )
    
    filename = f'debye_shielding_{n_cells}c{n_particles_per_cell}ppc'

    sim.run()
    sim.plot_fastest_particle_trajectory()
    sim.show_integrated_moments(save=True, filename=f'{filename}_moments')
    sim.save_snapshots_to_csv(filename)
    sim.show_snapshots(fps=15, save_animation=False, filename=filename, show_moments=True)
