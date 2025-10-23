import numpy as np
from tyranowskisim import TyranowskiSimulation
import constants
from pdfsampler import PdfSampler
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    v_domain = [-10, 10]
    x_domain = [-1, 1]
    n_particles = 1000

    def f(x, v):
        eta = 10
        a = 0.3
        v0 = 4
        sigma = 0.5
        return np.exp(-0.5*(x/eta)**2) / (np.sqrt(2*np.pi)*eta) * (np.exp(-0.5*v**2) / ((1+a)*np.sqrt(2*np.pi)) + a*np.exp(-0.5*((v - v0)/sigma)**2)/((1+a)*(np.sqrt(2*np.pi)*sigma)))

    seed = 480572
    beta = 6
    use_data = True

    if use_data:
        snapshots = np.loadtxt(f'tyranowski_snapshots_{n_particles}p.csv', dtype=float, delimiter=',')
        u0 = snapshots[:, 0]
        x0 = u0[:n_particles]
        v0 = u0[n_particles:]
    else:
        x0, v0 = PdfSampler.sample(f, (x_domain, v_domain), n_particles, seed=seed)
        snapshots = np.vstack((x0.reshape(-1, 1), v0.reshape(-1, 1)))

    def calculate_hamiltonian(x, v):
        return 0.5*np.sum(v**2 + 0.5*beta**2*x**4)

    dt = 0.0001
    end_time = 10
    interval = 100
    t = np.arange(0, end_time+0.1*dt*interval, dt*interval)
    h_ref = calculate_hamiltonian(x0, v0)

    sim = TyranowskiSimulation(
        particle_snapshots=snapshots,
        dt=dt, end_time=end_time,
        x_domain=x_domain,
        v_domain=v_domain,
        beta=beta,
        snapshot_interval=interval
    )
    
    type='PSD'
    sim.show_projection_error_function(step=20, type=type)

    n_modes = 120
    sim.run(n_modes=n_modes, type=type)
    sim.show_moments(8, compare=True)
    sim.show_statistics(t, h_ref, hfunc=calculate_hamiltonian, title='Tyranowski Cubic Statistics')
    sim.show_snapshots(fps=20, compare=True, save_animation=True, filename=f'tyranowski_cubic_{n_particles}p_{type}_{n_modes}m')