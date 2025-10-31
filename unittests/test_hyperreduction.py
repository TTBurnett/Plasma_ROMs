import sys
sys.path.append('..')
from particleromsimulation import RomSimulation
import pytest
import numpy as np
import constants
import matplotlib.pyplot as plt

def test_interpolation_hyperreduction():
    n_cells = 5
    n_particles_per_cell = 5

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

    print('Loading snapshots...')
    filename = f'debye_shielding_{n_cells}c{n_particles_per_cell}ppc'
    particle_snapshots = np.loadtxt(f'{filename}_particles.csv', dtype=float, delimiter=',')
    node_positions = np.loadtxt(f'{filename}_node_positions.csv', dtype=float, delimiter=',')

    print('Setting up simulation...')
    sim = RomSimulation(
        particle_snapshots=particle_snapshots,
        node_positions=node_positions,
        dt=dt, end_time=end_time,
        x_domain=(0, L),
        v_domain=(-10*v_T, 10*v_T),
        background_charge_density=background_charge_density,
        particle_weight_factor=weight_factor,
        particle_order=1,
        snapshot_interval=10,
        interpolation_snapshot_file=f'{filename}_interpolations.npz'
    )
    
    np.set_printoptions(formatter={'float': lambda x: f"{x:.2f}"})
    pod_type = 'POD'
    
    total_interpolation_size = 125
    total_particle_number = 25
    n_particle_modes = 25
    sim.setup_rom(n_particle_modes, total_particle_number, pod_type)
    sim.n_idx_tiling = np.tile([-1, 0, 1], sim.n_particles)
    sim.p_idx = np.repeat(np.arange(sim.n_particles), 3)
    sim.n_hyperreduction_points = 25
    true_ans = sim.get_interpolation_matrix(sim.psi_px @ sim.px).T
    
    error = np.empty((total_interpolation_size, total_particle_number))
    for i in range(1, total_interpolation_size+1):
        for j in range(1, total_particle_number+1):
            n_hyperreduction_points = j
            n_hyperreduction_modes = i
            n_pe_field_modes = i
            sim.setup_hyperreduction(n_hyperreduction_points, n_hyperreduction_modes, n_pe_field_modes, should_print=False)
            sim.interpolate_particles_to_field()
            ans = sim.unhyperreduce_and_reshape @ sim.interpolation.reshape(-1)
            error_value = np.linalg.norm(ans - true_ans) / np.linalg.norm(true_ans)
            error[i-1, j-1] = np.log10(error_value)
            print(i, j)
    
    X, Y = np.meshgrid(np.arange(1, total_particle_number+1), np.arange(1, total_interpolation_size+1))

    # Create 3D surface plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, error, cmap='viridis', edgecolor='none')

    # Labels and title
    ax.set_xlabel('# of Hyperreduction Points')
    ax.set_ylabel('# of Nonlinearity Modes')
    ax.set_zlabel('Relative Error')
    ax.set_title('Error Surface')

    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.6, label='Relative Error')

    plt.tight_layout()
    plt.show()
    
def test_pe_field_hyperreduction():
    n_cells = 5
    n_particles_per_cell = 5

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

    print('Loading snapshots...')
    filename = f'debye_shielding_{n_cells}c{n_particles_per_cell}ppc'
    particle_snapshots = np.loadtxt(f'{filename}_particles.csv', dtype=float, delimiter=',')
    node_positions = np.loadtxt(f'{filename}_node_positions.csv', dtype=float, delimiter=',')

    print('Setting up simulation...')
    sim = RomSimulation(
        particle_snapshots=particle_snapshots,
        node_positions=node_positions,
        dt=dt, end_time=end_time,
        x_domain=(0, L),
        v_domain=(-10*v_T, 10*v_T),
        background_charge_density=background_charge_density,
        particle_weight_factor=weight_factor,
        particle_order=1,
        snapshot_interval=10
    )
    
    np.set_printoptions(formatter={'float': lambda x: f"{x:.2f}"})
    pod_type = 'POD'
    
    total_interpolation_size = 125
    total_particle_number = 25
    n_particle_modes = 25
    sim.setup_rom(n_particle_modes, total_particle_number, pod_type)
    sim.n_idx_tiling = np.tile([-1, 0, 1], sim.n_particles)
    sim.p_idx = np.repeat(np.arange(sim.n_particles), 3)
    sim.n_hyperreduction_points = 25
    sim.interpolate_particles_to_field()
    sim.update_electric_field()
    sim.interpolate_field_to_particles()
    true_ans = sim.pe_field.copy()
    true_ne_field = sim.ne_field.copy()
    sim.interpolation_snapshot_file = f'{filename}_interpolations.npz'
    sim.should_hyperreduce = True
    
    error = np.empty((total_particle_number, total_particle_number))
    for i in range(1, total_particle_number+1):
        for j in range(1, total_particle_number+1):
            n_hyperreduction_points = j
            n_hyperreduction_modes = total_interpolation_size
            n_pe_field_modes = i
            sim.setup_hyperreduction(n_hyperreduction_points, n_hyperreduction_modes, n_pe_field_modes, should_print=False)
            sim.interpolate_particles_to_field()
            test_pe = sim.interpolation.T @ true_ne_field
            ans = sim.pe_to_pv @ test_pe
            error_value = np.linalg.norm(ans - true_ans) / np.linalg.norm(true_ans)
            error[i-1, j-1] = np.log10(error_value)
            print(i, j)
    
    X, Y = np.meshgrid(np.arange(1, total_particle_number+1), np.arange(1, total_particle_number+1))

    # Create 3D surface plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, error, cmap='viridis', edgecolor='none')

    # Labels and title
    ax.set_xlabel('# of Hyperreduction Points')
    ax.set_ylabel('# of Nonlinearity Modes')
    ax.set_zlabel('Relative Error')
    ax.set_title('Error Surface')

    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.6, label='Relative Error')

    plt.tight_layout()
    plt.show()
    
def test_combined_hyperreduction():
    n_cells = 5
    n_particles_per_cell = 5

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

    print('Loading snapshots...')
    filename = f'debye_shielding_{n_cells}c{n_particles_per_cell}ppc'
    particle_snapshots = np.loadtxt(f'{filename}_particles.csv', dtype=float, delimiter=',')
    node_positions = np.loadtxt(f'{filename}_node_positions.csv', dtype=float, delimiter=',')

    print('Setting up simulation...')
    sim = RomSimulation(
        particle_snapshots=particle_snapshots,
        node_positions=node_positions,
        dt=dt, end_time=end_time,
        x_domain=(0, L),
        v_domain=(-10*v_T, 10*v_T),
        background_charge_density=background_charge_density,
        particle_weight_factor=weight_factor,
        particle_order=1,
        snapshot_interval=10
    )
    
    np.set_printoptions(formatter={'float': lambda x: f"{x:.2f}"})
    pod_type = 'POD'
    
    total_particle_number = 25
    n_particle_modes = 25
    sim.setup_rom(n_particle_modes, total_particle_number, pod_type)
    sim.n_idx_tiling = np.tile([-1, 0, 1], sim.n_particles)
    sim.p_idx = np.repeat(np.arange(sim.n_particles), 3)
    sim.n_hyperreduction_points = 25
    sim.interpolate_particles_to_field()
    sim.update_electric_field()
    sim.interpolate_field_to_particles()
    true_ans = sim.pe_field.copy()
    sim.interpolation_snapshot_file = f'{filename}_interpolations.npz'
    sim.should_hyperreduce = True
    
    error = np.empty((total_particle_number, total_particle_number))
    for i in range(1, total_particle_number+1):
        for j in range(1, total_particle_number+1):
            n_hyperreduction_points = j
            n_hyperreduction_modes = i*n_cells
            n_pe_field_modes = i
            sim.setup_hyperreduction(n_hyperreduction_points, n_hyperreduction_modes, n_pe_field_modes, should_print=False)
            sim.interpolate_particles_to_field()
            sim.update_electric_field()
            sim.interpolate_field_to_particles()
            ans = sim.pe_field
            error_value = np.linalg.norm(ans - true_ans) / np.linalg.norm(true_ans)
            error[i-1, j-1] = np.log10(error_value)
            print(i, j, error_value)
    
    X, Y = np.meshgrid(np.arange(1, total_particle_number+1), np.arange(1, total_particle_number+1))

    # Create 3D surface plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, error, cmap='viridis', edgecolor='none')

    # Labels and title
    ax.set_xlabel('# of Hyperreduction Points')
    ax.set_ylabel('# of Nonlinearity Modes')
    ax.set_zlabel('Relative Error')
    ax.set_title('Error Surface')

    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.6, label='Relative Error')

    plt.tight_layout()
    plt.show()
    
test_combined_hyperreduction()