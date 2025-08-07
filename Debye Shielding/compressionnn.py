import sys
sys.path.append('..')
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
import constants
import romtools

class IdentityActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, x):
        return x + self.alpha*x*(torch.sigmoid(x) - 1)

class CompressionNN(nn.Module):
    def __init__(self, node_positions, psi_particles, psi_nodes):
        super().__init__()
        self.n_nodes = node_positions.shape[0]
        self.n_particles = psi_particles.shape[0]
        self.n_particle_modes = psi_particles.shape[1]
        self.n_node_modes = psi_nodes.shape[1]
        dx = node_positions[1] - node_positions[0]
        self.min_x = node_positions[0] - 0.5*dx
        self.L = dx*self.n_nodes

        # self.identity_one = nn.Parameter(torch.eye(self.n_particle_modes), requires_grad=True)
        # self.identity_two = nn.Parameter(torch.eye(self.n_particle_modes), requires_grad=True)
        # self.identity_three = nn.Parameter(torch.eye(self.n_particle_modes), requires_grad=True)
        # self.bias_one = nn.Parameter(torch.zeros(self.n_particle_modes, 1), requires_grad=True)
        # self.bias_two = nn.Parameter(torch.zeros(self.n_particle_modes, 1), requires_grad=True)
        # self.bias_three = nn.Parameter(torch.zeros(self.n_particle_modes, 1), requires_grad=True)
        self.bias_four = nn.Parameter(torch.zeros(self.n_particle_modes*self.n_node_modes, 1), requires_grad=True)
        # self.sigma1 = IdentityActivation()
        # self.sigma2 = IdentityActivation()
        # self.sigma3 = IdentityActivation()

        nx = np.repeat(node_positions, self.n_particles).reshape(-1, 1)
        self.nx = nn.Parameter(torch.from_numpy(nx), requires_grad=True)
        self.dx = nn.Parameter(torch.from_numpy(np.ones_like(nx)*dx), requires_grad=True)
        psi_p = np.tile(psi_particles, (self.n_nodes, 1))
        self.psi_p = nn.Parameter(torch.from_numpy(psi_p), requires_grad=True)
        self.psi_particles_sum = torch.from_numpy(np.sum(psi_particles, axis=0))
        reduce_matrix = np.kron(psi_nodes, psi_particles).T
        self.reduce_matrix = nn.Parameter(torch.from_numpy(reduce_matrix), requires_grad=True)
        self.lifted_dim = self.reduce_matrix.shape[1]

    def get_distance(self, x_modulus):
        return 0.5*self.L - torch.abs(torch.abs(x_modulus - self.nx) - 0.5*self.L)
    
    def b_spline(self, x):
        y = torch.zeros_like(x)
        mask1 = x <= 0.5
        y[mask1] = 0.75-x[mask1]**2
        mask2 = (x > 0.5) & (x < 1.5)
        y[mask2] = 0.5*(1.5-x[mask2])**2
        return y

    def activation(self, x):
        x_modulus = torch.remainder(x - self.min_x, self.L) + self.min_x
        x_ref = self.get_distance(x_modulus) / self.dx
        return self.b_spline(x_ref)

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        # h1 = self.sigma1(self.identity_one @ x.T + self.bias_one)
        # h2 = self.sigma2(self.identity_two @ h1 + self.bias_two)
        # h3 = self.sigma3(self.identity_three @ h2 + self.bias_three)
        self.h1 = self.activation(self.psi_p @ x.T)
        y = self.reduce_matrix @ self.h1 + self.bias_four
        interp = y.reshape(self.n_node_modes, self.n_particle_modes, -1).permute(2, 0, 1)
        return interp
        factor = self.psi_particles_sum / torch.sum(interp, dim=1)
        print(factor.max())
        return interp * factor[:, None, :]

    def drop_hidden_neurons(self, fraction, optimizer_type, norm_order=2, energy_order=1, **optimizer_kwargs):
        n_current_neurons = self.reduce_matrix.shape[1]
        sensitivities = torch.linalg.vector_norm(self.h1.grad, dim=1, ord=norm_order)
        sort_order = torch.argsort(sensitivities)
        energy = sensitivities[sort_order]**energy_order
        cum_energy = torch.cumsum(energy, dim=0) / torch.sum(energy)
        n_drop = torch.sum(cum_energy < fraction)
        self.lifted_dim = n_current_neurons-n_drop
        idx = sort_order[n_drop:]

        self.psi_p = nn.Parameter(self.psi_p.data[idx, :])
        self.nx = nn.Parameter(self.nx.data[idx])
        self.dx = nn.Parameter(self.dx.data[idx])
        self.reduce_matrix = nn.Parameter(self.reduce_matrix.data[:, idx])
        print(f'Reduced from {n_current_neurons} to {self.lifted_dim} hidden neurons ({1-self.lifted_dim/(self.n_particles*self.n_nodes):.2%} total compression)')
        # print(f'Alpha values are {self.sigma1.alpha.data.item():.2e}, {self.sigma2.alpha.data.item():.2e}, {self.sigma3.alpha.data.item():.2e}')
        return optimizer_type(self.parameters(), **optimizer_kwargs)

def nonlinearity(x_diff, dx):
    x_ref = np.abs(x_diff)/dx
    conditions = [x_ref <= 0.5, x_ref < 1.5]
    values = [0.75-x_ref**2, 0.5*(1.5-x_ref)**2]
    return np.select(conditions, values, default=0)

def get_distance(x1, x2, L):
    return 0.5*L - np.abs(np.abs(x1 - x2) - 0.5*L)

def get_interpolation_matrix(psi_x, psi_v, psi_Q, n_particle_modes, n_node_modes, x_hat, node_positions):
    if len(x_hat.shape) == 1:
        x_hat = x_hat.reshape(1, -1)
    dx = node_positions[1] - node_positions[0]
    L = node_positions.shape[0]*dx
    n_batch = 100
    x = ((psi_x @ x_hat.T).T + L/2) % L - L/2
    n0 = x.shape[0]
    n1 = x.shape[1]
    y_batch = np.zeros((n0, n_particle_modes, n_node_modes))
    for i in range(n0//n_batch+1):
        x_batch = x.reshape(n0, n1, 1)[i*n_batch:(i+1)*n_batch]
        dist = get_distance(x_batch, node_positions, L)
        interpolation = nonlinearity(dist, dx)
        y_batch[i*n_batch:(i+1)*n_batch] = psi_v.T @ np.einsum('ijk,kl', interpolation, psi_Q)
    return y_batch.transpose(0, 2, 1)

def get_pe_from_interpolator(interpolator):
    pe = interpolator.permute(0, 2, 1) @ re_field_operator @ (interpolator @ rp_ones + rbg_vect)
    return(pe.squeeze())

if __name__ == "__main__":
    n_particle_modes = 50
    n_node_modes = 50
    n_cells = 50
    n_particles_per_cell = 50

    print('Loading data...')
    filename = f'debye_shielding_{n_cells}c{n_particles_per_cell}ppc'
    particle_snapshots = np.loadtxt(f'{filename}_particles.csv', dtype=float, delimiter=',')
    node_snapshots = np.loadtxt(f'{filename}_nodes.csv', dtype=float, delimiter=',')
    node_positions = np.loadtxt(f'{filename}_node_positions.csv', dtype=float, delimiter=',')

    n_particles = particle_snapshots.shape[0] // 3
    n_cells = node_positions.shape[0]
    dx = node_positions[1] - node_positions[0]
    L = n_cells*dx
    px = particle_snapshots[:n_particles]
    pv = particle_snapshots[n_particles:2*n_particles]
    pe = particle_snapshots[2*n_particles:]
    nq = node_snapshots[:n_cells]
    ne = node_snapshots[n_cells:]
    particle_ones = np.ones((pv.shape[0], 1))
    node_ones = np.ones((nq.shape[0], 1))

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
    bg_charge_density = background_charge_density(node_positions).reshape(-1, 1)
    weight_factor = L*n0 / (n_particles_per_cell*n_cells)

    print('Creating projection matrices...')
    psi_p = romtools.get_basis_for_all(n_particle_modes, px, pv, particle_ones)
    psi_qe = np.identity(n_cells) #romtools.get_basis_for_all(n_node_modes, nq-bg_charge_density, ne, bg_charge_density)

    rx = (psi_p.T @ px).T

    A = np.zeros((n_cells, n_cells))
    for i in range(n_cells):
        A[i, i-1] = 1
        A[i, i] = -2
        if i+1 < n_cells:
            A[i, i+1] = 1
        else:
            A[i, 0] = 1
    B = np.zeros((n_cells, n_cells))
    for i in range(n_cells-1):
        B[i, i+1] = 1
        B[i, i-1] = -1
    B[-1, 0] = 1
    B[-1, -2] = -1
    Q = 0.5 * B @ np.linalg.pinv(A) / constants.epsilon0
    re_field_operator = torch.from_numpy(psi_qe.T @ Q @ psi_qe)
    rp_ones = torch.from_numpy(psi_p.T @ particle_ones * constants.q_electron * weight_factor)
    rbg_vect = torch.from_numpy(psi_qe.T @ bg_charge_density*dx)

    print('Generating Targets...')
    interp = get_interpolation_matrix(psi_p, psi_p, psi_qe, n_particle_modes, n_node_modes, rx, node_positions)

    model = CompressionNN(node_positions=node_positions, psi_particles=psi_p, psi_nodes=psi_qe)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0)
    print(f'Number of free parameters: {sum(p.numel() for p in model.parameters())}')

    x = torch.from_numpy(rx)
    y = torch.from_numpy(interp)

    n_data = rx.shape[0]
    n_batches = 10
    train_num = int(0.8*n_data)
    test_num = n_data-train_num
    x_training, x_testing = random_split(x, [train_num, test_num])
    y_training, y_testing = y[x_training.indices], y[x_testing.indices]
    train_dataset = TensorDataset(x_training[:], y_training)
    test_dataset = (x_testing[:], y_testing)
    dl_training = DataLoader(train_dataset, batch_size=n_data//n_batches)

    def train(n_epochs: int, network: nn.Module, dl_training: DataLoader, test_dataset: tuple, optimizer: torch.optim.Adam, loss_fn=nn.MSELoss(),
              save_model_every_n_epochs: int=None):
        training_loss = np.zeros(n_epochs)
        testing_loss = np.zeros(n_epochs)
        n = len(dl_training)

        def closure():
            optimizer.zero_grad()
            x, y = data
            y_hat = network(x)
            loss = loss_fn(y, y_hat)
            network.h1.retain_grad()
            loss.backward()
            training_loss[i] += loss.item()

        for i in range(n_epochs):
            # Testing
            network.eval()
            with torch.no_grad():
                x, y_test = test_dataset
                y_hat_test = network(x)
                loss = loss_fn(y_test, y_hat_test)
                pe = get_pe_from_interpolator(y_test)
                pe_hat = get_pe_from_interpolator(y_hat_test)
                testing_loss[i] = loss.item()

            # Training
            network.train()
            for data in dl_training:
                optimizer.step(closure)
            training_loss[i] /= n

            # Print performance
            print(f'Epoch {i+1} Losses: Training = {training_loss[i]:.3g}, Testing = {testing_loss[i]:.3g}')
            rel_errors = romtools.rel_error(y_test, y_hat_test, axis=0)
            print(f'Testing Relative Errors in Matrix: Min = {rel_errors.min().item():.3%}, Mean = {rel_errors.mean().item():.3%}, Max = {rel_errors.max().item():.3%}')
            rel_errors = romtools.rel_error(pe, pe_hat, axis=0)
            print(f'Testing Relative Errors in Electric Field: Min = {rel_errors.min().item():.3%}, Mean = {rel_errors.mean().item():.3%}, Max = {rel_errors.max().item():.3%}')

            if save_model_every_n_epochs != None:
                if not (i+1)%save_model_every_n_epochs:
                    print('Saving Model...')
                    torch.save(network.state_dict(), f'{filename}_neural_net_{n_particle_modes}pm{n_node_modes}nm')
                    print('Saved')

            # Drop Neurons
            if training_loss[i] < 1e-4 and network.lifted_dim > n_particle_modes*n_node_modes:
                optimizer = network.drop_hidden_neurons(0.01, optimizer_type=torch.optim.AdamW, energy_order=1, lr=1e-6, weight_decay=1e-2)

        return training_loss, testing_loss

    print('Starting Training...')
    training_loss, testing_loss = train(20000, model, dl_training, test_dataset, optimizer, loss_fn=nn.MSELoss(), save_model_every_n_epochs=100)
    torch.save(model.state_dict(), f'{filename}_interp_net_{n_particle_modes}pm{n_node_modes}nm')

    plt.plot(training_loss, label='Training')
    plt.plot(testing_loss, label='Testing')
    plt.ylabel('MSE Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Losses')
    plt.yscale('log')
    plt.show()