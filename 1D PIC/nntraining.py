import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
import constants
import romtools
import nntools

class Activation(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class NN(nn.Module):
    def __init__(self, input_size, output_size: tuple, hidden_layers, use_layer_norm=True):
        super().__init__()

        layers = []
        layers.append(nn.Linear(input_size, hidden_layers[0]))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_layers[0]))
        layers.append(Activation())
        n = len(hidden_layers)
        for i in range(n-1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_layers[i+1]))
            layers.append(Activation())
        layers.append(nn.Linear(hidden_layers[-1], output_size[0]*output_size[1]))
        self.net = nn.Sequential(*layers)
        self.output_size = output_size

    def forward(self, x, nnx):
        return self.net(torch.cat((x, nnx.unsqueeze(0).expand(x.shape[0], -1)), dim=1)).reshape(x.shape[0], self.output_size[0], self.output_size[1])

def nonlinearity(x_diff, dx):
        x_ref = np.abs(x_diff)/dx
        conditions = [x_ref <= 0.5, x_ref < 1.5]
        values = [0.75-x_ref**2, 0.5*(1.5-x_ref)**2]
        return np.select(conditions, values, default=0)

def get_distance(x1, x2, L):
    return 0.5*L - np.abs(np.abs(x1 - x2) - 0.5*L)

def get_interpolation_matrix(psi_x, psi_v, psi_Q, n_particle_modes, n_node_modes, x_hat, node_positions):
    dx = node_positions[1] - node_positions[0]
    L = node_positions[-1] - node_positions[0] + dx
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
    return y_batch

def get_pe_from_interpolator(interpolator):
    interpolator = interpolator*scale_y + shift_y
    pe = interpolator @ re_field_operator @ (interpolator.permute(0, 2, 1) @ rp_ones + rbg_vect)
    return(pe.squeeze())

if __name__ == "__main__":
    n_particle_modes = 250
    n_node_modes = 50
    n_cells = 50
    n_particles_per_cell = 60

    print('Loading data...')
    filename = f'two_stream_{n_cells}c{n_particles_per_cell}ppc'
    particle_snapshots = np.loadtxt(f'{filename}_particles.csv', dtype=float, delimiter=',')[:, 1:]
    node_snapshots = np.loadtxt(f'{filename}_nodes.csv', dtype=float, delimiter=',')[:, 1:]
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
    bg_charge_density = -constants.q_electron * n0
    mean_v = 0.2*constants.c
    electron_plasma_frequency = np.sqrt(n0*constants.q_electron**2 / (constants.epsilon0 * constants.m_electron))
    inv_w_pe = 1/electron_plasma_frequency
    resonance = 2*np.pi*mean_v*inv_w_pe
    max_x = resonance * 1.25
    weight_factor = 2*max_x*n0 / (n_particles_per_cell*n_cells)

    print('Creating projection matrices...')
    psi_p = romtools.get_basis_for_all(n_particle_modes, px, pv, particle_ones)
    psi_qe = romtools.get_basis_for_all(n_node_modes, nq-bg_charge_density, ne, node_ones)

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
    rbg_vect = torch.from_numpy(psi_qe.T @ node_ones*bg_charge_density*dx)

    input_size = n_particle_modes + n_cells
    output_size = (n_particle_modes, n_node_modes)
    hidden_layers = [200, 200]
    network = NN(input_size, output_size, hidden_layers, use_layer_norm=False).to(torch.double)
    lr = 1e-2
    optimizer =  torch.optim.SGD(network.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=np.sqrt(0.1), min_lr=1e-4, patience=20, threshold=1e-5)

    print('Generating Targets...')
    interp = get_interpolation_matrix(psi_p, psi_p, psi_qe, n_particle_modes, n_node_modes, rx, node_positions)
    rpe = (psi_p.T @ pe).T
    
    nnx, _, _ = nntools.shift_to_zero_one(node_positions)
    nrx, shift_x, scale_x = nntools.shift_to_zero_one(rx)
    n_interp, shift_y, scale_y = nntools.shift_to_zero_one(interp)
    np.savetxt(f'{filename}_interp_net_scaling_{n_particle_modes}pm{n_node_modes}nm', np.array([shift_x, scale_x, shift_y, scale_y]), header='shift x, scale x, shift y, scale y')

    node_x = torch.from_numpy(nnx)
    x = torch.from_numpy(nrx)
    y = torch.from_numpy(n_interp)

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
              scheduler=None, save_model_every_n_epochs: int=None):
        training_loss = np.zeros(n_epochs)
        testing_loss = np.zeros(n_epochs)
        n = len(dl_training)
        pe_weight = 1e-10

        for i in range(n_epochs):
            # Training
            network.train()
            
            for data in dl_training:
                def closure():
                    optimizer.zero_grad()
                    x, y = data
                    y_hat = network(x, node_x)
                    loss = loss_fn(y, y_hat)
                    loss.backward()
                    training_loss[i] += loss.item()
                    return loss
                    
                optimizer.step(closure)
            training_loss[i] /= n

            # Testing
            network.eval()
            with torch.no_grad():
                x, y = test_dataset
                y_hat = network(x, node_x)
                pe = get_pe_from_interpolator(y)
                pe_hat = get_pe_from_interpolator(y_hat)
                loss = loss_fn(y, y_hat)
                testing_loss[i] = loss.item()

            if scheduler != None:
                scheduler.step(training_loss[i])
                lr = scheduler.get_last_lr()[0]
            else:
                lr = 1e-3
            print(f'Epoch {i+1} Losses: Training = {training_loss[i]:.3g}, Testing = {testing_loss[i]:.3g}, lr = {lr:.2e}')
            rel_errors = romtools.rel_error(y*scale_y + shift_y, y_hat*scale_y + shift_y, axis=0)
            print(f'Testing Relative Errors in Matrix: Mean = {rel_errors.mean().item():.3%}, Max = {rel_errors.max().item():.3%}')
            rel_errors = romtools.rel_error(pe, pe_hat, axis=0)
            print(f'Testing Relative Errors in Electric Field: Mean = {rel_errors.mean().item():.3%}, Max = {rel_errors.max().item():.3%}')

            if save_model_every_n_epochs != None:
                if not (i+1)%save_model_every_n_epochs:
                    print('Saving Model...')
                    torch.save(network.state_dict(), f'{filename}_interp_net_{n_particle_modes}pm{n_node_modes}nm')
                    print('Saved')
        return training_loss, testing_loss

    network.load_state_dict(torch.load(f'{filename}_interp_net_{n_particle_modes}pm{n_node_modes}nm', weights_only=True))
    training_loss, testing_loss = train(5000, network, dl_training, test_dataset, optimizer, loss_fn=nn.MSELoss(),
                                        scheduler=scheduler, save_model_every_n_epochs=100)
    torch.save(network.state_dict(), f'{filename}_interp_net_{n_particle_modes}pm{n_node_modes}nm')

    plt.plot(training_loss, label='Training')
    plt.plot(testing_loss, label='Testing')
    plt.ylabel('MSE Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Losses')
    plt.yscale('log')
    plt.show()