import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
import constants
import romtools
import nntools

class NN(nn.Module):
    def __init__(self, input_size, output_size: tuple, hidden_layers, activation, 
                 re_field_operator, rp_ones, rbg_vect,
                 use_layer_norm=True):
        super().__init__()

        layers = []
        layers.append(nn.Linear(input_size, hidden_layers[0]))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_layers[0]))
        layers.append(activation)
        n = len(hidden_layers)
        for i in range(n-1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_layers[i+1]))
            layers.append(activation)
        layers.append(nn.Linear(hidden_layers[-1], output_size[0]*output_size[1]))
        self.net = nn.Sequential(*layers)
        self.output_size = output_size

        self.re_field_operator = torch.from_numpy(re_field_operator)
        self.rp_ones = torch.from_numpy(rp_ones)
        self.rbg_vect = torch.from_numpy(rbg_vect)

    def forward(self, x):
        interp = self.net(x).reshape(x.shape[0], self.output_size[0], self.output_size[1])
        pe_hat = interp @ self.re_field_operator @ (interp.permute(0, 2, 1) @ self.rp_ones + self.rbg_vect)
        return pe_hat.squeeze()

if __name__ == "__main__":
    n_particle_modes = 250
    n_node_modes = 60
    n_cells = 60
    n_particles_per_cell = 100

    print('Loading data...')
    filename = f'two_stream_{n_cells}c{n_particles_per_cell}ppc_circle'
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
    rpe = (psi_p.T @ pe).T

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
    re_field_operator = psi_qe.T @ Q @ psi_qe
    rp_ones = psi_p.T @ particle_ones * constants.q_electron * weight_factor
    rbg_vect = psi_qe.T @ node_ones*bg_charge_density*dx

    input_size = n_particle_modes
    output_size = (n_particle_modes, n_node_modes)
    hidden_layers = [500, 500]
    network = NN(input_size, output_size, hidden_layers, nn.SiLU(),
                 re_field_operator, rp_ones, rbg_vect).to(torch.double)
    lr = 1e-2
    optimizer =  torch.optim.Adam(network.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=np.sqrt(0.1), min_lr=1e-6)

    print('Generating Targets...')
    nrx, shift_x, scale_x = nntools.shift_to_zero_one(rx)
    nrpe, mean_y, std_y = nntools.normalize(rpe)
    np.savetxt(f'{filename}_pe_interp_net_scaling_{n_particle_modes}pm{n_node_modes}nm', np.array([shift_x, scale_x, mean_y, std_y]), header='shift x, scale x, shift y, scale y')

    x = torch.from_numpy(nrx)
    y = torch.from_numpy(nrpe)

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

        for i in range(n_epochs):
            # Training
            network.train()
            
            for data in dl_training:
                def closure():
                    optimizer.zero_grad()
                    x, y = data
                    y_hat = network(x)
                    loss = loss_fn(y_hat, y)
                    loss.backward()
                    training_loss[i] += loss.item()
                    return loss
                    
                optimizer.step(closure)
            training_loss[i] /= n

            # Testing
            network.eval()
            with torch.no_grad():
                x, y = test_dataset
                y_hat = network(x)
                loss = loss_fn(y_hat, y)
                testing_loss[i] = loss.item()
                rel_errors = romtools.rel_error(y*std_y + mean_y, y_hat*std_y + mean_y, axis=0)

            if scheduler != None:
                scheduler.step(training_loss[i])
                lr = scheduler.get_last_lr()[0]
            else:
                lr = 1e-3
            print(f'Epoch {i+1} Losses: Training = {training_loss[i]:.3g}, Testing = {testing_loss[i]:.3g}, lr = {lr:.2e}')
            print(f'Testing Relative Errors: Mean = {rel_errors.mean().item():.3%}, Max = {rel_errors.max().item():.3%}')

            if save_model_every_n_epochs != None:
                if not (i+1)%save_model_every_n_epochs:
                    print('Saving Model...')
                    torch.save(network.state_dict(), f'{filename}_pe_interp_net_{n_particle_modes}pm{n_node_modes}nm')
                    print('Saved')
        return training_loss, testing_loss

    training_loss, testing_loss = train(5000, network, dl_training, test_dataset, optimizer, loss_fn=nn.L1Loss(),
                                        scheduler=scheduler, save_model_every_n_epochs=100)
    torch.save(network.state_dict(), f'{filename}_pe_interp_net_{n_particle_modes}pm{n_node_modes}nm')

    plt.plot(training_loss, label='Training')
    plt.plot(testing_loss, label='Testing')
    plt.ylabel('MSE Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Losses')
    plt.yscale('log')
    plt.show()