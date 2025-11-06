import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
import nntools
import romtools

class NN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, activation=nn.ReLU(), use_batch_norm=False):
        super().__init__()

        layers = []
        layers.append(nn.Linear(input_size, hidden_layers[0]))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_layers[0]))
        layers.append(activation)
        n = len(hidden_layers)
        for i in range(n-1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_layers[i+1]))
            layers.append(activation)
        layers.append(nn.Linear(hidden_layers[-1], output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    n_cells = 80
    n_particles_per_cell = 200

    filename = f'two_stream_{n_cells}c{n_particles_per_cell}ppc'
    particle_snapshots = np.loadtxt(f'{filename}_particles.csv', dtype=float, delimiter=',')[:, 1:]
    node_snapshots = np.loadtxt(f'{filename}_nodes.csv', dtype=float, delimiter=',')[:, 1:]
    nt = particle_snapshots.shape[1]

    n_particles = particle_snapshots.shape[0] // 3
    px = particle_snapshots[:n_particles]
    pv = particle_snapshots[n_particles:2*n_particles]
    pe = particle_snapshots[2*n_particles:]

    nq = node_snapshots[:n_cells]
    ne = node_snapshots[n_cells:]

    n_node_modes = 40
    psi_q = np.linalg.svd(nq, full_matrices=False, compute_uv=True).U[:, :n_node_modes]
    print(romtools.proj_error(psi_q, nq))

    rq = psi_q.T @ nq

    nrq, meanq, stdq = nntools.normalize(rq)
    npx, meanx, stdx = nntools.normalize(px)

    x = torch.from_numpy(npx.T)
    y = torch.from_numpy(nrq.T)

    input_size = n_particles
    output_size = n_node_modes
    hidden_layers = [1000 for i in range(2)]
    network = NN(input_size, output_size, hidden_layers, activation=nn.SiLU() ,use_batch_norm=True).to(torch.double)
    n_param = sum(p.numel() for p in network.parameters() if p.requires_grad)
    optimizer =  torch.optim.Adam(network.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, min_lr=1e-5)

    n_batches = 10
    train_num = int(0.8*nt)
    test_num = nt-train_num
    x_training, x_testing = random_split(x, [train_num, test_num])
    y_training, y_testing = y[x_training.indices], y[x_testing.indices]
    train_dataset = TensorDataset(x_training[:], y_training)
    test_dataset = (x_testing[:], y_testing)
    dl_training = DataLoader(train_dataset, batch_size=nt//n_batches)

    def train(n_epochs: int, network: nn.Module, dl_training: DataLoader, test_dataset: tuple, optimizer: torch.optim.Adam, scheduler=None):
        loss_fn = nn.MSELoss()
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
                if i == n_epochs-1:
                    rel_error = (y_hat - y) / y
                    print(f'Mean Error: {torch.where(rel_error != torch.nan, torch.abs(rel_error), 0).mean().item():.2%}')
                    print(f'Max Error: {torch.where(rel_error != torch.nan, torch.abs(rel_error), 0).max().item():.2%}')

            if scheduler != None:
                scheduler.step(training_loss[i])
                lr = f'{scheduler.get_last_lr()[0]:.2e}'
            else:
                lr = 'N/A'
            print(f'Epoch {i+1} Losses: Training = {training_loss[i]:.3g}, Testing = {testing_loss[i]:.3g}, lr = {lr}')
        return training_loss, testing_loss

    training_loss, testing_loss = train(1000, network, dl_training, test_dataset, optimizer, scheduler)
    plt.plot(training_loss, label='Training')
    plt.plot(testing_loss, label='Testing')
    plt.ylabel('MSE Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Losses')
    plt.yscale('log')

    torch.save(network.state_dict(), f'{filename}_px_to_romq_net')
    np.savetxt(f'{filename}_px_to_romq_scaling', np.array([meanx, stdx, meanq, stdq]), header='meanx, stdx, meanq, stdq')

    plt.show()