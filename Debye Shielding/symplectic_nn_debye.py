import sys
sys.path.append('..')
import symplectic_nn as sympnet
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split

if __name__ == "__main__":
    n_cells = 40
    n_particles_per_cell = 20

    print('Loading data...')
    filename = f'debye_shielding_{n_cells}c{n_particles_per_cell}ppc'
    particle_snapshots = np.loadtxt(f'{filename}_particles.csv', dtype=float, delimiter=',')[:, 1:]

    n_particles = particle_snapshots.shape[0] // 2

    n_latent_particles = 20
    autoencoder_hidden_layers = [32, 32]
    flow_map_hidden_layers = [32, 32]
    activation = nn.SiLU()
    autoencoder = sympnet.SymplecticAutoencoder(n_particles, n_latent_particles, autoencoder_hidden_layers, henon_activation=activation, n_g_reflector_layers=10)
    flow_map = sympnet.HenonNet(n_latent_particles, flow_map_hidden_layers, activation)
    model = sympnet.SymplecticNN(autoencoder, flow_map)
    optimizer =  torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=np.sqrt(0.1), min_lr=1e-5)

    px = torch.tensor(particle_snapshots[:n_particles].T, dtype=torch.float, requires_grad=True)
    pv = torch.tensor(particle_snapshots[n_particles:].T, dtype=torch.float, requires_grad=True)

    print(px.shape)
    print(pv.shape)

    def train(n_epochs: int, model: sympnet.SymplecticNN, x, v, optimizer: torch.optim.Adam,
              scheduler=None, save_model_every_n_epochs: int=None):
        training_loss = np.zeros(n_epochs)
        model.train()
        n_data = x.shape[0]
        for i in range(n_epochs):
            optimizer.zero_grad()
            print('calculating output')
            output = model.train_forward(x, v, unroll_length=10)
            print('done')
            for i in range(n_data):
                x_hat, v_hat = output.get_predictions_at_time_idx(i)
                x_diff, v_diff = x_hat - x[i], v_hat - v[i]
                loss += torch.linalg.norm(x_diff) + torch.linalg.norm(v_diff)
            loss /= n_data
            training_loss[i] = loss
            loss.backward()
            optimizer.step(loss)

            if scheduler != None:
                scheduler.step(training_loss[i])
                lr = scheduler.get_last_lr()[0]
            else:
                lr = 1e-3
                
            print(f'Epoch {i+1} Losses: Training = {training_loss[i]:.3g}, Testing = {testing_loss[i]:.3g}, lr = {lr:.2e}')

            if save_model_every_n_epochs != None:
                if not (i+1)%save_model_every_n_epochs:
                    print('Saving Model...')
                    torch.save(model.state_dict(), f'{filename}_symplectic_nn_{n_cells}c{n_particles_per_cell}ppc')
                    print('Saved')
        return training_loss

    training_loss, testing_loss = train(5000, model, px, pv, optimizer,
                                        scheduler=scheduler, save_model_every_n_epochs=100)
    torch.save(model.state_dict(), f'{filename}_symplectic_nn_{n_cells}c{n_particles_per_cell}ppc')

    plt.plot(training_loss)
    plt.ylabel('MSE Loss')
    plt.xlabel('Epoch')
    plt.title('Losses')
    plt.yscale('log')
    plt.show()