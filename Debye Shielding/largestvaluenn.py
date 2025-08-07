import sys
sys.path.append('..')
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt

class NN(nn.Module):
    def __init__(self, n_nodes, hidden_layers, activation=nn.ReLU()):
        super().__init__()

        layers = []
        layers.append(nn.Linear(1, hidden_layers[0]))
        layers.append(activation)
        n = len(hidden_layers)
        for i in range(n-1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(activation)
        layers.append(nn.Linear(hidden_layers[-1], n_nodes))
        layers.append(nn.Softmax(dim=1))
        self.nn = nn.Sequential(*layers)

    def forward(self, x):
        return self.nn(x)

def spline(x_ref, order: int = 1):
    match order:
        case 0:
            return np.where(x_ref < 1, 1 - x_ref, 0)
        case 1:
            conditions = [x_ref <= 0.5, x_ref < 1.5]
            values = [0.75-x_ref**2, 0.5*(1.5-x_ref)**2]
            return np.select(conditions, values, default=0)
        case _:
            raise ValueError(f'Invalid spline order {order}. [0, 1] are supported.')
        
def get_score(y, y_hat):
    true_max = torch.argmax(y, dim=1)
    guess_max = torch.argmax(y_hat, dim=1)
    return torch.sum(torch.where(true_max == guess_max, 1, 0)) / y.shape[0]

def get_avg_confidence(y_hat):
    max_confidence = torch.max(y_hat, dim=1).values
    return torch.mean(max_confidence)

if __name__ == "__main__":
    n_cells = 200
    n_particles_per_cell = 100

    print('Loading data...')
    filename = f'debye_shielding_{n_cells}c{n_particles_per_cell}ppc'
    particle_snapshots = np.loadtxt(f'{filename}_particles.csv', dtype=float, delimiter=',')
    node_positions = np.loadtxt(f'{filename}_node_positions.csv', dtype=float, delimiter=',')

    n_particles = particle_snapshots.shape[0] // 3
    dx = node_positions[1] - node_positions[0]
    L = n_cells*dx
    px = particle_snapshots[:n_particles]
    x_min = node_positions[0] - 0.5*dx

    def get_distance(px, nx):
        x = (px - x_min) % L + x_min
        return 0.5*L - np.abs(np.abs(x - nx) - 0.5*L)

    def get_max_idx(x):
        x_ref = get_distance(x, node_positions) / dx
        interpolations = spline(x_ref)
        max_idx = np.argmax(interpolations, axis=1)
        result = np.zeros_like(interpolations)
        result[np.arange(result.shape[0]), max_idx] = 1
        return result

    print('Seting up neural network...')
    hidden_layers = [20, 20, 20, 20]
    model = NN(n_cells, hidden_layers=hidden_layers, activation=nn.SiLU())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=np.sqrt(0.1))

    print('Generating targets...')
    rpx = np.random.choice(px.ravel(), size=200000, replace=False).reshape(-1, 1)
    x = torch.from_numpy(rpx).float()
    y = torch.from_numpy(get_max_idx(rpx)).float()

    n_data = x.shape[0]
    n_batches = 100
    train_num = int(0.8*n_data)
    test_num = n_data-train_num
    x_training, x_testing = random_split(x, [train_num, test_num])
    y_training, y_testing = y[x_training.indices], y[x_testing.indices]
    train_dataset = TensorDataset(x_training[:], y_training)
    test_dataset = (x_testing[:], y_testing)
    dl_training = DataLoader(train_dataset, batch_size=n_data//n_batches)

    def train(n_epochs: int, model: nn.Module, dl_training: DataLoader, test_dataset: tuple, optimizer: torch.optim.Adam,
              loss_fn=nn.MSELoss(), scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau=None, save_model_every_n_epochs: int=None):
        training_loss = np.zeros(n_epochs)
        testing_loss = np.zeros(n_epochs)
        n = len(dl_training)

        def closure():
            optimizer.zero_grad()
            x, y = data
            y_hat = model(x)
            loss = loss_fn(y, y_hat)
            loss.backward()
            training_loss[i] += loss.item()
            return loss

        for i in range(n_epochs):
            # Testing
            model.eval()
            with torch.no_grad():
                x, y = test_dataset
                y_hat = model(x)
                score = get_score(y, y_hat)
                avg_confidence = get_avg_confidence(y_hat)
                loss = loss_fn(y, y_hat)
                testing_loss[i] = loss.item()

            # Training
            model.train()
            for data in dl_training:
                optimizer.step(closure)
            training_loss[i] /= n

            # Print performance
            if scheduler != None:
                print(f'Epoch {i+1} Learning Rate = {scheduler.get_last_lr()[0]:.2e}')
            print(f'Epoch {i+1} Losses: Training = {training_loss[i]:.3g}, Testing = {testing_loss[i]:.3g}')
            print(f'Epoch {i+1} Testing Statistics: Accuracy = {score:.2%}, Average Confidence = {avg_confidence:.2%}')

            scheduler.step(training_loss[i])
            if save_model_every_n_epochs != None:
                if not (i+1)%save_model_every_n_epochs:
                    print('Saving Model...')
                    torch.save(model.state_dict(), f'{filename}_max_idx_nn')
                    print('Saved')

        return training_loss, testing_loss

    print('Starting Training...')
    training_loss, testing_loss = train(20000, model, dl_training, test_dataset, optimizer, loss_fn=nn.MSELoss(), scheduler=scheduler, save_model_every_n_epochs=100)
    torch.save(model.state_dict(), f'{filename}_max_idx_nn')

    plt.plot(training_loss, label='Training')
    plt.plot(testing_loss, label='Testing')
    plt.ylabel('MSE Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Losses')
    plt.yscale('log')
    plt.show()