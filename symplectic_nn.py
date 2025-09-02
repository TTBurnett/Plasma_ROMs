from torch import nn
import torch

def embedding_operator(x, n):
    lifted_x = torch.zeros(n)
    lifted_x[:x.shape[0]] = x
    return lifted_x

class HenonMapping(nn.Module):
    def __init__(self, n_particles, n_hidden_neurons, activation):
        super().__init__()
        l1 = nn.Linear(n_particles, n_hidden_neurons)
        l2 = nn.Linear(n_hidden_neurons, 1)
        self.V = nn.Sequential(l1, activation, l2)
        self.eta = nn.Parameter(torch.randn(n_particles), requires_grad=True)
        self.n_particles = n_particles
    
    def forward(self, x, v):
        grad_v = torch.autograd.grad(self.V(v), v)[0]
        return v + self.eta, -x + grad_v
    
    def backward(self, y, w):
        v = y - self.eta
        grad_v = torch.autograd.grad(self.V(v), v)[0]
        return grad_v - w, v
    
class HenonLayer(nn.Module):
    def __init__(self, n_particles, n_hidden_neurons, activation, power=4):
        super().__init__()
        self.h_map = HenonMapping(n_particles, n_hidden_neurons, activation)
        self.power = power
        self.n_particles = n_particles
        
    def forward(self, x, v):
        for i in range(self.power):
            x, v = self.h_map.forward(x, v)
        return x, v
    
    def backward(self, y, w):
        for i in range(self.power):
            y, w = self.h_map.backward(y, w)
        return y, w
    
class HenonNet(nn.Module):
    def __init__(self, n_particles, hidden_layers, activation, power=4):
        super().__init__()
        self.layers = nn.ModuleList(HenonLayer(n_particles, n_neurons, activation, power) for n_neurons in hidden_layers)
        self.n_particles = n_particles
        
    def forward(self, x, v):
        for layer in self.layers:
            x, v = layer.forward(x, v)
        return x, v
    
    def backward(self, y, w):
        for layer in reversed(self.layers):
            y, w = layer.backward(y, w)
        return y, w
    
class GReflectorLayer(nn.Module):
    def __init__(self, n_particles):
        super().__init__()
        self.n_particles = n_particles
        self.eye = torch.eye(2*n_particles)
        self.u = nn.Parameter(torch.randn(2*n_particles, 1), requires_grad=True)
        self.beta = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.J = torch.zeros((2*n_particles, 2*n_particles))
        self.J[:n_particles, n_particles:] = torch.eye(n_particles)
        self.J[n_particles:, :n_particles] = -torch.eye(n_particles)
        
    def forward(self, x, v):
        X = torch.hstack((x, v)).T
        not_eye = self.beta * self.u @ self.u.T @ self.J / (self.u.T @ self.u)
        Y = (self.eye + not_eye) @ X
        return Y[:self.n_particles].T, Y[self.n_particles:].T
    
    def backward(self, y, w):
        Y = torch.hstack((y, w)).T
        not_eye = self.beta * self.u @ self.u.T @ self.J / (self.u.T @ self.u)
        X = (self.eye - not_eye) @ Y
        return X[:self.n_particles].T, X[self.n_particles:].T
    
class GReflector(nn.Module):
    def __init__(self, n_particles, n_layers):
        super().__init__()
        self.layers = nn.ModuleList(GReflectorLayer(n_particles) for i in range(n_layers))
        self.n_particles = n_particles
        
    def forward(self, x, v):
        for layer in self.layers:
            x, v = layer.forward(x, v)
        return x, v
    
    def backward(self, y, w):
        for layer in reversed(self.layers):
            y, w = layer.backward(y, w)
        return y, w
    
class SymplecticAutoencoder(nn.Module):
    def __init__(self, n_particles, n_latent_particles, henon_hidden_layers, henon_activation, n_g_reflector_layers):
        super().__init__()
        self.n_particles = n_particles
        self.n_latent_particles = n_latent_particles
        self.henon_net = HenonNet(n_particles, henon_hidden_layers, henon_activation)
        self.g_reflector = GReflector(n_particles, n_g_reflector_layers)
        
    def encode(self, x, v):
        y, w = self.g_reflector(*self.henon_net(x, v))
        return y[:self.n_latent_particles], w[:self.n_latent_particles]
    
    def decode(self, y, w):
        lifted_y, lifted_w = embedding_operator(y, self.n_particles), embedding_operator(w, self.n_particles)
        return self.henon_net.backward(*self.g_reflector.backward(lifted_y, lifted_w))
    
class SymplecticNN(nn.Module):
    class TrainingOutputs:
        def __init__(self, x_hat, v_hat):
            self.x_hat = x_hat
            self.v_hat = v_hat
            self.batch_size = x_hat.shape[0]
            self.trajectory_size = x_hat.shape[1]
            self.feature_size = x_hat.shape[2]
            
        def get_predictions_at_time_idx(self, time_idx):
            x = []
            v = []
            for i in range(self.trajectory_size):
                if time_idx-i >= self.batch_size: continue
                if time_idx-i < 0: continue
                x.append(self.x_hat[time_idx-i, i, :])
                v.append(self.v_hat[time_idx-i, i, :])
            return torch.row_stack(x), torch.row_stack(v)
    
    def __init__(self, autoencoder: SymplecticAutoencoder, flow_map: HenonNet):
        self.n_particles = autoencoder.n_particles
        self.autoencoder = autoencoder
        self.flow_map = flow_map
        
    def train_forward(self, x, v, unroll_length=5):
        output = torch.tensor([torch.hstack(self.eval_forward(xi, vi, trajectory_length=unroll_length)) for xi, vi in zip(x[:-unroll_length+1], v[:-unroll_length+1])])
        print(output.shape)
        x_hat = output[:, :, :self.n_particles]
        v_hat = output[:, :, self.n_particles:]
        print(x_hat.shape)
        print(v_hat.shape)
        return SymplecticNN.TrainingOutputs(x_hat, v_hat)
    
    def eval_forward(self, x0, v0, trajectory_length):
        x = torch.zeros((trajectory_length, x0.shape[0]))
        v = torch.zeros_like(x)
        y, w = self.autoencoder.encode(x0, v0)
        for i in range(trajectory_length):
            x[i], v[i] = self.autoencoder.decode(y, w)
            y, w = self.flow_map(y, w)
        return x, v