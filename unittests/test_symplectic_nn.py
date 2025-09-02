import pytest
import sys
sys.path.append('.')
from symplectic_nn import HenonNet, embedding_operator, GReflector, SymplecticAutoencoder
import torch
from torch import nn

def test_henon_net_invertibility():
    n_particles = 10
    hidden_layers = [5, 10, 5]
    hnet = HenonNet(n_particles, hidden_layers, nn.SiLU())
    xi = torch.randn(n_particles, requires_grad=True)
    vi = torch.randn(n_particles, requires_grad=True)
    print(xi)
    print(vi)
    y, w = hnet(xi, vi)
    assert not torch.allclose(xi, y)
    assert not torch.allclose(vi, w)
    xf, vf = hnet.backward(y, w)
    print(xf)
    print(vf)
    assert torch.allclose(xf, xi, atol=1e-7)
    assert torch.allclose(vf, vi, atol=1e-7)
    
def test_henon_net_parameters_are_found():
    n_particles = 10
    hidden_layers = [5, 10, 5]
    hnet = HenonNet(n_particles, hidden_layers, nn.SiLU())
    n_parameters = sum(p.numel() for p in hnet.parameters())
    print([p for p in hnet.parameters()])
    print('# of parameters:', n_parameters)
    assert n_parameters == 273
    
def test_inclusion_matrix():
    n = 100
    r = 20
    x = torch.randn(r)
    y = embedding_operator(x, n)
    assert y.shape[0] == n
    assert torch.allclose(y[:r], x)
    
def test_g_reflector_invertibility():
    n_particles = 100
    g_reflector = GReflector(n_particles, n_layers=10)
    xi = torch.randn(n_particles, requires_grad=True)
    vi = torch.randn(n_particles, requires_grad=True)
    y, w = g_reflector(xi, vi)
    print(xi)
    print(vi)
    assert not torch.allclose(xi, y)
    assert not torch.allclose(vi, w)
    xf, vf = g_reflector.backward(y, w)
    print(xf)
    print(vf)
    assert torch.allclose(xf, xi, atol=1e-5)
    assert torch.allclose(vf, vi, atol=1e-5)
    
def test_g_reflector_parameters_are_found():
    n_particles = 100
    n_layers = 10
    g_reflector = GReflector(n_particles, n_layers)
    n_parameters = sum(p.numel() for p in g_reflector.parameters())
    print('# of parameters:', n_parameters)
    assert n_parameters == (2*n_particles + 1)*n_layers
    
def test_g_reflector_invertibility_with_batches():
    batch_size=10
    n_particles = 100
    g_reflector = GReflector(n_particles, n_layers=10)
    xi = torch.randn(batch_size, n_particles, requires_grad=True)
    vi = torch.randn(batch_size, n_particles, requires_grad=True)
    y, w = g_reflector(xi, vi)
    print(xi)
    print(vi)
    assert not torch.allclose(xi, y)
    assert not torch.allclose(vi, w)
    xf, vf = g_reflector.backward(y, w)
    print(xf)
    print(vf)
    assert torch.allclose(xf, xi, atol=1e-5)
    assert torch.allclose(vf, vi, atol=1e-5)
    
def test_symplectic_autoencoder():
    n_particles = 100
    n_latent_particles = 10
    autoencoder = SymplecticAutoencoder(n_particles, n_latent_particles, henon_hidden_layers=[50, 60, 50],
                                        henon_activation=nn.SiLU(), n_g_reflector_layers=5)
    xi = torch.randn(n_particles, requires_grad=True)
    vi = torch.randn(n_particles, requires_grad=True)
    print(xi)
    print(vi)
    y, w = autoencoder.encode(xi, vi)
    print(y)
    print(w)
    assert y.shape[0] == n_latent_particles
    assert w.shape[0] == n_latent_particles
    xf, vf = autoencoder.decode(y, w)
    print(xf)
    print(vf)
    assert xf.shape[0] == n_particles
    assert vf.shape[0] == n_particles