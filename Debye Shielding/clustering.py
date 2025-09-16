import sys
sys.path.append('..')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import sklearn.cluster as cluster
import romtools

def cosine_similarity_distance(X, Y):
    return np.abs(1 - np.abs(X @ Y.T.conj()))

def show_snapshots(x, v, idx_list, x_domain, fps=10, repeat=True):
        print('Generating animation...')
        fig, ax = plt.subplots(1, 1, figsize=(10, 5), sharex=True)
        v_domain = (np.min(v), np.max(v))

        def show(i):
            ax.clear()
            for j, idx in enumerate(idx_list):
                ax.scatter(x[idx, i], v[idx, i], alpha=0.4, label=f'Cluster {j+1}')
            ax.set_title(f'Idx = {i}')
            ax.set_ylabel('$v_x$ (m/s)')
            ax.set_ylim(v_domain)
            ax.set_xlim(x_domain)
            ax.set_xlabel('$x$ (m)')
            ax.legend()

        ani = animation.FuncAnimation(fig=fig, func=show, frames=range(x.shape[1]), interval=1e3/fps, repeat=repeat)
        plt.show()

if __name__ == "__main__":
    n_cells = 100
    n_particles_per_cell = 100
    problem = 'debye_shielding'
    
    print('Loading data...')
    filename = f'{problem}_{n_cells}c{n_particles_per_cell}ppc'
    particle_snapshots = np.loadtxt(f'{filename}_particles.csv', dtype=float, delimiter=',')
    node_positions = np.loadtxt(f'{filename}_node_positions.csv', dtype=float, delimiter=',')
    n_particles = particle_snapshots.shape[0] // 2
    dx = node_positions[1] - node_positions[0]
    x_domain = (node_positions[0] - 0.5*dx, node_positions[-1] + 0.5*dx)
    L = x_domain[-1] - x_domain[0]
    px = particle_snapshots[:n_particles]
    pv = particle_snapshots[n_particles:]
    U = px + pv*1j
    # U -= U[:, 0].reshape(-1, 1)
    U = U / np.linalg.vector_norm(U, axis=1, keepdims=True)
    x = (px - x_domain[0]) % L + x_domain[0]
    
    # print('Splitting in time...')
    # # nt = px.shape[1]
    # # window_size = 5
    # # tr = [np.trace(cosine_similarity_distance(U[:, i-window_size:i], U[:, i:i+window_size])) for i in range(window_size, nt-window_size, window_size)]
    # # plt.plot(tr)
    # # plt.show()
    # split_idx = (1+np.argmax(tr))*window_size
    # U = U[:, split_idx:]
    # x = x[:, split_idx:]
    # pv = pv[:, split_idx:]
    # print(split_idx)
    
    print('Performing clustering...')
    distances = cosine_similarity_distance(U, U)
    clu = cluster.AgglomerativeClustering(n_clusters=3, metric='precomputed', linkage='average')
    labels = clu.fit_predict(distances)
    idx = [np.where(labels == i)[0] for i in range(np.max(labels)+1)]
    
    show_snapshots(x, pv, idx, x_domain, fps=20)
    romtools.get_pod_basis(particle_snapshots, plot_svs=True, plot_projection_errors=True, error_step=10)
    for i in idx:
        romtools.get_pod_basis(px[i], plot_svs=True, plot_projection_errors=True, error_step=10)
        romtools.get_pod_basis(pv[i], plot_svs=True, plot_projection_errors=True, error_step=10)