import numpy as np

class PdfSampler:
    '''
    A class used for randomly sampling from a pdf.
    '''
    def sample(pdf: callable, ranges: list[tuple], n_samples: int, seed=None) -> np.typing.NDArray[np.float64]:
        '''
        Args:
            `pdf`: the probability density function to sample from.
            `ranges`: a list of tuples containing the minimum and maximum value to sample from
            for each dimension of the pdf.
            `n_samples`: the number of samples to pull from the distribution.
        '''
        if seed != None:
            np.random.seed(seed)
        idx = 0
        n_dims = len(ranges)
        samples = np.zeros((n_dims, n_samples))

        coords = [np.linspace(r[0], r[1], n_samples) for r in ranges]
        grid = np.meshgrid(*coords)
        max_value = np.max(pdf(*grid))

        while idx < n_samples:
            # Sample uniformly from the bounding distribution
            sample = np.array([np.random.uniform(r[0], r[1]) for r in ranges])
            y_sample = np.random.uniform(0, max_value)

            # Check if the sample is accepted
            if y_sample < pdf(*sample):
                samples[:, idx] = sample
                idx += 1

        return samples

# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Define a 2D Gaussian PDF
    def gaussian(x, y):
        return np.exp(-0.5 * (x**2 + y**2)) / (2 * np.pi)
    
    def tophat(x, y):
        return np.ones_like(x)
    
    def sin(x, y):
        return 1 + np.sin((x + y)*np.pi)
    
    def gaussian_torus(x, y):
        return np.exp(-2*(np.sqrt(x**2+y**2) - 3)**2)

    # Sample points from the PDF
    x_samples, y_samples = PdfSampler.sample(sin, ranges=[(-5, 5), (-5, 5)], n_samples=10000)

    # Plot the sampled points
    plt.figure(figsize=(8, 8))
    plt.scatter(x_samples, y_samples, alpha=0.5)
    plt.title('Sampled Points from 2D PDF using CDF')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid()
    plt.show()

    plt.hist(x_samples, bins=np.linspace(0, 5, 50))
    plt.show()
