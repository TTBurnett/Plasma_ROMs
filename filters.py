import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from scipy import fft
from scipy.special import binom, factorial

class SplineFilter:
    def __init__(self, order: int, width: float = 1.0, offset: float = 0.0, coefficient: float = 1.0):
        self.spline = BSpline.basis_element(np.arange(order+1)-0.5*(order), extrapolate=False)
        self.width = width
        self.offset = offset
        self.coefficient = coefficient
    
    def __call__(self, x):
        return np.nan_to_num(self.coefficient * self.spline((x - self.offset) / self.width))

class SiacFilter:
    def __init__(self, n_moments: int, b_spline_order: int, width: float = 1.0):
        '''
        n_moments: The number of moments the filter should perfectly reproduce. Must be an even number.
        b_spline_order: The order of the b-spline. This will control how quickly high frequencies decay.
        '''
        RS = n_moments // 2
        numspline = n_moments + 1

        # Define matrix to determine kernel coefficients
        A=np.zeros((numspline,numspline))
        for m in np.arange(numspline):
            for gamma in np.arange(numspline):
                component = 0.
                for n in np.arange(m+1):
                    jsum = sum((-1)**(j+b_spline_order-1)*binom(b_spline_order-1,j)*((j-0.5*(b_spline_order-2))**(b_spline_order+n)-(j-0.5*b_spline_order)**(b_spline_order+n)) for j in np.arange(b_spline_order))
                    component += binom(m,n)*(gamma-RS)**(m-n)*factorial(n)/factorial(n+b_spline_order)*jsum
                    A[m, gamma] = component
        b = np.zeros(numspline)
        b[0] = 1
        c = np.linalg.solve(A, b)
        self.splines = [SplineFilter(b_spline_order, width=width, offset=(i-RS)*width, coefficient=coeff) for i, coeff in enumerate(c)]
    
    def __call__(self, x):
        return sum(spline(x) for spline in self.splines)

if __name__ == "__main__":
    plt.rcParams['figure.figsize'] = [10, 5]
    
    n = 1001
    min_x, max_x = -5, 5
    x = np.linspace(min_x, max_x, n)
    dx = x[1] - x[0]
    
    fig, ax = plt.subplots(2, 2)
    width = 0.1
    orders = [2, 3]
    for i, order in enumerate(orders):
        spline = SplineFilter(order=order, width=width)
        y_spline = spline(x)
        siac_filter = SiacFilter(6, order, width)
        y_siac = siac_filter(x)
        ax[0, i].plot(x, y_spline, 'k', label='Spline')
        ax[0, i].plot(x, y_siac, 'k--', label='SIAC')
        ax[0, i].set_xlim([-0.5, 0.5])
        ax[0, i].legend()
        ax[0, i].set_title(f'{order} Order')
        
        yf_spline = fft.rfft(y_spline)
        yf_siac = fft.rfft(y_siac)
        freq = fft.rfftfreq(n, dx)
        ax[1, i].plot(freq, np.abs(yf_spline) / np.abs(yf_spline).max(), 'k', label='Spline')
        ax[1, i].plot(-freq, np.abs(yf_spline) / np.abs(yf_spline).max(), 'k')
        ax[1, i].plot(freq, np.abs(yf_siac) / np.abs(yf_siac).max(), 'k--', label='SIAC')
        ax[1, i].plot(-freq, np.abs(yf_siac) / np.abs(yf_siac).max(), 'k--')
        ax[1, i].set_xlim([-5/width, 5/width])
        ax[1, i].set_title(f'{order} Order')
        ax[1, i].legend()
    plt.show()
    
    siac_filter = SiacFilter(8, 3, width)
    test_x = np.arange(min_x, max_x, width)
    test_x += np.random.randn(1)
    print(np.sum(siac_filter(test_x)))
    
    y = siac_filter(x)
    plt.plot(x, y)
    plt.show()
    yf = fft.rfft(y)
    plt.plot(freq, np.abs(yf) / np.abs(yf).max())
    plt.show()