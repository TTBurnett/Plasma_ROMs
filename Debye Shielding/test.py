import numpy as np

nidx = np.arange(10) * 10
print(np.repeat(nidx, 3) + np.tile([-1, 0, 1], 10))