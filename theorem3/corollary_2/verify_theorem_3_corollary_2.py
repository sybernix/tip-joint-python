import matplotlib.pyplot as plt
import numpy as np
from theorem2.linear_interpolation import interpolateLinear
from utils import get_project_root

img_path = str(get_project_root()) + "/img/noisy_lena.png"
img = plt.imread(img_path)
plt.imshow(img, cmap="gray")
plt.show()

m = 4
n = 2
gamma = 0.4
kappa = 0.3

H = np.concatenate((np.identity(m), np.zeros((m, n))), axis=1)
HT = H.transpose()

G = np.concatenate((np.zeros((n, m)), np.identity(n)), axis=1)
GT = G.transpose()

# initialize denoiser
psi = np.full((m, m), 1 / m)
mu = 0.5
laplacian = (1 / mu) * (np.linalg.pinv(psi) - (1 + gamma) * np.identity(m))

combined_laplacian = mu * np.matmul(np.matmul(HT, laplacian), H) + kappa *
