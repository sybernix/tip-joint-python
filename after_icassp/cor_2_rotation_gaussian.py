import numpy as np
import matplotlib.pyplot as plt
import ot
from skimage import io
from scipy.sparse.linalg import cg
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from scipy.ndimage import rotate
from skimage.util import img_as_float

from after_icassp.utils import gaussianFilter, preparePsi

noise_mean = 0
noise_variance = 0.04
sigma = 0.4
mu = 0.4
gamma = 0.5
kappa = 0.3
angle_degrees = 20

def getXYMaps(originalImage, angle_degrees):
    rows, cols = originalImage.shape
    alpha = -np.deg2rad(angle_degrees)

    R = np.array([[np.cos(alpha), -np.sin(alpha)],
                  [np.sin(alpha), np.cos(alpha)]])

    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
    coords = np.linalg.inv(R) @ np.vstack([(X - cols / 2).ravel(), (Y - rows / 2).ravel()])

    x_rot = coords[0].reshape(rows, cols) + cols / 2
    y_rot = coords[1].reshape(rows, cols) + rows / 2

    return x_rot, y_rot


# Main Code
img = io.imread("../img/peppers_gray.png")
img = img_as_float(img)
noisy_img = img + np.random.normal(noise_mean, np.sqrt(noise_variance), img.shape)

h, w = noisy_img.shape

max_disp = int(np.ceil(np.sqrt(h ** 2 + w ** 2) - max([h, w])))
noisy_img_padded = np.pad(noisy_img, ((max_disp, max_disp), (max_disp, max_disp)))

mapx, mapy = getXYMaps(noisy_img_padded, angle_degrees)
mapx = mapx[max_disp:max_disp + h, max_disp:max_disp + w]
mapy = mapy[max_disp:max_disp + h, max_disp:max_disp + w]

seq_out = np.zeros((h, w))
graph_output = np.zeros((h, w))

kernel_halfwidth = 1
kernel_size = 2 * kernel_halfwidth + 1
g = gaussianFilter(kernel_size, sigma)

p_h = 5
p_w = 5

for row in range(50, h - (h % p_h), p_h):
    psv = row
    pev = row + p_h

    for col in range(50, w - (w % p_w), p_w):
        psh = col
        peh = col + p_w

        p_mapx = mapx[psv: pev, psh: peh]
        p_mapy = mapy[psv: pev, psh: peh]

        x_min = int(np.floor(p_mapx.min()))
        y_min = int(np.floor(p_mapy.min()))

        x_max = int(np.ceil(p_mapx.max()))
        y_max = int(np.ceil(p_mapy.max()))

        img_patch = noisy_img_padded[y_min:y_max, x_min:x_max]
        ip_h, ip_w = img_patch.shape

        theta = np.zeros((p_h * p_w, ip_h * ip_w))

        for i in range(p_h):
            for j in range(p_w):
                theta_r = i * p_w + j

                x = p_mapx[i, j]  - x_min
                y = p_mapy[i, j] - y_min

                l = int(np.floor(x))
                t = int(np.floor(y))

                a = x - l
                b = y - t

                theta_lt_loc = (t - 1) * ip_w + l
                theta_rt_loc = theta_lt_loc + 1

                theta_lb_loc = t * ip_w + l
                theta_rb_loc = theta_lb_loc + 1

                theta[theta_r, theta_lt_loc] = (1 - b) * (1 - a)
                theta[theta_r, theta_rt_loc] = (1 - b) * a
                theta[theta_r, theta_lb_loc] = b * (1 - a)
                theta[theta_r, theta_rb_loc] = b * a

        y_flat = img_patch.T.ravel()
        M = theta.shape[1]
        N = M

        psi = preparePsi(g, y_max - y_min, x_max - x_min, kernel_halfwidth)
        psi = ot.sinkhorn(a=np.ones(psi.shape[0]), b=np.ones(psi.shape[1]), M=psi, reg=1e-3)
        psi_bar = preparePsi(g, p_h, p_w, kernel_halfwidth)
        psi_bar = ot.sinkhorn(a=np.ones(psi_bar.shape[0]), b=np.ones(psi_bar.shape[1]), M=psi_bar, reg=0.1)

        output_flat = np.dot(np.dot(psi_bar, theta), y_flat)
        output_mat = output_flat.reshape(p_h, p_w)
        seq_out[psv:pev, psh:peh] = output_mat

        # Calculate graph output
        H = np.zeros((M, M + N))
        np.fill_diagonal(H[:, :M], 1)

        G = np.zeros((N, M + N))
        np.fill_diagonal(G[:, M:], 1)

        theta_square = np.random.rand(M, M)
        theta_square /= theta_square.sum(axis=1, keepdims=True)
        theta_square[:theta.shape[0], :] = theta

        psi_bar_full = np.zeros((M, M))
        psi_bar_full[:p_h * p_w, :p_h * p_w] = psi_bar
        rans_psi_comp = np.random.rand(M - p_h * p_w, M - p_h * p_w)
        rans_psi_comp = rans_psi_comp @ rans_psi_comp.T
        rans_psi_comp = ot.sinkhorn(a=np.ones(rans_psi_comp.shape[0]), b=np.ones(rans_psi_comp.shape[1]), M=rans_psi_comp, reg=0.1)
        psi_bar_full[p_h * p_w:, p_h * p_w:] = rans_psi_comp

        # psi_bar_full = ot.sinkhorn(a=np.ones(psi_bar_full.shape[0]), b=np.ones(psi_bar_full.shape[1]), M=psi_bar_full, reg=1e-3)

        L = (np.linalg.inv(psi) - np.eye(M)) / mu
        Lbar = (np.linalg.inv(psi_bar_full) - np.eye(N)) / kappa

        Amn = np.linalg.inv(theta_square)
        A = np.zeros((M + N, M + N))
        A[:M, M:] = Amn

        I = np.eye(M + N)
        L_comb = mu * H.T @ L @ H + kappa * G.T @ Lbar @ G
        C = H.T @ H + gamma * (I - A).T @ H.T @ H @ (I - A) + L_comb
        C_inv = np.linalg.inv(C)
        # x, _ = cg(C, H.T @ y_flat, tol=1e-6, maxiter=3000)
        x_inv = C_inv @ H.T @ y_flat
        output_mat_graph = x_inv[M:M + p_h * p_w].reshape(p_h, p_w)
        graph_output[psv:pev, psh:peh] = output_mat_graph

plt.imshow(graph_output, cmap='gray')
plt.show()

groundtruth = rotate(img, angle_degrees, mode='nearest', reshape=False)
graph_psnr_value = psnr(graph_output, groundtruth)
seq_psnr_value = psnr(seq_out, groundtruth)

psnr_gain = graph_psnr_value - seq_psnr_value

graph_ssim_value = ssim(graph_output, groundtruth)
seq_ssim_value = ssim(seq_out, groundtruth)

ssim_gain = graph_ssim_value - seq_ssim_value


