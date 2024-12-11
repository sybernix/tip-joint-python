import numpy as np
import cv2
from skimage import img_as_float, metrics
from scipy.sparse.linalg import cg
from scipy.ndimage import rotate
from scipy.linalg import inv
from utils import prepare_psi, sinkhorn_knopp

# Load image
img = cv2.imread("../../img/lena_gray.png", cv2.IMREAD_GRAYSCALE)
img = img_as_float(img)

# Add Gaussian noise
noise_mean = 0
noise_variance = 0.04
noisy_img = img + np.random.normal(noise_mean, np.sqrt(noise_variance), img.shape)

# Parameters
sigma = 0.4
mu = 0.4
gamma = 0.5
kappa = 0.3
angle_degrees = 20

# Calculate padding
h, w = noisy_img.shape
max_disp = int(np.ceil(np.sqrt(h**2 + w**2) - max(h, w)))
img_padded = np.pad(noisy_img, ((max_disp, max_disp), (max_disp, max_disp)), mode='constant')

# Generate rotation maps
def get_xy_maps(original_image, angle_degrees):
    rows, cols = original_image.shape
    alpha = -np.radians(angle_degrees)  # Negative for image coordinate system
    R = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
    coords = np.linalg.inv(R) @ np.vstack([(X - cols / 2).ravel(), (Y - rows / 2).ravel()])
    x_rot = coords[0].reshape(rows, cols) + cols / 2
    y_rot = coords[1].reshape(rows, cols) + rows / 2
    return x_rot, y_rot

mapx, mapy = get_xy_maps(img_padded, angle_degrees)
mapx = mapx[max_disp:max_disp + h, max_disp:max_disp + w]
mapy = mapy[max_disp:max_disp + h, max_disp:max_disp + w]

# Gaussian kernel
def gaussian_filter_2d(size, sigma):
    kernel = np.zeros((size, size))
    offset = size // 2
    for x in range(size):
        for y in range(size):
            diff = ((x - offset)**2 + (y - offset)**2)
            kernel[x, y] = np.exp(-diff / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel


def calculate_graph_output(M, N, theta, psi_bar, p_h, p_w, mu, gamma, kappa, y):
    """
    Performs graph-based optimization to compute the output matrix.

    Parameters:
    M (int): Size of the input space (number of rows in H).
    N (int): Size of the output space (number of columns in G).
    theta (numpy.ndarray): The theta matrix (M x M).
    psi_bar (numpy.ndarray): The Psi matrix (p_h*p_w x p_h*p_w).
    p_h (int): Patch height.
    p_w (int): Patch width.
    mu (float): Regularization parameter for Psi.
    gamma (float): Graph Laplacian regularization parameter.
    kappa (float): Laplacian smoothing parameter.
    y (numpy.ndarray): Input data vector (M-dimensional).

    Returns:
    numpy.ndarray: Graph-based optimized output matrix (p_h x p_w).
    """
    # Step 1: Initialize matrices H and G
    H = np.zeros((M, M + N))
    np.fill_diagonal(H[:, :M], 1)

    G = np.zeros((N, M + N))
    np.fill_diagonal(G[:, M:], 1)

    # Step 2: Create a row-stochastic theta_square
    theta_square = np.random.rand(M, M)
    theta_square = theta_square / np.sum(theta_square, axis=1, keepdims=True)  # Row stochastic
    theta_square[:theta.shape[0], :theta.shape[1]] = theta

    # Step 3: Construct psi_bar_full
    psi_bar_full = np.zeros((M, M))
    psi_bar_full[:p_h * p_w, :p_h * p_w] = psi_bar

    random_component = np.random.rand(M - p_h * p_w, M - p_h * p_w)
    random_component = np.dot(random_component, random_component.T)  # Symmetric random component
    psi_bar_full[p_h * p_w:, p_h * p_w:] = random_component

    # Normalize using Sinkhorn-Knopp
    psi_bar_full = sinkhorn_knopp(psi_bar_full)

    # Step 4: Compute Lbar
    psi_inv = np.linalg.inv(psi_bar_full)
    Lbar = (psi_inv - np.eye(M)) / mu

    # Step 5: Compute Amn using CLIME inverse
    Amn = inv(theta_square)

    # Step 6: Build the A matrix
    A = np.zeros((M + N, M + N))
    A[:M, M:M + N] = Amn

    # Step 7: Construct the C matrix
    I = np.eye(M + N)
    C = np.dot(H.T, H) + gamma * np.dot((I - A).T, np.dot(H.T, np.dot(H, (I - A)))) + kappa * np.dot(G.T,
                                                                                                     np.dot(Lbar, G))

    # Step 8: Solve the system using conjugate gradient solver
    b = np.dot(H.T, y)
    x, _ = cg(C, b, tol=1e-6, maxiter=3000)

    # Step 9: Compute output matrix for the graph
    output_flat = x[M:M + p_h * p_w]
    output_mat_graph = output_flat.reshape((p_h, p_w))

    return output_mat_graph

kernel_halfwidth = 1
kernel_size = 2 * kernel_halfwidth + 1
g = gaussian_filter_2d(kernel_size, sigma)

# Image processing
p_h, p_w = 10, 10
seq_out = np.zeros((h, w))
graph_output = np.zeros((h, w))

for row in range(h // p_h):
    print(str(row) + "/" + str(h//p_h))
    psv = row * p_h
    pev = (row + 1) * p_h
    for col in range(w // p_w):
        psh = col * p_w
        peh = (col + 1) * p_w

        p_mapx = mapx[psv:pev, psh:peh]
        p_mapy = mapy[psv:pev, psh:peh]

        x_min = int(np.floor(p_mapx.min()))
        y_min = int(np.floor(p_mapy.min()))
        x_max = int(np.ceil(p_mapx.max()))
        y_max = int(np.ceil(p_mapy.max()))

        img_patch = img_padded[y_min:y_max, x_min:x_max]
        ip_h, ip_w = img_patch.shape

        # Construct theta
        theta = np.zeros((p_h * p_w, ip_h * ip_w))
        for i in range(p_h):
            for j in range(p_w):
                theta_r = i * p_w + j
                x = p_mapx[i, j] + 1 - x_min
                y = p_mapy[i, j] + 1 - y_min

                l = int(np.floor(x))
                t = int(np.floor(y))

                a = x - l
                b = y - t

                theta_lt_loc = (t - 1) * ip_w + l
                theta_rt_loc = theta_lt_loc + 1
                theta_lb_loc = t * ip_w + l
                theta_rb_loc = theta_lb_loc + 1

                if 0 <= theta_lt_loc < theta.shape[1]:
                    theta[theta_r, theta_lt_loc] = (1 - b) * (1 - a)
                if 0 <= theta_rt_loc < theta.shape[1]:
                    theta[theta_r, theta_rt_loc] = (1 - b) * a
                if 0 <= theta_lb_loc < theta.shape[1]:
                    theta[theta_r, theta_lb_loc] = b * (1 - a)
                if 0 <= theta_rb_loc < theta.shape[1]:
                    theta[theta_r, theta_rb_loc] = b * a

        y = img_patch.flatten()
        psi_bar = prepare_psi(g, p_h, p_w, kernel_halfwidth);
        output_flat = np.dot(np.dot(psi_bar, theta), y)
        output_mat = output_flat.reshape((p_h, p_w))
        seq_out[psv:pev, psh:peh] = output_mat
        M = len(theta[0])
        N = M
        graph_output[psv:pev, psh:peh] = output_mat
        # graph_output[psv:pev, psh:peh] = calculate_graph_output(M, N, theta, psi_bar, p_h, p_w, mu, gamma, kappa, y)

# Calculate metrics
groundtruth = rotate(img, angle_degrees, reshape=False, order=1)
graph_psnr = metrics.peak_signal_noise_ratio(graph_output, groundtruth)
seq_psnr = metrics.peak_signal_noise_ratio(seq_out, groundtruth)
psnr_gain = graph_psnr - seq_psnr

graph_ssim = metrics.structural_similarity(graph_output, groundtruth)
seq_ssim = metrics.structural_similarity(seq_out, groundtruth)
ssim_gain = graph_ssim - seq_ssim

print(f"PSNR Gain: {psnr_gain}")
print(f"SSIM Gain: {ssim_gain}")
