import numpy as np
import ot

def gaussianFilter(kernel_size, sigma):
    if kernel_size % 2 == 0:
        kernel_halfwidth = kernel_size // 2
    else:
        kernel_halfwidth = (kernel_size - 1) // 2

    Y, X = np.meshgrid(np.arange(-kernel_halfwidth, kernel_halfwidth + 1),
                       np.arange(-kernel_halfwidth, kernel_halfwidth + 1))

    filter_ = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
    gaussian_filter = filter_ / np.sum(filter_)

    return gaussian_filter


def preparePsi(filter_, patch_height, patch_width, kernel_halfwidth):
    psi = np.zeros((patch_height * patch_width, patch_height * patch_width))

    for row in range(patch_height):
        for col in range(patch_width):
            temp_patch = np.zeros((patch_height + 2 * kernel_halfwidth, patch_width + 2 * kernel_halfwidth))
            temp_patch[row: row + 2 * kernel_halfwidth + 1, col:col + 2 * kernel_halfwidth + 1] = filter_

            extracted_patch = temp_patch[kernel_halfwidth: kernel_halfwidth + patch_height,
                              kernel_halfwidth: kernel_halfwidth + patch_width]

            psi[row * patch_width + col, :] = extracted_patch.T.ravel()

    psi += 1e-8
    psi = ot.sinkhorn(a=np.ones(psi.shape[0]), b=np.ones(psi.shape[1]), M=psi, reg=1e-3)

    return psi
