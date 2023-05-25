import matplotlib.pyplot as plt
import numpy as np
from graph_interpolator import interpolateGraphFilter
from linear_interpolation import interpolateLinear
import cv2
from skimage.metrics import structural_similarity as ssim

img_path = '/Users/niruhan/Documents/source_codes/tip-joint-python/img/lena_gray_half.png'
groundtruth_path = '/Users/niruhan/Documents/source_codes/tip-joint-python/img/lena_gray.png'

img = plt.imread(img_path)
plt.imshow(img, cmap="gray")
plt.show()

graph_interpolated_img = interpolateGraphFilter(img)
plt.imshow(graph_interpolated_img, cmap="gray")
plt.show()

linear_interpolated_img = interpolateLinear(img)
plt.imshow(linear_interpolated_img, cmap="gray")
plt.show()

exp_dir = "/Users/niruhan/Documents/source_codes/tip-joint-python/exp_outs/6/"
plt.imsave(exp_dir + "linear_out.png", linear_interpolated_img, cmap="gray")
plt.imsave(exp_dir + "graph_out.png", graph_interpolated_img, cmap="gray")

ssd = np.sum((graph_interpolated_img - linear_interpolated_img) ** 2)

print(ssd)

groundtruth = plt.imread(groundtruth_path)
groundtruth = groundtruth.astype(np.float64)[:,:511]
psnr_linear = cv2.PSNR(linear_interpolated_img, groundtruth)
print("psnr linear = " + str(psnr_linear))

psnr_graph = cv2.PSNR(graph_interpolated_img, groundtruth)
print("psnr graph = " + str(psnr_graph))

data_range_l = linear_interpolated_img.max() - linear_interpolated_img.min()
ssim_linear = ssim(linear_interpolated_img, groundtruth, data_range=data_range_l)
print("ssim linear = " + str(ssim_linear))

data_range_g = graph_interpolated_img.max() - graph_interpolated_img.min()
ssim_graph = ssim(graph_interpolated_img, groundtruth, data_range=data_range_g)
print("ssim graph = " + str(ssim_graph))


