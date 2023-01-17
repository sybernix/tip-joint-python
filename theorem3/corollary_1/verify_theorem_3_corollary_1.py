import matplotlib.pyplot as plt
import numpy as np
from separable_graph_filter import graph_denoise_interpolate
from avg_filter_noisy_img import avg_filter
from theorem3.corollary_1.filter_img import filter_img
from theorem3.corollary_1.linear_interpolation import interpolateLinear
from utils import get_project_root

img_path = str(get_project_root()) + "/img/noisy_lena.png"
img = plt.imread(img_path)
plt.imshow(img, cmap="gray")
plt.show()

psi = np.array([[0.312816957047989, 0.247249820875324, 0.260255415738962, 0.179677806337725],
                    [0.247249820875324, 0.247335656093526, 0.278941892119199, 0.226472630911952],
                    [0.260255415738962, 0.278941892119199, 0.201335672588002, 0.259467019553837],
                    [0.179677806337726, 0.226472630911952, 0.259467019553837, 0.334382543196486]])
theta = np.array([[1 / 2, 1 / 2, 0, 0], [0, 0, 1 / 2, 1 / 2]])

graph_filtered_img = graph_denoise_interpolate(img, psi, theta)
plt.imshow(graph_filtered_img, cmap="gray")
plt.show()

# average filter and then linear interpolate
# avg_filtered_img = avg_filter(img)
filtered_img = filter_img(img, psi)
plt.imshow(filtered_img, cmap="gray")
plt.show()

linear_interpolated_avg_filtered_img = interpolateLinear(filtered_img, theta)
plt.imshow(linear_interpolated_avg_filtered_img, cmap="gray")
plt.show()

difference = graph_filtered_img - linear_interpolated_avg_filtered_img
plt.imshow(difference, cmap="gray")
plt.show()

ssd1 = np.sum(difference ** 2)
print(ssd1)

# linear interpolate and then average filter
# not relevant to theorem. no need
# linear_interpolated_img = interpolateLinear(img)
# plt.imshow(linear_interpolated_img, cmap="gray")
# plt.show()
#
# avg_filtered_interpolated_img = avg_filter(linear_interpolated_img)
# plt.imshow(avg_filtered_interpolated_img, cmap="gray")
# plt.show()
#
# ssd2 = np.sum((graph_filtered_img - avg_filtered_interpolated_img) ** 2)
# print(ssd2)

print("hi")
