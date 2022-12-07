import matplotlib.pyplot as plt
import numpy as np
from separable_graph_filter import graph_denoise_interpolate
from avg_filter_noisy_img import avg_filter
from theorem2.linear_interpolation import interpolateLinear
from utils import get_project_root

img_path = str(get_project_root()) + "/img/noisy_lena.png"
img = plt.imread(img_path)
plt.imshow(img, cmap="gray")
plt.show()

graph_filtered_img = graph_denoise_interpolate(img)
plt.imshow(graph_filtered_img, cmap="gray")
plt.show()

# average filter and then linear interpolate
avg_filtered_img = avg_filter(img)
plt.imshow(avg_filtered_img, cmap="gray")
plt.show()

linear_interpolated_avg_filtered_img = interpolateLinear(avg_filtered_img)
plt.imshow(linear_interpolated_avg_filtered_img, cmap="gray")
plt.show()

ssd1 = np.sum((graph_filtered_img - linear_interpolated_avg_filtered_img) ** 2)
print(ssd1)

# linear interpolate and then average filter
linear_interpolated_img = interpolateLinear(img)
plt.imshow(linear_interpolated_img, cmap="gray")
plt.show()

avg_filtered_interpolated_img = avg_filter(linear_interpolated_img)
plt.imshow(avg_filtered_interpolated_img, cmap="gray")
plt.show()

ssd2 = np.sum((graph_filtered_img - avg_filtered_interpolated_img) ** 2)
print(ssd2)

print("hi")
