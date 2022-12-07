import matplotlib.pyplot as plt
import numpy as np
from graph_interpolator import interpolateGraphFilter
from linear_interpolation import interpolateLinear

img_path = 'img/lena_gray.png'

img = plt.imread(img_path)
plt.imshow(img, cmap="gray")
plt.show()

graph_interpolated_img = interpolateGraphFilter(img)
plt.imshow(graph_interpolated_img, cmap="gray")
plt.show()

linear_interpolated_img = interpolateLinear(img)
plt.imshow(linear_interpolated_img, cmap="gray")
plt.show()

ssd = np.sum((graph_interpolated_img - linear_interpolated_img) ** 2)

print(ssd)


