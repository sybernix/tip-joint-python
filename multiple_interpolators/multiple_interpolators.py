import numpy as np
import matplotlib.pyplot as plt
from utils import get_project_root
from multiple_graph_interpolators import graph_multiple_interpolate
from linear_interpolator import linear_multiple_interpolate

gamma = 0.4
m = 4
n1 = 2
n2 = 3

img_path = str(get_project_root()) + "/img/lena_gray.png"

img = plt.imread(img_path)
plt.imshow(img, cmap="gray")
plt.show()

graph_interpolated_img = graph_multiple_interpolate(img, m, n1, n2, gamma)
plt.imshow(graph_interpolated_img, cmap="gray")
plt.show()

linear_interpolated_img = linear_multiple_interpolate(img)
plt.imshow(linear_interpolated_img, cmap="gray")
plt.show()

ssd = np.sum((graph_interpolated_img - linear_interpolated_img) ** 2)

print(ssd)
