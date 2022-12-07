import matplotlib.pyplot as plt
import numpy as np
from separable_graph_filter import graph_denoise_interpolate

img_path = '../img/noisy_lena.png'

img = plt.imread(img_path)
plt.imshow(img, cmap="gray")
plt.show()

graph_filtered_img = graph_denoise_interpolate(img)

plt.imshow(graph_filtered_img, cmap="gray")
plt.show()

print("hi")
