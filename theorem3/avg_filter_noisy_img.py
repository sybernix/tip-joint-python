import matplotlib.pyplot as plt
import numpy as np

img_path = '../img/noisy_lena.png'

img = plt.imread(img_path)
plt.imshow(img, cmap="gray")
plt.show()

h, w = img.shape
filtered_img = np.zeros((h, w))

for i in range(int(h / 2)):
    for j in range(int(w - 1)):
        row = i * 2
        col = j
        img_patch = img[row:row + 2, col:col + 2]
        filtered_img[row:row + 2, col:col + 2] = np.average(img_patch)

        print("hi")

plt.imshow(filtered_img, cmap="gray")
plt.show()

print("hi")
