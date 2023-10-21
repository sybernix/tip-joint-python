import numpy as np
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
import cv2


def getXYMaps(original_image, angle_degrees):
    rows, cols = original_image.shape
    alpha = -np.deg2rad(angle_degrees)  # Negative because y-axis is reversed in image coordinates

    R = np.array([[np.cos(alpha), -np.sin(alpha)],
                  [np.sin(alpha), np.cos(alpha)]])  # Rotation matrix

    x, y = np.meshgrid(np.arange(cols), np.arange(rows))

    coords = np.linalg.inv(R) @ np.vstack([x.flatten() - cols / 2, y.flatten() - rows / 2])
    x_rot = coords[0].reshape(rows, cols) + cols / 2
    y_rot = coords[1].reshape(rows, cols) + rows / 2

    return x_rot, y_rot


img = cv2.imread("../../img/lena_gray.png", cv2.IMREAD_GRAYSCALE) / 255.0  # Ensure the image is grayscale and normalize

h, w = img.shape

angle_degrees = 20
max_disp = int(np.ceil(np.sqrt(h ** 2 + w ** 2) - max(h, w)))

img_padded = np.pad(img, ((max_disp, max_disp), (max_disp, max_disp)))

# Update maps for the padded image
mapx, mapy = getXYMaps(img_padded, angle_degrees)

mapx = mapx[max_disp:max_disp + h, max_disp:max_disp + w]
mapy = mapy[max_disp:max_disp + h, max_disp:max_disp + w]

p_h, p_w = 10, 10
interpolated_output = np.zeros((h, w))

for row in range(h // p_h):
    psv = row * p_h
    pev = (row + 1) * p_h
    for col in range(w // p_w):
        psh = col * p_w
        peh = (col + 1) * p_w

        p_mapx = mapx[psv:pev, psh:peh]
        p_mapy = mapy[psv:pev, psh:peh]

        x_min, y_min = int(np.floor(p_mapx.min())), int(np.floor(p_mapy.min()))
        x_max, y_max = int(np.ceil(p_mapx.max())), int(np.ceil(p_mapy.max()))

        img_patch = img_padded[y_min:y_max + 1, x_min:x_max + 1]
        ip_h, ip_w = img_patch.shape

        theta = np.zeros((p_h * p_w, ip_h * ip_w))

        for i in range(p_h):
            for j in range(p_w):
                theta_r = i * p_w + j

                x = p_mapx[i, j] - x_min
                y = p_mapy[i, j] - y_min

                l, t = int(np.floor(x)), int(np.floor(y))
                a, b = x - l, y - t

                theta_lt_loc = t * ip_w + l
                theta_rt_loc = theta_lt_loc + 1
                theta_lb_loc = (t + 1) * ip_w + l
                theta_rb_loc = theta_lb_loc + 1

                theta[theta_r, theta_lt_loc] = (1 - b) * (1 - a)
                theta[theta_r, theta_rt_loc] = (1 - b) * a
                theta[theta_r, theta_lb_loc] = b * (1 - a)
                theta[theta_r, theta_rb_loc] = b * a

        y = img_patch.T.flatten()

        output_flat = theta @ y
        output_mat = output_flat.reshape(p_h, p_w)
        interpolated_output[psv:pev, psh:peh] = output_mat

plt.imshow(interpolated_output, cmap='gray')
plt.axis('off')
plt.show()
