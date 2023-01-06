import numpy as np
import matplotlib.pyplot as plt
from numpy import matmul as mul
from utils import get_project_root

img_path = str(get_project_root()) + "/img/lena_gray.png"

img = plt.imread(img_path)
# plt.imshow(img, cmap="gray")
# plt.show()


theta1 = np.array([[1 / 3, 2 / 3, 0, 0], [0, 0, 1 / 3, 2 / 3]])
theta1T = theta1.transpose()
theta2 = np.array([[1/3, 0, 2/3, 0], [1/8, 3/8, 1/8, 3/8], [0, 1/3, 0, 2/3]])
theta2T = theta2.transpose()

h, w = img.shape
interpolated_img = np.zeros((2 * h - 1, 2 * w - 1))
row = 0
col = 0

img_patch = img[row:row + 2, col:col + 2]
flattened_patch_1 = np.reshape(img_patch, (4, 1))

interp1_1 = mul(theta1, flattened_patch_1)
interp2_1 = mul(theta2, flattened_patch_1)
interpolated_patch_1 = np.zeros((3, 3))

interpolated_patch_1[0, 0] = img_patch[0, 0]
interpolated_patch_1[0, 2] = img_patch[0, 1]
interpolated_patch_1[2, 0] = img_patch[1, 0]
interpolated_patch_1[2, 2] = img_patch[1, 1]

interpolated_patch_1[0, 1] = interp1_1[0]
interpolated_patch_1[2, 1] = interp1_1[1]

interpolated_patch_1[1, 0] = interp2_1[0]
interpolated_patch_1[1, 1] = interp2_1[1]
interpolated_patch_1[1, 2] = interp2_1[2]

print("hi")

gamma = 0.4
m = 4
n1 = 2
n2 = 3

H = np.zeros((m, m + n1 + n2))
H[0, 0] = 1
H[1, 1] = 1
H[2, 2] = 1
H[3, 3] = 1
HT = H.transpose()

# theta1 = np.array([[1 / 2, 1 / 2, 0, 0], [0, 0, 1 / 2, 1 / 2]])
# theta1T = theta1.transpose()

A1mn = 2 * mul(theta1T, np.linalg.inv(mul(theta1, theta1T)))
A1 = np.zeros((m + n1 + n2, m + n1 + n2))
A1[0:m, m:m + n1] = A1mn
A1T = A1.transpose()

temp1 = mul(HT, H) - 2 * mul(mul(A1T, HT), H) + mul(mul(mul(A1T, HT), H), A1)

# theta2 = np.array([[0.5, 0, 0.5, 0], [0.25, 0.25, 0.25, 0.25], [0, 0.5, 0, 0.5]])
# theta2T = theta2.transpose()

A2mn = 2 * mul(theta2T, np.linalg.inv(mul(theta2, theta2T)))
A2 = np.zeros((m + n1 + n2, m + n1 + n2))
A2[0:m, m + n1:m + n1 + n2] = A2mn
A2T = A2.transpose()

temp2 = mul(HT, H) - 2 * mul(mul(A2T, HT), H) + mul(mul(mul(A2T, HT), H), A2)

coeff = mul(HT, H) + gamma * (temp1 + temp2)
inv_coeff = np.linalg.inv(coeff)

h, w = img.shape
interpolated_img = np.zeros((2 * h - 1, 2 * w - 1))

img_patch = img[row:row + 2, col:col + 2]
flattened_patch = np.reshape(img_patch, (4, 1))

x = mul(inv_coeff, mul(HT, flattened_patch))
interpolated_patch = x * (1 + gamma)
interpolated_patch_reshaped = np.zeros((3, 3))
interpolated_patch_reshaped[0, 0] = interpolated_patch[0, 0]
interpolated_patch_reshaped[0, 2] = interpolated_patch[1, 0]
interpolated_patch_reshaped[2, 0] = interpolated_patch[2, 0]
interpolated_patch_reshaped[2, 2] = interpolated_patch[3, 0]

interpolated_patch_reshaped[0, 1] = interpolated_patch[4, 0]
interpolated_patch_reshaped[2, 1] = interpolated_patch[5, 0]

interpolated_patch_reshaped[1, 0] = interpolated_patch[6, 0]
interpolated_patch_reshaped[1, 1] = interpolated_patch[7, 0]
interpolated_patch_reshaped[1, 2] = interpolated_patch[8, 0]

print("hi")
