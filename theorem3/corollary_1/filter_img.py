import numpy as np

def filter_img(img, filter):
    h, w = img.shape
    filtered_img = np.zeros((h, w))

    for i in range(int(h / 2)):
        for j in range(int(w / 2)):
            row = i * 2
            col = j * 2
            img_patch = img[row:row + 2, col:col + 2]
            flattened_patch = np.reshape(img_patch, (4, 1))
            output = np.matmul(filter, flattened_patch)

            filtered_img[row, col] = output[0][0]
            filtered_img[row, col + 1] = output[1][0]
            filtered_img[row + 1, col] = output[2][0]
            filtered_img[row + 1, col + 1] = output[3][0]
    return filtered_img