import cv2
import numpy as np
import matplotlib.pyplot as plt

# Hàm DCT 8x8
def dct(array):
    result = np.zeros_like(array, dtype=float)

    for i in range(8):
        for u in range(8):
            cu = 1 / np.sqrt(2) if u == 0 else 1
            sum_val = 0
            for v in range(8):
                sum_val += array[v, i] * np.cos((2 * v + 1) * np.pi * u / 16)
            result[u, i] = cu * sum_val / 2

    for j in range(8):
        for u in range(8):
            cu = 1 / np.sqrt(2) if u == 0 else 1
            sum_val = 0
            for v in range(8):
                sum_val += result[u, v] * np.cos((2 * v + 1) * np.pi * j / 16)
            array[u, j] = cu * sum_val / 2

    return array

# Hàm IDCT 8x8
def idct(array):
    reconstruction = np.zeros_like(array, dtype=float)

    for i in range(8):
        for v in range(8):
            sum_val = 0
            for u in range(8):
                cu = 1 / np.sqrt(2) if u == 0 else 1
                sum_val += array[i, u] * cu * np.cos((2 * v + 1) * np.pi * u / 16)
            sum_val *= 1/2
            reconstruction[i, v] = sum_val

    for j in range(8):
        for v in range(8):
            sum_val = 0
            for u in range(8):
                cu = 1 / np.sqrt(2) if u == 0 else 1
                sum_val += reconstruction[u, j] * cu * np.cos((2 * v + 1) * np.pi * u / 16)
            sum_val *= 1/2
            reconstruction[v, j] = sum_val

    return reconstruction

# Hàm lượng tử hóa ảnh xám sử dụng DCT
def dct_image_quantized(image, quantization_factor=1):
    height, width = image.shape
    block_size = 8
    blocks_w = width + (block_size - width % block_size) if width % block_size != 0 else width
    blocks_h = height + (block_size - height % block_size) if height % block_size != 0 else height

    new_image = np.zeros((blocks_h, blocks_w))
    new_image[:height, :width] = image

    new_image = new_image.astype(float)
    new_image -= 128

    result = np.zeros_like(new_image)

    for i in range(0, blocks_h, block_size):
        for j in range(0, blocks_w, block_size):
            block = new_image[i:i+block_size, j:j+block_size]
            result[i:i+block_size, j:j+block_size] = dct(block) / quantization_factor

    return result

# Hàm giải nén ảnh đã lượng tử hóa
def idct_image_quantized(result, quantization_factor=1):
    height, width = result.shape
    block_size = 8

    reconstruction = np.zeros_like(result)

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = result[i:i+block_size, j:j+block_size]
            reconstruction[i:i+block_size, j:j+block_size] = idct(block) * quantization_factor + 128

    return reconstruction

# Hàm hiển thị ảnh gốc và ảnh đã giải nén
def show_images(original_image, reconstructed_image, title1='Ảnh Gốc', title2='Ảnh Đã Giải Nén'):
    plt.gray()
    plt.subplot(121), plt.imshow(original_image), plt.axis('off'), plt.title(title1, size=10)
    plt.subplot(122), plt.imshow(reconstructed_image), plt.axis('off'), plt.title(title2, size=10)
    plt.show()

# Đọc ảnh
image = cv2.imread('og.jpg', cv2.IMREAD_GRAYSCALE)

# Áp dụng DCT với lượng tử hóa
quantization_factor = 10  # Thay đổi hệ số lượng tử hóa
result_quantized = dct_image_quantized(image, quantization_factor)

# Thực hiện giải nén trên kết quả đã lượng tử hóa
reconstruction_quantized = idct_image_quantized(result_quantized, quantization_factor)

# Hiển thị ảnh gốc và ảnh đã giải nén
show_images(image, reconstruction_quantized)

