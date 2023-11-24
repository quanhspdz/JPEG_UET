import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# DCT block 8x8
def dct(array, quantization_matrix):
    result = np.zeros_like(array, dtype=float)

    for u in range(8):
        for v in range(8):
            cu = 1 / np.sqrt(2) if u == 0 else 1
            cv = 1 / np.sqrt(2) if v == 0 else 1
            sum_val = 0
            for i in range(8):
                for j in range(8):
                    sum_val += array[i, j] * np.cos((2 * i + 1) * u * np.pi / 16) * np.cos((2 * j + 1) * v * np.pi / 16)
            result[u, v] = cu * cv * sum_val / 4

    result = np.round(result / quantization_matrix)
    return result

# IDCT block 8x8
def idct(array, quantization_matrix):
    reconstruction = np.zeros_like(array, dtype=float)

    for i in range(8):
        for j in range(8):
            sum_val = 0
            for u in range(8):
                for v in range(8):
                    cu = 1 / np.sqrt(2) if u == 0 else 1
                    cv = 1 / np.sqrt(2) if v == 0 else 1
                    sum_val += cu * cv * array[u, v] * np.cos((2 * i + 1) * u * np.pi / 16) * np.cos((2 * j + 1) * v * np.pi / 16)
            reconstruction[i, j] = sum_val / 4

    reconstruction = reconstruction * quantization_matrix
    return reconstruction

# DCT + quantization function for Y component
def dct_image_y(image, quantization_matrix):
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
            block = new_image[i:i + block_size, j:j + block_size]
            result[i:i + block_size, j:j + block_size] = dct(block, quantization_matrix)

    return result

# IDCT + Iquantization function for Y component
def idct_image_y(result, quantization_matrix):
    height, width = result.shape
    block_size = 8

    image = np.zeros((height, width))

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = result[i:i + block_size, j:j + block_size]
            image[i:i + block_size, j:j + block_size] = idct(block, quantization_matrix) + 128

    return image.clip(0, 255).astype(np.uint8)

# Encode Y component using Run-Length Encoding
def encode_rle(matrix):
    flatten_matrix = matrix.flatten()
    encoded_matrix = []
    count = 0
    for i in range(len(flatten_matrix)):
        if i == 0:
            count = 1
        elif flatten_matrix[i] == flatten_matrix[i - 1]:
            count += 1
        else:
            encoded_matrix.append((flatten_matrix[i - 1], count))
            count = 1
    encoded_matrix.append((flatten_matrix[-1], count))
    return encoded_matrix

# Decode Y component using Run-Length Decoding
def decode_rle(encoded_matrix, shape):
    decoded_matrix = np.zeros(shape)
    index = 0
    for value, count in encoded_matrix:
        decoded_matrix.flat[index:index + count] = value
        index += count
    return decoded_matrix.reshape(shape)

# Read the image
image = cv2.imread('vidu_1.png', cv2.IMREAD_GRAYSCALE)

# Quantization matrix for Y component
quantization_matrix_y = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                  [12, 12, 14, 19, 26, 58, 60, 55],
                                  [14, 13, 16, 24, 40, 57, 69, 56],
                                  [14, 17, 22, 29, 51, 87, 80, 62],
                                  [18, 22, 37, 56, 68, 109, 103, 77],
                                  [24, 35, 55, 64, 81, 104, 113, 92],
                                  [49, 64, 78, 87, 103, 121, 120, 101],
                                  [72, 92, 95, 98, 112, 100, 103, 99]])

# DCT + Quantization on the Y component
result_dct_y = dct_image_y(image, quantization_matrix_y)

# Encode the DCT coefficients using Run-Length Encoding
encoded_data = encode_rle(result_dct_y)

# Reconstruct the Y component
decoded_data = decode_rle(encoded_data, result_dct_y.shape)
reconstruction_dct_y = idct_image_y(decoded_data, quantization_matrix_y)

# Compression ratio calculation
original_size = os.path.getsize('vidu_1.png')
compressed_size = len(encoded_data) * 2  # Assuming each (value, count) pair requires 2 bytes
compression_ratio = original_size / compressed_size
print(f"Compression Ratio: {compression_ratio:.2f}")

# Visualize the results
plt.gray()
plt.subplot(131), plt.imshow(image, cmap='gray'), plt.axis('off'), plt.title('Original Image', size=10)
plt.subplot(132), plt.imshow(result_dct_y, cmap='gray'), plt.axis('off'), plt.title('Quantized DCT Coefficients', size=10)
plt.subplot(133), plt.imshow(reconstruction_dct_y, cmap='gray'), plt.axis('off'), plt.title('Decompressed Image', size=10)
plt.show()
