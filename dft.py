import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# DFT block 8x8
def dft(array):
    result = np.zeros_like(array, dtype=complex)

    for u in range(8):
        for v in range(8):
            sum_val = 0
            for i in range(8):
                for j in range(8):
                    sum_val += array[i][j] * np.exp(-2j * np.pi * (u * i / 8 + v * j / 8))
            result[u][v] = sum_val / 8

    return result

# IDFT block 8x8 
def idft(array):
    reconstruction = np.zeros_like(array, dtype=complex)

    for i in range(8):
        for j in range(8):
            sum_val = 0
            for u in range(8):
                for v in range(8):
                    sum_val += array[u][v] * np.exp(2j * np.pi * (u * i / 8 + v * j / 8))
            reconstruction[i][j] = sum_val / 8

    return reconstruction

# DFT + Quantization function 
def dft_image(image, quantization_matrix):
    if len(image.shape) == 2:  # Check if it is a grayscale image
        height, width = image.shape
        channels = 1
    else:  # Color image
        height, width, channels = image.shape
    
    block_size = 8
    blocks_w = width + (block_size - width % block_size) if width % block_size != 0 else width
    blocks_h = height + (block_size - height % block_size) if height % block_size != 0 else height

    new_image = np.zeros((blocks_h, blocks_w, channels))

    if channels == 1:
        new_image[:height, :width, 0] = image
    else:
        new_image[:height, :width, :] = image

    new_image = new_image.astype(float)
    new_image -= 128

    result = np.zeros_like(new_image, dtype=complex)

    for c in range(channels):
        for i in range(0, blocks_h, block_size):
            for j in range(0, blocks_w, block_size):
                block = new_image[i:i + block_size, j:j + block_size, c]
                result[i:i + block_size, j:j + block_size, c] = dft(block)

    return result

# IDFT + Iquantization function for an image
def idft_image(result, quantization_matrix):
    height, width, channels = result.shape
    block_size = 8

    image = np.zeros((height, width, channels), dtype=complex)

    for c in range(channels):
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                block = result[i:i + block_size, j:j + block_size, c]
                # Change back to the original pixel values in the range 0-255
                image[i:i + block_size, j:j + block_size, c] = idft(block)

    return image

# Run-Length Encoding function for complex values
def encode_rle_complex(matrix):
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

# Run-Length Decoding function for complex values
def decode_rle_complex(encoded_matrix, shape):
    decoded_matrix = np.zeros(shape, dtype=complex)
    index = 0
    for value, count in encoded_matrix:
        decoded_matrix.flat[index:index + count] = value
        index += count
    return decoded_matrix.reshape(shape)

# Read the image
image = cv2.imread('vidu_1.png')

# Define the quantization matrix for DFT
quantization_matrix_dft = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                    [12, 12, 14, 19, 26, 58, 60, 55],
                                    [14, 13, 16, 24, 40, 57, 69, 56],
                                    [14, 17, 22, 29, 51, 87, 80, 62],
                                    [18, 22, 37, 56, 68, 109, 103, 77],
                                    [24, 35, 55, 64, 81, 104, 113, 92],
                                    [49, 64, 78, 87, 103, 121, 120, 101],
                                    [72, 92, 95, 98, 112, 100, 103, 99]])

# DFT + Quantization on the input color image
result_dft = dft_image(image, quantization_matrix_dft)

# Encode the DFT coefficients using Run-Length Encoding
encoded_data_dft = encode_rle_complex(result_dft)
# Reconstruct the image
reconstruction_dft = idft_image(decode_rle_complex(encoded_data_dft, result_dft.shape), quantization_matrix_dft)
cv2.imwrite('decompress_dft.png', np.abs(reconstruction_dft).astype(np.uint8))
decoded_data_dft = decode_rle_complex(encoded_data_dft, result_dft.shape)
reconstruction_dft = idft_image(decoded_data_dft, quantization_matrix_dft)

original_size_dft = os.path.getsize('vidu_1.png')
compressed_size_dft = os.path.getsize('decompress_dft.png')

compression_ratio_dft = original_size_dft / compressed_size_dft
print(f"Compression Ratio (DFT): {compression_ratio_dft:.2f}")

# Plot the results for comparison
plt.gray()
plt.subplot(131), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.axis('off'), plt.title('Original Image', size=10)
plt.subplot(132), plt.imshow(np.abs(result_dft[0]), cmap='gray'), plt.axis('off'), plt.title('Quantized DFT Coefficients', size=10)
plt.subplot(133), plt.imshow(cv2.cvtColor(np.abs(reconstruction_dft).astype(np.uint8), cv2.COLOR_BGR2RGB)), plt.axis('off'), plt.title('Decompressed Image (DFT)', size=10)
plt.show()