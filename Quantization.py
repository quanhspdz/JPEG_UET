import numpy as np
import cv2
import scipy.fftpack as fft
from scipy.fftpack import idct
import matplotlib.pyplot as plt

# Hàm tích chập 1D giữa hai mảng
def convolve1D(data, kernel):
    return [sum(data[i:i+len(kernel)] * kernel) for i in range(len(data) - len(kernel) + 1)]

# Hàm biến đổi DCT 1D
def dct1D(data):
    N = len(data)
    dct = np.zeros(N)
    for k in range(N):
        sum_val = 0.0
        for n in range(N):
            sum_val += data[n] * np.cos((np.pi * k / N) * (n + 0.5))
        dct[k] = sum_val
    return dct

# Hàm biến đổi DCT 2D cho một khối 8x8
def dct2D(block):
    # Biến đổi DCT trên các hàng
    row_dct = [dct1D(row) for row in block]
    
    # Biến đổi DCT trên các cột của ma trận đã được biến đổi DCT từ trước
    col_dct = [dct1D(np.array(row_dct).T) for row_dct in block]

    return np.array(col_dct)

# Hàm lấy một khối 8x8 từ ảnh
def get_image_block(image, i, j):
    return image[i:i+8, j:j+8]

# Hàm nén ảnh bằng DCT với lượng tử hóa
def compress_image(image, quantization_matrix):
    height, width = image.shape
    compressed_image = np.zeros((height, width))

    for i in range(0, height, 8):
        for j in range(0, width, 8):
            block = get_image_block(image, i, j)
            dct_block = dct2D(block)
            #Chia từng khối DCT với ma trận quantization
            quantized_block = np.round(dct_block / quantization_matrix)
            compressed_image[i:i+8, j:j+8] = quantized_block

    return compressed_image

# Hàm giải nén ảnh bằng DCT với lượng tử hóa
def decompress_image(compressed_image, quantization_matrix):
    height, width = compressed_image.shape
    decompressed_image = np.zeros((height, width))

    for i in range(0, height, 8):
        for j in range(0, width, 8):
            block = get_image_block(compressed_image, i, j)
            dequantized_block = block * quantization_matrix
            idct_block = np.round(np.real(fft.idct(fft.idct(dequantized_block.T, type=2, norm='ortho').T, type=2, norm='ortho')))
            decompressed_image[i:i+8, j:j+8] = idct_block

    return decompressed_image

# Hàm chỉnh ảnh thành kích thước chia hết cho 8x8
def resize_image_to_multiple_of_8(image):
    height, width = image.shape
    new_height = ((height + 7) // 8) * 8
    new_width = ((width + 7) // 8) * 8
    resized_image = np.zeros((new_height, new_width), dtype=image.dtype)
    resized_image[0:height, 0:width] = image
    return resized_image

# Đọc ảnh gốc
original_image = cv2.imread('vidu.jpg', cv2.IMREAD_GRAYSCALE)

if original_image is None:
    print("Không thể đọc ảnh.")
else:
    # Kích thước ảnh
    height, width = original_image.shape[:2]

    print(f'Kích thước ảnh: {width}x{height}')

    # Cắt ảnh thành kích thước chia hết cho 8x8
    resized_image = resize_image_to_multiple_of_8(original_image)

    # Tạo ma trận lượng tử hóa
    quantization_matrix = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                    [12, 12, 14, 19, 26, 58, 60, 55],
                                    [14, 13, 16, 24, 40, 57, 69, 56],
                                    [14, 17, 22, 29, 51, 87, 80, 62],
                                    [18, 22, 37, 56, 68, 109, 103, 77],
                                    [24, 35, 55, 64, 81, 104, 113, 92],
                                    [49, 64, 78, 87, 103, 121, 120, 101],
                                    [72, 92, 95, 98, 112, 100, 103, 99]])

    # Nén ảnh sử dụng ma trận lượng tử hóa
    compressed_image = compress_image(resized_image, quantization_matrix)
    #Lưu ảnh
    cv2.imwrite('compressed_image.jpg', compressed_image)

    # Giải nén ảnh sử dụng ma trận lượng tử hóa
    decompressed_image = decompress_image(compressed_image, quantization_matrix)

    # Lưu ảnh nén và giải nén
    cv2.imwrite('decompressed_image.jpg', decompressed_image)

    # Hiển thị ảnh gốc và ảnh giải nén
    cv2.imshow('Compressed Image', compressed_image)
    cv2.imshow('Decompressed Image', decompressed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
