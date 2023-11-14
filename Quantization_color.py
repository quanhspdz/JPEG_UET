import cv2
import numpy as np
import matplotlib.pyplot as plt

# DCT block 8x8
def dct(array, quantization_matrix):
    result = np.zeros_like(array, dtype=float)

    # DCT theo hàng
    for i in range(8):
        for u in range(8):
            cu = 1 / np.sqrt(2) if u == 0 else 1
            sum_val = 0
            for v in range(8):
                sum_val += array[i][v] * np.cos((2 * v + 1) * np.pi * u / 16)
            result[i][u] = sum_val * cu * 1/2

    # DCT theo cột
    for j in range(8):
        for u in range(8):
            cu = 1 / np.sqrt(2) if u == 0 else 1
            sum_val = 0
            for v in range(8):
                sum_val += result[v][j] * np.cos((2 * v + 1) * np.pi * u / 16)
            result[u][j] = sum_val * cu * 1/2

    # Quantization
    result = np.round(result / quantization_matrix)

    return result

# IDCT block 8x8 
def idct(array, quantization_matrix):
    reconstruction = np.zeros_like(array, dtype=float)

    # IDCT theo hàng
    for i in range(8):
        for v in range(8):
            sum_val = 0
            for u in range(8):
                cu = 1 / np.sqrt(2) if u == 0 else 1
                sum_val += array[i][u] * cu * np.cos((2 * v + 1) * np.pi * u / 16)
            sum_val *= 1 / 2
            reconstruction[i][v] = sum_val

    # IDCT theo cột
    for j in range(8):
        for v in range(8):
            sum_val = 0
            for u in range(8):
                cu = 1 / np.sqrt(2) if u == 0 else 1
                sum_val += reconstruction[u][j] * cu * np.cos((2 * v + 1) * np.pi * u / 16)
            sum_val *= 1 / 2
            reconstruction[v][j] = sum_val

    # Lượng tử hóa ngược
    reconstruction = reconstruction * quantization_matrix #Nhân ngược lại với ma trận lượng tử hóa để được ảnh giải nén 
    return reconstruction

# DCT + quantization function 
def dct_image(image, quantization_matrix):
    if len(image.shape) == 2:  # Kiểm tra có là ảnh xám hay không
        height, width = image.shape
        channels = 1
    else:  # Ảnh màu
        height, width, channels = image.shape
    
    #Số lượng các khối được tính toán để  chia hết cho 8 
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

    result = np.zeros_like(new_image)

    for c in range(channels):
        for i in range(0, blocks_h, block_size):
            for j in range(0, blocks_w, block_size):
                block = new_image[i:i + block_size, j:j + block_size, c]
                result[i:i + block_size, j:j + block_size, c] = dct(block, quantization_matrix)

    return result

# IDCT + Iquantization function for an image
def idct_image(result, quantization_matrix):
    height, width, channels = result.shape
    block_size = 8

    image = np.zeros((height, width, channels))

    for c in range(channels):
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                block = result[i:i + block_size, j:j + block_size, c]
                # Chuyển về giá trị pixel gốc từ 0-255
                image[i:i + block_size, j:j + block_size, c] = idct(block, quantization_matrix) + 128 

    return image.clip(0, 255).astype(np.uint8) #Đảm bảo các giá trị pixel trong khoảng 0-255

# Đọc ảnh
image = cv2.imread('og.png')

# Ma trận lượng tử hóa
quantization_matrix = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                [12, 12, 14, 19, 26, 58, 60, 55],
                                [14, 13, 16, 24, 40, 57, 69, 56],
                                [14, 17, 22, 29, 51, 87, 80, 62],
                                [18, 22, 37, 56, 68, 109, 103, 77],
                                [24, 35, 55, 64, 81, 104, 113, 92],
                                [49, 64, 78, 87, 103, 121, 120, 101],
                                [72, 92, 95, 98, 112, 100, 103, 99]])

# DCT + Quantization trên ảnh màu đầu vào
result = dct_image(image, quantization_matrix)

# Tái tạo ảnh
reconstruction = idct_image(result, quantization_matrix)
cv2.imwrite('decompressed_image.jpg', reconstruction)

# hiển thị hình ảnh 
plt.gray()
plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.axis('off'), plt.title('Original Image', size=10)
plt.subplot(122), plt.imshow(cv2.cvtColor(reconstruction, cv2.COLOR_BGR2RGB)), plt.axis('off'), plt.title('Decompressed Image', size=10)
plt.show()
