import cv2
import numpy as np
from scipy.fftpack import dct, idct

def message_to_bin(message):
    binary_message = bin(int.from_bytes(message.encode(), 'big'))[2:]
    return binary_message.zfill(8 * ((len(binary_message) + 7) // 8))

def embed_data(image, data):
    data_bin = message_to_bin(data)
    data_size = len(data_bin)

    height, width = image.shape[:2]
    block_size = 8

    if len(data_bin) > height * width:
        raise ValueError("Data size is too large for the given image.")

    # Embed data using Discrete Cosine Transform (DCT)
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = image[i:i + block_size, j:j + block_size, 0]  # Assuming grayscale image

            # Apply 2D DCT to the block
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')

            # Embed data in the DC coefficient
            for m in range(min(block_size, data_size)):
                for n in range(min(block_size, data_size)):
                    dct_block[m, n] += int(data_bin[m * block_size + n])

            # Apply inverse DCT to the modified block
            image[i:i + block_size, j:j + block_size, 0] = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')

            data_size -= block_size

    return image

def extract_data(image):
    height, width = image.shape[:2]
    block_size = 8
    extracted_data = ''

    # Extract data using Discrete Cosine Transform (DCT)
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = image[i:i + block_size, j:j + block_size, 0]  # Assuming grayscale image

            # Apply 2D DCT to the block
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')

            # Extract data from the DC coefficient
            for m in range(block_size):
                for n in range(block_size):
                    extracted_data += str(int(dct_block[m, n]) % 2)

    return extracted_data

# Example Usage
original_image_path = '801281.jpg'
output_image_path = 'output.jpg'
message = "Hello, this is a more advanced secret message!"

# Read the original image
image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)

# Embedding data
embedded_image = embed_data(image.copy(), message)

# Save the embedded image
cv2.imwrite(output_image_path, embedded_image)

# Read the embedded image for extraction
embedded_image = cv2.imread(output_image_path, cv2.IMREAD_GRAYSCALE)

# Extracting data
extracted_message = extract_data(embedded_image)
print("Extracted Message:", extracted_message)
