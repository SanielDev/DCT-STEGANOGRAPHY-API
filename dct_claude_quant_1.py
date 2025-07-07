import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct
import cv2
import os
import math
import time
from skimage.metrics import structural_similarity as ssim

class ImprovedDCTSteganography:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.block_size = 8
        # Standard JPEG luminance quantization matrix
        self.quantization_matrix = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ])
        # Store embedding positions for extraction
        self.embedding_positions = []
        
    def _pad_image(self, image):
        h, w = image.shape
        new_h = ((h + self.block_size - 1) // self.block_size) * self.block_size
        new_w = ((w + self.block_size - 1) // self.block_size) * self.block_size
        padded = np.zeros((new_h, new_w))
        padded[:h, :w] = image
        return padded, (h, w)

    def _calculate_texture_mask(self, block):
        """Calculate texture masking factor based on local gradients."""
        gradient_x = np.diff(block, axis=1, prepend=block[:, :1])
        gradient_y = np.diff(block, axis=0, prepend=block[:1, :])
        texture_strength = np.mean(np.abs(gradient_x)) + np.mean(np.abs(gradient_y))
        return 1 + (texture_strength / 128)  # Normalize to reasonable range

    def _get_embedding_strength(self, dct_block, position, texture_mask):
        """Calculate adaptive embedding strength."""
        q_value = self.quantization_matrix[position]
        local_variance = np.var(dct_block)
        # Combine quantization, local variance, and texture masking
        return self.alpha * (q_value / 16) * (1 + local_variance/1000) * texture_mask

    def _select_embedding_position(self, dct_block):
        """Select optimal position for embedding based on HVS model."""
        perceptual_mask = np.abs(dct_block) / self.quantization_matrix
        # Avoid DC coefficient and high frequencies
        perceptual_mask[0, 0] = 0
        perceptual_mask[7:, :] = 0
        perceptual_mask[:, 7:] = 0
        return np.unravel_index(np.argmax(perceptual_mask), perceptual_mask.shape)

    def _split_into_blocks(self, image):
        padded_image, original_shape = self._pad_image(image)
        height, width = padded_image.shape
        blocks = []
        for i in range(0, height, self.block_size):
            for j in range(0, width, self.block_size):
                block = padded_image[i:i+self.block_size, j:j+self.block_size]
                blocks.append(block)
        return blocks, original_shape

    def _reconstruct_from_blocks(self, blocks, original_shape):
        h, w = original_shape
        new_h = ((h + self.block_size - 1) // self.block_size) * self.block_size
        new_w = ((w + self.block_size - 1) // self.block_size) * self.block_size
        
        image = np.zeros((new_h, new_w))
        block_idx = 0
        for i in range(0, new_h, self.block_size):
            for j in range(0, new_w, self.block_size):
                if block_idx < len(blocks):
                    image[i:i+self.block_size, j:j+self.block_size] = blocks[block_idx]
                    block_idx += 1
        return image[:h, :w]

    def _message_to_bits(self, message):
        length_bits = format(len(message), '016b')
        message_bits = ''.join(format(ord(char), '08b') for char in message)
        return [int(bit) for bit in length_bits + message_bits]

    def _bits_to_message(self, bits):
        if len(bits) < 16:
            return ""
        length_bits = ''.join(map(str, bits[:16]))
        message_length = int(length_bits, 2)
        
        message_bits = bits[16:16 + message_length * 8]
        message = ""
        for i in range(0, len(message_bits), 8):
            if i + 8 <= len(message_bits):
                byte = ''.join(map(str, message_bits[i:i+8]))
                message += chr(int(byte, 2))
        return message

    def get_max_message_length(self, image_path):
        """Calculate maximum possible message length for an image."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb_image[:, :, 0]
        blocks, _ = self._split_into_blocks(y_channel)
        
        return (len(blocks) - 16) // 8

    def embed(self, image_path, message):
        # Check message length
        max_length = self.get_max_message_length(image_path)
        if len(message) > max_length:
            raise ValueError(
                f"Message is too long!\n"
                f"Message length: {len(message)} characters\n"
                f"Maximum allowed length: {max_length} characters"
            )

        # Read and prepare image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)   #  RGB --> YCrCb
        y_channel, cr_channel, cb_channel = cv2.split(ycrcb_image)  # split into 3 different channels
        
        blocks, original_shape = self._split_into_blocks(y_channel)
        bits = self._message_to_bits(message)     # Converting the message into bits
        
        # Reset embedding positions
        self.embedding_positions = []
        
        # Process each block
        modified_blocks = []
        for idx, block in enumerate(blocks):
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')  # Block is getting transformed into Transform Domain
            
            if idx < len(bits):
                # Calculate texture mask
                texture_mask = self._calculate_texture_mask(block)
                
                # Select embedding position
                pos = self._select_embedding_position(dct_block)
                self.embedding_positions.append(pos)
                
                # Calculate adaptive strength
                strength = self._get_embedding_strength(dct_block, pos, texture_mask)
                
                # Embed bit  (Sign-Based Embedding)
                if bits[idx] == 1:
                    dct_block[pos] = abs(dct_block[pos]) + strength
                else:
                    dct_block[pos] = -abs(dct_block[pos]) - strength
            
            modified_block = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')
            modified_blocks.append(modified_block)
        
        # Reconstruct image
        modified_y = self._reconstruct_from_blocks(modified_blocks, original_shape)
        modified_y = np.clip(modified_y, 0, 255).astype(np.uint8)
        
        # Save embedding positions
        np.save('embedding_positions.npy', np.array(self.embedding_positions))
        
        # Reconstruct color image
        modified_ycrcb = cv2.merge([modified_y, cr_channel, cb_channel])
        modified_image = cv2.cvtColor(modified_ycrcb, cv2.COLOR_YCrCb2BGR)
        
        return modified_image

    def extract_text(self, stego_image_path):
        # Load embedding positions
        try:
            embedding_positions = np.load('embedding_positions.npy')
        except:
            raise ValueError("Could not load embedding positions file")
            
        stego_image = cv2.imread(stego_image_path)
        if stego_image is None:
            raise ValueError(f"Could not load stego image from {stego_image_path}")
        
        ycrcb_image = cv2.cvtColor(stego_image, cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb_image[:, :, 0]
        
        blocks, _ = self._split_into_blocks(y_channel)
        extracted_bits = []
        
        for idx, block in enumerate(blocks):
            if idx < len(embedding_positions):
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                pos = tuple(embedding_positions[idx].astype(int))
                extracted_bits.append(1 if dct_block[pos] > 0 else 0)
        
        return self._bits_to_message(extracted_bits)


def ensure_directories_exist():
    directories = ['Images', 'Embedded-image', 'Decoded-message']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")


def calculate_mse(original, stego):
    """
    Calculate the Mean Squared Error (MSE) between the original and stego images.
    The images should both be in the same color space and have the same dimensions.
    """
    # Convert images to floating point for accurate calculations
    original_float = original.astype(np.float64)
    stego_float = stego.astype(np.float64)

    # Compute the squared difference and average
    mse = np.mean((original_float - stego_float) ** 2)
    return mse

def calculate_psnr(mse, max_pixel_value=255.0):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) given the MSE and the
    maximum possible pixel value (default = 255 for 8-bit images).
    """
    if mse == 0:
        # MSE = 0 means no difference at all between the images
        return float('inf')  # Infinite PSNR
    psnr = 20 * math.log10(max_pixel_value / math.sqrt(mse))
    return psnr

def calculate_ncc(original, stego):
    """
    Calculate the Normalized Cross-Correlation (NCC) between two images.
    1 -> perfect correlation, -1 -> perfect inverse correlation, 0 -> no correlation.
    """
    # Flatten to 1D
    orig_f = original.astype(np.float64).ravel()
    steg_f = stego.astype(np.float64).ravel()

    # Subtract means
    orig_mean = np.mean(orig_f)
    steg_mean = np.mean(steg_f)

    numerator = np.sum((orig_f - orig_mean) * (steg_f - steg_mean))
    denominator = np.sqrt(np.sum((orig_f - orig_mean)**2) * np.sum((steg_f - steg_mean)**2))

    if denominator == 0:
        return 0
    return numerator / denominator

def calculate_ssim(original, stego):
    """
    Calculate the Structural Similarity Index (SSIM).
    This uses the grayscale channels for a straightforward measure.
    Returns a value between -1 and 1, where 1 is a perfect match.
    """
    # Convert to grayscale for SSIM
    orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    steg_gray = cv2.cvtColor(stego, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(orig_gray, steg_gray, full=True)
    return score


def main():
    ensure_directories_exist()
    steganography = ImprovedDCTSteganography(alpha=0.1)
    
    while True:
        print("\n=== Improved DCT Steganography Menu ===")
        print("1. Embed message in image")
        print("2. Extract message from image")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == "3":
            print("Exiting program...")
            break
            
        elif choice == "1":
            input_image = input("Enter the image name from Images folder: ")
            image_path = f"Images/{input_image}"
            
            try:
                max_length = steganography.get_max_message_length(image_path)
                print(f"\nMaximum message length for this image: {max_length} characters")
                
                secret_message = input("Enter the secret message to hide: ")
                print(f"Message length: {len(secret_message)} characters")
                
                print("\nStarting embedding process...")
                start_time = time.time()  # ⏱️ Start timer

                embedded_img = steganography.embed(image_path, secret_message)

                end_time = time.time()  # ⏱️ End timer
                elapsed_time = end_time - start_time
                print(f"✅ Text embedded successfully! Time taken: {elapsed_time:.4f} seconds")

                cv2.imwrite('Embedded-image/stego_image.png', embedded_img)
                # print("Text embedded successfully!")

                # =========================
                # Calculate MSE, PSNR, NCC and SSIM
                # =========================
                original_img = cv2.imread(image_path)
                if original_img is not None:
                    # Ensure both images are the same size (in case there's any unexpected shape mismatch)
                    if original_img.shape == embedded_img.shape:
                        mse_value = calculate_mse(original_img, embedded_img)
                        psnr_value = calculate_psnr(mse_value)
                        ncc_value = calculate_ncc(original_img, embedded_img)
                        ssim_value = calculate_ssim(original_img, embedded_img)
                        print(f"\nMSE:  {mse_value:.4f}")
                        print(f"PSNR: {psnr_value:.2f} dB")
                        print(f"NCC   : {ncc_value:.4f}")
                        print(f"SSIM  : {ssim_value:.4f}")
                    else:
                        print("Warning: Original and stego image shapes differ; cannot compute MSE/PSNR.")
                else:
                    print("Warning: Could not load the original image for MSE/PSNR calculation.")

            except Exception as e:
                print(f"Error: {str(e)}")
                
        elif choice == "2":
            try:
                print("\nStarting extraction process...")
                start_time = time.time()  # ⏱️ Start timer

                extracted_message = steganography.extract_text("Embedded-image/stego_image.png")

                end_time = time.time()  # ⏱️ End timer
                elapsed_time = end_time - start_time
                print(f"✅ Message extracted successfully in {elapsed_time:.4f} seconds")

                with open("Decoded-message/extracted_text.txt", "w") as f:
                    f.write(extracted_message)
                print("Message extracted successfully!")
                print(f"Extracted message length: {len(extracted_message)} characters")
                print("Extracted message saved in: Decoded-message/extracted_text.txt")

            except Exception as e:
                print(f"Error: {str(e)}")
                
        else:
            print("Invalid choice! Please enter 1 for embedding, 2 for extraction, or 3 to exit.")


if __name__ == "__main__":
    main()
