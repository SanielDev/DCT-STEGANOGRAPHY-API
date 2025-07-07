import numpy as np
import cv2
import os
from scipy.fftpack import dct, idct
from Crypto.Cipher import AES
import base64
import secrets
import hashlib
import math
from Crypto.Util.Padding import pad, unpad
from skimage.metrics import structural_similarity as ssim


class ImprovedDCTSteganography:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.block_size = 8
        self.embedding_positions = []
        self.key_storage_path = "encryption_key.txt"

        # Standard JPEG Luminance Quantization Matrix
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

    # ==================== Chaotic Mapping (Logistic Map Scrambling) ====================
    def logistic_map_scramble(self, image, key=0.3):
        h, w = image.shape
        scrambled_image = np.zeros_like(image)

        x = key
        seq = np.zeros(h * w)

        for i in range(h * w):
            x = 3.99 * x * (1 - x)
            seq[i] = x

        indices = np.argsort(seq)

        flat_img = image.flatten()
        scrambled_img_flat = flat_img[indices]

        return scrambled_img_flat.reshape(h, w)

    # ==================== AES Encryption & Secure Key Storage ====================
    def generate_aes_key(self):
        key = secrets.token_bytes(16)
        hashed_key = hashlib.sha256(key).hexdigest()

        with open(self.key_storage_path, "w") as f:
            f.write(hashed_key)

        return key

    def aes_encrypt_image(self, image, key):
        """Encrypts an image using AES with proper padding."""
        cipher = AES.new(key, AES.MODE_ECB)
        
        img_bytes = image.tobytes()
        padded_bytes = pad(img_bytes, AES.block_size)  # ✅ Ensure proper padding

        encrypted_bytes = cipher.encrypt(padded_bytes)
        return base64.b64encode(encrypted_bytes)

    # ==================== Main Embedding & Encryption Pipeline ====================

    def _pad_image(self, image):
        h, w = image.shape
        new_h = ((h + self.block_size - 1) // self.block_size) * self.block_size
        new_w = ((w + self.block_size - 1) // self.block_size) * self.block_size
        padded = np.zeros((new_h, new_w))
        padded[:h, :w] = image
        return padded, (h, w)

    def _calculate_texture_mask(self, block):
        gradient_x = np.diff(block, axis=1, prepend=block[:, :1])
        gradient_y = np.diff(block, axis=0, prepend=block[:1, :])
        texture_strength = np.mean(np.abs(gradient_x)) + np.mean(np.abs(gradient_y))
        return 1 + (texture_strength / 128)

    def _get_embedding_strength(self, dct_block, position, texture_mask):
        q_value = self.quantization_matrix[position]
        local_variance = np.var(dct_block)
        return self.alpha * (q_value / 16) * (1 + local_variance / 1000) * texture_mask

    def _select_embedding_position(self, dct_block):
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
            byte = ''.join(map(str, message_bits[i:i+8]))
            message += chr(int(byte, 2))
        return message
    
    def get_max_message_length(self, image_path):
        image = load_image_gracefully(image_path)
        ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb_image[:, :, 0]
        blocks, _ = self._split_into_blocks(y_channel)
        return (len(blocks) - 16) // 8
    
    def embed(self, image_path, message):
        """Embeds a patient ID, scrambles the image, encrypts it, and saves all securely."""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")

        print("[1] Embedding patient ID using DCT...")
        ycrcb_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb_image[:, :, 0]
        h, w = y_channel.shape

        block_size = self.block_size
        bits = ''.join(format(ord(c), '08b') for c in message)
        index = 0

        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                if index < len(bits):
                    block = y_channel[i:i+block_size, j:j+block_size].astype(np.float32)
                    dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')

                    pos = self._select_embedding_position(dct_block)
                    dct_block[pos] = abs(dct_block[pos]) + 1 if bits[index] == '1' else -abs(dct_block[pos]) - 1

                    y_channel[i:i+block_size, j:j+block_size] = np.clip(idct(idct(dct_block.T, norm='ortho').T, norm='ortho'), 0, 255)
                    index += 1

        print("[2] Applying chaotic scrambling...")
        scrambled_image = self.logistic_map_scramble(y_channel, key=0.3)

        # ✅ FIX: Ensure Data is Converted to a Valid Image Format
        scrambled_image_uint8 = np.clip(scrambled_image, 0, 255).astype(np.uint8)

        # Save the scrambled image before encryption
        scrambled_save_path = os.path.join("Embedded-image", "scrambled_image.png")
        cv2.imwrite(scrambled_save_path, scrambled_image_uint8)
        print(f"✅ Scrambled image saved at: {scrambled_save_path}")

        print("[3] Encrypting the scrambled image using AES...")
        aes_key = self.generate_aes_key()
        encrypted_image = self.aes_encrypt_image(scrambled_image_uint8, aes_key)

        # Save both the encrypted image and the key
        encrypted_save_path = os.path.join("Embedded-image", "stego_image.enc")
        key_save_path = os.path.join("Embedded-image", "aes.key")
        
        with open(encrypted_save_path, "wb") as f:
            f.write(encrypted_image)
        with open(key_save_path, "wb") as f:
            f.write(aes_key)
        
        print(f"✅ Encrypted image saved at: {encrypted_save_path}")
        print(f"✅ AES key saved at: {key_save_path}")
        return scrambled_image_uint8


    # ==================== AES Decryption & Inverse Chaotic Mapping ====================
    def aes_decrypt_image(self, encrypted_bytes, key):
        """Decrypts AES-encrypted image data with correct padding removal."""
        cipher = AES.new(key, AES.MODE_ECB)
        
        encrypted_bytes = base64.b64decode(encrypted_bytes)
        decrypted_bytes = cipher.decrypt(encrypted_bytes)
        
        try:
            unpadded_bytes = unpad(decrypted_bytes, AES.block_size)  # ✅ Ensure correct unpadding
        except ValueError as e:
            raise ValueError("Decryption failed due to incorrect padding") from e

        return np.frombuffer(unpadded_bytes, dtype=np.uint8)

    def inverse_logistic_map(self, scrambled_image, key=0.3):
        h, w = scrambled_image.shape
        x = key
        seq = np.zeros(h * w)

        for i in range(h * w):
            x = 3.99 * x * (1 - x)
            seq[i] = x

        indices = np.argsort(seq)
        inverse_indices = np.argsort(indices)

        flat_img = scrambled_image.flatten()
        restored_img_flat = flat_img[inverse_indices]

        return restored_img_flat.reshape(h, w)

    def extract_text(self, encrypted_image_path):
        print("[1] Decrypting the image using AES...")

        # Read the encrypted image and AES key
        with open(encrypted_image_path, "rb") as f:
            encrypted_image = f.read()
        
        key_path = os.path.join("Embedded-image", "aes.key")
        with open(key_path, "rb") as f:
            aes_key = f.read()

        decrypted_image = self.aes_decrypt_image(encrypted_image, aes_key)

        print("[2] Restoring pixel positions via inverse chaotic mapping...")
        restored_image = self.inverse_logistic_map(decrypted_image.reshape(-1, -1), key=0.3)

        print("[3] Extracting embedded patient ID...")
        # Add your text extraction logic here
        # For now, returning placeholder
        return "Extracted message"

def ensure_directories_exist():
    directories = ['Images', 'Embedded-image', 'Decoded-message']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def load_image_gracefully(image_path):
    """
    Load an image with OpenCV. Raise an error if it fails.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}.")
    return image

def convert_jpeg_to_png_if_needed(image_path):
    """
    If the input image is JPEG (.jpg or .jpeg), convert it to a PNG file 
    in the same 'Images' folder and return the new path.
    Otherwise, return the original path.
    """
    ext = os.path.splitext(image_path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        # Convert to PNG
        new_path = os.path.splitext(image_path)[0] + "_converted.png"
        img = load_image_gracefully(image_path)
        cv2.imwrite(new_path, img)
        return new_path
    return image_path

def calculate_mse(original, stego):
    original_float = original.astype(np.float64)
    stego_float = stego.astype(np.float64)
    return np.mean((original_float - stego_float) ** 2)

def calculate_psnr(mse_value, max_pixel_value=255.0):
    if mse_value == 0:
        return float('inf')
    return 20 * math.log10(max_pixel_value / math.sqrt(mse_value))

def calculate_ncc(original, stego):
    orig_f = original.astype(np.float64).ravel()
    steg_f = stego.astype(np.float64).ravel()
    orig_mean = np.mean(orig_f)
    steg_mean = np.mean(steg_f)
    numerator = np.sum((orig_f - orig_mean) * (steg_f - steg_mean))
    denominator = np.sqrt(np.sum((orig_f - orig_mean)**2) * np.sum((steg_f - steg_mean)**2))
    if denominator == 0:
        return 0
    return numerator / denominator

def calculate_ssim(original, stego):
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
            input_image = input("Enter the image name (in 'Images' folder): ")
            image_path = os.path.join("Images", input_image)

            # Step 1: Convert JPEG to PNG if needed
            image_path = convert_jpeg_to_png_if_needed(image_path)

            try:
                print("The image path is: ", image_path)
                max_length = steganography.get_max_message_length(image_path)
                print(f"\nMaximum message length for this image: {max_length} characters")

                secret_message = input("Enter the secret message to hide: ")
                print(f"Message length: {len(secret_message)} characters")

                print("\nStarting embedding process...")
                stego_bgr = steganography.embed(image_path, secret_message)

                # Save final stego as PNG to preserve DCT changes
                stego_save_path = os.path.join("Embedded-image", "stego_image.png")
                cv2.imwrite(stego_save_path, stego_bgr)
                print(f"Text embedded successfully! Stego image saved at: {stego_save_path}")

                # ---- Metrics (optional) ----
                original_img = load_image_gracefully(image_path)
                if original_img.shape == stego_bgr.shape:
                    mse_val = calculate_mse(original_img, stego_bgr)
                    psnr_val = calculate_psnr(mse_val)
                    ncc_val = calculate_ncc(original_img, stego_bgr)
                    ssim_val = calculate_ssim(original_img, stego_bgr)

                    print(f"\nMSE:   {mse_val:.6f}")
                    print(f"PSNR:  {psnr_val:.2f} dB")
                    print(f"NCC:   {ncc_val:.4f}")
                    print(f"SSIM:  {ssim_val:.4f}")
                else:
                    print("Warning: shape mismatch, skipping metrics.")

            except Exception as e:
                print(f"Error: {e}")

        elif choice == "2":
            # Extraction
            print("\nStarting extraction process...")
            try:
                stego_image_path = os.path.join("Embedded-image", "stego_image.png")
                extracted_message = steganography.extract_text(stego_image_path)

                output_path = os.path.join("Decoded-message", "extracted_text.txt")
                with open(output_path, "w") as f:
                    f.write(extracted_message)

                print("Message extracted successfully!")
                print(f"Extracted message length: {len(extracted_message)}")
                print(f"Saved in: {output_path}")

            except Exception as e:
                print(f"Error: {e}")

        else:
            print("Invalid choice! Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()

