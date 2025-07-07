import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct
import cv2
import os

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
        self.embedding_positions = []
        self.psnr_threshold = 35.0  # Minimum acceptable PSNR
        self.jpeg_quality = 95      # JPEG quality for saving

    def _check_image_format(self, image_path):
        """Check image format and return appropriate parameters"""
        extension = image_path.lower().split('.')[-1]
        if extension not in ['jpg', 'jpeg', 'png']:
            raise ValueError("Unsupported image format. Please use JPG or PNG")
        return extension in ['jpg', 'jpeg']

    def _assess_quality(self, original, modified):
        """Calculate PSNR to ensure quality"""
        mse = np.mean((original - modified) ** 2)
        if mse == 0:
            return float('inf')
        PIXEL_MAX = 255.0
        psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
        return psnr

    def _check_embedding_robustness(self, dct_block, position, strength):
        """Verify if embedding will survive JPEG compression"""
        q_value = self.quantization_matrix[position]
        # Simulate JPEG quantization
        quantized_value = round(dct_block[position] / q_value) * q_value
        modified_value = quantized_value + strength
        # Check if value will survive quantization
        return abs(modified_value - quantized_value) > q_value/2

    def _pad_image(self, image):
        h, w = image.shape
        new_h = ((h + self.block_size - 1) // self.block_size) * self.block_size
        new_w = ((w + self.block_size - 1) // self.block_size) * self.block_size
        padded = np.zeros((new_h, new_w))
        padded[:h, :w] = image
        return padded, (h, w)

    def _calculate_texture_mask(self, block):
        """Calculate texture masking factor based on local gradients"""
        gradient_x = np.diff(block, axis=1, prepend=block[:, :1])
        gradient_y = np.diff(block, axis=0, prepend=block[:1, :])
        texture_strength = np.mean(np.abs(gradient_x)) + np.mean(np.abs(gradient_y))
        return 1 + (texture_strength / 128)

    def _get_embedding_strength(self, dct_block, position, texture_mask, is_jpeg):
        """Calculate adaptive embedding strength"""
        q_value = self.quantization_matrix[position]
        local_variance = np.var(dct_block)
        strength = self.alpha * (q_value / 16) * (1 + local_variance/1000) * texture_mask
        
        if is_jpeg:
            # Increase strength for JPEG to ensure survival after compression
            strength *= 1.5
        
        return strength

    def _select_embedding_position(self, dct_block, is_jpeg):
        """Select optimal position for embedding based on HVS model"""
        perceptual_mask = np.abs(dct_block) / self.quantization_matrix
        
        # Avoid DC coefficient and high frequencies
        perceptual_mask[0, 0] = 0
        if is_jpeg:
            # Be more conservative with frequency selection for JPEG
            perceptual_mask[6:, :] = 0
            perceptual_mask[:, 6:] = 0
        else:
            perceptual_mask[7:, :] = 0
            perceptual_mask[:, 7:] = 0
            
        return np.unravel_index(np.argmax(perceptual_mask), perceptual_mask.shape)

    def _split_into_blocks(self, image):
        # Ensure image is 2D (single channel)
        if len(image.shape) > 2:
            # If image has multiple channels, only use the first channel
            image = image[:, :, 0]
        
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
        """Calculate maximum possible message length for an image"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert to grayscale if needed
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        is_jpeg = self._check_image_format(image_path)
        if is_jpeg:
            # Reduce capacity estimation for JPEG to ensure reliability
            return ((len(self._split_into_blocks(image)[0]) - 16) // 8) * 85 // 100
        else:
            return (len(self._split_into_blocks(image)[0]) - 16) // 8

    def embed(self, image_path, message):
        # Format check
        is_jpeg = self._check_image_format(image_path)
        if is_jpeg:
            print("Note: Using JPEG format. Embedding strength will be adjusted for reliability.")
        
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
        
        # Convert to YCrCb color space
        ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb_image[:, :, 0]  # Use only Y channel
        cr_channel = ycrcb_image[:, :, 1]
        cb_channel = ycrcb_image[:, :, 2]
        
        blocks, original_shape = self._split_into_blocks(y_channel)
        bits = self._message_to_bits(message)
        
        self.embedding_positions = []
        modified_blocks = []
        
        for idx, block in enumerate(blocks):
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            
            if idx < len(bits):
                texture_mask = self._calculate_texture_mask(block)
                pos = self._select_embedding_position(dct_block, is_jpeg)
                strength = self._get_embedding_strength(dct_block, pos, texture_mask, is_jpeg)
                
                if is_jpeg:
                    # Check robustness and adjust strength if needed
                    while not self._check_embedding_robustness(dct_block, pos, strength):
                        strength *= 1.2
                        if strength > self.alpha * 5:  # Maximum strength limit
                            print(f"Warning: Block {idx} may not survive JPEG compression")
                            break
                
                if bits[idx] == 1:
                    dct_block[pos] = abs(dct_block[pos]) + strength
                else:
                    dct_block[pos] = -abs(dct_block[pos]) - strength
                
                self.embedding_positions.append(pos)
            
            modified_block = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')
            modified_blocks.append(modified_block)
        
        modified_y = self._reconstruct_from_blocks(modified_blocks, original_shape)
        modified_y = np.clip(modified_y, 0, 255).astype(np.uint8)
        
        # Quality assessment
        psnr = self._assess_quality(y_channel, modified_y)
        if psnr < self.psnr_threshold:
            print(f"Warning: Image quality might be affected (PSNR: {psnr:.2f}dB)")
        
        # Save embedding positions
        np.save('embedding_positions.npy', np.array(self.embedding_positions))
        
        # Reconstruct and save image
        modified_ycrcb = cv2.merge([modified_y, cr_channel, cb_channel])
        modified_image = cv2.cvtColor(modified_ycrcb, cv2.COLOR_YCrCb2BGR)
        
        if is_jpeg:
            cv2.imwrite('Embedded-image/stego_image.jpg', modified_image, 
                       [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            print(f"Saved as JPEG with quality {self.jpeg_quality}")
        else:
            cv2.imwrite('Embedded-image/stego_image.png', modified_image)
            print("Saved as PNG for maximum reliability")
        
        return modified_image

    def extract_text(self, stego_image_path):
        # Check format
        is_jpeg = self._check_image_format(stego_image_path)
        
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

def main():
    ensure_directories_exist()
    steganography = ImprovedDCTSteganography(alpha=0.1)
    
    while True:
        print("\n=== Format-Aware DCT Steganography Menu ===")
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
                embedded_img = steganography.embed(image_path, secret_message)
                print("Text embedded successfully!")
                
            except Exception as e:
                print(f"Error: {str(e)}")
                
        elif choice == "2":
            try:
                print("\nStarting extraction process...")
                extracted_message = steganography.extract_text("Embedded-image/stego_image.png")
                with open("Decoded-message/extracted_text.txt", "w") as f:
                    f.write(extracted_message)
                print("Message extracted successfully!")
                print(f"Extracted message length: {len(extracted_message)} characters")
                print(f"Extracted message saved in: Decoded-message/extracted_text.txt")
            except Exception as e:
                print(f"Error: {str(e)}")
                
        else:
            print("Invalid choice! Please enter 1 for embedding, 2 for extraction, or 3 to exit.")

if __name__ == "__main__":
    main()