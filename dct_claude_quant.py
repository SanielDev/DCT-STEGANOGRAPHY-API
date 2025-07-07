import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct
import cv2
import os
import logging


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
        """Calculate texture masking factor based on local gradients"""
        gradient_x = np.diff(block, axis=1, prepend=block[:, :1])
        gradient_y = np.diff(block, axis=0, prepend=block[:1, :])
        texture_strength = np.mean(np.abs(gradient_x)) + np.mean(np.abs(gradient_y))
        return 1 + (texture_strength / 128)  # Normalize to reasonable range

    def _get_embedding_strength(self, dct_block, position, texture_mask):
        """Calculate adaptive embedding strength"""
        q_value = self.quantization_matrix[position]
        local_variance = np.var(dct_block)
        # Combine quantization, local variance, and texture masking
        return self.alpha * (q_value / 16) * (1 + local_variance/1000) * texture_mask

    def _select_embedding_position(self, dct_block):
        """Select optimal position for embedding based on HVS model"""
        # Calculate perceptual importance for each coefficient
        perceptual_mask = np.abs(dct_block) / self.quantization_matrix
        # Avoid DC coefficient and high frequencies
        perceptual_mask[0, 0] = 0
        perceptual_mask[7:, :] = 0
        perceptual_mask[:, 7:] = 0
        # Find position with highest capacity
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
        """Calculate maximum possible message length for an image"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb_image[:, :, 0]
        blocks, _ = self._split_into_blocks(y_channel)
        
        return (len(blocks) - 16) // 8

    def embed(self, image, message):
        """
        Embeds a text message into an image using Discrete Cosine Transform (DCT).
        - image: NumPy array (not a file path)
        - message: String to embed
        """
        # Check message length
        # max_length = self.get_max_message_length(image)
        max_length = 100
        if len(message) > max_length:
            raise ValueError(
                f"Message is too long!\n"
                f"Message length: {len(message)} characters\n"
                f"Maximum allowed length: {max_length} characters"
            )

        if image is None:
            raise ValueError("Received an invalid image for embedding.")

        ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)   # BGR to YCrCb conversion
        y_channel, cr_channel, cb_channel = cv2.split(ycrcb_image)   # Extracting the Y-channel for embedding

        blocks, original_shape = self._split_into_blocks(y_channel)   # we have array of 8x8 blocks and the original shape of the image
        bits = self._message_to_bits(message)

        # Reset embedding positions
        self.embedding_positions = []

        # Process each block
        modified_blocks = []
        for idx, block in enumerate(blocks):
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')

            if idx < len(bits):
                # Calculate texture mask
                texture_mask = self._calculate_texture_mask(block)

                # Select embedding position
                pos = self._select_embedding_position(dct_block)
                self.embedding_positions.append(pos)

                # Calculate adaptive strength
                strength = self._get_embedding_strength(dct_block, pos, texture_mask)

                # Embed bit
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

# def ensure_directories_exist():
#     directories = ['Images', 'Embedded-image', 'Decoded-message']
#     for directory in directories:
#         if not os.path.exists(directory):
#             os.makedirs(directory)
#             print(f"Created directory: {directory}")

def ensure_directories_exist():
    """Ensure necessary directories exist for storing results."""
    os.makedirs("Images", exist_ok=True)  # Ensure the folder exists
    os.makedirs("Embedded-image", exist_ok=True)
    os.makedirs("Decoded-message", exist_ok=True)

def load_image(image_path):
    """Load image using OpenCV, fallback to PIL if OpenCV fails."""
    image = cv2.imread(image_path)
    if image is None:
        logging.warning("⚠️ OpenCV failed to read image, using PIL for conversion...")
        img = Image.open(image_path)
        converted_path = os.path.join("Images", "converted_image.png")
        img.save(converted_path, "PNG")

        image = cv2.imread(converted_path)
        if image is None:
            raise ValueError(f"❌ OpenCV still cannot read {converted_path}")

    return image


def main():
    """CLI-based DCT Steganography."""
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
            input_image = input("Enter the image filename (must be inside 'Images' folder): ")
            image_path = os.path.abspath(os.path.join("Images", input_image)).strip()
            # image_path = f"Images/{input_image}"
            print(f"🔍 Checking image path: {image_path}")  # Debugging print
            image = cv2.imread(image_path)
            # image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)


            if image is None:
                print(f"❌ Error: OpenCV failed to read the image '{image_path}'")
                exit()

            if not os.path.exists(image_path):
                print(f"❌ Error: Image file '{input_image}' not found.")
                continue

            try:
                image = load_image(image_path)
                max_length = steganography.get_max_message_length(image)
                print(f"Maximum message length: {max_length} characters")

                secret_message = input("Enter the secret message to hide: ")
                if len(secret_message) > max_length:
                    print("❌ Error: Message too long!")
                    continue

                print("🔄 Embedding message...")
                embedded_img = steganography.embed(image, secret_message)

                output_path = os.path.join("Embedded-image", "stego_image.png")
                cv2.imwrite(output_path, embedded_img)
                print(f"✅ Text embedded successfully! Saved at: {output_path}")

            except Exception as e:
                print(f"❌ Error: {str(e)}")

        else:
            print("⚠️ Invalid choice! Enter 1 for embedding, 2 for extraction, or 3 to exit.")


if __name__ == "__main__":
    main()