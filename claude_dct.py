import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct
import cv2
import os

class DCTSteganography:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.block_size = 8
        
    def _pad_image(self, image):
        h, w = image.shape
        new_h = ((h + self.block_size - 1) // self.block_size) * self.block_size
        new_w = ((w + self.block_size - 1) // self.block_size) * self.block_size
        padded = np.zeros((new_h, new_w))
        padded[:h, :w] = image
        return padded, (h, w)

    def _split_into_blocks(self, image):
        padded_image, original_shape = self._pad_image(image)
        height, width = padded_image.shape
        blocks = []
        for i in range(0, height, self.block_size):
            for j in range(0, width, self.block_size):
                block = padded_image[i:i+self.block_size, j:j+self.block_size]
                blocks.append(block)
        return blocks, original_shape
    
    def get_max_message_length(self, image_path):
        """Calculate maximum possible message length for an image"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb_image[:, :, 0]
        blocks, _ = self._split_into_blocks(y_channel)
        
        # Subtract 16 bits used for length encoding and divide by 8 (bits per character)
        return (len(blocks) - 16) // 8
    
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
        all_bits = length_bits + message_bits
        return [int(bit) for bit in all_bits]
    
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
    
    def embed(self, image_path, message):
        # First check if message can fit
        max_length = self.get_max_message_length(image_path)
        message_length = len(message)
        
        if message_length > max_length:
            raise ValueError(
                f"Message is too long!\n"
                f"Message length: {message_length} characters\n"
                f"Maximum allowed length: {max_length} characters\n"
                f"Message exceeds capacity by {message_length - max_length} characters"
            )
        
        # Read image in color
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert from BGR to YCrCb color space
        ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y_channel, cr_channel, cb_channel = cv2.split(ycrcb_image)
        
        blocks, original_shape = self._split_into_blocks(y_channel)
        bits = self._message_to_bits(message)
        
        modified_blocks = []
        for idx, block in enumerate(blocks):
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            
            if idx < len(bits):
                if bits[idx] == 1:
                    dct_block[4, 4] = abs(dct_block[4, 4]) + self.alpha
                else:
                    dct_block[4, 4] = -abs(dct_block[4, 4]) - self.alpha
            
            modified_block = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')
            modified_blocks.append(modified_block)
        
        modified_y = self._reconstruct_from_blocks(modified_blocks, original_shape)
        modified_y = np.clip(modified_y, 0, 255).astype(np.uint8)
        
        modified_ycrcb = cv2.merge([modified_y, cr_channel, cb_channel])
        modified_image = cv2.cvtColor(modified_ycrcb, cv2.COLOR_YCrCb2BGR)
        
        return modified_image
    
    def extract_text(self, stego_image_path):
        stego_image = cv2.imread(stego_image_path)
        if stego_image is None:
            raise ValueError(f"Could not load stego image from {stego_image_path}")
        
        ycrcb_image = cv2.cvtColor(stego_image, cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb_image[:, :, 0]
        
        blocks, _ = self._split_into_blocks(y_channel)
        extracted_bits = []
        for block in blocks:
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            extracted_bits.append(1 if dct_block[4, 4] > 0 else 0)
        
        return self._bits_to_message(extracted_bits)

def ensure_directories_exist():
    directories = ['Images', 'Embedded-image', 'Decoded-message']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def main():
    ensure_directories_exist()
    steganography = DCTSteganography(alpha=0.1)
    
    while True:
        print("\n=== Steganography Menu ===")
        print("1. Embed message in image")
        print("2. Extract message from image")
        print("3. Exit")
        
        m = input("Enter your choice (1-3): ")
        
        if m == "3":
            print("Exiting program...")
            break
            
        elif m == "1":
            input_image = input("Enter the image name from Images folder: ")
            image_path = f"Images/{input_image}"
            
            try:
                # Get and display max message length before asking for message
                max_length = steganography.get_max_message_length(image_path)
                print(f"\nMaximum message length for this image: {max_length} characters")
                
                secret_message = input("Enter the secret message to hide: ")
                print(f"Message length: {len(secret_message)} characters")
                
                print("\nStarting embedding process...")
                embedded_img = steganography.embed(image_path, secret_message)
                cv2.imwrite('Embedded-image/stego_image.png', embedded_img)
                print(f"Text embedded successfully!")
                
            except Exception as e:
                print(f"Error: {str(e)}")
                
        elif m == "2":
            try:
                print("\nStarting extraction process...")
                extracted_message = steganography.extract_text("Embedded-image/stego_image.png")
                with open("Decoded-message/extracted_text.txt", "w") as f:
                    f.write(extracted_message)
                print(f"Message extracted successfully!")
                print(f"Extracted message length: {len(extracted_message)} characters")
                print(f"Extracted message saved in: Decoded-message/extracted_text.txt")
            except Exception as e:
                print(f"Error: {str(e)}")
                
        else:
            print("Invalid choice! Please enter 1 for embedding, 2 for extraction, or 3 to exit.")

if __name__ == "__main__":
    main()