import numpy as np
import cv2
import random

def text_to_bits(text):
    """Convert text to binary string"""
    # Convert each character to 8-bit binary
    binary = []
    for char in text:
        # Get 8-bit binary representation of each character
        char_binary = format(ord(char), '08b')
        # Convert each bit to integer and add to list
        binary.extend([int(bit) for bit in char_binary])
    return binary

def bits_to_text(bits):
    """Convert binary values back to text"""
    # Convert bits to string of 0s and 1s
    binary_str = ''.join(str(bit) for bit in bits)
    # Convert each 8 bits to a character
    text = ''
    for i in range(0, len(binary_str), 8):
        byte = binary_str[i:i+8]
        if byte == '00000000':  # Stop at delimiter
            break
        text += chr(int(byte, 2))
    return text

def embed_text(input_image_path, secret_text):
    """Embed text in image using DCT"""
    print("\n=== EMBEDDING PROCESS ===")
    print(f"Original text: {secret_text}")
    
    # Read and preprocess image
    im = cv2.imread(input_image_path)
    if len(im.shape) > 2:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.resize(im, (512, 512))
    
    # Convert message to binary
    binary_message = text_to_bits(secret_text)
    binary_message = binary_message + [0, 0, 0, 0, 0, 0, 0, 0]  # Add delimiter
    print(f"Binary message with delimiter: {binary_message}")
    print(f"Total bits to embed: {len(binary_message)}")
    
    # Generate random positions
    total_blocks = (512 * 512) // 64
    required_blocks = len(binary_message)
    random_positions = sorted(random.sample(range(total_blocks), required_blocks))
    print(f"Selected block positions (sorted): {random_positions}")
    
    # Save positions to file
    with open('Embedded-image/positions.txt', 'w') as f:
        f.write(','.join(map(str, random_positions)))
    
    # Embed bits
    dct_img = np.zeros_like(im, dtype=np.float32)
    block_index = 0
    bit_index = 0
    embedded_values = {}  # Dictionary to store embedded values for verification
    
    for i in range(0, 512, 8):
        for j in range(0, 512, 8):
            block = im[i:i+8, j:j+8].astype(np.float32)
            dct_block = cv2.dct(block)
            
            if block_index in random_positions:
                if bit_index < len(binary_message):
                    original_dc = dct_block[0, 0]
                    
                    # Make DC coefficient even/odd based on bit
                    if binary_message[bit_index] == 1:
                        modified_dc = np.floor(original_dc / 2) * 2 + 1  # Make odd
                    else:
                        modified_dc = np.floor(original_dc / 2) * 2  # Make even
                    
                    dct_block[0, 0] = modified_dc
                    
                    print(f"\nBlock {block_index} embedding:")
                    print(f"  Original DC: {original_dc}")
                    print(f"  Modified DC: {modified_dc}")
                    print(f"  Bit embedded: {binary_message[bit_index]}")
                    print(f"  Verification (mod 2): {int(modified_dc % 2)}")
                    
                    # Store the exact modified value
                    embedded_values[block_index] = modified_dc
                    
                    bit_index += 1
            
            dct_img[i:i+8, j:j+8] = dct_block
            block_index += 1
    
    # Save the embedded values
    np.save('Embedded-image/embedded_values.npy', embedded_values)
    
    # Inverse DCT and ensure values are properly quantized
    stego_img = np.zeros_like(im)
    for i in range(0, 512, 8):
        for j in range(0, 512, 8):
            block = cv2.idct(dct_img[i:i+8, j:j+8])
            stego_img[i:i+8, j:j+8] = np.round(block)  # Round to nearest integer
    
    return np.clip(stego_img, 0, 255).astype(np.uint8)

def extract_text(stego_image_path):
    """Extract hidden text from stego image"""
    print("\n=== EXTRACTION PROCESS ===")
    
    # Load the embedded values for verification
    embedded_values = np.load('Embedded-image/embedded_values.npy', allow_pickle=True).item()
    
    # Read stego image
    stego_img = cv2.imread(stego_image_path, cv2.IMREAD_GRAYSCALE)
    
    with open('Embedded-image/positions.txt', 'r') as f:
        random_positions = [int(x) for x in f.read().split(',')]
    
    extracted_bits = []
    block_index = 0
    
    for i in range(0, 512, 8):
        for j in range(0, 512, 8):
            if block_index in random_positions:
                block = stego_img[i:i+8, j:j+8].astype(np.float32)
                dct_block = cv2.dct(block)
                
                # Apply the same floor operation as in embedding
                current_dc = dct_block[0, 0]
                floored_dc = np.floor(current_dc / 2) * 2
                extracted_bit = 1 if abs(current_dc - (floored_dc + 1)) < abs(current_dc - floored_dc) else 0
                
                extracted_bits.append(extracted_bit)
                
                # Compare with embedded value
                if block_index in embedded_values:
                    embedded_dc = embedded_values[block_index]
                    print(f"\nBlock {block_index} extraction:")
                    print(f"  Current DC: {current_dc}")
                    print(f"  Floored DC: {floored_dc}")
                    print(f"  Embedded DC: {embedded_dc}")
                    print(f"  Extracted bit: {extracted_bit}")
                    print(f"  Verification: closest to {floored_dc if extracted_bit == 0 else floored_dc + 1}")
            
            block_index += 1
    
    # Convert bits to text
    text = ''
    for i in range(0, len(extracted_bits), 8):
        byte = extracted_bits[i:i+8]
        if byte == [0] * 8:  # Check for delimiter
            break
        char_code = int(''.join(str(bit) for bit in byte), 2)
        text += chr(char_code)
    
    return text



# if __name__ == "__main__":
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
        # Embedding process
        input_image = input("Enter the image name from Images folder: ")
        image_path = f"Images/{input_image}"
        secret_message = input("Enter the secret message to hide: ")
            
        try:
            print("\nStarting embedding process...")
            embedded_img = embed_text(image_path, secret_message)
            cv2.imwrite('Embedded-image/stego_image.png', embedded_img)
            print(f"Text embedded successfully!")
        except Exception as e:
            print(f"Error during embedding: {str(e)}")
                
    elif m == "2":
        # Extraction process
        try:
            print("\nStarting extraction process...")
            extracted_message = extract_text("Embedded-image/stego_image.png")
            with open("Decoded-message/extracted_text.txt", "w") as f:
                f.write(extracted_message)
            print(f"Message extracted successfully!")
            print(f"Extracted message saved in: Decoded-message/extracted_text.txt")
        except Exception as e:
            print(f"Error during extraction: {str(e)}")
                
    else:
        print("Invalid choice! Please enter 1 for embedding, 2 for extraction, or 3 to exit.")    