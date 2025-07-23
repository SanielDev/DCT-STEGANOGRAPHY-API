import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct
import cv2
import os
import logging
from typing import List
from bch_utils import bch_encode, bch_decode

# dct_claude_quant.py  (add above your class or into utils section)

def bits_to_bytes(bits: list[int]) -> bytes:
    # chop into groups of 8
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for bit in bits[i:i+8]:
            byte = (byte << 1) | bit
        out.append(byte)
    return bytes(out)

class ImprovedDCTSteganography:
    def __init__(self, alpha=0.012, repetition_factor=1): # before the alpha was 0.1 and no repetition_factor was there
        self.alpha = alpha
        self.repetition_factor = repetition_factor
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
    
    def _bytes_to_bits(self, data: bytes) -> list[int]:
        # return [ (b >> i) & 1 for b in buf for i in range(7,-1,-1) ]
        out = []
        for byte in data:
            for shift in range(7, -1, -1):          # big-endian
                out.append((byte >> shift) & 1)
        return out

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
        # perceptual_mask[0, 0] = 0
        # perceptual_mask[7:, :] = 0
        # perceptual_mask[:, 7:] = 0

        # --- newly added starts:
        perceptual_mask[:3, :] = 0     # zero rows 0,1,2
        perceptual_mask[6:, :] = 0     # zero rows 6,7
        perceptual_mask[:, :3] = 0     # zero cols 0,1,2
        perceptual_mask[:, 6:] = 0     # zero cols 6,7
        # --- newly added ends

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

    # def _message_to_bits(self, message):
    #     length_bits = format(len(message), '016b')
    #     message_bits = ''.join(format(ord(char), '08b') for char in message)
    #     return [int(bit) for bit in length_bits + message_bits]

    # Newly added
    def _message_to_bits(self, message):
        """16-bit length header + ASCII bits, then bit-repeat for robustness."""
        length_bits   = format(len(message), '016b')
        payload_bits  = ''.join(format(ord(c), '08b') for c in message)
        bitstream     = length_bits + payload_bits

        repeated = []
        for b in bitstream:
            repeated.extend([int(b)] * self.repetition_factor)
        return repeated


    # def _bits_to_message(self, bits):
    #     if len(bits) < 16:
    #         return ""
    #     length_bits = ''.join(map(str, bits[:16]))
    #     message_length = int(length_bits, 2)
        
    #     message_bits = bits[16:16 + message_length * 8]
    #     message = ""
    #     for i in range(0, len(message_bits), 8):
    #         if i + 8 <= len(message_bits):
    #             byte = ''.join(map(str, message_bits[i:i+8]))
    #             message += chr(int(byte, 2))
    #     return message

    # Newly added
    def _bits_to_message(self, bits):
        if len(bits) < 16 * self.repetition_factor:
            return ""

        # collapse N repeated bits → 1 bit (majority vote)
        def majority(chunk):
            return 1 if sum(chunk) >= (len(chunk) / 2) else 0

        singles = []
        for i in range(0, len(bits), self.repetition_factor):
            chunk = bits[i:i+self.repetition_factor]
            if len(chunk) < self.repetition_factor:
                break
            singles.append(majority(chunk))

        # first 16 bits = length
        if len(singles) < 16:
            return ""
        msg_len = int(''.join(map(str, singles[:16])), 2)
        need    = 16 + msg_len * 8
        if len(singles) < need:
            return ""

        data_bits = singles[16:need]
        chars = [
            chr(int(''.join(str(b) for b in data_bits[i:i+8]), 2))
            for i in range(0, len(data_bits), 8)
        ]
        return ''.join(chars)


    # def get_max_message_length(self, image_path):
    #     """Calculate maximum possible message length for an image"""
    #     image = cv2.imread(image_path)
    #     if image is None:
    #         raise ValueError(f"Could not load image from {image_path}")
        
    #     ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    #     y_channel = ycrcb_image[:, :, 0]
    #     blocks, _ = self._split_into_blocks(y_channel)
        
    #     return (len(blocks) - 16) // 8

    # Newly added
    def get_max_message_length(self, image_path):
        """
        Capacity (chars) = (available_blocks / r  − 16) / 8
        where r = repetition_factor
        """
        image  = load_image(image_path)
        y_chan = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        blocks, _ = self._split_into_blocks(y_chan)

        capacity_bits = len(blocks)               # 1 bit per block
        usable_bits   = capacity_bits // self.repetition_factor
        max_chars     = (usable_bits - 16) // 8
        return max(0, int(max_chars))
    
    # ───────────────────────────────────────────────────────────────────────────── (added on 15th july)
    def _select_blocks(self, y_plane: np.ndarray, keep_ratio: float = 0.4) -> np.ndarray:
        """
        Return the indices (ascending) of the top-variance 8×8 blocks.
        keep_ratio = 0.4  ⇒  use only the 40 % most textured blocks.
        """
        blocks, _  = self._split_into_blocks(y_plane)
        variances  = np.array([blk.var() for blk in blocks])
        k          = max(1, int(len(blocks) * keep_ratio))
        best       = np.argsort(variances)[-k:]          # k largest variances
        return np.sort(best)                             # keep original order
    # ─────────────────────────────────────────────────────────────────────────────

    # def embed(self, image_in, message):
    #     """
    #     Embeds a text message into an image using Discrete Cosine Transform (DCT).
    #     - image: NumPy array (not a file path)
    #     - message: String to embed
    #     """
    #     # -------------- (Beginning of newly added section)
    #     if isinstance(image_in, str):                      # file path given
    #         image = load_image(image_in)
    #         # max_len = self.get_max_message_length(image_in) # commented on july 15th
    #     else:                                              # numpy array given
    #         image = image_in
    #     # capacity = ((#blocks) – 16) // 8  (same maths you use elsewhere)
    #     y_chan = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    #     # ---------------- (newly added on july 15th)
    #     good_idx = self._select_blocks(y_chan, keep_ratio=0.40)   # <── NEW
    #     n_blocks = len(good_idx)
    #     # ------------------------------------------
    #     # n_blocks = len(self._split_into_blocks(y_chan)[0])
    #     # max_len = (n_blocks - 16) // 8
    #     max_len  = ((n_blocks // self.repetition_factor) - 16) // 8   # capacity (chars)

    # # ── 2. Capacity check ────────────────────────────────────────────────
    #     if len(message) > max_len:
    #         raise ValueError(
    #             f"Message too long ({len(message)} chars). "
    #             f"Capacity for this image is {max_len} chars.")
        
    #     # ----------- (End of newly added section)
    #     # Check message length
    #     # max_length = self.get_max_message_length(image)
    #     max_length = 100
    #     if len(message) > max_length:
    #         raise ValueError(
    #             f"Message is too long!\n"
    #             f"Message length: {len(message)} characters\n"
    #             f"Maximum allowed length: {max_length} characters"
    #         )

    #     if image is None:
    #         raise ValueError("Received an invalid image for embedding.")

    #     ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)   # BGR to YCrCb conversion
    #     y_channel, cr_channel, cb_channel = cv2.split(ycrcb_image)   # Extracting the Y-channel for embedding

    #     blocks, original_shape = self._split_into_blocks(y_channel)   # we have array of 8x8 blocks and the original shape of the image
    #     bits = self._message_to_bits(message)

    #     # Reset embedding positions
    #     self.embedding_positions = []

    #     # Process each block
    #     # modified_blocks = []

    #     # for idx, block in enumerate(blocks):
    #     #     dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')

    #     #     if idx < len(bits):
    #     #         # Calculate texture mask
    #     #         texture_mask = self._calculate_texture_mask(block)

    #     #         # Select embedding position
    #     #         pos = self._select_embedding_position(dct_block)
    #     #         self.embedding_positions.append(pos)

    #     #         # Calculate adaptive strength
    #     #         strength = self._get_embedding_strength(dct_block, pos, texture_mask)

    #     #         # ── NEW: damp strength for very small coefficients ───────────
    #     #         if abs(dct_block[pos]) < 10:          # weak-energy bin
    #     #             strength *= 0.5                   # embed more gently
    #     #         # -------------------------------------------------------------

    #     #         # Embed bit
    #     #         if bits[idx] == 1:
    #     #             dct_block[pos] = abs(dct_block[pos]) + strength
    #     #         else:
    #     #             dct_block[pos] = -abs(dct_block[pos]) - strength

    #     #     modified_block = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')
    #     #     modified_blocks.append(modified_block)

    #     # ----- (Newly added)
    #     # bit_idx = 0                              # pointer into bit-stream
    #     # modified_blocks = []
    #     # for block in blocks:
    #     #     # -------- stop early when all bits are written -----------------
    #     #     if bit_idx >= len(bits):
    #     #         modified_blocks.append(block)
    #     #         continue
    #     #     # ---------------------------------------------------------------

    #     #     dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')

    #     #     # 1) pick strongest coef in rows/cols 3-5
    #     #     pos = self._select_embedding_position(dct_block)

    #     #     # 2) if that coef is weak (<12), look for another in SAME block
    #     #     strength_scale = 1.0
    #     #     if abs(dct_block[pos]) < 12:
    #     #         dct_abs = np.abs(dct_block).copy()
    #     #         # keep only rows/cols 3-5
    #     #         dct_abs[:3, :] = dct_abs[6:, :] = 0
    #     #         dct_abs[:, :3] = dct_abs[:, 6:] = 0
    #     #         dct_abs[pos]   = 0                      # zero current pick
    #     #         alt_pos = np.unravel_index(np.argmax(dct_abs), dct_abs.shape)

    #     #         if abs(dct_block[alt_pos]) >= 15:
    #     #             pos = alt_pos                      # switch to stronger coef
    #     #         else:
    #     #             strength_scale = 0.5               # stay here but embed gently

    #     #     # 3) compute (and optionally damp) strength
    #     #     texture_mask = self._calculate_texture_mask(block)
    #     #     strength = self._get_embedding_strength(dct_block, pos, texture_mask) * strength_scale
    #     #     if abs(dct_block[pos]) < 10:               # extra safety
    #     #         strength *= 0.5

    #     #     # 4) embed current bit
    #     #     bit = bits[bit_idx]
    #     #     if bit == 1:
    #     #         dct_block[pos] =  abs(dct_block[pos]) + strength
    #     #     else:
    #     #         dct_block[pos] = -abs(dct_block[pos]) - strength

    #     #     # 5) record successful embed & advance
    #     #     self.embedding_positions.append(pos)
    #     #     bit_idx += 1

    #     #     # 6) inverse-DCT and store modified block
    #     #     modified_blocks.append(
    #     #         idct(idct(dct_block.T, norm='ortho').T, norm='ortho')
    #     #     )
    #     # # -------- capacity sanity check -----------------------------------
    #     # if bit_idx < len(bits):
    #     #     raise ValueError("Image capacity exhausted before embedding finished")
                
    #     # # --- (added ends)

    #     # --------------------------- (added on july 15th)
    #             # ------------------------------------------------------------------
    #     modified_blocks = []
    #     bit_idx = 0                                    # pointer into payload

    #     for blk_idx, block in enumerate(blocks):

    #         # stop when every bit is written
    #         if bit_idx >= len(bits):
    #             modified_blocks.append(block)
    #             continue

    #         dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')

    #         # ----- pick TWO reasonably strong coeffs in rows/cols 3-5 -----
    #         positions = []
    #         dct_abs   = np.abs(dct_block).copy()
    #         dct_abs[:3, :] = dct_abs[6:, :] = 0
    #         dct_abs[:, :3] = dct_abs[:, 6:] = 0

    #         for _ in range(2):                              # need a pair
    #             pos = np.unravel_index(np.argmax(dct_abs), dct_abs.shape)
    #             positions.append(pos)
    #             dct_abs[pos] = 0                            # zero-out & pick next

    #         # strength scaling (use the *stronger* of the two as reference)
    #         base_coef = max(abs(dct_block[p]) for p in positions)
    #         if base_coef < 12:                  # both are weak → embed gently
    #             str_scale = 0.5
    #         else:
    #             str_scale = 1.0

    #         texture_mask = self._calculate_texture_mask(block)
    #         strength = self._get_embedding_strength(dct_block, positions[0],
    #                                                 texture_mask) * str_scale

    #         # --------------- embed current bit into BOTH coeffs ------------
    #         bit = bits[bit_idx]
    #         for pos in positions:
    #             if bit == 1:
    #                 dct_block[pos] =  abs(dct_block[pos]) + strength
    #             else:
    #                 dct_block[pos] = -abs(dct_block[pos]) - strength

    #         # record (block-idx , pos) **for each coeff**  ▼ CHANGED
    #         # for pos in positions:
    #         #     self.embedding_positions.append((blk_idx, pos))
    #         for pos in positions:
    #             self.embedding_positions.append((int(blk_idx),
    #                                             int(pos[0]),
    #                                             int(pos[1])))

    #         bit_idx += 1                                            # next bit

    #         modified_blocks.append(
    #             idct(idct(dct_block.T, norm='ortho').T, norm='ortho')
    #         )

    #     # -------- capacity sanity check -----------------------------------
    #     if bit_idx < len(bits):
    #         raise ValueError("Image capacity exhausted before embedding finished")

    #     # -----------------------------------(marks the end of block added on july 15th)

    #     # Reconstruct image
    #     modified_y = self._reconstruct_from_blocks(modified_blocks, original_shape)
    #     modified_y = np.clip(modified_y, 0, 255).astype(np.uint8)

    #     # # Save embedding positions
    #     # np.save('embedding_positions.npy', np.array(self.embedding_positions))

    #     # --------------------------------(Newly added on 15th july)
    #     # ---------- Save embedding positions as an (N, 3) int-array ----------
    #     #     each row = (block_idx, row_in_block, col_in_block)
    #     embedding_arr = np.asarray(self.embedding_positions, dtype=np.int16)
    #     np.save("embedding_positions.npy", embedding_arr)

    #     # --------------------------------(marks the end of the block added on 15th july)

    #     # Reconstruct color image
    #     modified_ycrcb = cv2.merge([modified_y, cr_channel, cb_channel])
    #     modified_image = cv2.cvtColor(modified_ycrcb, cv2.COLOR_YCrCb2BGR)

    #     return modified_image

    # ----------------------------------------------(with reed-solomons) 16th july
    def embed(self, image_in, message: str):
        """
        Embed *message* into *image_in* using the two-coeff DCT scheme
        + BCH(15,11) forward-error-correction.
        • image_in : ndarray or path-string
        • message  : plain-text (latin-1) string
        """

        # ── 0. load / normalise the input image ────────────────────────────────
        if isinstance(image_in, str):
            image = load_image(image_in)           # your helper
        else:
            image = image_in.copy()

        if image is None:
            raise ValueError("Received an invalid image for embedding")

        # ── 1. capacity check ---------------------------------------------------
        y_chan   = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        good_idx = self._select_blocks(y_chan, keep_ratio=0.40)   # mid-texture 40 %
        capacity_bits = max(len(good_idx) - 16, 0)                # 16-bit guard
        # BCH-encode **before** computing the bit-length
        payload_bytes = bch_encode(message)
        bits          = self._bytes_to_bits(payload_bytes)

        if len(bits) > capacity_bits:
            raise ValueError(f"Message too long ({len(message)} chars) – "
                            f"capacity is {capacity_bits//8} chars")

        # ── 2. prep the Y channel blocks ---------------------------------------
        ycrcb                    = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y_plane, cr_plane, cb_plane = cv2.split(ycrcb)
        blocks, orig_shape       = self._split_into_blocks(y_plane)

        self.embedding_positions = []
        modified_blocks          = []
        bit_idx                  = 0
        good_set                 = set(good_idx)        # O(1) lookup

        # ── 3. main embedding loop ---------------------------------------------
        for blk_idx, block in enumerate(blocks):

            # -- skip blocks we decided not to use
            if blk_idx not in good_set:
                modified_blocks.append(block)
                continue

            # -- stop once all bits have been written
            if bit_idx >= len(bits):
                modified_blocks.append(block)
                continue

            dct_blk = dct(dct(block.T, norm="ortho").T, norm="ortho")

            # pick TWO strongest mid-band coefficients
            positions, dct_abs = [], np.abs(dct_blk).copy()
            dct_abs[:3, :] = dct_abs[6:, :] = 0
            dct_abs[:, :3] = dct_abs[:, 6:] = 0
            for _ in range(2):
                pos = np.unravel_index(np.argmax(dct_abs), dct_abs.shape)
                positions.append(pos)
                dct_abs[pos] = 0

            # adaptive strength
            base_coef = max(abs(dct_blk[p]) for p in positions)
            str_scale = 0.5 if base_coef < 12 else 1.0
            tex_mask  = self._calculate_texture_mask(block)
            strength  = self._get_embedding_strength(dct_blk, positions[0],
                                                    tex_mask) * str_scale

            # write one bit into *both* coefficients
            bit = bits[bit_idx]
            for pos in positions:
                dct_blk[pos] = ( abs(dct_blk[pos]) + strength
                                if bit == 1 else
                                -abs(dct_blk[pos]) - strength )

            # record (block,row,col) for each coefficient
            for pos in positions:
                self.embedding_positions.append((int(blk_idx),
                                                int(pos[0]),
                                                int(pos[1])))

            bit_idx += 1
            modified_blocks.append(
                idct(idct(dct_blk.T, norm="ortho").T, norm="ortho")
            )

        # ── 4. sanity check -----------------------------------------------------
        if bit_idx < len(bits):
            raise ValueError("Image capacity exhausted before embedding finished")

        # ── 5. rebuild image & save positions -----------------------------------
        mod_y = self._reconstruct_from_blocks(modified_blocks, orig_shape)
        mod_y = np.clip(mod_y, 0, 255).astype(np.uint8)

        np.save("embedding_positions.npy",
                np.asarray(self.embedding_positions, dtype=np.int16))

        merged = cv2.merge([mod_y, cr_plane, cb_plane])
        return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
    # ----------------------------------------------


    # def extract_text(self, stego_image_path):
    #     # Load embedding positions
    #     try:
    #         embedding_positions = np.load('embedding_positions.npy')
    #     except:
    #         raise ValueError("Could not load embedding positions file")
            
    #     stego_image = cv2.imread(stego_image_path)
    #     if stego_image is None:
    #         raise ValueError(f"Could not load stego image from {stego_image_path}")
        
    #     ycrcb_image = cv2.cvtColor(stego_image, cv2.COLOR_BGR2YCrCb)
    #     y_channel = ycrcb_image[:, :, 0]
        
    #     blocks, _ = self._split_into_blocks(y_channel)
    #     extracted_bits = []
        
    #     for idx, block in enumerate(blocks):
    #         if idx < len(embedding_positions):
    #             dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
    #             pos = tuple(embedding_positions[idx].astype(int))
    #             extracted_bits.append(1 if dct_block[pos] > 0 else 0)
        
    #     return self._bits_to_message(extracted_bits)

    # ---------------------------- (Newly added on 15th of july)
    # ──────────────────────────────────────────────────────────────────────
    # def extract_text(self, stego_image_path: str) -> str:
    #     """
    #     Extract the embedded message from *stego_image_path*.

    #     Each payload bit is stored in TWO DCT coefficients; we majority-vote
    #     them, rebuild the byte stream, then BCH-decode back to the original
    #     text.
    #     """
    #     # 1. load the (block,row,col) table ───────────────────────────────────
    #     try:
    #         pos_arr = np.load("embedding_positions.npy")      # shape (N,3)
    #     except Exception as e:
    #         raise ValueError("Could not load embedding_positions.npy") from e

    #     if pos_arr.ndim != 2 or pos_arr.shape[1] != 3 or len(pos_arr) % 2 != 0:
    #         raise ValueError("Corrupt embedding_positions format")

    #     pos_arr = pos_arr.astype(int)

    #     # 2. read stego & pre-compute DCT of all blocks ───────────────────────
    #     img = cv2.imread(stego_image_path)
    #     if img is None:
    #         raise ValueError(f"Could not read stego image: {stego_image_path}")

    #     y_plane = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    #     blocks, _ = self._split_into_blocks(y_plane)

    #     dct_blocks = [dct(dct(b.T, norm="ortho").T, norm="ortho") for b in blocks]

    #     # 3. majority vote for each coefficient *pair* ────────────────────────
    #     bits_out = []
    #     for i in range(0, len(pos_arr), 2):
    #         blk1, r1, c1 = pos_arr[i]
    #         blk2, r2, c2 = pos_arr[i + 1]

    #         coef1 = dct_blocks[blk1][r1, c1]
    #         coef2 = dct_blocks[blk2][r2, c2]

    #         vote = (1 if coef1 > 0 else -1) + (1 if coef2 > 0 else -1)
    #         bits_out.append(1 if vote > 0 else 0)

    #     # 4. rebuild byte-stream  ➜  BCH-decode  ➜  text ──────────────────────
    #     raw_bytes = bits_to_bytes(bits_out)
    #     msg = bch_decode(raw_bytes)            # returns str or None
    #     if msg is None:
    #         # raise ValueError("BCH decode failed (too many bit-errors)")
    #         return ""

    #     return msg

    # Claude code for extraction -----------------------------
    def extract_text(self, stego_image_path: str) -> str:
        """
        Extract the embedded message from *stego_image_path*.
        Each payload bit is stored in TWO DCT coefficients; we majority-vote them,
        rebuild the byte stream, then BCH-decode back to the original text.
        """
        # 1. load the (block,row,col) table ───────────────────────────────────
        try:
            pos_arr = np.load("embedding_positions.npy")  # shape (N,3)
        except Exception as e:
            raise ValueError("Could not load embedding_positions.npy") from e
        
        if pos_arr.ndim != 2 or pos_arr.shape[1] != 3 or len(pos_arr) % 2 != 0:
            raise ValueError("Corrupt embedding_positions format")
        
        pos_arr = pos_arr.astype(int)
        
        # 2. read stego & pre-compute DCT of all blocks ───────────────────────
        img = cv2.imread(stego_image_path)
        if img is None:
            raise ValueError(f"Could not read stego image: {stego_image_path}")
        
        y_plane = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        blocks, *_ = self._split_into_blocks(y_plane)
        dct_blocks = [dct(dct(b.T, norm="ortho").T, norm="ortho") for b in blocks]
        
        # 3. TRUE majority vote for each coefficient *pair* ───────────────────
        bits_out = []
        for i in range(0, len(pos_arr), 2):
            blk1, r1, c1 = pos_arr[i]
            blk2, r2, c2 = pos_arr[i + 1]
            
            coef1 = dct_blocks[blk1][r1, c1]
            coef2 = dct_blocks[blk2][r2, c2]
            
            # Count votes for bit = 1 (positive coefficients)
            positive_votes = (1 if coef1 > 0 else 0) + (1 if coef2 > 0 else 0)
            
            # True majority voting: more than half the votes
            if positive_votes > 1:  # 2 out of 2 votes for positive (bit = 1)
                bits_out.append(1)
            else:  # 0 or 1 out of 2 votes for positive (bit = 0)
                bits_out.append(0)
        
        # 4. rebuild byte-stream ➜ BCH-decode ➜ text ──────────────────────────
        raw_bytes = bits_to_bytes(bits_out)
        msg = bch_decode(raw_bytes)  # returns str or None
        
        if msg is None:
            # raise ValueError("BCH decode failed (too many bit-errors)")
            return ""
        
        return msg

    # ──────────────────────────────────────────────────────────────────────
    # ------------------------------------------------------- (Marks the end of extract_text() added on 15th july)

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