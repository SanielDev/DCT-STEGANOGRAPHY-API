import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import math

def compute_metrics(original_path, stego_path):
    # Load both images
    original = cv2.imread(original_path)
    stego = cv2.imread(stego_path)

    if original is None or stego is None:
        raise ValueError("❌ Could not read one or both images.")

    # Convert to grayscale for SSIM
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    stego_gray = cv2.cvtColor(stego, cv2.COLOR_BGR2GRAY)

    # Ensure same shape
    if original_gray.shape != stego_gray.shape:
        raise ValueError("❌ Image dimensions do not match.")

    # MSE calculation
    mse = np.mean((original_gray - stego_gray) ** 2)

    # PSNR calculation
    if mse == 0:
        psnr = float("inf")
    else:
        PIXEL_MAX = 255.0
        psnr = 10 * math.log10((PIXEL_MAX ** 2) / mse)

    # SSIM calculation
    ssim_score = ssim(original_gray, stego_gray)

    return {
        "MSE": round(mse, 4),
        "PSNR": round(psnr, 4),
        "SSIM": round(ssim_score, 4)
    }

orig = "Images/test_image_2.png"
stego = "Embedded-image/embedded_fd4c5726969843edb0c95a5975e7ffb5.png"


results = compute_metrics(orig, stego)

print("📊 Quality Metrics:")
print(f"PSNR : {results['PSNR']} dB")
print(f"MSE  : {results['MSE']}")
print(f"SSIM : {results['SSIM']}")