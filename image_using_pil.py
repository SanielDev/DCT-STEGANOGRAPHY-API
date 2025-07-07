from PIL import Image
import os

image_path = "Images/pepper_copy.png"
converted_path = "Images/pepper_fixed.png"

try:
    img = Image.open(image_path)
    img.save(converted_path, "PNG")  # Convert and save as a new PNG
    print(f"✅ Image re-saved as: {converted_path}")
except Exception as e:
    print(f"❌ Error converting image: {str(e)}")
