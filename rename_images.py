import os

image_dir = "Images"
valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

# Filter only image files, ignore hidden/system files like .DS_Store
files = [f for f in os.listdir(image_dir)
         if os.path.isfile(os.path.join(image_dir, f)) and
         os.path.splitext(f)[1].lower() in valid_exts]

# Sort alphabetically or however you want
files.sort()

# Rename to test_image_1.ext, test_image_2.ext ...
for idx, filename in enumerate(files, start=1):
    old_path = os.path.join(image_dir, filename)
    _, ext = os.path.splitext(filename)
    new_name = f"test_image_{idx}{ext}"
    new_path = os.path.join(image_dir, new_name)

    os.rename(old_path, new_path)
    print(f"✅ Renamed: {filename} ➜ {new_name}")
