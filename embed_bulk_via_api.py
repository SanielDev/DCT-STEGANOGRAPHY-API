import os
import random
import string
import requests

INPUT_FOLDER = "Images"
OUTPUT_FOLDER = "Embedded-image"
API_URL = "http://localhost:8000/embed"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def random_message(length=8):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

for filename in os.listdir(INPUT_FOLDER):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        input_path = os.path.join(INPUT_FOLDER, filename)
        message = random_message(8)

        with open(input_path, "rb") as f:
            files = {"file": (filename, f, "image/png")}
            data = {"patient_id": message}

            try:
                response = requests.post(API_URL, files=files, data=data)
                # result = response.json()
                try:
                    result = response.json()
                except Exception as e:
                    print(f"❌ Invalid JSON from server for {filename}:")
                    print(f"Status code: {response.status_code}")
                    print(f"Response body: {response.text}")
                    continue


                if "download_url" in result:
                    download_url = result["download_url"]
                    stego_response = requests.get(download_url)

                    if stego_response.status_code == 200:
                        output_path = os.path.join(OUTPUT_FOLDER, filename)
                        with open(output_path, "wb") as out_f:
                            out_f.write(stego_response.content)
                        print(f"✅ {filename} embedded with message '{message}'")
                    else:
                        print(f"❌ Failed to download embedded image for {filename}")
                else:
                    print(f"⚠️ Server error for {filename}: {result.get('error', 'Unknown error')}")

            except Exception as e:
                print(f"❌ Exception for {filename}: {e}")
