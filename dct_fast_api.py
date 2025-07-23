from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import numpy as np
import cv2
import os
import uuid
import logging
# from dct_claude_quant_1 import ImprovedDCTSteganography

# Import the class instead of functions
from dct_claude_quant import ImprovedDCTSteganography

app = FastAPI()

steganography = ImprovedDCTSteganography(alpha=0.012)


# Set up logging
logging.basicConfig(level=logging.INFO)

# Define a save path in the home directory
SAVE_DIR = os.path.expanduser("~/dct_files/")
os.makedirs(SAVE_DIR, exist_ok=True)  # Ensure the directory exists

@app.post("/embed")
async def embed_patient_id(file: UploadFile = File(...), patient_id: str = Form(...)):
    try:
        image_bytes = await file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            logging.error("❌ Failed to decode image!")
            return {"error": "Invalid image file"}

        # Save temporary file for get_max_message_length
        temp_path = os.path.join(SAVE_DIR, "temp.png")
        cv2.imwrite(temp_path, image)

        steganography = ImprovedDCTSteganography()
        max_length = steganography.get_max_message_length(temp_path)
        
        if len(patient_id) > max_length:
            return {"error": f"Patient ID too long. Maximum length: {max_length}"}
            
        embedded_image = steganography.embed(image, patient_id)

        filename = f"embedded_{uuid.uuid4().hex}.png"
        output_path = os.path.join(SAVE_DIR, filename)

        cv2.imwrite(output_path, embedded_image)
        logging.info(f"🧬 Embedding message: {patient_id} into {file.filename}")
        return {"download_url": f"http://localhost:8000/download/{filename}"}
        # return FileResponse(output_path, media_type="image/png", filename=filename)

    except Exception as e:
        logging.error(f"❌ Internal Server Error: {str(e)}", exc_info=True)
        return {"error": "An internal server error occurred"}

@app.post("/decode")
async def decode_patient_id(file: UploadFile = File(...)):
   """Decodes the hidden message from an uploaded image using the DCT function from `dct_claude_quant.py`."""
   try:
       logging.info("📥 Received image for decoding...")

       image_bytes = await file.read()
       np_arr = np.frombuffer(image_bytes, np.uint8)
       image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

       if image is None:
           logging.error("❌ Failed to decode image!")
           return {"error": "Invalid image file. Please upload a valid PNG or JPEG file."}

       logging.info("✅ Image loaded successfully. Starting DCT decoding...")

       # Save temporary file for extract_text
       temp_path = os.path.join(SAVE_DIR, "temp_decode.png")
       cv2.imwrite(temp_path, image)

       steganography = ImprovedDCTSteganography()
       extracted_message = steganography.extract_text(temp_path)

       if not extracted_message:
           logging.warning("⚠️ No hidden message found in the image.")
           return {"error": "No hidden message found in the image."}

       logging.info(f"✅ Extracted message: {extracted_message}")
       return {"extracted_message": extracted_message}

   except Exception as e:
       logging.error(f"❌ Internal Server Error: {str(e)}", exc_info=True)
       return {"error": "An internal server error occurred."}
    
# @app.post("/decode")
# async def decode_patient_id(file: UploadFile = File(...)):
#     try:
#         image_bytes = await file.read()
#         np_arr = np.frombuffer(image_bytes, np.uint8)
#         image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

#         if image is None:
#             logging.error("❌ Failed to decode image!")
#             return {"error": "Invalid image file"}

#         temp_path = os.path.join(SAVE_DIR, "temp_decode.png")
#         cv2.imwrite(temp_path, image)

#         steganography = ImprovedDCTSteganography()
#         extracted_message = steganography.extract_text(temp_path)

#         return {"extracted_message": extracted_message}

#     except Exception as e:
#         logging.error(f"❌ Internal Server Error: {str(e)}", exc_info=True)
#         return {"error": "An internal server error occurred"}

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Serves the embedded image for download."""
    file_path = os.path.join(SAVE_DIR, filename)

    print(f"Checking for file: {file_path}")

    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return {"error": f"File not found: {file_path}"}

    print(f"✅ File found, serving: {file_path}")
    return FileResponse(file_path, media_type="image/png", filename=filename)
