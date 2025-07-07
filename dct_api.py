from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from dct_claude_quant_1 import ImprovedDCTSteganography

app = Flask(__name__)

UPLOAD_FOLDER = "Embedded-image"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

dct_steganography = ImprovedDCTSteganography(alpha=0.1)

@app.route('/embed', methods=['POST'])
def embed_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    image = request.files['image']
    secret_message = request.form.get('secret_message')
    
    if not secret_message:
        return jsonify({"error": "Secret message is required"}), 400
    
    filename = secure_filename(image.filename)
    embedded_image_path = os.path.join(UPLOAD_FOLDER, f"{secret_message}_{filename}")
    
    temp_image_path = os.path.join("/tmp", filename)  # Use a temporary location to avoid modifying Images folder
    image.save(temp_image_path)
    
    try:
        embedded_img = dct_steganography.embed(temp_image_path, secret_message)
        cv2.imwrite(embedded_image_path, embedded_img)
        os.remove(temp_image_path)  # Clean up temporary image
        return jsonify({"message": "Image processed successfully", "saved_path": embedded_image_path}), 200
    except Exception as e:
        os.remove(temp_image_path)  # Ensure cleanup on failure
        return jsonify({"error": str(e)}), 500

@app.route('/extract', methods=['POST'])
def extract_text():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    image = request.files['image']
    filename = secure_filename(image.filename)
    temp_image_path = os.path.join("/tmp", filename)
    image.save(temp_image_path)
    
    try:
        extracted_message = dct_steganography.extract_text(temp_image_path)
        os.remove(temp_image_path)  # Clean up temporary image
        return jsonify({"message": "Extraction successful", "extracted_text": extracted_message}), 200   # return file_response
    except Exception as e:
        os.remove(temp_image_path)  # Ensure cleanup on failure
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
