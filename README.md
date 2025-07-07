# 🧠 DCT-Based Medical Report Steganography API

This project implements an **API-driven, secure and error-proof medical report handling system** using **DCT-based image steganography**. It is specifically designed to **embed patient IDs into medical images** (like X-rays, MRI scans, etc.) for **automated verification**, ensuring **privacy, accuracy, and tamper-proof storage** in hospital databases.

---

## 📌 Table of Contents

- [🧠 DCT-Based Medical Report Steganography API](#-dct-based-medical-report-steganography-api)
  - [📌 Table of Contents](#-table-of-contents)
  - [🎯 Problem Statement](#-problem-statement)
  - [💡 Solution Overview](#-solution-overview)
  - [🛠 Features](#-features)
  - [🚀 How to Run](#-how-to-run)
    - [🔧 Step 1: Create Virtual Environment](#-step-1-create-virtual-environment)
    - [📦 Step 2: Install Dependencies](#-step-2-install-dependencies)
    - [▶️ Step 3: Run API](#️-step-3-run-api)
  - [📤 API Endpoints](#-api-endpoints)
    - [➕ `/embed` \[POST\]](#-embed-post)
    - [🔍 `/extract` \[POST\]](#-extract-post)
  - [📁 Directory Structure](#-directory-structure)
  - [📦 Dependencies](#-dependencies)
  - [🔐 Security \& Use Case](#-security--use-case)
  - [📌 Future Enhancements](#-future-enhancements)
  - [🧑‍💻 Author](#-author)

---

## 🎯 Problem Statement

Hospitals currently rely on manual entry or non-validated automation for storing patient medical reports. This can result in:

- **Human errors** (wrong report mapped to wrong patient)
- **No security validation** between stored image and patient ID
- **No privacy control** in case of unauthorized report viewing

---

## 💡 Solution Overview

✅ This project uses **DCT (Discrete Cosine Transform)** and **adaptive steganography** to:

- **Embed the patient ID** securely into a medical image
- **Automate storage** of these verified stego-images
- Allow the **doctor to extract the embedded ID** for cross-validation before report viewing

The solution ensures **zero human dependency**, **no data mix-up**, and enhanced **patient confidentiality**.

---

## 🛠 Features

- 🔒 DCT-based adaptive image steganography
- 🧠 Texture and perceptual masking for imperceptibility
- 📷 Embed any patient ID into any image (X-ray, MRI, CT)
- ✅ REST API with `/embed` and `/extract` endpoints
- 🌐 No need for image to be in any specific folder
- 🔁 Support for image uploads and dynamic processing
- 💾 Stego-image returned automatically after upload
- 📜 Extraction matches embedded ID with ground truth

---

## 🚀 How to Run

### 🔧 Step 1: Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```
### 📦 Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```
### ▶️ Step 3: Run API
```bash
uvicorn dct_fast_api:app --reload
```

---

## 📤 API Endpoints

### ➕ `/embed` [POST]
Uploads an image and patient ID, returns a stego-image.

**Form Data:**
- `file`: image file (JPG/PNG)
- `patient_id`: patient ID string

**Returns:**  
Stego-image with the embedded patient ID.

---

### 🔍 `/extract` [POST]
Uploads a stego-image and extracts the embedded patient ID.

**Form Data:**
- `file`: stego-image

**Returns:**  
Extracted patient ID string.

---

## 📁 Directory Structure

```bash
.
├── dct_fast_api.py         # FastAPI app
├── dct_claude_quant.py     # Embedding & extraction logic
├── README.md               # This file
├── requirements.txt
├── Embedded-image/         # Output images
├── Decoded-message/        # Extracted text
├── Images/                 # Input images (optional)
├── screenshots/            # Screenshots for demonstration
```

---

## 📦 Dependencies

```bash
.
├── fastapi 
├── uvicorn     
├── opencv-python            
├── numpy
├── pillow       
├── scipy       
├── python-multipart 
```

** Installing dependencies **
```bash
pip install -r requirements.txt
```

---

## 🔐 Security & Use Case

- Ensures **privacy-preserving storage** of medical images  
- Prevents **unauthorized report access**  
- Protects against **mismatched patient-report mapping**  
- Automates the **workflow of hospital database entry**

---

## 📌 Future Enhancements

- 🔒 AES encryption of patient ID before embedding  
- 🖥️ GUI for hospital-side integration  
- 🔌 Plugin for PACS/RIS systems  
- 📜 Embed medical report hash + timestamp for audit trail  
- 📦 Embed multiple fields (name, ID, report date) in compressed form

---

## 🧑‍💻 Author

**Saniel Bhattarai**  
B.Tech in Computer Science Engineering  
Specialization: Information Security 
Final Year Capstone Project (2025)

**Riya Vaid**  
B.Tech in Computer Science Engineering  
Specialization: Information Security  
VIT Vellore  
Final Year Capstone Project (2025)