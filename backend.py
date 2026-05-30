import os
import time
import random
import numpy as np
import cv2
import pydicom
from scipy.ndimage import gaussian_filter
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import gdown

app = Flask(__name__)
CORS(app)

# ===============================
# MEMORY OPTIMIZATION (IMPORTANT)
# ===============================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ===============================
# FOLDERS
# ===============================
os.makedirs("uploads", exist_ok=True)
os.makedirs("static/processed", exist_ok=True)
os.makedirs("model", exist_ok=True)

# ===============================
# MODEL CONFIG
# ===============================
FILE_ID = "1m9j9KzQmyi293_Rlrv4jFlRrA_kMkGj1"
MODEL_PATH = "model/model_small.h5"

# ===============================
# LAZY MODEL LOADING (FIX CRASH)
# ===============================
model = None

def get_model():
    global model
    if model is None:
        print("[INFO] Loading model...")
        model = load_model(MODEL_PATH, compile=False)
        print("[INFO] Model loaded successfully")
    return model

# ===============================
# SAFE DOWNLOAD
# ===============================
def safe_download(file_id, output_path, retries=3):
    if os.path.exists(output_path):
        print(f"[INFO] Model already exists")
        return

    url = f"https://drive.google.com/uc?id={file_id}"

    for i in range(retries):
        try:
            print(f"[DOWNLOAD] attempt {i+1}")
            gdown.download(url, output_path, quiet=False)

            if os.path.exists(output_path) and os.path.getsize(output_path) > 10 * 1024 * 1024:
                print("[SUCCESS] Model downloaded")
                return

        except Exception as e:
            print(f"[ERROR] Attempt {i+1}: {e}")
            time.sleep(5)

    raise Exception("Model download failed")

# Download model once at startup
safe_download(FILE_ID, MODEL_PATH)

# ===============================
# LABELS
# ===============================
class_labels = [
    "Epidural",
    "Intraparenchymal",
    "Intraventricular",
    "Subarachnoid",
    "Subdural"
]

hemorrhage_descriptions = {
    "Epidural": "A collection of blood between the skull and dura mater. Often due to trauma and may require surgery.",
    "Intraparenchymal": "Bleeding within brain tissue, usually from hypertension or trauma. May cause swelling and need intensive care.",
    "Intraventricular": "Bleeding into the brain's ventricles, affecting cerebrospinal fluid circulation. Can lead to hydrocephalus.",
    "Subarachnoid": "Bleeding in the space between the brain and meninges. Often caused by an aneurysm rupture, requiring urgent intervention.",
    "Subdural": "Blood accumulation between the dura and arachnoid layer. Common in head trauma, requiring possible surgical drainage."
}

# ===============================
# IMAGE PROCESSING (OPTIMIZED)
# ===============================
def hu_normalization(image, slope, intercept):
    return image * slope + intercept

def window_image(image, center, width):
    min_val = center - width / 2
    max_val = center + width / 2
    img = np.clip(image, min_val, max_val)
    img = (img - min_val) / (max_val - min_val + 1e-6)
    return (img * 255).astype(np.uint8)

def sharpen(img):
    blur = gaussian_filter(img, sigma=1)
    return cv2.addWeighted(img, 1.5, blur, -0.5, 0)

def preprocess_dicom(path):
    dcm = pydicom.dcmread(path)
    image = dcm.pixel_array.astype(np.float32)

    hu = hu_normalization(image, dcm.RescaleSlope, dcm.RescaleIntercept)

    brain = sharpen(window_image(hu, 40, 80))
    subdural = sharpen(window_image(hu, 80, 200))
    bone = sharpen(window_image(hu, 600, 2800))

    brain = cv2.resize(brain, (256, 256))
    subdural = cv2.resize(subdural, (256, 256))
    bone = cv2.resize(bone, (256, 256))

    img = cv2.merge([brain, subdural, bone])
    img = np.expand_dims(img, axis=0)

    filename = f"processed_{random.randint(1000,9999)}.png"
    path_out = os.path.join("static/processed", filename)

    # FIX: replace matplotlib (heavy) with OpenCV
    cv2.imwrite(path_out, cv2.cvtColor(img[0], cv2.COLOR_RGB2BGR))

    return img, path_out

# ===============================
# SEVERITY
# ===============================
def classify_severity(score):
    if score < 0.3:
        return "Mild", "Observation only"
    elif score < 0.7:
        return "Moderate", "Hospital monitoring required"
    else:
        return "Severe", "Immediate medical attention required"

# ===============================
# ROUTES
# ===============================
@app.route("/processed/<filename>")
def get_image(filename):
    return send_from_directory("static/processed", filename)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    path = os.path.join("uploads", file.filename)
    file.save(path)

    try:
        model_input, img_path = preprocess_dicom(path)

        # 🔥 lazy model load (CRITICAL FIX)
        m = get_model()

        prediction = m.predict(model_input, verbose=0)

        label = class_labels[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        severity, advice = classify_severity(confidence)

        return jsonify({
            "image_url": img_path,
            "Predicted Hemorrhage Type": label,
            "Confidence": confidence,
            "Description": hemorrhage_descriptions[label],
            "Severity Level": severity,
            "Medical Suggestions": advice
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===============================
# START APP (RENDER SAFE)
# ===============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
