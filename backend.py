import os
import numpy as np
import cv2
import pydicom
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from tensorflow.keras.models import load_model
import random
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import gdown   # ✅ ADDED

app = Flask(__name__)
CORS(app)

# Ensure folders exist
output_dir = "static/processed"
os.makedirs(output_dir, exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("model", exist_ok=True)

# ===============================
# 🔥 GOOGLE DRIVE MODEL LOADER
# ===============================

MODEL_PATH_1 = "model/ICH-combined1.h5"
MODEL_PATH_2 = "model/ICH-combined2.h5"

FILE_ID_1 = "YOUR_FILE_ID_1"
FILE_ID_2 = "YOUR_FILE_ID_2"

def download_model(file_id, output_path):
    if not os.path.exists(output_path):
        print(f"Downloading {output_path} from Google Drive...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
        print(f"{output_path} downloaded successfully.")
    else:
        print(f"{output_path} already exists.")

# Download models before loading
download_model(FILE_ID_1, MODEL_PATH_1)
download_model(FILE_ID_2, MODEL_PATH_2)

# Load models
model1 = load_model(MODEL_PATH_1, compile=False)
model2 = load_model(MODEL_PATH_2, compile=False)

# ===============================
# Labels and descriptions
# ===============================

class_labels = ["Epidural", "Intraparenchymal", "Intraventricular", "Subarachnoid", "Subdural"]

hemorrhage_descriptions = {
    "Epidural": "A collection of blood between the skull and dura mater. Often due to trauma and may require surgery.",
    "Intraparenchymal": "Bleeding within brain tissue, usually from hypertension or trauma. May cause swelling and need intensive care.",
    "Intraventricular": "Bleeding into the brain's ventricles, affecting cerebrospinal fluid circulation. Can lead to hydrocephalus.",
    "Subarachnoid": "Bleeding in the space between the brain and meninges. Often caused by an aneurysm rupture, requiring urgent intervention.",
    "Subdural": "Blood accumulation between the dura and arachnoid layer. Common in head trauma, requiring possible surgical drainage."
}

# ===============================
# Helper functions
# ===============================

def classify_severity(prediction_score):
    if prediction_score < 0.3:
        return "Mild", "Observation and follow-up recommended. No immediate intervention required."
    elif 0.3 <= prediction_score < 0.7:
        return "Moderate", "Monitoring in a hospital setting is advised. CT scans may be needed."
    else:
        return "Severe", "Immediate medical attention required. Possible surgical intervention needed."

def hu_normalization(image, slope, intercept):
    return image * slope + intercept

def window_image(image, window_center, window_width):
    window_min = window_center - window_width / 2
    window_max = window_center + window_width / 2
    windowed = np.clip(image, window_min, window_max)
    windowed = (windowed - window_min) / (window_max - window_min)
    return (windowed * 255).astype(np.uint8)

def apply_sharpening(image):
    blurred = gaussian_filter(image, sigma=1)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

# ===============================
# DICOM preprocessing
# ===============================

def preprocess_dicom(dicom_path, target_size=(256, 256)):
    dcm = pydicom.dcmread(dicom_path)
    image = dcm.pixel_array.astype(np.float32)

    hu_image = hu_normalization(image, dcm.RescaleSlope, dcm.RescaleIntercept)

    brain_window = window_image(hu_image, 40, 80)
    subdural_window = window_image(hu_image, 80, 200)
    bone_window = window_image(hu_image, 600, 2800)

    brain = cv2.resize(apply_sharpening(brain_window), target_size)
    subdural = cv2.resize(apply_sharpening(subdural_window), target_size)
    bone = cv2.resize(apply_sharpening(bone_window), target_size)

    three_channel = cv2.merge([brain, subdural, bone])

    model1_input = np.expand_dims(np.stack([three_channel] * 5, axis=0), axis=0)
    model2_input = np.expand_dims(three_channel, axis=0)

    filename = f"processed_{random.randint(1000,9999)}.png"
    path = os.path.join("static/processed", filename)

    plt.imsave(path, three_channel)

    return model1_input, model2_input, path

# ===============================
# Routes
# ===============================

@app.route("/processed/<filename>")
def get_processed_image(filename):
    return send_from_directory("static/processed", filename)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    dicom_path = os.path.join("uploads", file.filename)
    file.save(dicom_path)

    try:
        model1_input, model2_input, processed_image_path = preprocess_dicom(dicom_path)

        predictions1 = model1.predict(model1_input)
        predictions2 = model2.predict(model2_input)

        avg_prediction = (predictions1 + predictions2) / 2   # ✅ FIXED

        predicted_label = class_labels[np.argmax(avg_prediction)]
        confidence = float(np.max(avg_prediction))

        severity, recommendation = classify_severity(confidence)
        description = hemorrhage_descriptions.get(predicted_label, "N/A")

        return jsonify({
            "image_url": processed_image_path,
            "Predicted Hemorrhage Type": predicted_label,
            "Confidence": confidence,
            "Description": description,
            "Severity Level": severity,
            "Medical Suggestions": recommendation
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===============================
# Run app
# ===============================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
