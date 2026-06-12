from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import json
import os
 
app = Flask(__name__)
 
# -----------------------------
# Load Model
# -----------------------------
model = tf.keras.models.load_model("waste_model.h5")
 
# -----------------------------
# Load Class Mapping (IMPORTANT FIX)
# -----------------------------
with open("class_indices.json") as f:
    class_indices = json.load(f)
 
# Convert to list (index → label)
classes = list(class_indices.keys())
 
# -----------------------------
# Stats Tracking
# -----------------------------
stats = {cls: 0 for cls in classes}

# -----------------------------
# Recycling Advice
# -----------------------------
def recycling_advice(label):
    if label in ["plastic", "glass", "metal"]:
        return "Put in recyclable bin"
    elif label in ["paper", "cardboard"]:
        return "Recycle or reuse"
    elif label in ["organic"]:
        return "Compostable waste"
    else:
        return "trash (Dispose of properly)"
 
# -----------------------------
# Prediction Function
# -----------------------------
def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.reshape(img, (1, 224, 224, 3))
 
    pred = model.predict(img)
    class_idx = np.argmax(pred)
 
    return classes[class_idx], float(np.max(pred))
 
# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return "✅ AI Garbage Classification API Running"
 
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
 
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
 
    path = "temp.jpg"
    file.save(path)
 
    label, confidence = predict_image(path)
    stats[label] += 1
 
    return jsonify({
        "type": label,
        "confidence": round(confidence, 2),
        "advice": recycling_advice(label)
    })
 
@app.route("/stats")
def get_stats():
    return jsonify(stats)
 
# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)