import os
import cv2
import numpy as np
import joblib
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
import time

from dataset.data_loader import load_data
from preprocessing.preprocess import preprocess_pipeline
from preprocessing.extraction import extract_all_traditional_features
from segmentation.efcm_imp import segment_wbc
from models.hybrid_model import build_cnn_feature_extractor, HybridEnsembleModel

app = Flask(__name__)
CORS(app)

print("Loading ML Pipeline...")
# Load Deep Feature Extractor Topology
_, feature_extractor = build_cnn_feature_extractor()

# Load specific trained models
cnn_model = tf.keras.models.load_model('saved_models/cnn_model.h5', compile=False)
rf_model = joblib.load('saved_models/rf_classifier.pkl')
svm_model = joblib.load('saved_models/svm_classifier.pkl')

print("Models loaded successfully.")

def is_valid_blood_smear(img_bgr):
    """
    Heuristic algorithm to detect if an image is completely out-of-distribution (e.g. a dog, a car).
    Blood smears typically have very specific color telemetry (high red/pink, dark blue/purple nuclei).
    Returns (is_valid: bool, reason: str)
    """
    if img_bgr is None: return False, "Invalid image data."
    
    # Check for extreme lack of contrast/texture (solid colors, pure white etc)
    if np.std(img_bgr) < 5:
        return False, "Image lacks cellular texture or contrast. Not a valid microscopic slide."
        
    b, g, r = cv2.split(img_bgr)
    mean_b, mean_g, mean_r = np.mean(b), np.mean(g), np.mean(r)
    
    # Blood smears should never be strictly dominated by Green or Blue vs Red
    # Hematoxylin (Purple/Blue) and Eosin (Pink/Red) both contain heavy Red telemetry.
    if mean_g > mean_r * 1.3:
        return False, "Color profile invalid. Microscopic blood smears are typically pink/purple, but this image is dominated by greens/other spectrums."
        
    # Check for intense macro darkness (night photos, black screens)
    if np.mean(img_bgr) < 30:
        return False, "Illumination is far too low for a standard microscopic slide."
        
    return True, "Valid"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        start_time = time.time()
        
        # 1. Read Image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Could not decode image'}), 400
            
        # 1.5 Out-Of-Distribution (OOD) Rejection Filter
        is_smear, reason = is_valid_blood_smear(img)
        if not is_smear:
            return jsonify({'error': f"OOD REJECTED: {reason}"}), 400
            
        img = cv2.resize(img, (224, 224))
        
        # 2. Preprocess (CLAHE, Blur)
        processed = preprocess_pipeline(img)
        model_ready_img = processed['model_ready']
        clahe_enhanced = processed['clahe_enhanced']
        
        # 3. Segmentation (for traditional features)
        mask, cells_img = segment_wbc(clahe_enhanced)
        
        # Calculate strict morphological count
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cells = len(contours)
        
        # 4. Feature Extraction
        deep_features = feature_extractor.predict(np.expand_dims(model_ready_img, axis=0))
        trad_features = extract_all_traditional_features(clahe_enhanced, mask)
        trad_features = np.expand_dims(trad_features, axis=0)
        
        combined_features = np.concatenate([deep_features, trad_features], axis=1)
        
        # 5. Ensemble Prediction
        cnn_raw = cnn_model.predict(np.expand_dims(model_ready_img, axis=0))
        rf_raw  = rf_model.predict_proba(combined_features)
        svm_raw = svm_model.predict_proba(combined_features)
        
        # Squeeze down to Python floats!
        cnn_prob = float(np.squeeze(cnn_raw)[()])
        
        # Scikit-learn proba returns shape (1, 2)
        rf_prob = float(rf_raw[0][1] if rf_raw.shape[1] > 1 else rf_raw[0][0])
        svm_prob = float(svm_raw[0][1] if svm_raw.shape[1] > 1 else svm_raw[0][0])
        
        # Soft Voting
        mean_prob = (cnn_prob + rf_prob + svm_prob) / 3.0
        is_leukemia = bool(mean_prob >= 0.5)
        inference_time = round(time.time() - start_time, 2)
        
        return jsonify({
            "diagnosis": "Leukemia Detected" if is_leukemia else "Normal Cells",
            "isNormal": not is_leukemia,
            "confidence": round(mean_prob * 100, 1),
            "ensembleDetails": {
                "cnn": round(float(cnn_prob) * 100, 1),
                "rf": round(float(rf_prob) * 100, 1),
                "svm": round(float(svm_prob) * 100, 1)
            },
            "inferenceTime": f"{inference_time}s",
            "cellsSegmented": int(cells),
            "reportId": f"REP-{np.random.randint(100000, 999999)}",
            "error": None
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

# Triggering Flask Auto-Reload to Load New 526-Feature Weights!
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

