import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf

from dataset.data_loader import load_data
from preprocessing.preprocess import preprocess_pipeline
from segmentation.efcm_imp import segment_wbc
from preprocessing.extraction import extract_all_traditional_features
from models.hybrid_model import build_cnn_feature_extractor, HybridEnsembleModel, build_autoencoder
from utils.metrics import evaluate_classification, plot_confusion_matrix, calculate_autoencoder_reconstruction_error

EPOCHS = 20
BATCH_SIZE = 16

def extract_features_for_dataset(X, feature_extractor, model_ready_X=None):
    """
    Extract both CNN deep features and traditional features (Texture, Shape, Statistical).
    """
    print("Extracting Features...")
    deep_features = feature_extractor.predict(model_ready_X)
    
    # Extract traditional features manually
    traditional_features = []
    print("Processing traditional features + segmentation masks...")
    for img in tqdm(X):
        mask, _ = segment_wbc(img)
        # We need the 8-bit unnormalized image for traditional extraction
        trad_feats = extract_all_traditional_features(img, mask)
        traditional_features.append(trad_feats)
        
    traditional_features = np.array(traditional_features)
    
    # Combine Deep + Traditional
    combined_features = np.concatenate([deep_features, traditional_features], axis=1)
    return combined_features


def main():
    print("=============================================")
    print(" Leukemia Detection ML Pipeline Verification ")
    print("=============================================")
    
    # 1. Dataset Loading
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data('./data')
    
    # If no real data, we simulate a tiny random set of RGB images for script verification
    if len(X_train) > 0 and type(X_train[0]) == str and "mock_img" in X_train[0]:
        print("Using random noise images to verify pipeline flow since no Kaggle data present locally...")
        X_train = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(50)]
        X_val = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(10)]
        X_test = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(10)]
        y_train = np.random.randint(0, 2, size=(50,))
        y_val = np.random.randint(0, 2, size=(10,))
        y_test = np.random.randint(0, 2, size=(10,))
    
    # 2. Preprocessing
    print("Preprocessing Train Data...")
    X_train_processed = [preprocess_pipeline(img) for img in X_train]
    X_train_model = np.array([res['model_ready'] for res in X_train_processed])
    X_train_clahe = np.array([res['clahe_enhanced'] for res in X_train_processed])
    
    print("Preprocessing Validation Data...")
    X_val_processed = [preprocess_pipeline(img) for img in X_val]
    X_val_model = np.array([res['model_ready'] for res in X_val_processed])
    X_val_clahe = np.array([res['clahe_enhanced'] for res in X_val_processed])
    
    print("Preprocessing Test Data...")
    X_test_processed = [preprocess_pipeline(img) for img in X_test]
    X_test_model = np.array([res['model_ready'] for res in X_test_processed])
    X_test_clahe = np.array([res['clahe_enhanced'] for res in X_test_processed])

    
    # 3. Model Building & Feature Extraction
    print("Building ResNet50 Classifier & Autoencoder...")
    cnn_model, feature_extractor = build_cnn_feature_extractor()
    cnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
                      loss='binary_crossentropy', metrics=['accuracy'])
                      
    autoencoder = build_autoencoder()
    autoencoder.compile(optimizer='adam', loss='mse')
    
    # 4. Training Primary CNN
    print("Training CNN...")
    history = cnn_model.fit(
        X_train_model, y_train,
        validation_data=(X_val_model, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )
    
    # Plot accuracy and loss curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.legend()
    plt.title('Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss')
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/training_curves.png')
    plt.close()

    # Train Autoencoder on Normal images (Class 0)
    print("Training Autoencoder for Anomaly Detection (Hodgkin patterns)...")
    X_train_normal = X_train_model[y_train == 0]
    if len(X_train_normal) > 0:
        autoencoder.fit(X_train_normal, X_train_normal, epochs=10, batch_size=BATCH_SIZE, verbose=1)
    
    # 5. Build Hybrid Ensemble Space
    print("Extracting Deep+Traditional features for ML Ensemble...")
    X_train_ensemble = extract_features_for_dataset(X_train_clahe, feature_extractor, X_train_model)
    X_test_ensemble  = extract_features_for_dataset(X_test_clahe, feature_extractor, X_test_model)
    
    ensemble = HybridEnsembleModel()
    ensemble.train_classifiers(X_train_ensemble, y_train)
    
    # 6. Evaluation
    cnn_test_preds = cnn_model.predict(X_test_model)
    
    print("Evaluating Ensemble Model (Voting = Soft)...")
    final_preds, final_probs = ensemble.predict_ensemble(X_test_ensemble, cnn_test_preds, mode='soft')
    
    acc, prec, rec, f1 = evaluate_classification(y_test, final_preds)
    plot_confusion_matrix(y_test, final_preds, save_path='outputs/conf_matrix.png')
    
    print(f"\nFinal Expected Accuracy Constraint Check: {acc*100:.2f}%")
    print("Verification completed successfully. Check 'outputs/' directory.")
    
    # Save the models
    import joblib
    os.makedirs('saved_models', exist_ok=True)
    cnn_model.save('saved_models/cnn_model.h5')
    autoencoder.save('saved_models/autoencoder.h5')
    joblib.dump(ensemble.rf_classifier, 'saved_models/rf_classifier.pkl')
    joblib.dump(ensemble.svm_classifier, 'saved_models/svm_classifier.pkl')
    print("Models saved.")

if __name__ == "__main__":
    main()
