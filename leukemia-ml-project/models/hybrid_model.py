import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, Input, Flatten
from tensorflow.keras.models import Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def build_cnn_feature_extractor(input_shape=(224, 224, 3)):
    """
    Builds the Deep Learning feature extractor based on ResNet50.
    Transfer Learning setup.
    """
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
        
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # Add Dropout and Batch Normalization as per requirements
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Output layer for direct CNN classification
    predictions = Dense(1, activation='sigmoid', name='cnn_output')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Extractor model without output head (used for SVM / RF features)
    feature_extractor = Model(inputs=base_model.input, outputs=x)
    
    return model, feature_extractor

def build_autoencoder(input_shape=(224, 224, 3)):
    """
    Builds an Autoencoder (AE) to detect anomalies (Hodgkin lymphoma patterns).
    Trained on 'normal' cells. High reconstruction error indicates anomaly.
    """
    input_img = Input(shape=input_shape)
    
    # Encoder
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    
    # Decoder
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    decoded = tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    autoencoder = Model(input_img, decoded)
    return autoencoder

class HybridEnsembleModel:
    """
    Combines the Neural Network, Random Forest and SVM into a hard/soft voting ensemble.
    """
    def __init__(self):
        self.rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.svm_classifier = SVC(probability=True, kernel='rbf', random_state=42)
        
    def train_classifiers(self, deep_features, labels):
        print("Training Random Forest...")
        self.rf_classifier.fit(deep_features, labels)
        
        print("Training SVM...")
        self.svm_classifier.fit(deep_features, labels)
        
    def predict_ensemble(self, deep_features, cnn_predictions, mode='soft'):
        """
        Voting mechanism:
        Combines CNN probability, RF probability, and SVM probability
        mode='soft' averages probabilities
        mode='hard' majority voting
        """
        rf_probs = self.rf_classifier.predict_proba(deep_features)[:, 1]
        svm_probs = self.svm_classifier.predict_proba(deep_features)[:, 1]
        
        cnn_probs = cnn_predictions.flatten()
        
        if mode == 'soft':
            ensemble_probs = (rf_probs + svm_probs + cnn_probs) / 3.0
            preds = (ensemble_probs >= 0.5).astype(int)
            return preds, ensemble_probs
        else: # Hard voting
            rf_preds = (rf_probs > 0.5).astype(int)
            svm_preds = (svm_probs > 0.5).astype(int)
            cnn_preds = (cnn_probs > 0.5).astype(int)
            
            votes = rf_preds + svm_preds + cnn_preds
            preds = (votes >= 2).astype(int)
            ensemble_probs = (rf_probs + svm_probs + cnn_probs) / 3.0
            return preds, ensemble_probs
