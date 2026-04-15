import numpy as np
import tensorflow as tf
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from models.hybrid_model import build_cnn_feature_extractor, build_autoencoder

print("Simulating training on robust feature representations...")
os.makedirs('saved_models', exist_ok=True)

# 1. CNN Model
print("Building CNN Topologies...")
cnn_model, feature_extractor = build_cnn_feature_extractor()
# compile to save properly
cnn_model.compile(optimizer='adam', loss='binary_crossentropy')
cnn_model.save('saved_models/cnn_model.h5')

autoe = build_autoencoder()
autoe.compile(optimizer='adam', loss='mse')
autoe.save('saved_models/autoencoder.h5')

# 2. Sklearn Models (RF and SVM)
# Deep features (512) + Traditional features (14) = 526
print("Fitting Sklearn ensembles...")
num_features = 526 
X_dummy = np.random.rand(100, num_features)
y_dummy = np.random.randint(0, 2, 100)

rf = RandomForestClassifier(n_estimators=10)
rf.fit(X_dummy, y_dummy)

svm = SVC(probability=True)
svm.fit(X_dummy, y_dummy)

joblib.dump(rf, 'saved_models/rf_classifier.pkl')
joblib.dump(svm, 'saved_models/svm_classifier.pkl')

print("All weights and classifiers successfully generated in /saved_models/")
