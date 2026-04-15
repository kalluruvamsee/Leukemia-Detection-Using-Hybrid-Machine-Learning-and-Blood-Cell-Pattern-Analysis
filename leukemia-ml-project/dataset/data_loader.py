import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(dataset_path, img_size=(224, 224), test_size=0.15, val_size=0.15):
    """
    Simulates loading image paths and labels from a structured directory.
    Expects folder structure: dataset_path/leukemia/ and dataset_path/normal/
    Splits into train, validation, and test (70/15/15) as required.
    """
    # In a real scenario, this would load Kaggle/ALL-IDB images
    # Since dataset downloading requires user API keys (like Kaggle.json), we provide the structure
    print("Scanning dataset directory...")
    all_images = []
    labels = []
    
    # We assume 'leukemia' folder corresponds to class 1, 'normal' to class 0
    leukemia_dir = os.path.join(dataset_path, 'leukemia')
    normal_dir = os.path.join(dataset_path, 'normal')
    
    if os.path.exists(leukemia_dir) and os.path.exists(normal_dir):
        for img_path in glob.glob(os.path.join(leukemia_dir, '*.jpg')) + glob.glob(os.path.join(leukemia_dir, '*.png')):
            all_images.append(img_path)
            labels.append(1)
            
        for img_path in glob.glob(os.path.join(normal_dir, '*.jpg')) + glob.glob(os.path.join(normal_dir, '*.png')):
            all_images.append(img_path)
            labels.append(0)
    else:
        print(f"Warning: Dataset folders not found in {dataset_path}. Creating mock paths for structural validation.")
        # If no dataset exists yet, we mock data paths for end-to-end functionality verification
        all_images = [f"mock_img_{i}.jpg" for i in range(100)]
        labels = [1 if i % 2 == 0 else 0 for i in range(100)]
        
    all_images = np.array(all_images)
    labels = np.array(labels)
    
    # Splitting to Train (70%), Val (15%), Test (15%)
    # First, separate 70% Train, 30% Temp (Val + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(all_images, labels, test_size=(test_size + val_size), random_state=42, stratify=labels)
    
    # Split the temporal 30% into equal halves: 15% Val, 15% Test
    ratio = test_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=ratio, random_state=42, stratify=y_temp)
    
    print(f"Dataset Split: Train: {len(X_train)} (70%), Validation: {len(X_val)} (15%), Test: {len(X_test)} (15%)")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

if __name__ == "__main__":
    load_data('./data')
