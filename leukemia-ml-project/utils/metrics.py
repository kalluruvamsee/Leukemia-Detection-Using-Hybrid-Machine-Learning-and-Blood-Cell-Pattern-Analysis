import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

def calculate_mse(imageA, imageB):
    """
    Calculate Mean Squared Error between two images.
    Used to compare original vs preprocessed.
    """
    if imageA.shape != imageB.shape:
        imageB = cv2.resize(imageB, (imageA.shape[1], imageA.shape[0]))
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def calculate_psnr(imageA, imageB):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).
    """
    mse = calculate_mse(imageA, imageB)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def plot_confusion_matrix(y_true, y_pred, classes=['Normal', 'Leukemia'], save_path=None):
    """
    Plots and saves Confusion Matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    if save_path:
        plt.savefig(save_path)
    plt.close()

def evaluate_classification(y_true, y_pred):
    """
    Computes Accuracy, Precision, Recall, F1
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print("\n--- Evaluation Metrics ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    return acc, prec, rec, f1

def calculate_autoencoder_reconstruction_error(y_true_imgs, y_pred_imgs):
    """
    Calculate MSE for Autoencoder recon.
    """
    errors = []
    for i in range(len(y_true_imgs)):
        err = calculate_mse(y_true_imgs[i], y_pred_imgs[i])
        errors.append(err)
    return np.array(errors)
