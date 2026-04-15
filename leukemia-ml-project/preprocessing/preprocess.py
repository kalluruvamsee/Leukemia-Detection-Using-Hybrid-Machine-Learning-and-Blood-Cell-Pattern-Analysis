import cv2
import numpy as np
import matplotlib.pyplot as plt

def resize_image(image, target_size=(224, 224)):
    """Resize input image to target dimensions."""
    if image is None: return None
    return cv2.resize(image, target_size)

def remove_noise(image, ksize=(5, 5)):
    """Apply Gaussian Blur for noise removal."""
    if image is None: return None
    return cv2.GaussianBlur(image, ksize, 0)

def enhance_contrast(image):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast."""
    if image is None: return None
    
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge and convert back to BGR
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

def normalize_image(image):
    """Normalize image pixel values to [0, 1]."""
    if image is None: return None
    return image.astype(np.float32) / 255.0

def preprocess_pipeline(image, target_size=(224, 224)):
    """
    End-to-end preprocessing pipeline combining:
    1. Resizing to 224x224
    2. Noise Removal via Gaussian Blur
    3. Contrast Enhancement via CLAHE
    4. Normalization to [0, 1] range
    """
    img_resized = resize_image(image, target_size)
    img_denoised = remove_noise(img_resized)
    img_clahe = enhance_contrast(img_denoised)
    # Return both unnomalized (for visualization) and normalized (for model)
    img_normalized = normalize_image(img_clahe)
    
    return {
        'original_resized': img_resized,
        'clahe_enhanced': img_clahe,
        'model_ready': img_normalized
    }

def show_before_after(original, processed):
    """Utility function to display before/after images side by side."""
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    if original.shape[-1] == 3:
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(original, cmap='gray')
    plt.title('Before Preprocessing')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    if processed.shape[-1] == 3:
        # Check if normalized [0, 1] or [0, 255]
        if processed.dtype == np.float32 or processed.dtype == np.float64:
            plt.imshow(processed)
        else:
            plt.imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(processed, cmap='gray')
    plt.title('After Preprocessing (CLAHE + Noise Removal)')
    plt.axis('off')
    
    plt.show()
