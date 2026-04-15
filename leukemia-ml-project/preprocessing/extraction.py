import cv2
import numpy as np
import scipy.stats as stats
from skimage.feature import graycomatrix, graycoprops

def extract_shape_features(binary_mask):
    """
    Extract Shape Features from segmented binary mask.
    Features: Area, Perimeter, Circularity, Aspect Ratio.
    """
    if binary_mask is None: return [0, 0, 0, 0]
    
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [0, 0, 0, 0]
        
    largest_contour = max(contours, key=cv2.contourArea)
    
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    
    # Circularity = 4*pi*Area / Perimeter^2
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = float(w) / h if h > 0 else 0
    
    return [area, perimeter, circularity, aspect_ratio]

def extract_glcm_texture(image):
    """
    Extract Texture Features using Gray-Level Co-occurrence Matrix (GLCM).
    Features: Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM.
    """
    if len(image.shape) == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = image
        
    # GLCM matrix computation (distances=[1], angles=[0, pi/4, pi/2, 3pi/4])
    glcm = graycomatrix(img_gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)
    
    # Calculate properties across angles and average them
    contrast = np.mean(graycoprops(glcm, 'contrast'))
    dissimilarity = np.mean(graycoprops(glcm, 'dissimilarity'))
    homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
    energy = np.mean(graycoprops(glcm, 'energy'))
    correlation = np.mean(graycoprops(glcm, 'correlation'))
    asm = np.mean(graycoprops(glcm, 'ASM'))
    
    return [contrast, dissimilarity, homogeneity, energy, correlation, asm]

def extract_statistical_features(image):
    """
    Extract Basic Statistical features.
    Features: Mean, Variance, Skewness, Kurtosis.
    """
    if len(image.shape) == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = image
        
    mean = np.mean(img_gray)
    variance = np.var(img_gray)
    skewness = stats.skew(img_gray.flatten())
    kurtosis = stats.kurtosis(img_gray.flatten())
    
    return [mean, variance, skewness, kurtosis]

def extract_all_traditional_features(image, mask):
    """
    Combines Shape, Texture, and Statistical Feature Arrays.
    """
    shape_feats = extract_shape_features(mask)
    glcm_feats = extract_glcm_texture(image)
    stat_feats = extract_statistical_features(image)
    
    # Returns a 1D vector of 14 features
    return np.concatenate([shape_feats, glcm_feats, stat_feats])
