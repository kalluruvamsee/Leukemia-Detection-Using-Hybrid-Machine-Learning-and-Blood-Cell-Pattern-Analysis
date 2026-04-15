import cv2
import numpy as np

def apply_efcm(image, n_clusters=3, max_iter=100, m=2.0, epsilon=1e-5):
    """
    Enhanced Fuzzy C-Means (EFCM) for Frequency-based clustering to segment WBC regions.
    Instead of standard FCM, EFCM focuses on histogram/frequency-based optimized FCM for speed and accuracy.
    Here we implement a robust 2D histogram approximation of EFCM.
    
    Args:
        image: Preprocessed RGB image from CLAHE
        n_clusters: number of segments (e.g., Background, RBCs, WBC nucleus)
        max_iter: max iterations for membership optimization
        m: fuzziness factor
    Returns:
        Segmented cluster mask focusing on the nucleus (densest class).
    """
    if image is None: return None
    
    # Using L channel for density/frequency based FCM is most effective for purple WBCs
    if len(image.shape) == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = image
        
    flat_image = img_gray.flatten().astype(np.float32)
    hist, bins = np.histogram(flat_image, bins=256, range=[0, 256])
    
    # Initialize random centroids from histogram peaks (simplified)
    np.random.seed(42)
    centroids = np.random.choice(bins[:-1], n_clusters)
    
    memberships = np.zeros((256, n_clusters))
    # Optimize Cluster Centers
    for iteration in range(max_iter):
        distances = np.abs(bins[:-1][:, np.newaxis] - centroids)
        # Avoid division by zero
        distances = np.fmax(distances, np.finfo(np.float64).eps)
        
        # Membership matrix calculation
        inv_dists = distances ** (-2.0 / (m - 1))
        memberships = inv_dists / inv_dists.sum(axis=1)[:, np.newaxis]
        
        # Update centroids via frequency
        num = np.sum(hist[:, np.newaxis] * (memberships ** m) * bins[:-1][:, np.newaxis], axis=0)
        den = np.sum(hist[:, np.newaxis] * (memberships ** m), axis=0)
        new_centroids = num / den
        
        if np.linalg.norm(new_centroids - centroids) < epsilon:
            break
        centroids = new_centroids

    # Map membership back to original image
    mapped_pixels = np.zeros_like(flat_image)
    for i, p in enumerate(flat_image):
        cluster_idx = np.argmax(memberships[int(p)])
        mapped_pixels[i] = centroids[cluster_idx]
    
    segmented_gray = mapped_pixels.reshape(img_gray.shape).astype(np.uint8)
    
    # Extract the nucleus (usually the lowest intensity component in L*a*b / grayscale inverted)
    # The lowest centroid generally belongs to the dark purple nucleus.
    nucleus_centroid = np.min(centroids)
    
    # Binary threshold for nucleus
    _, binary_mask = cv2.threshold(segmented_gray, nucleus_centroid + 10, 255, cv2.THRESH_BINARY_INV)
    return binary_mask


def apply_imp(binary_mask, iterations=2):
    """
    Iterative Morphological Processing (IMP)
    Applies Erosion, Dilation, Opening & Closing continuously
    to remove noise and refine segmented cells based on shape priors.
    """
    if binary_mask is None: return None
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    refined_mask = binary_mask.copy()
    
    for i in range(iterations):
        # 1. Erosion to remove tiny speckles
        refined_mask = cv2.erode(refined_mask, kernel, iterations=1)
        # 2. Dilation to restore cell boundaries
        refined_mask = cv2.dilate(refined_mask, kernel, iterations=1)
        
        # 3. Opening (Erosion followed by Dilation)
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        # 4. Closing (Dilation followed by Erosion) to fill holes in the nucleus
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
    return refined_mask

def segment_wbc(image):
    """
    Full pipeline wrapper for WBC Segmentation.
    Executes EFCM -> IMP sequentially.
    """
    # 1. Standard EFCM for raw mathematical morphology mapping
    efcm_mask = apply_efcm(image)
    
    # 2. Iterative Morphological Refinement
    final_mask = apply_imp(efcm_mask)
    
    # 3. Apply mask to original
    if len(image.shape) == 3:
        result_img = cv2.bitwise_and(image, image, mask=final_mask)
    else:
        result_img = cv2.bitwise_and(image, final_mask)
        
    return final_mask, result_img
