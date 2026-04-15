import cv2
import numpy as np
import os

def create_mock_blood_smear(filename, is_leukemia):
    # Base canvas (pinkish/white typical of blood smears)
    img = np.ones((400, 400, 3), dtype=np.uint8) * 240
    img[:, :, 0] = 230  # slightly blueish/pink
    img[:, :, 1] = 210
    img[:, :, 2] = 230

    # Draw RBCs (small red/pink circles)
    np.random.seed(42 if is_leukemia else 24)
    for _ in range(30):
        x, y = np.random.randint(20, 380, 2)
        r = np.random.randint(15, 25)
        cv2.circle(img, (x, y), r, (180, 150, 220), -1)
        cv2.circle(img, (x, y), r-3, (210, 190, 240), -1) # donut hole

    # Draw WBC (Purple)
    if is_leukemia:
        # Leukemia: Enlarged, highly irregular, multiple prominent nuclei, densely packed chromatin
        x, y = 200, 200
        # Cytoplasm
        cv2.circle(img, (x, y), 80, (200, 180, 220), -1)
        # Large irregular Nucleus
        cv2.ellipse(img, (x, y), (65, 50), 30, 0, 360, (120, 50, 140), -1)
        cv2.ellipse(img, (x-15, y+20), (40, 45), 60, 0, 360, (100, 30, 120), -1)
        cv2.ellipse(img, (x+25, y-10), (50, 55), 120, 0, 360, (110, 40, 130), -1)
        
        # Add texture/noise
        noise = np.random.normal(0, 15, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
    else:
        # Normal: Regular spherical WBC, distinct segmented nucleus
        x, y = 200, 200
        # Cytoplasm
        cv2.circle(img, (x, y), 50, (210, 190, 230), -1)
        # Segmented Nucleus (Neutrophil)
        cv2.circle(img, (x-15, y-10), 18, (130, 80, 150), -1)
        cv2.circle(img, (x+15, y+10), 18, (130, 80, 150), -1)
        cv2.circle(img, (x+10, y-15), 16, (130, 80, 150), -1)
        # Connecting threads
        cv2.line(img, (x-15, y-10), (x+10, y-15), (130, 80, 150), 4)
        cv2.line(img, (x+10, y-15), (x+15, y+10), (130, 80, 150), 4)

        # Add slight texture
        noise = np.random.normal(0, 5, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Save
    cv2.imwrite(filename, img)
    print(f"Created {filename}")

if __name__ == "__main__":
    create_mock_blood_smear("sample_leukemia_positive.jpg", is_leukemia=True)
    create_mock_blood_smear("sample_normal_negative.jpg", is_leukemia=False)
