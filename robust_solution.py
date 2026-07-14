import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def precision_remove_straight_lines(image_path):
    # 1. Load image and convert to binary inverse
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # 2. Extract Raw Horizontal & Vertical Paths
    # We use a narrower, sharp structuring element to find lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    
    raw_horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    raw_vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    
    # ======================================================================
    # PRECISION FILTER: ENFORCE MINIMUM RUN-LENGTH CONNECTIVITY
    # ======================================================================
    # Filter Horizontal Lines: Delete any segments that don't span a wide distance
    num_h, labels_h, stats_h, _ = cv2.connectedComponentsWithStats(raw_horizontal, connectivity=8)
    clean_horizontal = np.zeros_like(raw_horizontal)
    for i in range(1, num_h):
        # CC_STAT_WIDTH checks the exact horizontal length of the straight run
        if stats_h[i, cv2.CC_STAT_WIDTH] >= 80:  # Strict length gate
            clean_horizontal[labels_h == i] = 255
            
    # Filter Vertical Lines: Delete any segments that don't span a tall distance
    num_v, labels_v, stats_v, _ = cv2.connectedComponentsWithStats(raw_vertical, connectivity=8)
    clean_vertical = np.zeros_like(raw_vertical)
    for i in range(1, num_v):
        # CC_STAT_HEIGHT checks the exact vertical length of the straight run
        if stats_v[i, cv2.CC_STAT_HEIGHT] >= 80:  # Strict height gate
            clean_vertical[labels_v == i] = 255
    # ======================================================================
    
    # Combine the long, verified straight line paths into a master deletion mask
    final_straight_grid_mask = cv2.bitwise_or(clean_horizontal, clean_vertical)
    
    # Slightly thicken the deletion mask (1 pixel) to ensure clean subtraction boundaries
    thickening_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    final_straight_grid_mask = cv2.dilate(final_straight_grid_mask, thickening_kernel, iterations=1)
    
    # SUBTRACTION LAYER: Cut the straight lines out of the original artwork
    pure_wave_signals = cv2.subtract(binary, final_straight_grid_mask)
    
    # Bridge any microscopic gaps left behind where lines cut directly through the wave
    bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    pure_wave_signals = cv2.morphologyEx(pure_wave_signals, cv2.MORPH_CLOSE, bridge_kernel)
    
    # --- Visual Telemetry Verification Plots ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 11))
    
    axes[0].imshow(binary, cmap='gray')
    axes[0].set_title("1. Original Raw Input Mask (Contains wave and grid intersections)", fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(final_straight_grid_mask, cmap='gray')
    axes[1].set_title("2. Precision Extraction Mask: Long Straight Lines Only (Runs >= 80px)", fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(pure_wave_signals, cmap='gray')
    axes[2].set_title("3. Output Matrix: Curvy Waves Preserved with Straight Lines Erased", fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return pure_wave_signals

if __name__ == "__main__":
    TARGET_IMAGE = "10140238-0012.png"
    if os.path.exists(TARGET_IMAGE):
        precision_remove_straight_lines(TARGET_IMAGE)