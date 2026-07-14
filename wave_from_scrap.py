import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

def extract_ecg_signal_precise(image_path, crop_top_percent, crop_bottom_percent):
    # 1. Load Image Matrix
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
    
    original = img.copy()
    height, width, _ = img.shape
    
    # 2. CONVERT PERCENTAGES TO ABSOLUTE ROW COORD INDICES
    start_y = int(height * crop_top_percent)
    end_y = int(height * crop_bottom_percent)
    start_x = 0
    end_x = width
    
    print(f"[Debug Info] Image Height: {height}px, Width: {width}px")
    print(f"[Debug Info] Active Box Boundaries -> Top Row Y: {start_y}, Bottom Row Y: {end_y}")

    # Draw the explicit boundary block on original for tracking
    img_with_rectangle = original.copy()
    cv2.rectangle(img_with_rectangle, (start_x, start_y), (end_x, end_y), (0, 255, 0), 3)
    
    # 3. Crop Down to Selected Track Lane
    img_cropped = img[start_y:end_y, start_x:end_x]
    
    # 4. Filter Grid Lines
    img_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2HSV)
    
    lower_pink1 = np.array([0, 20, 20])   # Widened tolerance
    upper_pink1 = np.array([20, 255, 255])
    lower_pink2 = np.array([140, 20, 20])
    upper_pink2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_pink1, upper_pink1)
    mask2 = cv2.inRange(hsv, lower_pink2, upper_pink2)
    grid_mask = cv2.bitwise_or(mask1, mask2)
    
    grid_removed = img_gray.copy()
    grid_removed[grid_mask > 0] = 255 
    
    # 5. Local Matrix Binarization
    binary = cv2.adaptiveThreshold(
        grid_removed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10
    )
    
    # 6. Connected Component Path Analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    clean_binary = np.zeros_like(binary)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if 20 <= area <= 40000:  # Loosened restrictions to preserve signal elements
            clean_binary[labels == i] = 255
            
    # 7. Skeletonize Tracking
    bool_binary = clean_binary > 0
    if not np.any(bool_binary):
        bool_binary = binary > 0
        
    skeleton = skeletonize(bool_binary)
    skeleton_img = (skeleton * 255).astype(np.uint8)
    
    # 8. DATA EXTRACTION LAYER WITH FALLBACK DETECTS
    crop_height, crop_width = skeleton_img.shape
    signal_y = np.full(crop_width, np.nan)
    
    # Scan via skeleton first
    for x in range(crop_width):
        y_indices = np.where(skeleton_img[:, x] == 255)[0]
        if len(y_indices) > 0:
            signal_y[x] = np.mean(y_indices)
            
    # CRITICAL FALLBACK: If skeleton is missing fragments, pull directly from binary map
    fallback_count = 0
    for x in range(crop_width):
        if np.isnan(signal_y[x]):
            y_indices_binary = np.where(binary[:, x] == 255)[0]
            if len(y_indices_binary) > 0:
                signal_y[x] = np.mean(y_indices_binary)
                fallback_count += 1
                
    # Invert Y-axis so upward deflections are positive voltage peaks
    signal_y = crop_height - signal_y
    
    # 9. Array Structural Gap Interpolation
    x_axis = np.arange(crop_width)
    nans = np.isnan(signal_y)
    if np.any(nans) and not np.all(nans):
        signal_y[nans] = np.interp(x_axis[nans], x_axis[~nans], signal_y[~nans])
    elif np.all(nans):
        signal_y = np.zeros(crop_width)

    print(f"[Extraction Info] Signal extracted successfully. Width: {len(signal_y)} points. Fallback patches applied: {fallback_count}")

    # --- Plotting Visualization Windows ---
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    
    axes[0].imshow(cv2.cvtColor(img_with_rectangle, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"1. Main Layout (Box Window: Rows {start_y} to {end_y})")
    axes[0].axis('off')
    
    axes[1].imshow(binary, cmap='gray')
    axes[1].set_title("2. Cropped Binary Stream")
    axes[1].axis('off')
    
    axes[2].imshow(skeleton_img, cmap='gray')
    axes[2].set_title("3. Skeletonized Isolated Target Track")
    axes[2].axis('off')
    
    axes[3].plot(signal_y, color='blue', linewidth=1)
    axes[3].set_title("4. Output Digital 1D Numerical Waveform")
    axes[3].set_xlim(0, crop_width)
    
    plt.tight_layout()
    plt.show()
    
    return signal_y

if __name__ == "__main__":
    IMAGE_FILE = "Test.png"
    
    # ======================================================================
    # TUNE THESE TWO VALUES RIGHT HERE TO SHIFT OR SHRINK THE GREEN BOX!
    # ======================================================================
    CROP_TOP = 0.82      
    CROP_BOTTOM = 0.92   
    # ======================================================================
    
    try:
        extracted_vector = extract_ecg_signal_precise(IMAGE_FILE, crop_top_percent=CROP_TOP, crop_bottom_percent=CROP_BOTTOM)
    except FileNotFoundError as e:
        print(e)
    