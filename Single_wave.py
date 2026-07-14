import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

# Clinical Constants defined by the PhysioNet 2024 challenge standards
STANDARD_MM_PER_SEC = 25.0      # 25 mm per second horizontal paper speed
STANDARD_MM_PER_MV = 10.0       # 10 mm per millivolt vertical grid sensitivity

def extract_ecg_signal_precise(image_path, crop_top_percent, crop_bottom_percent, output_name="Test_lead_II"):
    # 1. Load Image Matrix
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
    
    original = img.copy()
    height, width, _ = img.shape
    
    # 2. Convert percentages to row coordinates
    start_y = int(height * crop_top_percent)
    end_y = int(height * crop_bottom_percent)
    start_x = 0
    end_x = width
    
    print(f"[Debug Info] Image Height: {height}px, Width: {width}px")
    print(f"[Debug Info] Active Box Boundaries -> Top Row Y: {start_y}, Bottom Row Y: {end_y}")

    # Draw the bounding box for verification tracking
    img_with_rectangle = original.copy()
    cv2.rectangle(img_with_rectangle, (start_x, start_y), (end_x, end_y), (0, 255, 0), 3)
    
    # 3. Crop to the selected rhythm lane
    img_cropped = img[start_y:end_y, start_x:end_x]
    
    # 4. Filter Grid Lines via HSV Color Space
    img_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2HSV)
    
    lower_pink1 = np.array([0, 20, 20])   
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
    
    # 6. Connected Component Cleaning
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    clean_binary = np.zeros_like(binary)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if 20 <= area <= 40000:  
            clean_binary[labels == i] = 255
            
    # 7. Skeletonize Tracking
    bool_binary = clean_binary > 0
    if not np.any(bool_binary):
        bool_binary = binary > 0
        
    skeleton = skeletonize(bool_binary)
    skeleton_img = (skeleton * 255).astype(np.uint8)
    
    # 8. Data Extraction Layer with Fallback Detects
    crop_height, crop_width = skeleton_img.shape
    signal_y = np.full(crop_width, np.nan)
    
    for x in range(crop_width):
        y_indices = np.where(skeleton_img[:, x] == 255)[0]
        if len(y_indices) > 0:
            signal_y[x] = np.mean(y_indices)
            
    fallback_count = 0
    for x in range(crop_width):
        if np.isnan(signal_y[x]):
            y_indices_binary = np.where(binary[:, x] == 255)[0]
            if len(y_indices_binary) > 0:
                signal_y[x] = np.mean(y_indices_binary)
                fallback_count += 1
                
    # Invert Y-axis so deflections match positive electrical directions
    signal_y = crop_height - signal_y
    
    # 9. Array Structural Gap Interpolation
    x_axis = np.arange(crop_width)
    nans = np.isnan(signal_y)
    if np.any(nans) and not np.all(nans):
        signal_y[nans] = np.interp(x_axis[nans], x_axis[~nans], signal_y[~nans])
    elif np.all(nans):
        signal_y = np.zeros(crop_width)

    print(f"[Extraction Info] Signal extracted successfully. Width: {len(signal_y)} points. Fallback patches applied: {fallback_count}")

    # ======================================================================
    # NEW STEP 10: CALIBRATION & PHYSICAL TRANSFORMATION LAYER
    # ======================================================================
    # For standard challenge sheets (approx 300 DPI), 1 mm is ~11.81 pixels.
    pixels_per_mm = 11.81 
    
    # Transform pixels to clinical physical units (Time: seconds, Voltage: mV)
    time_series = np.arange(crop_width) / (pixels_per_mm * STANDARD_MM_PER_SEC)
    voltage_series = signal_y / (pixels_per_mm * STANDARD_MM_PER_MV)
    
    # Center the voltage vector on a 0.0 mV baseline offset marker
    voltage_series = voltage_series - np.mean(voltage_series)
    
    # Create an output directory folder structurally inside your ECG_project space
    output_dir = "digitized_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the structured metrics directly into a CSV matrix format
    csv_path = os.path.join(output_dir, f"{output_name}.csv")
    df = pd.DataFrame({"Time_Seconds": time_series, "Voltage_mV": voltage_series})
    df.to_csv(csv_path, index=False)
    print(f"[Data Export] Clean clinical time-series safely saved to: {csv_path}")

    # --- Plotting Visualization Windows ---
    fig, axes = plt.subplots(4, 1, figsize=(12, 11))
    
    axes[0].imshow(cv2.cvtColor(img_with_rectangle, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"1. Main Layout (Box Window: Rows {start_y} to {end_y})")
    axes[0].axis('off')
    
    axes[1].imshow(binary, cmap='gray')
    axes[1].set_title("2. Cropped Binary Stream")
    axes[1].axis('off')
    
    axes[2].imshow(skeleton_img, cmap='gray')
    axes[2].set_title("3. Skeletonized Isolated Target Track")
    axes[2].axis('off')
    
    # Plotting using the newly calibrated physical time/mV axes
    axes[3].plot(time_series, voltage_series, color='darkred', linewidth=1)
    axes[3].set_title("4. Output Final Digitized Time-Series Array (Calibrated to Seconds & mV)")
    axes[3].set_xlabel("Time (Seconds)")
    axes[3].set_ylabel("Amplitude (mV)")
    axes[3].set_xlim(time_series[0], time_series[-1])
    axes[3].grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    
    # Save a high-DPI confirmation plot image asset to disk
    plot_path = os.path.join(output_dir, f"{output_name}_summary.png")
    plt.savefig(plot_path, dpi=300)
    print(f"[Data Export] Plot diagnostic image saved to: {plot_path}")
    
    plt.show()
    return time_series, voltage_series

if __name__ == "__main__":
    IMAGE_FILE = "Test.png"
    
    # Active bounding box parameters tested on your Inspiron system terminal
    CROP_TOP = 0.82      
    CROP_BOTTOM = 0.92   
    
    try:
        final_time, final_volt = extract_ecg_signal_precise(IMAGE_FILE, CROP_TOP, CROP_BOTTOM)
        print("\n--- Pipeline Fully Complete ---")
    except FileNotFoundError as e:
        print(e)