import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_separate_signals(extracted_csv, reference_csv, target_lead="II"):
    # 1. Load both CSV files
    print(f"[Loading] Extracted data from: {extracted_csv}")
    ext_df = pd.read_csv(extracted_csv)
    
    print(f"[Loading] Reference data from: {reference_csv}")
    ref_df = pd.read_csv(reference_csv)
    
    # 2. Extract raw numerical arrays
    ext_time = ext_df["Time_Seconds"].values
    ext_volt = ext_df["Voltage_mV"].values
    ref_volt = ref_df[target_lead].values
    
    # 3. RESAMPLING LAYER: Match the lengths exactly
    # Stretches/interpolates the 2200 extracted points into 5000 points
    x_ext_normalized = np.linspace(0, 1, len(ext_volt))
    x_ref_normalized = np.linspace(0, 1, len(ref_volt))
    resampled_ext_volt = np.interp(x_ref_normalized, x_ext_normalized, ext_volt)
    
    # 4. NORMALIZATION (Z-score standard scaling)
    # Brings both signals to a standard shared vertical height scale
    ref_final = (ref_volt - np.mean(ref_volt)) / np.std(ref_volt)
    ext_final = (resampled_ext_volt - np.mean(resampled_ext_volt)) / np.std(resampled_ext_volt)
    
    # 5. Create a shared 10-second timeline array
    shared_time = np.linspace(0, 10, len(ref_volt))
    
    # 6. Plotting - 2 Separate Rows
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Top Plot: Ground Truth Reference Dataset
    axes[0].plot(shared_time, ref_final, color="black", linewidth=1.5)
    axes[0].set_title(f"Official Dataset Ground Truth Reference Matrix (Lead {target_lead})", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Normalized Amplitude", fontsize=11)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    
    # Bottom Plot: Your Extracted Signal Output
    axes[1].plot(shared_time, ext_final, color="crimson", linewidth=1.2)
    axes[1].set_title(f"Your Digitized Extracted Signal (Resampled from {len(ext_volt)} to {len(ref_volt)} points)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Time Duration (Seconds)", fontsize=11)
    axes[1].set_ylabel("Normalized Amplitude", fontsize=11)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    
    # Set uniform horizontal view
    plt.xlim(0, 10)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Ensure these point to your active workspace files
    MY_OUTPUT = "digitized_outputs/Test_lead_II.csv"
    DATASET_REF = "548033375.csv"
    
    try:
        plot_separate_signals(MY_OUTPUT, DATASET_REF, target_lead="II")
    except Exception as e:
        print(f"\n[Execution Error] Process halted: {e}")