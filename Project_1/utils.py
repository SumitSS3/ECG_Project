import cv2
import numpy as np
from scipy.signal import savgol_filter, find_peaks

def extract_signal(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    h, w = thresh.shape
    signal = []

    for col in range(w):
        rows = np.where(thresh[:, col] > 0)[0]

        if len(rows) > 0:
            signal.append(np.min(rows))  # IMPORTANT FIX
        else:
            signal.append(signal[-1] if len(signal) > 0 else 0)

    signal = np.array(signal)

    # Normalize safely
    denom = np.max(signal) - np.min(signal)
    if denom == 0:
        denom = 1

    signal = (signal - np.min(signal)) / denom
    signal = 1 - signal  # invert ECG

    # Sampling rate assumption
    fs = 250
    time = np.arange(len(signal)) / fs

    return time, signal



def process_signal(signal):

    if len(signal) < 31:
        return signal, []

    smooth = savgol_filter(signal, 31, 3)

    norm = (smooth - np.mean(smooth)) / (np.std(smooth) + 1e-6)

    fs = 250

    peaks, _ = find_peaks(
        norm,
        distance=int(fs * 0.6),  # minimum 0.6 sec gap
        prominence=0.8
    )

    return smooth, peaks



def calculate_heart_rate(peaks, time):

    if len(peaks) < 2:
        return 0, 0

    rr_intervals = np.diff(time[peaks])
    rr_avg = np.mean(rr_intervals)

    if rr_avg == 0:
        return 0, 0

    heart_rate = 60 / rr_avg

    return round(heart_rate, 2), round(rr_avg, 2)

def get_stage(hr):

    if hr == 0:
        return "No Signal"
    elif hr < 40:
        return "Severe Bradycardia"
    elif hr < 50:
        return "Moderate Bradycardia"
    elif hr < 60:
        return "Mild Bradycardia"
    elif hr <= 100:
        return "Normal"
    elif hr <= 120:
        return "Mild Tachycardia"
    elif hr <= 150:
        return "Moderate Tachycardia"
    else:
        return "Severe Tachycardia"
