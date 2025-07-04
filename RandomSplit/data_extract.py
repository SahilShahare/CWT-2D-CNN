import wfdb
import numpy as np
import csv
import os
import pywt
import numpy as np
import pandas as pd
import cv2

import json

# Load paths from JSON file
with open("../paths.json", "r") as f:
    paths = json.load(f)

data_path = paths["afdb"]
os.makedirs('./csv', exist_ok=True)

beats_0 = 0
beats_1 = 0
for record_name in os.listdir(data_path):
    
    if record_name.endswith(".atr"):
        record_name = record_name.replace(".atr", "")
        if(record_name == '04936' or record_name =='05091' or record_name == '00735' or record_name =='03665'):
            continue
        print(record_name)
    else:
        continue
    
    try:
        
        record = wfdb.rdrecord(f"{data_path}/{record_name}")
        sig_len = len(record.p_signal[:,0])
        annotation = wfdb.rdann(f"{data_path}/{record_name}", "atr")
        
        file = open(f"./csv/{record_name}.csv", mode="w")
        writer = csv.writer(file)

        writer.writerow(["start", "end", "label"])
        
        annotation = wfdb.rdann(f"{data_path}/{record_name}", 'atr')
        ann = wfdb.rdann(f"{data_path}/{record_name}", 'qrs')
        
        r_peaks = np.array(ann.sample)
        
        rhythm_change_indices = annotation.sample
        rhythm_annotations = annotation.aux_note
        current_rhythm = 0

        for i in range(1, len(r_peaks)):
            rr_prev = r_peaks[i] - r_peaks[i - 1]
            start = int(r_peaks[i] - rr_prev / 3)
            end   = start+300

            if(end>sig_len):
                break
        
            if((current_rhythm < len(rhythm_change_indices)-1) and  r_peaks[i]>rhythm_change_indices[current_rhythm+1]):
                current_rhythm+=1
        
            if(rhythm_annotations[current_rhythm] == '(AFIB'):
                label=1
            elif(rhythm_annotations[current_rhythm] == '(N'):
                label=0
            else:
                label=2
            writer.writerow([start, end, label])
            if(label==0):
                beats_0+=1
            elif(label==1):
                beats_1+=1
            
        file.close()
    except Exception as e:
        print(f"Unable to read file: {record_name} {e}")
print(f"Total beats {beats_0} {beats_1}")

#############################################################################################################

def denoise_ecg(signal: np.ndarray) -> np.ndarray:
    signal = signal.astype(np.float16)
    # Decompose into 7 scales
    coeffs = pywt.wavedec(signal, 'db4', level=7)
    
    # Zero out first three detail coefficients (high-frequency noise)
    coeffs[1] = np.zeros_like(coeffs[1])  # Scale 1
    coeffs[2] = np.zeros_like(coeffs[2])  # Scale 2
    coeffs[3] = np.zeros_like(coeffs[3])  # Scale 3
    
    # Reconstruct denoised signal
    return pywt.waverec(coeffs, 'db4')


def apply_dwt(segment, wavelet='db5', level=5, output_size=(128, 128)):
    coeffs = pywt.wavedec(segment, wavelet, level=level)
    
    coeff_arr = np.concatenate([np.abs(c) for c in coeffs], axis=0)
    
    coeff_arr -= coeff_arr.min()
    if coeff_arr.max() > 0:
        coeff_arr = coeff_arr / coeff_arr.max()
    img = (coeff_arr * 255).astype(np.uint8)
    img_resized = cv2.resize(img.reshape(-1, 1), output_size, interpolation=cv2.INTER_AREA)
    return img_resized


def process_record(record_path, csv_path):
    # Load ECG signal using wfdb
    record = wfdb.rdrecord(record_path)
    signal = record.p_signal[:, 0]

    signal = denoise_ecg(signal)

    # Read corresponding CSV file for fragmentation and labeling
    annotations = pd.read_csv(csv_path)
    
    X_fragments = []
    y_labels = []

    start = annotations['start'].values
    end = annotations['end'].values
    label = annotations['label'].values
    i=0
    while(i<len(start)):
        af = 0
        j=0
        segment = []
        while(i<len(start) and j<5):
            if(label[i]==2):
                i+=1
                j=0
                af=0
                segment=[]
                break
            elif(label[i]==1):
                af+=1
            beat = signal[start[i]:end[i]].astype(np.float16)
            segment.append(apply_dwt(beat))
            j+=1
            i+=1
        if(j==5 and (af==0 or af==5)):
            X_fragments.append(np.array(segment))
            if(af==5):
                y_labels.append(1)
            else:
                y_labels.append(0)
    return np.array(X_fragments), np.array(y_labels)
    

# Process all records in the folder
def process_all_records(data_folder='afdb', csv_folder='csv'):
    """
    Process all records in the afdb folder based on CSV files in the csv folder.
    Args:
        data_folder: Path to folder containing ECG records (.dat files).
        csv_folder: Path to folder containing CSV files.
    Returns:
        X (input fragments) and y (labels) for all records.
    """
    X_all = []
    y_all = []

    for csv_file in os.listdir(csv_folder):
        if csv_file.endswith('.csv'):
            record_name = os.path.splitext(csv_file)[0]  # Extract record name from CSV filename
            record_path = os.path.join(data_folder, record_name)
            csv_path = os.path.join(csv_folder, csv_file)
            print(f"Reading {record_name}")
            # Process this record and append results to overall dataset
            X_fragments, y_labels = process_record(record_path, csv_path)
            X_all.extend(X_fragments)
            y_all.extend(y_labels)

    return np.array(X_all), np.array(y_all)

# Run processing function for all records
X, y = process_all_records(data_folder=paths["afdb"], csv_folder='./csv')

#################################################################################################################################
def balanced_data(X, y, num_samples_0, num_samples_1):
    # Find indices of each class
    idx_class_0 = np.where(y == 0)[0]
    idx_class_1 = np.where(y == 1)[0]

    np.random.shuffle(idx_class_0)
    np.random.shuffle(idx_class_1)

    idx_class_0 = idx_class_0[:num_samples_0]
    idx_class_1 = idx_class_1[:num_samples_1]
    
    # Combine balanced indices
    balanced_indices = np.concatenate([idx_class_0, idx_class_1])
    np.random.shuffle(balanced_indices)
    # Extract balanced data
    X_balanced = X[balanced_indices]
    y_balanced = y[balanced_indices]
    return X_balanced, y_balanced

X, y = balanced_data(X,y,50000,50000)

np.save('X.npy', X)
np.save('y.npy', y)

y_0 = np.where(y == 0)[0]
y_1 = np.where(y == 1)[0]

print("Processed input shape:", X.shape)
print("Processed labels shape:", y.shape)
print(f"Normal: {len(y_0)} AF: {len(y_1)}")

