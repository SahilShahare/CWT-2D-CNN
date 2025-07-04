import wfdb
import numpy as np
import csv
import os
import pywt
import numpy as np
import pandas as pd
import cv2
import glob
from scipy.signal import resample
from sklearn.model_selection import train_test_split
import json

# Load paths from JSON file
with open("../paths.json", "r") as f:
    paths = json.load(f)

        
        
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
def process_all_records(data_folder, csv_folder):
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


##############################################################################################################################

data_path = paths["afdb"]
os.makedirs('./csv', exist_ok=True)
os.makedirs('./csv/afdb', exist_ok=True)

files = glob.glob(os.path.join('./csv/afdb', "*"))

for f in files:
    if os.path.isfile(f):
        os.remove(f)
        
beats_0 = 0
beats_1 = 0
records_done = 0
for record_name in os.listdir(data_path):
    
    if record_name.endswith(".atr"):
        if(records_done == 10):
            break
        record_name = record_name.replace(".atr", "")
        if(record_name == '04936' or record_name =='05091' or record_name == '00735' or record_name =='03665'):
            continue
        records_done+=1
        print(record_name)
    else:
        continue
    
    try:
        
        record = wfdb.rdrecord(f"{data_path}/{record_name}")
        sig_len = len(record.p_signal[:,0])
        annotation = wfdb.rdann(f"{data_path}/{record_name}", "atr")
        
        file = open(f"./csv/afdb/{record_name}.csv", mode="w")
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
print(f"Total beats AFDB: {beats_0} {beats_1}")

# Run processing function for all records
X_afdb, y_afdb = process_all_records(data_folder=paths["afdb"], csv_folder='./csv/afdb')



X_afdb, y_afdb = balanced_data(X_afdb,y_afdb,20000,20000)

np.save('X_afdb.npy', X_afdb)
np.save('y_afdb.npy', y_afdb)

y_0 = np.where(y_afdb == 0)[0]
y_1 = np.where(y_afdb == 1)[0]

print("For AFDB:")
print("Processed input shape:", X_afdb.shape)
print("Processed labels shape:", y_afdb.shape)
print(f"Normal: {len(y_0)} AF: {len(y_1)}")

##################################################################################################################################################

data_path = paths["ltafdb"]
os.makedirs('./csv', exist_ok=True)
os.makedirs('./csv/ltafdb', exist_ok=True)

files = glob.glob(os.path.join('./csv/ltafdb', "*"))

for f in files:
    if os.path.isfile(f):
        os.remove(f)
        
beats_0 = 0
beats_1 = 0
records_done = 0
for record_name in os.listdir(data_path):
    
    if record_name.endswith(".atr"):
        if(records_done == 10):
            break
        record_name = record_name.replace(".atr", "")
        if(record_name == '04936' or record_name =='05091' or record_name == '00735' or record_name =='03665'):
            continue
        records_done+=1
        print(record_name)
    else:
        continue
    
    try:
        
        record = wfdb.rdrecord(f"{data_path}/{record_name}")
        sig_len = len(record.p_signal[:,0])
        annotation = wfdb.rdann(f"{data_path}/{record_name}", "atr")
        
        file = open(f"./csv/ltafdb/{record_name}.csv", mode="w")
        writer = csv.writer(file)

        writer.writerow(["start", "end", "label"])
        
        annotation = wfdb.rdann(f"{data_path}/{record_name}", 'atr')
        ann = wfdb.rdann(f"{data_path}/{record_name}", "qrs")
        
        r_peaks = np.array(ann.sample[np.array(ann.symbol) == 'N'])
        
        rhythm_change_indices = []
        rhythm_annotations =[]

        for i in range(len(annotation.symbol)):
            if(annotation.symbol[i]=='+'):
                rhythm_change_indices.append(annotation.sample[i])
                rhythm_annotations.append(annotation.aux_note[i])
                
        current_rhythm = 0

        for i in range(1, len(r_peaks)):
            rr_prev = r_peaks[i] - r_peaks[i - 1]
            start = int(r_peaks[i] - rr_prev / 3)
            end   = start+154

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
print(f"Total beats LTAFDB: {beats_0} {beats_1}")

def resample_signal(signal, original_rate= 128 , target_rate=250):
    num_samples = int(len(signal) * target_rate / original_rate)
    return resample(signal, num_samples)

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
            beat = resample_signal(beat)
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
def process_all_records(data_folder, csv_folder):
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
X_ltafdb, y_ltafdb = process_all_records(data_folder=paths["ltafdb"], csv_folder='./csv/ltafdb')



X_ltafdb, y_ltafdb = balanced_data(X_ltafdb,y_ltafdb,20000,20000)

np.save('X_ltafdb.npy', X_ltafdb)
np.save('y_ltafdb.npy', y_ltafdb)

y_0 = np.where(y_ltafdb == 0)[0]
y_1 = np.where(y_ltafdb == 1)[0]

print("For LTAFDB:")
print("Processed input shape:", X_ltafdb.shape)
print("Processed labels shape:", y_ltafdb.shape)
print(f"Normal: {len(y_0)} AF: {len(y_1)}")



#############################################################################################################################################



data_path = paths["shdb"]
os.makedirs('./csv', exist_ok=True)
os.makedirs('./csv/shdb', exist_ok=True)

files = glob.glob(os.path.join('./csv/shdb', "*"))

for f in files:
    if os.path.isfile(f):
        os.remove(f)
        
beats_0 = 0
beats_1 = 0
records_done = 0
for record_name in os.listdir(data_path):
    
    if record_name.endswith(".atr"):
        if(records_done == 10):
            break
        record_name = record_name.replace(".atr", "")
        if(record_name == '04936' or record_name =='05091' or record_name == '00735' or record_name =='03665'):
            continue
        records_done+=1
        print(record_name)
    else:
        continue
    
    try:
        
        record = wfdb.rdrecord(f"{data_path}/{record_name}")
        sig_len = len(record.p_signal[:,0])
        annotation = wfdb.rdann(f"{data_path}/{record_name}", "atr")
        
        file = open(f"./csv/shdb/{record_name}.csv", mode="w")
        writer = csv.writer(file)

        writer.writerow(["start", "end", "label"])
        
        annotation = wfdb.rdann(f"{data_path}/{record_name}", 'atr')
        ann = wfdb.rdann(f"{data_path}/{record_name}", 'qrs')
        
        r_peaks = np.array(ann.sample)
        
        rhythm_annotations = [label for label in annotation.aux_note if label != '']
        rhythm_change_indices = [val for label, val in zip(annotation.aux_note, annotation.sample) if label != '']
        current_rhythm = 0

        for i in range(1, len(r_peaks)):
            rr_prev = r_peaks[i] - r_peaks[i - 1]
            start = int(r_peaks[i] - rr_prev / 3)
            end   = start+240

            if(end>sig_len):
                break
        
            if((current_rhythm < len(rhythm_change_indices)-1) and  r_peaks[i]>=rhythm_change_indices[current_rhythm+1]):
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
print(f"Total beats SHDB: {beats_0} {beats_1}")

def resample_signal(signal, original_rate= 200 , target_rate=250):
    num_samples = int(len(signal) * target_rate / original_rate)
    return resample(signal, num_samples)

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
            beat = resample_signal(beat)
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
def process_all_records(data_folder, csv_folder):
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
X_shdb, y_shdb = process_all_records(data_folder=paths["shdb"], csv_folder='./csv/shdb')



X_shdb, y_shdb = balanced_data(X_shdb,y_shdb,20000,20000)

np.save('X_shdb.npy', X_shdb)
np.save('y_shdb.npy', y_shdb)

y_0 = np.where(y_shdb == 0)[0]
y_1 = np.where(y_shdb == 1)[0]

print("For SHDB:")
print("Processed input shape:", X_shdb.shape)
print("Processed labels shape:", y_shdb.shape)
print(f"Normal: {len(y_0)} AF: {len(y_1)}")