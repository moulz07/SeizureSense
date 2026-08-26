import os
import numpy as np
import mne
from scipy.signal import stft

DATA_PATH = "data"

# -----------------------------
# GET FILES
# -----------------------------
def get_seizure_files():
    file_path = os.path.join(DATA_PATH, "RECORDS-WITH-SEIZURES")

    files = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.endswith(".edf"):
                files.append(os.path.join(DATA_PATH, line))

    return files


def get_patient_files(patient_id):
    return [f for f in get_seizure_files() if patient_id in f]


# -----------------------------
# LOAD EEG
# -----------------------------
def load_eeg(file_path):
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    raw.filter(0.5, 40)

    data = raw.get_data()

    # normalize per channel
    data = (data - np.mean(data, axis=1, keepdims=True)) / \
           (np.std(data, axis=1, keepdims=True) + 1e-6)

    return data


# -----------------------------
# SEGMENT
# -----------------------------
def segment_signal(data, fs=256, window_sec=5):
    window = fs * window_sec
    step = window // 2

    segments = []
    for i in range(0, data.shape[1] - window, step):
        segments.append((data[:, i:i+window], i/fs, (i+window)/fs))

    return segments


# -----------------------------
# SPECTROGRAM
# -----------------------------
def to_spectrogram(segment):
    specs = []

    for ch in segment:
        f, t, Zxx = stft(ch, fs=256, nperseg=128, noverlap=64)

        # keep EEG band
        mask = (f >= 0.5) & (f <= 30)
        Zxx = np.abs(Zxx[mask])

        specs.append(Zxx)

    return np.array(specs)


# -----------------------------
# SEIZURE TIMES
# -----------------------------
def get_seizure_times(file_path):
    import re

    folder = os.path.dirname(file_path)

    summary = None
    for f in os.listdir(folder):
        if "summary" in f.lower():
            summary = os.path.join(folder, f)
            break

    if summary is None:
        return []

    seizure_times = []
    fname = os.path.basename(file_path)

    with open(summary, "r") as f:
        lines = f.readlines()

    inside = False
    for i in range(len(lines)):
        line = lines[i]

        if fname in line:
            inside = True

        if inside:
            if "Seizure Start Time" in line:
                start = float(re.findall(r'\d+', line)[0])
                end = float(re.findall(r'\d+', lines[i+1])[0])
                seizure_times.append((start, end))

            if "File Name" in line and fname not in line:
                break

    return seizure_times


# -----------------------------
# LABEL
# -----------------------------
def label_segment(start, end, seizure_times):
    for s, e in seizure_times:
        if (s - 300) <= start <= e:
            return 1
    return 0


# -----------------------------
# BUILD DATASET (STABLE)
# -----------------------------
def build_dataset(files):
    np.random.seed(42)

    X, y = [], []

    for file in files:
        print("Processing:", file)

        data = load_eeg(file)
        seizure_times = get_seizure_times(file)

        if len(seizure_times) == 0:
            continue

        segments = segment_signal(data)

        for seg, start, end in segments:
            spec = to_spectrogram(seg)
            label = label_segment(start, end, seizure_times)

            X.append(spec)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    # balance dataset (deterministic)
    c0 = np.where(y == 0)[0]
    c1 = np.where(y == 1)[0]

    n = min(len(c0), len(c1))

    idx = np.concatenate([c0[:n], c1[:n]])
    np.random.shuffle(idx)

    X = X[idx]
    y = y[idx]

    print("\nDataset:", X.shape)
    print("Classes:", np.bincount(y))

    return X, y