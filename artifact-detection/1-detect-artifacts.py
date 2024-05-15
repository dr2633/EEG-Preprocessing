import mne
import numpy as np
import os

# Set parameters for fif path
sub = 'pilot-3'
stim = 'Jobs2'
seg = 'segment_2'

base_path = '/Users/derekrosenzweig/Documents/GitHub/EEG-Preprocessing'
fif_path = f'{base_path}/segmented_data/{sub}/{sub}_{seg}_{stim}_eeg.fif'

# Load the data in MNE
raw = mne.io.read_raw_fif(fif_path, preload=True)

# Get channel names starting with 'E'
eeg_channels = [ch for ch in raw.ch_names if ch.startswith('E')]

# Calculate mean and standard deviation across EEG channels
data = raw.get_data(picks=eeg_channels)
mean_data = np.mean(data, axis=1)
std_data = np.std(data, axis=1)

# Define threshold for abnormal activity
threshold = 3 * std_data

# Identify abnormal segments
abnormal_segments = []
for ch_idx, channel in enumerate(data):
    avg_channel = np.mean(channel)  # Average channel activity across time
    abnormal_indices = np.where(np.abs(mean_data - avg_channel) > threshold[ch_idx])[0]
    if len(abnormal_indices) > 0:
        start_idx = abnormal_indices[0]
        end_idx = abnormal_indices[-1]
        start_time = start_idx * 1000 / raw.info['sfreq']  # Convert start index to milliseconds
        end_time = end_idx * 1000 / raw.info['sfreq']  # Convert end index to milliseconds
        abnormal_segments.append((start_time, end_time))

# Print start and stop time of abnormal segments
for start_time, end_time in abnormal_segments:
    print(f"Abnormal segment start: {start_time:.2f} ms, stop: {end_time:.2f} ms")
