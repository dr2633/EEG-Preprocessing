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

# Identify channels that correlate with E127 and E128
blink_channels = ['E127', 'E128']
correlated_channels = []
for ch_name in blink_channels:
    ch_idx = raw.ch_names.index(ch_name)
    try:
        corr_matrix = np.corrcoef(raw._data)
    except Exception as e:
        print(f"Error computing correlation matrix for channel {ch_name}: {e}")
        continue
    correlated_channels.extend(np.where(np.abs(corr_matrix[ch_idx, :]) > 0.7)[0])

# Identify the timepoints where blink artifacts occur
blink_artifact_indices = []
for ch_name in blink_channels:
    ch_idx = raw.ch_names.index(ch_name)
    try:
        channel_data = raw._data[ch_idx]
        blink_threshold = 1  # Adjust as needed
        blink_artifact_indices.extend(np.where(channel_data > blink_threshold)[0])
    except Exception as e:
        print(f"Error processing data for channel {ch_name}: {e}")

# Print the timepoints where blink artifacts occur
blink_times = raw.times[blink_artifact_indices]
print("Timepoints where blink artifacts occur:")
for time in blink_times:
    print(f"{time:.3f} seconds")
