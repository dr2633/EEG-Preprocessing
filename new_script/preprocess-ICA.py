import mne
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# Set parameters for fif path
sub = 'pilot-3'
stim = 'AttFast'
seg = 'segment_8'
comp = 'with-ica1'  # two options: 'no-ica' or 'with-ica'

base_path = '/Users/derekrosenzweig/Documents/GitHub/EEG-Preprocessing'
word_path = f'{base_path}/annotations/words/tsv/{stim}-words.tsv'
fif_path = f'{base_path}/segmented_data/{sub}/{sub}_{seg}_{stim}_eeg.fif'

# Load the data in MNE
raw = mne.io.read_raw_fif(fif_path, preload=True)

# Remove bad channels here

# Get sampling rate
sampling_rate = raw.info['sfreq']

# Calculate mean and standard deviation across all electrodes
data = raw.get_data()  # Get the EEG data
mean_data = np.mean(data, axis=1)  # Calculate mean across channels
std_data = np.std(data, axis=1)  # Calculate standard deviation across channels

# Define threshold for identifying bad electrodes (e.g., 5 SD)
threshold = 10 * np.mean(std_data)

# Find indices of electrodes exceeding the threshold
bad_indices = np.where(np.abs(mean_data) > threshold)[0]

# Convert indices to channel names
bad_channels = [raw.ch_names[idx] for idx in bad_indices]

# Interpolate bad electrodes
raw.info['bads'] = bad_channels


# Define the path to save the bad electrodes TSV file
bad_electrodes_path = f'/Users/derekrosenzweig/Documents/GitHub/EEG-Preprocessing/segmented_data/{sub}/bad-elecs.tsv'

# Save the list of bad electrodes to a TSV file
bad_electrodes_df = pd.DataFrame({'bad_electrodes': bad_channels})
bad_electrodes_df.to_csv(bad_electrodes_path, sep='\t', index=False)

# Apply ICA to remove blink artifacts
ica = mne.preprocessing.ICA(n_components=20, random_state=97)
ica.fit(raw)

# Plot ICA components for visual inspection
ica.plot_components()

# Exclude blink-related components interactively
ica.exclude = [18,19,20]  # Set to your excluded components

# Apply ICA artifact removal
ica.apply(raw)

# Apply Notch filter
raw.notch_filter(60)

# Apply bandpass filter
raw.filter(l_freq=1.0, h_freq=15.0)

# Re-reference the data using the 'VREF' channel
raw.set_eeg_reference(['VREF'])

# Extract only the channels starting with 'E'
eeg_channels = [ch for ch in raw.ch_names if ch.startswith('E')]
raw = raw.pick_channels(eeg_channels)

# Apply common average reference (CAR) across remaining electrodes
raw_car = raw.set_eeg_reference('average', projection=True)

# Z-score the data
data = raw_car.get_data()
mean_data = np.mean(data, axis=1)
std_data = np.std(data, axis=1)
zscore_data = (data - mean_data[:, np.newaxis]) / std_data[:, np.newaxis]

# Create a new Raw object with z-scored data
raw_zscored = mne.io.RawArray(zscore_data, raw_car.info)

# Load metadata for words from annotations
word_info = pd.read_csv(word_path, delimiter='\t', encoding='utf-8')

# Create word epochs
word_onsets = (word_info['Start'].values * sampling_rate).astype(int)
word_events = np.column_stack((word_onsets, np.zeros_like(word_onsets), np.ones_like(word_onsets)))

# Ensure picks for epoching are within the valid range
picks = mne.pick_types(raw_zscored.info, eeg=True)

word_epochs = mne.Epochs(raw_zscored, word_events, tmin=-0.2, tmax=0.6, baseline=None, reject=None, flat=None, picks=picks)
word_epochs.metadata = word_info  # Assign metadata to word epochs

# Define the directory path for saving word epochs
word_epochs_dir = os.path.join(base_path, 'derivatives', 'individual', 'word_epochs')
os.makedirs(word_epochs_dir, exist_ok=True)

# Save word epochs with a specific filename
word_epochs_filename = f'word-epo-{sub}-{stim}-{seg}-epo.fif'
word_epochs_filepath = os.path.join(word_epochs_dir, word_epochs_filename)
word_epochs.save(word_epochs_filepath, overwrite=True)
print(f"Word epochs saved to: {word_epochs_filepath}")

# Average epochs for evoked response
word_evoked = word_epochs.average()

# Define the path for saving the evoked figure
evoked_fig_name = f'word-evoked-{sub}-{stim}_{seg}.jpg'
evoked_fig_path = os.path.join(base_path, 'vis', 'individual', 'word_evoked', comp, sub, evoked_fig_name)
os.makedirs(os.path.dirname(evoked_fig_path), exist_ok=True)

# Plot the evoked response and retrieve the Figure object
fig = word_evoked.plot_joint()

# Save the figure directly from the Figure object
fig.savefig(evoked_fig_path, format='jpg', dpi=300)
