import mne
import numpy as np
import pandas as pd
import os
from mne.preprocessing import ICA
import json

# Set parameters for fif path
sub = 'pilot-3'
stim = 'Jobs3'
seg = 'segment_3'
comp = 'no-ica'

base_path = '/Users/derekrosenzweig/Documents/GitHub/EEG-Preprocessing'
fif_path = f'{base_path}/segmented_data/{sub}/{sub}_{seg}_{stim}_eeg.fif'
word_path = f'{base_path}/annotations/words/tsv/{stim}-words.tsv'

# Load the data in MNE
raw = mne.io.read_raw_fif(fif_path, preload=True)

# Mark channels as bad
raw.info['bads'] = ['E105', 'E52', 'E127','E128']

# Set Notch filter
raw.notch_filter(60)

# Apply ICA before filtering
ica = ICA(n_components=20, random_state=35)
ica.fit(raw)

ica.plot_sources(raw)

# Select components from ICA to remove
ica.exclude = []

# Save the excluded components to a JSON file
excluded_components = {
    'subject': sub,
    'segment': seg,
    'stimulus': stim,
    'excluded_components': ica.exclude
}
json_dir = os.path.join(base_path, 'derivatives', 'individual', 'ica_excluded_components')
os.makedirs(json_dir, exist_ok=True)
json_filename = f'{sub}_{seg}_{stim}_excluded_components.json'
json_filepath = os.path.join(json_dir, json_filename)
with open(json_filepath, 'w') as json_file:
    json.dump(excluded_components, json_file)

ica.apply(raw)


# Set band-pass filter
raw.filter(l_freq=1.0, h_freq=15.0)

# Extract only the channels starting with 'E'
eeg_channels = [ch for ch in raw.ch_names if ch.startswith('E')]
raw.pick_channels(eeg_channels)

# Apply common average reference (CAR) across remaining electrodes
raw_car = raw.set_eeg_reference('average', projection=True)

# Load metadata for words from annotations
word_info = pd.read_csv(word_path, delimiter='\t', encoding='utf-8')

# Create word epochs
sampling_rate = raw.info['sfreq']
word_onsets = (word_info['Start'].values * sampling_rate).astype(int)
word_events = np.column_stack((word_onsets, np.zeros_like(word_onsets), np.ones_like(word_onsets)))

# Check if the number of word events matches the number of rows in word_info
print("Number of word events:", len(word_events))
print("Number of rows in word_info:", len(word_info))

# Ensure that the lengths match
min_length = min(len(word_events), len(word_info))
word_events = word_events[:min_length]
word_info = word_info.iloc[:min_length]

word_epochs = mne.Epochs(raw_car, word_events, tmin=-0.1, tmax=0.3, baseline=None, reject=None, flat=None, preload=True)
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
print(f"Evoked response figure saved to: {evoked_fig_path}")
