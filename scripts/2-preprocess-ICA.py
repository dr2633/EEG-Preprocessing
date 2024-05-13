import mne
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
import json
from scipy.stats import zscore

# Set parameters for fif path
sub = 'pilot-2'
stim = 'Jobs1'
seg = 'segment_1'
comp = 'with-ica' # two options: 'no-ica' or 'with-ica'

base_path = '/Users/derekrosenzweig/Documents/GitHub/EEG-Preprocessing'
word_path = f'{base_path}/annotations/words/tsv/{stim}-words.tsv'
fif_path = f'{base_path}/segmented_data/{sub}/{sub}_{seg}_{stim}_eeg.fif'

# Load the data in MNE
raw = mne.io.read_raw_fif(fif_path, preload=True)

# Get sampling rate
sampling_rate = raw.info['sfreq']


# Calculate mean and standard deviation across all electrodes
data = raw.get_data()  # Get the EEG data
mean_data = np.mean(data, axis=1)  # Calculate mean across channels
std_data = np.std(data, axis=1)  # Calculate standard deviation across channels

# Define threshold for identifying bad electrodes (e.g., 5 SD)
threshold = 6 * np.mean(std_data)

# Find indices of electrodes exceeding the threshold
bad_indices = np.where(np.abs(mean_data) > threshold)[0]

# Convert indices to channel names
bad_channels = [raw.ch_names[idx] for idx in bad_indices]

# Interpolate bad electrodes
raw.info['bads'] = bad_channels
raw.interpolate_bads()

# Define the path to save the bad electrodes TSV file
bad_electrodes_path = f'/Users/derekrosenzweig/Documents/GitHub/EEG-Preprocessing/segmented_data/{sub}/bad-elecs.tsv'

# Save the list of bad electrodes to a TSV file
bad_electrodes_df = pd.DataFrame({'bad_electrodes': bad_channels})
bad_electrodes_df.to_csv(bad_electrodes_path, sep='\t', index=False)

# Apply ICA before filtering
ica = ICA(n_components=20, random_state=35)
ica.fit(raw)

# Define the directory path for saving ICA figures
ica_fig_dir = os.path.join(base_path, 'vis', 'individual', 'ICA', sub)
os.makedirs(ica_fig_dir, exist_ok=True)

# Save ICA component plot
ica_components_fig = ica.plot_components()
ica_components_fig.savefig(os.path.join(ica_fig_dir, f'{sub}_{seg}_{stim}_components.jpg'))
plt.close(ica_components_fig)

# Save ICA source plot
ica_sources_fig = ica.plot_sources(raw)
ica_sources_fig.savefig(os.path.join(ica_fig_dir, f'{sub}_{seg}_{stim}_sources.jpg'))
plt.close(ica_sources_fig)

# Select components from ICA to remove
ica.exclude = [17]

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

# Set Notch filter
raw.notch_filter(60)

# Set filter
raw.filter(l_freq=1.0, h_freq=15.0)

# Re-reference the data using the 'VREF' channel
raw.set_eeg_reference(['VREF'])

# Extract only the channels starting with 'E'
eeg_channels = [ch for ch in raw.ch_names if ch.startswith('E')]
raw = raw.pick_channels(eeg_channels)

# Apply common average reference (CAR) across remaining electrodes
raw_car = raw.set_eeg_reference('average', projection=True)


# Load metadata for words from annotations
word_info = pd.read_csv(word_path, delimiter='\t', encoding='utf-8')

# Create word epochs
word_onsets = (word_info['Start'].values * sampling_rate).astype(int)
word_events = np.column_stack((word_onsets, np.zeros_like(word_onsets), np.ones_like(word_onsets)))
word_epochs = mne.Epochs(raw_car, word_events, tmin=-0.2, tmax=.6, baseline=None, reject=None, flat=None)
word_epochs.metadata = word_info  # Assign metadata to word epochs

# Define the directory path for saving word epochs
word_epochs_dir = os.path.join(base_path, 'derivatives', 'individual', 'word_epochs')
os.makedirs(word_epochs_dir, exist_ok=True)

# # Save word epochs with a specific filename
# word_epochs_filename = f'word-epo-{sub}-{stim}-{seg}-epo.fif'
# word_epochs_filepath = os.path.join(word_epochs_dir, word_epochs_filename)
# word_epochs.save(word_epochs_filepath, overwrite=True)
# print(f"Word epochs saved to: {word_epochs_filepath}")

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