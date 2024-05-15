import mne
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# Set parameters for fif path
sub = 'pilot-3'
stim = 'Jobs2'
seg = 'segment_2'
comp = 'with-ica'  # two options: 'no-ica' or 'with-ica'

base_path = '/Users/derekrosenzweig/Documents/GitHub/EEG-Preprocessing'
word_path = f'{base_path}/annotations/words/tsv/{stim}-words.tsv'
fif_path = f'{base_path}/segmented_data/{sub}/{sub}_{seg}_{stim}_eeg.fif'

# Load the data in MNE
raw = mne.io.read_raw_fif(fif_path, preload=True)

# Remove bad channels
# These channels are manually identified based on noise or consistent artifacts
bad_channels = ['E105', 'E106', 'E107', 'E119', 'E120', 'E121', 'E122', 'E123', 'E124', 'E125', 'E126', 'E127', 'E128',
                'E8', 'E9', 'E10', 'E11', 'E12', 'E13', 'E14', 'E15', 'E16', 'E17', 'E18', 'E19', 'E20', 'E21', 'E22',
                'E23', 'E24', 'E25','E52', 'E26', 'E32']

# Exclude bad channels from further analysis
raw.info['bads'] = bad_channels

# Save the raw data with bad channels marked
raw.save(fif_path[:-8] + '_eeg.fif', overwrite=True)

# Plot raw data
# raw.plot(scalings='auto')
#
# # Save the figure
# fig_path = os.path.join(base_path, 'vis', 'individual', 'raw_data', f'raw_data_{sub}_{seg}_{stim}.png')
# os.makedirs(os.path.dirname(fig_path), exist_ok=True)
# plt.savefig(fig_path)

# Show the plot
plt.show()

print(raw.info['bads'])

# Apply ICA to remove blink artifacts
ica = mne.preprocessing.ICA(n_components=20, random_state=97)
ica.fit(raw)


ica.plot_sources(raw)
# Plot ICA components for visual inspection
# ica.plot_components()

# Exclude blink-related components interactively
ica.exclude = [15,17]  # Set to your excluded components

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
word_onsets = (word_info['Start'].values * raw_zscored.info['sfreq']).astype(int)
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
