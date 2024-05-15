import mne
import numpy as np
import pandas as pd
import os

# Set parameters for fif path
sub = 'pilot-2'
stim = 'Jobs1'
seg = 'segment_1'
comp = 'no-ica'  # or 'with-ica' depending on your analysis

base_path = '/Users/derekrosenzweig/Documents/GitHub/EEG-Preprocessing'
fif_path = f'{base_path}/segmented_data/{sub}/{sub}_{seg}_{stim}_eeg.fif'
word_path = f'{base_path}/annotations/words/tsv/{stim}-words.tsv'

# Load the data in MNE
raw = mne.io.read_raw_fif(fif_path, preload=True)

# Plot raw data
raw.plot(scalings='auto')

# Calculate the mean and standard deviation for each channel
eeg_data = raw.get_data()
channel_means = np.mean(eeg_data, axis=1)
channel_stds = np.std(eeg_data, axis=1)

# Create a DataFrame to store the summary statistics
summary_stats = pd.DataFrame({
    'Channel': raw.ch_names,
    'Mean': channel_means,
    'Standard Deviation': channel_stds
})

# Print the summary statistics
print("\nSummary Statistics:")
print(summary_stats)

# Save the summary statistics to a CSV file
summary_stats_filepath = os.path.join(base_path, 'segmented_data', 'summary_statistics.csv')
summary_stats.to_csv(summary_stats_filepath, index=False)
print(f"\nSummary statistics saved to '{summary_stats_filepath}'")

# Load metadata for words from annotations
word_info = pd.read_csv(word_path, delimiter='\t', encoding='utf-8')

# Create word epochs
sampling_rate = raw.info['sfreq']
word_onsets = (word_info['Start'].values * sampling_rate).astype(int)
word_events = np.column_stack((word_onsets, np.zeros_like(word_onsets), np.ones_like(word_onsets)))
word_epochs = mne.Epochs(raw, word_events, tmin=-0.2, tmax=0.6, baseline=None, preload=True)
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
