import mne
import os

# Set parameters for fif path
sub = 'pilot-3'
stim = 'Jobs2'
seg = 'segment_2'

base_path = '/Users/derekrosenzweig/Documents/GitHub/EEG-Preprocessing'
fif_path = f'{base_path}/segmented_data/{sub}/{sub}_{seg}_{stim}_eeg.fif'

# Load the data in MNE
raw = mne.io.read_raw_fif(fif_path, preload=True)

# Plot raw data
raw.plot(scalings='auto')

# Show the plot interactively in the terminal
raw.plot(scalings='auto').show()

# Bad data segment between
# 39.25s - 43.3s

