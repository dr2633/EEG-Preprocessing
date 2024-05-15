import mne
import os
import matplotlib.pyplot as plt

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

# Save the figure
fig_path = os.path.join(base_path, 'vis', 'individual', 'raw_data', f'raw_data_{sub}_{seg}_{stim}.png')
os.makedirs(os.path.dirname(fig_path), exist_ok=True)
plt.savefig(fig_path)

# Show the plot
plt.show()


# pilot-3

# E105,E106,E107 bad channels

#Blinking artifacts
# E119 - 128
# E8 - 25
