# This is a bonus script that I use to plot the ICA sources in my terminal
# I use this for quick interactive visualization

import mne
import matplotlib.pyplot as plt
from mne.preprocessing import ICA

# Set parameters for fif path
sub = 'pilot-3'
stim = 'Jobs2'
seg = 'segment_2'

base_path = '/Users/derekrosenzweig/Documents/GitHub/EEG-Preprocessing'
fif_path = f'{base_path}/segmented_data/{sub}/{sub}_{seg}_{stim}_eeg.fif'

# Load the data in MNE
raw = mne.io.read_raw_fif(fif_path, preload=True)

# Apply ICA before plotting sources
ica = ICA(n_components=10, random_state=42)
ica.fit(raw)

# Plot ICA sources
ica_sources_fig = ica.plot_sources(raw)
plt.show()  # Display the plot


ica_components_fig = ica.plot_components()
plt.show()