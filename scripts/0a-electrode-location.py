import mne
import os

# Define the subject
sub = 'pilot-2'

# Set file path to the raw EEG data
file = f'/Users/derekrosenzweig/Documents/GitHub/EEG-Speech/data/{sub}_20240417_020809.mff'

# Read in the EEG data with read_raw_egi
raw = mne.io.read_raw_egi(file)



# Plot all EEG electrodes and their topographical locations
fig = raw.plot_sensors(show_names=True)

# Add title and customize the plot if needed
fig.suptitle('EEG Electrodes Topographical Locations', fontsize=16)

# Show the plot
fig.show()
