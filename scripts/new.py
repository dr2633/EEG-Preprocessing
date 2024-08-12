import mne
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
import json
from scipy.stats import zscore

# Set parameters for fif path
sub = 'pilot-3'
stim = 'Jobs1'
seg = 'segment_1'
comp = 'ica'

base_path = '/Users/derekrosenzweig/Documents/GitHub/EEG-Preprocessing'
word_path = f'{base_path}/annotations/words/tsv/{stim}-words.tsv'
phoneme_path = f'{base_path}/annotations/phonemes/tsv/{stim}-phonemes.tsv'
fif_path = f'{base_path}/segmented_data/{sub}/{sub}_{seg}_{stim}_eeg.fif'

# Load the data in MNE
raw = mne.io.read_raw_fif(fif_path, preload=True)

# Get sampling rate
sampling_rate = raw.info['sfreq']

# Set Notch filter
raw.notch_filter(60)

# Set filter
raw.filter(l_freq=1.0, h_freq=15.0)

# For visualizing raw data in terminal
# raw.plot()

# find bad channels from gui
raw.info['bads'] = ['E105', 'E52']

# Apply common average reference (CAR) across remaining electrodes
raw_car = raw.set_eeg_reference('average', projection=True)

# Apply ICA before filtering
ica = ICA(n_components=20, random_state=35)
ica.fit(raw_car)

# When running in terminal to visualize sources
ica.plot_sources(raw_car)

# List of components to exclude
components_to_exclude = [1,2]

# Remove just ICs that we know are blinky
ica.exclude = components_to_exclude

# remove just ics that we know are blinky
ica.apply(raw_car)


# Load metadata for words from annotations
word_info = pd.read_csv(word_path, delimiter='\t', encoding='utf-8')

# Create word epochs
word_onsets = (word_info['Start'].values * sampling_rate).astype(int)
word_events = np.column_stack((word_onsets, np.zeros_like(word_onsets), np.ones_like(word_onsets)))
word_epochs = mne.Epochs(raw_car, word_events, tmin=-3, tmax=3, baseline=None, reject=None, flat=None)
word_epochs.metadata = word_info  # Assign metadata to word epochs

word_evoked = word_epochs.average()


# Define the path for saving the evoked figure
evoked_fig_name = f'word-evoked-{sub}-{stim}_{seg}.jpg'
evoked_fig_path = os.path.join(base_path, 'vis', 'new', 'word_evoked', comp, sub, evoked_fig_name)
os.makedirs(os.path.dirname(evoked_fig_path), exist_ok=True)

# Plot the evoked response and retrieve the Figure object
fig = word_evoked.plot_joint()

# Save the figure directly from the Figure object
fig.savefig(evoked_fig_path, format='jpg', dpi=300)


phoneme_info = pd.read_csv(phoneme_path, delimiter='\t', encoding='utf-8')

phoneme_onsets = (phoneme_info['Start'].values * sampling_rate).astype(int)
phoneme_events = np.column_stack((phoneme_onsets, np.zeros_like(phoneme_onsets), np.ones_like(phoneme_onsets)))
min_len = min(len(phoneme_events), len(phoneme_info))
phoneme_events = phoneme_events[:min_len]
phoneme_info = phoneme_info[:min_len]
phoneme_epochs = mne.Epochs(raw_car, phoneme_events, tmin=-3, tmax=3, preload=True, baseline=None,
                            event_repeated='drop')

# Align metadata with epochs
phoneme_epochs.metadata = phoneme_info.iloc[phoneme_epochs.selection]

phoneme_evoked = phoneme_epochs.average()

# Define the path for saving the evoked figure
evoked_fig_name = f'phoneme-evoked-{sub}-{stim}_{seg}.jpg'
evoked_fig_path = os.path.join(base_path, 'vis', 'new', 'phoneme_evoked', comp, sub, evoked_fig_name)
os.makedirs(os.path.dirname(evoked_fig_path), exist_ok=True)

# Plot the evoked response and retrieve the Figure object
fig = phoneme_evoked.plot_joint()

# Save the figure directly from the Figure object
fig.savefig(evoked_fig_path, format='jpg', dpi=300)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, KFold

# Decoding whether a phoneme is voiced or unvoiced using logistic regression
clf = make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear'))
cv = KFold(5, shuffle=True)

y = (phoneme_epochs.metadata['manner'] == 'f').astype(int).values
X = phoneme_epochs.get_data()  # Get the epoch data

accuracy_scores = []

# Perform decoding across time points
for tt in range(X.shape[2]):
    print(f'Time point {tt}')
    X_ = X[:, :, tt]
    scores = cross_val_score(clf, X_, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    accuracy_scores.append(scores.mean())

# Convert accuracy scores to numpy array for easier indexing
accuracy_scores = np.array(accuracy_scores)

# Plot the decoding results
plt.figure(figsize=(10, 5))
times = phoneme_epochs.times
plt.plot(times, accuracy_scores, label='AUC Score')
plt.axhline(0.5, color='r', linestyle='--', label='Chance Level')
plt.xlabel('Time (s)')
plt.ylabel('AUC Score')
plt.title('Decoding Fricatives')
plt.legend()
plt.grid(True)

# Define the path for saving the decoding figure
decoding_fig_name = f'decoding-results-{sub}-{stim}_{seg}.jpg'
decoding_fig_path = os.path.join(base_path, 'vis', 'new', 'decoding_results', comp, sub, decoding_fig_name)
os.makedirs(os.path.dirname(decoding_fig_path), exist_ok=True)

# Save the decoding figure
plt.savefig(decoding_fig_path, format='jpg', dpi=300)
plt.show()