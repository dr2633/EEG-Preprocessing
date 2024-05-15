import mne
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
import json
from scipy.stats import zscore

def save_ica_plots_and_json(ica, subject, segment, stimulus, round, base_path):
    ica_fig_dir = os.path.join(base_path, 'vis', 'individual', f'ica_{round}_filtering', subject)
    os.makedirs(ica_fig_dir, exist_ok=True)

    ica_components_fig = ica.plot_components()
    ica_components_fig.savefig(os.path.join(ica_fig_dir, f'{subject}_{segment}_{stimulus}_components_{round}.jpg'))
    plt.close(ica_components_fig)

    ica_sources_fig = ica.plot_sources(raw)
    ica_sources_fig.savefig(os.path.join(ica_fig_dir, f'{subject}_{segment}_{stimulus}_sources_{round}.jpg'))
    plt.close(ica_sources_fig)

    excluded_components = {
        'subject': subject,
        'segment': segment,
        'stimulus': stimulus,
        f'excluded_components_{round}': ica.exclude
    }
    json_dir = os.path.join(base_path, 'derivatives', 'individual', f'ica_{round}_filtering', 'excluded_components')
    os.makedirs(json_dir, exist_ok=True)
    json_filename = f'{subject}_{segment}_{stimulus}_excluded_components_{round}.json'
    json_filepath = os.path.join(json_dir, json_filename)
    with open(json_filepath, 'w') as json_file:
        json.dump(excluded_components, json_file)

if __name__ == '__main__':
    # Set parameters for fif path
    subject = 'pilot-2'
    stimulus = 'Jobs1'
    segment = 'segment_1'

    base_path = '/Users/derekrosenzweig/Documents/GitHub/EEG-Preprocessing'
    word_path = os.path.join(base_path, 'annotations', 'words', 'tsv', f'{stimulus}-words.tsv')
    fif_path = os.path.join(base_path, 'segmented_data', '../scratch/pilot-2', f'{subject}_{segment}_{stimulus}_eeg.fif')

    # Load the data in MNE
    raw = mne.io.read_raw_fif(fif_path, preload=True)

    # Get sampling rate
    sampling_rate = raw.info['sfreq']

    # Apply ICA before filtering
    ica_before = ICA(n_components=10, random_state=42)
    ica_before.fit(raw)
    ica_before.exclude = [8,9]  # Example excluded components
    save_ica_plots_and_json(ica_before, subject, segment, stimulus, 'before', base_path)
    ica_before.apply(raw)

    # Set Notch filter
    raw.notch_filter(60)

    # Set filter
    raw.filter(l_freq=1.0, h_freq=15.0)

    # Re-reference the data using the 'VREF' channel
    raw.set_eeg_reference(['VREF'])

    # Extract only the channels starting with 'E'
    eeg_channels = [ch for ch in raw.ch_names if ch.startswith('E')]
    raw = raw.pick_channels(eeg_channels)

    # Apply common average reference
    raw_car = raw.set_eeg_reference('average', projection=True)

    # Apply ICA after filtering
    ica_after = ICA(n_components=10, random_state=42)
    ica_after.fit(raw)
    ica_after.exclude = []  # Example excluded components
    save_ica_plots_and_json(ica_after, subject, segment, stimulus, 'after', base_path)
    ica_after.apply(raw)

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

    # Save word epochs with a specific filename
    word_epochs_filename = f'word-epo-{subject}-{stimulus}-{segment}-epo.fif'
    word_epochs_filepath = os.path.join(word_epochs_dir, word_epochs_filename)
    word_epochs.save(word_epochs_filepath, overwrite=True)
    print(f"Word epochs saved to: {word_epochs_filepath}")

    # Average epochs for evoked response
    word_evoked = word_epochs.average()

    # Define the path for saving the evoked figure
    evoked_fig_name = f'word-evoked-{subject}-{stimulus}_{segment}.jpg'
    evoked_fig_path = os.path.join(base_path, 'vis', 'individual', 'word_evoked', subject, evoked_fig_name)
    os.makedirs(os.path.dirname(evoked_fig_path), exist_ok=True)

    # Plot the evoked response and retrieve the Figure object
    fig = word_evoked.plot_joint()

    # Save the figure directly from the Figure object
    fig.savefig(evoked_fig_path, format='jpg', dpi=300)