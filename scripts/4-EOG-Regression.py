import mne
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import zscore
from mne.preprocessing import ICA, EOGRegression
import json

def run_ica_and_eog_regression(sub, stim, seg, base_path):
    """
    Runs ICA and EOG regression on the EEG data to reduce the impact of artifacts.

    Parameters:
    - sub: Subject identifier.
    - stim: Stimulus identifier.
    - seg: Segment identifier.
    - base_path: Base directory path for the project.

    Returns:
    - epochs_clean: Epochs after ICA and EOG regression.
    - evoked_clean: Evoked response after ICA and EOG regression.
    """
    # Set parameters for fif path
    word_path = f'{base_path}/annotations/words/tsv/{stim}-words.tsv'
    phoneme_path = f'{base_path}/annotations/phonemes/tsv/{stim}-phonemes.tsv'
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
    threshold = 4 * np.mean(std_data)

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
    ica = ICA(n_components=10, random_state=42)
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

    # Set Notch filter
    raw.notch_filter(60)

    # Set filter
    raw.filter(l_freq=1, h_freq=15.0)

    # Re-reference the data using the 'VREF' channel
    raw.set_eeg_reference(['VREF'])

    # Extract only the channels starting with 'E'
    eeg_channels = [ch for ch in raw.ch_names if ch.startswith('E')]
    raw = raw.pick_channels(eeg_channels)

    # Change channel types to 'eog'
    eog_channels = ['E126', 'E127']
    raw.set_channel_types({ch: 'eog' for ch in eog_channels})

    # Apply common average reference
    raw_car = raw.set_eeg_reference('average', projection=True)

    # Load metadata for words from annotations
    word_info = pd.read_csv(word_path, delimiter='\t', encoding='utf-8')

    # Create word epochs

    # Create word epochs
    word_onsets = (word_info['Start'].values * sampling_rate).astype(int)
    word_events = np.column_stack((word_onsets, np.zeros_like(word_onsets), np.ones_like(word_onsets)))
    word_epochs = mne.Epochs(raw_car, word_events, tmin=-0.2, tmax=0.6, baseline=None, reject=None, flat=None,
                             preload=True)

    # Drop bad epochs
    word_epochs.drop_bad()

    # Assign metadata to word epochs
    word_epochs.metadata = word_info.iloc[:len(word_epochs.events)]

    # Run EOG regression
    model = EOGRegression(picks="eeg", picks_artifact="eog").fit(word_epochs)

    # Plot regression coefficients (EOG channels) as topomap
    fig_regression = model.plot(vlim=(None, 0.4))
    fig_regression.set_size_inches(3, 2)
    plt.show()

    # Apply EOG regression to the epochs
    epochs_clean = model.apply(word_epochs)
    # epochs_clean.apply_baseline()

    # Compute the evoked response on the corrected data
    evoked_clean = epochs_clean.average()

    # Define the directory path for saving figures
    fig_dir = os.path.join(base_path, 'vis', 'individual', 'EOG', sub)
    os.makedirs(fig_dir, exist_ok=True)

    # Save the regression coefficients figure
    regression_fig_path = os.path.join(fig_dir, f'EOG_regression_{sub}_{stim}_{seg}.png')
    fig_regression.savefig(regression_fig_path)
    plt.close(fig_regression)

    # Plot the evoked response and save the figure
    fig_evoked = evoked_clean.plot_joint()
    evoked_fig_path = os.path.join(fig_dir, f'EOG_evoked_{sub}_{stim}_{seg}.png')
    fig_evoked.savefig(evoked_fig_path)
    plt.close(fig_evoked)

    return epochs_clean, evoked_clean

if __name__ == '__main__':
    # Set parameters for fif path
    sub = 'pilot-3'
    stim = 'Jobs3'
    seg = 'segment_3'
    base_path = '/Users/derekrosenzweig/Documents/GitHub/EEG-Preprocessing'

    # Run ICA and EOG regression
    epochs_clean, evoked_clean = run_ica_and_eog_regression(sub, stim, seg, base_path)