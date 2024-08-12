# When running epochs with window size -3 to 3 s, the process gets interrupted


import mne
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, KFold
import os
import csv
import pandas as pd


def load_and_preprocess_data(fif_path):
    raw = mne.io.read_raw_fif(fif_path, preload=True)
    raw.filter(l_freq=1.0, h_freq=30.0)
    #raw.set_eeg_reference(['VREF'])
    eeg_channels = [ch for ch in raw.ch_names if ch.startswith('E')]
    raw = raw.pick_channels(eeg_channels)
    raw_car = raw.set_eeg_reference('average', projection=True)
    return raw_car


def create_phoneme_epochs(raw_car, phoneme_info, sampling_rate):
    phoneme_onsets = (phoneme_info['Start'].values * sampling_rate).astype(int)
    phoneme_events = np.column_stack((phoneme_onsets, np.zeros_like(phoneme_onsets), np.ones_like(phoneme_onsets)))
    min_len = min(len(phoneme_events), len(phoneme_info))
    phoneme_events = phoneme_events[:min_len]
    phoneme_info = phoneme_info[:min_len]
    phoneme_epochs = mne.Epochs(raw_car, phoneme_events, tmin=-1, tmax=1, preload=True, baseline=None,
                                event_repeated='drop')

    # Align metadata with epochs
    phoneme_epochs.metadata = phoneme_info.iloc[phoneme_epochs.selection]

    return phoneme_epochs

def filter_epochs(epochs, desired_phonation_value, desired_manner_value, desired_place_value, desired_roundness_value, desired_frontback_value):
    filtered_epochs_phonation = epochs[epochs.metadata['phonation'] == desired_phonation_value]
    filtered_epochs_manner = epochs[epochs.metadata['manner'] == desired_manner_value]
    filtered_epochs_place = epochs[epochs.metadata['place'] == desired_place_value]
    filtered_epochs_roundness = epochs[epochs.metadata['roundness'] == desired_roundness_value]
    filtered_epochs_frontback = epochs[epochs.metadata['frontback'] == desired_frontback_value]
    filtered_epochs = mne.concatenate_epochs([filtered_epochs_phonation, filtered_epochs_manner, filtered_epochs_place, filtered_epochs_roundness, filtered_epochs_frontback])
    return filtered_epochs


def perform_decoding(filtered_epochs, desired_phonation_value, desired_manner_value, desired_place_value, desired_roundness_value, desired_frontback_value):
    clf = make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear'))
    cv = KFold(5, shuffle=True)
    accuracy_dict = {}

    for feat, label, color in zip(['phonation', 'manner', 'place', 'roundness', 'frontback'],
                                  ['Voiced', 'Fricatives', 'Vowels', 'Rounded', 'Front'],
                                  ['indigo', 'darkorchid', 'plum', 'pink', 'palevioletred']):
        desired_value = {
            'phonation': desired_phonation_value,
            'manner': desired_manner_value,
            'place': desired_place_value,
            'roundness': desired_roundness_value,
            'frontback': desired_frontback_value
        }[feat]
        y = (filtered_epochs.metadata[feat] == desired_value).astype(int)
        accuracy_scores = np.empty(filtered_epochs.get_data(copy=True).shape[-1])

        for tt in range(accuracy_scores.shape[0]):
            X_ = filtered_epochs.get_data(copy=True)[:, :, tt]
            scores = cross_val_score(clf, X_, y, scoring='roc_auc', cv=cv, n_jobs=-1)
            accuracy_scores[tt] = scores.mean()

        accuracy_dict[feat] = accuracy_scores

    return accuracy_dict


def visualize_results(filtered_epochs, accuracy_dict, fig_path, sub, seg, stim):
    y_min = 0.45
    y_max = 0.75

    fig, ax = plt.subplots(1, figsize=(10, 5))
    plt.title(f"Decoding Accuracy for {sub}")

    for feat, label, color in zip(['phonation', 'manner', 'place', 'roundness', 'frontback'],
                                  ['Voiced', 'Fricatives', 'Vowels', 'Rounded', 'Front'],
                                  ['indigo', 'darkorchid', 'plum', 'pink', 'palevioletred']):
        ax.plot(filtered_epochs.times, accuracy_dict[feat], label=label, color=color)

    ax.axvline(x=0, color='grey', linestyle='--')
    ax.axhline(y=0.5, color='grey', linestyle='--')
    ax.set_xlabel("Time (ms) relative to phoneme onset")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(y_min, y_max)
    ax.legend()
    plt.savefig(f'{fig_path}/{sub}_{seg}_{stim}_logistic.jpg', dpi=300, bbox_inches='tight')
    plt.show()


def save_accuracy_scores(accuracy_dict, base_path):
    csv_file_path = os.path.join(base_path, 'accuracy_scores.csv')
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Key', 'Accuracy Scores'])
        for key, scores in accuracy_dict.items():
            writer.writerow([key, scores])


def main():
    # Set parameters for fif path
    sub = 'pilot-3'
    stim = 'Jobs1'
    seg = 'segment_1'
    comp = 'no-ica'

    base_path = '/Users/derekrosenzweig/Documents/GitHub/EEG-Preprocessing'
    phoneme_path = f'{base_path}/annotations/phonemes/tsv/{stim}-phonemes.tsv'
    fif_path = f'{base_path}/segmented_data/{sub}/{sub}_{seg}_{stim}_eeg.fif'
    fig_path = os.path.join(base_path, 'vis', 'individual', 'phoneme-decode')
    os.makedirs(fig_path, exist_ok=True)

    sampling_rate = mne.io.read_raw_fif(fif_path, preload=False).info['sfreq']

    raw_car = load_and_preprocess_data(fif_path)
    phoneme_info = pd.read_csv(phoneme_path, delimiter='\t', encoding='utf-8')
    phoneme_epochs = create_phoneme_epochs(raw_car, phoneme_info, sampling_rate)

    desired_phonation_value = 'v'
    desired_manner_value = 'f'
    desired_place_value = 'm'
    desired_roundness_value = 'r'
    desired_frontback_value = 'f'

    filtered_epochs = filter_epochs(phoneme_epochs, desired_phonation_value, desired_manner_value, desired_place_value, desired_roundness_value, desired_frontback_value)
    filtered_epochs.resample(100)

    accuracy_dict = perform_decoding(filtered_epochs, desired_phonation_value, desired_manner_value, desired_place_value, desired_roundness_value, desired_frontback_value)
    visualize_results(filtered_epochs, accuracy_dict, fig_path, sub, seg, stim)
    save_accuracy_scores(accuracy_dict, base_path)
    print("Decoding analysis completed.")


if __name__ == '__main__':
    main()
