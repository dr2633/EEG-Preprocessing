import numpy as np
import mne

def detect_artifacts(raw_data, threshold_factor=8):
    """
    Automatically detect artifacts in the EEG data.

    Parameters:
    raw_data (numpy.ndarray): Raw EEG data (channels x samples).
    threshold_factor (float): Factor to multiply the channel variance threshold (default=3).

    Returns:
    tuple: Tuple containing start and end time indices of the detected artifact, or None if no artifact is detected.
    """
    # Calculate the variance of each channel across the entire recording
    channel_variances = np.var(raw_data, axis=1)

    # Identify channels with unusually high variance
    high_variance_channels = \
    np.where(channel_variances > np.mean(channel_variances) + threshold_factor * np.std(channel_variances))[0]

    # Initialize variables to track the start and end time indices of the artifact
    artifact_start_idx = None
    artifact_end_idx = None

    # Loop through all segments of 1-second duration with 50% overlap
    segment_length = raw_data.shape[1] // 2
    for i in range(0, raw_data.shape[1], segment_length):
        segment_start_idx = i
        segment_end_idx = min(i + segment_length, raw_data.shape[1])

        # Check activity during the segment for each high variance channel
        for ch_idx in high_variance_channels:
            channel_activity = raw_data[ch_idx, segment_start_idx:segment_end_idx]
            channel_mean_activity = np.mean(channel_activity)
            channel_std_activity = np.std(channel_activity)

            # Calculate threshold for detecting artifacts (e.g., 3 SD from mean activity)
            threshold = channel_mean_activity + threshold_factor * channel_std_activity

            # Check if any activity during the segment exceeds the threshold
            if np.any(channel_activity > threshold):
                if artifact_start_idx is None:
                    artifact_start_idx = segment_start_idx
                artifact_end_idx = segment_end_idx

    # Return the start and end time indices of the detected artifact
    if artifact_start_idx is not None and artifact_end_idx is not None:
        return artifact_start_idx, artifact_end_idx
    else:
        return None

# Set parameters for fif path
sub = 'pilot-3'
stim = 'Jobs2'
seg = 'segment_2'

base_path = '/Users/derekrosenzweig/Documents/GitHub/EEG-Preprocessing'
fif_path = f'{base_path}/segmented_data/{sub}/{sub}_{seg}_{stim}_eeg.fif'

# Load the data in MNE
raw = mne.io.read_raw_fif(fif_path, preload=True)

# Extract EEG data
raw_data = raw.get_data()

# Detect artifacts in the EEG data
artifact_indices = detect_artifacts(raw_data)
if artifact_indices is not None:
    start_time = raw.times[artifact_indices[0]]
    end_time = raw.times[artifact_indices[1]]
    print("Artifact detected from {}s to {}s.".format(start_time, end_time))
else:
    print("No artifact detected.")
