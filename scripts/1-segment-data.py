import mne
import numpy as np
import librosa
import os
import pandas as pd
from collections import defaultdict

sub = 'pilot-3'

base_path = '/Users/derekrosenzweig/Documents/GitHub/EEG-Preprocessing'

# Set file path
file = f'{base_path}/data/{sub}.mff'

# Set the directory path to save the segmented data
segmented_data_dir = f'{base_path}/segmented_data'
sub_dir = os.path.join(segmented_data_dir, sub)
os.makedirs(sub_dir, exist_ok=True)

# Read in mff data with read_raw_egi
raw = mne.io.read_raw_egi(file)

# Get channel names
channel_names = raw.ch_names
print("\nChannel names:")
print(channel_names)

# Find the trigger channel
trigger_channel_name = 'STI 014'
trigger_channel_index = channel_names.index(trigger_channel_name)
print(f"\nTrigger channel '{trigger_channel_name}' found at index:", trigger_channel_index)

# Find the reference channel
reference_channel_name = 'VREF'
reference_channel_index = channel_names.index(reference_channel_name)
print(f"\nReference channel '{reference_channel_name}' found at index:", reference_channel_index)

# Extract EEG data
eeg_data = raw.get_data()

# Print shape of EEG data
print("\nShape of EEG data:", np.shape(eeg_data))

# Access trigger channel data
trigger_channel_data = eeg_data[trigger_channel_index, :]

# Find events for channel 1 only
events = mne.find_events(raw, stim_channel=trigger_channel_name)

# Filter events where the third column is equal to 1
events_channel_1 = events[events[:, 2] == 1]

# Print the first few events for channel 1
print("\nFirst few events for channel 1:")
print(events_channel_1[:5])

# Get the sample rate
sample_rate = raw.info['sfreq']
print(f"\nSample rate: {sample_rate} Hz")

# Print info
print("\nRaw info:")
print(raw.info)

# Create a DataFrame to store events and timestamps
event_timestamps = pd.DataFrame(events_channel_1, columns=['Sample', 'Offset', 'Event'])
event_timestamps['Timestamp'] = event_timestamps['Sample'] / sample_rate

# Set the directory path to save the CSV file
save_dir = f'{base_path}/segmented_data/stim-onset'

# Create the directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Save the DataFrame as a CSV file
csv_filename = f'{sub}_event_timestamps.csv'
csv_filepath = os.path.join(save_dir, csv_filename)
event_timestamps.to_csv(csv_filepath, index=False)
print(f"\nEvent timestamps saved to '{csv_filepath}'")

# Sort the events by timestamp
event_timestamps = event_timestamps.sort_values('Timestamp')

# Initialize variables to keep track of segments
segments = []
segment_start = None
prev_timestamp = None
prev_event = None

# Iterate through the events
for _, row in event_timestamps.iterrows():
    timestamp = row['Timestamp']
    event = row['Event']

    # Check if at the start of a new segment
    if segment_start is None:
        segment_start = timestamp
        prev_event = event
    else:
        # If the difference between the current and previous timestamp is greater than 20 seconds or
        # the current event is different from the previous event, it's the start of a new segment
        if (timestamp - prev_timestamp > 20) or (event != prev_event):
            segment_end = prev_timestamp
            segment_duration = segment_end - segment_start
            segments.append((segment_start, segment_end, prev_event, segment_duration))
            segment_start = timestamp
            prev_event = event

    prev_timestamp = timestamp

# Add the last segment if it exists
if segment_start is not None:
    segment_end = prev_timestamp
    segment_duration = segment_end - segment_start
    segments.append((segment_start, segment_end, prev_event, segment_duration))

# Create a DataFrame from the segments list
segments_df = pd.DataFrame(segments, columns=['start', 'end', 'event', 'duration'])

# Save the segments DataFrame as a CSV file
segments_csv_filename = f'{sub}_segments.csv'
segments_csv_filepath = os.path.join(save_dir, segments_csv_filename)
segments_df.to_csv(segments_csv_filepath, index=False)
print(f"\nSegments saved to '{segments_csv_filepath}'")

# Load the WAV durations DataFrame
wav_durations_df = pd.read_csv(f'{base_path}/segmented_data/stim-onset/wav_durations.csv')

# Define the order of WAV files
wav_files = ['Jobs1.wav', 'Jobs2.wav', 'Jobs3.wav',
             'BecFast.wav','AttSlow.wav', 'CampFast.wav',
             'BecSlow.wav', 'AttFast.wav','CampSlow.wav'
             'Jobs1.wav','Jobs2.wav', 'Jobs3.wav']

# Get the sample rate from the raw data
sampling_rate = raw.info['sfreq']

# Iterate through the segments and WAV files
for i, (_, segment) in enumerate(segments_df.iterrows()):
    filename = wav_files[i]
    start_time = segment['start']
    end_time = segment['end']
    wav_duration = wav_durations_df.loc[wav_durations_df['filename'] == filename, 'duration'].values[0]

    # Convert times to samples
    start_sample = int(start_time * sampling_rate)
    end_sample = int(end_time * sampling_rate)

    # Extract the segment
    segment_data = raw.get_data(start=start_sample, stop=end_sample)

    # Create a new Raw object for the segment
    segment_raw = mne.io.RawArray(segment_data, raw.info, verbose='WARNING')

    # Save the segment as a FIF file
    segment_filename = f'{sub}_segment_{i + 1}_{filename.split(".")[0]}_eeg.fif'
    segment_filepath = os.path.join(sub_dir, segment_filename)
    segment_raw.save(segment_filepath, overwrite=True)
    print(f"Segment {i + 1} saved as '{segment_filepath}'")