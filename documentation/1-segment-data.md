# EEG Data Segmentation

This script performs the segmentation of EEG data based on event timestamps and saves the segmented data as FIF files.

## Imports

The script starts by importing the necessary libraries:
- `mne`: MNE-Python library for processing EEG data
- `numpy`: NumPy library for numerical operations
- `librosa`: Librosa library for audio processing
- `os`: Built-in Python module for file and directory operations
- `pandas`: Pandas library for data manipulation and analysis
- `defaultdict`: Default dictionary from the collections module

## Variables

- `sub`: Subject identifier
- `base_path`: Base path for the project directory

## Data Reading

1. Set the file path for the MFF data file using the `base_path` and `sub` variables.
2. Set the directory path to save the segmented data.
3. Read in the MFF data using `mne.io.read_raw_egi()` and store it in the `raw` variable.

## Channel Information

1. Get the channel names from `raw.ch_names` and print them.
2. Find the trigger channel index by searching for the channel name 'STI 014' in `channel_names`.
3. Find the reference channel index by searching for the channel name 'VREF' in `channel_names`.

## EEG Data Extraction

1. Extract the EEG data using `raw.get_data()` and store it in `eeg_data`.
2. Print the shape of `eeg_data`.
3. Access the trigger channel data by indexing `eeg_data` with the trigger channel index.

## Event Detection

1. Find events for channel 1 using `mne.find_events()` with the trigger channel name.
2. Filter events where the third column is equal to 1 and store them in `events_channel_1`.
3. Print the first few events for channel 1.

## Sample Rate

1. Get the sample rate from `raw.info['sfreq']` and print it.

## Event Timestamps

1. Create a DataFrame `event_timestamps` to store events and timestamps.
2. Calculate timestamps by dividing the 'Sample' column by the sample rate.
3. Set the directory path to save the CSV file.
4. Save the `event_timestamps` DataFrame as a CSV file.

## Segmentation

1. Sort the events in `event_timestamps` by timestamp.
2. Initialize variables to keep track of segments.
3. Iterate through the events and identify the start of new segments based on timestamp differences and event changes.
4. Append each segment's start time, stop time, event, and duration to the `segments` list.
5. Create a DataFrame `segments_df` from the `segments` list.

## Segment Extraction and Saving

1. Load the `segments_df` DataFrame from the CSV file.
2. Load the `wav_durations_df` DataFrame containing WAV file durations.
3. Define the order of WAV files in `wav_files`.
4. Get the sample rate from `raw.info['sfreq']`.
5. Iterate through the segments and WAV files:
   - Get the filename, start time, and WAV duration for the current segment.
   - Convert duration from seconds to samples.
   - Calculate the end time.
   - Convert start and end times to samples.
   - Extract the segment data using `raw.get_data()` with the start and end samples.
   - Create a new Raw object for the segment using `mne.io.RawArray()`.
   - Save the segment as a FIF file with a unique filename.

## Output

The script outputs the following:
- Channel names
- Trigger channel index
- Reference channel index
- Shape of EEG data
- First few events for channel 1
- Sample rate
- Raw info
- CSV file path for event timestamps
- FIF file paths for each segmented data file

The segmented data files are saved in the specified directory with filenames containing the subject identifier, segment number, and corresponding WAV file name.