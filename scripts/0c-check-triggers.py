import os
import soundfile as sf
import numpy as np

# Directory containing the modified WAV files
wav_eeg_dir = '/Users/derekrosenzweig/PycharmProjects/annotation-categories/stimuli/wav_eeg'

# Function to count triggers in the WAV file
def count_triggers(filename):
    try:
        # Load the modified WAV file
        y, sr = sf.read(filename)

        # Extract the triggers from the second channel
        triggers = y[:, 1]

        # Threshold for trigger detection (adjust as needed)
        trigger_threshold = 0.5

        # Count the number of triggers
        num_triggers = np.sum(triggers > trigger_threshold)

        return num_triggers

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return 0

# Initialize total trigger count
total_triggers = 0

# Loop through each file in the directory
for filename in os.listdir(wav_eeg_dir):
    if filename.endswith('.wav'):
        wav_file = os.path.join(wav_eeg_dir, filename)
        num_triggers = count_triggers(wav_file)
        total_triggers += num_triggers
        print(f"Number of triggers in {filename}: {num_triggers}")

print(f"Total number of triggers across all WAV files: {total_triggers}")