import os
import librosa
import pandas as pd

# Set base path
base_path = '/Users/derekrosenzweig/Documents/GitHub/EEG-Preprocessing'

# Directory containing WAV files
stim_dir = f'{base_path}/stimuli/wav_eeg'

# Function to get the duration of a WAV file
def get_wav_duration(wav_file):
    y, sr = librosa.load(wav_file, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    return duration

# Create a list to store the file names and durations
wav_durations = []

# Loop through all files in the directory
for filename in os.listdir(stim_dir):
    if filename.endswith('.wav'):
        wav_file = os.path.join(stim_dir, filename)
        wav_duration = get_wav_duration(wav_file)
        wav_durations.append({'filename': filename, 'duration': wav_duration})

# Create a DataFrame from the list of dictionaries
df = pd.DataFrame(wav_durations)

# Set the directory path to save the CSV file
save_dir = f'{base_path}/segmented_data/stim-onset'

# Create the directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Save the DataFrame as a CSV file
csv_filename = 'wav_durations.csv'
csv_filepath = os.path.join(save_dir, csv_filename)
df.to_csv(csv_filepath, index=False)
print(f"WAV file durations saved to '{csv_filepath}'")