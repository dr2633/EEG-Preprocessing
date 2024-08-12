import numpy as np
from scipy.io import wavfile
import os

def create_tone_with_silence_and_impulse(duration, tone_freq, tone_duration, silence_duration, sample_rate):
    # Calculate samples for tone and silence
    tone_samples = int(tone_duration * sample_rate)
    silence_samples = int(silence_duration * sample_rate)
    total_samples = int(duration * sample_rate)

    # Create time array for a single tone
    t_tone = np.linspace(0, tone_duration, tone_samples, False)

    # Generate single tone
    single_tone = np.sin(2 * np.pi * tone_freq * t_tone)

    # Apply a simple envelope to avoid clicks
    envelope = np.ones_like(single_tone)
    ramp_samples = int(0.01 * sample_rate)  # 10 ms ramp
    envelope[:ramp_samples] = np.linspace(0, 1, ramp_samples)
    envelope[-ramp_samples:] = np.linspace(1, 0, ramp_samples)
    single_tone *= envelope

    # Create the full audio with alternating tone and silence
    full_audio = np.zeros(total_samples)
    impulse_channel = np.zeros(total_samples)
    for i in range(0, total_samples, tone_samples + silence_samples):
        if i + tone_samples <= total_samples:
            full_audio[i:i + tone_samples] = single_tone
            impulse_channel[i] = 1  # Add impulse at the start of each tone

    return full_audio, impulse_channel

# Parameters
duration = 10 * 60  # Total duration in seconds (10 minutes)
tone_freq = 1000  # Hz
tone_duration = 0.2  # 200 ms
silence_duration = 0.8  # 800 ms
sample_rate = 44100  # Hz

# Create the audio and impulse channel
audio, impulse_channel = create_tone_with_silence_and_impulse(duration, tone_freq, tone_duration, silence_duration, sample_rate)

# Normalize audio to prevent clipping
audio = audio / np.max(np.abs(audio))

# Create stereo audio (audio in left channel, impulses in right channel)
stereo_audio = np.column_stack((audio, impulse_channel))

# Define the output path
output_path = '/Users/derekrosenzweig/Documents/GitHub/Speech-Decoding/eeg-tones/wav/repeated_tones_with_impulses.wav'

# Save as WAV file
wavfile.write(output_path, sample_rate, (stereo_audio * 32767).astype(np.int16))

print(f"WAV file '{output_path}' has been created.")