# EEG Preprocessing Pipeline

This repository contains a preprocessing pipeline for EEG data using the MNE library in Python. The pipeline performs various steps to clean and prepare the EEG data for further analysis.

## Pipeline Steps

1. **Loading Data**: The raw EEG data is loaded from a FIF file using MNE. Load the data directly from the Box folder.
- Raw data is stored in the directory 'data'
- Segmented data for each trial is stored in 'segmented_data'
- Script for extracting event markers for segmenting trials is available in 'scripts'

2. **Independent Component Analysis (ICA)**:
   - ICA is applied to the raw data to identify and separate independent components.
   - The ICA components and their sources are plotted and saved for visual inspection.
   - Components representing artifacts are manually selected and excluded.
   - The excluded components are saved to a JSON file for reproducibility.

3. **Filtering**:
   - A notch filter at 60 Hz is applied to remove power line noise.
   - A bandpass filter with a lower frequency of 0.1 Hz and an upper frequency of 40.0 Hz is applied to remove low-frequency drifts and high-frequency noise.

4. **Re-referencing**:
   - The data is re-referenced using the 'VREF' channel.
   - Channels starting with 'E' are extracted.

5. **EOG Regression**:
   - The channel types of 'E126' and 'E127' are changed to 'eog' for EOG regression.
   - Common average reference is applied to the data.
   - Word annotation metadata is loaded from a TSV file.
   - Word epochs are created based on the word onsets obtained from the annotation metadata.
   - EOG regression is performed on the word epochs to remove EOG artifacts.
   - The regression coefficients are plotted as a topomap.
   - The cleaned evoked response is computed and saved.
  
  
