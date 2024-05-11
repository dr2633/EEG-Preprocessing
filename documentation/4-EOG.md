### EOG Regression and ICA in EEG Preprocessing
Derek Rosenzweig



This script applies Independent Component Analysis (ICA) and EOG regression to the EEG data
to reduce the impact of blink artifacts.

**Steps**

1. Loads the raw EEG data from a FIF file.
2. Applies ICA before filtering.
   - Fits the ICA model with a specified number of components.
   - Plots and saves the ICA component and source plots.
   - Selects components to exclude based on manual inspection.
   - Saves the excluded components to a JSON file.
   - Applies the ICA to the raw data.
3. Applies a notch filter at 60 Hz to remove power line noise.
4. Applies a bandpass filter with a lower frequency of 0.1 Hz and an upper frequency of 40.0 Hz.
5. Re-references the data using the 'VREF' channel.
6. Extracts only the channels starting with 'E'.
7. Changes the channel types of 'E126' and 'E127' to 'eog'.
8. Applies common average reference.
9. Loads word annotation metadata from a TSV file.
10. Creates word epochs based on the word onsets obtained from the annotation metadata.
11. Runs EOG regression on the word epochs.
    - Fits the EOG regression model to the word epochs.
    - Plots and saves the regression coefficients as a topomap.
    - Applies the EOG regression to the epochs.
    - Computes the evoked response on the corrected data.
    - Plots and saves the cleaned evoked response.
12. Returns the cleaned epochs and evoked response.

**Functions**

- run_ica_and_eog_regression(sub, stim, seg, base_path):
  - Runs ICA and EOG regression on the EEG data to reduce the impact of artifacts.
  - Parameters:
    - sub: Subject identifier.
    - stim: Stimulus identifier.
    - seg: Segment identifier.
    - base_path: Base directory path for the project.
  - Returns:
    - epochs_clean: Epochs after ICA and EOG regression.
    - evoked_clean: Evoked response after ICA and EOG regression.

**Parameters**

- sub: Subject identifier.
- stim: Stimulus identifier.
- seg: Segment identifier.
- base_path: Base directory path for the project.

**Paths**

- word_path: Path to the word annotation TSV file.
- phoneme_path: Path to the phoneme annotation TSV file.
- fif_path: Path to the raw EEG data FIF file.

**Output**

- ICA component and source plots saved in the 'vis/individual/ICA/{sub}' directory.
- JSON file containing the excluded components saved in the 'derivatives/individual/ica_excluded_components' directory.
- EOG regression coefficients topomap saved in the 'vis/individual/EOG/{sub}' directory.
- Cleaned evoked response plot saved in the 'vis/individual/EOG/{sub}' directory.

**Dependencies**

- MNE: A Python package for processing and analyzing EEG data.
- NumPy: A library for numerical computing in Python.
- Pandas: A library for data manipulation and analysis.
- Matplotlib: A plotting library for creating static, animated, and interactive visualizations.
- SciPy: A library for scientific computing and statistical functions.
- JSON: A built-in Python module for working with JSON data.

**Notes**

- The number of ICA components and the random seed for ICA can be modified if desired.
- The excluded components for ICA are manually selected based on visual inspection of the ICA component and source plots.
- The script saves the excluded components to a JSON file for reproducibility and future reference.
- The EOG regression model is fitted to the word epochs and applied to remove EOG artifacts.
- The cleaned epochs and evoked response are returned by the script for further analysis or processing.
