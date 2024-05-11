### Preprocessing with ICA Before and After Filtering

This script applies Independent Component Analysis (ICA) twice to the EEG data, once before filtering
and once after filtering.

**Steps**

1. Loads the raw EEG data from a FIF file.
2. Applies the first round of ICA before filtering.
   - Fits the ICA model with a specified number of components.
   - Plots and saves the ICA component and source plots.
   - Selects components to exclude based on manual inspection.
   - Saves the excluded components to a JSON file.
   - Applies the ICA to the raw data.
3. Applies a notch filter at 60 Hz to remove power line noise.
4. Applies a bandpass filter with a lower frequency of 1.0 Hz and an upper frequency of 15.0 Hz.
5. Re-references the data using the 'VREF' channel.
6. Extracts only the channels starting with 'E'.
7. Applies common average reference.
8. Applies the second round of ICA after filtering.
   - Fits the ICA model with a specified number of components.
   - Plots and saves the ICA component and source plots.
   - Selects components to exclude based on manual inspection.
   - Saves the excluded components to a JSON file.
   - Applies the ICA to the raw data.
9. Loads word annotation metadata from a TSV file.
10. Creates word epochs based on the word onsets obtained from the annotation metadata.
11. Saves the word epochs to a file.
12. Averages the word epochs to obtain the evoked response.
13. Plots and saves the evoked response figure.

**Functions**

- save_ica_plots_and_json(ica, subject, segment, stimulus, round, base_path):
  - Saves the ICA component plots, source plots, and JSON files for the excluded components.
  - Parameters:
    - ica: The ICA object.
    - subject: Subject identifier.
    - segment: Segment identifier.
    - stimulus: Stimulus identifier.
    - round: Round of ICA application ('before' or 'after' filtering).
    - base_path: Base directory path for the project.

**Parameters**

- subject: Subject identifier.
- stimulus: Stimulus identifier.
- segment: Segment identifier.

**Paths**

-base_path: Base directory path for the project.
- word_path: Path to the word annotation TSV file.
- fif_path: Path to the raw EEG data FIF file.

**Output**

- ICA component and source plots for the first round of ICA (before filtering).
- JSON file containing the excluded components for the first round of ICA.
- ICA component and source plots for the second round of ICA (after filtering).
- JSON file containing the excluded components for the second round of ICA.
- Word epochs saved to a FIF file.
- Evoked response figure saved as a JPEG file.

**Dependencies**

- MNE: A Python package for processing and analyzing EEG data.
- NumPy: A library for numerical computing in Python.
- Pandas: A library for data manipulation and analysis.
- Matplotlib: A plotting library for creating static, animated, and interactive visualizations.
- SciPy: A library for scientific computing and statistical functions.
- JSON: A built-in Python module for working with JSON data.

**Notes** 

- The script assumes that the necessary dependencies are installed.
- The script uses specific file paths and directory structures, which may need to be adjusted based on the user's setup.
- The number of ICA components and the random seed for ICA can be modified if desired.
- The excluded components for each round of ICA are manually selected based on visual inspection of the ICA component and source plots.
- The script saves the excluded components to JSON