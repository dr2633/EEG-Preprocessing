### Preprocessing Script with ICA 

Derek Rosenzweig

**Steps**

1. Loads the raw EEG data from a FIF file.
2. Applies ICA.
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
8. Loads word annotation metadata from a TSV file.
9. Creates word epochs based on the word onsets obtained from the annotation metadata.
10. Saves the word epochs to a file.
11. Averages the word epochs to obtain the evoked response.
12. Plots and saves the evoked response figure.

**Parameters**
- sub: Subject identifier.
- stim: Stimulus identifier.
- seg: Segment identifier.

**Paths**
- base_path: Base directory path for the project.
- word_path: Path to the word annotation TSV file.
- fif_path: Path to the raw EEG data FIF file.

**Output** 
- ICA component and source plots.
- JSON file containing the excluded components.
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
- The number of ICA components and the random seed for ICA can be modified if desired.
- The excluded components are manually selected based on visual inspection of the ICA component and source plots.
- The script saves the excluded components to a JSON file, allowing for reproducibility and future reference.
