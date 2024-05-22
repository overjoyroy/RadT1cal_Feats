# T1w_Preprocess

This is a basic pipeline to preprocess T1-weighted (T1w) Structural MRIs for visualization and further processing. This tool is designed for command-line use.

## Overview

The preprocessing pipeline includes four primary steps:

1. **Reorientation** - Aligning the MRI images to a standard orientation.
2. **Skull Stripping** - Removing non-brain tissues from the MRI images.
3. **Bias Correction** - Correcting intensity non-uniformities in the MRI images.
4. **Registration** - Aligning an atlas to the patient space for standardized anatomical referencing.

## Features

- Written primarily in Python.
- Utilizes the Nipype library for workflow management.
- Employs tools such as FSL and ANTs for processing tasks.
- Compatible with the Brain Imaging Data Structure (BIDS) Standard.

## Inputs and Outputs

### Inputs
At minimum:
- **Parent Data Directory**: The directory containing the MRI data.
- **Subject ID**: Identifier for the subject/patient.
- **Output Directory**: Directory where the processed outputs will be saved.

### Outputs

1. **Preprocessed Patient T1 MRI**: The MRI after all preprocessing steps.
2. **Atlas in Patient Space**: An atlas registered to the patient’s anatomical space.

## Usage

To use this tool, run the command:
```
python3 t1_preproc.py -p [data_dir_path] -sid [subject-id] -o [output_path]
```

## Development Status

This tool is a work in progress. It is currently fully functional but is undergoing refinement to enhance user flexibility and input handling.

## Feedback and Contributions

If you have any concerns or suggestions regarding the code or its implementation, please contact the authors at pirc@chp.edu.


