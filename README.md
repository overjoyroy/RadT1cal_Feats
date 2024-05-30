# RadT1cal_Feats

RadT1cal_Feats is a pipeline designed to preprocess T1-weighted (T1w) Structural MRIs for visualization and further analysis. This tool extracts radiomics features for each region of interest (ROI) based on a provided anatomical atlas. It is intended for command-line use.

## Overview

The preprocessing pipeline includes four primary steps:

1. **Reorientation** - Aligning the MRI images to a standard orientation.
2. **Skull Stripping** - Removing non-brain tissues from the MRI images.
3. **Bias Correction** - Correcting intensity non-uniformities in the MRI images.
4. **Registration** - Aligning an atlas to the patient space for standardized anatomical referencing.
5. **Radiomics** - Calculating and collecting first order radiomics features per ROI

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
3. **Radiomics Features**: Radiomics features per ROI

## Usage Instructions

### Running the Tool Directly

To use this tool directly, run the following command:
```
python3 t1_preproc.py -p [data_dir_path] -sid [subject-id] -o [output_path] -tem [template_path] -seg [segment_path]
```

### Using Docker (Recommended)

Using Docker is recommended to simplify the installation of necessary dependencies (including FSL, ANTs, and relevant Python libraries). There are two ways to use Docker: building and running the container locally, or using a prebuilt Docker image from Docker Hub.

**Option 1: Building and Running the Container Locally**
1. Clone the Git repository and navigate to the directory containing the Dockerfile.
2. Build the Docker container:
```
docker build -t my_RadT1cal_Feats_container .
```
3. Run the Docker container:
```
docker run -v [data_dir_path]:/data/my_data -v [output_path]:/data/output --rm -u $UID:$UID my_RadT1cal_Feats_container -p /data/my_data -sid [subject-id] -o [output_path] 
```

Explanation of Docker Flags:

```--rm```: Automatically removes the container after it exits. <br>
```-v```: Mounts your data and output directories into the Docker container.<br>
```-u``` $UID:$UID: Runs the Docker container as the same user as on the host machine to avoid file permission issues.

*Notes:
If the output directory is a subdirectory of the data directory (e.g., \[data_dir_path\]/derivatives), you only need to mount the data directory once and provide the output path relative to the mounted data directory (e.g., /data/my_data/derivatives). Output files will be accessible on the host machine from where the output path was mounted. Default anatomical templates and atlas (MNI152 and AALv3_CombinedThalami respectively) are included so the -tem and -seg flags are optional.*

**Option 2: Using a Prebuilt Docker Image**
1. Pull the prebuilt Docker image from Docker Hub:
```
docker pull jor115/RadT1cal_Feats
```
2. Run the Docker container using the pulled image:
```
docker run -v [data_dir_path]:/data/my_data -v [output_path]:/data/output --rm -u $UID:$UID my_RadT1cal_Feats_container -p /data/my_data -sid [subject-id] -o [output_path] 
```
*Note: The docker run command is identical to the one used for running a locally built container, but you do not need to download the source code or build the container locally.*

## Development Status

This tool is a work in progress. It is currently fully functional but is undergoing refinement to enhance user flexibility and input handling.

## Feedback and Contributions

If you have any concerns or suggestions regarding the code or its implementation, please contact the authors at pirc@chp.edu.

## Acknowledgement

The primary authors of this work are Joy Roy and Rafael Ceschin, both members of the Department of Biomedical Informatics, University of Pittsburgh School of Medicine and the Pediatric Imaging Research Center, Department of Radiology, Children's Hospital of Pittsburgh.
