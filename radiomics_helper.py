import nibabel as nib
import numpy as np
import os
import pandas as pd
import radiomics
from radiomics import featureextractor

def getMaxROI(atlas):
    img = nib.load(atlas)
    data = img.get_fdata()
    return round(np.max(data))

def create_binaryROI_mask(nifti_path, roi_value, output_path=None):
    img = nib.load(nifti_path)
    data = img.get_fdata()
    binary_mask = np.where(data == roi_value, 1, 0)
    roi_present = np.any(binary_mask != 0)
    
    if roi_present: # Save the binary mask as a new NIfTI file
        new_img = nib.Nifti1Image(binary_mask, img.affine, img.header)
        if output_path == None:
            output_name = '{}_roi{}.nii.gz'.format(os.path.basename(nifti_path)[:-7], roi_value)
            output_path = os.path.join(os.getcwd(), output_name)
        elif os.path.isdir(output_path):
            output_name = '{}_roi{}.nii.gz'.format(os.path.basename(nifti_path)[:-7], roi_value)
            output_path = os.path.join(output_path, output_name)
        nib.save(new_img, output_path)
        print(f"Binary mask saved to {output_path}")
        return output_path
    else:
        return None

def getROIVolume(brain_path, roi_mask_path):
    import nibabel as nib
    import numpy as np

    # Load the T1-weighted MRI and ROI mask (assuming both are NIfTI files)
    t1_img = nib.load(brain_path)
    roi_mask = nib.load(roi_mask_path)

    # Extract the data arrays
    t1_data = t1_img.get_fdata()
    roi_data = roi_mask.get_fdata()

    # Ensure the ROI mask is binary (1 for ROI, 0 for background)
    roi_data = roi_data > 0

    # Mask
    t1_roi_data = t1_data * roi_data

    # Calculate the voxel volume (assuming the affine matrix has the voxel dimensions)
    voxel_dims = t1_img.header.get_zooms()
    voxel_volume = np.prod(voxel_dims)

    # Count the number of voxels in the ROI
    num_voxels_in_roi = np.count_nonzero(t1_roi_data)

    # Calculate the total volume of the ROI
    roi_volume = num_voxels_in_roi * voxel_volume

    return roi_volume

def getAllROIFeats(atlas, brain, maxROI):    
    featVecs = {}
    volVecs  = {}
    for i in range(1, maxROI+1): # start at atlas val 1 because 0 is null
        roi_mask = create_binaryROI_mask(atlas, i, output_path = os.getcwd())
        if roi_mask == None:
            continue

        # Volume
        roi_vol = getROIVolume(brain, roi_mask)
        volVecs[i] = roi_vol
        
        # Radiomics
        extractor = radiomics.featureextractor.RadiomicsFeatureExtractor()
        extractor.enableAllFeatures()
        featureVector = extractor.execute(brain, roi_mask)
        featVecs[i] = featureVector

        # Clean-up
        os.remove(roi_mask)

    return volVecs, featVecs 

def saveOutput(data, atlas, suffix, outpath=None):
    if outpath == None:
        outpath = os.path.join(os.getcwd(), '{}_{}.csv'.format(os.path.basename(atlas)[:-7], suffix))
    elif os.path.isdir(outpath):
        outpath = os.path.join(outpath, '{}_{}.csv'.format(os.path.basename(atlas)[:-7], suffix))
    data.to_csv(outpath, index=False)
    return outpath 

def getAndStoreROIFeats(atlas, brain, maxROI, outpath=None):    
    v, f = getAllROIFeats(atlas, brain, maxROI)

    ### Volume
    roi_volumes_pd = pd.DataFrame(v.items(), columns=[['ROI', "Volume_mm3"]])
    volOutpath = saveOutput(roi_volumes_pd, atlas, 'volumes')
    
    ### Radiomics
    features = list(sorted(filter(lambda k: k.startswith("original_"), f[1])))
    roi_features_pd = pd.DataFrame(f).T[features]
    roi_features_pd = roi_features_pd.rename_axis('ROI').reset_index() #make the index a column indicating ROI
    radOutpath = saveOutput(roi_features_pd, atlas, 'radiomicsFeatures')

    return volOutpath, radOutpath

if __name__ == "__main__":
    print('')