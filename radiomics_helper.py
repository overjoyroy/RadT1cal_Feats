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

def getAllROIFeats(aal, brain, maxROI):    
    featVecs = {}
    maxROI = 2
    for i in range(0,maxROI+1):
        roi_mask = create_binaryROI_mask(aal, i, output_path = os.getcwd())
        if roi_mask == None:
            continue
            
        extractor = radiomics.featureextractor.RadiomicsFeatureExtractor()
        extractor.disableAllFeatures()
        extractor.enableFeatureClassByName('firstorder')
        # Alternative; only enable 'Mean' and 'Skewness' features in firstorder
        # extractor.enableFeaturesByName(firstorder=['Mean', 'Skewness'])
    
        featureVector = extractor.execute(brain, roi_mask)
        featVecs[i] = featureVector
        os.remove(roi_mask)

    return featVecs

def getAndStoreROIFeats(aal, brain, maxROI, outpath=None):    
    f = getAllROIFeats(aal, brain, maxROI)
    features = list(sorted(filter(lambda k: k.startswith("original_"), f[0])))
    roi_features_pd = pd.DataFrame(f).T[features]
    roi_features_pd = roi_features_pd.rename_axis('ROI').reset_index() #make the index a column indicating ROI

    if outpath == None:
        outpath = os.path.join(os.getcwd(), '{}_radiomicsFeatures.csv'.format(os.path.basename(aal)[:-7]))
    elif os.isDir(outpath):
        outpath = os.path.join(outpath, '{}_radiomicsFeatures.csv'.format(os.path.basename(aal)[:-7]))

    print('I will first output the rad features to: {}'.format(outpath))

    roi_features_pd.to_csv(outpath, index=False)

    return outpath

if __name__ == "__main__":
    print('')