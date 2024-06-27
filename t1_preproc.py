#bin/python3
import argparse
import nibabel as nib
import nipy as nipy
import nipype.interfaces.io as nio 
import nipype.interfaces.fsl as fsl 
import nipype.interfaces.ants as ants 
import nipype.interfaces.utility as util 
import nipype.pipeline.engine as pe
import numpy as np
import os, sys
import time
import radiomics_helper # Custom module

DATATYPE_SUBJECT_DIR = 'anat'
DATATYPE_FILE_SUFFIX = 'T1w'

def makeParser():
    parser = argparse.ArgumentParser(
                        prog='T1_Preproc', 
                        usage='This program preprocesses T1 MRIs for later use with an MSN development pipeline.',
                        epilog='BUG REPORTING: Report bugs to pirc@chp.edu or more directly to Joy Roy at the Childrens Hospital of Pittsburgh.'
        )
    parser.add_argument('--version', action='version', version='%(prog)s 0.1')
    parser.add_argument('-p','--parentDir', nargs=1, required=True,
                        help='Path to the parent data directory. BIDS compatible datasets are encouraged.')
    parser.add_argument('-sid','--subject_id', nargs=1, required=True,
                        help='Subject ID used to indicate which patient to preprocess')
    parser.add_argument('-spath','--subject_t1_path', nargs=1, required=False,
                        help='Path to a subjects T1 scan. This is not necessary if subject ID is provided as the T1 will be automatically found using the T1w.nii.gz extension')
    parser.add_argument('-ses_id','--session_id', nargs=1, required=False,
                        help='Session ID used to indicate which session to look for the patient to preprocess')
    parser.add_argument('-tem','--template', nargs=1, required=False,
                        help='Template to be used to register into patient space. Default is MNI152lin_T1_2mm_brain.nii.gz')
    parser.add_argument('-seg','--segment', nargs=1, required=False,
                        help='Atlas to be used to identify brain regions in patient space. This is used in conjunction with the template. Please ensure that the atlas is in the same space as the template. Default is the AALv3 template.')
    parser.add_argument('-o','--outDir', nargs=1, required=True,
                        help='Path to the \'derivatives\' folder or chosen out folder.')
    parser.add_argument('--testmode', required=False, action='store_true',
                        help='Activates TEST_MODE to make pipeline finish faster for quicker debugging')

    return parser 


# This was developed instead of using the default parameter in the argparser
# bc argparser only returns a list or None and you can't do None[0]. 
# Not all variables need a default but need to be inspected whether they are None
def vetArgNone(variable, default):
    if variable==None:
        return default
    else:
        return variable[0]

def makeOutDir(outDirName, args, enforceBIDS=True):
    # outDir = ''
    # if os.path.basename(args.outDir[0]) == 'derivatives':
    #     outDir = os.path.join(args.outDir[0], outDirName, args.subject_id[0])
    # elif args.outDir[0] == args.parentDir[0]:
    #     print("Your outdir is the same as your parent dir!")
    #     print("Making a derivatives folder for you...")
    #     outDir = os.path.join(args.outDir[0], 'derivatives', outDirName, args.subject_id[0])
    # elif os.path.basename(args.outDir[0]) == args.subject_id[0]:
    #     print('The given out directory seems to be at a patient level rather than parent level')
    #     print('It is hard to determine if your out directory is BIDS compliant')
    # elif 'derivatives' in args.outDir[0]:
    #     outDir = os.path.join(args.outDir[0], outDirName, args.subject_id[0])

    outDir = os.path.join(args.outDir[0], outDirName, args.subject_id[0])

    if not os.path.exists(outDir):
        os.makedirs(outDir, exist_ok=True)

    return outDir

# Nipype nodes built on python-functions need to reimport libraries seperately
def getMaxROI(atlas_path):
    import nibabel as nib
    import numpy as np
    
    img = nib.load(atlas_path)
    data = img.get_fdata()
    return round(np.max(data))

# Nipype nodes built on python-functions need to reimport libraries seperately
def CalcROIFeatures(patient_atlas_path, brain_path, maxROI, outDir_path=None):
    import radiomics_helper
    return radiomics_helper.getAndStoreROIFeats(patient_atlas_path, brain_path, maxROI, outDir_path)

def buildWorkflow(patient_T1_path, template_path, segment_path, outDir, subjectID, test=False):

    ########## PREPWORK
    preproc = pe.Workflow(name='preproc')

    # Sole purpose is to store the original T1w image
    input_node = pe.Node(interface=util.IdentityInterface(fields=['T1w']),name='input')
    input_node.inputs.T1w = patient_T1_path

    # Sole purpose is to store the MNI152 Template
    template_feed = pe.Node(interface=util.IdentityInterface(fields=['template']), name='template_MNI')
    template_feed.inputs.template = template_path

    # Sole purpose is to store the AAL template
    segment_feed = pe.Node(interface=util.IdentityInterface(fields=['segment']), name='segment_AAL')
    segment_feed.inputs.segment = segment_path

    # All files should be placed in a directory with the patient's ID
    datasink = pe.Node(nio.DataSink(parameterization=False), name='sinker')
    datasink.inputs.base_directory = outDir
    ########## END PREPROCESS T1W

    ########## PREPROCESS T1W
    # Reorient to make comparing visually images easier
    reorient2std_node = pe.Node(interface=fsl.Reorient2Std(), name='reorient2std')
    preproc.connect(input_node, 'T1w', reorient2std_node, 'in_file')
    preproc.connect(reorient2std_node, 'out_file', datasink, '{}.@reorient'.format(DATATYPE_SUBJECT_DIR))

    # Brain Extraction, 
    brain_extract = pe.Node(interface=fsl.BET(frac=0.50, mask=True, robust=True), name='bet')
    preproc.connect(reorient2std_node, 'out_file', brain_extract, 'in_file')
    preproc.connect(brain_extract, 'out_file', datasink, '{}.@brain'.format(DATATYPE_SUBJECT_DIR))

    # FSL Fast used for bias field correction
    fast_bias_extract = pe.Node(interface=fsl.FAST(output_biascorrected=True), name='fast')
    preproc.connect(brain_extract, 'out_file', fast_bias_extract, 'in_files')
    preproc.connect(fast_bias_extract, 'restored_image', datasink, '{}.@nobias'.format(DATATYPE_SUBJECT_DIR))

    # ants for both linear and nonlinear registration
    antsReg = pe.Node(interface=ants.Registration(), name='antsRegistration')
    antsReg.inputs.transforms = ['Affine', 'SyN']
    antsReg.inputs.transform_parameters = [(2.0,), (0.25, 3.0, 0.0)]
    antsReg.inputs.number_of_iterations = [[1500, 200], [100, 50, 30]]
    if test==True:
        antsReg.inputs.number_of_iterations = [[5, 5], [5, 5, 5]]
    antsReg.inputs.dimension = 3
    antsReg.inputs.write_composite_transform = False
    antsReg.inputs.collapse_output_transforms = False
    antsReg.inputs.initialize_transforms_per_stage = False
    antsReg.inputs.metric = ['Mattes']*2
    antsReg.inputs.metric_weight = [1]*2 # Default (value ignored currently by ANTs)
    antsReg.inputs.radius_or_number_of_bins = [32]*2
    antsReg.inputs.sampling_strategy = ['Random', None]
    antsReg.inputs.sampling_percentage = [0.05, None]
    antsReg.inputs.convergence_threshold = [1.e-8, 1.e-9]
    antsReg.inputs.convergence_window_size = [20]*2
    antsReg.inputs.smoothing_sigmas = [[1,0], [2,1,0]]
    antsReg.inputs.sigma_units = ['vox'] * 2
    antsReg.inputs.shrink_factors = [[2,1], [3,2,1]]
    antsReg.inputs.use_histogram_matching = [True, True] # This is the default
    antsReg.inputs.output_warped_image = 'output_warped_image.nii.gz'

    preproc.connect(template_feed, 'template', antsReg, 'moving_image')
    preproc.connect(fast_bias_extract, 'restored_image', antsReg, 'fixed_image')
    preproc.connect(antsReg, 'warped_image', datasink, '{}.@warpedTemplate'.format(DATATYPE_SUBJECT_DIR))

    antsAppTrfm = pe.Node(interface=ants.ApplyTransforms(), name='antsApplyTransform')
    antsAppTrfm.inputs.dimension = 3
    antsAppTrfm.inputs.interpolation = 'NearestNeighbor'
    antsAppTrfm.inputs.default_value = 0

    preproc.connect(segment_feed, 'segment', antsAppTrfm, 'input_image')
    preproc.connect(fast_bias_extract, 'restored_image', antsAppTrfm, 'reference_image')
    preproc.connect(antsReg, 'reverse_forward_transforms', antsAppTrfm, 'transforms')
    preproc.connect(antsReg, 'reverse_forward_invert_flags', antsAppTrfm, 'invert_transform_flags')
    preproc.connect(antsAppTrfm, 'output_image', datasink, '{}.@warpedAtlas'.format(DATATYPE_SUBJECT_DIR))
    ########## END PREPROCESS T1W


    ########## GET ROI FEATURES
    GetMaxROI_node = pe.Node(interface=util.Function(input_names=['atlas_path'], output_names=['max_roi'], function=getMaxROI), name='GetMaxROI')
    preproc.connect(segment_feed, 'segment', GetMaxROI_node, 'atlas_path')

    GetROIF_node = pe.Node(interface=util.Function(input_names=['patient_atlas_path', 'brain_path', 'maxROI', 'outDir_path'], output_names=['volOutpath', 'radOutpath'], function=CalcROIFeatures), name='CalculateROIFeatures')
    preproc.connect(antsAppTrfm, 'output_image', GetROIF_node, 'patient_atlas_path')
    preproc.connect(fast_bias_extract, 'restored_image', GetROIF_node, 'brain_path')
    preproc.connect(GetMaxROI_node, 'max_roi', GetROIF_node, 'maxROI')
    preproc.connect(GetROIF_node, 'radOutpath', datasink, '{}.@radiomicsFeatures'.format(DATATYPE_SUBJECT_DIR))
    preproc.connect(GetROIF_node, 'volOutpath', datasink, '{}.@volume'.format(DATATYPE_SUBJECT_DIR))
    ########## END GET ROI FEATURES


    return preproc


def main():

    ################################################################################
    ### PREPWORK
    ################################################################################
    parser = makeParser()
    args   = parser.parse_args()

    outDir        = ''
    outDirName    = 'RadT1cal_Features'
    session       = vetArgNone(args.session_id, None)
    template_path = vetArgNone(args.template, '/app/Template/MNI152lin_T1_2mm_brain.nii.gz') #path in docker container
    segment_path  = vetArgNone(args.segment, '/app/Template/AAL3v1_CombinedThalami.nii.gz') #path in docker container
    enforceBIDS   = True
    outDir        = makeOutDir(outDirName, args, enforceBIDS)
    TEST_MODE     = args.testmode
    if TEST_MODE:
        print("!!YOU ARE USING TEST MODE!!")

    for i in os.listdir(args.parentDir[0]):
        if i[:3] == 'ses':
            if session == None:
                raise Exception("Your data is sorted into sessions but you did not indicate a session to process. Please provide the Session.")

    if session != None:
        patient_T1_dir = os.path.join(args.parentDir[0], session, args.subject_id[0], DATATYPE_SUBJECT_DIR)
    else:
        patient_T1_dir = os.path.join(args.parentDir[0], args.subject_id[0], DATATYPE_SUBJECT_DIR)

    ## The following behavior only takes the first T1 seen in the directory. 
    ## The code could be expanded to account for multiple runs
    for i in os.listdir(patient_T1_dir):
        if i[-10:] =='T1w.nii.gz':
            patient_T1_path = os.path.join(patient_T1_dir, i)


    ################################################################################
    ### PREPROCESS T1w MRI + GET RADIOMICS FEATURES
    ################################################################################

    preproc = buildWorkflow(patient_T1_path, template_path, segment_path, outDir, args.subject_id[0], test=TEST_MODE)
    tic     = time.time()
    preproc.run()
    toc     = time.time()
    print('\nElapsed Time to Preprocess: {}s\n'.format(tic-toc))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\tReceived Keyboard Interrupt, ending program.\n")
        sys.exit(2)