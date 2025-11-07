import numpy as np
import nibabel as nib
import re
import os
os.environ["OMP_NUM_THREADS"] = "15"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "15" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "15"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "15" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "15"  # export NUMEXPR_NUM_THREADS=1
import ants
import argparse
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
from sys import argv
from glob import glob
import subprocess
from subject import Subject
from analysis import tumor_striatum_analysis
from utils import get_file, get_dynamic_parameters
from NET_module import postprocess
from NET_module import main

def find_subject_ids(data_dir):
    get_id = lambda fn: re.sub('sub-','',os.path.basename(fn).split('_')[0])
    pet_images_list = Path(data_dir).rglob('*_ses-01_pet.nii.gz')
    return [ get_id(fn) for fn in pet_images_list ]

def get_parser():
    parser = ArgumentParser(usage="useage: ")
    parser.add_argument("-i",dest="data_dir", default='pediatric/', help="Path for input file directory")
    parser.add_argument("-o",dest="out_dir", default='output/', help="Path for output file directory")
    parser.add_argument("-s",dest="stx_fn", default='atlas/mni_icbm152_t1_tal_nlin_asym_09c.nii.gz', help="Path for stereotaxic template file")
    parser.add_argument("-a",dest="atlas_fn", default='atlas/dka_atlas_eroded.nii.gz', help="Path for stereotaxic label file")
    parser.add_argument("-b",dest="atlas_brats", default='atlas/T1.nii.gz', help="Path for stereotaxic label file")
    parser.add_argument("-vol_MRI", dest="flair_tumor", default='Lesions_TL/final_preds_fold3/', help="Path for MRI volume")
    parser.add_argument("-DL", dest="dl_flag", action="store_true", help="U-Net segmentation and Postprocess")
    return parser

def run_unet_inference_unix(data_dir):
    """
    Running U-NET model for automatic segmentation. Running Post-Process for getting lesions in NIFTI format.
    """
    print('\n Run U-NET model...\n ________________________\n')

    # default parameters for U-NET model
    args = argparse.Namespace(
        exec_mode="predict",
        data="./brats_flair/val_3d/test",
        results="./results",
        not_val=None,
        config=None,
        logname="logs.json",
        task="train",
        gpus=1,
        nodes=1,
        learning_rate=0.0008,
        gradient_clip_val=0,
        negative_slope=0.01,
        tta=True,
        tb_logs=False,
        deep_supervision=False,
        amp=True,
        focal=False,
        save_ckpt=False,
        nfolds=5,
        seed=None,
        ckpt_path="./results/checkpoints/epoch=146-dice=88.05.ckpt",
        ckpt_store_dir="./results",
        resume_training=False,
        fold=0,
        patience=100,
        batch_size=2,
        val_batch_size=4,
        momentum=0.99,
        weight_decay=0.0001,
        save_preds=True,
        num_workers=8,
        epochs=1000,
        warmup=5,
        norm="instance",
        depth=6,
        min_fmap=2,
        deep_supr_num=2,
        res_block=False,
        filters=[64, 96, 128, 192, 256, 384, 512],
        oversampling=0.4,
        overlap=0.5,
        scheduler=False,
        freeze=-1
        )
    
    preds_dir = main.run_unet(args)
    
    print("Segmentation completed")
    print('\n Run U-NET model...\n ________________________\n')

    # Postprocess.py to obtain predictions in NIfTI
    preds_dirs = [preds_dir]
    images_dir = Path("./brats_flair/val/images")
    output_dir = Path("./Lesions_TL/final_preds_fold3")

    postprocess.run_postprocess(predictions_dirs=preds_dirs,
                            images_dir=images_dir,
                            output_dir=output_dir,
                            output_type="postop")

    print("Completed PostProcessing. Running Pediatric FDOPA Pipeline...")

def lesions_exist(path):
    """
    Check if in 'Lesions_TL/final_preds_fold3/' there are NIFTI files
    """
    lesion_dir = Path(path)
    if not lesion_dir.exists():
        return False
    nii_files = list(lesion_dir.glob("*.nii")) + list(lesion_dir.glob("*.nii.gz"))
    return len(nii_files) > 0
    
def check_or_segment(opts):

    """
    If Lesions_TL/final_preds3 is empy run U-NET, otherwise run pipepline
    """
    
    if opts.dl_flag:

        lesion_path = Path(opts.flair_tumor)

        if lesions_exist(lesion_path):
            print(f"Found lesions in {lesion_path}. No segmentation needed. Running FDOPA Pipeline...")
        else:
            print(f'\n No lesions in {lesion_path}. Running UNet...\n')
            run_unet_inference_unix("/mnt/nas_biolab/data/michele/DL/brats_flair_test_pipeline/val_3d/test")
            lesion_path = Path("Lesions_TL/final_preds_fold3/")
        
        opts.flair_tumor = str(lesion_path)+'/'
    else:
        opts.flair_tumor = str("/home/michele/pediatric_fdopa_Transfer_Learning/tumor_MRI")+'/'
    
    return opts

if __name__ == '__main__' :

    opts = get_parser().parse_args()
    print('\n Pediatric FDOPA Pipeline\n ________________________\n')

    opts = check_or_segment(opts)

    print('\tOptions')
    print('\t\tData directory:', opts.data_dir)
    print('\t\tOutput directory:', opts.out_dir)
    print('\t\tTemplate:',opts.stx_fn)
    print('\t\tAtlas:', opts.atlas_fn)
    print('\t\tAtlas Brats:', opts.atlas_brats)
    print('\t\tTumor:', opts.flair_tumor)
    print()

    tumor_striatum_csv = opts.out_dir+os.sep+'tumor_striatum_ibrido.csv'
    print('\tOutputs')
    print('\t\tTumor striatum ratio csv:'+tumor_striatum_csv)
    print()

    dynamic_parameters = opts.out_dir+os.sep+'Dynamic_Parameters_ibrido.csv'
    print('\tOutputs')
    print('\t\tDynamic Parameters csv:'+dynamic_parameters)
    print()

    H_tumor_percentage = opts.out_dir+os.sep+'H_tumor_percentage_ibrido.csv'
    print('\tOutputs')
    print('\t\tH_tumor_percentage csv:'+H_tumor_percentage)
    print()

    subject_id_list = find_subject_ids(opts.data_dir)
    print('\tRuntime parameters:')
    print(f'\t\tSubject IDs: {subject_id_list}')
    print()
        
    # Create a list of instances of the Subject class. 
    subject_list = [ Subject(opts.data_dir, opts.out_dir, sub, opts.stx_fn, opts.atlas_fn, opts.atlas_brats, opts.flair_tumor) for sub in subject_id_list ]
    
    # Do initial processing for each subject (e.g., alignment to MRI and stereotaxic atlas)
    [ subj.process() for subj in subject_list ] 

    # Do analysis to find maximum tumor and striatum PET values
    subject_list = [ tumor_striatum_analysis(subj, subj.roi_labels, subj.ref_labels) for subj in subject_list ]
    tumor_striatum_df = pd.concat([ subject.suvr_df for subject in subject_list ])
    tumor_striatum_df.to_csv(tumor_striatum_csv, index=False)

    # Initialize lists to store data
    dy_param_data = []
    ratio_data = []

    for subject in subject_list:
    # Checking if PET file exists
        if (Path(subject.data_dir+'/sub-'+subject.sub+'/ses-02').is_dir()):
            # Extract dynamic parameters
            dy_param_data.append(subject.dy_df)
            if (subject.bool_flag):
                ratio_data.append({'subject': subject.sub, 'percentage': subject.tum_percentage})
        
    if dy_param_data:
        dy_param_df = pd.concat(dy_param_data)
    else:
        dy_param_df = pd.DataFrame()  

    
    if ratio_data:
        df_ratio = pd.DataFrame(ratio_data)
    else:
        df_ratio = pd.DataFrame() 

    # Write dataframes to CSV files
    dy_param_df.to_csv(dynamic_parameters, index=False)
    df_ratio.to_csv(H_tumor_percentage, index=False)

    print()




