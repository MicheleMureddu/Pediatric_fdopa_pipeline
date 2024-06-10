import matplotlib 
matplotlib.rcParams['figure.facecolor'] = '1.'
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import pandas as pd
import re
import os
import ants
import argparse
import csv
import statistics
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import label
from scipy.ndimage import center_of_mass
from scipy.ndimage import gaussian_filter
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from scipy.stats import linregress
from argparse import ArgumentParser
from pathlib import Path
from skimage.morphology import dilation, erosion
from sys import argv
from glob import glob
import seaborn as sns
from ref_tumor_seg import ref_seg, binary_mask

def get_stats_for_labels(vol, atlas, labels):
    total=0
    n=0
    maximum=np.min(vol)

    for l in labels :
        idx = atlas == l
        
        label_n = np.sum(idx)
        if label_n > 0 :
            pet_values_in_label = vol[ idx ]
            total += np.sum(vol[idx])
            max_in_label = np.max(vol[idx])
            median_in_label = statistics.median(vol[idx])
            l_std = np.std(vol[idx])
            maximum = max_in_label if max_in_label > maximum else maximum

            n += label_n
        else :
            print(f'Error: label {l} not found in atlas volume where it was expected. Skipping')
            exit(1)

    average = total / n

    return average, maximum, l_std, median_in_label

def get_new_atlas(atlas_vol, roi_labels):
    
    # Roi labels for creating a unique region for the striatum
    index_caudato = atlas_vol == roi_labels[0]
    index_putamen = atlas_vol == roi_labels[1]
    index_pallidum = atlas_vol == roi_labels[2]
    
    # Striatum is defined by the OR operaton between the masks of Pallidum, Caudato and Putamen
    index_striatum = index_caudato | index_putamen | index_pallidum

    striatum_label = np.max(atlas_vol) + 1
    striatum_atlas_vol = np.copy(atlas_vol)
    striatum_atlas_vol[index_striatum] = striatum_label
    
    return striatum_atlas_vol, striatum_label

def get_tumor_lab(atlas_vol,suvr_max):
    
    tumor_label = np.max(atlas_vol) + 1
    tumor_atlas_vol = np.copy(atlas_vol)
    # Condition for creating tumor VOI
    tumor_atlas_vol[suvr_max > 1] = tumor_label
    
    return tumor_atlas_vol, tumor_label

def get_H_Uptake_lab(atlas_vol, mT, sT, vol, tum_atlas,tumor_label):

    HU_label = np.max(atlas_vol) + 1
    new_atlas_vol = np.copy(atlas_vol)
    # Finding the tumor region with higher uptake
    H_mask = (tum_atlas == tumor_label) & (vol > mT + 2*sT) 
    new_atlas_vol[H_mask] = HU_label
    
    return new_atlas_vol, HU_label

def get_M_Uptake_lab(atlas_vol, mT, sT, vol, tum_atlas,tumor_label):

    MU_label = np.max(atlas_vol) + 2
    new_atlas_vol = np.copy(atlas_vol)
    # Finding the tumor region with medium uptake
    M_mask = (tum_atlas == tumor_label) & (vol > mT)  & (vol < mT +2*sT)
    new_atlas_vol[M_mask] = MU_label
    
    
    return new_atlas_vol, MU_label

def get_L_Uptake_lab(atlas_vol, mT, sT, vol, tum_atlas,tumor_label):

    LU_label = np.max(atlas_vol) + 3
    new_atlas_vol = np.copy(atlas_vol)
    # Finding the tumor region with lower uptake
    L_mask = (tum_atlas == tumor_label) & (vol > mT -2*sT) & (vol < mT)
    new_atlas_vol[L_mask] = LU_label
    
    return new_atlas_vol, LU_label

def control_ratio(tumor_MRI_vol, tumor_max, tumor_max_manuale, ref_max, roi_max, tumor_atlas_vol, tumor_labels):
    
    tumor_atlas = np.zeros_like(tumor_atlas_vol)
    tumor_label = 0
    
    # Selecting the automatic or semi-automatic approach for tumor analysis
    if (tumor_max/ref_max >= 1) & (tumor_max/roi_max >= 1) & (tumor_max_manuale/ref_max >= 1) & (tumor_max_manuale/roi_max >= 1):
        tumor_atlas = tumor_atlas_vol
        tumor_label = tumor_labels
    else:
        tumor_label = 1
        tumor_atlas = tumor_MRI_vol
        
    return tumor_atlas, tumor_label

def variable_def(subj):

    ### Input data ###    
    pet_hd = nib.load(subj.pet)
    pet_3d = pet_hd.get_fdata()
    
    atlas_hd = nib.load(subj.atlas_space_pet)
    atlas_vol = np.rint(atlas_hd.get_fdata()).astype(int)

    brain_mask,_ = binary_mask(subj)
    pet_3d[(brain_mask == 0)] = 0
    
    # Static parameters for tumor definition
    _, ref_max, _, _ = get_stats_for_labels(pet_3d, atlas_vol, [subj.ref_labels])
    
    # Striatum definition
    s_atlas_vol, striatum_label = get_new_atlas(atlas_vol, subj.roi_labels)
    subj.striatum_atlas = s_atlas_vol
    subj.striatum_label = striatum_label
    _, roi_max, _, _ = get_stats_for_labels(pet_3d, subj.striatum_atlas, [subj.striatum_label])
    
    # SUVr volume
    suvr_max = pet_3d / ref_max
    subj.suvr_m = suvr_max
    tumor_atlas_vol, tumor_labels = get_tumor_lab(subj.striatum_atlas, subj.suvr_m)
    
    ### just in case tumors is located in both hemispheres ###
    if(not(np.any(tumor_atlas_vol == tumor_labels))):
        subj.ref_labels = 47 
        _, ref_max, _, _ = get_stats_for_labels(pet_3d, atlas_vol, [subj.ref_labels])
        suvr_max = pet_3d / ref_max
        subj.suvr_m = suvr_max
        tumor_atlas_vol, tumor_labels = get_tumor_lab(subj.striatum_atlas, subj.suvr_m)

    # tumor volume from MRI
    tumor_MRI_hd = nib.load(subj.volume_MRI)
    tumor_MRI_vol = np.rint(tumor_MRI_hd.get_fdata()).astype(int)

    # Comparing of tumor parameters between PET and FLAIR tumor
    _, tumor_max, _, _ = get_stats_for_labels(pet_3d, tumor_atlas_vol, [tumor_labels])
    _, tumor_max_manuale, _, _ = get_stats_for_labels(pet_3d, tumor_MRI_vol, [1])
    subj.tumor_atlas, subj.tumor_label = control_ratio(tumor_MRI_vol,tumor_max,tumor_max_manuale, ref_max, roi_max, tumor_atlas_vol, tumor_labels)
    nib.Nifti1Image(subj.tumor_atlas, nib.load(subj.atlas_space_pet).affine, dtype = np.int64).to_filename(subj.prefix+'original_tum_volume.nii.gz')
    
    if (subj.tumor_label != 1):
        ### elimination of controlateral striatum ####
        subj.tumor_atlas[subj.striatum_atlas == subj.striatum_label] = 0
            
        ### elimination of omolateral striatum ###
        if subj.roi_labels[0] == 11:
            omolateral_striatum = (subj.striatum_atlas == 50) | (subj.striatum_atlas == 51) | (subj.striatum_atlas == 52) 
            if ((tumor_MRI_vol == 1) & (omolateral_striatum)).any() == False:
                subj.tumor_atlas[omolateral_striatum] = 0
        else:
            omolateral_striatum = (subj.striatum_atlas == 11) | (subj.striatum_atlas == 12) | (subj.striatum_atlas == 13) 
            if ((tumor_MRI_vol == 1) & (omolateral_striatum)).any() == False:
                subj.tumor_atlas[omolateral_striatum] = 0
                
        ### refinement of tumor segmentation ###
        subj.tumor_atlas = ref_seg(subj)

    return subj.tumor_atlas, subj.tumor_label, subj.striatum_atlas, subj.striatum_label, subj.suvr_m

def tumor_striatum_analysis(subject, roi_labels, ref_labels):
    
    print('\tTumor Striatum Analysis')
    print(f'\t\tPET:\t{subject.pet}')
    print(f'\t\tAtlas:\t{subject.atlas_space_pet}')
    print(f'\t\tRoi Labels:\t{roi_labels}')
    print(f'\t\tReference Labels:\t{ref_labels}')
    
    subject.pet_suvr = subject.prefix+'suvr_max.nii.gz'
    subject.suvr_csv = subject.prefix+'suvr_max.json'

    if not os.path.exists(subject.pet_suvr) or not os.path.exists(subject.suvr_csv) or subject.clobber :
        
        pet_hd = nib.load(subject.pet)
        pet_vol = pet_hd.get_fdata()

        atlas_hd = nib.load(subject.atlas_space_pet)
        atlas_vol = np.rint(atlas_hd.get_fdata()).astype(int)
      
        ### Static parameters ###
        roi_avg, roi_max, _, _ = get_stats_for_labels(pet_vol, subject.striatum_atlas, [subject.striatum_label])
        ref_avg, ref_max, _, _ = get_stats_for_labels(pet_vol, atlas_vol, [ref_labels])
        tumor_avg, tumor_max, _, _ = get_stats_for_labels(pet_vol,  subject.tumor_atlas, [subject.tumor_label])

        ts_ratio = np.round(tumor_max / roi_max,3)
        tn_ratio = np.round(tumor_max / ref_max,3)

        nib.Nifti1Image(subject.suvr_m, nib.load(subject.pet).affine).to_filename(subject.pet_suvr)
        nib.Nifti1Image(subject.tumor_atlas, atlas_hd.affine , header = atlas_hd.header).to_filename(subject.prefix+'tumor_atlas.nii.gz')
        
        suvr_dict = {'sub':[subject.sub],'tumor_label':[subject.tumor_label],'tumor_max':[tumor_max],'tumor_avg':[tumor_avg], 'striatum_label':[subject.striatum_label], 'striatum_max':[roi_max], 'straitum_avg':[roi_avg], 'ts_ratio':[ts_ratio], 'reference_label':[ref_labels], 'reference_max':[ref_max], 'reference_avg':[ref_avg], 'tn_ratio': [tn_ratio]}
        subject.suvr_df = pd.DataFrame(suvr_dict)
        subject.suvr_df.to_csv(subject.suvr_csv, index=False)

        print(f'\t\tWriting SUVR_max volume to {subject.pet_suvr}')
        
    else :
        subject.suvr_df = pd.read_csv(subject.suvr_csv)

    return subject




