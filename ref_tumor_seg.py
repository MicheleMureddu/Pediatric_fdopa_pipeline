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

def binary_mask(subject):
    '''
    This function cretates a binary mask of FLAIR MRI skull stripped. This mask is used to eliminate skull from [18F]F-DOPA PET. 
    '''
    brain_mask_hd = nib.load(subject.brain)
    brain_mask = brain_mask_hd.get_fdata()
    brain_mask[brain_mask != 0] = 1
    kernel = np.ones([3,3])
    brain_mask_eroded = np.zeros_like(brain_mask)

    for i in range(brain_mask.shape[0]):
        brain_mask_eroded[i,:,:] = erosion(brain_mask[i,:,:], kernel)
    return brain_mask_eroded, brain_mask

def create_tumor_label_volume(atlas, tumor_label, tumor_flair):
    '''
    This function eliminate small connceted components from PET tumour and combined this mask with the FLAIR tumour. the output will be used for 
    calculation of the probability distance map
    '''
    tumor_label_volume = np.zeros_like(atlas)
    tumor_label_volume[(atlas == tumor_label) | (tumor_flair == 1)] = 1

    labels, _ = label(tumor_label_volume)
    label_ids, counts = np.unique(labels, return_counts=True)
    label_ids = label_ids[1:] 
    counts = counts[1:] 

    # Remove small connected components
    for label_id, count in zip(label_ids, counts):
        if count < np.max(counts):
            tumor_label_volume[labels == label_id] = 0
            
    return tumor_label_volume

def sinus_sag(subj):
    '''
    The static image representing sinus sagittallis is obtained by a weighted sum of the first two frames of 4D PET.
    From this image, the sinus probability map is obtained by normalizing its maximum value.
    '''
    img = nib.load(subj.pet4d)
    vol_4d = img.get_fdata()
    sinus = np.sum(vol_4d[:, :, :, :2] * subj.frame_weight[:2], axis=3)
    _,brain_mask = binary_mask(subj)
    sinus[brain_mask == 0] = 0
    sinus_max = np.max(sinus)
    sinus_map = sinus/sinus_max
    
    return sinus_map

def calculate_distance_map(volume:np.array, step:list, exponent:float=2):
    '''
    Probability distance map is obtained by applying a gaussian filter to tumour label volume.
    '''
    volume = volume.astype(float)
    x,y,z = np.rint(center_of_mass(volume)).astype(int)

    xstep, ystep, zstep = step

    volume_inverse = np.ones_like(volume) 
    volume_inverse[x,y,z] = 0

    X, Y, Z = np.meshgrid(np.arange(volume.shape[0]), np.arange(volume.shape[1]), np.arange(volume.shape[2]), indexing='ij')

    distance_map = np.sqrt( 
            (xstep*(X - x))**2 + 
            (ystep*(Y - y))**2 + 
            (zstep*(Z - z))**2
            ) 

    distance_map = np.power(distance_map, exponent)

    fwhm = 5
    sigma = fwhm / 2.355

    distance_map = gaussian_filter(volume, sigma=sigma)

    distance_map /=np.max(distance_map)

    return distance_map

def create_tumor_dictionary(distance_map, sinus_map, tumor_index, tumor_idx):
    '''
    This function cretaes the dictionary that will contain the probability values of distance map and sinus map. This values will be used for classification.
    '''
    df_tumor = pd.DataFrame({

        'distance': distance_map[tumor_idx],
        'sinus': sinus_map[tumor_idx],
        'index': tumor_index,
        'location': 'tumor'

    })
    
    df_tumor['distance'] = (df_tumor['distance'] - df_tumor['distance'].min()) / (df_tumor['distance'].max() - df_tumor['distance'].min())
    df_tumor['sinus'] = (df_tumor['sinus'] - df_tumor['sinus'].min()) / (df_tumor['sinus'].max() - df_tumor['sinus'].min())

    return df_tumor

def ref_seg(subj):

    subj.sinus = subj.prefix + 'sinus_map.nii.gz'
    subj.tumor_lab = subj.prefix + 'tumor_label_volume.nii.gz'
    subj.distance_map = subj.prefix + 'distance_map.nii.gz'
    subj.volume_seg = subj.prefix + 'segmented_volume.nii.gz'

    atlas_hd = nib.load(subj.atlas_space_pet)
    atlas_vol = np.rint(atlas_hd.get_fdata()).astype(int)

    tumor_MRI_hd = nib.load(subj.volume_MRI)
    tumor_MRI_vol = np.rint(tumor_MRI_hd.get_fdata()).astype(int)

    if not os.path.exists(subj.tumor_lab):
        tumor_label_volume = create_tumor_label_volume(subj.tumor_atlas, subj.tumor_label, tumor_MRI_vol)
        nib.Nifti1Image(tumor_label_volume, nib.load(subj.atlas_space_pet).affine, dtype = np.int64).to_filename(subj.tumor_lab)
    else:
        tumorlab_hd = nib.load(subj.atlas_space_pet)
        tumor_label_volume = np.rint(tumorlab_hd.get_fdata()).astype(int)

    xstep, ystep, zstep = atlas_hd.header.get_zooms()

    if not os.path.exists(subj.distance_map):
        distance_map = calculate_distance_map(tumor_label_volume, [xstep, ystep, zstep])
        nib.Nifti1Image(distance_map, nib.load(subj.atlas_space_pet).affine).to_filename(subj.distance_map)
    else:
        distance_hd = nib.load(subj.distance_map)
        distance_map = distance_hd.get_fdata()

    if (Path(subj.data_dir+'/sub-'+subj.sub+'/ses-02').is_dir()):

        tumor_idx = (subj.tumor_atlas == subj.tumor_label).ravel()
        nvox = np.product(subj.tumor_atlas.shape)
        index_range = np.arange(nvox).astype(int)
        tumor_index = index_range[tumor_idx]

        if not os.path.exists(subj.sinus):
            sinus_map = sinus_sag(subj)
            nib.Nifti1Image(sinus_map, nib.load(subj.pet).affine).to_filename(subj.sinus)
        else:
            sinus_hd = nib.load(subj.sinus)
            sinus_map = sinus_hd.get_fdata()

        distance_map = distance_map.ravel()
        sinus_map = sinus_map.ravel()
        df_tumor = create_tumor_dictionary(distance_map, sinus_map, tumor_index, tumor_idx)

        if not os.path.exists(subj.volume_seg):
            
            ### K-means ###
            np.random.seed(42)
            from sklearn.cluster import KMeans

            n_clusters = 2
            kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init = 10)
            kmeans.fit(df_tumor[['distance', 'sinus']])
            df_tumor['label'] = kmeans.predict(df_tumor.iloc[:, :2])

            
            plt.figure()
            sns.scatterplot(data=df_tumor, x='distance', y='sinus', hue='label', alpha=0.1)
            plt.savefig(subj.prefix + '_clusters.png')
            plt.close()

            from sklearn.svm import SVC
            from sklearn.model_selection import GridSearchCV

            svm_model = SVC()
            param_grid = {
                'C': [0.1, 1, 10, 100],    # Regularization
                'gamma': [1, 0.1, 0.01, 0.001],  # Kernel Parameter
                'kernel': ['rbf']  # Kernel RBF
            }

            grid_search = GridSearchCV(svm_model, param_grid, cv=5)
            grid_search.fit(df_tumor[['distance', 'sinus']], df_tumor['label'])
            best_svm_model = grid_search.best_estimator_
            df_tumor['label_svm'] = best_svm_model.predict(df_tumor.iloc[:, :2])
            mean_distances_svm = df_tumor.groupby('label_svm')[['distance', 'sinus']].mean()
            max_mean_label_svm = mean_distances_svm.idxmax()[0]
            df_tumor['label_svm'] = df_tumor['label_svm'].apply(lambda x: 1 if x == max_mean_label_svm else 0)

            plt.figure()
            sns.scatterplot(data=df_tumor, x='distance', y='sinus', hue='label_svm', alpha=0.1)
            plt.savefig(subj.prefix + '_clusters_SVM.png')
            plt.close()
            segmented_volume = np.zeros_like(atlas_vol).reshape(-1,)
            segmented_volume[df_tumor['index'].values] = df_tumor['label_svm'].values + 1
            segmented_volume = segmented_volume.reshape(atlas_hd.shape)
            segmented_volume[segmented_volume == 1] = 0
            segmented_volume[segmented_volume == 2] = 2037
            nib.Nifti1Image(segmented_volume, atlas_hd.affine, dtype= np.int64).to_filename(subj.volume_seg)

        else:
            segmented_volume_hd = nib.load(subj.volume_seg)
            segmented_volume = np.rint(segmented_volume_hd.get_fdata()).astype(int) 
            
    else:
        if not os.path.exists(subj.volume_seg):
            distance_mask = distance_map > 0
            mask = (subj.tumor_atlas == subj.tumor_label) & (distance_mask)
            segmented_volume = np.zeros_like(atlas_vol)
            segmented_volume[mask == 1] = 2037
            nib.Nifti1Image(segmented_volume, atlas_hd.affine, dtype= np.int64).to_filename(subj.volume_seg)
        else:
            segmented_volume_hd = nib.load(subj.volume_seg)
            segmented_volume = np.rint(segmented_volume_hd.get_fdata()).astype(int) 
    
    return segmented_volume
