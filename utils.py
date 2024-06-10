import matplotlib 
matplotlib.rcParams['figure.facecolor'] = '1.'
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import re
import os
import ants
import argparse
import pandas as pd
import skimage.transform as skTrans
import csv
from skimage.morphology import dilation, erosion
from scipy.stats import linregress
from qc import ImageParam
from analysis import get_stats_for_labels,get_H_Uptake_lab, get_M_Uptake_lab, get_L_Uptake_lab
from argparse import ArgumentParser
from pathlib import Path
from sys import argv
from glob import glob
import seaborn

### Utility functions

def get_tacs(subj, roi, ref, times, tac_csv, qc_png= None, qc_sub_region_png = None, clobber= False):

    '''
        Inputs:
            subj: Subject class with its characteristics
            roi: striatum labels
            ref: reference labels
            times: initial time frames (seconds)
            tac_csv: csv file containing the TAC values for each region considered, this file will be created if it does not already exist.
            qc_png: TAC image
            qc_sub_region_png: TAC sub regions image
            clobber: bool, overwrite
    '''
    '''
        Outputs:
            df: these variable will contain tac_csv
    '''
    df = pd.DataFrame({'frame':[], 'label':[],'region':[], 'value':[],'std':[]})
    df_sub_r = pd.DataFrame({'frame':[], 'label':[],'region':[], 'value':[],'std':[]})
    
    ### Variables for Dynamic analysis ### 
    img = nib.load(subj.pet4d)
    vol_4d = img.get_fdata()
    
    pet_hd = nib.load(subj.pet)
    pet_3d = pet_hd.get_fdata()
    
    atlas_hd = nib.load(subj.atlas_space_pet)
    atlas_vol = np.rint(atlas_hd.get_fdata()).astype(int)

    tumor_avg, _, t_std, _ = get_stats_for_labels(pet_3d, subj.tumor_atlas, [subj.tumor_label])
    
    all_lab = [subj.striatum_label] + [ref] + [subj.tumor_label]

    print('\tExtract Time Activity Curves')
    print('\t\tPet:',subj.pet)
    print('\t\tAtlas:',subj.atlas_space_pet)
    print('\t\tReference Labels:', [ref])
    print('\t\tStriatum\tLabels:', roi)
    print('\t\t New Striatum\tLabels:', [subj.striatum_label])
    print('\t\tTumor\tLabel:', [subj.tumor_label])
    print('\t\tTAC csv:', tac_csv)
    print('\t\tTAC QC:', qc_png)
    print()

    clobber=True
    if not os.path.exists(tac_csv) or clobber :

        rows = []
        rows_sub_r = []
    
    for label in all_lab:

           for frame in range(vol_4d.shape[3]) :

               pet_frame = vol_4d[:,:,:,frame]
               assert np.sum(subj.tumor_atlas == label) > 0 or np.sum(subj.striatum_atlas == label) > 0  , f'Error: could not find {label} in atlas'

               if label == subj.tumor_label:
                   values = pet_frame[subj.tumor_atlas == label]
               else:
                   values = pet_frame[subj.striatum_atlas == label]

               if np.sum(np.abs(values)) > 0 :
                   avg = np.mean(values)
                   dev = np.std(values)
               else:
                    avg=0
                    dev = 0
                    
               new_row = pd.DataFrame({'time':[times[frame]],'frame':[frame], 'region':[label], 'value':[avg],'std':[dev]})
               rows.append(new_row)
             
    df = pd.concat(rows)

    # Plotting TAC of the tumor lesion, the controlateral striatum and the controlateral cerebral white matter
    if type(qc_png) == str :
        plt.figure()
        plt.title('Time Activity Curves')
        cmap = seaborn.color_palette(palette = "colorblind", n_colors = 3)
        for idx, l in enumerate(all_lab):
            time = df['time'][df['region'] == l].to_numpy()
            value = df['value'][df['region'] == l].to_numpy()
            std = df['std'][df['region'] == l].to_numpy()

            if l == 2:
                legend = 'Left Cortical White Matter'
            elif l == 41:
                legend = 'Right Cortical White Matter'
            elif l == subj.striatum_label:
                if subj.roi_labels[0] == 11:
                    legend = 'Left Striatum'
                else:
                    legend = 'Right Striatum'
            elif l == 12:
                legend = 'Left Putamen'
            elif l == 47:
                legend = 'Right Cerebellum White Matter'
            else:
                legend = 'Tumor'
            
            
            plt.plot(time, value,label = legend, color = cmap[idx])
            plt.fill_between(time, value-std, value+std, alpha=0.1)
            
        plt.legend(loc= 'lower right')
        plt.xlabel('Time [s]')
        plt.ylabel('Value [Bq/ml]')
        plt.savefig(qc_png, dpi = 600)
        plt.close()
            
        df.to_csv(tac_csv)

    else :
        df = pd.read_csv(tac_csv)

    # calculating atlases for tumor sub-regions
    H_atlas_vol, HU_label = get_H_Uptake_lab(subj.striatum_atlas, tumor_avg, t_std, pet_3d, subj.tumor_atlas, subj.tumor_label)
    M_atlas_vol, MU_label = get_M_Uptake_lab(subj.striatum_atlas, tumor_avg, t_std, pet_3d, subj.tumor_atlas, subj.tumor_label)
    L_atlas_vol, LU_label = get_L_Uptake_lab(subj.striatum_atlas, tumor_avg, t_std, pet_3d, subj.tumor_atlas, subj.tumor_label)
    all_new_lab = [HU_label, MU_label, LU_label]

    if((np.any(H_atlas_vol == HU_label))):

        subj.bool_flag = True
        # These variables will be called in pediatric_fdopa_pipeline.py
        tumor_voxel = np.shape(pet_3d[subj.tumor_atlas == subj.tumor_label])[0]
        H_tumor_voxel = np.shape(pet_3d[H_atlas_vol == HU_label])[0]

        subj.tum_percentage = np.round((H_tumor_voxel / tumor_voxel) *100,3)
    
        AT = np.zeros((*pet_3d.shape, 3))
        AT[:,:,:,0] = H_atlas_vol
        AT[:,:,:,1] = M_atlas_vol
        AT[:,:,:,2] = L_atlas_vol
    
        for l in range(0, 3):
            for frame in range(vol_4d.shape[3]) :
                pet_frame = vol_4d[:,:,:,frame]
                assert np.sum(AT[:,:,:,l] == all_new_lab[l]) > 0, f'Error: could not find {all_new_lab[l]} in atlas'
                values = pet_frame[AT[:,:,:,l] == all_new_lab[l]]
                if np.sum(np.abs(values)) > 0 :
                    avg = np.mean(values)
                    dev = np.std(values)
                else:
                    avg=0
                    dev = 0
            
                new_row_sub_r = pd.DataFrame({'time':[times[frame]],'frame':[frame], 'region':[all_new_lab[l]], 'value':[avg],'std':[dev]})
                rows_sub_r.append(new_row_sub_r)
            
        df_sub_r = pd.concat(rows_sub_r)
        df_sub_r.to_csv(subj.tacs_sub_regions_csv)

        #plotting the TACs of tumor sub-regions vs the mean of the tumor region
        if type(qc_sub_region_png) == str:
            plt.figure()
            plt.title('Tumor TAC vs tumor sub-regions TACs')
            cmap = seaborn.color_palette(palette = "colorblind", n_colors = 10)
            time = df['time'][df['region'] == subj.tumor_label].to_numpy()
            value = df['value'][df['region'] == subj.tumor_label].to_numpy()
            std = df['std'][df['region'] == subj.tumor_label].to_numpy()
            color = [cmap[3], cmap[1], cmap[2]]
        
            for l in range(0, 3):
                value_sub_r = df_sub_r['value'][df_sub_r['region'] == all_new_lab[l]].to_numpy()
                std_sub_r = df_sub_r['std'][df_sub_r['region'] == all_new_lab[l]].to_numpy()
                if all_new_lab[l] == HU_label:
                    legend1 = 'Highest Uptake Tumor'
                elif all_new_lab[l] == MU_label:
                    legend1 = 'Mid Uptake Tumor'
                else:
                    legend1 = 'Lowest Uptake Tumor'
                
                plt.plot(time, value_sub_r,color = color[l], label = legend1)
                plt.fill_between(time, value_sub_r-std_sub_r, value_sub_r+std_sub_r, color = color[l], alpha=0.1)
            
            legend = 'Tumor'
            plt.plot(time, value,label = legend, color = cmap[0])
            plt.fill_between(time, value-std, value+std, alpha=0.1)
                
        
            plt.legend(loc= 'lower right')
            plt.xlabel('Time [s]')
            plt.ylabel('Value [Bq/ml]')
            plt.savefig(subj.tacs_sub_regions_qc_plot, dpi = 600)
            plt.close()

        # saving the three masks as Nifti volumes
        H_atlas_vol[H_atlas_vol != HU_label] = 0
        M_atlas_vol[M_atlas_vol != MU_label] = 0
        L_atlas_vol[L_atlas_vol != LU_label] = 0
        
        nib.Nifti1Image(H_atlas_vol, atlas_hd.affine , header = atlas_hd.header).to_filename(subj.prefix+'H_tumor_atlas.nii.gz')
        nib.Nifti1Image(M_atlas_vol, atlas_hd.affine , header = atlas_hd.header).to_filename(subj.prefix+'M_tumor_atlas.nii.gz')
        nib.Nifti1Image(L_atlas_vol, atlas_hd.affine , header = atlas_hd.header).to_filename(subj.prefix+'L_tumor_atlas.nii.gz')
    else:
        subj.bool_flag = False
        
    return df

def get_dynamic_parameters(subject, qc_png=None):

    '''
    This function calculates dynamic parameters from time activity curve csv file. 
    Time-To-Peak (TTP) is calculated as t(s) in correspondence of the maximum of the TAC
    Slope tumour (SlopeT) is calculated by a linear regression of tumour TAC in interval (2 -> end of acquisition)
    Slope striatum (SlopeS) is calculated by a linear regression of striatum TAC in interval (2 -> end of acquisition)
    Dynamic Slope Ratio (DSR) is calculated as the fraction between SlopeT/SlopeS
    '''

    print('\tExtract Dynamic Parameters: ')
    print(f'\t\tTime To Peak')
    print(f'\t\tDynamic Slope Ratio')
    print()

    subject.dyn_csv = subject.prefix+'dyn.json'

    if not os.path.exists(subject.dyn_csv) or subject.clobber :
        tumor = subject.tacs[subject.tacs['region'] == subject.tumor_label]
        striatum = subject.tacs[subject.tacs['region'] == subject.striatum_label]
        time = tumor['time']
    
        striatum = striatum['value'].to_numpy()
        tum = tumor['value'].to_numpy()
        t = tumor['time'].to_numpy()

        max_index = tum.argmax()
        TTP = t[max_index]
        
        # the time interval is 25.5 minutes: modified for new PET instrumentation with larger intervals
        if sum(subject.frame_time_start == 120) == 1:
            index_1 = np.where(t == 120)
        else:
            index_1 = np.where(t == 150)

        index_l = np.argmax(t)

        slopeT, interceptT, rT, pT, seT = linregress(t[int(index_1[0]):(index_l+1)], tum[int(index_1[0]):(index_l+1)])
        slopeS, interceptS, rS, pS, seS = linregress(t[int(index_1[0]):(index_l+1)], striatum[int(index_1[0]):(index_l+1)])
        
        reg_line = slopeT*t[int(index_1[0]):(index_l+1)] + interceptT
        reg_line_s = slopeS*t[int(index_1[0]):(index_l+1)] + interceptS
        
        DSR = slopeT/slopeS

        plt.figure()
        plt.title('Subject '+ str(subject.sub) + ' Regression line')
        cmap = seaborn.color_palette(palette = "colorblind", n_colors = 10)
        plt.scatter(t, tum,label = 'Tumor',s = 15, color = cmap[0])
        plt.plot(t, tum,color = cmap[0], linewidth = 1.5)
        plt.scatter(t[int(index_1[0])], reg_line[0], color = cmap[1],s = 15)
        plt.scatter(t[index_l], reg_line[-1], color = cmap[1],s = 15)
        plt.plot(t[int(index_1[0]):(index_l+1)], reg_line , color = cmap[1],ls='--', label = 'Regression Line Tumor',linewidth = 1.5)

        plt.scatter(t, striatum, color = cmap[2],label = 'Striatum',s = 15)
        plt.plot(t, striatum,color = cmap[2],linewidth = 1.5)
        plt.scatter(t[int(index_1[0])], reg_line_s[0], color = cmap[3],s = 15)
        plt.scatter(t[index_l], reg_line_s[-1], color = cmap[3],s = 15)
        plt.plot(t[int(index_1[0]):(index_l+1)], reg_line_s , color = cmap[3],ls='--', label = 'Regression Line Striatum',linewidth = 1.5)

        plt.legend(loc = 'lower right')
        plt.xlabel('Time [s]')
        plt.ylabel('Value [Bq/ml]')
        plt.savefig(qc_png, dpi = 600)
        plt.close()
    
        dyn_dict = {'Subject':[subject.sub],'TTP':[TTP], 'Slope_Tumor':[slopeT],'Slope_Striatum':[slopeS], 'DSR':[DSR]}
        subject.dy_df = pd.DataFrame(dyn_dict)    
        subject.dy_df.to_csv(subject.dyn_csv, index=False)

    else :
        subject.dy_df = pd.read_csv(subject.dyn_csv)

    return subject

def get_file(data_dir, string):
    
    lst = list(Path(data_dir).rglob(string))
    if len(lst)==1 :
        return str(lst[0])
    else :

        print('Could not find single file for string')
        print(lst)
        exit(1)

def transform(prefix, fx, mv, tfm, interpolator='linear', qc_filename=None, clobber=False):
    print('\tTransforming')
    print('\t\tFixed',fx)
    print('\t\tMoving',mv)
    print('\t\tTransformations:', tfm)
    print('\t\tQC:', qc_filename)
    print()
    
    out_fn = prefix +  re.sub('.nii.gz','_rsl.nii.gz', os.path.basename(mv))
    if not os.path.exists(out_fn) or clobber :
        img_rsl = ants.apply_transforms(fixed= ants.image_read(fx), 
                                        moving=ants.image_read(mv), 
                                        transformlist=tfm,
                                        interpolator=interpolator,
                                        verbose=True
                                        )
        ants.image_write( img_rsl, out_fn )
        
        if type(qc_filename) == str :
            ImageParam(fx, qc_filename, out_fn, duration=600,  nframes=15, dpi=200, alpha=[0.4]   ).volume2gif()
    return out_fn

def align(fx, mv, transform_method='SyNAggro', init=[], outprefix='', qc_filename=None) :
   
    warpedmovout =  outprefix + 'fwd.nii.gz'
    warpedfixout =  outprefix + 'inv.nii.gz'
    fwdtransforms = outprefix+'Composite.h5'
    invtransforms = outprefix+'InverseComposite.h5'

    print(f'\tAligning\n\t\tFixed: {fx}\n\t\tMoving: {mv}\n\t\tTransform: {transform_method}')
    print(f'\t\tQC: {qc_filename}\n')
    output_files = warpedmovout, warpedfixout, fwdtransforms, invtransforms
    if False in [os.path.exists(fn) for fn in output_files ] :
        out = ants.registration(fixed = ants.image_read(fx), 
                                moving = ants.image_read(mv), 
                                type_of_transform = transform_method, 
                                init=init,
                                verbose=True,
                                outprefix=outprefix,
                                write_composite_transform=True
                                )
        ants.image_write(out['warpedmovout'], warpedmovout)
        ants.image_write(out['warpedfixout'], warpedfixout)
        
        if type(qc_filename) == str :
            ImageParam(fx, qc_filename, warpedmovout, duration=600,  nframes=15, dpi=200, alpha=[0.3],  edge_2=1, cmap1=plt.cm.Greys, cmap2=plt.cm.Reds ).volume2gif()

    return output_files


