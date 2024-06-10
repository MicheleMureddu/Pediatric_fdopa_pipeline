import nibabel as nib
import numpy as np
from ref_tumor_seg import calculate_distance_map

'''
Regions of Interests: 
2 & 41: Left and Right Cerebral White Matter
16: Pons
10 & 49: Left and Right Thalamus
'''

def region_selection(subject):

    tumor_MRI_hd = nib.load(subject.volume_MRI)
    tum_flair = np.rint(tumor_MRI_hd.get_fdata()).astype(int)
    atlas_hd = nib.load(subject.atlas_space_pet)
    atlas_vol = np.rint(atlas_hd.get_fdata()).astype(int)
    
    xstep, ystep, zstep = atlas_hd.header.get_zooms()
    distance_map = calculate_distance_map(tum_flair, [xstep, ystep, zstep])
        
    max_prob_41 = np.max(distance_map[atlas_vol == 41])
    max_prob_2 = np.max(distance_map[atlas_vol == 2])
    max_prob_16 = np.max(distance_map[atlas_vol == 16])
    max_prob_10 = np.max(distance_map[atlas_vol == 10])
    max_prob_49 = np.max(distance_map[atlas_vol == 49])
        
    mask_2_41 = (atlas_vol == 2) | (atlas_vol == 41)
    mask_2 = (atlas_vol == 2)
    mask_41 = (atlas_vol == 41)
    and_mask = (tum_flair == 1) & (mask_2_41 == 1)
    mask_10 = atlas_vol == 10
    mask_49 = atlas_vol == 49

    if(np.any(and_mask)):
        if(np.any((tum_flair == 1) & (mask_2 == 1)) and not np.any((tum_flair == 1) & (mask_41 == 1))):

            if(np.any((mask_10) & (tum_flair == 1))):
                if(max_prob_10> max_prob_49):
                    subject.ref_labels = 41
                else:
                    subject.ref_labels = 2
            elif((max_prob_16 > max_prob_2)):
                subject.ref_labels = 2
            else:
                subject.ref_labels = 41

        elif(np.any((tum_flair == 1) & (mask_41 == 1)) and not np.any((tum_flair == 1) & (mask_2 == 1))):

            if(np.any((mask_49) & (tum_flair == 1))):
                if( max_prob_10 > max_prob_49):
                    subject.ref_labels = 41
                else:
                    subject.ref_labels = 2
            elif((max_prob_16 > max_prob_41)):
                subject.ref_labels = 2
            else:
                subject.ref_labels = 2

        elif(np.any((tum_flair == 1) & (mask_41 == 1)) and np.any((tum_flair == 1) & (mask_2 == 1))):
            
            if(max_prob_41 > max_prob_2):
                subject.ref_labels = 2
            else:
                subject.ref_labels = 41
    else:
        subject.ref_labels = 2
    
    if (subject.ref_labels == 2):
        subject.roi_labels = [11,12,13]
    else:
        subject.roi_labels = [50,51,52]
  
    return subject
