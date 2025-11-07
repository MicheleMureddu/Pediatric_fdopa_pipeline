import numpy as np
import nibabel as nib
import os
from glob import glob
from scipy.ndimage.measurements import label
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

# inspired by the NVIDIA nnU-Net GitHub repository available at:
# https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Segmentation/nnUNet

# inspired by
# Bianconi, Andrea, et al. "Deep learning-based algorithm for postoperative glioblastoma MRI segmentation: 
# a promising new tool for tumor burden assessment." Brain Informatics 10.1 (2023): 26.

### modified by Mureddu Michele

def back_to_original_labels(pred, preop):
    """
    Convert back the triplet (ET, TC, WT) to the original (NCR, ED, ET) for a given prdiction.
    :param pred: prediction
    :param preop: boolean value to check if postprocessing is applied for pre-operative segmentation.
    :return: converted prediction
    """
    pred = pred[0]
    bin_pred = (pred > 0.40).astype(np.uint8)
    if (np.sum(bin_pred == 1)) == 0:
        bin_pred = (pred > 0.30).astype(np.uint8)
    
    # transpose to fit BraTS orientation
    bin_pred = np.transpose(bin_pred, (2, 1, 0)).astype(np.uint8)

    return bin_pred

def prepare_preditions(example, images_dir, out_dir, preop):

    """
    Convert back to original BraTS labels and save as NIfTI.
    :param example: example file
    :param preop: boolean value to check if postprocessing is applied for pre-operative segmentation.
    :return: post-processed NIfTI file
    """

    fname = example[0].split("/")[-1].split(".")[0]

    preds = [np.load(f) for f in example]

    # convert back to original BraTS labels
    p = back_to_original_labels(np.mean(preds, 0), preop)

    # save as NIfTI
    img = nib.load(images_dir/f"{fname}.nii.gz")

    base_name = example[0].split("/")[-1].split(".")[0]
    
    if "-" in base_name:
        prefix = "-".join(base_name.split("-")[:-1])
    else:
        prefix = base_name
    
    fnamenew = prefix + "-000"

    nib.save(
        nib.Nifti1Image(p, img.affine, header=img.header),
        out_dir/(fnamenew + "-seg.nii.gz"),
    )


def run_postprocess(predictions_dirs, images_dir, output_dir, output_type="postop"):
    
    """
    Convert predictions .npy to final NIfTI segmentations.
    :param predictions_dirs: list of folders containing .npy prediction files
    :param images_dir: folder with original NIfTI images (for affine/header)
    :param output_dir: folder to save final NIfTI segmentations
    :param output_type: 'preop' or 'postop'
    """

    examples = list(zip(*[sorted(glob(f"{p}/*.npy")) for p in predictions_dirs]))
    print("Preparing final predictions...")
    for example in examples:
        prepare_preditions(example, images_dir, output_dir, preop=(output_type=="preop"))
    print("Finished!")
