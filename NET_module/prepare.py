import numpy as np
import nibabel
import json
import os
import time
from glob import glob
from subprocess import call
from joblib import Parallel, delayed
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser


# inspired by the NVIDIA nnU-Net GitHub repository available at:
# https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Segmentation/nnUNet

# inspired by
# Bianconi, Andrea, et al. "Deep learning-based algorithm for postoperative glioblastoma MRI segmentation: 
# a promising new tool for tumor burden assessment." Brain Informatics 10.1 (2023): 26.

### modified by Mureddu Michele

def load_nifti(directory, example_id, modality):
    """
    Read NIfTI file.
    :param directory: patient file directory
    :param example_id: patient file id
    :param modality: patient scan modality
    :return: nibabel.load()
    """
    return nibabel.load(os.path.join(directory, example_id + "-" + modality + ".nii.gz"))


def load_channels(directory, example_id, modalities=("flair", "t1", "t1ce", "t2")):
    """
    Load all indicated MRI scan modalities.
    :param directory: patient directory
    :param example_id: patient file id
    :param modalities: iterable with patient scan modalities. Choose one or more between ("flair", "t1", "t1ce", "t2")
    :return: modalities, list with nibabel.load() for each modality
    """
    try:
        _ = (el for el in modalities)
    except TypeError:
        print(f"{modalities} is not iterable.")
    scans = [load_nifti(directory=directory, example_id=example_id, modality=modality) for modality in modalities]
    assert len(scans) >= 1, "at least one scan modality is required"

    return scans


def get_data(nifti, dtype="int16"):
    """
    Retrieve NIfTI file data as numpy array.
    :param nifti: NIfTI file
    :param dtype: numpy matrix dtype (default "int16". If different, "uint8" is used)
    :return: NIfTI file data as numpy array
    """
    if dtype == "int16":
        data = np.abs(nifti.get_fdata().astype(np.int16))
        data[data == -32768] = 0  # outlier value
        return data

    return nifti.get_fdata().astype(np.uint8)


def prepare_nifti(directory, modalities=("flair", "t1", "t1ce", "t2")):
    """
    Prepare stacked NIfTI containing all modalities.
    If present, convert segmentation to uint8 and assign label 3 to enhancing tumor (BraTS assigns 4 by default).
    :param directory: patient file directory
    :param modalities: iterable with patient scan modalities. Choose one or more between ("flair", "t1", "t1ce", "t2")
    """
    # retrieve patient file id
    example_id = directory.split("/")[-1]
    scans = load_channels(directory=directory, example_id=example_id, modalities=modalities)

    # retrieve homogeneous affine and header metadata
    affine = scans[0].affine
    header = scans[0].header

    # stack modalities, create NIfTI and save it
    dataobj = np.stack([get_data(nifti=scan) for scan in scans], axis=-1)
    img = nibabel.nifti1.Nifti1Image(dataobj=dataobj, affine=affine, header=header)
    nibabel.save(img=img, filename=os.path.join(directory, example_id + ".nii.gz"))

    if os.path.exists(os.path.join(directory, example_id + "-seg.nii.gz")):
        # segmentation data exists -> processing the same way as above
        segmentation = load_nifti(directory=directory, example_id=example_id, modality="seg")
        affine = segmentation.affine
        header = segmentation.header
        dataobj = get_data(nifti=segmentation, dtype="uint8")
        # assigning label 3 to enhancing tumor
        if np.max(dataobj)== 1:
            seg_1 = dataobj == 1
            dataobj[seg_1] = 1
        else:
            dataobj[dataobj == 4] = 3
            seg_1 = dataobj == 1
            seg_2 = dataobj == 2
            seg_3 = dataobj == 3
            dataobj[seg_1 | seg_2 | seg_3] = 1
        img = nibabel.nifti1.Nifti1Image(dataobj=dataobj, affine=affine, header=header)
        nibabel.save(img=img, filename=os.path.join(directory, example_id + "-seg.nii.gz"))


def prepare_dirs(data, train):
    """
    Prepare directories splitting between images and labels
    :param data: outer patients directory
    :param train: boolean value to determine if directory contains training data
    """
    images_path = os.path.join(data, "images")
    labels_path = os.path.join(data, "labels")
    call(f"mkdir {images_path}", shell=True)
    if train:
        call(f"mkdir {labels_path}", shell=True)

    # return a possibly-empty list of path names that match pathname
    directories = glob(pathname=os.path.join(data, "BraTS*"))
    for directory in directories:
        if "-" in directory.split("/")[-1]:
            files = glob(pathname=os.path.join(directory, "*.nii.gz"))
            for file in files:
                if ("t2f" in file) or ("t1n" in file) or ("t1c" in file) or ("t2w" in file):
                    continue
                if "-seg" in file:
                    # move segmentation file to labels directory
                    call(f"mv {file} {labels_path}", shell=True)
                else:
                    call(f"mv {file} {images_path}", shell=True)
        # delete explored directory
        call(f"rm -rf {directory}", shell=True)


def prepare_dataset_json(data, train, modalities=("t2f", "t1n", "t1c", "t2w")):
    """
    Prepare BraTS2021 dataset as a json dictionary.
    :param data: outer patients directory
    :param train: boolean value to determine if directory contains training data
    :param modalities: iterable with patient scan modalities. Choose one or more between ("t2f", "t1", "t1c", "t2")
    """
    # match all possible images and labels
    images = glob(os.path.join(data, "images", "*"))
    labels = glob(os.path.join(data, "labels", "*"))

    # keep only filenames (drop hierarchical path) and sort
    images = sorted([img.replace(data + "/", "") for img in images])
    labels = sorted([lbl.replace(data + "/", "") for lbl in labels])

    # create dictionaries for both modalities and labels
    modality = {}
    idx = 0
    if "t2f" in modalities:
        modality[f"{idx}"] = "FLAIR"
        idx += 1
    if "t1n" in modalities:
        modality[f"{idx}"] = "T1"
        idx += 1
    if "t1c" in modalities:
        modality[f"{idx}"] = "T1CE"
        idx += 1
    if "t2w" in modalities:
        modality[f"{idx}"] = "T2"
        idx += 1
    labels_dict = {"0": "background", "1": "whole tumor"} #, "2": "non-enhancing tumor", "3": "enhancing tumour"}
    if train:
        key = "training"
        data_pairs = [{"image": image, "label": label} for (image, label) in zip(images, labels)]
    else:
        key = "test"
        data_pairs = [{"image": image} for image in images]

    dataset = {
        "labels": labels_dict,
        "modality": modality,
        key: data_pairs,
    }

    # create json
    with open(os.path.join(data, "dataset.json"), "w") as outfile:
        json.dump(dataset, outfile)


def prepare_dataset(data, train, modalities=("flair", "t1", "t1ce", "t2")):
    """
    Prepare BraTS dataset in its final form.
    :param data: outer patients directory
    :param modalities: iterable with patient scan modalities. Choose one or more between ("flair", "t1", "t1ce", "t2")
    :param train: boolean value to determine if directory contains training data
    """
    print(f"Preparing BraTS21 dataset from: {data}")
    start = time.time()
    # parallel running jobs mapping
    Parallel(n_jobs=os.cpu_count())(
        delayed(prepare_nifti)(directory, modalities) for directory in sorted(glob(os.path.join(data, "BraTS*")))
    )
    prepare_dirs(data=data, train=train)
    prepare_dataset_json(data=data, train=train, modalities=modalities)
    end = time.time()
    print(f"Preparing time: {(end - start):.2f}")


# define the ArgumentParser
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--flair", action="store_true", help="Confirm that FLAIR modality is present.")
parser.add_argument("--t1", action="store_true", help="Confirm that T1 modality is present.")
parser.add_argument("--t1ce", action="store_true", help="Confirm that T1CE modality is present.")
parser.add_argument("--t2", action="store_true", help="Confirm that T2 modality is present.")


if __name__ == "__main__":
    # retrieve the modalities
    args = parser.parse_args()
    modalities = []
    if args.flair:
        modalities.append("t2f")
    if args.t1:
        modalities.append("t1n")
    if args.t1ce:
        modalities.append("t1c")
    if args.t2:
        modalities.append("t2")

    # prepare BraTS dataset
    prepare_dataset(data="./brats_flair/train", train=True, modalities=tuple(modalities))
    prepare_dataset(data="./brats_flair/val", train=False,  modalities=tuple(modalities))
    print("Finished!")
