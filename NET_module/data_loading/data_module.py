
import numpy as np
import glob
import os

from pytorch_lightning import LightningDataModule
from sklearn.model_selection import KFold

from NET_module.utils.utils import get_config_file, get_task_code, print0
from NET_module.data_loading.dali_loader import fetch_dali_loader


# inspired by the NVIDIA nnU-Net GitHub repository available at:
# https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Segmentation/nnUNet

# DataModule makes use of the NVIDIA Data Loading Library (DALI)
# read more at: https://docs.nvidia.com/deeplearning/dali/user-guide/docs/

# inspired by
# Bianconi, Andrea, et al. "Deep learning-based algorithm for postoperative glioblastoma MRI segmentation: 
# a promising new tool for tumor burden assessment." Brain Informatics 10.1 (2023): 26.

class DataModule(LightningDataModule):
    def __init__(self, args):
        """
        Initialize the LightningDataModule.
        A datamodule encapsulates the five steps involved in data processing in PyTorch:
            download / tokenize / process
            clean and (maybe) save to disk
            load inside Dataset
            apply transforms (rotate, tokenize, etc…)
            wrap inside a DataLoader
        :param args: args
        """
        super().__init__()
        self.args = args
        self.data_path = get_data_path(args)
        self.kfold = get_kfold_splitter(args.nfolds)
        self.kwargs = {
            "seed": self.args.seed,
            "gpus": self.args.gpus,
            "overlap": self.args.overlap,
            "num_workers": self.args.num_workers,
            "oversampling": self.args.oversampling,
            "patch_size": get_config_file(self.args)["patch_size"],
            "probability": 0.15,
        }
        self.train_images, self.train_labels, self.val_images, self.val_labels, self.test_images = ([],) * 5

    def setup(self, stage=None):
        """
        Create the starting datasets splitting between training and validation.
        :param stage: used to separate setup logic. Here initialized to None since all stages have been set-up.
        """
        meta = load_data(self.data_path, "*_meta.npy")
        images = load_data(self.data_path, "*_x.npy")
        self.test_images = images.copy()
        test_meta = meta.copy()

        if self.args.exec_mode != "predict":
            # retrieve train and validation
            labels = load_data(self.data_path, "*_y.npy")
            orig_lbl = load_data(self.data_path, "*_orig_lbl.npy")
            train_idx, val_idx = list(self.kfold.split(images))[self.args.fold]
            orig_lbl, meta = get_split(orig_lbl, val_idx), get_split(meta, val_idx)
            self.kwargs.update({"orig_lbl": orig_lbl, "meta": meta})
            self.train_images, self.train_labels = get_split(images, train_idx), get_split(labels, train_idx)
            self.val_images, self.val_labels = get_split(images, val_idx), get_split(labels, val_idx)
        else:
            # prediction only
            self.kwargs.update({"meta": test_meta})
        print0(f"{len(self.train_images)} training, {len(self.val_images)} validation, "
               f"{len(self.test_images)} test examples.")

    def train_dataloader(self):
        """
        Fetch the train DALI data loader.
        :return: train DALI data loader
        """
        return fetch_dali_loader(self.train_images, self.train_labels, self.args.batch_size, "train", **self.kwargs)

    def val_dataloader(self):
        """
        Fetch the eval DALI data loader.
        :return: eval DALI data loader
        """
        return fetch_dali_loader(self.val_images, self.val_labels, 1, "eval", **self.kwargs)

    def test_dataloader(self):
        """
        Fetch the test DALI data loader.
        :return: test DALI data loader
        """
        return fetch_dali_loader(self.test_images, None, 1, "test", **self.kwargs)


class DataModulePostop(LightningDataModule):
    def __init__(self, args):
        """
        Initialize the LightningDataModule.
        A datamodule encapsulates the five steps involved in data processing in PyTorch:
            download / tokenize / process
            clean and (maybe) save to disk
            load inside Dataset
            apply transforms (rotate, tokenize, etc…)
            wrap inside a DataLoader
        :param args: args
        """
        super().__init__()
        self.args = args
        self.data_path = get_data_path(args)
        self.kfold = get_kfold_splitter(args.nfolds)
        self.kwargs = {
            "seed": self.args.seed,
            "gpus": self.args.gpus,
            "overlap": self.args.overlap,
            "num_workers": self.args.num_workers,
            "oversampling": self.args.oversampling,
            "patch_size": get_config_file(self.args)["patch_size"],
            "probability": 0.5,
        }
        self.not_val = open(self.args.not_val).read().splitlines() if self.args.not_val is not None else None
        self.train_images, self.train_labels, self.val_images, self.val_labels, self.test_images = ([],) * 5

    def setup(self, stage=None):
        """
        Create the starting datasets splitting between training and validation.
        :param stage: used to separate setup logic. Here initialized to None since all stages have been set-up.
        """
        meta = load_data(self.data_path, "*_meta.npy")
        images = load_data(self.data_path, "*_x.npy")
        to_split = sorted(list(os.path.split(name)[-1].split("_")[1].split("N")[0] for name in images))
        if self.not_val is not None:
            # remove not volumetric scans before k-fold split
            to_split = list(filter(lambda name: os.path.split(name)[-1].split("_")[1].split("N")[0] not in self.not_val, images))
            to_split = sorted(list(set([os.path.split(name)[-1].split("_")[1].split("N")[0] for name in to_split])))
        self.test_images = images.copy()
        test_meta = meta.copy()

        if self.args.exec_mode != "predict":
            # retrieve train and validation
            train_idx, val_idx = list(self.kfold.split(to_split))[self.args.fold]
            train_patients = get_split(to_split, train_idx)
            val_patients = get_split(to_split, val_idx)
            orig_lbl = []
            meta = []
            for patient in val_patients:
                orig_lbl += sorted(glob.glob(os.path.join(self.data_path, f"{patient}*_orig_lbl.npy")))
                meta += sorted(glob.glob(os.path.join(self.data_path, f"{patient}*_meta.npy")))
            self.kwargs.update({"orig_lbl": orig_lbl, "meta": meta})
            self.train_images = []
            self.train_labels = []
            for patient in train_patients:
                self.train_images += sorted(glob.glob(os.path.join(self.data_path, f"BraTS2021_{patient}*_x.npy")))
                self.train_labels += sorted(glob.glob(os.path.join(self.data_path, f"BraTS2021_{patient}*_y.npy")))
            if self.not_val is not None:
                # add not volumetric scans for training
                for patient in self.not_val:
                    self.train_images += sorted(glob.glob(os.path.join(self.data_path, f"BraTS2021_{patient}*_x.npy")))
                    self.train_labels += sorted(glob.glob(os.path.join(self.data_path, f"BraTS2021_{patient}*_y.npy")))
            self.val_images = []
            self.val_labels = []
            for patient in val_patients:
                self.val_images += sorted(glob.glob(os.path.join(self.data_path, f"BraTS2021_{patient}*_x.npy")))
                self.val_labels += sorted(glob.glob(os.path.join(self.data_path, f"BraTS2021_{patient}*_y.npy")))
        else:
            # prediction only
            self.kwargs.update({"meta": test_meta})
        print0(f"{len(self.train_images)} training, {len(self.val_images)} validation, "
               f"{len(self.test_images)} test examples.")

    def train_dataloader(self):
        """
        Fetch the train DALI data loader.
        :return: train DALI data loader
        """
        return fetch_dali_loader(self.train_images, self.train_labels, self.args.batch_size, "train", **self.kwargs)

    def val_dataloader(self):
        """
        Fetch the eval DALI data loader.
        :return: eval DALI data loader
        """
        return fetch_dali_loader(self.val_images, self.val_labels, 1, "eval", **self.kwargs)

    def test_dataloader(self):
        """
        Fetch the test DALI data loader.
        :return: test DALI data loader
        """
        return fetch_dali_loader(self.test_images, None, 1, "test", **self.kwargs)


def get_split(data, idx):
    """
    Retrieve data split for a given set of indices idx.
    :param data: data
    :param idx: set of indices
    :return: list with corresponding split
    """
    return list(np.array(data)[idx])


def load_data(path, files_pattern, non_empty=True):
    """
    Retrieve all filenames including a given files_pattern from a path.
    :param path: path
    :param files_pattern: recurrent files pattern
    :param non_empty: boolean value to assert whether desired files exist or not
    :return: list of filenames
    """
    data = sorted(glob.glob(os.path.join(path, files_pattern)))
    if non_empty:
        assert len(data) > 0, f"No data found in {path} with pattern {files_pattern}"

    return data


def get_kfold_splitter(nfolds):
    """
    Retrieve the sklearn.model_selection.KFold for splitting data.
    :param nfolds: desired number of folds
    :return: sklearn.model_selection.KFold()
    """
    return KFold(n_splits=nfolds, shuffle=True, random_state=12345)


def get_data_path(args):
    """
    Retrieve data path.
    :param args: args
    :return: data path
    """
    if args.data != "./brats_flair":
        return args.data

    data_path = os.path.join(args.data, get_task_code(args))
    if args.exec_mode == "predict":
        data_path = os.path.join(data_path, "test")

    return data_path
