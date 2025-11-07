
import numpy as np
import torch
import ctypes
import os
import pickle

from subprocess import run
from pytorch_lightning.utilities import rank_zero_only


# inspired by the NVIDIA nnU-Net GitHub repository available at:
# https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Segmentation/nnUNet

# inspired by
# Bianconi, Andrea, et al. "Deep learning-based algorithm for postoperative glioblastoma MRI segmentation: 
# a promising new tool for tumor burden assessment." Brain Informatics 10.1 (2023): 26.


@rank_zero_only
def print0(text):
    """
    Print text in rank_zero_only mode.
    :param text: text
    """
    print(text)


def get_task_code(args):
    """
    Retrieve task code.
    :param args: main args
    :return: {args.task}_3d
    """
    return f"{args.task}_3d"


def get_config_file(args):
    """
    Load config pickle file.
    :param args: main args
    :return: loaded pickle file
    """
    if args.data != "./brats_flair":
        path = os.path.join(args.data, "config.pkl")
    else:
        task_code = get_task_code(args)
        path = os.path.join(args.data, task_code, "config.pkl")
        
    return pickle.load(open(path, "rb"))


def set_cuda_devices(args):
    """
    Set requested cuda devices.
    :param args: main args
    """
    # assert that it is possible to request {args.gpus} cuda devices 
    assert args.gpus <= torch.cuda.device_count(), f"Requested {args.gpus} gpus, available {torch.cuda.device_count()}."
    
    device_list = ",".join([str(i) for i in range(args.gpus)])
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", device_list)


def verify_ckpt_path(args):
    """
    Verify the good definition of specified checkpoint paths.
    :param args: main args
    :return: either a valid checkpoint path or None if no checkpoint is present
    """
    if args.resume_training:
        resume_path_ckpt = os.path.join(
            args.ckpt_path if args.ckpt_path is not None else "", "checkpoints", f"fold{args.fold}", "last.ckpt"
        )
        resume_path_results = os.path.join(args.results, "checkpoints", f"fold{args.fold}", "last.ckpt")
        
        if os.path.exists(resume_path_ckpt):
            return resume_path_ckpt
        
        if os.path.exists(resume_path_results):
            return resume_path_results
        
        print("[Warning] Checkpoint not found. Starting training from scratch.")
        return None
    
    return args.ckpt_path


def make_empty_dir(path):
    """
    Create directory at specified path.
    :param path: requested path
    """
    run(["rm", "-rf", path])
    os.makedirs(path)


def get_stats(prediction, target, class_idx):
    """
    Retrieve true positives, false negatives and false positives.
    :param prediction: prediction 
    :param target: target 
    :param class_idx: class index
    :return: true positives, false negatives, false positives
    """
    tp = np.logical_and(prediction == class_idx, target == class_idx).sum()
    fn = np.logical_and(prediction != class_idx, target == class_idx).sum()
    fp = np.logical_and(prediction == class_idx, target != class_idx).sum()
    
    return tp, fn, fp


def set_granularity():
    """
    Set L2 granularity to 128.
    """
    _libcudart = ctypes.CDLL("libcudart.so")
    pValue = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
    _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
    _libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))
    assert pValue.contents.value == 128
