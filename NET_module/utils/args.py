
import os
import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace

# inspired by the NVIDIA nnU-Net GitHub repository available at:
# https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Segmentation/nnUNet

# inspired by
# Bianconi, Andrea, et al. "Deep learning-based algorithm for postoperative glioblastoma MRI segmentation: 
# a promising new tool for tumor burden assessment." Brain Informatics 10.1 (2023): 26.

def positive_int(value):
    """
    Function to assert if value is a positive int.
    :param value: value
    :return: positive int value
    """
    ivalue = int(value)
    assert ivalue > 0, f"Argparse error. Expected positive integer but got {value}"

    return ivalue


def non_negative_int(value):
    """
    Function to assert if value is a non negative int.
    :param value: value
    :return: non negative int value
    """
    ivalue = int(value)
    assert ivalue >= 0, f"Argparse error. Expected non-negative integer but got {value}"

    return ivalue


def geq_minus_one_int(value):
    """
    Function to assert if value is >= -1.
    :param value: value
    :return: non negative int value
    """
    ivalue = int(value)
    assert ivalue >= -1, f"Argparse error. Expected non-negative integer but got {value}"

    return ivalue


def float_0_1(value):
    """
    Function to assert if value is a float in the range [0, 1].
    :param value: value
    :return: float value
    """
    fvalue = float(value)
    assert 0 <= fvalue <= 1, f"Argparse error. Expected float value to be in range (0, 1), but got {value}"

    return fvalue


def get_main_args():
    """
    Function to retrieve all command line args.
    :return: args
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    arg = parser.add_argument
    # list all args
    arg("--exec_mode", type=str, choices=["train", "predict"], default="train", help="Execution mode to run the model")
    arg("--data", type=str, default="./data", help="Path to data directory")
    arg("--results", type=str, default="./results", help="Path to results directory")
    arg("--not_val", type=str, default=None, help="Path to .txt file with not volumetric patients' names")
    arg("--config", type=str, default=None, help="Config file with arguments")
    arg("--logname", type=str, default="logs.json", help="Name of dlloger output")
    arg("--task", type=str, choices=["train", "val"], default="train", help="Choose between train or val on BraTS")
    arg("--gpus", type=non_negative_int, default=1, help="Number of gpus")
    arg("--nodes", type=non_negative_int, default=1, help="Number of nodes")
    arg("--learning_rate", type=float, default=0.0008, help="Learning rate")
    arg("--gradient_clip_val", type=float, default=0, help="Gradient clipping norm value")
    arg("--negative_slope", type=float, default=0.01, help="Negative slope for LeakyReLU")
    arg("--tta", action="store_true", help="Enable test time augmentation")
    arg("--tb_logs", action="store_true", help="Log metrics to tensoboard")
    arg("--deep_supervision", action="store_true", help="Enable deep supervision")
    arg("--amp", action="store_true", help="Enable automatic mixed precision")
    arg("--focal", action="store_true", help="Use focal loss instead of cross entropy")
    arg("--save_ckpt", action="store_true", help="Enable saving checkpoint")
    arg("--nfolds", type=positive_int, default=5, help="Number of cross-validation folds")
    arg("--seed", type=non_negative_int, default=None, help="Random seed")
    arg("--ckpt_path", type=str, default=None, help="Path for loading checkpoint")
    arg("--ckpt_store_dir", type=str, default="./results", help="Path for saving checkpoint")
    arg("--resume_training", action="store_true", help="Resume training from the last checkpoint")
    arg("--fold", type=non_negative_int, default=0, help="Fold number")
    arg("--patience", type=positive_int, default=100, help="Early stopping patience")
    arg("--batch_size", type=positive_int, default=2, help="Batch size")
    arg("--val_batch_size", type=positive_int, default=4, help="Validation batch size")
    arg("--momentum", type=float, default=0.99, help="Momentum factor")
    arg("--weight_decay", type=float, default=0.0001, help="Weight decay (L2 penalty)")
    arg("--save_preds", action="store_true", help="Enable prediction saving")
    arg("--num_workers", type=non_negative_int, default=8, help="Number of subprocesses to use for data loading")
    arg("--epochs", type=non_negative_int, default=1000, help="Number of training epochs.")
    arg("--warmup", type=non_negative_int, default=5, help="Warmup iterations before collecting statistics")
    arg("--norm", type=str, choices=["instance", "batch", "group"], default="instance", help="Normalization layer")
    arg("--depth", type=non_negative_int, default=5, help="The depth of the encoder")
    arg("--min_fmap", type=non_negative_int, default=4, help="Minimal dimension of feature map in the bottleneck")
    arg("--deep_supr_num", type=non_negative_int, default=2, help="Number of deep supervision heads")
    arg("--res_block", action="store_true", help="Enable residual blocks")
    arg("--filters", nargs="+", help="[Optional] Set U-Net filters", default=None, type=int)
    arg("--oversampling", type=float_0_1, default=0.4, help="Probability of crop to have some region with positive label")
    arg("--overlap", type=float_0_1, default=0.5, help="Amount of overlap between scans during sliding window inference")
    arg("--scheduler", action="store_true", help="Enable cosine rate scheduler with warmup")
    arg("--freeze", type=geq_minus_one_int, default=-1, help="Number of levels to freeze during training")

    args = parser.parse_args()
    if args.config is not None:
        config = json.load(open(args.config, "r"))
        args = vars(args)
        args.update(config)
        args = Namespace(**args)

    # create folder if not present
    if not os.path.isdir(args.results):
        os.mkdir(args.results)

    with open(f"{args.results}/params.json", "w") as f:
        json.dump(vars(args), f)

    return args
