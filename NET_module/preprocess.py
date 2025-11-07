
import os
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from NET_module.data_preprocessing.preprocessor import Preprocessor
from NET_module.utils.utils import get_task_code


# inspired by the NVIDIA nnU-Net GitHub repository available at:
# https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Segmentation/nnUNet

# inspired by
# Bianconi, Andrea, et al. "Deep learning-based algorithm for postoperative glioblastoma MRI segmentation: 
# a promising new tool for tumor burden assessment." Brain Informatics 10.1 (2023): 26.

### modified by Mureddu Michele

# retrieve args from command line
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--data", type=str, default="./brats_flair", help="Path to data directory")
parser.add_argument("--results", type=str, default="./brats_flair", help="Path for saving results directory")
parser.add_argument(
    "--exec_mode",
    type=str,
    default="training",
    choices=["training", "test"],
    help="Mode for data preprocessing",
)
parser.add_argument("--ohe", action="store_true", help="Add one-hot-encoding for foreground voxels (voxels > 0)")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--task", type=str, choices=["train", "val"], help="Choose between train or val on BraTS")
parser.add_argument("--dim", type=int, default=3, choices=[2, 3], help="Data dimension to prepare")
parser.add_argument("--n_jobs", type=int, default=-1, help="Number of parallel jobs for data preprocessing")


if __name__ == "__main__":
    args = parser.parse_args()
    # run the Preprocessor
    start = time.time()
    Preprocessor(args).run()
    task_code = get_task_code(args)
    path = os.path.join(args.data, task_code)
    if args.exec_mode == "test":
        path = os.path.join(path, "test")
    end = time.time()
    print(f"Pre-processing time: {(end - start):.2f}")
