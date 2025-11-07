
import os
import dllogger as logger
from pytorch_lightning.utilities import rank_zero_only
from dllogger import JSONStreamBackend, StdOutBackend, Verbosity


# inspired by the NVIDIA nnU-Net GitHub repository available at:
# https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Segmentation/nnUNet

# logger makes use of the NVIDIA DLLogger for Python (DLLogger)
# read more at: https://github.com/NVIDIA/dllogger

# inspired by
# Bianconi, Andrea, et al. "Deep learning-based algorithm for postoperative glioblastoma MRI segmentation: 
# a promising new tool for tumor burden assessment." Brain Informatics 10.1 (2023): 26.

class DLLogger:
    def __init__(self, log_dir, filename, append=True):
        """
        __init__ call to the NVIDIA DLLogger.
        :param log_dir: log saving directory
        :param filename: file name
        :param append: whether to append every epoch in the same file
        """
        super().__init__()
        self._initialize_dllogger(log_dir, filename, append)

    @rank_zero_only
    def _initialize_dllogger(self, log_dir, filename, append):
        """
        Initialize the NVIDIA DLLogger (called in the __init__ method).
        :param log_dir: log saving directory
        :param filename: file name
        :param append: whether to append every epoch in the same file
        """
        # JSONStreamBackend saves JSON formatted lines into a file, adding time stamps for each line
        # StdOutBackend is a vanilla backend that holds no buffers and that prints the provided values to stdout
        backends = [
            JSONStreamBackend(Verbosity.VERBOSE, os.path.join(log_dir, filename), append=append),
            StdOutBackend(Verbosity.VERBOSE),
        ]
        logger.init(backends=backends)

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        """
        Log metrics at a given step in the DLLogger.
        :param metrics: metrics to log
        :param step: current epoch
        """
        if step is None:
            step = ()
        logger.log(step=step, data=metrics)

    @rank_zero_only
    def log_metadata(self, metric, metadata):
        """
        Log metric and update metadata in each backend.
        :param metric: metrics to log
        :param metadata: metadata
        """
        logger.metadata(metric, metadata)

    @rank_zero_only
    def flush(self):
        """
        Flush logger at training end.
        """
        logger.flush()
