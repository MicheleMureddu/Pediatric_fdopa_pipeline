import torch

from monai.metrics import compute_meandice, do_metric_reduction, compute_hausdorff_distance
from monai.networks.utils import one_hot
from torchmetrics import Metric

# inspired by the NVIDIA nnU-Net GitHub repository available at:
# https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Segmentation/nnUNet

# metric computing makes use of the MONAI toolkit available at:
# https://github.com/Project-MONAI/MONAI

# inspired by
# Bianconi, Andrea, et al. "Deep learning-based algorithm for postoperative glioblastoma MRI segmentation: 
# a promising new tool for tumor burden assessment." Brain Informatics 10.1 (2023): 26.

### modified by Mureddu Michele

class Dice(Metric):
    def __init__(self, n_class, preop=True):
        """
        Initialize Dice metric.
        :param n_class: number of classes
        """
        super().__init__(dist_sync_on_step=False)
        self.n_class = n_class
        self.preop = preop
        self.add_state("loss", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("steps", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("dice", default=torch.zeros((n_class,)), dist_reduce_fx="sum")

    def update(self, prediction, target, loss):
        """
        Update step for Dice metric.
        :param prediction: prediction
        :param target: target
        :param loss: computed loss
        """
        # apply sigmoid to binarize tumor guess (one class for each channel)
        prediction = (torch.sigmoid(prediction) > 0.5).int()
 
        target_whole = (target ==1) > 0
        
        # stack to match prediction shape
        target = torch.stack([target_whole], dim=1)
        self.steps += 1
        self.loss += loss
        self.dice += self.compute_metric(prediction, target, compute_meandice, 1, 0)

    def compute(self):
        """
        Compute step for Dice score and loss.
        :return: Dice score scaled on [0, 100], loss
        """
        return 100 * self.dice / self.steps, self.loss / self.steps

    def compute_metric(self, prediction, target, metric_fn, best_metric, worst_metric):
        """
        Dice metric computing step (called inside the update step).
        :param prediction: prediction 
        :param target: target 
        :param metric_fn: metric to apply (here monai.metrics.compute_meandice)
        :param best_metric: best possible metric (used when neither prediction nor target present label voxels)
        :param worst_metric: worst possible metric (used for NaN and Inf)
        :return: computed metric_fn
        """
        metric = metric_fn(prediction, target, include_background=True)
        # evaluate as worst_metric both NaN and Inf cases
        metric = torch.nan_to_num(metric, nan=worst_metric, posinf=worst_metric, neginf=worst_metric)
        metric = do_metric_reduction(metric, "mean_batch")[0]

        for i in range(self.n_class):
            if (target != 1).all():
                # evaluate as best_metric the eventuality in which neither prediction nor target present label voxels
                metric[i - 1] += best_metric if (prediction != 1).all() else worst_metric

        return metric


class Hausdorff95(Metric):
    def __init__(self, n_class, preop=True):
        """
        Initialize Hausdorff95 metric.
        :param n_class: number of classes
        """
        super().__init__(dist_sync_on_step=False)
        self.n_class = n_class
        self.preop = preop
        self.add_state("loss", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("steps", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("hausdorff95", default=torch.zeros((n_class,)), dist_reduce_fx="sum")

    def update(self, prediction, target, loss):
        """
        Update step for Hausdorff95 metric.
        :param prediction: prediction
        :param target: target
        :param loss: computed loss
        """
        # apply sigmoid to binarize tumor guess (one class for each channel)
        prediction = (torch.sigmoid(prediction) > 0.5).int()
        target_whole = (target == 1) > 0
        #target_core = ((target == 1) + (target == 3)) > 0
        #target_enh = target == 1 + 2 * self.preop  # 3 if preop, 1 otherwise
        # stack to match prediction shape
        target = torch.stack([target_whole], dim=1)

        self.steps += 1
        self.loss += loss
        # one-hot for compute_hausdorff_distance
        pred_whole = one_hot(prediction[:, 0].unsqueeze(1), num_classes=2, dim=1)
        target_whole = one_hot(target_whole.unsqueeze(1), num_classes=2, dim=1)
        h95_whole = self.compute_metric(pred_whole, target_whole, compute_hausdorff_distance, 0, 373.13)
        #pred_core = one_hot(prediction[:, 1].unsqueeze(1), num_classes=2, dim=1)
        #target_core = one_hot(target_core.unsqueeze(1), num_classes=2, dim=1)
        #h95_core = self.compute_metric(pred_core, target_core, compute_hausdorff_distance, 0, 373.13)
        #pred_enh = one_hot(prediction[:, 2].unsqueeze(1), num_classes=2, dim=1)
        #target_enh = one_hot(target_enh.unsqueeze(1), num_classes=2, dim=1)
        #h95_enh = self.compute_metric(pred_enh, target_enh, compute_hausdorff_distance, 0, 373.13)
        self.hausdorff95 += torch.cat([h95_whole])

    def compute(self):
        """
        Compute step for Hausdorff95 score and loss.
        :return: Hausdorff95, loss
        """
        return self.hausdorff95 / self.steps, self.loss / self.steps

    def compute_metric(self, prediction, target, metric_fn, best_metric, worst_metric):
        """
        Hausdorff95 metric computing step (called inside the update step).
        :param prediction: prediction 
        :param target: target 
        :param metric_fn: metric to apply (here monai.metrics.compute_hausdorff_distance)
        :param best_metric: best possible metric (used when neither prediction nor target present label voxels)
        :param worst_metric: worst possible metric (used for NaN and Inf)
        :return: computed metric_fn
        """
        metric = metric_fn(prediction, target, percentile=95, include_background=False)
        # evaluate as worst_metric both NaN and Inf cases
        metric = torch.nan_to_num(metric, nan=worst_metric, posinf=worst_metric, neginf=worst_metric)
        metric = do_metric_reduction(metric, "mean_batch")[0]

        for i in range(1):
            if (target != 1).all():
                # evaluate as best_metric the eventuality in which neither prediction nor target present label voxels
                metric[i - 1] += best_metric if (prediction != 1).all() else worst_metric

        if torch.cuda.is_available():
            metric = metric.cuda()

        return metric
