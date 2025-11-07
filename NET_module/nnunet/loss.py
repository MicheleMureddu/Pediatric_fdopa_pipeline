
import torch.nn as nn

from monai.losses import DiceLoss, GeneralizedDiceLoss, FocalLoss


# inspired by the NVIDIA nnU-Net GitHub repository available at:
# https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Segmentation/nnUNet

# loss computing makes use of the MONAI toolkit available at:
# https://github.com/Project-MONAI/MONAI

# inspired by
# Bianconi, Andrea, et al. "Deep learning-based algorithm for postoperative glioblastoma MRI segmentation: 
# a promising new tool for tumor burden assessment." Brain Informatics 10.1 (2023): 26.

### modified by Mureddu Michele

class LossBraTS(nn.Module):
    def __init__(self, focal, preop=True):
        """
        Initialize the loss for the BraTS task.
        :param focal: boolean value to determine if summing FocalLoss() (BCEWithLogitsLoss() otherwise)
        :param preop: boolean value to determine if applying pre- or post-operative segmentation
        """
        super(LossBraTS, self).__init__()
        self.dice = DiceLoss(sigmoid=True, batch=True)
        #self.dice = GeneralizedDiceLoss(sigmoid=True, batch=True)
        self.ce = FocalLoss(gamma=2.0, to_onehot_y=False) if focal else nn.BCEWithLogitsLoss()
        self.preop = preop

    def _loss(self, prediction, target):
        """
        Compute the overall loss as the sum between the DiceLoss() and FocalLoss()/BCEWithLogitsLoss().
        :param prediction: prediction
        :param target: target
        :return: sum of the two losses
        """
        return self.dice(prediction, target) + self.ce(prediction, target.float())

    def forward(self, prediction, target):
        """
        Forward pass of the loss.
        :param prediction: prediction
        :param target: target
        :return: sum of the three computed losses (one for each BraTS class)
        """
        
        target_whole = (target == 1) > 0
       
        pred_whole = prediction[:, 0].unsqueeze(1)

        loss_whole = self._loss(pred_whole, target_whole)
       
        
        return loss_whole