import numpy as np
import pytorch_lightning as pl
import torch
import os

### from apex.optimizers import FusedAdam
from torch.optim import AdamW
from monai.inferers import sliding_window_inference
from monai.networks.nets import DynUNet
from monai.optimizers.lr_scheduler import WarmupCosineSchedule
from pytorch_lightning.utilities import rank_zero_only
from scipy.special import expit
from skimage.transform import resize
from NET_module.data_loading.data_module import get_data_path, load_data
from NET_module.nnunet.loss import LossBraTS
from NET_module.nnunet.metrics import Dice, Hausdorff95
from NET_module.utils.logger import DLLogger
from NET_module.utils.utils import get_config_file, print0


# inspired by the NVIDIA nnU-Net GitHub repository available at:
# https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Segmentation/nnUNet

# metric computing and model inference make use of the MONAI toolkit available at:
# https://github.com/Project-MONAI/MONAI

# Adam optimizer makes use of the NVIDIA A Pytorch Extension Library (Apex)
# read more at: https://nvidia.github.io/apex/

# inspired by
# Bianconi, Andrea, et al. "Deep learning-based algorithm for postoperative glioblastoma MRI segmentation: 
# a promising new tool for tumor burden assessment." Brain Informatics 10.1 (2023): 26.

### modified by Mureddu Michele


def flip(data, axis):
    """
    Flip function for test time augmentation.
    :param data: data
    :param axis: flip axis
    :return: flipped data along given axis
    """
    return torch.flip(data, dims=axis)


class NNUnet(pl.LightningModule):
    def __init__(self, args):
        """
        Initialize the nnU-Net framework for the BraTS task.
        :param args: args
        """
        super(NNUnet, self).__init__()
        self.save_hyperparameters()
        self.args = args
        self.build_nnunet()
        self.best_temp_dice, self.best_epoch, self.test_idx = (0,) * 3
        self.best_temp_hausdorff95 = 373.13
        self.train_loss = []
        self.test_imgs = []
        self.learning_rate = args.learning_rate
        self.loss = LossBraTS(self.args.focal, self.args.freeze >= 0)
        self.tta_flips = [[2], [3], [4], [2, 3], [2, 4], [3, 4], [2, 3, 4]]
        self.dice = Dice(self.n_class, self.args.freeze >= 0)
        self.hausdorff95 = Hausdorff95(self.n_class, self.args.freeze >= 0)
        if self.args.exec_mode == "train":
            self.dllogger = DLLogger(args.results, f"fold{args.fold}_{args.logname}")

    def forward(self, image):
        """
        nnU-Net forward method.
        :param image: input image
        :return: output
        """
        return torch.argmax(self.model(image), 1)

    def _forward(self, image):
        """
        Apply test time augmentation inference if specified, sliding window one otherwise
        :param image: input image
        :return: inference output
        """
        return self.tta_inference(image) if self.args.tta else self.sliding_window_inference(image)

    def compute_loss(self, prediction, target):
        """
        Compute the DiceLoss (with deep supervision if enabled).
        :param prediction: prediction
        :param target: target
        :return: computed loss
        """
        if self.args.deep_supervision:
            # apply deep supervision
            loss, weights = 0.0, 0.0
            for i in range(prediction.shape[1]):
                loss += self.loss(prediction[:, i], target) * (0.5 ** i)
                weights += 0.5 ** i

            return loss / weights

        return self.loss(prediction, target)

    def training_step(self, batch, batch_idx):
        """
        Define the training step.
        :param batch: batch
        :param batch_idx: batch index
        :return: batch loss
        """
        if batch_idx == 0:
            self.train_loss = []
        image, label = batch["image"], batch["label"]
        prediction = self.model(image)
        loss = self.compute_loss(prediction, label)
        self.train_loss.append(loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Define the validation step.
        :param batch: batch
        :param batch_idx: batch index
        """
        image, label = batch["image"], batch["label"]
        prediction = self._forward(image)
        loss = self.loss(prediction, label)
        self.dice.update(prediction, label[:,0], loss)
        self.hausdorff95.update(prediction, label[:,0], loss)

    def test_step(self, batch, batch_idx):
        """
        Define the test step, eventually saving outputs.
        :param batch: batch
        :param batch_idx: batch index
        """
        image = batch["image"]
        prediction = self._forward(image).squeeze(0).cpu().detach().numpy()
        if self.args.save_preds:
            prediction = expit(prediction)
            # resize to original shape
            meta = batch["meta"][0].cpu().detach().numpy()
            min_d, max_d = meta[0, 0], meta[1, 0]
            min_h, max_h = meta[0, 1], meta[1, 1]
            min_w, max_w = meta[0, 2], meta[1, 2]
            n_class, original_shape, cropped_shape = prediction.shape[0], meta[2], meta[3]
            if not all(cropped_shape == prediction.shape[1:]):
                resized_pred = np.zeros((n_class, *cropped_shape))
                for i in range(n_class):
                    resized_pred[i] = resize(
                        prediction[i], cropped_shape, order=3, mode="edge", cval=0, clip=True, anti_aliasing=False
                    )
                prediction = resized_pred
            final_pred = np.zeros((n_class, *original_shape))
            final_pred[:, min_d:max_d, min_h:max_h, min_w:max_w] = prediction

            self.save_mask(final_pred)

    def get_unet_params(self):
        """
        Compute and return the required parameters in order to build nnU-Net.
        :return: in_channels, out_channels, kernels list, strides list
        """
        config = get_config_file(self.args)
        patch_size, spacings = config["patch_size"], config["spacings"]
        strides, kernels, sizes = [], [], patch_size[:]
        while True:
            spacing_ratio = [spacing / min(spacings) for spacing in spacings]
            stride = [
                2 if ratio <= 2 and size >= 2 * self.args.min_fmap else 1 for (ratio, size) in zip(spacing_ratio, sizes)
            ]
            kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
            if all(s == 1 for s in stride):
                break
            sizes = [i / j for i, j in zip(sizes, stride)]
            spacings = [i * j for i, j in zip(spacings, stride)]
            kernels.append(kernel)
            strides.append(stride)
            if len(strides) == self.args.depth:
                break
        strides.insert(0, len(spacings) * [1])
        kernels.append(len(spacings) * [3])

        return config["in_channels"], config["n_class"], kernels, strides, patch_size

    def build_nnunet(self):
        """
        Build the actual nnU-Net model.
        """
        in_channels, out_channels, kernels, strides, self.patch_size = self.get_unet_params()
        print(f"out_channels: {out_channels}")
        self.n_class = out_channels - 1
        ### modified here for binary classification (original: out_channels = 3)
        out_channels = 1

        self.model = DynUNet(
            spatial_dims= 3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernels,
            strides=strides,
            upsample_kernel_size=strides[1:],
            filters=self.args.filters,
            norm_name=("INSTANCE", {"affine": True}),
            act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            deep_supervision=self.args.deep_supervision,
            deep_supr_num=self.args.deep_supr_num,
            res_block=self.args.res_block,
            trans_bias=True,
        )
        print0(f"Filters: {self.model.filters},\nKernels: {kernels}\nStrides: {strides}")

    def tta_inference(self, image):
        """
        Apply inference with test time augmentation (flip).
        :param image: input image
        :return: tta inference
        """
        pred = self.sliding_window_inference(image)
        for flip_idx in self.tta_flips:
            pred += flip(self.sliding_window_inference(flip(image, flip_idx)), flip_idx)
        pred /= len(self.tta_flips) + 1

        return pred

    def sliding_window_inference(self, image):
        """
        Call to monai.infers.sliding_window_inference for inference.
        :param image: input image
        :return: sliding window inference output
        """
        return sliding_window_inference(
            inputs=image,
            roi_size=self.patch_size,
            sw_batch_size=self.args.val_batch_size,
            predictor=self.model,
            overlap=self.args.overlap,
            mode="gaussian",
        )

    def round(self, tensor):
        """
        Round tensor mean to two decimal digits float.
        :param tensor: tensor
        :return: rounded tensor mean
        """
        return round(torch.mean(tensor).item(), 2)

    def validation_epoch_end(self, outputs):
        """
        Define the validation-end step.
        :param outputs: outputs
        """
        dice, dice_loss = self.dice.compute()
        self.dice.reset()
        hausdorff95, hausdorff95_loss = self.hausdorff95.compute()
        self.hausdorff95.reset()
        
        torch.cuda.empty_cache()
        # Update metrics
        dice_mean = torch.mean(dice)
        if dice_mean >= self.best_temp_dice:
            self.best_temp_dice = dice_mean
            self.best_mean_dice = dice
            self.best_epoch = self.current_epoch
        hausdorff95_mean = torch.mean(hausdorff95)
        if hausdorff95_mean <= self.best_temp_hausdorff95:
            self.best_temp_hausdorff95 = hausdorff95_mean
            self.best_mean_hausdorff95 = hausdorff95

        metrics = {}
        metrics["Dice"] = self.round(dice)
        metrics["Val Loss"] = self.round(dice_loss)
        metrics["Max Dice"] = self.round(self.best_mean_dice)
        metrics["Hausdorff95"] = self.round(hausdorff95)
        metrics["Min Hausdorff95"] = self.round(self.best_mean_hausdorff95)
        metrics["Best epoch"] = self.best_epoch
        metrics["Train Loss"] = round(sum(self.train_loss) / len(self.train_loss), 4)
        # update for each one of the overlapping regions as well
        metrics["Dice-1"] = self.round(dice)
        metrics["Hausdorff95-1"] = self.round(hausdorff95)

        self.dllogger.log_metrics(step=self.current_epoch, metrics=metrics)
        self.dllogger.flush()
        if self.args.tb_logs:
            # tensorboard logger
            self.logger.log_metrics(metrics, step=self.current_epoch)
        self.log("vloss", metrics["Val Loss"])
        self.log("dice", metrics["Dice"])
        self.log("hausdorff95", metrics["Hausdorff95"])

    @rank_zero_only
    def on_fit_end(self):
        """
        Define the fit-end step. Log metrics and flush.
        """
        metrics = {}
        metrics["dice_score"] = round(self.best_temp_dice.item(), 2)
        metrics["hausdorff95_score"] = round(self.best_temp_hausdorff95.item(), 2)
        metrics["train_loss"] = round(sum(self.train_loss) / len(self.train_loss), 4)
        metrics["val_loss"] = round(1 - self.best_temp_dice.item() / 100, 4)
        metrics["Epoch"] = self.best_epoch

        self.dllogger.log_metrics(step=(), metrics=metrics)
        self.dllogger.flush()

    def configure_optimizers(self):
        """
        Configure the Adam optimizer.
        """
        print(f"parameters: {self.parameters()}")
        # requires_grad prevents a gradient from being computed at first but it does not prevent other steps outside
        # of a gradient that might update parameters and thus ask for the gradient to pass through that layer, for
        # example: normalization, optimizers with parameters, etc.
        #
        #  the parameter cannot be given to the optimizer in order to freeze a parameter(s)
        optimizer = AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate,
                              weight_decay=self.args.weight_decay, fused=True)

        if self.args.scheduler:
            # apply warmup cosine scheduler
            scheduler = {
                "scheduler": WarmupCosineSchedule(
                    optimizer=optimizer,
                    warmup_steps=0,
                    t_total=self.args.epochs * len(self.trainer.datamodule.train_dataloader()),
                    cycles=0.25,
                ),
                "interval": "step",
                "frequency": 1,
            }
            return {"optimizer": optimizer, "monitor": "val_loss", "lr_scheduler": scheduler}

        return {"optimizer": optimizer, "monitor": "val_loss"}
    
    def save_mask(self, prediction):
        """
        Save the mask output after testing as numpy array.
        :param prediction: prediction
        """
        if self.test_idx == 0:
            data_path = get_data_path(self.args)
            self.test_imgs = load_data(data_path, "*_x.npy", non_empty=False)

        fname = os.path.basename(self.test_imgs[self.test_idx]).replace("_x", "")
        np.save(os.path.join(self.save_dir, fname), prediction, allow_pickle=False)
        self.test_idx += 1
