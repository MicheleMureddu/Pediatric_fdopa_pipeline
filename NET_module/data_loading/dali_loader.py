
import os
import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.math as math
import nvidia.dali.ops as ops
import nvidia.dali.types as types

from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator


# inspired by the NVIDIA nnU-Net GitHub repository available at:
# https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Segmentation/nnUNet

# data augmentation makes use of the NVIDIA Data Loading Library (DALI)
# read more at: https://docs.nvidia.com/deeplearning/dali/user-guide/docs/

# inspired by
# Bianconi, Andrea, et al. "Deep learning-based algorithm for postoperative glioblastoma MRI segmentation: 
# a promising new tool for tumor burden assessment." Brain Informatics 10.1 (2023): 26.

def random_augmentation(probability, augmented, original):
    """
    Perform random augmentation with a given probability.
    :param probability: probability of performing random augmentation
    :param augmented: augmented image
    :param original: original image
    :return: either one or the other
    """
    condition = fn.cast(fn.random.coin_flip(probability=probability), dtype=types.DALIDataType.BOOL)
    neg_condition = condition ^ True

    return condition * augmented + neg_condition * original


class GenericPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, **kwargs):
        """
        Initialize the generic data loading pipeline.
        :param batch_size: batch size
        :param num_threads: number of threads
        :param device_id: device id
        :param kwargs: kwargs
        """
        super().__init__(batch_size, num_threads, device_id)
        self.kwargs = kwargs
        self.dim = 3
        self.device = device_id
        self.patch_size = kwargs["patch_size"]
        self.load_to_gpu = kwargs["load_to_gpu"]
        self.prob = kwargs["probability"]
        self.input_x = self.get_reader(kwargs["imgs"])
        self.input_y = self.get_reader(kwargs["lbls"]) if kwargs["lbls"] is not None else None

    def get_reader(self, data):
        """
        Retrieve the reader.
        :param data: list with data paths
        :return: nvidia.dali.ops.readers.Numpy()
        """
        return ops.readers.Numpy(
            files=data,
            device="cpu",
            read_ahead=True,
            dont_use_mmap=True,
            pad_last_batch=True,
            shard_id=self.device,
            seed=self.kwargs["seed"],
            num_shards=self.kwargs["gpus"],
            shuffle_after_epoch=self.kwargs["shuffle"],
        )

    def load_data(self):
        """
        Load data (eventually to gpu).
        :return: pair (image, label) if labels are present, otherwise image only.
        """
        img = self.input_x(name="ReaderX")
        if self.load_to_gpu:
            img = img.gpu()
        img = fn.reshape(img, layout="CDHW")
        if self.input_y is not None:
            lbl = self.input_y(name="ReaderY")
            if self.load_to_gpu:
                lbl = lbl.gpu()
            lbl = fn.reshape(lbl, layout="CDHW")
            return img, lbl
        return img

    def crop(self, data):
        return fn.crop(data, crop=self.patch_size, out_of_bounds_policy="pad")

    def crop_fn(self, img, lbl):
        img, lbl = self.crop(img), self.crop(lbl)
        return img, lbl

    def transpose_fn(self, img, lbl):
        img, lbl = fn.transpose(img, perm=(1, 0, 2, 3)), fn.transpose(lbl, perm=(1, 0, 2, 3))
        return img, lbl


class TrainPipeline(GenericPipeline):
    def __init__(self, batch_size, num_threads, device_id, **kwargs):
        """
        Initialize the training-specific data loading pipeline.
        :param batch_size: batch size
        :param num_threads: number of threads
        :param device_id: device id
        :param kwargs: kwargs
        """
        super().__init__(batch_size, num_threads, device_id, **kwargs)
        self.oversampling = kwargs["oversampling"]
        self.crop_shape = types.Constant(np.array(self.patch_size), dtype=types.INT64)
        self.crop_shape_float = types.Constant(np.array(self.patch_size), dtype=types.FLOAT)

    @staticmethod
    def slice_fn(img):
        """
        Slice the given image.
        :param img: image
        :return: sliced image
        """
        return fn.slice(img, 1, 3, axes=[0])

    def resize(self, data, interp_type):
        """
        Resize the given data.
        :param data: data
        :param interp_type: interpolation method
        :return: resized image
        """
        return fn.resize(data, interp_type=interp_type, size=self.crop_shape_float)

    def biased_crop_fn(self, img, label):
        """
        Randomly crop an image from the input pair, guaranteeing that some foreground (i.e. tumor) is present with
        a given probability.
        :param image: image
        :param label: label
        :return: cropped image, cropped label
        """
        roi_start, roi_end = fn.segmentation.random_object_bbox(
            label,
            device="cpu",
            background=0,
            format="start_end",
            cache_objects=True,
            foreground_prob=self.oversampling,
        )
        anchor = fn.roi_random_crop(label, roi_start=roi_start, roi_end=roi_end, crop_shape=[1, *self.patch_size])
        anchor = fn.slice(anchor, 1, 3, axes=[0])  # drop channels from anchor
        img, label = fn.slice(
            [img, label], anchor, self.crop_shape, axis_names="DHW", out_of_bounds_policy="pad", device="cpu"
        )
        return img.gpu(), label.gpu()

    def zoom_fn(self, img, lbl):
        """
        Resize the input pair in a zoomed way.
        :param img: image
        :param lbl: label
        :return: possibly zoomed image, possibly zoomed label
        """
        scale = random_augmentation(self.prob, fn.random.uniform(range=(0.7, 1.0)), 1.0)
        d, h, w = [scale * x for x in self.patch_size]
        if self.dim == 2:
            d = self.patch_size[0]
        img, lbl = fn.crop(img, crop_h=h, crop_w=w, crop_d=d), fn.crop(lbl, crop_h=h, crop_w=w, crop_d=d)
        img, lbl = self.resize(img, types.DALIInterpType.INTERP_CUBIC), self.resize(lbl, types.DALIInterpType.INTERP_NN)
        return img, lbl

    def noise_fn(self, img):
        """
        Apply Gaussian random noise with mean 0 and sampled standard deviation to an image.
        :param img: input image
        :return: possibly noisy image
        """
        img_noised = img + fn.random.normal(img, stddev=fn.random.uniform(range=(0.0, 0.33)))
        return random_augmentation(self.prob, img_noised, img)

    def blur_fn(self, img):
        """
        Apply Gaussian blurring with standard deviation of the Gaussian Kernel sampled uniformly.
        :param img: image
        :return: possibly blurred image
        """
        img_blurred = fn.gaussian_blur(img, sigma=fn.random.uniform(range=(0.5, 1.5)))
        return random_augmentation(self.prob, img_blurred, img)

    def brightness_fn(self, img):
        """
        Modify image brightness bu multiplying input voxels by a randomly sampled value.
        :param img: image
        :return: possibly brightened image
        """
        brightness_scale = random_augmentation(self.prob, fn.random.uniform(range=(0.7, 1.3)), 1.0)
        return img * brightness_scale

    def contrast_fn(self, img):
        """
        Modify image contrast by multiplying input voxels by a randomly sampled value, clipping then
        to their original value range.
        :param img: image
        :return: possibly contrasted image
        """
        scale = random_augmentation(self.prob, fn.random.uniform(range=(0.65, 1.5)), 1.0)
        return math.clamp(img * scale, fn.reductions.min(img), fn.reductions.max(img))

    def flips_fn(self, img, lbl):
        """
        Flip input pair volume, independently for each axis, with probability of 0.15.
        :param img: image
        :param lbl: label
        :return: eventually flipped pair (image, label)
        """
        kwargs = {
            "horizontal": fn.random.coin_flip(probability=self.prob),
            "vertical": fn.random.coin_flip(probability=self.prob),
        }
        if self.dim == 3:
            kwargs.update({"depthwise": fn.random.coin_flip(probability=self.prob)})
        return fn.flip(img, **kwargs), fn.flip(lbl, **kwargs)

    def define_graph(self):
        """
        Define the whole data loading pipeline.
        :return: end-of-pipeline pair (image, label)
        """
        img, lbl = self.load_data()
        img, lbl = self.biased_crop_fn(img, lbl)
        img, lbl = self.zoom_fn(img, lbl)
        img, lbl = self.flips_fn(img, lbl)
        img = self.noise_fn(img)
        img = self.blur_fn(img)
        img = self.brightness_fn(img)
        img = self.contrast_fn(img)
        if self.dim == 2:
            img, lbl = self.transpose_fn(img, lbl)
        return img, lbl


class EvalPipeline(GenericPipeline):
    def __init__(self, batch_size, num_threads, device_id, **kwargs):
        """
        Initialize the evaluation-specific data loading pipeline.
        :param batch_size: batch size
        :param num_threads: number of threads
        :param device_id: device id
        :param kwargs: kwargs
        """
        super().__init__(batch_size, num_threads, device_id, **kwargs)

    def define_graph(self):
        """
        Define the whole data loading pipeline.
        :return: retrieved pair (image, label)
        """
        img, lbl = self.load_data()

        return img, lbl


class TestPipeline(GenericPipeline):
    def __init__(self, batch_size, num_threads, device_id, **kwargs):
        """
       Initialize the testing-specific data loading pipeline.
       :param batch_size: batch size
       :param num_threads: number of threads
       :param device_id: device id
       :param kwargs: kwargs
       """
        super().__init__(batch_size, num_threads, device_id, **kwargs)
        # labels are not present -> retrieve metadata instead
        self.input_meta = self.get_reader(kwargs["meta"])

    def define_graph(self):
        """
        Define the whole data loading pipeline.
        :return: retrieved pair (image, metadata)
        """
        img = self.load_data()
        meta = self.input_meta(name="ReaderM")

        return img, meta


PIPELINES = {
    "train": TrainPipeline,
    "eval": EvalPipeline,
    "test": TestPipeline,
}


class LightningWrapper(DALIGenericIterator):
    def __init__(self, pipe, **kwargs):
        """
        Initialize the DALI iterator for classification tasks for PyTorch.
        It returns 2 outputs (image and label) in the form of PyTorch’s Tensor.
        :param pipe: list of pipelines to use
        :param kwargs: kwargs
        """
        super().__init__(pipe, **kwargs)

    def __next__(self):
        """
        Retrieve next pair.
        :return: next pair
        """
        out = super().__next__()[0]

        return out


def fetch_dali_loader(imgs, lbls, batch_size, mode, **kwargs):
    """
    Fetch the DALI iterator loaded with desired pipelines.
    :param imgs: images
    :param lbls: labels
    :param batch_size: batch size
    :param mode: desired pipeline. Choose one between ["train", "eval", "test]
    :param kwargs: kwargs
    :return: loaded DALI iterator
    """
    assert len(imgs) > 0, "Empty list of images!"
    if lbls is not None:
        assert len(imgs) == len(lbls), f"Number of images ({len(imgs)}) not matching number of labels ({len(lbls)})"


    pipeline = PIPELINES[mode]
    shuffle = True if mode == "train" else False
    dynamic_shape = True if mode in ["eval", "test"] else False
    load_to_gpu = True if mode in ["eval", "test", "benchmark"] else False
    pipe_kwargs = {"imgs": imgs, "lbls": lbls, "load_to_gpu": load_to_gpu, "shuffle": shuffle, **kwargs}
    output_map = ["image", "meta"] if mode == "test" else ["image", "label"]

    rank = int(os.getenv("LOCAL_RANK", "0"))
    pipe = pipeline(batch_size, kwargs["num_workers"], rank, **pipe_kwargs)
    
    return LightningWrapper(
        pipe,
        auto_reset=True,
        reader_name="ReaderX",
        output_map=output_map,
        dynamic_shape=dynamic_shape,
    )
