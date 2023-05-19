from pycocotools.coco import COCO
import numpy as np
import imageio.v2 as imageio
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import tensorflow as tf

# from keras.utils.image_utils import load_img, img_to_array
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from pathlib import Path
import cv2 as cv
import imgaug as ia

from imgaug.augmentables.polys import Polygon, PolygonsOnImage

import src.utils.data_utils as data_utils
import src.data.data_augmentation as dataug


class COCOBinaryClassificationGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        data_dir,
        classes,
        positive_classes,
        negative_classes,
        data_type=None,
        batch_size=32,
        target_size=(224, 224),
        shuffle=True,
        augmenter=None,
    ):
        self.data_dir = data_dir
        self.data_type = data_type
        self.classes = classes
        self.positive_classes = positive_classes
        self.negative_classes = negative_classes
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.augmenter = augmenter

        # Initialize the COCO API
        if data_type is None:
            annFile_str = str(
                Path(data_dir).joinpath(
                    "annotations", "instances_{}.json".format("default")
                )
            )
        elif data_type == "train":
            annFile_str = str(
                Path(data_dir).joinpath(
                    "annotations", "instances_{}.json".format("train")
                )
            )
        elif (data_type == "val") or (data_type == "validation"):
            annFile_str = str(
                Path(data_dir).joinpath(
                    "annotations", "instances_{}.json".format("val")
                )
            )
        elif data_type == "test":
            annFile_str = str(
                Path(data_dir).joinpath(
                    "annotations", "instances_{}.json".format("test")
                )
            )
        else:
            raise NotImplementedError("Unknown data_type argument passed!")

        self.coco = COCO(annFile_str)

        # Get image IDs and annotations for specified classes
        self.annotations = data_utils.get_annotations(self.coco, self.classes)

        # Create dictionary to map annotation ID to image ID
        self.ann_dict = {ann["id"]: ann["image_id"] for ann in self.annotations}

        # Create dictionary to map image ID to annotations
        self.img_dict = data_utils.imgID_2_annotations(self.annotations)

        # Create dictionary to map image IDs to file paths
        self.imgid_2_file = data_utils.imgID_2_filepaths(
            self.coco, self.data_dir, self.data_type
        )

        # Create dictionary to map filename to image IDs
        self.file_2_imgid = data_utils.file_2_imgid(self.coco)

        # Create list of image IDs
        self.img_ids = list(self.img_dict.keys())

        # Shuffle image IDs if requested
        if self.shuffle:
            np.random.shuffle(self.img_ids)

    def __len__(self):
        return int(np.ceil(len(self.img_ids) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.img_ids)

    def get_batch_img_ids(self, idx):
        return self.img_ids[idx * self.batch_size : (idx + 1) * self.batch_size]

    def __getitem__(self, idx):
        batch_img_ids = self.get_batch_img_ids(idx)

        batch_X_images = []
        batch_y_targets = []
        for img_id in batch_img_ids:
            binary_classlabel = data_utils.get_binary_class(
                img_id, self.coco, self.img_dict, self.positive_classes
            )
            img_array = data_utils.load_img(
                img_id, self.coco, self.data_dir, self.data_type
            )

            img_array_augmented = self.augmenter(image=img_array)

            batch_X_images.append(img_array_augmented)
            batch_y_targets.append(np.array(binary_classlabel))

        batch_X_images = np.array(batch_X_images)
        batch_y_targets = np.array(batch_y_targets)

        return batch_X_images, batch_y_targets
