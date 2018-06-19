"""
Mask R-CNN configuration for KITTI segmentated dataset
For more information and how to use see
https://github.com/matterport/Mask_RCNN
Created by Žygintas Bergeris
Based on code by Waleed Abdulla
https://github.com/matterport/Mask_RCNN/tree/master/samples/nucleus
"""

import matplotlib
import os
import sys
import json
import datetime
import numpy as np
import logging
import skimage.io

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize
from mrcnn.model import log


import itertools
import math
import re
import random
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon

from mrcnn.visualize import display_images


# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/kitti/")


############################################################
#  Configurations
############################################################

class KittiConfig(Config):
    """Configuration for training on the traffic sign segmentation dataset."""
    # Give the configuration a recognizable name
    NAME = "kitti"

    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + traffic sign

    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = 150 // IMAGES_PER_GPU
    VALIDATION_STEPS = (50 // IMAGES_PER_GPU)

    DETECTION_MIN_CONFIDENCE = 0.9

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet101"

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 2.0

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 50

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 50

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimzer
    # implementation.
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    RPN_ANCHOR_STRIDE = 1


class KittiInferenceConfig(KittiConfig):

    # Don't resize image for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7


############################################################
#  Dataset
############################################################

class KittiDataset(utils.Dataset):

    def load_kitti(self, dataset_dir, subset):
        """Load a subset of the nuclei dataset.

        dataset_dir: Root directory of the dataset
        subset: Subset to load. Either the name of the sub-directory,
                such as stage1_train, stage1_test, ...etc. or, one of:
                * train: stage1_train excluding validation images
                * val: validation images from VAL_IMAGE_IDS
        """

        # Add classes. We have one class.
        self.add_class("kitti", 1, "traffic sign")

        # Adds subset to path and adds image_2 folder with images (dataset design)
        dataset_dir = os.path.join(dataset_dir, subset)
        dataset_dir = os.path.join(dataset_dir, "image_2")
        print("Dataset at:", dataset_dir)

        # Reads all image names and gets image ids
        # Images in dataset named as XXXXXX_XX.png where first 6 digits are image id
        image_ids = next(os.walk(dataset_dir))[2]
        ids =  [str.rsplit(x, ".png")[0] for x in image_ids]
        ids =  [str.rsplit(x, "_")[0] for x in image_ids]
        
        # Saves image ids
        for image_id in ids:
            self.add_image(
                "kitti",
                image_id=int(image_id),
                path=os.path.join(dataset_dir, "{}_10.png".format(image_id)))

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        info = self.image_info[image_id]

        # Get mask directory from image path
        mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "masks")

        # Read mask files from .png images
        # Mask files for each image saved in separate folders named with image_id
        mask = []
        for file in os.listdir(mask_dir):
            fa = sys.intern(os.fsdecode(file))
            if fa == sys.intern(str(image_id)):
                for file in os.listdir(os.path.join(mask_dir, fa)):
                    f = sys.intern(os.fsdecode(file))
                    if f.endswith(".png"):
                        m = skimage.io.imread(os.path.join(mask_dir, fa, f)).astype(np.bool)
                        mask.append(m)
        mask = np.asarray(mask)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        mask = np.moveaxis(mask, [2,1,0], [1,0,2])
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def load_image(self, image_id):
        """Load the specified image and return a [H,W] Numpy array.
        """

        info = self.image_info[image_id]
        image = skimage.io.imread(info['path'])
        return image

    def image_reference(self, image_id):
        """Return the path of the image."""

        info = self.image_info[image_id]    
        return info["path"]

    
############################################################
#  Training
############################################################

def train(model, dataset_dir, subset):
    """Train the model."""
    # Training dataset.
    dataset_train = KittiDataset()
    dataset_train.load_kitti(dataset_dir, subset)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = KittiDataset()
    dataset_val.load_kitti(dataset_dir, "testing")
    dataset_val.prepare()

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')

############################################################
#  Detection
############################################################

def detect(model, dataset_dir, subset):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "reusults_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = KittiDataset()
    dataset.load_kitti(dataset_dir, subset)
    dataset.prepare()
    # Load over images
    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # Save image with masks
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'],
            show_bbox=False, show_mask=True,
            title="Predictions")
        plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))

############################################################
#  Misc
############################################################

def compute_metrics(model, dataset_dir, subset):
    """Runs object detection and computes mAP and IoU for detected objects"""

    # Load dataset
    dataset = KittiDataset()
    dataset.load_kitti(dataset_dir, subset)
    dataset.prepare()
    
    # Calculates IoU and AP for each image
    APs = []
    IoUs = []
    for image_id in dataset.image_ids:
        # Load image
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset, config,
                                   image_id, use_mini_mask=False)
        # Run object detection
        results = model.detect([image], verbose=0)
        # Compute AP and IoU
        r = results[0]
        if (len(gt_class_id) == 0):
            continue
        AP, precisions, recalls, overlaps =\
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                              r['rois'], r['class_ids'], r['scores'], r['masks'])
        IoUs.append(overlaps)
        APs.append(AP)
    # Prints mAP - mean of all APs
    print("mAP : ", np.mean(APs))

    # Calculates and prints mean IoU
    vals = []
    for x in IoUs:
        z = x[np.nonzero(x)]
        for y in z:
            vals.append(y)
    print("IoU: ", np.mean(vals))


############################################################
#  Command Line
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for traffic sign detection')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'training', 'detect' or 'evaluate")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--subset', required=False,
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run prediction on")
    args = parser.parse_args()

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    if args.subset:
        print("Subset: ", args.subset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "training":
        config = KittiConfig()
    else:
        config = KittiInferenceConfig()
    config.display()

    # Create model
    if args.command == "training":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train, detect or evaluate
    if args.command == "training":
        train(model, args.dataset, args.subset)
    elif args.command == "detect":
        detect(model, args.dataset, args.subset)
    elif args.command == "evaluate":
        compute_metrics(model, args.dataset, args.subset)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))
