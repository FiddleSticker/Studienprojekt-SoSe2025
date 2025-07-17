"""Main execution file for training"""

import os
import shutil
from roboflow import Roboflow
from ultralytics import YOLO

from src import constants as c
from src.augment import augment_images


if __name__ == "__main__":
    # Download dataset from Roboflow
    rf = Roboflow(api_key="f2Zm3PDjLvH2GrAMKRuE")
    project = rf.workspace("test-p4ko4").project("test2-xzapr")
    version = project.version(5)  # Dataset mit 80/16/16 Split aber ohne Shuffle
    dataset = version.download("yolov9")  # Vlt anpassen an Yolo11

    # Moving dataset to fixed location and shuffling
    if os.path.isdir(c.DATASET_LOCATION):
        shutil.rmtree(c.DATASET_LOCATION)
    os.rename(dataset.location, c.DATASET_LOCATION)
    c.shuffle_sets(c.DATASET_LOCATION, 80, 16, 16)
    augment_images(os.path.join(c.DATASET_LOCATION, "train"), n_augmentations=10)

    # Training model
    model = YOLO("yolov9t.pt") # CHOOSE RIGHT MODEL HERE
    model.train(
        data=os.path.join(c.DATASET_LOCATION, "data.yaml"),
        epochs=500,
        imgsz=640,
        # Ab hier auskommentieren um native augmentation zu disablen
        augment=False,
        mosaic=0.0,
        mixup=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        flipud=0.0,
        fliplr=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        bgr=0.0,
        cutmix=0.0,
    )
